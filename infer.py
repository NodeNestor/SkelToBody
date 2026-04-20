"""
Inference: skeleton mesh -> full body mesh.

Load the base TRELLIS.2 shape-flow denoiser + shape decoder, attach the trained
SkeletonAdapter + LoRA, encode the input skeleton via SC-VAE, sample body SLAT
via rectified flow with CFG, decode to a textured mesh, export GLB.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).parent / 'train'))
sys.path.insert(0, str(Path(__file__).parent / 'data'))

import o_voxel
import trellis2.models as models
import trellis2.modules.sparse as sp

from skeleton_adapter import SkeletonAdapter, AdapterConfig
from dataset import SHAPE_LATENT_MEAN, SHAPE_LATENT_STD, denormalize
from mesh_to_slat import load_mesh_tensors, mesh_to_sparse_dual_grid
from train import inject_lora


@torch.no_grad()
def encode_skeleton(skel_glb: str, encoder, grid_size: int = 512):
    v, f = load_mesh_tensors(skel_glb)
    verts_st, inter_st = mesh_to_sparse_dual_grid(v, f, grid_size)
    z = encoder(verts_st.cuda(), inter_st.cuda())
    torch.cuda.synchronize()
    return z.feats.float(), z.coords[:, 1:].int()


@torch.no_grad()
def sample_body_slat(denoiser, adapter, sk_feats, sk_coords, num_steps: int = 20, cfg: float = 3.0,
                     dilate: int = 3):
    """Rectified-flow sampling with classifier-free guidance.

    The body voxel grid is *dilated* outward from the skeleton so the shape flow
    has room to generate body mass beyond bone (skin, fat, muscle). This is a
    cheap substitute for a real SS flow model: given grid resolution R, we
    expand each skeleton voxel by `dilate` cells in every direction.
    """
    device = 'cuda'
    sk_f_norm = (sk_feats.cpu().numpy() - SHAPE_LATENT_MEAN) / SHAPE_LATENT_STD
    sk_f = torch.from_numpy(sk_f_norm).to(device).to(torch.bfloat16)
    sk_c = sk_coords.to(device)

    cond, _ = adapter([sk_f], [sk_c])
    cond = cond.float()
    uncond = torch.zeros_like(cond)

    # Dilate skeleton voxel coords outward so the body can extend beyond bone.
    # Build a set of unique 3D offsets within a cube of radius `dilate`.
    offsets = torch.stack(torch.meshgrid(
        torch.arange(-dilate, dilate + 1, device=device),
        torch.arange(-dilate, dilate + 1, device=device),
        torch.arange(-dilate, dilate + 1, device=device),
        indexing='ij'), dim=-1).reshape(-1, 3)   # (D^3, 3)
    # Combine: (N_skel, D^3, 3) → unique → clip to [0, 31] (SLAT sparse res)
    expanded = (sk_c.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
    expanded = expanded.clamp(0, 31)
    body_c = torch.unique(expanded, dim=0)
    print(f'       dilated to {body_c.shape[0]} body voxels (from {sk_c.shape[0]} skeleton)')

    coords4 = torch.cat([torch.zeros_like(body_c[:, 0:1]), body_c], dim=-1).int()
    x = torch.randn(body_c.shape[0], 32, device=device, dtype=torch.float32)
    x_st = sp.SparseTensor(x, coords4)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.tensor([i * dt], device=device)
        v_cond = denoiser(x_st, t, cond).feats
        v_uncond = denoiser(x_st, t, uncond).feats
        v = v_uncond + cfg * (v_cond - v_uncond)
        x_st = x_st.replace(x_st.feats + v * dt)

    return x_st.feats.cpu().numpy(), sk_c.cpu().numpy()


def decode_slat_to_mesh(feats_norm, coords, shape_decoder, resolution: int = 512):
    feats = denormalize(feats_norm)
    coords4 = np.concatenate([np.zeros((coords.shape[0], 1), dtype=np.int32), coords.astype(np.int32)], axis=1)
    st = sp.SparseTensor(
        torch.from_numpy(feats).float().cuda(),
        torch.from_numpy(coords4).int().cuda(),
    )
    out = shape_decoder(st)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skeleton', required=True, help='skeleton mesh (GLB/OBJ)')
    ap.add_argument('--ckpt', required=True, help='trained .pt from train.py')
    ap.add_argument('--out', required=True, help='output GLB')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--shape_encoder', default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')
    ap.add_argument('--shape_decoder', default='microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
    ap.add_argument('--denoiser', default='microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16')
    ap.add_argument('--lora_rank', type=int, default=16, help='must match training rank')
    ap.add_argument('--dilate', type=int, default=3,
                    help='expand skeleton voxels outward by N cells so body can grow beyond bone')
    args = ap.parse_args()

    print('[load] shape encoder'); encoder = models.from_pretrained(args.shape_encoder).eval().cuda()
    print('[load] shape decoder'); decoder = models.from_pretrained(args.shape_decoder).eval().cuda()
    print('[load] denoiser');      denoiser = models.from_pretrained(args.denoiser).eval().cuda()

    # Re-inject LoRA scaffolding FIRST (same as training) so the checkpoint's
    # module paths match (e.g. "blocks.0.attn.qkv.lora_a.weight").
    inject_lora(denoiser, rank=args.lora_rank)

    adapter = SkeletonAdapter(AdapterConfig()).to('cuda').to(torch.bfloat16)
    ck = torch.load(args.ckpt, map_location='cuda')
    adapter.load_state_dict(ck['adapter'])
    lora = ck.get('lora', {})
    loaded = 0
    for name, p in denoiser.named_parameters():
        if name in lora:
            p.data.copy_(lora[name].to(p.device, p.dtype))
            loaded += 1
    print(f'[load] restored {loaded} LoRA params')

    print('[run] encode skeleton ...')
    sk_f, sk_c = encode_skeleton(args.skeleton, encoder)
    print(f'       got {sk_c.shape[0]} skeleton tokens')

    print('[run] sample body SLAT ...')
    body_f, body_c = sample_body_slat(denoiser, adapter, sk_f, sk_c, num_steps=args.steps, cfg=args.cfg,
                                       dilate=args.dilate)

    print('[run] decode to mesh ...')
    mesh = decode_slat_to_mesh(body_f, body_c, decoder)
    # Decoder returns a list (one item per batch element); we only have batch=1.
    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0]

    print(f'[save] writing {args.out}')
    # Prefer o_voxel's full postprocessing if CuMesh is installed; fall back to trimesh OBJ.
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=mesh.attrs if hasattr(mesh, 'attrs') else None,
            coords=mesh.coords if hasattr(mesh, 'coords') else None,
            attr_layout=mesh.layout if hasattr(mesh, 'layout') else None,
            voxel_size=mesh.voxel_size if hasattr(mesh, 'voxel_size') else 1.0 / 512,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=200000, texture_size=1024, remesh=True, remesh_band=1, remesh_project=0,
            verbose=True,
        )
        glb.export(args.out, extension_webp=True)
    except Exception as e:
        print(f'[warn] o_voxel.postprocess.to_glb failed ({e}); falling back to trimesh export')
        import trimesh as _tm
        out = _tm.Trimesh(vertices=mesh.vertices.cpu().numpy(),
                          faces=mesh.faces.cpu().numpy(), process=False)
        out.export(args.out)
    print('[done]')


if __name__ == '__main__':
    main()
