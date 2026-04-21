"""
Two-stage inference: skeleton mesh -> full body mesh.

1. SS stage:  skel occupancy (64^3) -> SS VAE encoder -> skel SS latent
              -> SSConditioner -> cond tokens
              -> SS flow (+ LoRA) rectified-flow sample -> body SS latent
              -> SS VAE decoder -> body occupancy (64^3)
              -> body sparse coords (in 32^3 grid)

2. Shape stage: skel mesh -> shape VAE encoder -> skel SLAT
                -> SkeletonAdapter -> cond tokens
                -> shape flow (+ LoRA) rectified-flow sample at body coords -> body SLAT
                -> shape VAE decoder -> body mesh -> GLB
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
from ss_adapter import SSConditioner
from dataset import SHAPE_LATENT_MEAN, SHAPE_LATENT_STD, denormalize
from mesh_to_slat import load_mesh_tensors, mesh_to_sparse_dual_grid
from mesh_to_ss import mesh_to_occupancy_grid
from train import inject_lora


@torch.no_grad()
def ss_stage(skeleton_glb: str, ss_encoder, ss_decoder, ss_flow, ss_cond: SSConditioner,
             ss_steps: int = 20, cfg: float = 3.0, out_res: int = 32) -> torch.Tensor:
    """Returns body sparse coords (N, 4) with batch idx 0 at grid_res=out_res."""
    device = 'cuda'
    # 1. skel mesh -> 64^3 occupancy -> SS latent
    grid = mesh_to_occupancy_grid(skeleton_glb, res=64).to(device)
    skel_ss = ss_encoder(grid)                                   # (1, 8, 16, 16, 16)

    # 2. SS flow sample conditioned on skel_ss
    cond = ss_cond(skel_ss.to(torch.bfloat16)).float()           # (1, 4096, 1024)
    uncond = torch.zeros_like(cond)
    x = torch.randn_like(skel_ss).float()
    dt = 1.0 / ss_steps
    for i in range(ss_steps):
        t = torch.tensor([i * dt], device=device)
        v_cond = ss_flow(x, t, cond)
        v_uncond = ss_flow(x, t, uncond)
        v = v_uncond + cfg * (v_cond - v_uncond)
        x = x + v * dt

    # 3. decode body SS latent -> occupancy (1, 1, 64, 64, 64) boolean
    decoded = ss_decoder(x) > 0
    if out_res != decoded.shape[2]:
        ratio = decoded.shape[2] // out_res
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()      # (N, 4)
    print(f'       SS stage produced {coords.shape[0]} body voxel coords at res={out_res}')
    return coords


@torch.no_grad()
def encode_skeleton_slat(skeleton_glb: str, shape_encoder, grid_size: int = 512):
    v, f = load_mesh_tensors(skeleton_glb)
    verts_st, inter_st = mesh_to_sparse_dual_grid(v, f, grid_size)
    z = shape_encoder(verts_st.cuda(), inter_st.cuda())
    return z.feats.float(), z.coords[:, 1:].int()


@torch.no_grad()
def shape_stage(body_coords: torch.Tensor, skel_feats: torch.Tensor, skel_coords: torch.Tensor,
                shape_flow, shape_decoder, adapter: SkeletonAdapter,
                num_steps: int = 30, cfg: float = 3.0):
    device = 'cuda'
    # Skeleton -> cond tokens
    sk_f_norm = (skel_feats.cpu().numpy() - SHAPE_LATENT_MEAN) / SHAPE_LATENT_STD
    sk_f = torch.from_numpy(sk_f_norm).to(device).to(torch.bfloat16)
    sk_c = skel_coords.to(device)
    cond, _ = adapter([sk_f], [sk_c])
    cond = cond.float()
    uncond = torch.zeros_like(cond)

    # Sample body SLAT at the body_coords we got from SS stage
    body_c = body_coords.to(device)
    x = torch.randn(body_c.shape[0], 32, device=device)
    x_st = sp.SparseTensor(x, body_c.int())

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.tensor([i * dt], device=device)
        v_cond = shape_flow(x_st, t, cond).feats
        v_uncond = shape_flow(x_st, t, uncond).feats
        v = v_uncond + cfg * (v_cond - v_uncond)
        x_st = x_st.replace(x_st.feats + v * dt)

    # Denormalize the predicted SLAT features
    feats = denormalize(x_st.feats.cpu().numpy())
    feats_t = torch.from_numpy(feats).float().cuda()
    st = sp.SparseTensor(feats_t, body_c.int())

    # Decode to mesh
    mesh = shape_decoder(st)
    if isinstance(mesh, (list, tuple)):
        mesh = mesh[0]
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skeleton', required=True)
    ap.add_argument('--shape_ckpt', required=True, help='trained shape-flow checkpoint (ckpt_*.pt)')
    ap.add_argument('--ss_ckpt', required=True, help='trained SS-flow checkpoint (ss_ckpt_*.pt)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ss_steps', type=int, default=20)
    ap.add_argument('--shape_steps', type=int, default=30)
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--lora_rank', type=int, default=16)
    ap.add_argument('--shape_encoder', default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')
    ap.add_argument('--shape_decoder', default='microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
    ap.add_argument('--shape_flow', default='microsoft/TRELLIS.2-4B/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16')
    ap.add_argument('--ss_encoder', default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
    ap.add_argument('--ss_decoder', default='microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')
    ap.add_argument('--ss_flow', default='microsoft/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16')
    args = ap.parse_args()

    device = 'cuda'
    print('[load] shape encoder');  shape_encoder = models.from_pretrained(args.shape_encoder).eval().to(device)
    print('[load] shape decoder');  shape_decoder = models.from_pretrained(args.shape_decoder).eval().to(device)
    print('[load] shape flow');     shape_flow    = models.from_pretrained(args.shape_flow).eval().to(device)
    print('[load] SS encoder');     ss_encoder    = models.from_pretrained(args.ss_encoder).eval().to(device)
    print('[load] SS decoder');     ss_decoder    = models.from_pretrained(args.ss_decoder).eval().to(device)
    print('[load] SS flow');        ss_flow       = models.from_pretrained(args.ss_flow).eval().to(device)

    # Attach LoRA to both flows and restore from checkpoints.
    inject_lora(shape_flow, rank=args.lora_rank)
    inject_lora(ss_flow,    rank=args.lora_rank)

    adapter = SkeletonAdapter(AdapterConfig()).to(device).to(torch.bfloat16)
    ss_cond = SSConditioner().to(device).to(torch.bfloat16)

    sh_ck = torch.load(args.shape_ckpt, map_location=device)
    adapter.load_state_dict(sh_ck['adapter'])
    for name, p in shape_flow.named_parameters():
        if name in sh_ck['lora']:
            p.data.copy_(sh_ck['lora'][name].to(p.device, p.dtype))
    print(f'[load] shape ckpt @ step {sh_ck.get("step")}')

    ss_ck = torch.load(args.ss_ckpt, map_location=device)
    ss_cond.load_state_dict(ss_ck['conditioner'])
    for name, p in ss_flow.named_parameters():
        if name in ss_ck['lora']:
            p.data.copy_(ss_ck['lora'][name].to(p.device, p.dtype))
    print(f'[load] ss ckpt @ step {ss_ck.get("step")}')

    print('[run] SS stage -> body voxel coords')
    body_coords = ss_stage(args.skeleton, ss_encoder, ss_decoder, ss_flow, ss_cond,
                           ss_steps=args.ss_steps, cfg=args.cfg, out_res=32)
    if body_coords.shape[0] < 50:
        print('[warn] SS stage produced very few voxels; model is probably undertrained')

    print('[run] encode skeleton SLAT')
    sk_f, sk_c = encode_skeleton_slat(args.skeleton, shape_encoder)

    print('[run] shape stage -> body mesh')
    mesh = shape_stage(body_coords, sk_f, sk_c, shape_flow, shape_decoder, adapter,
                       num_steps=args.shape_steps, cfg=args.cfg)

    print(f'[save] {args.out}')
    try:
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces,
            attr_volume=getattr(mesh, 'attrs', None),
            coords=getattr(mesh, 'coords', None),
            attr_layout=getattr(mesh, 'layout', None),
            voxel_size=getattr(mesh, 'voxel_size', 1 / 512),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=200000, texture_size=1024, remesh=True, remesh_band=1, remesh_project=0,
            verbose=False,
        )
        glb.export(args.out, extension_webp=True)
    except Exception as e:
        print(f'[warn] o_voxel export failed ({e}); trimesh fallback')
        import trimesh as _tm
        out = _tm.Trimesh(vertices=mesh.vertices.cpu().numpy(),
                          faces=mesh.faces.cpu().numpy(), process=False)
        out.export(args.out)
    print('[done]')


if __name__ == '__main__':
    main()
