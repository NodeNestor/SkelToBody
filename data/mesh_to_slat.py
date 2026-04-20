"""
Encode a mesh (in [-0.5, 0.5]^3) to a TRELLIS.2 shape SLAT:

    mesh  -->  dual-grid O-Voxel  -->  sparse shape latent (32 channels)

Batch-processes a directory of (skeleton, body) GLB pairs and saves
paired .npz files with {skel_feats, skel_coords, body_feats, body_coords}.

Requires: trellis2, o_voxel, torch, a GPU.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import trimesh

import o_voxel
import trellis2.models as models
import trellis2.modules.sparse as sp


torch.set_grad_enabled(False)


def load_mesh_tensors(glb_path: str):
    m = trimesh.load(glb_path, process=False, force='mesh')
    v = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32))
    f = torch.from_numpy(np.asarray(m.faces, dtype=np.int64))
    # safety: clamp to the valid cube (CT segmentation can have tiny overshoots)
    v = torch.clamp(v, -0.5 + 1e-6, 0.5 - 1e-6)
    return v, f


def mesh_to_sparse_dual_grid(vertices: torch.Tensor, faces: torch.Tensor, grid_size: int):
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=vertices, faces=faces,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
        timing=False,
    )
    dual_vertices = dual_vertices * grid_size - voxel_indices
    dual_vertices = torch.clamp(dual_vertices, 0, 1)  # normalized [0,1] per voxel

    # wrap as sparse tensors following the repo's convention
    coords4 = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
    verts_st = sp.SparseTensor(dual_vertices.float(), coords4)
    inter_st = verts_st.replace(intersected.bool())
    return verts_st, inter_st


def encode_mesh(glb_path: str, encoder, grid_size: int = 512):
    v, f = load_mesh_tensors(glb_path)
    verts_st, inter_st = mesh_to_sparse_dual_grid(v, f, grid_size)
    z = encoder(verts_st.cuda(), inter_st.cuda())
    torch.cuda.synchronize()
    return z.feats.cpu().numpy().astype(np.float32), z.coords[:, 1:].cpu().numpy().astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs_dir', required=True, help='dir of <specimen>/{skeleton,body}.glb')
    ap.add_argument('--out_dir', required=True, help='dir to save .npz SLAT pairs')
    ap.add_argument('--resolution', type=int, default=512, help='dual-grid O-Voxel resolution')
    ap.add_argument('--enc_pretrained', type=str,
                    default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Loading shape encoder from {args.enc_pretrained} ...')
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()

    specimens = [p for p in Path(args.pairs_dir).iterdir() if p.is_dir()]
    print(f'Encoding {len(specimens)} specimens at resolution {args.resolution}')

    ok = fail = 0
    for sp_dir in specimens:
        out_path = Path(args.out_dir) / f'{sp_dir.name}.npz'
        if out_path.exists():
            continue
        skel = sp_dir / 'skeleton.glb'
        body = sp_dir / 'body.glb'
        if not (skel.exists() and body.exists()):
            fail += 1
            continue
        try:
            sk_f, sk_c = encode_mesh(str(skel), encoder, args.resolution)
            bo_f, bo_c = encode_mesh(str(body), encoder, args.resolution)
            np.savez_compressed(out_path,
                                skel_feats=sk_f, skel_coords=sk_c,
                                body_feats=bo_f, body_coords=bo_c)
            ok += 1
            if ok % 20 == 0:
                print(f'  encoded {ok} pairs (failed {fail})')
        except Exception as e:
            print(f'  [skip] {sp_dir.name}: {e}')
            fail += 1
            torch.cuda.empty_cache()

    print(f'Done. ok={ok}, fail={fail}')


if __name__ == '__main__':
    main()
