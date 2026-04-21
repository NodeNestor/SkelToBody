"""
Second tokenizer: encode mesh -> 16x16x16x8 Sparse Structure latent via TRELLIS.2's
SS VAE. This complements the SLAT tokenizer (mesh_to_slat.py) and is required
to train the SS flow stage (which predicts body voxel occupancy from skeleton
voxel occupancy).

Output: extends the existing .npz pair files in-place with fields:
    skel_ss  (8, 16, 16, 16)  float16
    body_ss  (8, 16, 16, 16)  float16
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import trimesh

import o_voxel
import trellis2.models as models


torch.set_grad_enabled(False)


def mesh_to_occupancy_grid(glb_path: str, res: int = 64) -> torch.Tensor:
    """Return a dense (1, 1, res, res, res) float tensor with 1.0 at occupied voxels."""
    m = trimesh.load(glb_path, process=False, force='mesh')
    v = torch.from_numpy(np.asarray(m.vertices, dtype=np.float32))
    f = torch.from_numpy(np.asarray(m.faces, dtype=np.int64))
    v = torch.clamp(v, -0.5 + 1e-6, 0.5 - 1e-6)
    voxel_indices, _, _ = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=v, faces=f,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2,
        timing=False,
    )
    grid = torch.zeros(1, 1, res, res, res, dtype=torch.float32)
    idx = voxel_indices.long().clamp(0, res - 1)
    grid[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    return grid


def encode_ss(glb_path: str, encoder, res: int = 64) -> np.ndarray:
    grid = mesh_to_occupancy_grid(glb_path, res=res).cuda()
    z = encoder(grid)
    return z.detach().cpu().numpy().astype(np.float16)[0]  # drop batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs_dir', required=True, help='dir of <specimen>/{skeleton,body}.glb')
    ap.add_argument('--slats_dir', required=True,
                    help='dir of <specimen>.npz (output of mesh_to_slat.py); we extend in place')
    ap.add_argument('--input_res', type=int, default=64, help='occupancy grid resolution going in')
    ap.add_argument('--enc_pretrained', type=str,
                    default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16')
    args = ap.parse_args()

    print(f'[ss] loading {args.enc_pretrained}')
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()

    pairs_dir = Path(args.pairs_dir)
    slats_dir = Path(args.slats_dir)
    specimens = sorted([p for p in pairs_dir.iterdir() if p.is_dir()])
    print(f'[ss] encoding {len(specimens)} specimens at {args.input_res}^3 -> 16^3 latent')

    ok = fail = 0
    for sp_dir in specimens:
        slat_path = slats_dir / f'{sp_dir.name}.npz'
        if not slat_path.exists():
            fail += 1; continue
        try:
            skel_ss = encode_ss(str(sp_dir / 'skeleton.glb'), encoder, args.input_res)
            body_ss = encode_ss(str(sp_dir / 'body.glb'), encoder, args.input_res)
            # merge with existing slat data
            prev = dict(np.load(slat_path))
            prev['skel_ss'] = skel_ss
            prev['body_ss'] = body_ss
            np.savez_compressed(slat_path, **prev)
            ok += 1
            if ok % 20 == 0:
                print(f'  ss-encoded {ok} pairs  (failed {fail})')
        except Exception as e:
            fail += 1
            print(f'  [skip] {sp_dir.name}: {e}')
            torch.cuda.empty_cache()

    print(f'Done. ok={ok}  fail={fail}')


if __name__ == '__main__':
    main()
