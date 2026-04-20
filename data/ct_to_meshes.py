"""
Load an oVert / MorphoSource CT volume and produce a paired
(skeleton_mesh, body_mesh) in a shared coordinate frame,
normalized to the [-0.5, 0.5]^3 cube that TRELLIS.2 expects.

oVert specimens are fluid-preserved industrial CT: values are raw
16-bit density, not calibrated Hounsfield. We pick thresholds by
histogram (Otsu on bone class, Otsu on body-vs-background).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tifffile
import trimesh
from skimage import measure, filters, morphology


def load_ct_volume(path: str) -> np.ndarray:
    p = Path(path)
    if p.is_dir():
        slices = sorted(p.glob('*.tif*'))
        if not slices:
            raise FileNotFoundError(f'No TIFF slices in {path}')
        vol = np.stack([tifffile.imread(str(s)) for s in slices], axis=0)
    else:
        vol = tifffile.imread(str(p))
    if vol.ndim != 3:
        raise ValueError(f'Expected 3D volume, got shape {vol.shape}')
    return vol.astype(np.float32)


def threshold_body(vol: np.ndarray) -> np.ndarray:
    t = filters.threshold_otsu(vol)
    mask = vol > t
    mask = morphology.binary_closing(mask, morphology.ball(2))
    mask = morphology.remove_small_objects(mask, min_size=1000)
    return mask


def threshold_bone(vol: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    body_vals = vol[body_mask]
    t = filters.threshold_otsu(body_vals)
    bone = (vol > t) & body_mask
    bone = morphology.binary_opening(bone, morphology.ball(1))
    bone = morphology.remove_small_objects(bone, min_size=200)
    return bone


def volume_to_mesh(mask: np.ndarray, spacing=(1, 1, 1), smooth: int = 0) -> trimesh.Trimesh:
    verts, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing, allow_degenerate=False
    )
    m = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    if smooth > 0:
        trimesh.smoothing.filter_taubin(m, iterations=smooth)
    return m


def normalize_pair(body: trimesh.Trimesh, skel: trimesh.Trimesh) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Center and uniformly scale both meshes using the BODY bounding box, so the pair shares a frame."""
    mn = body.vertices.min(0)
    mx = body.vertices.max(0)
    center = (mn + mx) / 2
    scale = 0.99 / float((mx - mn).max())
    for m in (body, skel):
        m.vertices = (m.vertices - center) * scale
    assert body.vertices.min() >= -0.5 and body.vertices.max() <= 0.5
    return body, skel


def process_volume(ct_path: str, out_dir: str, specimen_id: str, smooth: int = 3):
    vol = load_ct_volume(ct_path)
    body_mask = threshold_body(vol)
    bone_mask = threshold_bone(vol, body_mask)

    if bone_mask.sum() < 500 or body_mask.sum() < 5000:
        raise RuntimeError(f'Segmentation failed for {specimen_id} (bone={bone_mask.sum()}, body={body_mask.sum()})')

    body_mesh = volume_to_mesh(body_mask, smooth=smooth)
    skel_mesh = volume_to_mesh(bone_mask, smooth=smooth)
    body_mesh, skel_mesh = normalize_pair(body_mesh, skel_mesh)

    out = Path(out_dir) / specimen_id
    out.mkdir(parents=True, exist_ok=True)
    body_mesh.export(out / 'body.glb')
    skel_mesh.export(out / 'skeleton.glb')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ct', required=True, help='CT volume (TIFF file or dir of TIFF slices)')
    ap.add_argument('--out', required=True, help='Output directory for paired meshes')
    ap.add_argument('--id', required=True, help='Specimen ID (used as subdir name)')
    ap.add_argument('--smooth', type=int, default=3, help='Taubin smoothing iterations')
    args = ap.parse_args()
    out = process_volume(args.ct, args.out, args.id, smooth=args.smooth)
    print(f'Wrote pair to {out}')


if __name__ == '__main__':
    main()
