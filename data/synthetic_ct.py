"""
Generate a synthetic CT-like volume that roughly mimics an oVert specimen:
an ellipsoid 'body' with a simplified 'skeleton' (spine + ribs + skull).
Used to smoke-test the entire pipeline (segment -> mesh -> SLAT -> train)
without needing a real MorphoSource download.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import tifffile


def make_ellipsoid(shape, center, radii) -> np.ndarray:
    Z, Y, X = np.indices(shape, dtype=np.float32)
    cz, cy, cx = center
    rz, ry, rx = radii
    return ((Z - cz) / rz) ** 2 + ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2 <= 1.0


def make_cylinder(shape, p0, p1, radius) -> np.ndarray:
    Z, Y, X = np.indices(shape, dtype=np.float32)
    pts = np.stack([Z, Y, X], axis=-1)
    p0 = np.array(p0, dtype=np.float32)
    p1 = np.array(p1, dtype=np.float32)
    axis = p1 - p0
    L = np.linalg.norm(axis)
    axis /= L + 1e-8
    v = pts - p0
    t = (v @ axis).clip(0, L)
    proj = p0 + t[..., None] * axis
    d = np.linalg.norm(pts - proj, axis=-1)
    return d <= radius


def synthesize(shape=(256, 128, 128), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    vol = np.zeros(shape, dtype=np.float32)

    body_mask = make_ellipsoid(shape, (Z * 0.5, Y * 0.5, X * 0.5),
                               (Z * 0.45, Y * 0.35, X * 0.35))
    vol[body_mask] = 0.35  # soft tissue

    # spine: cylinder along Z
    spine = make_cylinder(shape, (Z * 0.12, Y * 0.5, X * 0.5),
                          (Z * 0.88, Y * 0.5, X * 0.5), radius=3.0)
    # skull: small dense sphere at head
    skull = make_ellipsoid(shape, (Z * 0.9, Y * 0.5, X * 0.5),
                           (Z * 0.08, Y * 0.12, X * 0.12))
    # ribs: series of angled cylinders off the spine
    ribs = np.zeros(shape, dtype=bool)
    for z_frac in np.linspace(0.25, 0.7, 10):
        for side in (-1, 1):
            ribs |= make_cylinder(
                shape,
                (Z * z_frac, Y * 0.5, X * 0.5),
                (Z * z_frac, Y * 0.5 + side * Y * 0.28, X * 0.5 + side * X * 0.18),
                radius=1.8,
            )
    # limbs: four bones
    limbs = np.zeros(shape, dtype=bool)
    for z_frac, y_dir in [(0.3, 1), (0.3, -1), (0.7, 1), (0.7, -1)]:
        limbs |= make_cylinder(
            shape,
            (Z * z_frac, Y * 0.5, X * 0.5),
            (Z * z_frac, Y * 0.5 + y_dir * Y * 0.4, X * 0.2),
            radius=2.0,
        )

    bone = spine | skull | ribs | limbs
    vol[bone] = 1.0  # dense bone

    # noise
    vol += rng.normal(0, 0.02, shape).astype(np.float32)
    # scale to 16-bit like real CT
    vol = np.clip(vol, 0, 1)
    vol = (vol * 60000).astype(np.uint16)
    return vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output TIFF path (single multi-page)')
    ap.add_argument('--shape', type=int, nargs=3, default=(256, 128, 128), help='Z Y X')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    vol = synthesize(tuple(args.shape), args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(args.out, vol, compression='deflate')
    print(f'Wrote {args.out}  shape={vol.shape}  dtype={vol.dtype}  '
          f'body~{(vol > 12000).sum()/vol.size:.2%}  bone~{(vol > 45000).sum()/vol.size:.2%}')


if __name__ == '__main__':
    main()
