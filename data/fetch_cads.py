"""
Pull CADS-dataset (mrmrx/CADS-dataset) from Hugging Face — a multi-source CT
collection that aggregates TotalSegmentator + MSD + AMOS + KiTS + etc. Most
subsets provide bone + body labels in NIfTI format, so we can reuse the same
segmentation logic as fetch_totalsegmentator.

Usage (inside the container):
    python data/fetch_cads.py --out real_pairs/meshes_cads --max_specimens 200
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))
from ct_to_meshes import threshold_body, volume_to_mesh, normalize_pair


# The CADS subsets each expose a different label scheme. These ID sets are
# documented per-subset on the HF dataset card (TotalSegmentator + overlap).
# We accept any mask voxel >0 that equals a listed bone ID.
BONE_IDS_PER_SUBSET = {
    'TotalSegmentator': set(range(25, 51)) | set(range(69, 79)) | {91} | set(range(92, 118)),
    'default': None,  # use entire mask as "bone-or-organ" and depend on CT threshold for body
}


def find_pair(sample_dir: Path) -> tuple[Path, Path] | None:
    """Generic recursive search — returns (image.nii.gz, mask.nii.gz) under a specimen directory."""
    imgs = list(sample_dir.rglob('*.nii.gz'))
    if not imgs:
        return None
    # Heuristic: files named with '_seg' / 'mask' / 'label' → mask
    masks = [p for p in imgs if any(k in p.name.lower() for k in ('seg', 'mask', 'label'))]
    vols = [p for p in imgs if p not in masks]
    if not (masks and vols):
        return None
    return vols[0], masks[0]


def bones_from_mask(mask: np.ndarray, subset_hint: str | None) -> np.ndarray:
    ids = BONE_IDS_PER_SUBSET.get(subset_hint or 'default') or BONE_IDS_PER_SUBSET['TotalSegmentator']
    return np.isin(mask, np.fromiter(ids, dtype=mask.dtype))


def process_one(ct_nii: Path, mask_nii: Path, out_dir: Path, specimen_id: str, subset_hint: str, smooth: int = 3):
    ct = nib.load(str(ct_nii))
    lbl = nib.load(str(mask_nii))
    vol = np.asarray(ct.dataobj).astype(np.float32)
    mask = np.asarray(lbl.dataobj).astype(np.int32)
    if vol.shape != mask.shape:
        raise RuntimeError(f'{specimen_id}: shape mismatch')
    body = threshold_body(vol)
    skel = bones_from_mask(mask, subset_hint)
    if skel.sum() < 500 or body.sum() < 5000:
        raise RuntimeError(f'{specimen_id}: insufficient coverage')
    body_mesh = volume_to_mesh(body, smooth=smooth)
    skel_mesh = volume_to_mesh(skel, smooth=smooth)
    body_mesh, skel_mesh = normalize_pair(body_mesh, skel_mesh)
    out = out_dir / specimen_id
    out.mkdir(parents=True, exist_ok=True)
    body_mesh.export(out / 'body.glb')
    skel_mesh.export(out / 'skeleton.glb')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache_dir', default='/workspace/cache/hf')
    ap.add_argument('--max_specimens', type=int, default=0)
    ap.add_argument('--subsets', nargs='+',
                    default=['TotalSegmentator'],
                    help='which CADS subset folders to pull (0037_totalsegmentator etc.)')
    ap.add_argument('--smooth', type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print('[hf] snapshot_download mrmrx/CADS-dataset (filtered by subset patterns)')
    patterns = [f'*{s}*' for s in args.subsets]
    local = snapshot_download(
        repo_id='mrmrx/CADS-dataset',
        repo_type='dataset',
        cache_dir=args.cache_dir,
        allow_patterns=patterns,
    )
    print(f'[hf] local root: {local}')

    # Walk to find specimen dirs — CADS stores them under subset/<case>/
    root = Path(local)
    specimens = []
    for subset_dir in root.rglob('*'):
        if subset_dir.is_dir():
            pair = find_pair(subset_dir)
            if pair:
                specimens.append((subset_dir.name, pair))

    print(f'[walk] found {len(specimens)} candidate specimens')
    if args.max_specimens > 0:
        specimens = specimens[:args.max_specimens]

    subset_hint = args.subsets[0]
    ok = fail = 0
    for i, (name, (ct_nii, mask_nii)) in enumerate(specimens, 1):
        try:
            process_one(ct_nii, mask_nii, out, name, subset_hint, smooth=args.smooth)
            ok += 1
            if ok % 10 == 0: print(f'  {ok} pairs done')
        except Exception as e:
            fail += 1
            print(f'  [fail {i}/{len(specimens)}] {name}: {e}')

    print(f'Done. ok={ok} fail={fail}')


if __name__ == '__main__':
    main()
