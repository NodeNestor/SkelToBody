"""
Pull TotalSegmentator-CT-Lite from Hugging Face and convert each CT+mask pair
into our standard (skeleton.glb, body.glb) format.

Dataset: https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite
Structure:
    Images/<case>.nii.gz   — whole-body CT volume
    Masks/<case>.nii.gz    — integer mask, 117 anatomical classes

Bone class IDs we union into the skeleton mask (from the dataset card):
    25-50  (sacrum + vertebrae S1..C1)
    69-78  (long bones: humerus, femur, scapula, clavicula, hip)
    91     (skull)
    92-115 (ribs left 1-12, right 1-12)
    116-117 (sternum, costal cartilages)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download


BONE_IDS = set(range(25, 51)) | set(range(69, 79)) | {91} | set(range(92, 118))


def bone_mask_from_labels(mask: np.ndarray) -> np.ndarray:
    """Return boolean 3D mask where True = any bone class."""
    flat = np.isin(mask, np.fromiter(BONE_IDS, dtype=mask.dtype))
    return flat


def process_one(ct_nii: Path, mask_nii: Path, out_dir: Path, specimen_id: str, smooth: int = 3):
    """Load CT + labels, make (skeleton, body) meshes, save to out_dir/specimen_id/."""
    # lazy import so we don't need these at top-level parse time
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ct_to_meshes import threshold_body, volume_to_mesh, normalize_pair

    ct = nib.load(str(ct_nii))
    lbl = nib.load(str(mask_nii))
    vol = np.asarray(ct.dataobj).astype(np.float32)
    mask = np.asarray(lbl.dataobj).astype(np.int32)

    if vol.shape != mask.shape:
        raise RuntimeError(f'{specimen_id}: shape mismatch CT{vol.shape} vs mask{mask.shape}')

    # --- body surface: Otsu on CT gives outer skin contour ---
    body_mask = threshold_body(vol)

    # --- skeleton: direct pull from TotalSegmentator bone labels ---
    skel_mask = bone_mask_from_labels(mask)

    if skel_mask.sum() < 500 or body_mask.sum() < 5000:
        raise RuntimeError(f'{specimen_id}: insufficient coverage '
                           f'(bone={skel_mask.sum()}, body={body_mask.sum()})')

    body_mesh = volume_to_mesh(body_mask, smooth=smooth)
    skel_mesh = volume_to_mesh(skel_mask, smooth=smooth)
    body_mesh, skel_mesh = normalize_pair(body_mesh, skel_mesh)

    out = out_dir / specimen_id
    out.mkdir(parents=True, exist_ok=True)
    body_mesh.export(out / 'body.glb')
    skel_mesh.export(out / 'skeleton.glb')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output dir for <specimen>/{skeleton,body}.glb pairs')
    ap.add_argument('--cache_dir', default='/workspace/cache/hf', help='HF cache dir')
    ap.add_argument('--max_specimens', type=int, default=0, help='stop after N pairs (0=all)')
    ap.add_argument('--smooth', type=int, default=3)
    ap.add_argument('--skip_download', action='store_true',
                    help='assume data already present in cache_dir')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print('[hf] downloading TotalSegmentator-CT-Lite (may be ~23 GB) ...')
        local = snapshot_download(
            repo_id='YongchengYAO/TotalSegmentator-CT-Lite',
            repo_type='dataset',
            cache_dir=args.cache_dir,
        )
    else:
        import glob
        cands = glob.glob(os.path.join(args.cache_dir, '**', 'Images*'), recursive=True)
        if not cands:
            raise RuntimeError(f'Could not find Images* under {args.cache_dir}')
        local = str(Path(cands[0]).parent)
    print(f'[hf] data root: {local}')

    # The dataset ships as Images.zip + Masks.zip. Selectively extract only the
    # first N case files (matching pairs) to avoid the full ~22 GB unpack.
    import zipfile
    images_zip = Path(local) / 'Images.zip'
    masks_zip = Path(local) / 'Masks.zip'
    extract_root = Path(local) / 'extracted'
    extract_root.mkdir(parents=True, exist_ok=True)

    def _case_from_member(n: str) -> str:
        name = os.path.basename(n)
        # strip .nii / .nii.gz suffixes
        if name.endswith('.nii.gz'):
            name = name[:-7]
        elif name.endswith('.nii'):
            name = name[:-4]
        return name

    # Peek into both zips and find the intersection of case IDs
    with zipfile.ZipFile(images_zip) as zi, zipfile.ZipFile(masks_zip) as zm:
        img_members = {_case_from_member(n): n for n in zi.namelist() if n.endswith('.nii.gz')}
        mask_members = {_case_from_member(n): n for n in zm.namelist() if n.endswith('.nii.gz')}
        common = sorted(set(img_members) & set(mask_members))
        print(f'[zip] {len(img_members)} image entries, {len(mask_members)} mask entries, '
              f'{len(common)} matched cases')

        selected = common if args.max_specimens <= 0 else common[:args.max_specimens]
        print(f'[zip] extracting {len(selected)} case pairs')

        for case in selected:
            img_name = img_members[case]
            out_img = extract_root / 'Images' / f'{case}.nii.gz'
            out_img.parent.mkdir(parents=True, exist_ok=True)
            if not out_img.exists():
                with zi.open(img_name) as src, open(out_img, 'wb') as dst:
                    dst.write(src.read())
            mask_name = mask_members[case]
            out_mask = extract_root / 'Masks' / f'{case}.nii.gz'
            out_mask.parent.mkdir(parents=True, exist_ok=True)
            if not out_mask.exists():
                with zm.open(mask_name) as src, open(out_mask, 'wb') as dst:
                    dst.write(src.read())

    img_dir = extract_root / 'Images'
    mask_dir = extract_root / 'Masks'
    cases = sorted([p.stem.replace('.nii', '') for p in img_dir.glob('*.nii.gz')])
    print(f'[data] found {len(cases)} cases')

    if args.max_specimens > 0:
        cases = cases[:args.max_specimens]

    ok = fail = 0
    for i, case in enumerate(cases, 1):
        ct_nii = img_dir / f'{case}.nii.gz'
        mask_nii = mask_dir / f'{case}.nii.gz'
        if not (ct_nii.exists() and mask_nii.exists()):
            fail += 1
            print(f'  [skip {i}/{len(cases)}] {case}: missing file')
            continue
        try:
            process_one(ct_nii, mask_nii, out, case, smooth=args.smooth)
            ok += 1
            if ok % 10 == 0:
                print(f'  {ok} pairs done')
        except Exception as e:
            fail += 1
            print(f'  [fail {i}/{len(cases)}] {case}: {e}')

    print(f'Done. ok={ok} fail={fail} -> {out}')


if __name__ == '__main__':
    main()
