"""
Batch-process a directory of oVert specimens into (skeleton, body) GLB pairs.
Expects layout:
    overt_data/
        <specimen_id>/
            tiff_stack/        or  volume.tif
Output:
    pairs/
        <specimen_id>/
            skeleton.glb
            body.glb
"""
import argparse
import concurrent.futures as futures
from pathlib import Path

from ct_to_meshes import process_volume


def find_volume(specimen_dir: Path) -> Path | None:
    for cand in ('volume.tif', 'volume.tiff', 'ct.tif', 'ct.tiff'):
        if (specimen_dir / cand).exists():
            return specimen_dir / cand
    for sub in ('tiff_stack', 'tiffs', 'slices'):
        d = specimen_dir / sub
        if d.is_dir() and any(d.glob('*.tif*')):
            return d
    tifs = list(specimen_dir.glob('*.tif*'))
    if tifs:
        return specimen_dir if len(tifs) > 5 else tifs[0]
    return None


def _process_one(task: tuple[str, str, int]) -> tuple[str, str]:
    sp_path, out_dir, smooth = task
    sp = Path(sp_path)
    vol = find_volume(sp)
    if vol is None:
        return sp.name, 'no_volume'
    try:
        process_volume(str(vol), out_dir, sp.name, smooth=smooth)
        return sp.name, 'ok'
    except Exception as e:
        return sp.name, f'fail:{e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Root dir containing <specimen_id>/ subdirs')
    ap.add_argument('--out', required=True, help='Output dir for paired meshes')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--smooth', type=int, default=3)
    args = ap.parse_args()

    root = Path(args.root)
    specimens = [p for p in root.iterdir() if p.is_dir()]
    print(f'Found {len(specimens)} specimen dirs')

    tasks = [(str(sp), args.out, args.smooth) for sp in specimens]
    if args.workers <= 1:
        for t in tasks:
            name, status = _process_one(t)
            print(f'{name}\t{status}')
    else:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for name, status in ex.map(_process_one, tasks):
                print(f'{name}\t{status}')


if __name__ == '__main__':
    main()
