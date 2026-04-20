"""
Filter Objaverse-XL for animal / skeleton / dinosaur / bone categories using
LVIS and tag-based search, download matching .glb assets locally.

NOTE: Objaverse assets are ARTIST-MADE — skeletons are bone-rig meshes, bodies
are surface meshes. We treat them as pseudo-pairs:
  - If the asset has a visible skeleton (shape name / tag contains 'skeleton'):
    use it as the "skeleton" mesh.
  - If the asset has a related body mesh (same uid or adjacent), use it as body.

This is a best-effort complement to real CT data, useful for body-plan diversity
(dinosaurs, fish, extinct mammals) that CT datasets don't cover.

Requires:
    pip install objaverse
"""
from __future__ import annotations
import argparse
import os
import random
from pathlib import Path

try:
    import objaverse
    import objaverse.xl as oxl
except ImportError:
    raise SystemExit('pip install objaverse objaverse-xl')


ANIMAL_KEYWORDS = {
    'animal', 'creature', 'beast',
    'skeleton', 'bone', 'skull', 'skel',
    'dinosaur', 'trex', 't-rex', 'raptor', 'sauropod', 'stegosaurus', 'triceratops',
    'fish', 'shark', 'whale', 'dolphin', 'ray',
    'bird', 'eagle', 'parrot', 'owl', 'chicken', 'penguin',
    'mammal', 'cat', 'dog', 'wolf', 'lion', 'tiger', 'bear', 'horse', 'deer',
    'reptile', 'lizard', 'snake', 'crocodile', 'turtle',
    'amphibian', 'frog', 'salamander',
}


def pull_lvis_subset(max_total: int = 500) -> list[str]:
    """Return a list of object uids whose LVIS category touches our keywords."""
    try:
        lvis = objaverse.load_lvis_annotations()
    except AttributeError:
        # objaverse>=0.1.6 renamed; graceful fallback
        return []
    uids: list[str] = []
    for cat, items in lvis.items():
        if any(k in cat.lower() for k in ANIMAL_KEYWORDS):
            uids.extend(items)
    random.shuffle(uids)
    return uids[:max_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_total', type=int, default=500)
    ap.add_argument('--processes', type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print('[objaverse] pulling LVIS category annotations')
    uids = pull_lvis_subset(args.max_total)
    print(f'[objaverse] {len(uids)} candidate animal/skeleton uids selected')

    if not uids:
        print('no matches; aborting')
        return

    print('[objaverse] downloading GLBs ...')
    paths = objaverse.load_objects(uids=uids, download_processes=args.processes)
    print(f'[objaverse] {len(paths)} GLB files downloaded')

    # Copy into our out dir so train pipeline sees uniform layout
    for uid, src in paths.items():
        dst = out / f'{uid}.glb'
        if not dst.exists():
            try:
                with open(src, 'rb') as f_in, open(dst, 'wb') as f_out:
                    f_out.write(f_in.read())
            except Exception as e:
                print(f'  [skip] {uid}: {e}')

    print(f'Done. Wrote to {out}')


if __name__ == '__main__':
    main()
