"""
Download CT media from MorphoSource using their public REST API.

MorphoSource v2 API:
    https://www.morphosource.org/api/media?q=...
    Each 'media' item may have a 'download_url' OR require clicking through a
    data-use agreement in the browser.

We restrict to records that are (a) CT image series, and (b) have an Open
access level. For records gated behind a DUA, we print the URL and skip.

Usage:
    python data/download_morphosource.py \
        --out /workspace/overt_data \
        --per_clade 20 \
        --clades mammalia aves squamata anura actinopterygii \
        [--token YOUR_MS_API_TOKEN]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print('pip install requests'); sys.exit(1)


API = 'https://www.morphosource.org/api'


def search(clade: str, per_page: int = 20, token: str | None = None) -> list[dict]:
    """Return a list of media records (CT image series) for a given clade."""
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    params = {
        'q': f'taxonomy_ssim:"{clade}" AND media_type_sim:"CT Image Series"',
        'per_page': per_page,
        'sort': 'created_at desc',
    }
    r = requests.get(f'{API}/media', params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json().get('response', {}).get('media', [])


def describe(m: dict) -> str:
    taxo = (m.get('taxonomy_ssim') or ['?'])[0]
    title = m.get('title_sim', m.get('title_tesim', '?'))
    access = m.get('publication_status_sim', m.get('access_ssim', '?'))
    return f'{m.get("ark")}  {taxo}  access={access}  {title}'


def fetch(m: dict, out_dir: Path, token: str | None) -> tuple[str, str]:
    """Try to download the primary file for a media record.
    Returns (status, detail). status: 'ok' | 'skip' | 'fail'."""
    ark = m.get('ark') or m.get('id')
    safe = ark.replace('/', '_').replace(':', '_')
    specimen_dir = out_dir / safe
    specimen_dir.mkdir(parents=True, exist_ok=True)

    # write record metadata
    (specimen_dir / 'metadata.json').write_text(json.dumps(m, indent=2))

    # MorphoSource stores files per media; fetch the media detail
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        r = requests.get(f'{API}/media/{m.get("id")}', headers=headers, timeout=60)
        r.raise_for_status()
        detail = r.json()
    except Exception as e:
        return 'fail', f'detail fetch error: {e}'

    files = detail.get('files') or detail.get('response', {}).get('files', [])
    if not files:
        return 'skip', 'no files field (likely DUA-gated; open in browser)'

    # Pick the largest TIFF/volume file
    vol_files = [f for f in files if (f.get('name', '').lower().endswith(('.tif', '.tiff', '.zip')))]
    if not vol_files:
        vol_files = files
    vol_files.sort(key=lambda f: f.get('size', 0), reverse=True)
    target = vol_files[0]
    url = target.get('download_url') or target.get('url')
    if not url:
        return 'skip', 'no download_url; DUA required'

    # Stream download
    out_path = specimen_dir / target.get('name', 'volume.tif')
    try:
        with requests.get(url, stream=True, headers=headers, timeout=300) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return 'ok', str(out_path)
    except Exception as e:
        return 'fail', f'download error: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--clades', nargs='+', default=[
        'Mammalia', 'Aves', 'Squamata', 'Anura', 'Actinopterygii',
    ])
    ap.add_argument('--per_clade', type=int, default=20)
    ap.add_argument('--token', default=os.environ.get('MORPHOSOURCE_TOKEN'),
                    help='optional API token (env MORPHOSOURCE_TOKEN)')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    totals = {'ok': 0, 'skip': 0, 'fail': 0}
    for clade in args.clades:
        print(f'\n=== {clade} ===')
        try:
            media = search(clade, per_page=args.per_clade, token=args.token)
        except Exception as e:
            print(f'  search failed: {e}'); continue
        print(f'  found {len(media)} records')
        for m in media:
            print('  ' + describe(m))
            status, detail = fetch(m, out, args.token)
            totals[status] += 1
            print(f'    -> {status}: {detail}')
            time.sleep(0.3)

    print(f'\nDone. ok={totals["ok"]} skip={totals["skip"]} fail={totals["fail"]}')
    if totals['skip']:
        print('Many records are gated by Data-Use Agreements.  Either:')
        print('  1. Get an API token at https://www.morphosource.org and re-run with --token')
        print('  2. Manually approve DUAs in the web UI, then rerun — your account is whitelisted after.')


if __name__ == '__main__':
    main()
