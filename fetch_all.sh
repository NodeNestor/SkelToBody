#!/usr/bin/env bash
# Fetch every paired (skeleton, body) dataset we can currently access.
# Requires Docker container running.
# Some paths require an account (flagged as MANUAL below).

set -e
cd /workspace
mkdir -p real_pairs/meshes

echo "=== [1] TotalSegmentator — 1228 human CTs ==="
python data/fetch_totalsegmentator.py --out real_pairs/meshes --max_specimens 1228

echo "=== [2] CADS-dataset — multi-source human CTs (superset) ==="
python data/fetch_cads.py --out real_pairs/meshes --max_specimens 500 \
    --subsets TotalSegmentator || echo "CADS failed, continuing"

echo "=== [3] Objaverse-XL — artist-made animal/skeleton/dinosaur subset ==="
python data/fetch_objaverse_animals.py --out real_pairs/objaverse_animals --max_total 500 \
    || echo "Objaverse failed, continuing"

echo "=== [4] SMAL synthetic quadruped pairs (needs /workspace/smal_weights/) ==="
if [ -f /workspace/smal_weights/smal_CVPR2017.pkl ]; then
    python data/synth_smal.py --out real_pairs/meshes --n_samples 1000
else
    echo "  [skip] SMAL weights not found — register at smal.is.tue.mpg.de"
fi

echo "=== [5] MorphoSource oVert (MANUAL — needs \$MORPHOSOURCE_TOKEN) ==="
if [ -n "${MORPHOSOURCE_TOKEN:-}" ]; then
    python data/download_morphosource.py --out real_pairs/morphosource \
        --clades Mammalia Aves Squamata Anura Actinopterygii --per_clade 50 \
        --token "$MORPHOSOURCE_TOKEN"
else
    echo "  [skip] MORPHOSOURCE_TOKEN not set"
fi

echo ""
echo "=== COUNTS ==="
echo "meshes/: $(ls real_pairs/meshes 2>/dev/null | wc -l) specimens"
echo "objaverse/: $(ls real_pairs/objaverse_animals 2>/dev/null | wc -l) assets"
