#!/usr/bin/env bash
# Full real-data pipeline on TotalSegmentator-CT-Lite from Hugging Face.
# Run INSIDE the Docker container (docker compose run --rm skel2body bash -c "bash real_run.sh").

set -e
cd /workspace

N_SPECIMENS=${N_SPECIMENS:-100}
STEPS=${STEPS:-8000}
RESOLUTION=${RESOLUTION:-128}

echo "=== [1/3] fetch TotalSegmentator-CT-Lite from HF and segment to (skel, body) pairs ==="
python data/fetch_totalsegmentator.py \
    --out real_pairs/meshes \
    --cache_dir /workspace/cache/hf \
    --max_specimens $N_SPECIMENS \
    --smooth 3

echo "=== [2/3] encode meshes -> SLAT pairs via TRELLIS.2 SC-VAE ==="
python data/mesh_to_slat.py \
    --pairs_dir real_pairs/meshes \
    --out_dir real_pairs/slats_${RESOLUTION} \
    --resolution $RESOLUTION

echo "=== [3/3] train skeleton adapter + LoRA on real pairs ==="
python train/train.py \
    --pairs_dir real_pairs/slats_${RESOLUTION} \
    --output_dir checkpoints/real_run1 \
    --steps $STEPS \
    --save_every 500 \
    --log_every 25 \
    --grad_accum 4 \
    --lr 1e-4

echo "=== DONE. checkpoints in /workspace/checkpoints/real_run1/ ==="
