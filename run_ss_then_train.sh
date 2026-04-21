#!/usr/bin/env bash
# Full proper-fix pipeline: SS encoding + SS flow training + two-stage inference
# Run inside the docker container.

set -e
cd /workspace

RESOLUTION=${RESOLUTION:-128}
PAIRS=${PAIRS:-real_pairs/meshes}
SLATS=${SLATS:-real_pairs/slats_128}
SS_STEPS=${SS_STEPS:-6000}

echo "=== [A] ensure shape SLATs exist ==="
if [ ! -d "$SLATS" ]; then
    python data/mesh_to_slat.py --pairs_dir $PAIRS --out_dir $SLATS --resolution $RESOLUTION
else
    echo "  (already present)"
fi

echo "=== [B] encode SS latents into the existing pair files ==="
python data/mesh_to_ss.py --pairs_dir $PAIRS --slats_dir $SLATS --input_res 64

echo "=== [C] train the SS flow ==="
python train/train_ss.py \
    --pairs_dir $SLATS \
    --output_dir checkpoints/ss_run1 \
    --steps $SS_STEPS \
    --save_every 500 --log_every 25 \
    --grad_accum 4 --lr 1e-4

echo "=== DONE. SS ckpt in checkpoints/ss_run1/, shape ckpt in checkpoints/real_run1/ ==="
