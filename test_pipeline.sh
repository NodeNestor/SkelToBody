#!/usr/bin/env bash
# Smoke-test the entire SkelToBody pipeline using synthetic CT volumes.
# Run INSIDE the Docker container (not on the Windows host).

set -e
cd /workspace

# Clean previous smoke-test artifacts so steps don't silently skip
rm -rf synth

echo "=== [1/5] synthesize fake CT volumes ==="
mkdir -p synth/overt
for i in 0 1 2; do
    python data/synthetic_ct.py --out synth/overt/specimen_$i/volume.tif --seed $i
done

echo "=== [2/5] segment CT -> (skeleton, body) mesh pairs ==="
python data/batch_segment.py --root synth/overt --out synth/pairs/meshes --workers 2

echo "=== [3/5] encode meshes -> SLAT pairs ==="
python data/mesh_to_slat.py --pairs_dir synth/pairs/meshes --out_dir synth/pairs/slats --resolution 256

echo "=== [4/5] run 10 training steps as sanity check ==="
# No --quantize: bf16 path, avoids bnb Linear4bit/SparseTensor incompat.
# Model is 3B, ~6GB in bf16; LoRA + adapter + activations must fit in remaining VRAM.
python train/train.py \
    --pairs_dir synth/pairs/slats \
    --output_dir synth/checkpoints \
    --steps 10 \
    --save_every 10 \
    --log_every 1 \
    --grad_accum 2

echo "=== [5/5] run inference on the first synthetic skeleton ==="
python infer.py \
    --skeleton synth/pairs/meshes/specimen_0/skeleton.glb \
    --ckpt synth/checkpoints/ckpt_000010.pt \
    --out synth/body_out.glb \
    --steps 4 --cfg 2.0

echo "=== PIPELINE OK ==="
