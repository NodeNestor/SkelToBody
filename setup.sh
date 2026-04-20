#!/usr/bin/env bash
# One-shot setup: clone TRELLIS.2 upstream + submodules, then build the Docker image.
# Re-runnable: will skip steps that are already done.
set -e

if [ ! -d "TRELLIS.2" ]; then
    echo "[setup] cloning microsoft/TRELLIS.2 ..."
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git
else
    echo "[setup] TRELLIS.2 already present; fetching submodules just in case"
    (cd TRELLIS.2 && git submodule update --init --recursive)
fi

mkdir -p cache overt_data pairs checkpoints smal_weights inference_out

echo "[setup] building docker image (first run ~20 min) ..."
docker compose build

echo ""
echo "[setup] done. Start a shell in the container with:"
echo "    docker compose run --rm skel2body bash"
