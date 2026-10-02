#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?Require immutable source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
timeout --signal=TERM --kill-after=10s 60s docker run --rm --network none --memory 8g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env CUDA_VISIBLE_DEVICES='' --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/results/cari-refinement-assets.json,dst=$ROOT/results/cari-refinement-assets.json,readonly" \
 world-reward/cari4d-source:0.1 "$CODE/infra/cari_refinement_preflight.py"
