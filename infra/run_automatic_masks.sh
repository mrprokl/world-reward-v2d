#!/usr/bin/env bash
# Run only the existing, pinned SAM2+GroundingDINO runtime. No build or download.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
test -s "$ROOT/results/input-manifest.json"
test -s "$ROOT/weights/grounding_dino/model.safetensors"
test -s "$ROOT/weights/sam2/sam2.1_hiera_large.pt"
docker image inspect world-reward/grounding:0.1 >/dev/null
# WR_CODE must be the launcher-created immutable committed checkout, mounted
# read-only. Episode/hyperparameter arguments are forwarded without evaluation.
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env HOME="$ROOT/cache" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env WANDB_MODE=disabled \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  world-reward/grounding:0.1 python "$CODE/infra/automatic_masks.py" --root "$ROOT" "$@"
