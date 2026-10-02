#!/usr/bin/env bash
# Wait for the exact SAM2 image, then run one automatic segmentation gate.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
until docker image inspect world-reward/sam2:7c0d3b9 >/dev/null 2>&1; do
  state=$(systemctl show world-reward-build.service -p ActiveState --value)
  if [[ "$state" != active && "$state" != activating ]]; then
    echo "SAM2 image unavailable and runtime build terminal: $state" >&2
    exit 1
  fi
  sleep 30
done
# Prove native units, shared identity and official conversion before predictions.
docker run --rm --gpus all --network host \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env PYTHONPATH="$ROOT/code/src" --env HOME="$ROOT/cache" \
  --mount "type=bind,src=$ROOT/code,dst=$ROOT/code,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  world-reward/sam2:7c0d3b9 python "$ROOT/code/infra/mhr_smoke.py" --root "$ROOT"
docker build --network host --file "$ROOT/code/infra/Dockerfile.grounding" \
  --tag world-reward/grounding:0.1 "$ROOT/code/infra"
docker image inspect world-reward/grounding:0.1 --format '{{json .}}' \
  > "$ROOT/results/image-grounding.json"
test -s "$ROOT/weights/grounding_dino/model.safetensors"
docker run --rm --gpus all --network host \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env PYTHONPATH="$ROOT/code/src" --env HOME="$ROOT/cache" \
  --mount "type=bind,src=$ROOT/code,dst=$ROOT/code,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  world-reward/grounding:0.1 python "$ROOT/code/infra/automatic_masks.py" \
  --root "$ROOT" --episode 15
