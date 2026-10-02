#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Wait for the exact derivative; don't restart the active builder.
until docker image inspect world-reward/sam3d-runtime:0.1 >/dev/null 2>&1; do
  state=$(systemctl show world-reward-sam3d-runtime-build -p ActiveState --value)
  if [[ "$state" != active && "$state" != activating ]]; then
    echo "Repaired image unavailable, build terminal: $state" >&2; exit 1
  fi
  sleep 30
done
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env HOME="$ROOT/cache" --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH="$CODE/src" --env HF_HOME="$ROOT/weights/sam3d/hf_home" \
  --env TORCH_HOME="$ROOT/weights/sam3d/torch_home" \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  world-reward/sam3d-runtime:0.1 python "$CODE/infra/object_smoke.py" --root "$ROOT"
