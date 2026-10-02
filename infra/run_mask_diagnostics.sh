#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
docker run --rm --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env PYTHONPATH="$CODE/src" --env WR_ROOT="$ROOT" \
  --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  world-reward/grounding:0.1 python "$CODE/infra/mask_diagnostics.py"
