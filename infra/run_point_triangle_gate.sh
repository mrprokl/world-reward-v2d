#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/point_triangle_gate.py" "$@"
