#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
test -s "$ROOT/results/auxiliary-assets.json"
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env HOME="$ROOT/cache" --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH="$CODE/src:/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose" \
  --env TORCH_HOME="$ROOT/weights/cari4d/sam3d_body/torch_home" \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/body_smoke.py" --root "$ROOT"
