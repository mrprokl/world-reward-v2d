#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
DEST="$ROOT/validation/hands_rgb_v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -e "$DEST" && ! -L "$DEST" ]]
mkdir -p "$ROOT/validation"
mkdir "$DEST"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$DEST"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 8g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_RENDER_OUTPUT_RESERVED=1 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$DEST,dst=$DEST" \
  "$IMAGE" python "$CODE/infra/hand_synthetic_render.py" "$@"
