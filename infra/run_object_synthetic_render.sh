#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo "No object render gate arguments supported" >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" ]]
mkdir -p "$ROOT/validation"
DEST="$ROOT/validation/objects_rgb_v1"
mkdir "$DEST"
chown scenesmith:scenesmith "$DEST"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 4g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" --env WR_IMAGE_ID="$IMAGE" --env WR_RENDER_OUTPUT_RESERVED=1 \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$DEST,dst=$DEST" \
  "$IMAGE" python "$CODE/infra/object_synthetic_render.py"
