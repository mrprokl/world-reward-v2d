#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable source}"
BASE="$ROOT/validation/objects_rgb_v1"
OUT="$BASE/quality"
[[ ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=5s 60s docker run --rm --network none --memory 8g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$BASE/inputs/manifest.json,dst=$BASE/inputs/manifest.json,readonly" \
  --mount "type=bind,src=$BASE/observations-v2/report.json,dst=$BASE/observations-v2/report.json,readonly" \
  --mount "type=bind,src=$BASE/proposals-v2,dst=$BASE/proposals-v2,readonly" \
  --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
  --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" python "$CODE/infra/object_synthetic_evaluate.py" "$@"
