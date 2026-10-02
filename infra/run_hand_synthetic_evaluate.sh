#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
BASE="$ROOT/validation/hands_rgb_v1"
OUT="$BASE/quality"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=5s 60s docker run --rm --network none --memory 4g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
  --env WR_HAND_EVAL_RESERVED=1 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
  --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" \
  --mount "type=bind,src=$BASE/predictions,dst=$BASE/predictions,readonly" \
  --mount "type=bind,src=$BASE/official_proposals,dst=$BASE/official_proposals,readonly" \
  --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
  --mount "type=bind,src=$OUT,dst=$OUT" \
  "$IMAGE" python "$CODE/infra/hand_synthetic_evaluate.py" "$@"
