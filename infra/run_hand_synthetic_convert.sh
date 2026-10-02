#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Public official conversion accepts no arguments' >&2; exit 2; }
BASE="$ROOT/validation/hands_rgb_v1"
[[ ! -e "$BASE/official_proposals" && ! -L "$BASE/official_proposals" ]] || { echo 'Frozen official proposals exist' >&2; exit 2; }
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
mkdir "$BASE/official_proposals"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$BASE/official_proposals"
# Public artifacts only; neither private truth nor the whole validation root is mounted.
timeout --signal=TERM --kill-after=10s 190s docker run --rm --gpus all --network none --memory 16g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" --env WR_IMAGE_ID="$IMAGE" \
  --env WR_HAND_CONVERT_OUTPUT_RESERVED=1 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly" \
  --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
  --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
  --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
  --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" \
  --mount "type=bind,src=$BASE/predictions-v2,dst=$BASE/predictions-v2,readonly" \
  --mount "type=bind,src=$BASE/official_proposals,dst=$BASE/official_proposals" \
  "$IMAGE" python "$CODE/infra/hand_synthetic_convert.py"
