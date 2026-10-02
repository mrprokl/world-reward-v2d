#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?Require immutable source}"
REVISION="${WR_CODE_REVISION:?}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
BASE="$ROOT/validation/joint_rgb_v1"
for DIR in "$ROOT" "$ROOT/validation" "$BASE" "$BASE/inputs" "$BASE/predictions" "$BASE/predictions_camera_v1" "$BASE/eval_private"; do
 [[ -d "$DIR" && ! -L "$DIR" ]] || exit 2
done
OUT="$BASE/quality_camera_v1"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
mkdir "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=5s 63s docker run --rm --network none --memory 8g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/predictions,dst=$BASE/predictions,readonly" \
 --mount "type=bind,src=$BASE/predictions_camera_v1,dst=$BASE/predictions_camera_v1,readonly" \
 --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" python "$CODE/infra/joint_rgb_evaluate.py" --camera-source learned
