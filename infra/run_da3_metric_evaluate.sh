#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; BASE="$ROOT/validation/tudl_rgb_v1"
OUT="$BASE/da3_quality_v1"; [[ ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
mkdir "$OUT"; chown 1000:1000 "$OUT"
timeout --signal=TERM --kill-after=5s 93s docker run --rm --network none --memory 8g --cpus 4 \
 --user 1000:1000 --entrypoint python --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" \
 --env WR_IMAGE_ID="$IMAGE" --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/predictions_v1,dst=$BASE/predictions_v1,readonly" \
 --mount "type=bind,src=$BASE/da3_predictions_v1,dst=$BASE/da3_predictions_v1,readonly" \
 --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/da3_metric_evaluate.py"
