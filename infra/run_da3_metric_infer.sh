#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
(( $# == 0 )) || { echo 'DA3 metric inference accepts no arguments' >&2; exit 2; }
BASE="$ROOT/validation/tudl_rgb_v1"
OUT="$BASE/da3_predictions_v1"
[[ "$ROOT" == /srv/scenesmith/world-reward && ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/da3-metric:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=10s 603s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/research/da3_metric_v1,dst=$ROOT/vendor/research/da3_metric_v1,readonly" \
 --mount "type=bind,src=$ROOT/weights/research/da3_metric_v1,dst=$ROOT/weights/research/da3_metric_v1,readonly" \
 --mount "type=bind,src=$ROOT/results/da3-metric-acquisition-v1.json,dst=$ROOT/results/da3-metric-acquisition-v1.json,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/da3_metric_infer.py"
