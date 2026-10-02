#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
(( $# == 0 )) || { echo 'Perspective RGB inference accepts no arguments' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/perspective_rgb_v1"
OUT="$BASE/predictions_v1"
[[ "$ROOT" == /srv/scenesmith/world-reward && -d "$ROOT" && ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && -d "$BASE/inputs" && ! -L "$BASE/inputs" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3 ]]
mkdir "$OUT";chmod 755 "$OUT";chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$WR_CODE_REVISION" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/weights/geocalib-pinhole-v1-frontend,dst=$ROOT/weights/geocalib-pinhole-v1-frontend,readonly" \
 --mount "type=bind,src=$ROOT/results/geocalib-assets-v1.json,dst=$ROOT/results/geocalib-assets-v1.json,readonly" \
 --mount "type=bind,src=$ROOT/results/geocalib-frontend-smoke-v1.json,dst=$ROOT/results/geocalib-frontend-smoke-v1.json,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/perspective_rgb_infer.py"
