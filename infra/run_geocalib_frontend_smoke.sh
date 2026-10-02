#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
(( $# == 0 )) || { echo 'GeoCalib frontend smoke accepts no arguments' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
[[ "$ROOT" == /srv/scenesmith/world-reward && -d "$ROOT" && ! -L "$ROOT" && ! -L "$ROOT/results" && ! -L "$ROOT/weights" ]]
ASSETS="$ROOT/weights/geocalib-pinhole-v1-frontend"
RECEIPT="$ROOT/results/geocalib-assets-v1.json"
OUT="$ROOT/results/geocalib-frontend-smoke-v1.json"
[[ -d "$ASSETS" && ! -L "$ASSETS" && -f "$RECEIPT" && ! -L "$RECEIPT" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3 ]]
# A single pre-reserved report is RW, not the entire results or task tree.
(set -o noclobber; : > "$OUT")
chmod 644 "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$WR_CODE_REVISION" --env WR_IMAGE_ID="$IMAGE" --env WR_GEOCALIB_REPORT_RESERVED=1 \
 --env PYTHONPATH="$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ASSETS,dst=$ASSETS,readonly" \
 --mount "type=bind,src=$RECEIPT,dst=$RECEIPT,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/geocalib_frontend.py"
