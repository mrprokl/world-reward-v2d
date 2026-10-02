#!/usr/bin/env bash
# Probe-only numerical diagnosis: never replace existing conversion/submission.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
[[ $# == 2 || $# == 4 ]] || exit 2
[[ "$1" == --episode && ( "$2" == 0 || "$2" == 15 ) ]] || exit 2
EPISODE="$2"; SOURCE=forward
if [[ $# == 4 ]]; then
  [[ "$3" == --bundle-source && ( "$4" == forward || "$4" == refined ) ]] || exit 2
  SOURCE="$4"
fi
[[ "$ROOT" == /srv/scenesmith/world-reward ]]
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"
OUT="$BASE/cari_converter_diagnostic_${SOURCE}_v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/outputs" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=10s 903s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
 --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
 --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
 --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari_converter.py" \
 --episode "$EPISODE" --bundle-source "$SOURCE" --diagnostic-only
