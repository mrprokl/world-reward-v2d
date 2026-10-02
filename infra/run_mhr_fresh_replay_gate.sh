#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Fresh replay gate accepts no arguments' >&2; exit 2; }
[[ ! -e "$ROOT/results/mhr-fresh-replay.json" && ! -L "$ROOT/results/mhr-fresh-replay.json" ]] || { echo 'Frozen fresh replay report exists' >&2; exit 2; }
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=10s 63s docker run --rm --gpus all --network none --memory 16g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  "$IMAGE" python "$CODE/infra/mhr_fresh_replay_gate.py"
