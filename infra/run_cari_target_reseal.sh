#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
OUT="$ROOT/outputs/episode_000000/cari_target_reseal_refined_v1"
[[ "$ROOT" == /srv/scenesmith/world-reward && ! -L "$ROOT" && ! -L "$ROOT/outputs" ]]
[[ ! -L "$ROOT/outputs/episode_000000" && ! -e "$OUT" && ! -L "$OUT" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown 1000:1000 "$OUT"
timeout --signal=TERM --kill-after=10s 303s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user 1000:1000 --entrypoint python --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
 --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
 --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
 --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/cari_target_reseal.py"
