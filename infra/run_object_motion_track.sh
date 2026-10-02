#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?}"
BASE="$ROOT/validation/object_motion_v1"
OUT="$BASE/tracking-v2"
[[ ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=10s 303s docker run --rm --gpus all --network none --memory 16g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$BASE/observations,dst=$BASE/observations,readonly" \
 --mount "type=bind,src=$BASE/anchors,dst=$BASE/anchors,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/object_motion_track.py"
