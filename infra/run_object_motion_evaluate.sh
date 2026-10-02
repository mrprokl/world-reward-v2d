#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?}"
BASE="$ROOT/validation/object_motion_v1"
OUT="$BASE/quality"
[[ ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/topology-cpu:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=5s 123s docker run --rm --network none --memory 8g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$BASE/inputs/manifest.json,dst=$BASE/inputs/manifest.json,readonly" \
 --mount "type=bind,src=$BASE/observations/report.json,dst=$BASE/observations/report.json,readonly" \
 --mount "type=bind,src=$BASE/anchors,dst=$BASE/anchors,readonly" \
 --mount "type=bind,src=$BASE/tracking,dst=$BASE/tracking,readonly" \
 --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/object_motion_evaluate.py"
