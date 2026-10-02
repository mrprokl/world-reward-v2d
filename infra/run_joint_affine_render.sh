#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
DEST="$ROOT/validation/joint_affine_rgb_v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -e "$DEST" && ! -L "$DEST" ]]
mkdir -p "$ROOT/validation"; mkdir "$DEST"; chmod 755 "$DEST"; chown scenesmith:scenesmith "$DEST"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 16g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env HOME=/tmp \
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$DEST,dst=$DEST" "$IMAGE" "$CODE/infra/joint_affine_render.py"
