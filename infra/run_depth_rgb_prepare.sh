#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
BASE="$ROOT/validation/depth_rgb_v1"
[[ -d "$ROOT" && ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -e "$BASE" && ! -L "$BASE" ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir "$BASE"; chmod 700 "$BASE"; chown scenesmith:scenesmith "$BASE"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env PYTHONPATH="$CODE/src:$CODE/infra" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
 --mount "type=bind,src=$ROOT/results/mhr-finger-semantics-v4.json,dst=$ROOT/results/mhr-finger-semantics-v4.json,readonly" \
 --mount "type=bind,src=$BASE,dst=$BASE" "$IMAGE" "$CODE/infra/depth_rgb_render.py"
