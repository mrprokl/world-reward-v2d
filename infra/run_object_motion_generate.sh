#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?}"
BASE="$ROOT/validation/object_motion_v1"
OUT="$BASE/anchors"
[[ ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
export DOCKER_HOST="unix://$ROOT/docker.sock"
PIN=abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
IMAGE="$(docker image inspect world-reward/sam3d-runtime:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=10s 243s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env HOME="$ROOT/cache" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH="$ROOT/vendor/mv-sam3d/$PIN:$CODE/src" \
 --env HF_HOME="$ROOT/weights/sam3d/hf_home" --env TORCH_HOME="$ROOT/weights/sam3d/torch_home" \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/mv-sam3d/$PIN,dst=$ROOT/vendor/mv-sam3d/$PIN,readonly" \
 --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
 --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
 --mount "type=bind,src=$BASE/observations,dst=$BASE/observations,readonly" \
 --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" --mount "type=bind,src=$OUT,dst=$OUT" \
 "$IMAGE" "$CODE/infra/object_motion_generate.py"
