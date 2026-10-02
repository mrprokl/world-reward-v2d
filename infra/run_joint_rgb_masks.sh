#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/grounding:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
INPUT="$ROOT/validation/joint_rgb_v1/inputs"; OUTPUT="$ROOT/validation/joint_rgb_v1/automatic_masks"
[[ ! -L "$OUTPUT" && ! -e "$OUTPUT" && -s "$INPUT/manifest.json" ]]
mkdir "$OUTPUT"; chown scenesmith:scenesmith "$OUTPUT"
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 12g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env HOME="$ROOT/cache" --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env WANDB_MODE=disabled \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$INPUT,dst=$INPUT,readonly" \
 --mount "type=bind,src=$OUTPUT,dst=$OUTPUT" --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
 --mount "type=bind,src=$ROOT/results/image-grounding.json,dst=$ROOT/results/image-grounding.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" "$IMAGE" python "$CODE/infra/joint_rgb_masks.py"
