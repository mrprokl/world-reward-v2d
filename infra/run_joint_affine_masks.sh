#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]]
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/grounding:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
BASE="$ROOT/validation/joint_affine_rgb_v1"; INPUT="$BASE/inputs"; OUT="$BASE/automatic_masks"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" && -s "$INPUT/manifest.json" ]]
mkdir "$OUT"; chmod 755 "$OUT"; chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 183s docker run --rm --gpus all --network none --memory 16g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" \
 --env HOME=/tmp --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env WANDB_MODE=disabled --env XDG_CACHE_HOME=/tmp/world-reward-cache --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$INPUT,dst=$INPUT,readonly" \
 --mount "type=bind,src=$ROOT/weights/grounding_dino,dst=$ROOT/weights/grounding_dino,readonly" \
 --mount "type=bind,src=$ROOT/weights/sam2,dst=$ROOT/weights/sam2,readonly" \
 --mount "type=bind,src=$ROOT/results/image-grounding.json,dst=$ROOT/results/image-grounding.json,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/joint_affine_masks.py"
