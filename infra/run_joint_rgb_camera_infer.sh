#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Learned-camera joint RGB inference accepts no arguments' >&2; exit 2; }
BASE="$ROOT/validation/joint_rgb_v1"
OUT="$BASE/predictions_camera_v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -L "$BASE" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=10s 603s docker run --rm --gpus all --network none --memory 32g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
 --env PYTHONPATH="$CODE/src:/workspace/v2d_sam3d_body/lib" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env MOMENTUM_ENABLED=0 --env WANDB_MODE=disabled \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/vendor/video_to_data,dst=$ROOT/vendor/video_to_data,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body,dst=$ROOT/weights/cari4d/sam3d_body,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal,dst=$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal,readonly" \
 --mount "type=bind,src=$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c,dst=$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c,readonly" \
 --mount "type=bind,src=$ROOT/results/weights-acquisition.json,dst=$ROOT/results/weights-acquisition.json,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/joint_rgb_infer.py" --camera-source learned
