#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]]
timeout --signal=TERM --kill-after=10s 303s docker run --rm --gpus all --network none --memory 12g --cpus 4 \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env WR_IMAGE_ID="$IMAGE" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/weights/mhr,dst=$ROOT/weights/mhr,readonly" \
  --mount "type=bind,src=$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3,dst=$ROOT/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  "$IMAGE" python "$CODE/infra/mhr_finger_semantics_gate.py" "$@"
