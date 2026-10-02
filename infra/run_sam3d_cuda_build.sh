#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
docker build --network host --tag world-reward/sam3d-cuda:0.1 \
  --file "$CODE/infra/Dockerfile.sam3d_cuda" "$CODE/infra"
docker image inspect world-reward/sam3d-cuda:0.1 --format '{{json .}}' \
  > "$ROOT/results/image-sam3d-cuda.json"
docker run --rm --network none world-reward/sam3d-cuda:0.1 pip freeze \
  > "$ROOT/results/pip-sam3d-cuda.txt"
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env HOME="$ROOT/cache" --env NVIDIA_DRIVER_CAPABILITIES=all \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  world-reward/sam3d-cuda:0.1 python "$CODE/infra/kernel_smoke.py" \
  --report "$ROOT/results/kernels-sam3d-cuda.json" --flash-attention
