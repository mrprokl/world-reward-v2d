#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
docker build --network host --tag world-reward/sam3d-runtime:0.1 \
  --file "$CODE/infra/Dockerfile.sam3d_runtime" "$CODE/infra"
docker image inspect world-reward/sam3d-runtime:0.1 --format '{{json .}}' \
  > "$ROOT/results/image-sam3d-runtime.json"
