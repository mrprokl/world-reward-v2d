#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
"$ROOT/code/.venv/bin/python" "$CODE/infra/restore_source.py"
docker build --network none --tag world-reward/cari4d-source:0.1 \
  --file "$CODE/infra/Dockerfile.cari4d_source" \
  "$ROOT/vendor/video_to_data/reconstruction/modules"
docker image inspect world-reward/cari4d-source:0.1 --format '{{json .}}' \
  > "$ROOT/results/image-cari4d-source.json"
