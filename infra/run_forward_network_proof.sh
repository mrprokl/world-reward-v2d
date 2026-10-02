#!/usr/bin/env bash
set -euo pipefail
if (( $# )); then echo 'Legacy forward network proof has no user-tunable arguments' >&2; exit 2; fi
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
docker run --rm --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/jobs,dst=$ROOT/jobs,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/forward_network_proof.py"
