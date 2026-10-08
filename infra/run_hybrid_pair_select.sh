#!/usr/bin/env bash
set -euo pipefail
set +x
ROOT=/srv/scenesmith/world-reward
CODE="${WR_CODE:?committed code required}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CODE/src:$CODE/infra"
exec timeout --signal=TERM --kill-after=20s 1260s python3 -B "$CODE/infra/hybrid_pair_select.py"
