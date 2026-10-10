#!/usr/bin/env bash
# Source closure: /infra/sam31_recovery.py
set +x
set -euo pipefail
[[ $# -le 1 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_sam31_recovery/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONPATH="$CODE/src:$CODE/infra"
exec /usr/bin/timeout --signal=TERM --kill-after=30s 5550s \
 /usr/bin/python3 -B "$CODE/infra/sam31_recovery.py" "$@"
