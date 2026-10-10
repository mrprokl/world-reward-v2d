#!/usr/bin/env bash
# Source closure: /infra/sam31_late_anchor_run.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_sam31_late_anchor/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONPATH="$CODE/src:$CODE/infra"
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/timeout --signal=TERM --kill-after=20s 390s \
 /usr/bin/python3 -B "$CODE/infra/sam31_late_anchor_run.py"
