#!/usr/bin/env bash
# Source closure: /infra/full4d_focus.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
  && "$CODE" == "$ROOT/jobs/$REV/run_full4d_focus/code" \
  && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
exec /usr/bin/timeout --signal=TERM --kill-after=15s 360s \
  /usr/bin/python3 -B "$CODE/infra/full4d_focus.py"
