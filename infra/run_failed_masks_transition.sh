#!/usr/bin/env bash
# Azure-host CPU only: preserve an independently pinned failed mask stage.
# Source closure: /infra/failed_masks_transition.py.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_failed_masks_transition/code" ]] || exit 2
# Cooperating frontend lock only; do not reset/restart the failed original unit.
[[ -d "$ROOT/jobs" && ! -L "$ROOT/jobs/.world-reward-h100.lock" ]] || exit 1
if [[ -e "$ROOT/jobs/.world-reward-h100.lock" ]];then
 [[ -f "$ROOT/jobs/.world-reward-h100.lock" ]] || exit 1
fi
exec 9>>"$ROOT/jobs/.world-reward-h100.lock"
flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || exit 1
ulimit -v 1048576
env PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 30s \
 python3 -I -B "$CODE/infra/failed_masks_transition.py"
