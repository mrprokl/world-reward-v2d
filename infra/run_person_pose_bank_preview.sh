#!/usr/bin/env bash
# CPU display only; all saved native banks, no model, target or heavy local data.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_person_pose_bank_preview/code" ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CODE/infra"
timeout --signal=TERM --kill-after=10s 100s python3 -B "$CODE/infra/person_pose_bank_preview.py"
