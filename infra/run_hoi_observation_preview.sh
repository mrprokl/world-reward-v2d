#!/usr/bin/env bash
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_hoi_observation_preview/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_hoi_observation_preview.sh" ]] || exit 2
exec env -i PATH=/usr/bin:/bin timeout --signal=TERM --kill-after=5s 150s /usr/bin/python3 -I -B "$CODE/infra/hoi_observation_preview.py" --code "$CODE" --revision "$REV"
