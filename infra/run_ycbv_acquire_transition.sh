#!/usr/bin/env bash
# Source closure: /infra/ycbv_acquire_transition.py /infra/atomic_metadata.py
# /configs/ycbv_point_failed_acquisition_pins.json
set +x
set -euo pipefail
[[ $# == 0 && "$(id -u)" == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_acquire_transition/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_acquire_transition.sh" && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
timeout --signal=TERM --kill-after=5s 120s /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="$DOCKER_HOST" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" <<'PY'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.argv=[str(code/'infra/ycbv_acquire_transition.py')]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
