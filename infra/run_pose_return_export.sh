#!/usr/bin/env bash
# Binary stdout ONLY; forced readonly exporter shares the actual server snapshot.
# Source closure: /infra/pose_return_export.py /infra/pose_return_inputs.py /infra/pose_peer_inputs.py
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_pose_return_server/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_return_export.sh" && "$(hostname -s)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 SSH_CONNECTION="${SSH_CONNECTION:-}" SSH_ORIGINAL_COMMAND="${SSH_ORIGINAL_COMMAND:-}" PYTHONDONTWRITEBYTECODE=1 \
 timeout --signal=TERM --kill-after=10s 600s python3 -I -B - "$CODE" "$@" <<'PY'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.argv=[str(code/'infra/pose_return_export.py'),*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
