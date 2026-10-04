#!/usr/bin/env bash
# Azure-private CPU transfer only; no arrays/models/GT are decoded or used.
# Source closure: /infra/pose_return_pull.py /infra/pose_return_inputs.py /infra/pose_peer_inputs.py
# /infra/frontend_peer_receive.py /infra/pose_peer_server.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_pose_return_pull/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_return_pull.sh" && "$(hostname -s)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 timeout --signal=TERM --kill-after=10s 600s nice -n 15 python3 -I -B - "$CODE" "$@" <<'PY'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.argv=[str(code/'infra/pose_return_pull.py'),*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
