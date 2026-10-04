#!/usr/bin/env bash
# Azure CPU only; original dense pose report and archive never leave Azure.
# Source closure: /infra/pose_return_inventory.py /infra/pose_return_inputs.py /infra/pose_peer_inputs.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_pose_return_inventory/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_return_inventory.sh" && "$(hostname -s)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 timeout --signal=TERM --kill-after=10s 300s nice -n 15 python3 -I -B - "$CODE" "$@" <<'PY'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.argv=[str(code/'infra/pose_return_inventory.py'),*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
