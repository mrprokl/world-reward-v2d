#!/usr/bin/env bash
# /infra/hocap_boots_saved_audit.py /infra/hocap_boots_track.py
# /infra/mediapipe_cpu_runtime_verify.py /configs/hocap_boots_protocol_v1.json
# CPU saved-only. Historical HOSTFAIL is immutable; no private references.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_hocap_boots_saved_audit/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_hocap_boots_saved_audit.sh" ]] || exit 2
exec /usr/bin/timeout --signal=TERM --kill-after=10s 180s /usr/bin/env -i \
 PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$@" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
path=code/'infra/hocap_boots_saved_audit.py';sys.argv=[str(path),'--dispatch',*sys.argv[2:]]
runpy.run_path(str(path),run_name='__main__')
PYCONTROL
