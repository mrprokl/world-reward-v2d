#!/usr/bin/env bash
# Full source closure: /infra/hocap_boots_track.py /infra/hocap_amg_bank.py
# /infra/robotap_boots_infer.py /infra/tracker_noise_experiment.py
# /configs/hocap_boots_protocol_v1.json /configs/hocap_point_retention_protocol_v1.json
# No private extraction receipt, raw metadata, labels or challenge assets mounted.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_hocap_boots_track/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_hocap_boots_track.sh" ]] || exit 2
exec /usr/bin/timeout --signal=TERM --kill-after=10s 1860s /usr/bin/env -i \
 PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$@" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
driver=code/'infra/hocap_boots_track.py';sys.argv=[str(driver),'--dispatch',*sys.argv[2:]]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
