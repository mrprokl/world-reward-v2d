#!/usr/bin/env bash
# Source closure: /infra/hocap_amg_bank.py /infra/bridge_frontend_bindings.py
# Source closure: /infra/frontend_selected_assets.py /infra/frontend_sam2_kernel_gate.py
# Source closure: /configs/frontend_grounding_source_pins.json /configs/frontend_asset_archive_pins.json
# RGB only: original extraction receipt is authenticated on HOST, never mounted.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_hocap_amg_bank/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_hocap_amg_bank.sh" ]] || exit 2
exec /usr/bin/timeout --signal=TERM --kill-after=10s 960s /usr/bin/env -i \
 PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$@" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
driver=code/'infra/hocap_amg_bank.py';sys.argv=[str(driver),'--dispatch',*sys.argv[2:]]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
