#!/usr/bin/env bash
# Source closure: /infra/ycbv_archive_headers.py /configs/ycbv_point_inventory_failed_pins.json /configs/ycbv_archive_header_failed_pins.json
set +x
set -euo pipefail
[[ $# == 0 && "$(id -u)" == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_archive_headers/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_archive_headers.sh" && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
timeout --signal=TERM --kill-after=5s 310s /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$CODE" <<'PYRUN'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));runpy.run_path(str(code/'infra/ycbv_archive_headers.py'),run_name='__main__')
PYRUN
