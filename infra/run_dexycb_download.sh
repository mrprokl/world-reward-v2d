#!/usr/bin/env bash
# Azure host CPU only: two original public archives, never labels or GPU models.
# Source closure: /infra/dexycb_download.py /infra/dexycb_acquire.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_dexycb_download/code" && "$(uname -s)" == Linux ]] || exit 2
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=130s 7240s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" <<'PYCONTROL'
from pathlib import Path
import runpy, sys
code = Path(sys.argv[1]); driver = code / 'infra/dexycb_download.py'
sys.path.insert(0, str(code / 'infra'))
sys.argv = [str(driver)]
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
