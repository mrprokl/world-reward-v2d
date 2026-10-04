#!/usr/bin/env bash
# Azure host CPU: public source/model bytes only, no installation or inference.
# Source closure: /infra/mediapipe_hands_acquire.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_mediapipe_hands_acquire/code" && "$(uname -s)" == Linux ]] || exit 2
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=10s 320s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" <<'PYCONTROL'
from pathlib import Path
import runpy, sys
driver = Path(sys.argv[1]) / 'infra/mediapipe_hands_acquire.py'
sys.argv = [str(driver)]
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
