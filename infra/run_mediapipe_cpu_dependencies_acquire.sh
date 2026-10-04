#!/usr/bin/env bash
# Source closure: /infra/mediapipe_cpu_dependencies_acquire.py /infra/mediapipe_hands_acquire.py
# Azure CPU readonly qualification of previously exact-downloaded wheels only.
set -euo pipefail
[[ $# == 1 && "$1" == --verify-acquired ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_mediapipe_cpu_dependencies_acquire/code" && "$(uname -s)" == Linux ]] || exit 2
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=10s 320s runuser -u scenesmith -- env \
 -u WR_NAMESPACE_LEASE WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" <<'PYCONTROL'
from pathlib import Path
import runpy, sys
code = Path(sys.argv[1]); driver = code / 'infra/mediapipe_cpu_dependencies_acquire.py'
sys.path.insert(0, str(code / 'infra')); sys.argv = [str(driver), '--verify-acquired']
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
