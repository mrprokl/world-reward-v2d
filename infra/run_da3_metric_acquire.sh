#!/usr/bin/env bash
# Network is needed only here for pinned public assets; no Docker/GPU/install.
set -euo pipefail
umask 022
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
(( $# == 0 )) || { echo 'Pinned DA3 metric acquisition accepts no arguments' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
[[ "$(id -u scenesmith)" == 1000 ]]
python3 - "$ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]); paths=[root/'weights/research/da3_metric_v1',root/'vendor/research/da3_metric_v1',root/'results/da3-metric-acquisition-v1.json']
if not root.is_dir() or root.resolve()!=root.absolute(): raise ValueError('Canonical task root required')
for path in paths:
    if path.exists() or path.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents):
        raise FileExistsError('DA3 reserved targets/ancestors must be fresh regular paths')
PY
for OUT in "$ROOT/weights/research/da3_metric_v1" "$ROOT/vendor/research/da3_metric_v1"; do
 mkdir -p "$OUT"; chmod 755 "$OUT"; chown "1000:$(id -g scenesmith)" "$OUT"
done
ulimit -v 4194304
timeout --signal=TERM --kill-after=10s 723s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE_REVISION="$WR_CODE_REVISION" WR_DA3_OUTPUTS_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 "$CODE/infra/da3_metric_acquire.py"
