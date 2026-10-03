#!/usr/bin/env bash
# Public HTTPS host-Python acquisition only, matching DA3: no Docker/GPU/install.
set -euo pipefail
umask 022
(( $# == 0 )) || { echo 'Pinned DWPose acquisition accepts no arguments' >&2; exit 2; }
ROOT="${WR_ROOT:?Require Azure task root}"
CODE="${WR_CODE:?Require immutable committed source}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$(uname -s)" == Linux && "$(id -u)" == 0 ]]
[[ "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ && "$(id -u scenesmith)" == 1000 ]]
python3 - "$ROOT" "$CODE" <<'PY'
from pathlib import Path
import sys
root, code = map(Path, sys.argv[1:])
if not root.is_dir() or root.resolve() != root.absolute() or not code.is_dir() or code.resolve() != code.absolute():
    raise ValueError('Canonical task root/immutable source required')
for path in (root/'weights/dwpose_native_v1', root/'results/dwpose-acquisition-v1.json'):
    if path.exists() or path.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents):
        raise FileExistsError('Fresh regular acquisition targets/ancestors required')
PY
OUT="$ROOT/weights/dwpose_native_v1"
mkdir -p "$OUT"; chmod 755 "$OUT"; chown "1000:$(id -g scenesmith)" "$OUT"
ulimit -v 4194304
timeout --signal=TERM --kill-after=10s 603s runuser -u scenesmith -- env -i \
 PATH=/usr/bin:/bin HOME=/home/scenesmith WR_ROOT="$ROOT" WR_CODE_REVISION="$WR_CODE_REVISION" \
 WR_DWPOSE_OUTPUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 "$CODE/infra/dwpose_acquire.py"
