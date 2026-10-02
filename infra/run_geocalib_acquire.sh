#!/usr/bin/env bash
# Public source/weight acquisition runs on Azure CPU only, never on the laptop.
set -euo pipefail
umask 022
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward ]]
[[ "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ && "$(id -u scenesmith)" == 1000 ]]
python3 - "$ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]);out=root/'weights/geocalib-pinhole-v1-frontend';receipt=root/'results/geocalib-assets-v1.json'
if not root.is_dir() or root.resolve()!=root.absolute():raise ValueError('Canonical task root required')
for path in (out,receipt):
 if path.exists() or path.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents):
  raise FileExistsError('Fresh regular GeoCalib targets/ancestors required')
PY
OUT="$ROOT/weights/geocalib-pinhole-v1-frontend"
mkdir "$OUT";chmod 755 "$OUT";chown "1000:$(id -g scenesmith)" "$OUT"
ulimit -v 2097152
timeout --signal=TERM --kill-after=10s 303s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE_REVISION="$WR_CODE_REVISION" WR_GEOCALIB_OUTPUTS_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 "$CODE/infra/geocalib_assets.py"
