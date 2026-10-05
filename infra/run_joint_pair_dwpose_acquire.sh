#!/usr/bin/env bash
# Azure-only public assets; original source helpers unchanged, no GPU/install.
# Source closure: /infra/joint_pair_dwpose_acquire.py /infra/dwpose_acquire.py
# Source closure: /infra/dwpose_wheel_audit.py /infra/mediapipe_cpu_runtime_verify.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_joint_pair_dwpose_acquire/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_joint_pair_dwpose_acquire.sh" ]] || exit 2
python3 -I -B - "$ROOT" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
for p in (root/'weights/dwpose_joint_pair_v1',root/'results/dwpose-joint-pair-acquisition-v1.json'):
 if p.exists() or p.is_symlink() or any(x.is_symlink()for x in p.parents):
  raise FileExistsError('Fresh canonical acquisition required')
PY
OUT="$ROOT/weights/dwpose_joint_pair_v1"
mkdir -m 755 "$OUT"; chown "1000:$(id -g scenesmith)" "$OUT"
ulimit -v 4194304
exec timeout --signal=TERM --kill-after=10s 610s runuser -u scenesmith -- env -i \
 PATH=/usr/bin:/bin HOME=/home/scenesmith WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 WR_OUTPUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/joint_pair_dwpose_acquire.py"
