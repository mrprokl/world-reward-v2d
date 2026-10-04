#!/usr/bin/env bash
# CPU-only exact inputs; archive and dense data never leave Azure.
# Source closure: /infra/pose_peer_inventory.py /infra/pose_peer_inputs.py
# /infra/object_pose_smoke.py /infra/body_smoke.py /infra/camera_render.py
# /infra/volume_geometry_loader.py /infra/volume_mesh_pin_inventory.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_pose_peer_inventory/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_peer_inventory.sh" \
 && "$(hostname -s)" == scenesmith-ncc-h100-01 ]] || exit 2
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 timeout --signal=TERM --kill-after=10s 1800s nice -n 10 python3 -I -B - "$CODE" "$@" <<'PY'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.argv=[str(code/'infra/pose_peer_inventory.py'),*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
