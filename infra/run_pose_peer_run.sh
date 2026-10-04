#!/usr/bin/env bash
# Unchanged full pose algorithm; only independently pinned public inputs RO.
# Source closure: /infra/pose_peer_run.py /infra/pose_peer_inputs.py
# /infra/object_pose_smoke.py /infra/body_smoke.py /infra/camera_render.py
# /infra/volume_geometry_loader.py /infra/volume_mesh_pin_inventory.py
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_pose_peer_run/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_pose_peer_run.sh" \
 && "$(hostname -s)" == world-reward-ncc-h100-02 && "$(uname -s)" == Linux ]] || exit 2
LOCK="$ROOT/jobs/.world-reward-h100.lock"
python3 -I -B - "$LOCK" <<'PY'
from pathlib import Path
import stat,sys
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Original existing GPU lock required')
PY
exec 9<"$LOCK";flock --timeout 43200 9
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=10s 10830s \
 python3 -I -B "$CODE/infra/pose_peer_run.py" "$@"
