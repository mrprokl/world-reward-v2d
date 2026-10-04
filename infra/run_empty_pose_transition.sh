#!/usr/bin/env bash
# Source closure: /infra/empty_pose_transition.py /infra/failed_masks_transition.py
# /infra/volume_mesh_pin_inventory.py /infra/run_volume_mesh_pin_inventory.sh
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_empty_pose_transition/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_empty_pose_transition.sh" && "$(uname -s)" == Linux && "$(id -u)" == 0 ]] || exit 2
LOCK="$ROOT/jobs/.world-reward-h100.lock"
python3 -I -B - "$LOCK" <<'PYLOCK'
from pathlib import Path
import stat,sys
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Existing canonical GPU lock required')
PYLOCK
exec 9<"$LOCK";flock --nonblock 9
exec env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" "$@" <<'PYTRANSITION'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));driver=code/'infra/empty_pose_transition.py';sys.argv=[str(driver),*sys.argv[2:]]
runpy.run_path(str(driver),run_name='__main__')
PYTRANSITION
