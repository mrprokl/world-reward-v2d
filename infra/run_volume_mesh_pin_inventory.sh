#!/usr/bin/env bash
# Readonly stdlib host control; emits only actual tiny CPU mesh pin JSON.
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
[[ "$CODE" == "$ROOT/jobs/$REV/run_volume_mesh_pin_inventory/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_volume_mesh_pin_inventory.sh" ]]
exec env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" "$CODE/infra/volume_mesh_pin_inventory.py" "$@" <<'PYCONTROL'
from pathlib import Path
import runpy,stat,sys
code=Path(sys.argv[1]);source=code/'infra/volume_mesh_pin_inventory.py'
if (code.resolve()!=code or not code.is_dir() or Path(sys.argv[2])!=source
    or source.resolve()!=source or any(path.is_symlink() for path in(source,*source.parents))
    or not stat.S_ISREG(source.lstat().st_mode) or source.stat().st_mode&0o222):
 raise ValueError('Actual immutable standalone stdlib inventory source required')
for name in ('infra/run_volume_mesh_pin_inventory.sh','infra/object_budget_volume.py'):
 path=code/name
 if (path.resolve()!=path or any(parent.is_symlink()for parent in(path,*path.parents))
     or not stat.S_ISREG(path.lstat().st_mode)or path.stat().st_mode&0o222):raise ValueError('Actual readonly inventory/producer source closure required')
sys.argv=[str(source),*sys.argv[3:]]
runpy.run_path(str(source),run_name='__main__')
PYCONTROL
