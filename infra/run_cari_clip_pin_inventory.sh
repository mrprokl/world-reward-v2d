#!/usr/bin/env bash
# Readonly stdlib control host: outputs only tiny actual frontend pin JSON.
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
# The inventory revision may be newer than the dispatched frontend. The CLI
# independently supplies that original producer revision and unchanged source
# SHA; Python verifies both rather than trusting a report's first-seen values.
exec env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" "$CODE/infra/cari_clip_pin_inventory.py" "$@" <<'PYCONTROL'
from pathlib import Path
import runpy,stat,sys
code=Path(sys.argv[1]);names=('infra/cari_clip_pin_inventory.py','infra/cari_clip_inputs.py','infra/cari96_inputs.py')
if code.resolve()!=code or not code.is_dir():raise ValueError('Canonical immutable code required')
if Path(sys.argv[2])!=code/names[0]:raise ValueError('Exact source-bound inventory entrypoint required')
for name in names:
 path=code/name
 if (path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents))
     or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_mode&0o222):
  raise ValueError('Immutable regular report-only helper closure required')
sys.path.insert(0,str(code/'infra'))
sys.argv=[str(code/names[0]),*sys.argv[3:]]
runpy.run_path(sys.argv[0],run_name='__main__')
PYCONTROL
