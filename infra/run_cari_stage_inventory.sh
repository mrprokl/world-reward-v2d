#!/usr/bin/env bash
# Stdlib-only readonly Azure control operation. Output is only tiny pin JSON.
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
python3 -I -B - "$CODE" <<'PYSAFE'
from pathlib import Path
import stat,sys
code=Path(sys.argv[1]);script=code/'infra/cari_stage_inventory.py'
if (code.resolve()!=code or not code.is_dir() or script.resolve()!=script
    or any(p.is_symlink() for p in (script,*script.parents))
    or not stat.S_ISREG(script.lstat().st_mode) or script.stat().st_mode&0o222):
 raise ValueError('Immutable canonical code inventory entrypoint required')
PYSAFE
exec env PYTHONDONTWRITEBYTECODE=1 python3 -I -B "$CODE/infra/cari_stage_inventory.py" "$@"
