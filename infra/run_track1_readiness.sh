#!/usr/bin/env bash
# Read-only Azure control host, metadata-only ffprobe, no Docker/GPU/models.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == /* && "$REV" =~ ^[0-9a-f]{40}$ ]]
exec timeout --signal=TERM --kill-after=3s 123s env PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" "$CODE/infra/track1_readiness.py" <<'PYCONTROL'
from pathlib import Path
import os,runpy,stat,sys
code=Path(sys.argv[1]);script=Path(sys.argv[2])
if script!=code/'infra/track1_readiness.py':raise ValueError('Exact readonly entrypoint required')
for path in (code,script,code/'infra/run_track1_readiness.sh'):
 if (not path.is_absolute() or str(path)!=os.path.abspath(path)
     or any(p.is_symlink() for p in (path,*path.parents))):
  raise ValueError('Canonical source and ancestors required')
 if path!=code and (not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_mode&0o222):
  raise ValueError('Immutable regular stdlib source required')
if not code.is_dir():raise ValueError('Existing immutable code directory required')
sys.argv=[str(script)]
runpy.run_path(str(script),run_name='__main__')
PYCONTROL
