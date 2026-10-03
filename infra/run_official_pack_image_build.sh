#!/usr/bin/env bash
# Azure CPU-only image acquisition/build, never model/data/GPU work.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_official_pack_image_build/code" ]]
python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PYSAFE'
from pathlib import Path
import re,stat,sys
root,code=map(Path,sys.argv[1:3]);revision=sys.argv[3];entry=Path(sys.argv[4])
if (not root.is_dir() or not code.is_dir() or entry!=code/'infra/run_official_pack_image_build.sh'
    or any(path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)) for path in (root,code,entry))):
 raise ValueError('Canonical actual immutable CPU launcher source required')
markers={}
for name in ('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Canonical original identity markers required')
 markers[name]=path.read_bytes()
if markers['revision']!=(revision+'\n').encode() or not re.fullmatch(b'[0-9a-f]{64}\n',markers['source-sha256']):
 raise ValueError('Original source archive/revision markers differ')
for path in (code,*code.rglob('*')):
 if path.is_symlink() or (not path.is_dir() and not stat.S_ISREG(path.lstat().st_mode)) or path.stat().st_mode&0o222:
  raise ValueError('Complete immutable original CPU source closure required')
for name in ('infra/official_pack_image_build.py','infra/run_official_pack_image_build.sh'):
 if not (code/name).is_file():raise ValueError('Complete own CPU builder closure required')
report=root/'results/official-pack-image-build.json'
if report.exists() or report.is_symlink() or not report.parent.is_dir():raise ValueError('Exclusive build report required')
if {name:(code.parent/name).read_bytes() for name in markers}!=markers:raise ValueError('Source markers changed during preflight')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
timeout --signal=TERM --kill-after=10s 303s python3 -I -B "$CODE/infra/official_pack_image_build.py"
