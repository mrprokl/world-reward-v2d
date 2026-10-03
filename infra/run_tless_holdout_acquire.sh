#!/usr/bin/env bash
# Azure host CPU acquisition only; no GPU, Docker, installation or inference.
# Source closure: /infra/tless_holdout_acquire.py /infra/tudl_holdout_acquire.py
# /infra/tudl_acquire.py /infra/tudl_holdout_inputs.py /configs/tless_frame_holdout_protocol.json.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_tless_holdout_acquire/code" ]] || exit 2
OUT="$ROOT/validation/tless_frame_holdout_v1"
source_identity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
root,code,revision,executing=sys.argv[1:]; root,code,executing=map(Path,(root,code,executing))
if (not root.is_dir() or not code.is_dir() or code!=root/'jobs'/revision/'run_tless_holdout_acquire'/'code'
    or executing!=code/'infra/run_tless_holdout_acquire.sh'):
 raise ValueError('Exact immutable T-LESS CPU acquisition source namespace required')
for path in (root,code,executing):
 if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical original source without symlinks required')
for name in ('infra/run_tless_holdout_acquire.sh','infra/tless_holdout_acquire.py','infra/tudl_holdout_acquire.py',
             'infra/tudl_acquire.py','infra/tudl_holdout_inputs.py','configs/tless_frame_holdout_protocol.json'):
 if not (code/name).is_file(): raise ValueError('Complete frozen stdlib helper/protocol closure required')
digest=hashlib.sha256()
for name in ('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)) or not stat.S_ISREG(path.lstat().st_mode):
  raise ValueError('Original regular canonical dispatch markers required')
 value=path.read_bytes()
 if (name=='revision' and value!=(revision+'\n').encode()
     or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',value)):
  raise ValueError('Exact original revision/archive markers required')
 digest.update(name.encode()+b'\0'+value+b'\0')
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)) or mode&0o222:
  raise ValueError('Complete source snapshot must remain readonly and regular')
 if stat.S_ISREG(mode):
  digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PYSOURCE
}
SOURCE_BEFORE="$(source_identity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$OUT" <<'PYOUTPUT'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:])
if (out!=root/'validation/tless_frame_holdout_v1' or not out.is_absolute() or out.resolve()!=out
    or any(p.is_symlink() for p in (out,*out.parents)) or not out.parent.is_dir() or out.exists()):
 raise ValueError('Absent canonical holdout under existing validation parent required; no overwrite/resume/cleanup')
PYOUTPUT
ulimit -v 16777216
mkdir -m 755 "$OUT"; chown 1000:1000 "$OUT"
set +e
timeout --signal=TERM --kill-after=10s 603s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_TLESS_HOLDOUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" <<'PYCONTROL'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]); driver=code/'infra/tless_holdout_acquire.py'
sys.path.insert(0,str(code/'infra')); sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
STATUS=$?
set -e
SOURCE_AFTER="$(source_identity)"
if [[ "$SOURCE_AFTER" != "$SOURCE_BEFORE" ]]; then
 echo 'Original T-LESS acquisition source/markers changed' >&2; exit 1
fi
exit "$STATUS"
