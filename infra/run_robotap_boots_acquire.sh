#!/usr/bin/env bash
# VM02 host CPU acquisition of pinned external RoboTAP/BootsTAPIR; no Docker/GPU/labels here.
# Source closure: /infra/robotap_boots_acquire.py /configs/robotap_boots_protocol.json.
# Opaque private pickles, licence evidence and disposable ZIP are driver-owned.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_robotap_boots_acquire/code" ]] || exit 2
OUT="$ROOT/validation/robotap_boots_v1"
source_identity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
root,code,revision,executing=sys.argv[1:];root,code,executing=map(Path,(root,code,executing))
if (not root.is_dir() or not code.is_dir() or code!=root/'jobs'/revision/'run_robotap_boots_acquire'/'code'
    or executing!=code/'infra/run_robotap_boots_acquire.sh'):
 raise ValueError('Exact actual immutable RoboTAP/Boots acquisition namespace required')
for path in (root,code,executing):
 if not path.is_absolute() or path.resolve()!=path or any(parent.is_symlink()for parent in(path,*path.parents)):
  raise ValueError('Canonical original source without symlinks required')
for name in ('infra/run_robotap_boots_acquire.sh','infra/robotap_boots_acquire.py',
 'configs/robotap_boots_protocol.json'):
 if not(code/name).is_file():raise ValueError('Complete current and original stdlib acquisition source required')
digest=hashlib.sha256()
for name in ('revision','source-sha256'):
 path=code.parent/name
 if (path.resolve()!=path or any(parent.is_symlink()for parent in(path,*path.parents))
     or not stat.S_ISREG(path.lstat().st_mode)):
  raise ValueError('Canonical regular original dispatch markers required')
 value=path.read_bytes()
 if (name=='revision'and value!=(revision+'\n').encode()
     or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',value)):
  raise ValueError('Exact original revision/archive markers required')
 digest.update(name.encode()+b'\0'+value+b'\0')
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink()or(not stat.S_ISDIR(mode)and not stat.S_ISREG(mode))or mode&0o222:
  raise ValueError('Complete actual code must remain readonly and regular')
 if stat.S_ISREG(mode):
  digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PYSOURCE
}
SOURCE_BEFORE=""
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$SOURCE_BEFORE" ]];then
  SOURCE_AFTER="$(source_identity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$SOURCE_AFTER" != "$SOURCE_BEFORE" ]];then
   echo 'Original RoboTAP/Boots acquisition source/markers changed' >&2;STATUS=1
  fi
 fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SOURCE_BEFORE="$(source_identity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$OUT" <<'PYOUTPUT'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:])
if (out!=root/'validation/robotap_boots_v1'or not out.is_absolute()or out.resolve()!=out
    or any(parent.is_symlink()for parent in(out,*out.parents))or not out.parent.is_dir()or out.exists()):
 raise ValueError('Absent holdout output under existing validation parent required; no overwrite/resume/cleanup')
PYOUTPUT
# Limit virtual memory on the actual Linux control host, not an isolated CPU
# throughput claim. No GPU lock, Docker runtime or package installation occurs.
ulimit -v 16777216
mkdir -m 700 "$OUT"
chown 1000:1000 "$OUT"
export WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV"
export WR_ROBOTAP_OUTPUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1
set +e
timeout --signal=TERM --kill-after=10s 1203s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_ROBOTAP_OUTPUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B - "$CODE" <<'PYCONTROL'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);driver=code/'infra/robotap_boots_acquire.py'
sys.path.insert(0,str(code/'infra'))
sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
STATUS=$?
set -e
exit "$STATUS"
