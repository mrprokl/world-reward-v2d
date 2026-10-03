#!/usr/bin/env bash
# CPU read-only asset/Git/Docker-inspect inventory; NO export, transfer or GPU.
# Source closure: /infra/frontend_replica_inventory.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_replica_inventory/code" && "$(uname -s)" == Linux ]] || exit 2
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
from pathlib import Path
import hashlib,re,stat,sys
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
if not root.is_dir() or not code.is_dir() or entry!=code/'infra/run_frontend_replica_inventory.sh':raise ValueError('Actual inventory source required')
digest=hashlib.sha256()
for path in (root,code,entry):
 if path.resolve()!=path or any(p.is_symlink() for p in(path,*path.parents)):raise ValueError('Canonical inventory namespace required')
for name in('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Original dispatch markers required')
 raw=path.read_bytes()
 if(name=='revision' and raw!=(rev+'\n').encode() or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):raise ValueError('Exact original dispatch markers required')
 digest.update(name.encode()+b'\0'+raw)
for path in(code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or mode&0o222 or not(stat.S_ISREG(mode) or stat.S_ISDIR(mode)):raise ValueError('Complete source readonly and regular required')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for name in('infra/frontend_replica_inventory.py','infra/run_frontend_replica_inventory.sh'):
 if not(code/name).is_file():raise ValueError('Complete stdlib inventory source required')
print(digest.hexdigest())
PY
}
BEFORE="$(integrity)"
finish() {
 STATUS=$?; trap - EXIT INT TERM; set +e
 AFTER="$(integrity)"; CHECK=$?
 if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Frozen inventory source changed' >&2;STATUS=1;fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
OUT="$ROOT/results/frontend-replica-inventory-$REV"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
path=Path(sys.argv[1])
if not path.parent.is_dir() or path.exists() or path.is_symlink() or path.resolve()!=path or any(p.is_symlink() for p in path.parents):raise ValueError('Absent owned inventory output required; no overwrite/resume')
PY
mkdir -m 700 "$OUT"
export WR_FRONTEND_INVENTORY_RESERVED=1 PYTHONDONTWRITEBYTECODE=1
# Host root can read the task-private socket. No Docker run/save/load/build,
# GPU lock, credentials, challenge data or private validation path is queried.
ulimit -v 4194304
timeout --signal=TERM --kill-after=10s 603s nice -n 15 ionice -c 3 python3 -I -B "$CODE/infra/frontend_replica_inventory.py"
