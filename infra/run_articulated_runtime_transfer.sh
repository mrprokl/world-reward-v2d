#!/usr/bin/env bash
# Source closure: /infra/articulated_runtime_transfer.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_articulated_runtime_transfer/code" ]] || exit 2
[[ $# == 1 && "$1" == export || $# == 7 && "$1" == import ]] || exit 2
SECRET="$ROOT/results/articulated-bootstrap-credentials/$REV/$1.blob-url"
# Host-only transient secret, not a source/config/prediction artifact. The
# dispatcher installs it600 in the dedicated700 directory without printing.
exec timeout --signal=TERM --kill-after=30s 7230s python3 -I -B - "$CODE" "$SECRET" "$@" <<'PY'
from pathlib import Path
import os,stat,sys,runpy
code,secret=map(Path,sys.argv[1:3]);s=secret.lstat()
assert secret.resolve()==secret and not any(x.is_symlink()for x in(secret,*secret.parents))
assert stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o600 and 0<s.st_size<8192
os.environ['WR_BOOTSTRAP_BLOB_URL']=secret.read_text();secret.unlink()
sys.argv=[str(code/'infra/articulated_runtime_transfer.py'),*sys.argv[3:]]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
