#!/usr/bin/env bash
# Source closure: /infra/articulated_runtime_transfer.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_articulated_runtime_transfer/code" ]] || exit 2
[[ $# == 1 && "$1" == export || $# == 7 && "$1" == import ]] || exit 2
exec timeout --signal=TERM --kill-after=30s 7230s python3 -I -B - "$CODE" "$@" <<'PY'
from pathlib import Path
import os,sys,runpy,re
code=Path(sys.argv[1]);args=sys.argv[2:]
revision=os.environ['WR_CODE_REVISION'] if args==['export'] else args[args.index('--export-revision')+1]
assert re.fullmatch('[0-9a-f]{40}',revision)
# Unsigned private blob URL; the narrowly scoped VM identity provides an
# in-memory bearer token. No token/SAS appears in commands or files.
os.environ['WR_BOOTSTRAP_MANAGED_IDENTITY']='1'
os.environ['WR_BOOTSTRAP_BLOB_URL']='https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+revision+'.tar'
sys.argv=[str(code/'infra/articulated_runtime_transfer.py'),*args]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
