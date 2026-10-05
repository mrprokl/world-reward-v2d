#!/usr/bin/env bash
# Source closure: /infra/hocap_acquire.py /infra/mediapipe_cpu_runtime_verify.py
# Only five public publisher archives on owned Azure data disk; no GPU/model.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_hocap_acquire/code" ]] || exit 2
LEASE="$(python3 -I -B - "$CODE" "$REV" <<'PYBOOTSTRAP'
import json, os, pwd, shutil, sys
from pathlib import Path
code, revision = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(code/'infra'))
import hocap_acquire as h
p, source = h.source_binding(h.runtime(code), code, revision)
target = h.canonical(h.DESTINATION)
h.require(str(target) == p['output'] and not target.exists(), 'Fresh fixed dataset namespace only')
h.require(shutil.disk_usage(target.parent).free >= p['minimum_free_bytes'], 'Available data disk below25GiB')
account = pwd.getpwnam('scenesmith')
h.require(account.pw_uid == account.pw_gid == 1000, 'Expected existing account required')
target.mkdir(mode=0o700); os.chown(target, account.pw_uid, account.pw_gid)
s = target.lstat()
print(json.dumps(dict(device=s.st_dev, inode=s.st_ino, source_sha256=source['closure_sha256'])))
PYBOOTSTRAP
)"
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=65s 3660s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_HOCAP_NAMESPACE_LEASE="$LEASE" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" <<'PYCONTROL'
import runpy, sys
from pathlib import Path
code = Path(sys.argv[1]); sys.path.insert(0, str(code/'infra'))
driver = code/'infra/hocap_acquire.py'; sys.argv = [str(driver)]
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
