#!/usr/bin/env bash
# Source closure: /infra/hocap_extract.py /infra/hocap_acquire.py /infra/mediapipe_cpu_runtime_verify.py
# Saved archives only; no network, model, inference or old-source execution.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_hocap_extract/code" ]] || exit 2
LEASE="$(python3 -I -B - "$CODE" "$REV" <<'PYBOOTSTRAP'
import json, os, pwd, shutil, sys
from pathlib import Path
code, revision = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0,str(code/'infra'))
import hocap_extract as h
rt,_ = h.runtime(code); p,source = h.source_binding(rt,code,revision)
target = rt.canonical(Path(p['output']))
rt.require(not target.exists(), 'Fresh fixed extraction namespace required')
rt.require(shutil.disk_usage(target.parent).free >= p['minimum_free_bytes'], 'Selected-extraction disk space insufficient')
account = pwd.getpwnam('scenesmith')
rt.require(account.pw_uid == account.pw_gid == 1000, 'Existing UID1000 account required')
target.mkdir(mode=0o700); os.chown(target,account.pw_uid,account.pw_gid)
s = target.lstat()
print(json.dumps(dict(device=s.st_dev,inode=s.st_ino,source_sha256=source['closure_sha256'])))
PYBOOTSTRAP
)"
ulimit -v 3145728
exec timeout --signal=TERM --kill-after=65s 660s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_HOCAP_EXTRACTION_LEASE="$LEASE" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
driver=code/'infra/hocap_extract.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
