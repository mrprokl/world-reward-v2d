#!/usr/bin/env bash
# Source closure: /infra/form_hoi_external_dev.py /infra/form_hoi_external_acquire.py /infra/mediapipe_cpu_runtime_verify.py
# Four frozen FORM DEV only; first archive reuse, opaque private references, no GPU/model.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_form_hoi_external_dev/code" \
 && "$(id -u)" == 0 ]] || exit 2
LEASE="$(python3 -I -B - "$CODE" "$REV" <<'PYBOOTSTRAP'
import json,os,pwd,shutil,sys
from pathlib import Path
code,revision=Path(sys.argv[1]),sys.argv[2]
sys.path.insert(0,str(code/'infra'))
import form_hoi_external_dev as h
rt=h.runtime(code);cfg,_,source,_=h.source_binding(rt,code,revision)
base=h.canonical(h.DATA)
if not base.exists():base.mkdir(mode=0o755)
h.require(shutil.disk_usage(base).free>=cfg['minimum_free_bytes'],'Azure disk below6GiB')
target=h.canonical(base/revision);h.require(not target.exists(),'Fresh four-DEV output required')
account=pwd.getpwnam('scenesmith');h.require(account.pw_uid==account.pw_gid==1000,'Existing UID1000 required')
target.mkdir(mode=0o700);os.chown(target,1000,1000);s=target.lstat()
print(json.dumps(dict(device=s.st_dev,inode=s.st_ino,source_sha256=source['closure_sha256'])))
PYBOOTSTRAP
)"
ulimit -v 3145728
exec timeout --signal=TERM --kill-after=20s 1560s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_FORM_DEV_NAMESPACE_LEASE="$LEASE" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
driver=code/'infra/form_hoi_external_dev.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
