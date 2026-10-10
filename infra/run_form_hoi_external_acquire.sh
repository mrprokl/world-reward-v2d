#!/usr/bin/env bash
# Source closure: /infra/form_hoi_external_acquire.py /infra/mediapipe_cpu_runtime_verify.py
# Public publisher DEV only, opaque references quarantined; no GPU/model or RESERVED.
set +x
set -euo pipefail
[[ $# -le 1 ]] || exit 2
PROFILE="${1:-inventory_first}"
[[ "$PROFILE" == inventory_first || "$PROFILE" == acquire_dev ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_form_hoi_external_acquire/code" \
 && "$(id -u)" == 0 ]] || exit 2
LEASE="$(python3 -I -B - "$CODE" "$REV" "$PROFILE" <<'PYBOOTSTRAP'
import json, os, pwd, shutil, sys
from pathlib import Path
code, revision, profile = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(code/'infra'))
import form_hoi_external_acquire as h
p, source = h.source_binding(h.runtime(code), code, revision)
h.cohort(p, profile)
base = h.canonical(h.DATA)
if not base.exists(): base.mkdir(mode=0o755)
h.require(base.is_dir() and shutil.disk_usage(base).free >= 6 << 30, 'Available Azure data disk below6GiB')
target = h.canonical(base/f'{profile}-{revision}')
h.require(not target.exists(), 'Fresh fixed data namespace required')
account = pwd.getpwnam('scenesmith')
h.require(account.pw_uid == account.pw_gid == 1000, 'Expected existing UID1000 account required')
target.mkdir(mode=0o700); os.chown(target, account.pw_uid, account.pw_gid)
s = target.lstat()
print(json.dumps(dict(device=s.st_dev, inode=s.st_ino, source_sha256=source['closure_sha256'])))
PYBOOTSTRAP
)"
SECONDS_CAP=660
[[ "$PROFILE" == inventory_first ]] || SECONDS_CAP=1260
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=20s "${SECONDS_CAP}s" runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_FORM_EXTERNAL_NAMESPACE_LEASE="$LEASE" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" "$PROFILE" <<'PYCONTROL'
import runpy, sys
from pathlib import Path
code, profile = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(code/'infra'))
driver = code/'infra/form_hoi_external_acquire.py'; sys.argv = [str(driver), profile]
runpy.run_path(str(driver), run_name='__main__')
PYCONTROL
