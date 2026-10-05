#!/usr/bin/env bash
# Source closure: /infra/trellis_runtime_acquire.py /configs/trellis_runtime_protocol_v1.json /infra/mediapipe_cpu_runtime_verify.py /infra/mediapipe_hands_acquire.py
# Public source/model bytes only; no builds, imports, inference or datasets.
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_trellis_runtime_acquire/code" ]] || exit 2
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=5s 3630s python3 -I -B - "$ROOT" "$CODE" "$REV" <<'PYCONTROL'
import importlib.util,json,os,pwd,sys
from pathlib import Path
root,code,revision=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
path=code/'infra/trellis_runtime_acquire.py'
spec=importlib.util.spec_from_file_location('wr_trellis_bootstrap',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rt,mp=m.helpers(code);source=m.binding(rt,root,code,revision);config=m.manifest(rt,code)
rt.require(not(root/config['report']).exists()and m.DATA.parent.is_dir(),'Fresh receipt and existing data parent required')
rt.require(m.shutil.disk_usage(m.DATA.parent).free>=config['minimum_free_bytes'],'At least 30GB free before creating owned namespace')
account=pwd.getpwnam('scenesmith');rt.require(account.pw_uid==1000,'Exact unprivileged owner required')
lease=mp.create_namespace_lease([m.DATA],account.pw_uid,account.pw_gid,source['closure_sha256'])
os.initgroups(account.pw_name,account.pw_gid);os.setgid(account.pw_gid);os.setuid(account.pw_uid)
os.environ['WR_NAMESPACE_LEASE']=json.dumps(lease);os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.execv(sys.executable,[sys.executable,'-I','-B','-c',
 'import runpy,sys;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name="__main__")',str(path)])
PYCONTROL
