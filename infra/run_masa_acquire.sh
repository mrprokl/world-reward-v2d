#!/usr/bin/env bash
# Source closure: /infra/masa_acquire.py /configs/masa_acquisition_v1.json /infra/mediapipe_cpu_runtime_verify.py /infra/mediapipe_hands_acquire.py
# Public source/checkpoint bytes only; no package installation/model/image/data.
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 ]] || exit 2
[[ "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_masa_acquire/code" ]] || exit 2
ulimit -v 2097152
exec timeout --signal=TERM --kill-after=1s 610s python3 -I -B - "$ROOT" "$CODE" "$REV" <<'PYCONTROL'
import importlib.util,json,os,pwd,shutil,signal,sys,time
from pathlib import Path
started=time.monotonic()
def expired(*_): raise TimeoutError('Bounded acquisition bootstrap')
signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,600)
root,code,revision=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
path=code/'infra/masa_acquire.py'
spec=importlib.util.spec_from_file_location('wr_masa_bootstrap',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rt,mp=m.helpers(code);source=m.binding(rt,root,code,revision);m.manifest(rt,code)
rt.require(not(root/m.REPORT).exists(),'Fresh receipt required before bootstrap')
rt.require(m.BASE.parent.is_dir() and shutil.disk_usage(m.BASE.parent).free>=m.FREE,'Minimum free space unavailable')
account=pwd.getpwnam('scenesmith');rt.require(account.pw_uid==1000,'Exact unprivileged identity required')
lease=mp.create_namespace_lease([m.BASE],account.pw_uid,account.pw_gid,source['closure_sha256'])
os.initgroups(account.pw_name,account.pw_gid);os.setgid(account.pw_gid);os.setuid(account.pw_uid)
os.environ['WR_NAMESPACE_LEASE']=json.dumps(lease)
os.environ['WR_ACQUISITION_STARTED']=repr(started)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.execv(sys.executable,[sys.executable,'-I','-B','-c',
 'import runpy,sys;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name="__main__")',str(path)])
PYCONTROL
