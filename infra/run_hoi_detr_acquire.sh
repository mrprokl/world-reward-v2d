#!/usr/bin/env bash
# Source closure: /infra/hoi_detr_acquire.py /configs/hoi_detr_acquisition_v1.json /infra/mediapipe_cpu_runtime_verify.py /infra/mediapipe_hands_acquire.py
# Azure acquisition only. No model/native imports, installs, datasets or GPU.
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_hoi_detr_acquire/code" ]] || exit 2
ulimit -v 2097152
exec env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s 960s python3 -I -B - "$ROOT" "$CODE" "$REV" <<'PYCONTROL'
import importlib.util,json,os,pwd,sys,time
from pathlib import Path
started=time.monotonic()
root,code,revision=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
path=code/'infra/hoi_detr_acquire.py'
spec=importlib.util.spec_from_file_location('wr_hoi_acquire_bootstrap',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rt,mp=m.helpers(code);source=m.binding(rt,root,code,revision);config=m.manifest(rt,code)
rt.require(not(root/config['report']).exists()and m.DATA.parent.is_dir(),'Fresh receipt and existing data parent required')
rt.require(m.shutil.disk_usage(m.DATA.parent).free>=config['minimum_free_bytes'],'At least 16GiB free before namespace creation')
account=pwd.getpwnam('scenesmith');rt.require(account.pw_uid==1000,'Exact unprivileged owner required')
lease=mp.create_namespace_lease([m.DATA],account.pw_uid,account.pw_gid,source['closure_sha256'])
os.initgroups(account.pw_name,account.pw_gid);os.setgid(account.pw_gid);os.setuid(account.pw_uid)
os.environ['WR_NAMESPACE_LEASE']=json.dumps(lease);os.environ['WR_STARTED_MONOTONIC']=str(started)
os.execv(sys.executable,[sys.executable,'-I','-B','-c',
 'import runpy,sys;sys.argv=[sys.argv[1]];runpy.run_path(sys.argv[0],run_name="__main__")',str(path)])
PYCONTROL
