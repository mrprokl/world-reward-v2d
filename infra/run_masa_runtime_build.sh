#!/usr/bin/env bash
# Source closure: /infra/masa_runtime_build.py /configs/masa_runtime_v1.json /infra/mediapipe_cpu_runtime_verify.py /infra/mediapipe_hands_acquire.py
# CPU import/installation only; never model/image/dataset/operator execution.
set +x
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_masa_runtime_build/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_masa_runtime_build.sh" ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 1810s /usr/bin/python3 -I -B - "$CODE/infra/masa_runtime_build.py" <<'PYBUILD'
import os,runpy,sys,time
os.environ['WR_BUILD_STARTED']=repr(time.monotonic())
path=sys.argv[1];sys.argv=[path]
runpy.run_path(path,run_name='__main__')
PYBUILD
