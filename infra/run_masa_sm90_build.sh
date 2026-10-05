#!/usr/bin/env bash
# Source closure: /infra/masa_sm90_build.py /configs/masa_sm90_build_v1.json /configs/masa_runtime_v1.json /infra/masa_runtime_build.py /infra/mediapipe_cpu_runtime_verify.py /infra/mediapipe_hands_acquire.py
# CPU full original MMCV build only. No GPU, checkpoint, model or dataset.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_masa_sm90_build/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_masa_sm90_build.sh" ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent LANG=C.UTF-8 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 2430s /usr/bin/python3 -I -B - \
 "$CODE/infra/masa_sm90_build.py" --revision "$REV" "$@" <<'PYBUILD'
import os,runpy,sys,time
os.environ['WR_BUILD_STARTED']=repr(time.monotonic())
path=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(path,run_name='__main__')
PYBUILD
