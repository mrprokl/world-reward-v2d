#!/usr/bin/env bash
# Source closure: /infra/form_eval_numba_runtime.py /infra/mediapipe_cpu_runtime_verify.py
# Build a NEW child on CPU02; no data/reference/model/GPU, no base mutation.
set +x
set -euo pipefail
[[ $# == 0 && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_form_eval_numba_runtime/code" ]] || exit 2
# Separate CPU-runtime build lease, never the H100/model lease. Read-only flock
# does not truncate another producer's lock and prevents duplicate child builds.
LOCK="$ROOT/jobs/.world-reward-form-eval-numba-runtime.lock"
/usr/bin/python3 -I -B - "$LOCK" <<'PYLOCK'
import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1])
if p.resolve()!=p or any(x.is_symlink() for x in(p,*p.parents)):raise SystemExit(2)
try:fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o400);os.close(fd)
except FileExistsError:pass
s=p.lstat()
if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=0:raise SystemExit(2)
PYLOCK
exec 9<"$LOCK";flock -n 9
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" \
 WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra" \
 timeout --signal=TERM --kill-after=20s 960s /usr/bin/python3 -B \
 "$CODE/infra/form_eval_numba_runtime.py" build
