#!/usr/bin/env bash
# Closure: /infra/vcoco_fit_cal_acquire.py /infra/vcoco_fit_cal_context.py
# Closure: /infra/vcoco_fit_cal_freeze.py /infra/run_vcoco_fit_cal_freeze.sh
# One original acquisition of all48 frozen identities, no references or model.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_fit_cal_acquire/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_fit_cal_acquire.sh" ]] || exit 2
umask 077
ulimit -v 8388608
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
 WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s 1215s \
 nice -n 10 ionice -c 3 /usr/bin/python3 -I -B "$CODE/infra/vcoco_fit_cal_acquire.py"
