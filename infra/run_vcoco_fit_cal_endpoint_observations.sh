#!/usr/bin/env bash
# Closure: /infra/vcoco_fit_cal_endpoint_run.py /infra/vcoco_fit_cal_endpoint_observations.py
# One public-only48 native endpoint run; original GDI/OWL, no FIT/roles.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_fit_cal_endpoint_observations/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_fit_cal_endpoint_observations.sh" ]] || exit 2
umask 077
ulimit -v 8388608
exec timeout --signal=TERM --kill-after=5s 615s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_CODE="$CODE" WR_CODE_REVISION="$REV" /usr/bin/python3 -I -B "$CODE/infra/vcoco_fit_cal_endpoint_run.py"
