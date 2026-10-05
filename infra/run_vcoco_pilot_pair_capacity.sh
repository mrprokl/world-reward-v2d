#!/usr/bin/env bash
# Closure: /infra/vcoco_pilot_pair_capacity.py /configs/vcoco_pilot_pair_capacity_v1.json
set +x
set -euo pipefail
[[ $# == 28 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_pilot_pair_capacity/code" ]] || exit 2
umask 077
ulimit -v 6291456
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 timeout --signal=TERM --kill-after=5s 195s /usr/bin/python3 -I -B "$CODE/infra/vcoco_pilot_pair_capacity.py" "$@"
