#!/usr/bin/env bash
# Source closure: /infra/vcoco_full_hoi_run.py /infra/vcoco_public_observation_context.py
# Source closure: /infra/vcoco_cardinality_observation_adapters.py /infra/hoi_detr_native_model.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_full_hoi_observations/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_full_hoi_observations.sh" ]] || exit 2
umask 077
ulimit -v 8388608
exec timeout --signal=TERM --kill-after=10s 1860s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/vcoco_full_hoi_run.py" "$@"
