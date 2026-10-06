#!/usr/bin/env bash
# Source closure: /infra/vcoco_full_interaction_join.py /infra/vcoco_full_pose_replica.py
# Source closure: /infra/vcoco_full_interaction_core.py /infra/vcoco_full_hoi_run.py
# VM02 offline CPU saved48 numerical join; no RGB/GPU/models/references.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_full_interaction_join/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_full_interaction_join.sh" ]] || exit 2
exec timeout --signal=TERM --kill-after=10s 615s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/vcoco_full_interaction_join.py" "$@"
