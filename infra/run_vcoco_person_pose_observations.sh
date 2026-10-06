#!/usr/bin/env bash
# Source closure: /infra/vcoco_person_pose_observations.py /infra/vcoco_observation_replica.py
# Source closure: /infra/dwpose_smoke.py /infra/keypoint_rgb_dwpose.py
# Azure VM01 only; B47 CPU, no private references/model changes/selection.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_person_pose_observations/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_person_pose_observations.sh" ]] || exit 2
exec timeout --signal=TERM --kill-after=10s 615s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/vcoco_person_pose_observations.py" "$@"
