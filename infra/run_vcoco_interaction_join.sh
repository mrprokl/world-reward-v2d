#!/usr/bin/env bash
# Source closure: /infra/vcoco_interaction_join_run.py /infra/vcoco_hoi_saved_replica.py
# Source closure: /infra/vcoco_person_pose_observations.py /infra/vcoco_interaction_observations.py
# Azure VM01 B47 CPU, public saved observations only; no GPU/RGB/models/roles.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == scenesmith-ncc-h100-01 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_interaction_join/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_interaction_join.sh" ]] || exit 2
exec timeout --signal=TERM --kill-after=10s 615s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/vcoco_interaction_join_run.py" "$@"
