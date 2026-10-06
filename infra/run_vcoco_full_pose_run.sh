#!/usr/bin/env bash
# Source closure: /infra/vcoco_full_pose_run.py /infra/vcoco_public_pose_core.py
# Source closure: /infra/vcoco_public_observation_context.py /infra/dwpose_smoke.py
# VM01/B47 CPU only, original public48, no references/selection or GPU.
set +x
set -euo pipefail
umask 077
ulimit -v 8388608
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_full_pose_run/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_full_pose_run.sh" ]] || exit 2
exec timeout --signal=TERM --kill-after=10s 615s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/vcoco_full_pose_run.py" "$@"
