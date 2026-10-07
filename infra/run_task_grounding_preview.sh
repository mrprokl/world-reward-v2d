#!/usr/bin/env bash
# Source closure: /infra/task_grounding_preview.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_task_grounding_preview/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent \
 DOCKER_HOST="unix://$ROOT/docker.sock" WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 300s /usr/bin/python3 -B "$CODE/infra/task_grounding_preview.py"
