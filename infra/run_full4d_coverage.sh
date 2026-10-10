#!/usr/bin/env bash
# Remote durable host orchestrator: all models/data/outputs remain on Azure.
set +x
set -euo pipefail
[[ $# == 4 && "$1" == --baseline-report-bytes && "$2" =~ ^[1-9][0-9]*$ \
  && "$3" == --baseline-report-sha256 && "$4" =~ ^[0-9a-f]{64}$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
  && "$CODE" == "$ROOT/jobs/$REV/run_full4d_coverage/code" \
  && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent \
  DOCKER_HOST="unix://$ROOT/docker.sock" WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra" \
  /usr/bin/timeout --signal=TERM --kill-after=30s 28920s \
  /usr/bin/python3 -B "$CODE/infra/full4d_coverage.py" "$@"
