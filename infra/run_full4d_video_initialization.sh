#!/usr/bin/env bash
# Source closure: /infra/full4d_video.py
set +x
set -euo pipefail
[[ $# == 1 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
PRODUCER="$1"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
  && "$PRODUCER" =~ ^[0-9a-f]{40}$ && "$REV" != "$PRODUCER" \
  && "$CODE" == "$ROOT/jobs/$REV/run_full4d_video_initialization/code" \
  && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent \
  DOCKER_HOST="unix://$ROOT/docker.sock" WR_ROOT="$ROOT" WR_CODE="$CODE" \
  WR_CODE_REVISION="$REV" WR_PRODUCER_REVISION="$PRODUCER" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra" \
  /usr/bin/timeout --signal=TERM --kill-after=30s 650s \
  /usr/bin/python3 -B "$CODE/infra/full4d_video.py" --initialization-host
