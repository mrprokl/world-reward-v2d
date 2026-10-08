#!/usr/bin/env bash
# Source closure: /infra/full4d_publish.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
  && "$CODE" == "$ROOT/jobs/$REV/run_full4d_publish/code" \
  && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent PYTHONDONTWRITEBYTECODE=1 \
  WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONPATH="$CODE/infra:$CODE/src" \
  /usr/bin/timeout --signal=TERM --kill-after=10s 600s \
  /usr/bin/python3 -B "$CODE/infra/full4d_publish.py" "$@"
