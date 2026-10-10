#!/usr/bin/env bash
# Source closure: /infra/form_public_transfer.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == "$ROOT/jobs/$REV/run_form_public_transfer/code" \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$(id -u)" == 0 ]] || exit 2
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 \
 timeout --signal=TERM --kill-after=10s 600s python3 -B "$CODE/infra/form_public_transfer.py" "$@"
