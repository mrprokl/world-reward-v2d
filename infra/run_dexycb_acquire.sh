#!/usr/bin/env bash
# CPU-only original archives and private bytes; never a predictor/model stage.
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == "$ROOT/jobs/$REV/run_dexycb_acquire/code" && "$REV" =~ ^[0-9a-f]{40}$ ]]
exec timeout --signal=TERM --kill-after=130s 1810s runuser -u scenesmith -- \
 env WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/dexycb_acquire.py" "$@"
