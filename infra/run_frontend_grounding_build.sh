#!/usr/bin/env bash
# Azure VM02 CPU-only immutable source/wheel/notice acquisition and NEW v5 child.
# Source closure: /infra/frontend_grounding_build.py /configs/frontend_grounding_source_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_grounding_build/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_frontend_grounding_build.sh" ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 1803s \
 /usr/bin/python3 -I -B "$CODE/infra/frontend_grounding_build.py"
