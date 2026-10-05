#!/usr/bin/env bash
set +x
set -euo pipefail
[[ $# == 0 || $# == 1 && ( "$1" == --metadata-bound-range-v2 || "$1" == --dimensioned-directory-v3 ) ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_mmhoi_inventory/code" ]] || exit 2
umask 077
LIMIT=190s
[[ ${1:-} != --dimensioned-directory-v3 ]] || LIMIT=610s
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s "$LIMIT" /usr/bin/python3 -I -B "$CODE/infra/mmhoi_inventory.py" "$@"
