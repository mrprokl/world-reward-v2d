#!/usr/bin/env bash
# Closure: /infra/vcoco_hoi_saved_replica.py
# Exact Azure-only saved HOI banks; no model or RGB/reference decode.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_hoi_saved_replica/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_hoi_saved_replica.sh" \
 && ( "$(hostname)" == scenesmith-ncc-h100-01 || "$(hostname)" == world-reward-ncc-h100-02 ) ]] || exit 2
umask 077
ulimit -v 1048576
exec timeout --signal=TERM --kill-after=5s 195s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 /usr/bin/python3 -I -B "$CODE/infra/vcoco_hoi_saved_replica.py" "$@"
