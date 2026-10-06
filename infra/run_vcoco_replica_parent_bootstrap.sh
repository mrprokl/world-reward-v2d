#!/usr/bin/env bash
# Source closure: /infra/vcoco_replica_parent_bootstrap.py /infra/vcoco_observation_replica.py
# Only private-parent bootstrap on VM01; no replica retry, Blob or model here.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_replica_parent_bootstrap/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_replica_parent_bootstrap.sh" ]] || exit 2
exec timeout --signal=TERM --kill-after=5s 45s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/vcoco_replica_parent_bootstrap.py" "$@"
