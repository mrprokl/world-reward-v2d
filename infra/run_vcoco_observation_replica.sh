#!/usr/bin/env bash
# Source closure: /infra/vcoco_observation_replica.py /infra/articulated_runtime_transfer.py
# Source closure: /infra/runtime_image_archive.py /infra/vcoco_pilot_endpoint_bank.py
# Azure-only label-blind bytes; no Docker/image/source clone or native inference.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_observation_replica/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_observation_replica.sh" ]] || exit 2
ulimit -v 262144
exec timeout --signal=TERM --kill-after=10s 315s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B "$CODE/infra/vcoco_observation_replica.py" "$@"
