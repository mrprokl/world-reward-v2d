#!/usr/bin/env bash
# Source closure: /infra/raster_prefix_runtime.py /infra/mediapipe_cpu_runtime_verify.py
# One isolated child; no upstream model/data downloads or base mutation.
set +x
set -euo pipefail
[[ $# == 0 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_raster_prefix_runtime/code" ]] || exit 2
# Exclusive lease prevents duplicate inference during actual qualification.
exec 9<"$ROOT/jobs/.world-reward-h100.lock";flock -n 9
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 2
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" \
 WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra" \
 timeout --signal=TERM --kill-after=20s 2100s /usr/bin/python3 -B \
 "$CODE/infra/raster_prefix_runtime.py" build
