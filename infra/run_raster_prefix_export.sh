#!/usr/bin/env bash
# Source closure: /infra/raster_prefix_activation.py /infra/mediapipe_cpu_runtime_verify.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_raster_prefix_export/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 && $# == 3 ]] || exit 2
[[ "$1" == "$ROOT/results/raster-prefix-runtime-"*/report.json \
 && "$2" =~ ^[1-9][0-9]*$ && "$3" =~ ^[0-9a-f]{64}$ ]] || exit 2
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/infra:$CODE/src"
exec timeout --signal=TERM --kill-after=10s 180s /usr/bin/python3 -B -c \
 'from pathlib import Path; import sys; from raster_prefix_activation import export; print(export(Path(sys.argv[1]),dict(bytes=int(sys.argv[2]),sha256=sys.argv[3])))' "$@"
