#!/usr/bin/env bash
# Source closure: /infra/video_depth_assets.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ $# -eq 0 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_video_depth_assets/code" && "$(uname -s)" == Linux \
 && "$(hostname)" =~ ^(scenesmith-ncc-h100-01|world-reward-ncc-h100-02)$ && "$(id -u)" == 0 ]] || exit 2
# Acquisition-only stdlib process: no CUDA, data mount, pip or inherited secrets.
timeout --signal=TERM --kill-after=10s 600s /usr/bin/env -i \
 PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 /usr/bin/python3 -I -B "$CODE/infra/video_depth_assets.py"
