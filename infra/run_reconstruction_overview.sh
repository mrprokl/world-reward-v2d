#!/usr/bin/env bash
# Closure: /infra/reconstruction_overview.py /infra/reconstruction_preview.py /infra/camera_render.py /infra/cari_clip_inputs.py /infra/mediapipe_cpu_runtime_verify.py
# Frozen export/RGB QA only; no GPU, model, labels, optimisation or crops.
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_reconstruction_overview/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONDONTWRITEBYTECODE=1
exec timeout --signal=TERM --kill-after=10s 210s python3 -I -B "$CODE/infra/reconstruction_overview.py" --host "$ROOT" "$CODE" "$REV"
