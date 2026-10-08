#!/usr/bin/env bash
set +x
set -euo pipefail
export DOCKER_HOST=unix:///srv/scenesmith/world-reward/docker.sock
export PYTHONPATH="${WR_CODE:?}/src:$WR_CODE/infra"
export PYTHONDONTWRITEBYTECODE=1
exec timeout --signal=TERM --kill-after=20s 360s python3 -B "$WR_CODE/infra/gemini_initial.py"
