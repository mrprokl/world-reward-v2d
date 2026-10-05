#!/usr/bin/env bash
# Source closure: /infra/articulated_point_study.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_articulated_point_study/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
exec timeout --signal=TERM --kill-after=30s 56300s python3 -I -B "$CODE/infra/articulated_point_study.py"
