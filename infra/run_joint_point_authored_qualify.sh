#!/usr/bin/env bash
# Source closure: /infra/joint_point_authored_qualify.py
# Fresh authored actual-MHR runtime pair only; never the closed EP21 record.
set -euo pipefail
[[ $# == 0 || ( $# == 2 && "$1" == --control && "$2" == native_repeat ) ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_joint_point_authored_qualify/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
exec timeout --signal=TERM --kill-after=10s 1440s python3 -I -B "$CODE/infra/joint_point_authored_qualify.py" "$@"
