#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_cgal_sum_control/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/cgal_sum_control.py /infra/cgal_sum_control.cpp
# /infra/certified_solid_build.py /infra/certified_solid_source_job.py
exec timeout --signal=TERM --kill-after=35s 640s python3 -I -B "$CODE/infra/cgal_sum_control.py"
