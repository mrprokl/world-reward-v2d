#!/usr/bin/env bash
set -euo pipefail
[[ ( $# == 2 || ( $# == 3 && $3 == --query-requalification ) ) && $1 == --episode && $2 =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_object_budget_solid/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/object_budget_solid.py /infra/solid_chart_v2_qualify.py
# /infra/solid_chart_v2_build.py /infra/oriented_solid_compiler.py /infra/object_budget_endpoint.py
exec timeout --signal=TERM --kill-after=10s 1960s python3 -I -B "$CODE/infra/object_budget_solid.py" "$@"
