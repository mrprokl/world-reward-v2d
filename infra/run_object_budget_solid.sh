#!/usr/bin/env bash
set -euo pipefail
if [[ ( $# == 4 && $1 == --episode && $2 =~ ^(0|[1-9]|[12][0-9])$ && $3 == --domain && $4 == surface ) ||
      ( $# == 4 && $1 == --domain && $2 == surface && $3 == --control && $4 == surface_consumer_v1 ) ]]; then
  SURFACE=1
else
  [[ ( $# == 2 || ( $# == 3 && $3 == --query-requalification ) ) && $1 == --episode && $2 =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
  SURFACE=0
fi
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_object_budget_solid/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/object_budget_solid.py /infra/solid_chart_v2_qualify.py
# /infra/solid_chart_v2_build.py /infra/oriented_solid_compiler.py /infra/object_budget_endpoint.py
 # Source closure: /infra/surface_qslim_qualify.py /infra/surface_identity_qualify.py
 # /infra/official_pack_geometry.py /infra/mesh_precision_diagnostic.py /src/world_reward/surface_budget.py
 # /src/world_reward/surface_pose_geometry.py
if (( SURFACE )); then
  exec timeout --signal=TERM --kill-after=10s 740s python3 -I -B "$CODE/infra/object_budget_solid.py" "$@"
fi
exec timeout --signal=TERM --kill-after=10s 1960s python3 -I -B "$CODE/infra/object_budget_solid.py" "$@"
