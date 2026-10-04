#!/usr/bin/env bash
# Build-only CPU job: no mesh, GPU, native qualification, install or adoption.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_solid_chart_v2_build/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/solid_chart_v2_build.py /infra/mesh_conditioned_chart_v2_source.py
# /infra/mesh_conditioned_chart_v2.hpp /src/world_reward/mesh_conditioning_v2.py
# /infra/object_budget_conditioned.py /infra/certified_solid_source_job.py
# /infra/certified_solid_build.py /infra/mesh_serialization_compile.py
exec timeout --signal=TERM --kill-after=10s 960s python3 -I -B "$CODE/infra/solid_chart_v2_build.py"
