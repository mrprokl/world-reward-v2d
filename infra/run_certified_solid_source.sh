#!/usr/bin/env bash
# Diagnostic only: no geometry proposal, QEM, prediction replacement or GPU.
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_certified_solid_source/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/certified_solid_source_job.py /infra/certified_solid_source.py
# /infra/certified_solid_build.py /infra/mesh_precision_diagnostic.py
exec timeout --signal=TERM --kill-after=10s 450s python3 -I -B "$CODE/infra/certified_solid_source_job.py" "$@"
