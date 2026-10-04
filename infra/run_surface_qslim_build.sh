#!/usr/bin/env bash
# Standalone CPU compile/build-info only; no QEM, data, GPU or installation.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_surface_qslim_build/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/surface_qslim_build.py /infra/surface_qslim.cpp
# /infra/certified_solid_build.py /configs/mesh_serialization_compiler_protocol_v1.json
exec timeout --signal=TERM --kill-after=10s 960s python3 -I -B "$CODE/infra/surface_qslim_build.py"
