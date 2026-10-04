#!/usr/bin/env bash
set -euo pipefail
if (( $# != 0 ));then
 [[ $# == 1 && "$1" == --reuse-qualified-runtime ]] || exit 2
fi
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?}"
REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$CODE" == "$ROOT/jobs/$REV/run_certified_solid_build/code" ]] || exit 2
[[ -f "$CODE/../revision" && ! -L "$CODE/../revision" && -f "$CODE/../source-sha256" && ! -L "$CODE/../source-sha256" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
# Source closure: /infra/certified_solid_build.py /infra/certified_solid_query.cpp
# /infra/certified_solid_controls.py /src/world_reward/oriented_solid_forest.py
exec python3 -I -B "$CODE/infra/certified_solid_build.py" "$@"
