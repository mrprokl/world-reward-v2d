#!/usr/bin/env bash
# Exact blind closure: /infra/ycbv_point_objects.py /src/world_reward/pointmap.py
# /src/world_reward/mesh_geometry.py; no acquisition, recipe or private models.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_point_objects/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_point_objects.sh" && "$(uname -s)" == Linux ]] || exit 2
# Existing lock only: open readonly, never truncate/create; retained throughout.
exec 9<"$ROOT/jobs/.world-reward-h100.lock"
exec /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_OBJECTS_MODE=host PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B "$CODE/infra/ycbv_point_objects.py"
