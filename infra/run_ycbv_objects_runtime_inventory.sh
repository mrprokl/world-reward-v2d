#!/usr/bin/env bash
# CPU metadata only: /infra/ycbv_objects_runtime_inventory.py /infra/ycbv_point_objects.py
# Existing helper closure authenticates constants; no model or GPU execution.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_objects_runtime_inventory/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_objects_runtime_inventory.sh" && "$(uname -s)" == Linux ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" <<'PY'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.argv=[str(code/'infra/ycbv_objects_runtime_inventory.py')]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
