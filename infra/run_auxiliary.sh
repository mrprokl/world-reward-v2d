#!/usr/bin/env bash
# Small immutable source snapshot; downloads go directly to the Azure disk.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require an immutable committed job code snapshot}"
"$ROOT/code/.venv/bin/python" "$CODE/infra/acquire_auxiliary.py"
