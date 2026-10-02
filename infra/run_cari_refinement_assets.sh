#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?Require immutable source}"
"$ROOT/code/.venv/bin/python" "$CODE/infra/acquire_cari_refinement_assets.py"
