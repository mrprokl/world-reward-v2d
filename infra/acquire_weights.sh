#!/usr/bin/env bash
# Run on Azure after bootstrap; use the environment already installed there.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
export WR_ROOT="$ROOT"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
test -x "$ROOT/code/.venv/bin/python"
cd "$ROOT/code"
exec "$ROOT/code/.venv/bin/python" -u infra/acquire_weights.py
