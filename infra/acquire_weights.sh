#!/usr/bin/env bash
# Run on Azure after bootstrap; use the environment already installed there.
set -euo pipefail
if [[ $# == 1 && "$1" == --mhr-release-only ]]; then
  ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
  [[ "$(uname -s)" == Linux && "$ROOT" == /srv/scenesmith/world-reward \
    && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/acquire_weights/code" ]] || exit 2
  exec timeout --signal=TERM --kill-after=10s 320s python3 -I -B "$CODE/infra/acquire_weights.py" "$@"
fi
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
export WR_ROOT="$ROOT"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_DISABLE_PROGRESS_BARS=1
test -x "$ROOT/code/.venv/bin/python"
cd "$ROOT/code"
exec "$ROOT/code/.venv/bin/python" -u infra/acquire_weights.py
