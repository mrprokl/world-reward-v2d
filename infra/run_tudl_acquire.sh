#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
(( $# == 0 )) || { echo 'Pinned TUD-L acquisition accepts no arguments' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/validation/tudl_rgb_v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/validation" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir "$OUT"
chmod 755 "$OUT"
chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
export WR_ROOT="$ROOT" WR_TUDL_OUTPUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1
ulimit -v 16777216
timeout --signal=TERM --kill-after=10s 603s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" WR_CODE_REVISION="$WR_CODE_REVISION" WR_TUDL_OUTPUT_RESERVED=1 PYTHONDONTWRITEBYTECODE=1 \
 python3 "$CODE/infra/tudl_acquire.py"
