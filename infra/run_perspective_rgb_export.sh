#!/usr/bin/env bash
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/transfer/perspective-v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/transfer" && ! -e "$OUT" && ! -L "$OUT" ]]
mkdir -p "$ROOT/transfer"; mkdir "$OUT"; chmod 700 "$OUT"; chown 1000:1000 "$OUT"
timeout --signal=TERM --kill-after=5s 63s nice -n 10 ionice -c 3 runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" PYTHONPATH="$CODE/src" PYTHONDONTWRITEBYTECODE=1 python3 "$CODE/infra/perspective_rgb_transfer.py" \
 export --archive "$OUT/cohort.tar" > "$OUT/export.json"
chmod 400 "$OUT/cohort.tar"; chmod 444 "$OUT/export.json"
