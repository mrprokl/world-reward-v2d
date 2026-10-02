#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[0-9a-f]{64}$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "${WR_CODE_REVISION:?}" =~ ^[0-9a-f]{40}$ ]]
OUT="$ROOT/transfer/perspective-v1"
[[ ! -L "$ROOT" && ! -L "$ROOT/transfer" && ! -L "$OUT" && -f "$OUT/cohort.tar" && ! -L "$OUT/cohort.tar" && ! -e "$OUT/import.json" && ! -L "$OUT/import.json" ]]
chmod 711 "$ROOT/transfer"
runuser -u scenesmith -- test -r "$OUT/cohort.tar"
timeout --signal=TERM --kill-after=5s 63s runuser -u scenesmith -- env \
 WR_ROOT="$ROOT" PYTHONPATH="$CODE/src" PYTHONDONTWRITEBYTECODE=1 python3 "$CODE/infra/perspective_rgb_transfer.py" \
 extract --archive "$OUT/cohort.tar" --sha256 "$1" > "$OUT/import.json"
chmod 444 "$OUT/import.json"
