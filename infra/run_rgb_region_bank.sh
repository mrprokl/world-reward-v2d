#!/usr/bin/env bash
# Reusable new RGB-only bank, explicit file mounts exclude all scene recipes.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_rgb_region_bank/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_rgb_region_bank.sh" ]] || exit 2
exec timeout --signal=TERM --kill-after=10s 1270s env -i \
 PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 python3 -I -B "$CODE/infra/rgb_region_bank.py" "$@"
