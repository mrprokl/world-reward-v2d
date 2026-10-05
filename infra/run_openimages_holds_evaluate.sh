#!/usr/bin/env bash
# /infra/openimages_holds_evaluate.py /configs/openimages_holds_evaluate_v1.json
# CPU-only private references after every original prediction is sealed.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ $# == 0 && "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_openimages_holds_evaluate/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_openimages_holds_evaluate.sh" ]] || exit 2
exec /usr/bin/timeout --signal=TERM --kill-after=5s 660s /usr/bin/env -i \
 PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B "$CODE/infra/openimages_holds_evaluate.py" --code "$CODE" --revision "$REV"
