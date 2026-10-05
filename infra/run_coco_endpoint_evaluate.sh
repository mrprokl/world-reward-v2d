#!/usr/bin/env bash
# Source closure: /infra/coco_endpoint_evaluate.py /infra/rgb_endpoint_bank.py /infra/coco_endpoint_prepare.py /src/world_reward/endpoint_proposal_recall.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_coco_endpoint_evaluate/code" ]] || exit 2
umask 077
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 timeout --signal=TERM --kill-after=5s 190s /usr/bin/python3 -I -B "$CODE/infra/coco_endpoint_evaluate.py" "$@"
