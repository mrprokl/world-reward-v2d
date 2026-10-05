#!/usr/bin/env bash
# Source closure: /infra/coco_endpoint_prepare.py /infra/coco_proposal_prepare.py /infra/mediapipe_cpu_runtime_verify.py /infra/openimages_joint_pair_acquire.py /infra/run_coco_proposal_prepare.sh /configs/coco_proposal_v1.json /configs/coco_endpoint_v2.json
set +x
set -euo pipefail
[[ $# == 1 && "$1" == --census || $# == 5 && "$1" == --acquire ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_coco_endpoint_prepare/code" ]] || exit 2
umask 077
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s 310s /usr/bin/python3 -I -B "$CODE/infra/coco_endpoint_prepare.py" "$@"
