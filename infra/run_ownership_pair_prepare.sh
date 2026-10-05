#!/usr/bin/env bash
# Source closure: /infra/run_ownership_pair_prepare.sh /infra/ownership_pair_prepare.py /infra/ownership_pair_census.py /infra/openimages_joint_pair_census.py /infra/openimages_joint_pair_acquire.py /infra/coco_proposal_prepare.py /infra/coco_endpoint_prepare.py /infra/mediapipe_cpu_runtime_verify.py /infra/run_coco_proposal_prepare.sh /infra/run_coco_endpoint_prepare.sh /infra/run_ownership_pair_census.sh /infra/run_ownership_pair_census_v2.sh /configs/ownership_pair_prepare_v1.json /configs/ownership_pair_census_v1.json /configs/ownership_pair_census_v2.json /configs/coco_proposal_v1.json /configs/coco_endpoint_v2.json /configs/proposal_external_census_v1.json
set +x
set -euo pipefail
[[ $# == 1 && "$1" == --select || $# == 5 && "$1" == --acquire ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_ownership_pair_prepare/code" ]] || exit 2
LIMIT=190; [[ "$1" != --acquire ]] || LIMIT=310
umask 077
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
  WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s "${LIMIT}s" \
  /usr/bin/python3 -I -B "$CODE/infra/ownership_pair_prepare.py" "$@"
