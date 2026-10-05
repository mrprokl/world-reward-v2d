#!/usr/bin/env bash
# Source closure: /infra/vcoco_role_census.py /infra/run_vcoco_role_census.sh /configs/vcoco_role_census_v1.json /infra/metadata_json_stream.py /infra/mediapipe_cpu_runtime_verify.py /infra/openimages_joint_pair_acquire.py /infra/coco_proposal_prepare.py /infra/vcoco_role_stream.py /src/world_reward/vcoco_role_reference.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_role_census/code" ]] || exit 2
umask 077
ulimit -v 8388608
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
  WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s 1215s \
  nice -n 10 ionice -c 3 /usr/bin/python3 -I -B "$CODE/infra/vcoco_role_census.py"
