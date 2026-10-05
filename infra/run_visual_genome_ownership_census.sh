#!/usr/bin/env bash
# Source closure: /infra/visual_genome_ownership_census.py /infra/run_visual_genome_ownership_census.sh /infra/metadata_json_stream.py /infra/mediapipe_cpu_runtime_verify.py /infra/openimages_joint_pair_acquire.py /infra/coco_proposal_prepare.py /configs/visual_genome_ownership_census_v1.json /configs/ownership_pair_prepare_v1.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_visual_genome_ownership_census/code" ]] || exit 2
umask 077
ulimit -v 16777216
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
  WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s 615s \
  nice -n 10 ionice -c 3 /usr/bin/python3 -I -B "$CODE/infra/visual_genome_ownership_census.py"
