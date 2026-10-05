#!/usr/bin/env bash
# Source closure: /infra/run_vcoco_metadata_acquire.sh /infra/vcoco_metadata_acquire.py /infra/visual_genome_metadata_acquire.py /infra/mediapipe_cpu_runtime_verify.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_metadata_acquire/code" ]] || exit 2
umask 077
ulimit -v 8388608
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
  WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s 615s \
  nice -n 10 ionice -c 3 /usr/bin/python3 -I -B "$CODE/infra/vcoco_metadata_acquire.py"
