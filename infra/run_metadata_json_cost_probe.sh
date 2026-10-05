#!/usr/bin/env bash
# Source closure: /infra/metadata_json_cost_probe.py /infra/run_metadata_json_cost_probe.sh /infra/metadata_json_stream.py /infra/mediapipe_cpu_runtime_verify.py /infra/visual_genome_metadata_acquire.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_metadata_json_cost_probe/code" ]] || exit 2
umask 077
ulimit -v 16777216
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent CUDA_VISIBLE_DEVICES=-1 \
  WR_CODE="$CODE" WR_CODE_REVISION="$REV" timeout --signal=TERM --kill-after=5s 615s \
  nice -n 10 ionice -c 3 /usr/bin/python3 -I -B "$CODE/infra/metadata_json_cost_probe.py"
