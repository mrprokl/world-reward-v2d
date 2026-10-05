#!/usr/bin/env bash
# Source closure: /infra/hoi_detr_model_qualify.py /configs/hoi_detr_model_qualify_v1.json /infra/hoi_detr_runtime_verify.py /configs/hoi_detr_runtime_v1.json /infra/hoi_detr_acquire.py /configs/hoi_detr_acquisition_v1.json /infra/mediapipe_cpu_runtime_verify.py /infra/mediapipe_hands_acquire.py /src/world_reward/hoi_detr_observations.py
# Original full model: procedural default or explicit licensed external RGB probe.
# Never challenge images, dataset builders or GT inputs.
set -euo pipefail
[[ ( $# == 0 || ( $# == 1 && "$1" == --external-rgb ) ) && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_hoi_detr_model_qualify/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_hoi_detr_model_qualify.sh" ]] || exit 2
exec env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s 1860s /usr/bin/python3 -I -B "$CODE/infra/hoi_detr_model_qualify.py" "$@"
