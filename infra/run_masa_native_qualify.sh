#!/usr/bin/env bash
# Source closure: /infra/masa_native_qualify.py /configs/masa_native_qualification_v1.json /infra/mediapipe_cpu_runtime_verify.py /infra/masa_acquire.py /configs/masa_acquisition_v1.json /infra/masa_runtime_build.py /configs/masa_runtime_v1.json /configs/masa_sm90_build_v1.json
# One procedural CUDA/model contract; no RGB dataset, GT or tracking filters.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_masa_native_qualify/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_masa_native_qualify.sh" ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 /usr/bin/timeout --signal=TERM --kill-after=10s 630s /usr/bin/python3 -I -B "$CODE/infra/masa_native_qualify.py" \
 --revision "$REV" "$@"
