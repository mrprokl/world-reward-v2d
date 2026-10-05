#!/usr/bin/env bash
# Source closure: /infra/rgb_endpoint_bank.py /infra/owlv2_native_qualify.py /infra/openimages_joint_pair_gdi.py /infra/mediapipe_cpu_runtime_verify.py
set +x
set -euo pipefail
[[ $# == 6 && "$1" == --manifest-bytes && "$3" == --manifest-sha256 && "$5" == --images ]] || exit 2
[[ "$2" =~ ^[1-9][0-9]*$ && "$4" =~ ^[0-9a-f]{64}$ && "$6" =~ ^[1-9][0-9]*$ ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_rgb_endpoint_bank/code" ]] || exit 2
umask 077
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s 640s /usr/bin/python3 -I -B "$CODE/infra/rgb_endpoint_bank.py" "$@"
