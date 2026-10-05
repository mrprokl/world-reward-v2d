#!/usr/bin/env bash
# Source closure: /infra/owlv2_native_qualify.py /infra/mediapipe_cpu_runtime_verify.py /infra/bridge_frontend_bindings.py /infra/frontend_selected_assets.py /infra/frontend_sam2_kernel_gate.py
set +x
set -euo pipefail
[[ $# == 6 && "$1" == --acquisition-revision && "$3" == --acquisition-report-bytes && "$5" == --acquisition-report-sha256 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_owlv2_native_qualify/code" ]] || exit 2
umask 077
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s 130s /usr/bin/python3 -I -B "$CODE/infra/owlv2_native_qualify.py" "$@"
