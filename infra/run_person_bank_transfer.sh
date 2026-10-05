#!/usr/bin/env bash
# Source closure: /infra/person_bank_transfer.py /infra/articulated_runtime_transfer.py /infra/mediapipe_cpu_runtime_verify.py
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_person_bank_transfer/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_person_bank_transfer.sh" ]] || exit 2
exec env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
    timeout --signal=TERM --kill-after=5s 305s /usr/bin/python3 -I -B "$CODE/infra/person_bank_transfer.py" "$@"
