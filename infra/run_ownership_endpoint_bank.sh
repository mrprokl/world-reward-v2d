#!/usr/bin/env bash
# Source closure: /infra/ownership_endpoint_bank.py /infra/rgb_endpoint_bank.py /src/world_reward/rgb_bank_inputs.py
set +x
set -euo pipefail
[[ $# == 6 && "$1" == --manifest-bytes && "$3" == --manifest-sha256 && "$5" == --images ]] || exit 2
[[ "$2" =~ ^[1-9][0-9]*$ && "$4" =~ ^[0-9a-f]{64}$ && "$6" =~ ^[1-9][0-9]*$ && "$6" -le 96 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
   && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
   && "$CODE" == "$ROOT/jobs/$REV/run_ownership_endpoint_bank/code" ]] || exit 2
umask 077
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  timeout --signal=TERM --kill-after=5s 640s /usr/bin/python3 -I -B "$CODE/infra/ownership_endpoint_bank.py" "$@"
