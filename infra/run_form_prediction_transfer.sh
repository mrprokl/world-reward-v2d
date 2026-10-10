#!/usr/bin/env bash
# Source closure: /infra/form_prediction_transfer.py /infra/full4d_publish.py /infra/form_hoi_external_dev.py /infra/form_hoi_external_acquire.py /infra/form_prediction_after_terminal.py /infra/terminal_success.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$CODE" == "$ROOT/jobs/$REV/run_form_prediction_transfer/code" \
 && "$REV" =~ ^[0-9a-f]{40}$ && "$(id -u)" == 0 ]] || exit 2
if [[ "${1:-}" == --after-terminal ]]; then
 [[ $# == 5 && "$2" =~ ^world-reward-[a-z0-9][a-z0-9-]{0,80}(\.service)?$ \
  && "$3" == publish && "$4" =~ ^[0-9a-f]{40}$ && "$5" =~ ^[0-9a-f]{40}$ ]] || exit 2
 env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
  PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=10s 43320s python3 -B \
  "$CODE/infra/form_prediction_after_terminal.py" "$2" "$4" "$5"
 shift 2
fi
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 PYTHONPATH="$CODE/src:$CODE/infra" PYTHONDONTWRITEBYTECODE=1 \
 timeout --signal=TERM --kill-after=10s 960s python3 -B "$CODE/infra/form_prediction_transfer.py" "$@"
