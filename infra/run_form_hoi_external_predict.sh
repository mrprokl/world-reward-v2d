#!/usr/bin/env bash
# Source closure: /infra/form_hoi_external_predict.py
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_form_hoi_external_predict/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
/usr/bin/python3 -c 'import numpy,scipy' >/dev/null
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra"
SECONDS_LIMIT=15500
for argument in "$@"; do [[ "$argument" != --cohort-stage ]] || SECONDS_LIMIT=62000; done
exec /usr/bin/timeout --signal=TERM --kill-after=30s "${SECONDS_LIMIT}s" \
 /usr/bin/python3 -B "$CODE/infra/form_hoi_external_predict.py" "$@"
