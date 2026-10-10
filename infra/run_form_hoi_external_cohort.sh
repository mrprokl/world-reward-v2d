#!/usr/bin/env bash
# Source closure: /infra/form_hoi_external_cohort.py /infra/form_hoi_external_predict.py /infra/run_form_hoi_external_predict.sh
# This wrapper intentionally uses the original predictor's immutable ENTRY.
set +x
set -euo pipefail
[[ $# == 2 && "$1" =~ ^(localize|all)$ && "$2" =~ ^[0-9a-f]{40}$ ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_form_hoi_external_predict/code" \
 && "$(hostname)" == scenesmith-ncc-h100-01 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$CODE/src:$CODE/infra"
exec /usr/bin/timeout --signal=TERM --kill-after=30s 62000s \
 /usr/bin/python3 -B "$CODE/infra/form_hoi_external_predict.py" \
 --cohort-stage "$1" --dev-revision "$2"
