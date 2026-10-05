#!/usr/bin/env bash
# /infra/hocap_point_retention_eval.py /infra/mediapipe_cpu_runtime_verify.py
# /configs/hocap_point_retention_protocol_v1.json /configs/robotap_boots_inference_pins.json
# CPU-only: two sealed predictions/public provenance and ONLY labels.zip references.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_hocap_point_retention_eval/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_hocap_point_retention_eval.sh" ]] || exit 2
exec /usr/bin/timeout --signal=TERM --kill-after=10s 360s /usr/bin/env -i \
 PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B "$CODE/infra/hocap_point_retention_eval.py" --dispatch "$@"
