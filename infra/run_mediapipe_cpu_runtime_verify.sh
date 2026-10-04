#!/usr/bin/env bash
# Source closure: /infra/mediapipe_cpu_runtime_verify.py /infra/Dockerfile.mediapipe_cpu
# Source closure: /infra/mediapipe_cpu_dependencies_acquire.py /infra/mediapipe_hands_acquire.py
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_mediapipe_cpu_runtime_verify/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_mediapipe_cpu_runtime_verify.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent \
 DOCKER_HOST="unix://$ROOT/docker.sock" DOCKER_BUILDKIT=0 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 755s /usr/bin/python3 -I -B "$CODE/infra/mediapipe_cpu_runtime_verify.py"
