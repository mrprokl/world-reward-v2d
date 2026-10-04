#!/usr/bin/env bash
# Source closure: /infra/mediapipe_hand_evaluate.py /infra/mediapipe_hand_scan.py /infra/mediapipe_cpu_runtime_verify.py
# Source closure: /src/world_reward/hand_evaluation.py /src/world_reward/hand_observations.py
# Source closure: /infra/hand_mask_infer.py /src/world_reward/hand_mask_evaluation.py /src/world_reward/hand_mask_proposals.py
# Source closure: /src/world_reward/automatic_candidate_bank.py
set +x
set -euo pipefail
[[ $# == 0 ]] || { [[ $# == 2 && "$1" == --mask-cohort && "$2" == v2 ]] || exit 2; }
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_mediapipe_hand_evaluate/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_mediapipe_hand_evaluate.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent \
 DOCKER_HOST="unix://$ROOT/docker.sock" WR_CODE="$CODE" WR_CODE_REVISION="$REV" \
 /usr/bin/timeout --signal=TERM --kill-after=10s 330s /usr/bin/python3 -I -B "$CODE/infra/mediapipe_hand_evaluate.py" "$@"
