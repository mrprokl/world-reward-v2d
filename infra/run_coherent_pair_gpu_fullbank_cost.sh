#!/usr/bin/env bash
# Complete manufactured P4/O3600/K64 marginal cost, no FIT or model.
# Closure: /infra/coherent_pair_gpu_fullbank_cost.py /infra/coherent_pair_gpu_probe.py
# Closure: /infra/coherent_pair_cost_probe.py /infra/mediapipe_cpu_runtime_verify.py
# Closure: /src/world_reward/coherent_pair_packed.py /src/world_reward/coherent_pair_packed_score.py
# Closure: /src/world_reward/coherent_pair_packed_torch.py /src/world_reward/coherent_pair_cache.py
# Closure: /src/world_reward/coherent_pair_learning.py /src/world_reward/coherent_route_scorer.py
# Closure: /src/world_reward/coherent_pair_marginal.py /src/world_reward/interaction_candidate_evidence.py
# Closure: /src/world_reward/interaction_tuple_evidence.py /src/world_reward/person_pose_observations.py
# Closure: /src/world_reward/hoi_detr_observations.py /src/world_reward/__init__.py
set +x
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname -s)" == scenesmith-ncc-h100-01 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_coherent_pair_gpu_fullbank_cost/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_coherent_pair_gpu_fullbank_cost.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock" PYTHONDONTWRITEBYTECODE=1
LOCK="$ROOT/jobs/.world-reward-h100.lock"
[[ ! -L "$LOCK" && ( ! -e "$LOCK" || -f "$LOCK" ) ]] || exit 2
exec 9>>"$LOCK";flock --nonblock 9
APPS="$(timeout --signal=TERM --kill-after=2s 8s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || exit 1
exec timeout --signal=TERM --kill-after=25s 1215s /usr/bin/python3 -I -B "$CODE/infra/coherent_pair_gpu_fullbank_cost.py" host "$CODE" "$REV"
