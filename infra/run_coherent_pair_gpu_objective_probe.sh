#!/usr/bin/env bash
# New full-bank objective/short solver; no real FIT or model.
# Closure: /infra/coherent_pair_gpu_objective_probe.py /infra/coherent_pair_gpu_fullbank_cost.py
# Closure: /infra/run_coherent_pair_gpu_fullbank_cost.sh /infra/coherent_pair_gpu_probe.py
# Closure: /infra/coherent_pair_cost_probe.py /infra/mediapipe_cpu_runtime_verify.py
# Closure: /src/world_reward/coherent_pair_marginal_objective.py /src/world_reward/coherent_pair_packed.py
# Closure: /src/world_reward/coherent_pair_packed_score.py /src/world_reward/coherent_pair_packed_torch.py
# Closure: /src/world_reward/coherent_pair_cache.py /src/world_reward/coherent_pair_learning.py
# Closure: /src/world_reward/coherent_route_scorer.py /src/world_reward/coherent_pair_marginal.py
# Closure: /src/world_reward/interaction_candidate_evidence.py /src/world_reward/interaction_tuple_evidence.py
# Closure: /src/world_reward/person_pose_observations.py /src/world_reward/hoi_detr_observations.py
# Closure: /src/world_reward/__init__.py
set +x
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname -s)" == scenesmith-ncc-h100-01 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_coherent_pair_gpu_objective_probe/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_coherent_pair_gpu_objective_probe.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock" PYTHONDONTWRITEBYTECODE=1
LOCK="$ROOT/jobs/.world-reward-h100.lock"
[[ ! -L "$LOCK" && ( ! -e "$LOCK" || -f "$LOCK" ) ]] || exit 2
exec 9>>"$LOCK";flock --nonblock 9
APPS="$(timeout --signal=TERM --kill-after=2s 8s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || exit 1
exec timeout --signal=TERM --kill-after=25s 1215s /usr/bin/python3 -I -B "$CODE/infra/coherent_pair_gpu_objective_probe.py" host "$CODE" "$REV"
