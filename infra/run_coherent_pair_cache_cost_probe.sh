#!/usr/bin/env bash
# NEW data-free paired cache control; no retry of closed FIT/reference studies.
# Closure: /infra/coherent_pair_cache_cost_probe.py /infra/coherent_pair_cost_probe.py
# Closure: /src/world_reward/coherent_pair_cache.py /configs/coherent_pair_cache_cost_probe_v1.json
set +x
set -euo pipefail
[[ $# == 0 && "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname -s)" == scenesmith-ncc-h100-01 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_coherent_pair_cache_cost_probe/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_coherent_pair_cache_cost_probe.sh" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock" PYTHONDONTWRITEBYTECODE=1
exec timeout --signal=TERM --kill-after=20s 740s python3 -I -B "$CODE/infra/coherent_pair_cache_cost_probe.py" host "$CODE" "$REV"
