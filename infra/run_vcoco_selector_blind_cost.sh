#!/usr/bin/env bash
# Closure: /infra/vcoco_selector_blind_cost.py /infra/vcoco_selector_blind_cost_native.py
# One predeclared blind full32 label-free cost profile on Azure VM02.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$(uname -s)" == Linux && "$(id -u)" == 0 && "$(hostname)" == world-reward-ncc-h100-02 \
 && "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_vcoco_selector_blind_cost/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_vcoco_selector_blind_cost.sh" ]] || exit 2
umask 077
LOCK="$ROOT/jobs/.world-reward-h100.lock"
[[ ! -L "$LOCK" && ( ! -e "$LOCK" || -f "$LOCK" ) ]] || exit 2
exec 9>>"$LOCK"; flock --nonblock 9
APPS="$(timeout --signal=TERM --kill-after=2s 8s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || exit 1
exec timeout --signal=TERM --kill-after=10s 3615s env -i PATH=/usr/bin:/bin HOME=/nonexistent \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" DOCKER_HOST="unix://$ROOT/docker.sock" \
 PYTHONDONTWRITEBYTECODE=1 python3 -I -B "$CODE/infra/vcoco_selector_blind_cost.py"
