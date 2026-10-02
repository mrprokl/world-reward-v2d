#!/usr/bin/env bash
# Independent research, queued behind an exact GPU unit (not its predictions).
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?Require immutable source}"
(( $# == 0 || $# == 2 )) || exit 2
if (( $# )); then
 [[ "$1" == --wait-for && "$2" =~ ^world-reward-[a-z][a-z0-9-]{0,50}(\.service)?$ ]] || exit 2
 UNIT="${2%.service}.service"; WAITED=0
 while true; do
  LOAD="$(systemctl show "$UNIT" -p LoadState --value)"
  [[ "$LOAD" == loaded ]] || { echo 'GPU predecessor unit not loaded' >&2; exit 2; }
  STATE="$(systemctl show "$UNIT" -p ActiveState --value)"
  case "$STATE" in
   active|activating|deactivating|reloading)
    (( WAITED < 43200 )) || { echo 'GPU serialization exceeded12h' >&2; exit 2; }
    sleep 30; WAITED=$((WAITED+30)) ;;
   inactive|failed) break ;; # Independent hypothesis: predecessor failure is not an input.
   *) echo 'Unknown GPU predecessor state' >&2; exit 2 ;;
  esac
 done
fi
bash "$CODE/infra/run_joint_rgb_render.sh"
bash "$CODE/infra/run_joint_rgb_masks.sh"
bash "$CODE/infra/run_joint_rgb_infer.sh"
bash "$CODE/infra/run_joint_rgb_evaluate.sh"
