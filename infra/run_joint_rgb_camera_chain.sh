#!/usr/bin/env bash
# One new K-coherent hypothesis on the same public RGB; no GT or rerender.
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
   inactive|failed) break ;; # Different episode/hypothesis; no dependency on its output.
   *) echo 'Unknown GPU predecessor state' >&2; exit 2 ;;
  esac
 done
fi
BASE="$ROOT/validation/joint_rgb_v1"
[[ -f "$BASE/inputs/manifest.json" && -f "$BASE/automatic_masks/report.json" ]]
for target in predictions_camera_v1 quality_camera_v1; do
 [[ ! -e "$BASE/$target" && ! -L "$BASE/$target" ]] || { echo "Frozen camera target exists: $target" >&2; exit 2; }
done
bash "$CODE/infra/run_joint_rgb_camera_infer.sh"
bash "$CODE/infra/run_joint_rgb_camera_evaluate.sh"
