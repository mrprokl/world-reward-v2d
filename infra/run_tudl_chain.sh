#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable source}"
[[ $# == 2 && "$1" == --wait-for && "$2" =~ ^world-reward-[a-z0-9-]+$ ]] || exit 2
DEPENDENCY="$2"; START="$SECONDS"
while true; do
 STATE="$(systemctl show "$DEPENDENCY" --property=ActiveState --value)"
 case "$STATE" in
  active|activating|deactivating) (( SECONDS-START < 43200 )) || { echo 'Bounded GPU queue wait expired' >&2; exit 2; }; sleep 30 ;;
  inactive|failed) break ;;
  *) echo 'Unknown exact GPU predecessor state' >&2; exit 2 ;;
 esac
done
# Independent external cohort: predecessor predictions are not inference inputs.
[[ -f "$ROOT/validation/tudl_rgb_v1/inputs/manifest.json" ]]
bash "$CODE/infra/run_tudl_infer.sh"
bash "$CODE/infra/run_tudl_evaluate.sh"
