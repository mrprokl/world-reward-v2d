#!/usr/bin/env bash
# Complete the missing native final refinement; never replace forward outputs.
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?Require immutable source}"
source "$CODE/infra/cari_wrapper_common.sh"
wr_parse_cari_arguments refine "$@"
printf -v EPISODE '%06d' "$WR_EPISODE"
BASE="$ROOT/outputs/episode_$EPISODE"
for target in cari_refined cari_conversion_refined final_schema_refined; do
 [[ ! -e "$BASE/$target" && ! -L "$BASE/$target" ]] || { echo "Frozen target exists: $target" >&2; exit 2; }
done
# Acquisition is CPU/network only. No large data, no vendor modifications.
bash "$CODE/infra/run_cari_refinement_assets.sh"
bash "$CODE/infra/run_cari_refinement_preflight.sh"
if [[ -n "$WR_WAIT_FOR" ]]; then
 UNIT="$WR_WAIT_FOR"; WAITED=0
 while true; do
  LOAD="$(systemctl show "$UNIT" -p LoadState --value)"
  [[ "$LOAD" == loaded ]] || { echo 'GPU predecessor unit not loaded' >&2; exit 2; }
  STATE="$(systemctl show "$UNIT" -p ActiveState --value)"
  case "$STATE" in
   active|activating|deactivating|reloading)
    (( WAITED < 43200 )) || { echo 'GPU serialization exceeded12h' >&2; exit 2; }
    sleep 30; WAITED=$((WAITED+30)) ;;
   inactive|failed) break ;; # Independent input episode; failure is not a dependency.
   *) echo 'Unknown GPU predecessor state' >&2; exit 2 ;;
  esac
 done
fi
bash "$CODE/infra/run_cari_refine.sh" --episode "$WR_EPISODE" --no-wait
bash "$CODE/infra/run_cari_converter.sh" --episode "$WR_EPISODE" --bundle-source refined --no-wait
bash "$CODE/infra/run_final_episode_gate.sh" --episode "$WR_EPISODE" --bundle-source refined
