#!/usr/bin/env bash
# Existing automatic/full initializers and frozen qualified mesh, no GT/restarts.
set -euo pipefail
(( $# == 0 )) || exit 2
ROOT="${WR_ROOT:?}"
CODE="${WR_CODE:?Require immutable source}"
BASE="$ROOT/outputs/episode_000000"
for target in object_pose_full cari_inputs cari_forward cari_conversion final_schema; do
 [[ ! -e "$BASE/$target" && ! -L "$BASE/$target" ]] || { echo "Frozen target exists: $target" >&2; exit 2; }
done
bash "$CODE/infra/run_object_pose_smoke.sh" --episode 0 --full-video --mesh-source volume
bash "$CODE/infra/run_cari_prepare.sh" --episode 0
bash "$CODE/infra/run_cari_forward.sh" --episode 0
bash "$CODE/infra/run_cari_converter.sh" --episode 0
bash "$CODE/infra/run_final_episode_gate.sh" --episode 0
