#!/usr/bin/env bash
# Explicit continuation after preflight-only queue failure; no asset mutation.
set -euo pipefail
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"
source "$CODE/infra/cari_wrapper_common.sh"
wr_parse_cari_arguments refine "$@"
[[ -z "$WR_WAIT_FOR" ]] || { echo 'Continuation requires no active GPU predecessor' >&2; exit 2; }
printf -v EPISODE '%06d' "$WR_EPISODE"
BASE="$ROOT/outputs/episode_$EPISODE"
for target in cari_refined cari_conversion_refined final_schema_refined; do
 [[ ! -e "$BASE/$target" && ! -L "$BASE/$target" ]] || { echo 'Never resume or overwrite prediction outputs' >&2; exit 2; }
done
[[ -f "$ROOT/results/cari-refinement-assets.json" ]]
# The existing verified CPU preflight/asset identities remain immutable. Python
# consumer rechecks every source/asset/input before constructing the optimizer.
bash "$CODE/infra/run_cari_refine.sh" --episode "$WR_EPISODE" --no-wait
bash "$CODE/infra/run_cari_converter.sh" --episode "$WR_EPISODE" --bundle-source refined --no-wait
bash "$CODE/infra/run_final_episode_gate.sh" --episode "$WR_EPISODE" --bundle-source refined
