#!/usr/bin/env bash
# Serial remote-only initializers for a new clip with automatic masks already
# produced. No generic report reuse, implicit retries or historical-unit waits.
# Each existing pinned child wrapper enforces its own provenance/numerical gates.
set -euo pipefail

EPISODE='' WAIT_FOR='' episode_seen=0 wait_seen=0
while (( $# )); do
  case "$1" in
    --episode)
      if (( episode_seen || $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9]|[12][0-9])$ ]]; then
        echo 'Require exactly one --episode integer in 0..29' >&2; exit 2
      fi
      EPISODE="$2"; episode_seen=1; shift 2 ;;
    --wait-for)
      if (( wait_seen || $# < 2 )) || [[ ! "$2" =~ ^world-reward-[a-z][a-z0-9-]{0,50}(\.service)?$ ]]; then
        echo 'Require at most one --wait-for world-reward-<job>[.service] unit' >&2; exit 2
      fi
      WAIT_FOR="${2%.service}.service"; wait_seen=1; shift 2 ;;
    *) echo "Unsupported initializer argument: $1" >&2; exit 2 ;;
  esac
done
if (( ! episode_seen )); then
  echo 'An explicit --episode integer in 0..29 is required' >&2; exit 2
fi

ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
source "$CODE/infra/cari_wrapper_common.sh"
printf -v EPISODE_PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$EPISODE_PADDED"
CURRENT_STAGE=preflight
wr_initializer_phase() {
  printf '{"stage":"%s","phase":"%s","timestamp_utc":"%s"}\n' \
    "$CURRENT_STAGE" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap 'status=$?; wr_initializer_phase fail; exit "$status"' ERR

wr_initializer_phase start
# Use only the report waiter, not wr_parse_cari_arguments: there must be no
# episode-15 historical dependency or implicit wait when --wait-for is absent.
wr_require_dependency_report "$BASE/automatic_masks/report.json" "$WAIT_FOR"
if [[ ! -s "$ROOT/results/camera-render.json" ]]; then
  echo 'Require the existing camera/depth gate report before initializers' >&2
  false
fi
# Preflight all target directories before consuming GPU time. Reports are not
# automatically trusted/reused here. Reject even complete or broken-link targets.
for target in body_smoke depth_smoke scale_smoke object_grounded body_full depth_full body_full/cari_adapter; do
  if [[ -e "$BASE/$target" || -L "$BASE/$target" ]]; then
    echo "Frozen initializer target already exists: $BASE/$target; do not overwrite" >&2
    false
  fi
done
wr_initializer_phase pass

CURRENT_STAGE=body_smoke
wr_initializer_phase start
bash "$CODE/infra/run_body_smoke.sh" --episode "$EPISODE" --inference-type body
wr_initializer_phase pass

CURRENT_STAGE=depth_smoke
wr_initializer_phase start
bash "$CODE/infra/run_depth_smoke.sh" --episode "$EPISODE"
wr_initializer_phase pass

CURRENT_STAGE=scale_smoke
wr_initializer_phase start
bash "$CODE/infra/run_scale_smoke.sh" --episode "$EPISODE"
wr_initializer_phase pass

CURRENT_STAGE=object_grounded
wr_initializer_phase start
bash "$CODE/infra/run_object_smoke.sh" --episode "$EPISODE" --aligned-pointmap
wr_initializer_phase pass

CURRENT_STAGE=body_full
wr_initializer_phase start
bash "$CODE/infra/run_body_smoke.sh" --episode "$EPISODE" --full-video --inference-type body
wr_initializer_phase pass

CURRENT_STAGE=depth_full
wr_initializer_phase start
bash "$CODE/infra/run_depth_smoke.sh" --episode "$EPISODE" --full-video
wr_initializer_phase pass

CURRENT_STAGE=cari_adapter
wr_initializer_phase start
bash "$CODE/infra/run_cari_body_adapter_smoke.sh" --episode "$EPISODE"
wr_initializer_phase pass
