#!/usr/bin/env bash
# One clean-episode native route, not resume, packaging or upload. The immutable
# Azure launcher binds the source revision. Child wrappers retain actual input,
# source/model/provenance and numerical checks; no eligibility/accuracy claim.
set -euo pipefail

EPISODE='' episode_seen=0
while (( $# )); do
  case "$1" in
    --episode)
      if (( episode_seen || $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9]|[12][0-9])$ ]]; then
        echo 'Require exactly one --episode integer in 0..29' >&2; exit 2
      fi
      EPISODE="$2"; episode_seen=1; shift 2 ;;
    *) echo "Unsupported clean-episode argument: $1" >&2; exit 2 ;;
  esac
done
if (( ! episode_seen )); then
  echo 'An explicit --episode integer in 0..29 is required' >&2; exit 2
fi

ROOT="${WR_ROOT:?Require absolute remote runtime root}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
for path in "$ROOT" "$CODE"; do
  if [[ "$path" != /* || "$path" == / || "$path" =~ (^|/)\.\.?(/|$) || "$path" == *$'\n'* || "$path" == *$'\r'* ]]; then
    echo 'Runtime root/source must be absolute paths without traversal or newlines' >&2; exit 2
  fi
done
if [[ ! "$REVISION" =~ ^[0-9a-f]{40}$ || ! -d "$CODE/infra" || -L "$CODE" ]]; then
  echo 'Require launcher-bound immutable source directory and 40-hex revision' >&2; exit 2
fi
printf -v EPISODE_PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$EPISODE_PADDED"
CURRENT_STAGE=preflight
wr_episode_phase() {
  printf '{"stage":"%s","phase":"%s","timestamp_utc":"%s"}\n' \
    "$CURRENT_STAGE" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap 'status=$?; wr_episode_phase fail; exit "$status"' ERR

wr_episode_phase start
if [[ -L "$ROOT/outputs" || -L "$BASE" ]]; then
  echo 'Episode output roots cannot be symlinks' >&2
  false
fi
# Check every producing directory before the first GPU call. Existing reports
# are not trusted/reused; even broken symlinks are frozen occupied targets.
for target in automatic_masks body_smoke depth_smoke scale_smoke object_grounded \
              body_full depth_full object_pose_full body_full/cari_adapter \
              cari_inputs cari_forward cari_conversion; do
  if [[ -e "$BASE/$target" || -L "$BASE/$target" ]]; then
    echo "Frozen episode target already exists: $BASE/$target; no overwrite or implicit resume" >&2
    false
  fi
done
if [[ ! -s "$ROOT/results/camera-render.json" ]]; then
  echo 'Require the existing camera/depth gate report before the clean route' >&2
  false
fi
for child in run_automatic_masks.sh run_episode_initializers.sh run_object_pose_smoke.sh \
             run_cari_prepare.sh run_cari_forward.sh run_cari_converter.sh; do
  if [[ ! -f "$CODE/infra/$child" || -L "$CODE/infra/$child" ]]; then
    echo "Required immutable child wrapper is absent or a symlink: $child" >&2
    false
  fi
done
wr_episode_phase pass

CURRENT_STAGE=automatic_masks
wr_episode_phase start
bash "$CODE/infra/run_automatic_masks.sh" --episode "$EPISODE"
wr_episode_phase pass

CURRENT_STAGE=episode_initializers
wr_episode_phase start
bash "$CODE/infra/run_episode_initializers.sh" --episode "$EPISODE"
wr_episode_phase pass

CURRENT_STAGE=object_pose_full
wr_episode_phase start
bash "$CODE/infra/run_object_pose_smoke.sh" --episode "$EPISODE" --full-video
wr_episode_phase pass

# Each producer has returned in this same process. Explicit report-only mode
# avoids historical episode-15 unit waits; actual child validators still check
# the exact selected producer artifacts. No guessed or self-unit dependency.
CURRENT_STAGE=cari_inputs
wr_episode_phase start
bash "$CODE/infra/run_cari_prepare.sh" --episode "$EPISODE" --no-wait
wr_episode_phase pass

CURRENT_STAGE=cari_forward
wr_episode_phase start
bash "$CODE/infra/run_cari_forward.sh" --episode "$EPISODE" --no-wait
wr_episode_phase pass

CURRENT_STAGE=cari_conversion
wr_episode_phase start
bash "$CODE/infra/run_cari_converter.sh" --episode "$EPISODE" --no-wait
wr_episode_phase pass
