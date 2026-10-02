#!/usr/bin/env bash
# Reuse three frozen anchors; preserve failed tracking-v1 report without reruns.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_object_motion_track.sh"
bash "$CODE/infra/run_object_motion_evaluate.sh"
