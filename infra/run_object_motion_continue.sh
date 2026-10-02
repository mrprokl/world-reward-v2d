#!/usr/bin/env bash
# Public RGB inputs/observations already frozen; never overwrite preceding stages.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_object_motion_generate.sh"
bash "$CODE/infra/run_object_motion_track.sh"
bash "$CODE/infra/run_object_motion_evaluate.sh"
