#!/usr/bin/env bash
# New immutable RGB cohort then public observations, serial GPU use only.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_object_motion_render.sh"
bash "$CODE/infra/run_object_motion_observations.sh"
