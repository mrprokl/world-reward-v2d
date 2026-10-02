#!/usr/bin/env bash
# Fresh J3 synthesis then automatic public masks, no old cohort tuning.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?}"
bash "$CODE/infra/run_joint_affine_render.sh"
bash "$CODE/infra/run_joint_affine_masks.sh"
