#!/usr/bin/env bash
# Native public observations -> public fit -> independent private quality.
# Separate from synthesis to keep the immutable control bundle under100KB.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?}"
bash "$CODE/infra/run_joint_affine_infer.sh"
bash "$CODE/infra/run_joint_affine_fit.sh"
bash "$CODE/infra/run_joint_affine_evaluate.sh"
