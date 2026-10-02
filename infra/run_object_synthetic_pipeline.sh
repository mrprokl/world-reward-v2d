#!/usr/bin/env bash
# Serial independent RGB cohort, automatic observations, generation, evaluation.
set -euo pipefail
(( $# == 0 )) || { echo 'No object pipeline arguments accepted' >&2; exit 2; }
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_object_synthetic_render.sh"
bash "$CODE/infra/run_object_synthetic_observations.sh"
bash "$CODE/infra/run_object_synthetic_generate.sh"
bash "$CODE/infra/run_object_synthetic_evaluate.sh"
