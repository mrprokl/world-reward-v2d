#!/usr/bin/env bash
# Preserve completed RGB and failed original observations, new compatible cache.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_object_synthetic_observations.sh"
bash "$CODE/infra/run_object_synthetic_generate.sh"
bash "$CODE/infra/run_object_synthetic_evaluate.sh"
