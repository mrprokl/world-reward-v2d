#!/usr/bin/env bash
# Serial, fail-fast chain. No stage accepts evaluation truth as model input.
set -euo pipefail
(( $# == 0 )) || { echo 'No hand pipeline arguments accepted' >&2; exit 2; }
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_hand_synthetic_render.sh"
bash "$CODE/infra/run_hand_synthetic_masks.sh"
bash "$CODE/infra/run_hand_synthetic_infer.sh"
bash "$CODE/infra/run_hand_synthetic_convert.sh"
bash "$CODE/infra/run_hand_synthetic_evaluate.sh"
