#!/usr/bin/env bash
# Resume after unchanged frozen RGB/masks. Original inference failure retained.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_hand_synthetic_infer.sh"
bash "$CODE/infra/run_hand_synthetic_convert.sh"
bash "$CODE/infra/run_hand_synthetic_evaluate.sh"
