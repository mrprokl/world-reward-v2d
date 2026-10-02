#!/usr/bin/env bash
# Fail-fast, serial GPU backends followed by isolated CPU evaluation.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?}"
bash "$CODE/infra/run_depth_rgb_infer.sh" --backend moge
bash "$CODE/infra/run_depth_rgb_infer.sh" --backend da3
bash "$CODE/infra/run_depth_rgb_evaluate.sh"
