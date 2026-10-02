#!/usr/bin/env bash
# One isolated D76 chain, never re-run frozen MoGe or mount sensor truth upstream.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?}"
bash "$CODE/infra/run_da3_runtime_gate.sh"
bash "$CODE/infra/run_tudl_prediction_import.sh" --archive-sha256 \
 81e3226c43c6311f98c7a43f704b80a9dc69ce4bb0d233d971ffc6c88f281881
bash "$CODE/infra/run_da3_metric_infer.sh"
bash "$CODE/infra/run_da3_metric_evaluate.sh"
