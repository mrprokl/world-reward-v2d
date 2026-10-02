#!/usr/bin/env bash
# Existing bounded public acquisition drivers; heavy bytes remain on Azure.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?}"
bash "$CODE/infra/run_da3_metric_acquire.sh"
bash "$CODE/infra/run_da3_dependency_acquire.sh"
