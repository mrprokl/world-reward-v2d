#!/usr/bin/env bash
# Preserve the original decoded-artifact/report failure; serialization-only retry.
set -euo pipefail
(( $# == 0 )) || exit 2
CODE="${WR_CODE:?Require immutable source}"
bash "$CODE/infra/run_object_synthetic_generate.sh"
bash "$CODE/infra/run_object_synthetic_evaluate.sh"
