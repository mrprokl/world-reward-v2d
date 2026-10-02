#!/usr/bin/env bash
# One fail-fast cloud lane; private truth is mounted only by the final CPU stage.
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[0-9a-f]{64}$ ]] || exit 2
CODE="${WR_CODE:?}"
bash "$CODE/infra/run_perspective_rgb_import.sh" "$1"
bash "$CODE/infra/run_perspective_rgb_infer.sh"
bash "$CODE/infra/run_perspective_rgb_evaluate.sh"
