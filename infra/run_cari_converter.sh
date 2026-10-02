#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
source "$CODE/infra/cari_wrapper_common.sh"
wr_parse_cari_arguments converter "$@"
wr_cari_dependency converter
PYTHON_ARGUMENTS=(--episode "$WR_EPISODE")
# Preserve the exact legacy argv when the original forward route is selected.
if [[ "$WR_BUNDLE_SOURCE" == refined ]]; then PYTHON_ARGUMENTS+=(--bundle-source refined); fi
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env PYTHONPATH="$CODE/src" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/cari_converter.py" "${PYTHON_ARGUMENTS[@]}"
