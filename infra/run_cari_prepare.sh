#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
source "$CODE/infra/cari_wrapper_common.sh"
wr_parse_cari_arguments prepare "$@"
wr_cari_dependency prepare
set -- --episode "$WR_EPISODE"
if [[ "$WR_MESH_SOURCE" == solid ]]; then set -- "$@" --mesh-source solid; fi
docker run --rm --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env PYTHONPATH="$CODE/src" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/data,dst=$ROOT/data,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/cari_prepare.py" "$@"
