#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
source "$CODE/infra/cari_wrapper_common.sh"
# Adapter parsing has no legacy episode-15 wait; an explicit wait addresses
# the selected final-conversion producer, never an unrelated old episode.
wr_parse_cari_arguments adapter "$@"
printf -v EPISODE_PADDED '%06d' "$WR_EPISODE"
BASE="$ROOT/outputs/episode_$EPISODE_PADDED"
wr_require_dependency_report "$BASE/cari_conversion/report.json" "$WR_WAIT_FOR"
wr_require_dependency_report "$BASE/body_hands_smoke/report.json"
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="${WR_CODE_REVISION:?}" --env PYTHONPATH="$CODE/src" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor,dst=$ROOT/vendor,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results,readonly" \
  --mount "type=bind,src=$ROOT/outputs,dst=$ROOT/outputs" \
  world-reward/cari4d-source:0.1 python "$CODE/infra/finger_transfer_smoke.py" --episode "$WR_EPISODE"
