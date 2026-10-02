#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo "No partial gauge arguments supported" >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=$(docker image inspect world-reward/sam3d-runtime:0.1 --format '{{.Id}}')
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
PIN=abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
# Same frozen producer image, but deliberately no GPU, model weights or RGB.
docker run --rm --network none --cpus 2 --memory 4g \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" --env WR_SOURCE_IMAGE_ID="$IMAGE" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor/mv-sam3d/$PIN,dst=$ROOT/vendor/mv-sam3d/$PIN,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  "$IMAGE" python "$CODE/infra/multiview_gauge_gate.py"
