#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
image="${1:?Image name sam3d or cari4d}"
if [[ "$image" != sam3d && "$image" != cari4d ]]; then
  echo 'Unknown audited runtime' >&2; exit 1
fi
arguments=()
if [[ "$image" == sam3d ]]; then arguments+=(--flash-attention); fi
docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env HOME="$ROOT/cache" --env NVIDIA_DRIVER_CAPABILITIES=all \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  "world-reward/$image:7c0d3b9" python "$CODE/infra/kernel_smoke.py" \
  --report "$ROOT/results/kernels-$image.json" "${arguments[@]}"
