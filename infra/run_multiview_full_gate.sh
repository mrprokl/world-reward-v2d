#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || { echo "No full native gate arguments supported" >&2; exit 2; }
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid source revision" >&2; exit 2; }
PIN=abb04b5e8af5bc33b0265bdf19937e76bbb6bcdd
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=$(docker image inspect world-reward/sam3d-runtime:0.1 --format '{{.Id}}')
[[ "$IMAGE" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "Invalid runtime image digest" >&2; exit 2; }
timeout --signal=TERM --kill-after=10s 303s docker run --rm --gpus all --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" --env WR_IMAGE_ID="$IMAGE" \
  --env HOME="$ROOT/cache" --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH="$ROOT/vendor/mv-sam3d/$PIN:$CODE/src" \
  --env HF_HOME="$ROOT/weights/sam3d/hf_home" --env TORCH_HOME="$ROOT/weights/sam3d/torch_home" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/vendor/mv-sam3d/$PIN,dst=$ROOT/vendor/mv-sam3d/$PIN,readonly" \
  --mount "type=bind,src=$ROOT/weights,dst=$ROOT/weights,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  --mount "type=bind,src=$ROOT/cache,dst=$ROOT/cache" \
  "$IMAGE" python "$CODE/infra/multiview_full_gate.py"
