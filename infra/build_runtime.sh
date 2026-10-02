#!/usr/bin/env bash
# Build the audited official runtime on the isolated World Reward Docker store.
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
MODULES="$ROOT/vendor/video_to_data/reconstruction/modules"
test "$(git -c safe.directory="$ROOT/vendor/video_to_data" \
  -C "$ROOT/vendor/video_to_data" rev-parse HEAD)" = \
  7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
for MODULE in sam2 cari4d sam3d; do
  docker build --network host --tag "world-reward/$MODULE:7c0d3b9" \
    --file "$MODULES/v2d_$MODULE/docker/Dockerfile" "$MODULES"
  docker image inspect "world-reward/$MODULE:7c0d3b9" \
    --format '{{json .}}' > "$ROOT/results/image-$MODULE.json"
done
docker run --rm --gpus all --network host world-reward/cari4d:7c0d3b9 \
  python -c 'import torch; assert torch.cuda.is_available(); a=torch.randn(32,32,device="cuda"); print(torch.cuda.get_device_name(),float((a@a.T).sum())); import pytorch3d, kaolin, nvdiffrast; print("cuda_runtime_imports=pass")'
printf '%s\n' 'runtime-build-and-cuda-smoke=pass'
