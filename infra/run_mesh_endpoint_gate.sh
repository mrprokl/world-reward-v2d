#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Independent endpoint QEM gate accepts no configurable controls' >&2; exit 2; }
IMAGE_ID="$(python3 - "$ROOT/results/image-topology-cpu.json" <<'PYIMAGE'
import json,re,sys
r=json.load(open(sys.argv[1])); assert r['stage']=='world_reward_topology_cpu_build' and r['status']=='pass'
assert r['pymeshlab_version']=='2025.7.post1' and r['wheel_sha256']=='c3c1b01f101334b14469ace3b004382cd313b80a128f551a1da77e3053f09c30'
assert re.fullmatch(r'sha256:[0-9a-f]{64}',r['base_image_id']) and re.fullmatch(r'sha256:[0-9a-f]{64}',r['image_id'])
print(r['image_id'])
PYIMAGE
)"
docker run --rm --cpus 4 --memory 4g --network none \
  --user "$(id -u scenesmith):$(id -g scenesmith)" \
  --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REVISION" --env WR_TOPOLOGY_IMAGE_ID="$IMAGE_ID" \
  --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
  --mount "type=bind,src=$ROOT/results,dst=$ROOT/results" \
  "$IMAGE_ID" python "$CODE/infra/mesh_endpoint_gate.py"
