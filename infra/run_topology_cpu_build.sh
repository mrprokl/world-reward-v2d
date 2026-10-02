#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Topology CPU build accepts no arguments' >&2; exit 2; }
REPORT="$ROOT/results/image-topology-cpu.json"
[[ ! -e "$REPORT" && ! -L "$REPORT" ]] || { echo 'Frozen topology build report exists' >&2; exit 2; }
BASE_ID="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$BASE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Invalid immutable base image ID' >&2; exit 2; }
# Dockerfile FROM cannot use a bare daemon image ID with BuildKit. A local
# content-addressed tag binds the inspected ID without pulling a registry base.
BASE_REFERENCE="world-reward/topology-base:${BASE_ID#sha256:}"
docker image tag "$BASE_ID" "$BASE_REFERENCE"
[[ "$(docker image inspect "$BASE_REFERENCE" --format '{{.Id}}')" == "$BASE_ID" ]] || exit 2
docker build --pull=false --network host --build-arg "BASE_IMAGE=$BASE_REFERENCE" \
  --tag world-reward/topology-cpu:0.1 --file "$CODE/infra/Dockerfile.topology_cpu" "$CODE/infra"
[[ "$(docker image inspect "$BASE_REFERENCE" --format '{{.Id}}')" == "$BASE_ID" ]] || { echo 'Local base binding changed during build' >&2; exit 2; }
IMAGE_ID="$(docker image inspect world-reward/topology-cpu:0.1 --format '{{.Id}}')"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
# Independent import/filter gate is CPU-only and offline, using the immutable ID.
docker run --rm --network none "$IMAGE_ID" python -c "import pymeshlab, importlib.metadata as m; assert m.version('pymeshlab') == '2025.7.post1'; print('topology_cpu_import=pass')"
python3 - "$REPORT" "$BASE_ID" "$BASE_REFERENCE" "$IMAGE_ID" "$REVISION" "$CODE/infra/Dockerfile.topology_cpu" <<'PYREPORT'
import hashlib,json,pathlib,sys
path,base,reference,image,revision,source=sys.argv[1:]
report={'stage':'world_reward_topology_cpu_build','status':'pass','base_image_id':base,'base_image_reference':reference,'image_id':image,'producer_revision':revision,'dockerfile_sha256':hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest(),'pymeshlab_version':'2025.7.post1','wheel_sha256':'c3c1b01f101334b14469ace3b004382cd313b80a128f551a1da77e3053f09c30','license':'GPL-3.0; commercial use permitted, redistribution copyleft obligations','device':'cpu','build_network':'host exact hash-pinned wheel only','import_gate_network':'none','challenge_inputs_used':False}
with open(path,'x') as handle: handle.write(json.dumps(report,indent=2)+'\n')
PYREPORT
