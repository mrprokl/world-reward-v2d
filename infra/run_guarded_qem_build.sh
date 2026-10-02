#!/usr/bin/env bash
set -euo pipefail
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Guarded QEM build accepts no arguments' >&2; exit 2; }
REPORT="$ROOT/results/image-guarded-qem.json"
[[ ! -e "$REPORT" && ! -L "$REPORT" ]] || { echo 'Frozen guarded QEM build report exists' >&2; exit 2; }
# Literal source closure: $CODE/infra/mesh_guarded_qem.cpp and $CODE/infra/build_guarded_qem.sh.
BASE_ID="$(python3 - "$ROOT/results/image-topology-cpu.json" <<'PYBASE'
import json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]); assert p.is_file() and not p.is_symlink()
r=json.loads(p.read_text()); assert r['stage']=='world_reward_topology_cpu_build' and r['status']=='pass'
assert r['pymeshlab_version']=='2025.7.post1' and r['wheel_sha256']=='c3c1b01f101334b14469ace3b004382cd313b80a128f551a1da77e3053f09c30'
assert re.fullmatch(r'sha256:[0-9a-f]{64}',r['image_id']); print(r['image_id'])
PYBASE
)"
[[ "$(docker image inspect world-reward/topology-cpu:0.1 --format '{{.Id}}')" == "$BASE_ID" ]] || { echo 'Frozen topology base mismatch' >&2; exit 2; }
BASE_REFERENCE="world-reward/guarded-qem-base:${BASE_ID#sha256:}"
docker image tag "$BASE_ID" "$BASE_REFERENCE"
[[ "$(docker image inspect "$BASE_REFERENCE" --format '{{.Id}}')" == "$BASE_ID" ]] || exit 2
timeout --signal=TERM --kill-after=10s 660s docker build --pull=false --network host \
  --build-arg "BASE_IMAGE=$BASE_REFERENCE" --tag world-reward/guarded-qem:0.1 \
  --file "$CODE/infra/Dockerfile.guarded_qem" "$CODE/infra"
[[ "$(docker image inspect "$BASE_REFERENCE" --format '{{.Id}}')" == "$BASE_ID" ]] || { echo 'Local base binding changed' >&2; exit 2; }
IMAGE_ID="$(docker image inspect world-reward/guarded-qem:0.1 --format '{{.Id}}')"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
# Offline CPU-only binary ABI and SHA gate. No challenge/image/data volume mounted.
BUILD_JSON="$(docker run --rm --network none --read-only --entrypoint python3 "$IMAGE_ID" -c "import hashlib,json,pathlib,subprocess; p=pathlib.Path('/opt/world-reward/guarded-qem'); r=json.loads((p/'build.json').read_text()); assert r['status']=='pass'; assert hashlib.sha256((p/'mesh_guarded_qem').read_bytes()).hexdigest()==r['binary_sha256']; i=json.loads(subprocess.check_output([str(p/'mesh_guarded_qem'),'--build-info'],text=True)); assert all(r[k]==v for k,v in i.items()); print(json.dumps(r,separators=(',',':')))")"
python3 - "$REPORT" "$BASE_ID" "$BASE_REFERENCE" "$IMAGE_ID" "$REVISION" "$CODE" "$BUILD_JSON" "$ROOT/results/image-topology-cpu.json" <<'PYREPORT'
import hashlib,json,pathlib,sys
path,base,reference,image,revision,code,raw,base_report=sys.argv[1:]
root=pathlib.Path(code); build=json.loads(raw)
sources={name:hashlib.sha256((root/'infra'/name).read_bytes()).hexdigest() for name in ['mesh_guarded_qem.cpp','build_guarded_qem.sh','run_guarded_qem_build.sh','Dockerfile.guarded_qem']}
assert build['source_cpp_sha256']==sources['mesh_guarded_qem.cpp'] and build['build_script_sha256']==sources['build_guarded_qem.sh']
r={**build,'stage':'world_reward_guarded_qem_build','status':'pass','base_image_id':base,'base_image_reference':reference,'image_id':image,'producer_revision':revision,'source_sha256':sources,'base_build_report_sha256':hashlib.sha256(pathlib.Path(base_report).read_bytes()).hexdigest(),'device':'cpu','build_network':'host fixed immutable source commits only','binary_gate_network':'none','accuracy_verified':False,'production_adoption_performed':False}
with open(path,'x') as handle: json.dump(r,handle,indent=2);handle.write('\n')
PYREPORT
