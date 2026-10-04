#!/usr/bin/env bash
set -euo pipefail
# Explicit separate source-bound predicate compiler; historical build unchanged.
if [[ "${1:-}" == --serialization ]]; then
  (( $# == 1 )) || exit 2
  export DOCKER_HOST="unix://${WR_ROOT:?}/docker.sock"
  exec python3 -I -B "${WR_CODE:?}/infra/mesh_serialization_compile.py"
fi
ROOT="${WR_ROOT:-/srv/scenesmith/world-reward}"
CODE="${WR_CODE:?Require immutable committed source}"
REVISION="${WR_CODE_REVISION:?Require immutable source revision}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
(( $# == 0 )) || { echo 'Volume QEM build accepts no arguments' >&2; exit 2; }
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
REPORT="$ROOT/results/image-volume-qem.json"
[[ ! -e "$REPORT" && ! -L "$REPORT" ]] || { echo 'Frozen volume build report exists' >&2; exit 2; }
# Literal closure: $CODE/infra/mesh_volume_qem.cpp, $CODE/infra/build_volume_qem.sh,
# $CODE/infra/mesh_guarded_qem.cpp (identity only; never copied/modified in base).
BASE="$(python3 - "$ROOT/results/image-guarded-qem.json" "$CODE/infra/mesh_guarded_qem.cpp" <<'PYBASE'
import hashlib,json,pathlib,re,sys
p,cpp=map(pathlib.Path,sys.argv[1:]); assert p.is_file() and not p.is_symlink() and cpp.is_file() and not cpp.is_symlink()
r=json.loads(p.read_text()); digest=hashlib.sha256(cpp.read_bytes()).hexdigest()
assert r['stage']=='world_reward_guarded_qem_build' and r['status']=='pass' and r['block_intersections'] is True
assert r['libigl_revision']=='40e7900ccbd767f1f360e0eb10f0f1a6432e0993' and r['eigen_revision']=='3147391d946bb4b6c68edd901f2add6ac1f31f8c'
assert r['target_faces']==4096 and r['source_cpp_sha256']==digest
assert re.fullmatch(r'sha256:[0-9a-f]{64}',r['image_id']); print(r['image_id']+' '+digest)
PYBASE
)"
read -r BASE_ID BASE_CPP_SHA256 <<< "$BASE"
[[ "$(docker image inspect world-reward/guarded-qem:0.1 --format '{{.Id}}')" == "$BASE_ID" ]] || { echo 'Frozen guarded base mismatch' >&2; exit 2; }
BASE_REFERENCE="world-reward/volume-qem-base:${BASE_ID#sha256:}"
docker image tag "$BASE_ID" "$BASE_REFERENCE"
[[ "$(docker image inspect "$BASE_REFERENCE" --format '{{.Id}}')" == "$BASE_ID" ]] || exit 2
timeout --signal=TERM --kill-after=10s 660s docker build --pull=false --network none \
  --build-arg "BASE_IMAGE=$BASE_REFERENCE" --build-arg "BASE_CPP_SHA256=$BASE_CPP_SHA256" \
  --tag world-reward/volume-qem:0.1 --file "$CODE/infra/Dockerfile.volume_qem" "$CODE/infra"
[[ "$(docker image inspect "$BASE_REFERENCE" --format '{{.Id}}')" == "$BASE_ID" ]] || exit 2
IMAGE_ID="$(docker image inspect world-reward/volume-qem:0.1 --format '{{.Id}}')"
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 2
# Offline CPU ABI gate only: no geometry, predictions, assets, models or GPU.
BUILD_JSON="$(docker run --rm --network none --read-only --entrypoint python3 "$IMAGE_ID" -c "import hashlib,json,pathlib,subprocess; p=pathlib.Path('/opt/world-reward/volume-qem'); r=json.loads((p/'build.json').read_text()); assert r['status']=='pass'; assert hashlib.sha256((p/'mesh_volume_qem').read_bytes()).hexdigest()==r['binary_sha256']; i=json.loads(subprocess.check_output([str(p/'mesh_volume_qem'),'--build-info'],text=True)); assert all(r[k]==v for k,v in i.items()); print(json.dumps(r,separators=(',',':')))")"
python3 - "$REPORT" "$BASE_ID" "$BASE_REFERENCE" "$IMAGE_ID" "$REVISION" "$CODE" "$BUILD_JSON" "$ROOT/results/image-guarded-qem.json" <<'PYREPORT'
import hashlib,json,pathlib,sys
path,base,reference,image,revision,code,raw,base_report=sys.argv[1:]
root=pathlib.Path(code); build=json.loads(raw)
sources={name:hashlib.sha256((root/'infra'/name).read_bytes()).hexdigest() for name in ['mesh_volume_qem.cpp','build_volume_qem.sh','run_volume_qem_build.sh','Dockerfile.volume_qem','mesh_guarded_qem.cpp']}
assert build['source_cpp_sha256']==sources['mesh_volume_qem.cpp'] and build['base_source_cpp_sha256']==sources['mesh_guarded_qem.cpp']
assert build['build_script_sha256']==sources['build_volume_qem.sh']
r={**build,'stage':'world_reward_volume_qem_build','status':'pass','base_image_id':base,'base_image_reference':reference,
    'image_id':image,'producer_revision':revision,'source_sha256':sources,
    'base_build_report_sha256':hashlib.sha256(pathlib.Path(base_report).read_bytes()).hexdigest(),
    'device':'cpu','binary_gate_network':'none','accuracy_verified':False,'production_adoption_performed':False}
with open(path,'x') as stream: json.dump(r,stream,indent=2);stream.write('\n')
PYREPORT
