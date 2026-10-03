#!/usr/bin/env bash
# Twelve NEW analytic object-only RGBs; private truth is never mounted.
# Source closure: /infra/analytic_rgbd_infer.py /infra/analytic_rgbd_inputs.py.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_analytic_rgbd_infer/code" && "$(uname -s)" == Linux ]] || exit 2
BASE="$ROOT/validation/analytic_rgbd_holdout_v1"
OUT="$BASE/anchor_predictions_v1"
JOB="${CODE%/code}"
CONTAINER="world-reward-analytic-anchor-$REV"
OWNED_CONTAINER=0
BEFORE=""
export DOCKER_HOST="unix://$ROOT/docker.sock"
PINNED_IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
MOGE="$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal"
BLOB="$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c"
MOGE_REPORT="$ROOT/results/weights-acquisition.json"
DA3_SOURCE="$ROOT/vendor/research/da3_metric_v1"
DA3_WEIGHTS="$ROOT/weights/research/da3_metric_v1"
DA3_REPORT="$ROOT/results/da3-metric-acquisition-v1.json"
WHEEL="$ROOT/vendor/research/da3_dependencies_v1/addict-2.4.0-py3-none-any.whl"
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,re,stat,sys
root,code,revision,executing=sys.argv[1:];root,code,executing=map(Path,(root,code,executing))
def canonical(path):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):
  raise ValueError('Canonical actual source/asset top-level paths required')
for path in(root,code,executing):canonical(path)
if (not root.is_dir()or not code.is_dir()or code!=root/'jobs'/revision/'run_analytic_rgbd_infer'/'code'
    or executing!=code/'infra/run_analytic_rgbd_infer.sh'):
 raise ValueError('Exact actual immutable analytic launcher namespace required')
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 path=code.parent/name;canonical(path)
 if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Regular original dispatch markers required')
 value=path.read_bytes()
 if(name=='revision'and value!=(revision+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',value)):
  raise ValueError('Exact original revision/archive markers required')
 digest.update(name.encode()+b'\0'+value)
for name in('infra/run_analytic_rgbd_infer.sh','infra/analytic_rgbd_infer.py','infra/analytic_rgbd_inputs.py',
             'configs/analytic_rgbd_input_pins.json','src/world_reward/__init__.py'):
 if not(code/name).is_file():raise ValueError('Complete committed producer/public-reader/pin source required')
# All configs remain in the original immutable dispatch archive, but never
# enter the GPU container. Only this complete numerical/import closure does.
infra={'run_analytic_rgbd_infer.sh','analytic_rgbd_infer.py','analytic_rgbd_inputs.py',
 'rgbd_anchor_contract.py','tudl_anchor_infer.py','tudl_holdout_inputs.py','tudl_infer.py',
 'object_synthetic_observations.py','da3_metric_infer.py','da3_metric_acquire.py'}
src={'world_reward/__init__.py','world_reward/data.py','world_reward/pointmap.py'}
for directory,names in(('infra',infra),('src',src)):
 if {str(p.relative_to(code/directory))for p in(code/directory).rglob('*')if p.is_file()}!=names:
  raise ValueError('Complete blind-only source closure required; manufacture source must not enter inference')
for path in(code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink()or(not stat.S_ISDIR(mode)and not stat.S_ISREG(mode))or mode&0o222:
  raise ValueError('Complete actual source must be readonly and regular')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
sys.path.insert(0,str(code/'infra'))
import analytic_rgbd_inputs as public
pins=public.strict_json((code/'configs/analytic_rgbd_input_pins.json').read_bytes())
directory=root/'validation/analytic_rgbd_holdout_v1/inputs'
records,receipt=public.public_inputs(directory,pins)
lock=root/'jobs/.world-reward-h100.lock'
if lock.is_symlink()or(lock.exists()and not stat.S_ISREG(lock.lstat().st_mode)):
 raise ValueError('Canonical regular cooperating GPU lock required')
for name in sorted(pins['public_files']):
 digest.update(name.encode()+b'\0'+bytes.fromhex(pins['public_files'][name]['sha256']))
directories=('weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal',
 'vendor/research/da3_metric_v1','weights/research/da3_metric_v1')
files=('weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c',
 'results/weights-acquisition.json','results/da3-metric-acquisition-v1.json',
 'vendor/research/da3_dependencies_v1/addict-2.4.0-py3-none-any.whl')
for name in directories+files:
 path=root/name;canonical(path);s=path.lstat()
 if(name in directories and not stat.S_ISDIR(s.st_mode)or name in files and(not stat.S_ISREG(s.st_mode)or s.st_size<=0)):
  raise ValueError('Exact original model/source/receipt/wheel asset paths required')
 digest.update((name+':'+str((s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns))).encode())
 if name.startswith('results/')or name.endswith('.whl'):
  value=path.read_bytes();digest.update(hashlib.sha256(value).digest())
  if name.endswith('.whl')and(len(value)!=3832 or hashlib.sha256(value).hexdigest()!='249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc'):
   raise ValueError('Exact original Addict wheel byte count/SHA required')
# HF repo snapshots contain legitimate audited links to the separately mounted
# flat blob. The model loader verifies their exact targets and content SHA.
print(digest.hexdigest())
PYINTEGRITY
}
cleanup_container() {
 # Killing the docker CLI alone does not necessarily stop its GPU workload.
 # Every lifecycle action targets only this newly reserved revision container.
 [[ "$OWNED_CONTAINER" == 1 ]] || return 0
 local remaining
 remaining="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")" || return 1
 if [[ -n "$remaining" ]];then
  timeout --signal=TERM --kill-after=2s 12s docker stop --time 5 "$CONTAINER" >/dev/null 2>&1 || true
  remaining="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")" || return 1
  if [[ -n "$remaining" ]];then
   timeout --signal=TERM --kill-after=2s 5s docker kill "$CONTAINER" >/dev/null 2>&1 || true
   timeout --signal=TERM --kill-after=2s 5s docker rm --force "$CONTAINER" >/dev/null 2>&1 || true
  fi
 fi
 remaining="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")" || return 1
 if [[ -n "$remaining" ]];then
  echo 'Owned analytic inference container still exists after bounded cleanup' >&2;return 1
 fi
}
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 cleanup_container || STATUS=1
 if [[ -n "$BEFORE" ]];then
  AFTER="$(integrity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then
   echo 'Original anchor sources/public bytes/asset identities changed' >&2;STATUS=1
  fi
  IMAGE_AFTER="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
  if [[ "$IMAGE_AFTER" != "$PINNED_IMAGE" ]];then echo 'Exact VM02 anchor image changed' >&2;STATUS=1;fi
 fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
BEFORE="$(integrity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PYOUTPUT'
from pathlib import Path
import sys
out=Path(sys.argv[1])
if(not out.parent.is_dir()or out.exists()or out.resolve()!=out or any(p.is_symlink()for p in(out,*out.parents))):
 raise ValueError('Fresh analytic blind prediction output required; no overwrite/retry/cleanup')
PYOUTPUT
SOURCES="$(env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" <<'PYSOURCES'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(sys.argv[1])/'infra'))
from analytic_rgbd_inputs import filenames
print('\n'.join(('manifest.json',*filenames())))
PYSOURCES
)"
IMAGE="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE" != "$PINNED_IMAGE" ]];then echo 'Exact VM02 anchor image differs' >&2;exit 1;fi
exec 9>"$ROOT/jobs/.world-reward-h100.lock"
flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${APPS//[[:space:]]/}" ]];then echo 'GPU busy; analytic inference not started' >&2;exit 1;fi
EXISTING="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")"
if [[ -n "$EXISTING" ]];then
 echo 'Existing analytic revision container preserved; no retry or replacement' >&2;exit 1
fi
MOUNTS=(--mount "type=bind,src=$CODE/infra,dst=$CODE/infra,readonly"
 --mount "type=bind,src=$CODE/src,dst=$CODE/src,readonly"
 --mount "type=bind,src=$CODE/configs/analytic_rgbd_input_pins.json,dst=$CODE/configs/analytic_rgbd_input_pins.json,readonly"
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly"
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly")
while IFS= read -r name;do
 path="$BASE/inputs/$name";MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done <<< "$SOURCES"
for path in "$MOGE" "$BLOB" "$MOGE_REPORT" "$DA3_SOURCE" "$DA3_WEIGHTS" "$DA3_REPORT" "$WHEEL";do
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir -m 700 "$OUT";chown 1000:1000 "$OUT"
OWNED_CONTAINER=1
set +e
timeout --signal=TERM --kill-after=10s 303s docker run --rm --name "$CONTAINER" --gpus all --network none --memory 32g --cpus 4 \
 --user 1000 --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra:$WHEEL" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/analytic_rgbd_infer.py"
STATUS=$?
set -e
exit "$STATUS"
