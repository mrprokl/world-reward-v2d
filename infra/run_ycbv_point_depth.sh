#!/usr/bin/env bash
# Source closure: /infra/ycbv_point_depth.py /infra/tudl_holdout_inputs.py
# /infra/object_synthetic_observations.py /src/world_reward/pointmap.py
# Three frame-zero native RGB preflights only; no YCB acquisition/GT mount.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_point_depth/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_point_depth.sh" && "$(uname -s)" == Linux ]] || exit 2
BASE="$ROOT/validation/ycbv_point_pose_v1";OUT="$BASE/depth_init_v1";JOB="${CODE%/code}"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
NAME="world-reward-ycbv-point-depth-${REV:0:12}";CIDFILE='';IMAGE_BEFORE='';LOCK_BEFORE=''
MOGE="$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal"
BLOB="$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c"
MODEL_RECEIPT="$ROOT/results/weights-acquisition.json"
export DOCKER_HOST="unix://$ROOT/docker.sock"
integrity() {
 /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$CODE" "$OUT" "$MOGE" "$BLOB" "$MODEL_RECEIPT" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,runpy,stat,sys
code,out,moge,blob,receipt=map(Path,sys.argv[1:]);sys.path.insert(0,str(code/'infra'))
driver=runpy.run_path(str(code/'infra/ycbv_point_depth.py'),run_name='host_preflight')
proof=driver['host_proof'](driver['ROOT'],code,__import__('os').environ['WR_CODE_REVISION'])
digest=hashlib.sha256(proof.encode())
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents))or(not stat.S_ISDIR(mode)and not stat.S_ISREG(mode))or mode&0o222:raise ValueError('Readonly canonical original complete code required')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for path in (moge,blob,receipt):
 driver['files']._canonical(path);before=driver['files']._state(path)
 if (path==moge and not stat.S_ISDIR(before[2]))or(path!=moge and(not stat.S_ISREG(before[2])or before[4]<=0)):raise ValueError('Original narrow model asset paths required')
 digest.update((str(path)+str(before)).encode())
 if path==receipt:digest.update(hashlib.sha256(path.read_bytes()).digest())
 if before!=driver['files']._state(path):raise ValueError('Original model asset changed')
driver['files']._canonical(out)
if not out.parent.is_dir():raise ValueError('Existing public cohort parent required')
lock=driver['ROOT']/'jobs/.world-reward-h100.lock'
driver['files']._canonical(lock)
if not lock.exists():raise ValueError('Existing original cooperating GPU lock required')
state=driver['files']._state(lock)
if not stat.S_ISREG(state[2])or state[3]!=1:raise ValueError('Regular original GPU lock without aliases required')
digest.update(str(state).encode())
print(proof+' '+digest.hexdigest())
PYINTEGRITY
}
lock_fd_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$ROOT/jobs/.world-reward-h100.lock" "/proc/$$/fd/9" <<'PYLOCK'
from pathlib import Path
import hashlib,stat,sys
path,fd=map(Path,sys.argv[1:]);before=path.lstat();opened=fd.stat()
if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents))or not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or(before.st_dev,before.st_ino)!=(opened.st_dev,opened.st_ino)or before!=path.lstat():raise ValueError('FD9 must retain exact original unaliased GPU lock inode')
print(hashlib.sha256(str((before.st_dev,before.st_ino,before.st_mode,before.st_nlink,before.st_size,before.st_mtime_ns,before.st_ctime_ns)).encode()).hexdigest())
PYLOCK
}
BEFORE="$(integrity)";PROOF="${BEFORE%% *}"
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$CIDFILE" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   IDS="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 ]] || STATUS=1
   if [[ -n "$IDS" ]];then
    OWNED="$(timeout --signal=TERM --kill-after=2s 5s docker inspect "$CID" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$OWNED" == "$IMAGE|/$NAME|run_ycbv_point_depth|$REV" ]];then
     timeout --signal=TERM --kill-after=2s 15s docker rm -f "$CID" >/dev/null 2>&1 || STATUS=1
     IDS="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 && -z "$IDS" ]] || STATUS=1
    else STATUS=1;fi
   fi
   chmod 400 "$CIDFILE"
  else STATUS=1;fi
 fi
 AFTER="$(integrity)";[[ $? == 0 && "$AFTER" == "$BEFORE" ]] || STATUS=1
 if [[ -n "$LOCK_BEFORE" ]];then AFTER="$(lock_fd_identity)";[[ $? == 0 && "$AFTER" == "$LOCK_BEFORE" ]] || STATUS=1;fi
 if [[ -n "$IMAGE_BEFORE" ]];then AFTER="$(docker image inspect "$IMAGE" --format '{{.Id}}')";[[ $? == 0 && "$AFTER" == "$IMAGE_BEFORE" ]] || STATUS=1;fi
 exit "$STATUS"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
VM="$(/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - <<'PYVM'
import json,urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request('http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01',headers={'Metadata':'true'}),timeout=5)as stream:data=stream.read(65537)
if len(data)>65536:raise ValueError('Azure metadata byte bound exceeded')
value=json.loads(data)
if value.get('name')!='world-reward-ncc-h100-02'or str(value.get('resourceGroupName','')).lower()!='world-reward-research':raise ValueError('Owned Azure VM02 required')
print('verified')
PYVM
)";[[ "$VM" == verified ]]
[[ ! -e "$OUT" && ! -L "$OUT" ]]
IMAGE_BEFORE="$(docker image inspect "$IMAGE" --format '{{.Id}}')";[[ "$IMAGE_BEFORE" == "$IMAGE" ]]
exec 9<"$ROOT/jobs/.world-reward-h100.lock";LOCK_BEFORE="$(lock_fd_identity)";flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU busy; RGB preflight not started' >&2;exit 1; }
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
MOUNTS=()
for name in infra/ycbv_point_depth.py infra/run_ycbv_point_depth.sh infra/tudl_holdout_inputs.py infra/object_synthetic_observations.py src/world_reward/__init__.py src/world_reward/data.py src/world_reward/pointmap.py configs/ycbv_point_input_pins.json;do
 MOUNTS+=(--mount "type=bind,src=$CODE/$name,dst=$CODE/$name,readonly")
done
for path in "$BASE/inputs" "$JOB/revision" "$JOB/source-sha256" "$MOGE" "$BLOB" "$MODEL_RECEIPT";do
 MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly")
done
mkdir -m 700 "$OUT";chown 1000:1000 "$OUT";CIDFILE="$OUT/.container.cid"
set +e
timeout --signal=TERM --kill-after=10s 330s docker run --rm --interactive --name "$NAME" --cidfile "$CIDFILE" --label world-reward.job=run_ycbv_point_depth --label "world-reward.revision=$REV" \
 --gpus all --network none --memory 32g --cpus 4 --user 1000:1000 --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=512m \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp XDG_CACHE_HOME=/tmp \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_AZURE_VM02_VERIFIED=1 WR_YCBV_HOST_PROOF_SHA256="$PROOF" \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B - "$CODE" <<'PYDEPTH'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')];driver=code/'infra/ycbv_point_depth.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYDEPTH
STATUS=$?;set -e
exit "$STATUS"
