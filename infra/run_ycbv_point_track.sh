#!/usr/bin/env bash
# Source closure: /src/world_reward/native_frame_map.py
# Source closure: /infra/ycbv_point_track.py /infra/ycbv_point_depth.py
# /infra/camera_render.py /infra/robotap_boots_infer.py
# Same frozen full96 native pool: baseline and automatic Boots unary; no GT.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_point_track/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_point_track.sh" && "$(uname -s)" == Linux ]] || exit 2
BASE="$ROOT/validation/ycbv_point_pose_v2";OUT="$BASE/comparison_v1";JOB="${CODE%/code}"
IMAGE=sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4
PARENT=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
NAME="world-reward-ycbv-point-track-${REV:0:12}";CIDFILE='';LOCK_BEFORE='';IMAGE_BEFORE=''
MOGE="$ROOT/weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal"
BLOB="$ROOT/weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c"
MODEL_RECEIPT="$ROOT/results/weights-acquisition.json"
export DOCKER_HOST="unix://$ROOT/docker.sock"
control() {
 /usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$1" <<'PYCONTROL'
from pathlib import Path
import hashlib,json,os,runpy,stat,sys
code=Path(os.environ['WR_CODE']);sys.path[:0]=[str(code/'infra'),str(code/'src')];d=runpy.run_path(str(code/'infra/ycbv_point_track.py'),run_name='host_control');root=d['ROOT']
if sys.argv[1]=='proof':
 proof=d['host_proof'](root,code,os.environ['WR_CODE_REVISION']);digest=hashlib.sha256(proof.encode())
 for name in ('weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal','weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c','results/weights-acquisition.json'):
  path=root/name;d['depth'].files._canonical(path);s=d['depth'].files._state(path)
  d['require'](stat.S_ISDIR(s[2])if name.endswith('normal')else stat.S_ISREG(s[2])and s[4]>0,'Exact original narrow MoGe asset required');digest.update((name+str(s)).encode())
  if name.endswith('.json'):digest.update(hashlib.sha256(path.read_bytes()).digest())
 lock=root/'jobs/.world-reward-h100.lock';d['depth'].files._canonical(lock);s=lock.lstat()
 d['require'](stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Existing original unaliased GPU lock required');digest.update(str(d['depth'].files._state(lock)).encode())
 out=root/d['BASE']/d['OUTPUT'];d['depth'].files._canonical(out);d['require'](out.parent.is_dir(),'Existing cohort required')
 print(proof+' '+digest.hexdigest())
elif sys.argv[1]=='mounts':
 pins=d['validate_track_pins'](d['depth'].files.strict_json((code/d['PIN_FILE']).read_bytes()))
 for name in (*d['SOURCE_FILES'],d['PIN_FILE'],d['depth'].PIN_FILE):print(code/name)
 for name in ('revision','source-sha256'):print(code.parent/name)
 print(root/d['BASE']/'inputs')
 for kind,(folder,*_)in d['STAGES'].items():
  print(root/d['BASE']/folder/'report.json')
  for name in sorted(pins[kind]['files']):print(root/d['BASE']/folder/name)
 for name in d['BOOTS_SOURCE']:print(root/d['BOOTS_BASE']/'tapnet_source'/name)
 print(root/d['BOOTS_BASE']/'bootstapir_checkpoint_v2.pt')
else:raise ValueError('No control override allowed')
PYCONTROL
}
lock_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$ROOT/jobs/.world-reward-h100.lock" "/proc/$$/fd/9" <<'PYLOCK'
from pathlib import Path
import hashlib,stat,sys
path,fd=map(Path,sys.argv[1:]);s=path.lstat();opened=fd.stat()
if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or(s.st_dev,s.st_ino)!=(opened.st_dev,opened.st_ino):raise ValueError('FD9 must retain original GPU lock inode')
print(hashlib.sha256(str((s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)).encode()).hexdigest())
PYLOCK
}
image_identity() {
 local CHILD BASEIMAGE
 CHILD="$(docker image inspect "$IMAGE" --format '{{json .}}')";BASEIMAGE="$(docker image inspect "$PARENT" --format '{{json .}}')"
 /usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$CHILD" "$BASEIMAGE" "$IMAGE" "$PARENT" <<'PYIMAGE'
import hashlib,json,sys
child,parent=map(json.loads,sys.argv[1:3]);c,p=child['RootFS']['Layers'],parent['RootFS']['Layers']
if child['Id']!=sys.argv[3]or parent['Id']!=sys.argv[4]or child['Architecture']!='amd64'or child['Os']!='linux'or len(c)!=47 or len(p)!=44 or c[:44]!=p or hashlib.sha256(json.dumps(c,separators=(',',':')).encode()).hexdigest()!='0f1bf78024834b90e5841b8de8893fba8eec4176bf36c071516d0c1205d84fdc':raise ValueError('Exact measured Boots47-layer/parent44-prefix image required')
print(hashlib.sha256(json.dumps({'child':child['Id'],'parent':parent['Id'],'layers':c},sort_keys=True).encode()).hexdigest())
PYIMAGE
}
BEFORE="$(control proof)";PROOF="${BEFORE%% *}"
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$CIDFILE" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   IDS="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 ]] || STATUS=1
   if [[ -n "$IDS" ]];then
    OWNED="$(timeout --signal=TERM --kill-after=2s 5s docker inspect "$CID" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$OWNED" == "$IMAGE|/$NAME|run_ycbv_point_track|$REV" ]];then timeout --signal=TERM --kill-after=2s 15s docker rm -f "$CID" >/dev/null 2>&1 || STATUS=1;else STATUS=1;fi
    IDS="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 && -z "$IDS" ]] || STATUS=1
   fi
   chmod 400 "$CIDFILE"
  else STATUS=1;fi
 fi
 AFTER="$(control proof)";[[ $? == 0 && "$AFTER" == "$BEFORE" ]] || STATUS=1
 if [[ -n "$LOCK_BEFORE" ]];then AFTER="$(lock_identity)";[[ $? == 0 && "$AFTER" == "$LOCK_BEFORE" ]] || STATUS=1;fi
 if [[ -n "$IMAGE_BEFORE" ]];then AFTER="$(image_identity)";[[ $? == 0 && "$AFTER" == "$IMAGE_BEFORE" ]] || STATUS=1;fi
 exit "$STATUS"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - <<'PYVM'
import json,urllib.request
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request('http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01',headers={'Metadata':'true'}),timeout=5)as s:raw=s.read(65537)
if len(raw)>65536:raise ValueError('Bounded Azure metadata required')
v=json.loads(raw)
if v.get('name')!='world-reward-ncc-h100-02'or str(v.get('resourceGroupName','')).lower()!='world-reward-research':raise ValueError('Owned Azure VM02 required')
PYVM
[[ ! -e "$OUT" && ! -L "$OUT" ]];IMAGE_BEFORE="$(image_identity)"
exec 9<"$ROOT/jobs/.world-reward-h100.lock";LOCK_BEFORE="$(lock_identity)";flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)";[[ -z "${APPS//[[:space:]]/}" ]]
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
PATHS="$(control mounts)";MOUNTS=()
while IFS= read -r path;do MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done <<< "$PATHS"
for path in "$MOGE" "$BLOB" "$MODEL_RECEIPT";do MOUNTS+=(--mount "type=bind,src=$path,dst=$path,readonly");done
mkdir -m 700 "$OUT";CIDFILE="$OUT/.container.cid"
set +e
timeout --signal=TERM --kill-after=10s 3630s docker run --rm --interactive --name "$NAME" --cidfile "$CIDFILE" --label world-reward.job=run_ycbv_point_track --label "world-reward.revision=$REV" \
 --gpus all --network none --memory 32g --cpus 4 --user 0:0 --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=512m \
 "${MOUNTS[@]}" --mount "type=bind,src=$OUT,dst=$OUT" --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp XDG_CACHE_HOME=/tmp CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_AZURE_VM02_VERIFIED=1 WR_YCBV_HOST_PROOF_SHA256="$PROOF" \
 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B - "$CODE" <<'PYTRACK'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')];driver=code/'infra/ycbv_point_track.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYTRACK
STATUS=$?;set -e
exit "$STATUS"
