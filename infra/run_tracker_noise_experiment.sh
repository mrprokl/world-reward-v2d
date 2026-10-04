#!/usr/bin/env bash
set +x
set -euo pipefail
[[ $# == 2 && "$1" == --stage && "$2" =~ ^(public|infer|evaluate)$ ]] || exit 2
STAGE="$2";ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_tracker_noise_experiment/code" ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
BASE="$ROOT/validation/tracker_noise_v1";OUT="$BASE/$STAGE"
CONTROL="$ROOT/results/tracker-noise-control-$STAGE-$REV"
LOCK="$ROOT/jobs/.world-reward-h100.lock";LOCK_OPEN=0
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
SECONDS_LIMIT=135;GPU=()
START=$SECONDS;TOTAL=900
if [[ "$STAGE" == infer ]];then
 IMAGE=sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4
 SECONDS_LIMIT=915;TOTAL=1350;GPU=(--gpus all)
fi
NAME="world-reward-tracker-noise-$STAGE-${REV:0:12}";CID="$CONTROL/.container.cid";BEFORE='';IMAGE_BEFORE=''
# Source closure: /infra/tracker_noise_experiment.py /infra/robotap_boots_acquire.py
# /infra/robotap_boots_public.py /infra/robotap_boots_infer.py /src/world_reward/tracker_noise_null.py
limit() { local left=$((TOTAL-SECONDS+START));((left>0));if ((left<$1));then echo "$left";else echo "$1";fi; }
control() {
 timeout --signal=TERM --kill-after=5s "$(limit 135)s" /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$STAGE" "$1" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'))
p=code/'infra/tracker_noise_experiment.py';sys.argv=[str(p),'--stage',sys.argv[2],'--control',sys.argv[3]]
runpy.run_path(str(p),run_name='__main__')
PYCONTROL
}
image_identity() { timeout 5s docker image inspect "$IMAGE" --format '{{.Id}}|{{.Architecture}}|{{.Os}}|{{json .RootFS}}'; }
gpu_idle() { local p;p="$(timeout 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)";[[ -z "${p//[[:space:]]/}" ]]; }
lock_identity() {
 python3 -I -B - "$LOCK" "${1:-path}" <<'PYLOCK'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1]);s=p.lstat()
assert p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents))and stat.S_ISREG(s.st_mode)and s.st_nlink==1
if sys.argv[2]=='fd':assert(os.fstat(9).st_dev,os.fstat(9).st_ino)==(s.st_dev,s.st_ino)
print(str(s.st_dev)+':'+str(s.st_ino))
PYLOCK
}
finish() {
 local status=$? cid owned after ids
 trap - EXIT INT TERM;set +e
 if [[ -f "$CID" && ! -L "$CID" ]];then
  cid="$(cat "$CID")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]];then
   ids="$(docker ps -aq --no-trunc --filter "id=$cid")";[[ $? == 0 ]] || status=1
   if [[ -n "$ids" ]];then
    owned="$(docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$owned" == "$IMAGE|/$NAME|run_tracker_noise_experiment|$REV" ]];then
     timeout 15s docker rm -f "$cid" >/dev/null || status=1
     ids="$(timeout 5s docker ps -aq --no-trunc --filter "id=$cid")";[[ $? == 0 && -z "$ids" ]] || status=1
    else status=1;fi
   fi
  else status=1;fi
  chmod 400 "$CID"
 fi
 if [[ -n "$BEFORE" ]];then after="$(control preflight)";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1;fi
 if [[ -n "$IMAGE_BEFORE" ]];then after="$(image_identity)";[[ $? == 0 && "$after" == "$IMAGE_BEFORE" ]] || status=1;fi
 if [[ "$LOCK_OPEN" == 1 ]];then [[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || status=1;gpu_idle || status=1;exec 9>&-;fi
 ((SECONDS-START<=TOTAL)) || status=1
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
BEFORE="$(control preflight)";IMAGE_BEFORE="$(image_identity)";[[ "$IMAGE_BEFORE" == "$IMAGE|amd64|linux|"* ]]
[[ ! -e "$OUT" && ! -L "$OUT" && ! -e "$CONTROL" && ! -L "$CONTROL" ]]
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
if [[ "$STAGE" == infer ]];then
 control mirror >/dev/null
 LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK";LOCK_OPEN=1
 [[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]];flock --nonblock 9;gpu_idle
fi
MOUNT_PATHS="$(control mounts)";MOUNTS=()
while IFS=$'\t' read -r src dst;do
 [[ "$src" == "$ROOT/"* && "$dst" == "$ROOT/"* && "$src" != *','* && "$dst" != *','* ]]
 MOUNTS+=(--mount "type=bind,src=$src,dst=$dst,readonly")
done <<< "$MOUNT_PATHS"
if [[ ! -e "$BASE" && ! -L "$BASE" ]];then mkdir -m 755 "$BASE";fi
[[ -d "$BASE" && ! -L "$BASE" ]]
mkdir -m 700 "$CONTROL" "$OUT";chown 1000:1000 "$OUT"
MOUNTS+=(--mount "type=bind,src=$OUT,dst=$OUT")
set +e
timeout --signal=TERM --kill-after=10s "$(limit "$SECONDS_LIMIT")s" docker run --rm --interactive --name "$NAME" --cidfile "$CID" \
 --label world-reward.job=run_tracker_noise_experiment --label "world-reward.revision=$REV" \
 "${GPU[@]}" --network none --memory 32g --cpus 4 --user 1000:1000 --read-only --cap-drop ALL --security-opt no-new-privileges \
 --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m "${MOUNTS[@]}" --entrypoint /usr/bin/env "$IMAGE" -i \
 PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_VM02_VERIFIED=1 \
 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B - "$CODE" "$STAGE" <<'PYSTAGE'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));sys.path.insert(0,str(code/'src'))
p=code/'infra/tracker_noise_experiment.py';sys.argv=[str(p),'--stage',sys.argv[2]]
runpy.run_path(str(p),run_name='__main__')
PYSTAGE
STATUS=$?;set -e
exit "$STATUS"
