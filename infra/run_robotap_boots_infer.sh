#!/usr/bin/env bash
# Source closure: /infra/robotap_boots_infer.py /infra/robotap_boots_acquire.py
# /configs/robotap_boots_protocol.json /configs/robotap_boots_inference_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_robotap_boots_infer/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_robotap_boots_infer.sh" ]] || exit 2
IMAGE=sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4
OUT="$ROOT/validation/robotap_boots_v1/infer_v2";LOCK="$ROOT/jobs/.world-reward-h100.lock"
NAME="world-reward-robotap-boots-infer-${REV:0:12}";CIDFILE="$OUT/.container.cid"
export DOCKER_HOST="unix://$ROOT/docker.sock"
BEFORE='';IMAGE_BEFORE='';LOCK_BEFORE='';LOCK_OPEN=0;PROOF_READY=0
control() {
 /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$CODE" "$1" <<'PYCONTROL'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));driver=code/'infra/robotap_boots_infer.py';sys.argv=[str(driver),sys.argv[2]]
runpy.run_path(str(driver),run_name='__main__')
PYCONTROL
}
image_identity() { timeout --signal=TERM --kill-after=2s 5s docker image inspect "$IMAGE" --format '{{.Id}}|{{.Architecture}}|{{.Os}}|{{json .RootFS}}'; }
lock_identity() {
 python3 -I -B - "$LOCK" "${1:-path}" <<'PYLOCK'
from pathlib import Path
import os,stat,sys
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Canonical existing single-link GPU lock required')
if sys.argv[2]=='fd'and(os.fstat(9).st_dev,os.fstat(9).st_ino)!=(s.st_dev,s.st_ino):raise ValueError('GPU lock inode changed')
print(str(s.st_dev)+':'+str(s.st_ino))
PYLOCK
}
gpu_idle() {
 local apps
 apps="$(timeout --signal=TERM --kill-after=2s 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
 [[ -z "${apps//[[:space:]]/}" ]]
}
finish() {
 local status=$? after owned cid ids
 trap - EXIT INT TERM;set +e
 if [[ -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  cid="$(cat "$CIDFILE")"
  if [[ "$cid" =~ ^[0-9a-f]{64}$ ]];then
   ids="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$cid")";[[ $? == 0 ]] || status=1
   if [[ -n "$ids" ]];then
    owned="$(docker inspect "$cid" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$owned" == "$IMAGE|/$NAME|run_robotap_boots_infer|$REV" ]];then
     timeout --signal=TERM --kill-after=2s 15s docker rm -f "$cid" >/dev/null || status=1
     ids="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$cid")";[[ $? == 0 && -z "$ids" ]] || status=1
    else status=1;fi
   fi
   chmod 400 "$CIDFILE"
  else status=1;fi
 fi
 if [[ -n "$BEFORE" ]];then
  MODE=--preflight;[[ "$PROOF_READY" == 1 ]] && MODE=--verify
  after="$(control "$MODE")";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
 fi
 if [[ -n "$IMAGE_BEFORE" ]];then after="$(image_identity)";[[ $? == 0 && "$after" == "$IMAGE_BEFORE" ]] || status=1;fi
 if [[ "$LOCK_OPEN" == 1 ]];then
  after="$(lock_identity fd)";[[ $? == 0 && "$after" == "$LOCK_BEFORE" ]] || status=1
  gpu_idle || status=1;exec 9>&-
 fi
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
BEFORE="$(control --preflight)"
[[ ! -e "$OUT" && ! -L "$OUT" ]]
IMAGE_BEFORE="$(image_identity)";[[ "$IMAGE_BEFORE" == "$IMAGE|amd64|linux|"* ]]
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
PROOF="$(control --publish-runtime-proof)";PROOF_READY=1
[[ "$PROOF" == "$ROOT/results/robotap-boots-runtime-proof-$REV/report.json" ]]
LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]];flock --nonblock 9;gpu_idle
MOUNT_PATHS="$(control --mounts)"
MOUNTS=()
while IFS=$'\t' read -r src dst;do
 [[ "$src" == "$ROOT/"* && "$dst" == "$ROOT/"* && "$src" != *','* && "$dst" != *','* ]]
 MOUNTS+=(--mount "type=bind,src=$src,dst=$dst,readonly")
done <<< "$MOUNT_PATHS"
[[ ${#MOUNTS[@]} -ge 20 ]]
mkdir -m 700 "$OUT";chown 1000:1000 "$OUT"
MOUNTS+=(--mount "type=bind,src=$OUT,dst=$OUT")
set +e
timeout --signal=TERM --kill-after=10s 915s docker run --rm --interactive --name "$NAME" --cidfile "$CIDFILE" --label world-reward.job=run_robotap_boots_infer --label "world-reward.revision=$REV" \
 --gpus all --network none --memory 32g --cpus 4 --user 1000:1000 --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=64m \
 "${MOUNTS[@]}" --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp XDG_CACHE_HOME=/tmp \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_AZURE_VM02_VERIFIED=1 WR_RUNTIME_PROOF_MIRROR=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B - "$CODE" <<'PYINFER'
import runpy,sys
from pathlib import Path
code=Path(sys.argv[1]);sys.path.insert(0,str(code/'infra'));driver=code/'infra/robotap_boots_infer.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYINFER
STATUS=$?;set -e
exit "$STATUS"
