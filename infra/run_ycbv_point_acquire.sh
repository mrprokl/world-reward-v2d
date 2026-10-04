#!/usr/bin/env bash
# Source closure: /infra/ycbv_point_acquire.py /infra/tudl_acquire.py
# /configs/ycbv_point_protocol_v2.json.
# Source closure: /src/world_reward/native_frame_map.py /src/world_reward/__init__.py
# Azure VM02 CPU/public HTTPS only; no GPU.
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ && "$CODE" == "$ROOT/jobs/$REV/run_ycbv_point_acquire/code" && "${BASH_SOURCE[0]}" == "$CODE/infra/run_ycbv_point_acquire.sh" ]] || exit 2
OUT="$ROOT/validation/ycbv_point_pose_v2";JOB="${CODE%/code}"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
NAME="world-reward-ycbv-point-acquire-${REV:0:12}"
export DOCKER_HOST="unix://$ROOT/docker.sock"
source_identity() {
 /usr/bin/env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$CODE" "$REV" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
code=Path(sys.argv[1]);revision=sys.argv[2];digest=hashlib.sha256()
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents))or(not stat.S_ISDIR(mode)and not stat.S_ISREG(mode))or mode&0o222:raise ValueError('Readonly canonical full source required')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for name in ('revision','source-sha256'):
 path=code.parent/name;before=path.lstat();data=path.read_bytes()
 if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents))or not stat.S_ISREG(before.st_mode)or before!=path.lstat():raise ValueError('Original dispatch markers required')
 if(name=='revision'and data!=(revision+'\n').encode())or(name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',data)):raise ValueError('Exact dispatch markers required')
 digest.update(name.encode()+b'\0'+data)
print(digest.hexdigest())
PYSOURCE
}
# Intentionally closed: a distinct amendment and independent audit are required.
/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -B - "$CODE" <<'PYAUTH'
from pathlib import Path
import sys
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')]
from ycbv_point_acquire import require_execution
require_execution()
PYAUTH
BEFORE="$(source_identity)";IMAGE_BEFORE='';CIDFILE=''
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$CIDFILE" && -f "$CIDFILE" && ! -L "$CIDFILE" ]];then
  CID="$(cat "$CIDFILE")"
  if [[ "$CID" =~ ^[0-9a-f]{64}$ ]];then
   IDS="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 ]] || STATUS=1
   if [[ -n "$IDS" ]];then
    OWNED="$(docker inspect "$CID" --format '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}')"
    if [[ "$OWNED" == "$IMAGE|/$NAME|run_ycbv_point_acquire|$REV" ]];then
     timeout --signal=TERM --kill-after=2s 15s docker rm -f "$CID" >/dev/null || STATUS=1
     IDS="$(timeout --signal=TERM --kill-after=2s 5s docker ps -aq --no-trunc --filter "id=$CID")";[[ $? == 0 && -z "$IDS" ]] || STATUS=1
    else STATUS=1;fi
   fi
   chmod 400 "$CIDFILE"
  else STATUS=1;fi
 fi
 AFTER="$(source_identity)";[[ $? == 0 && "$AFTER" == "$BEFORE" ]] || STATUS=1
 if [[ -n "$IMAGE_BEFORE" ]];then AFTER="$(docker image inspect "$IMAGE" --format '{{.Id}}')";[[ $? == 0 && "$AFTER" == "$IMAGE_BEFORE" ]] || STATUS=1;fi
 exit "$STATUS"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
/usr/bin/env -i PATH=/usr/bin:/bin WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -I -B - "$CODE" <<'PYPREFLIGHT'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')];driver=code/'infra/ycbv_point_acquire.py';sys.argv=[str(driver),'--preflight']
runpy.run_path(str(driver),run_name='__main__')
PYPREFLIGHT
IMAGE_BEFORE="$(docker image inspect "$IMAGE" --format '{{.Id}}')";[[ "$IMAGE_BEFORE" == "$IMAGE" ]]
[[ -z "$(docker ps -aq --filter "name=^/$NAME$")" ]]
mkdir -m 700 "$OUT";chown 1000:1000 "$OUT";CIDFILE="$OUT/.container.cid"
set +e
timeout --signal=TERM --kill-after=10s 3810s docker run --rm --interactive --name "$NAME" --cidfile "$CIDFILE" --label world-reward.job=run_ycbv_point_acquire --label "world-reward.revision=$REV" \
 --network host --memory 16g --cpus 4 --user 1000:1000 --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,noexec,nosuid,size=64m \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" --mount "type=bind,src=$OUT,dst=$OUT" \
 --entrypoint /usr/bin/env "$IMAGE" -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp XDG_CACHE_HOME=/tmp \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" WR_IMAGE_ID="$IMAGE" WR_AZURE_VM02_VERIFIED=1 WR_YCBV_V2_AUTHORIZED=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
 /opt/conda/bin/python -I -B - "$CODE" <<'PYACQUIRE'
from pathlib import Path
import runpy,sys
code=Path(sys.argv[1]);sys.path[:0]=[str(code/'infra'),str(code/'src')];driver=code/'infra/ycbv_point_acquire.py';sys.argv=[str(driver)]
runpy.run_path(str(driver),run_name='__main__')
PYACQUIRE
STATUS=$?;set -e
exit "$STATUS"
