#!/usr/bin/env bash
# NEW data-free CPU triangle reference; actual CUDA raster is diagnostic only.
# Source closure: /infra/triangle_ray_gate.py
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_triangle_ray_gate/code" && "$(uname -s)" == Linux ]] || exit 2
OUT="$ROOT/validation/triangle_ray_gate_v1"; JOB="${CODE%/code}"
NAME="world-reward-triangle-ray-${REV:0:12}"
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
LOCK="$ROOT/jobs/.world-reward-h100.lock"
integrity() {
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,re,stat,sys
from pathlib import Path
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
def canonical(p):
 if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents)):raise ValueError('Canonical original code required')
for p in(root,code,entry):canonical(p)
if entry!=code/'infra/run_triangle_ray_gate.sh':raise ValueError('Actual frozen wrapper required')
d=hashlib.sha256()
for name in('revision','source-sha256'):
 p=code.parent/name;canonical(p);s=p.lstat()
 if not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or not 1<=s.st_size<=100:raise ValueError('Original bounded markers required')
 raw=p.read_bytes()
 if(name=='revision'and raw!=(rev+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):raise ValueError('Original source marker bytes required')
 d.update(name.encode()+b'\0'+raw)
for p in(code,*sorted(code.rglob('*'))):
 canonical(p);s=p.lstat()
 if s.st_mode&0o222 or not(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)):raise ValueError('Readonly regular complete code required')
 if stat.S_ISREG(s.st_mode):
  if s.st_nlink!=1 or not 0<=s.st_size<=2_000_000:raise ValueError('Bounded original code file required')
  raw=p.read_bytes();after=p.lstat()
  if any(getattr(s,k)!=getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Code changed during hashing')
  d.update(str(p.relative_to(code)).encode()+b'\0'+hashlib.sha256(raw).digest())
for name in('infra/triangle_ray_gate.py','infra/run_triangle_ray_gate.sh'):
 if not(code/name).is_file():raise ValueError('Complete new gate source required')
print(d.hexdigest())
PY
}
lock_identity() {
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$LOCK" "${1:-path}" <<'PY'
import os,stat,sys
from pathlib import Path
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not stat.S_ISREG(s.st_mode)or s.st_nlink!=1:raise ValueError('Original cooperative lock required')
if sys.argv[2]=='fd'and(os.fstat(9).st_dev,os.fstat(9).st_ino)!=(s.st_dev,s.st_ino):raise ValueError('Lock inode changed')
print(str(s.st_dev)+':'+str(s.st_ino))
PY
}
container_ids() {
 timeout --signal=TERM --kill-after=2s 4s docker ps -a --filter "name=^/${NAME}$" --format '{{.ID}}'
}
gpu_idle() {
 local apps
 apps="$(timeout --signal=TERM --kill-after=2s 4s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
 [[ -z "${apps//[[:space:]]/}" ]]
}
BEFORE="$(integrity)"; OWNED=0; LOCK_OPEN=0; LOCK_BEFORE=''
cleanup() {
 local status=$? ids after
 trap - EXIT INT TERM;set +e
 if (( OWNED ));then
  timeout --signal=TERM --kill-after=1s 4s docker stop --time 2 "$NAME" >/dev/null 2>&1 || true
  timeout --signal=TERM --kill-after=1s 3s docker kill "$NAME" >/dev/null 2>&1 || true
  timeout --signal=TERM --kill-after=1s 3s docker rm "$NAME" >/dev/null 2>&1 || true
  ids="$(container_ids)";[[ $? == 0 && -z "$ids" ]] || status=1
 fi
 after="$(integrity)";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1
 if (( LOCK_OPEN ));then
  after="$(lock_identity fd)";[[ $? == 0 && "$after" == "$LOCK_BEFORE" ]] || status=1
  if (( OWNED ));then gpu_idle || status=1;fi
  exec 9>&-
 fi
 if [[ "$status" == 0 ]];then
  printf '{"stage":"triangle_ray_gate_wrapper","status":"pass","source_unchanged":true,"owned_container_absent":true,"actual_GPU_compute_empty_after":true}\n'
 fi
 exit "$status"
}
trap cleanup EXIT;trap 'exit 143' TERM;trap 'exit 130' INT
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
if p.exists()or p.is_symlink()or p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not p.parent.is_dir():raise ValueError('Exclusive new private output required')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(timeout --signal=TERM --kill-after=2s 4s docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]] || exit 1
IDS="$(container_ids)";[[ -z "$IDS" ]] || exit 1
LOCK_BEFORE="$(lock_identity)";exec 9<>"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
flock --nonblock 9;gpu_idle || exit 1
[[ "$BEFORE" == "$(integrity)" && ! -e "$OUT" && ! -L "$OUT" ]] || exit 1
mkdir -m 700 "$OUT";chown 1000:1000 "$OUT";OWNED=1
timeout --signal=TERM --kill-after=10s 103s docker run --rm --name "$NAME" --gpus all --network none --memory 4g --cpus 2 \
 --user 1000 --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache \
 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/triangle_ray_gate.py"
