#!/usr/bin/env bash
# Four NEW authored RGB anchors only; no SAM/Body/MoGe inference or GPU.
# Source closure: /infra/bridge_rgb_anchor_render.py /infra/triangle_ray_gate.py /infra/run_triangle_ray_gate.sh
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[[ $# == 6 && "$1" == --gate-sha256 && "$2" =~ ^[0-9a-f]{64}$ \
 && "$3" == --gate-bytes && "$4" =~ ^[1-9][0-9]{0,6}$ && "$5" == --gate-producer && "$6" =~ ^[0-9a-f]{40}$ ]] || exit 2
ARGS=("$@")
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_bridge_rgb_anchor_render/code" && "$(uname -s)" == Linux && "$(id -u)" == 0 ]] || exit 2
OUT="$ROOT/validation/bridge_rgb_anchor_v1";JOB="${CODE%/code}";GATE="$ROOT/validation/triangle_ray_gate_v1/report.json"
MODEL=/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
NAME="world-reward-bridge-rgb-anchor-${REV:0:12}"
integrity() {
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$CODE" "$REV" "$GATE" "$MODEL" "${ARGS[1]}" "${ARGS[3]}" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,re,stat,sys
from pathlib import Path
code,rev,gate,model,gh,gb,entry=sys.argv[1:];code,gate,model,entry=map(Path,(code,gate,model,entry))
def canonical(p):
 if p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents)):raise ValueError('Canonical source/model/gate required')
def hashfile(p,sha=None,size=None,readonly=False):
 canonical(p);s=p.lstat()
 if not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or s.st_size<=0 or readonly and s.st_mode&0o222:raise ValueError('Original regular input required')
 digest=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
 h=digest.hexdigest()
 a=p.lstat()
 if any(getattr(s,k)!=getattr(a,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink'))or sha is not None and h!=sha or size is not None and s.st_size!=size:raise ValueError('Original byte/source identity changed')
 return h
canonical(code)
if entry!=code/'infra/run_bridge_rgb_anchor_render.sh':raise ValueError('Actual frozen wrapper required')
d=hashlib.sha256()
for name in('revision','source-sha256'):
 p=code.parent/name;hashfile(p)
 if p.stat().st_size>100:raise ValueError('Bounded markers required')
 raw=p.read_bytes()
 if(name=='revision'and raw!=(rev+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):raise ValueError('Original exact markers required')
 d.update(name.encode()+b'\0'+raw)
for p in(code,*sorted(code.rglob('*'))):
 canonical(p);s=p.lstat()
 if s.st_mode&0o222 or not(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)):raise ValueError('Readonly regular source required')
 if stat.S_ISREG(s.st_mode):
  if s.st_size>2_000_000:raise ValueError('Bounded code only required')
  d.update(str(p.relative_to(code)).encode()+b'\0'+hashfile(p,readonly=True).encode())
for name in('infra/bridge_rgb_anchor_render.py','infra/run_bridge_rgb_anchor_render.sh','infra/triangle_ray_gate.py','infra/run_triangle_ray_gate.sh'):
 if not(code/name).is_file():raise ValueError('Complete source closure required')
d.update(hashfile(gate,gh,int(gb),True).encode())
d.update(hashfile(model,'352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc',696110248,True).encode())
print(d.hexdigest())
PY
}
container_ids() { timeout --signal=TERM --kill-after=2s 4s docker ps -a --filter "name=^/${NAME}$" --format '{{.ID}}'; }
BEFORE="$(integrity)";OWNED=0
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
 exit "$status"
}
trap cleanup EXIT;trap 'exit 143' TERM;trap 'exit 130' INT
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
if p.exists()or p.is_symlink()or p.resolve()!=p or any(a.is_symlink()for a in(p,*p.parents))or not p.parent.is_dir():raise ValueError('Fresh private anchor output required')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(timeout --signal=TERM --kill-after=2s 4s docker image inspect "$IMAGE" --format '{{.Id}}')" == "$IMAGE" ]] || exit 1
IDS="$(container_ids)";[[ -z "$IDS" ]] || exit 1
umask 077;mkdir -m 700 "$OUT";OWNED=1
timeout --signal=TERM --kill-after=10s 183s docker run --rm --name "$NAME" --network none --memory 12g --cpus 4 \
 --user 0 --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env CUDA_VISIBLE_DEVICES= --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$GATE,dst=$GATE,readonly" \
 --mount "type=bind,src=$MODEL,dst=$ROOT/weights/mhr/mhr_model.pt,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/bridge_rgb_anchor_render.py" "${ARGS[@]}"
