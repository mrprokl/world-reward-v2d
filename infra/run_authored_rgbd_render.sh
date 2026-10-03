#!/usr/bin/env bash
# New authored-only RGBD manufacture. No data, models or challenge assets mounted.
# Source closure: /infra/authored_rgbd_render.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_authored_rgbd_render/code" && "$(uname -s)" == Linux ]] || exit 2
OUT="$ROOT/validation/authored_rgbd_holdout_v1"; JOB="${CODE%/code}"
IMAGE_PIN=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,re,stat,sys
from pathlib import Path
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
if code!=root/'jobs'/rev/'run_authored_rgbd_render/code' or entry!=code/'infra/run_authored_rgbd_render.sh':
 raise ValueError('Exact immutable new manufacture source namespace required')
for path in (root,code,entry):
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical actual source paths required')
digest=hashlib.sha256()
for name in ('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Original regular dispatch markers required')
 raw=path.read_bytes()
 if (name=='revision' and raw!=(rev+'\n').encode() or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):
  raise ValueError('Original exact revision/archive dispatch markers required')
 digest.update(name.encode()+b'\0'+raw)
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or mode&0o222 or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
  raise ValueError('Complete frozen source must be readonly and regular')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for name in ('infra/authored_rgbd_render.py','infra/run_authored_rgbd_render.sh','configs/authored_rgbd_protocol.json'):
 if not (code/name).is_file():raise ValueError('Complete manufacture source and preregistered protocol required')
print(digest.hexdigest())
PY
}
BEFORE="$(integrity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
if out.exists() or out.resolve()!=out or any(p.is_symlink() for p in (out,*out.parents)) or not out.parent.is_dir():
 raise ValueError('Fresh exclusive authored cohort required; no overwrite or rescue')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect "$IMAGE_PIN" --format '{{.Id}}')"
[[ "$IMAGE" == "$IMAGE_PIN" ]] || exit 1
exec 9>"$ROOT/jobs/.world-reward-h100.lock"
flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU busy; manufacture not started' >&2; exit 1; }
mkdir -m 700 "$OUT"; chown 1000:1000 "$OUT"
set +e
timeout --signal=TERM --kill-after=10s 363s docker run --rm --gpus all --network none --memory 8g --cpus 4 \
 --user 1000 --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" \
 --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" --env WR_RENDER_OUTPUT_RESERVED=1 \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/authored_rgbd_render.py" \
 --protocol "$CODE/configs/authored_rgbd_protocol.json"
STATUS=$?
set -e
AFTER="$(integrity)"
[[ "$BEFORE" == "$AFTER" ]] || { echo 'Frozen manufacture source changed' >&2; exit 1; }
[[ "$(docker image inspect "$IMAGE_PIN" --format '{{.Id}}')" == "$IMAGE_PIN" ]] || exit 1
exit "$STATUS"
