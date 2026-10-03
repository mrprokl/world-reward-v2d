#!/usr/bin/env bash
# Independent four-face scalar CUDA diagnostic. No data/models/cohort mounts.
# Source closure: /infra/raster_precision_probe.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_raster_precision_probe/code" && "$(uname -s)" == Linux ]] || exit 2
OUT="$ROOT/validation/raster_precision_probe_v1"; JOB="${CODE%/code}"
NAME="world-reward-raster-precision-${REV:0:12}"
IMAGE_PIN=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,re,stat,sys
from pathlib import Path
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
if code!=root/'jobs'/rev/'run_raster_precision_probe/code' or entry!=code/'infra/run_raster_precision_probe.sh':
 raise ValueError('Exact immutable diagnostic source namespace required')
digest=hashlib.sha256()
for path in (root,code,entry):
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Canonical source paths required')
for name in ('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Original regular dispatch markers required')
 raw=path.read_bytes()
 if (name=='revision' and raw!=(rev+'\n').encode() or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):
  raise ValueError('Original exact source markers required')
 digest.update(name.encode()+b'\0'+raw)
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or mode&0o222 or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):raise ValueError('Frozen source must be readonly regular')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for name in ('infra/raster_precision_probe.py','infra/run_raster_precision_probe.sh'):
 if not (code/name).is_file():raise ValueError('Complete independent diagnostic source required')
print(digest.hexdigest())
PY
}
BEFORE="$(integrity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
if out.exists() or out.resolve()!=out or any(p.is_symlink() for p in (out,*out.parents)) or not out.parent.is_dir():
 raise ValueError('Fresh exclusive independent diagnostic destination required')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
[[ "$(docker image inspect "$IMAGE_PIN" --format '{{.Id}}')" == "$IMAGE_PIN" ]] || exit 1
if docker container inspect "$NAME" >/dev/null 2>&1; then echo 'Diagnostic container name already exists; refuse reuse' >&2; exit 1; fi
OWNED=0
cleanup() {
 local status=$?; trap - EXIT TERM INT
 if [[ "$OWNED" == 1 ]]; then
  timeout --signal=TERM --kill-after=1s 4s docker stop --time 2 "$NAME" >/dev/null 2>&1 || true
  timeout --signal=TERM --kill-after=1s 3s docker kill "$NAME" >/dev/null 2>&1 || true
  timeout --signal=TERM --kill-after=1s 3s docker rm "$NAME" >/dev/null 2>&1 || true
  if timeout --signal=TERM --kill-after=1s 2s docker container inspect "$NAME" >/dev/null 2>&1; then echo 'Own diagnostic container remains' >&2; status=1; fi
 fi
 exit "$status"
}
trap cleanup EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
exec 9>"$ROOT/jobs/.world-reward-h100.lock"; flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU busy; independent diagnostic not started' >&2; exit 1; }
mkdir -m 700 "$OUT"; chown 1000:1000 "$OUT"; OWNED=1
set +e
timeout --signal=TERM --kill-after=10s 33s docker run --rm --name "$NAME" --gpus all --network none --memory 4g --cpus 2 \
 --user 1000 --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE_PIN" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache \
 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE_PIN" -B "$CODE/infra/raster_precision_probe.py"
STATUS=$?; set -e
[[ "$BEFORE" == "$(integrity)" ]] || { echo 'Frozen diagnostic source changed' >&2; exit 1; }
[[ "$(docker image inspect "$IMAGE_PIN" --format '{{.Id}}')" == "$IMAGE_PIN" ]] || exit 1
exit "$STATUS"
