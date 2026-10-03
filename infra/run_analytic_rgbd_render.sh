#!/usr/bin/env bash
# CPU-only new analytic RGBD manufacture: no GPU, models, data or challenge mount.
# Source closure: /infra/analytic_rgbd_render.py
set -euo pipefail; [[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_analytic_rgbd_render/code" && "$(uname -s)" == Linux ]] || exit 2
OUT="$ROOT/validation/analytic_rgbd_holdout_v1"; JOB="${CODE%/code}"
IMAGE_PIN=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,re,stat,sys
from pathlib import Path
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
if code!=root/'jobs'/rev/'run_analytic_rgbd_render/code' or entry!=code/'infra/run_analytic_rgbd_render.sh':
 raise ValueError('Exact immutable analytic manufacture source namespace required')
for path in (root,code,entry):
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Canonical actual source paths required')
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
for name in ('infra/analytic_rgbd_render.py','infra/run_analytic_rgbd_render.sh','configs/analytic_rgbd_protocol.json'):
 if not (code/name).is_file():raise ValueError('Complete manufacture source and preregistered protocol required')
print(digest.hexdigest())
PY
}
BEFORE="$(integrity)"; env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
if out.exists() or out.resolve()!=out or any(p.is_symlink() for p in (out,*out.parents)) or not out.parent.is_dir():
 raise ValueError('Fresh exclusive analytic cohort required; no overwrite or rescue')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect "$IMAGE_PIN" --format '{{.Id}}')"; [[ "$IMAGE" == "$IMAGE_PIN" ]] || exit 1
CONTAINER="world-reward-analytic-rgbd-${REV:0:12}-$$"
OWNED=0; RESERVED=0
cleanup() {
 [[ "$OWNED" == 1 ]] || return 0
 timeout --signal=TERM --kill-after=2s 4s docker stop --time 2 "$CONTAINER" >/dev/null 2>&1 || true
 timeout --signal=TERM --kill-after=2s 4s docker kill "$CONTAINER" >/dev/null 2>&1 || true
 timeout --signal=TERM --kill-after=2s 4s docker rm --force "$CONTAINER" >/dev/null 2>&1 || true
 local remaining; remaining="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^${CONTAINER}$")" || return 1
 [[ -z "$remaining" ]] || return 1
 OWNED=0
}
fail_output() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$OUT" "$1" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);private=out/'eval_private';private.mkdir(mode=0o700,exist_ok=True);receipt=private/'render-report.json'
for folder in (out/'inputs',private):
 if folder.is_dir():
  for path in folder.iterdir():
   if path!=receipt:path.unlink()
report=json.loads(receipt.read_text()) if receipt.is_file() else {'schema':'world_reward.analytic_rgbd_manufacture.v1','stage':'analytic_object_rgbd_manufacture'}
report.update(manufacture_status=report.get('status'),status='fail',phase='wrapper_failed',wrapper_exit_status=int(sys.argv[2]),arrays_removed=True,public_files={},private_files={})
if receipt.exists():receipt.chmod(0o600)
receipt.write_text(json.dumps(report,allow_nan=False)+'\n');receipt.chmod(0o400)
PY
}
finish() { local status=$?; trap - EXIT TERM INT; cleanup || exit 1; [[ "$status" == 0 || "$RESERVED" == 0 ]] || fail_output "$status"; exit "$status"; }
trap finish EXIT; trap 'exit 143' TERM; trap 'exit 130' INT
EXISTING="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^${CONTAINER}$")"; [[ -z "$EXISTING" ]] || exit 1
mkdir -m 700 "$OUT"; RESERVED=1; chown 1000:1000 "$OUT"
OWNED=1; set +e
timeout --signal=TERM --kill-after=10s 363s docker run --rm --name "$CONTAINER" --network none --memory 8g --cpus 4 \
 --user 1000 --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" \
 --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" --env WR_RENDER_OUTPUT_RESERVED=1 \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env HOME=/tmp \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/analytic_rgbd_render.py" \
 --protocol "$CODE/configs/analytic_rgbd_protocol.json"
STATUS=$?; cleanup || exit 1; set -e
AFTER="$(integrity)"; [[ "$BEFORE" == "$AFTER" && "$(docker image inspect "$IMAGE_PIN" --format '{{.Id}}')" == "$IMAGE_PIN" ]] || STATUS=1
exit "$STATUS"
