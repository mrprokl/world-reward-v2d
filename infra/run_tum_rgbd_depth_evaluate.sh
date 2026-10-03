#!/usr/bin/env bash
# Frozen predictions first, then private TUM sensor truth; CPU only and no models.
# Source closure: /infra/tum_rgbd_depth_evaluate.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_tum_rgbd_depth_evaluate/code" ]] || exit 2
BASE="$ROOT/validation/tum_rgbd_depth_holdout_v1"; OUT="$BASE/quality_anchor_v1"; JOB="${CODE%/code}"
IMAGE_PIN=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,re,stat,sys
from pathlib import Path
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
if code!=root/'jobs'/rev/'run_tum_rgbd_depth_evaluate/code' or entry!=code/'infra/run_tum_rgbd_depth_evaluate.sh':
 raise ValueError('Exact immutable private evaluation namespace required')
for path in (root,code,entry):
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical actual source paths required')
digest=hashlib.sha256()
for name in ('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Original regular markers required')
 raw=path.read_bytes()
 if (name=='revision' and raw!=(rev+'\n').encode() or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):
  raise ValueError('Exact original dispatch markers required')
 digest.update(name.encode()+b'\0'+raw)
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or mode&0o222 or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
  raise ValueError('Complete source must remain readonly and regular')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
for name in ('infra/tum_rgbd_depth_evaluate.py','infra/run_tum_rgbd_depth_evaluate.sh',
             'configs/tum_rgbd_depth_input_pins.json','configs/tum_rgbd_depth_prediction_pins.json'):
 if not (code/name).is_file():raise ValueError('Complete committed source and independently frozen actual pins required')
print(digest.hexdigest())
PY
}
BEFORE="$(integrity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$BASE" "$OUT" <<'PY'
from pathlib import Path
import sys
base,out=map(Path,sys.argv[1:])
for path in (base,out,base/'inputs',base/'anchor_predictions_v1',base/'eval_private'):
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical original frozen split paths required')
 if path!=out and not path.is_dir():raise ValueError('Complete existing frozen split required')
if out.exists():raise ValueError('Absent private evaluation output required; no retries or overwrite')
PY
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(timeout --signal=TERM --kill-after=2s 4s docker image inspect "$IMAGE_PIN" --format '{{.Id}}')"
[[ "$IMAGE" == "$IMAGE_PIN" ]] || exit 1
CONTAINER="wr-tum-rgbd-depth-evaluate-${REV:0:12}"
EXISTING="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")"
[[ -z "$EXISTING" ]] || exit 1
OWNED_CONTAINER=0
cleanup() {
 [[ "$OWNED_CONTAINER" == 1 ]] || return 0
 local remaining
 remaining="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")" || return 1
 if [[ -n "$remaining" ]]; then
  timeout --signal=TERM --kill-after=2s 10s docker stop --time 2 "$CONTAINER" >/dev/null 2>&1 || true
  timeout --signal=TERM --kill-after=2s 10s docker kill "$CONTAINER" >/dev/null 2>&1 || true
  timeout --signal=TERM --kill-after=2s 10s docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
 fi
 remaining="$(timeout --signal=TERM --kill-after=2s 4s docker ps -aq --filter "name=^/${CONTAINER}$")" || return 1
 if [[ -n "$remaining" ]]; then
  echo 'Owned whole-support evaluation container survives bounded cleanup' >&2; return 1
 fi
}
finish() {
 STATUS=$?; trap - EXIT INT TERM; set +e
 cleanup || STATUS=1
 AFTER="$(integrity)"; CHECK=$?
 if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]]; then
  echo 'Frozen evaluation source changed' >&2; STATUS=1
 fi
 if [[ "$(timeout --signal=TERM --kill-after=2s 4s docker image inspect "$IMAGE_PIN" --format '{{.Id}}')" != "$IMAGE_PIN" ]]; then STATUS=1; fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -m 700 "$OUT"; chown 1000:1000 "$OUT"
OWNED_CONTAINER=1
set +e
timeout --signal=TERM --kill-after=10s 183s docker run --name "$CONTAINER" --rm --network none --memory 8g --cpus 4 \
 --user 1000 --entrypoint python --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" \
 --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" --env "PYTHONPATH=$CODE/src:$CODE/infra" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/anchor_predictions_v1,dst=$BASE/anchor_predictions_v1,readonly" \
 --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/tum_rgbd_depth_evaluate.py"
STATUS=$?
set -e
exit "$STATUS"
