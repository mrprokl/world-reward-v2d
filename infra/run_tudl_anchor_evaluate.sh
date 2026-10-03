#!/usr/bin/env bash
# CPU-only, once-frozen twelve-frame depth comparison. No prediction refit.
# Source closure: /infra/tudl_anchor_evaluate.py /infra/tudl_anchor_infer.py
# /infra/tudl_holdout_inputs.py /infra/tudl_evaluate.py.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_tudl_anchor_evaluate/code" ]] || exit 2
BASE="$ROOT/validation/tudl_frame_holdout_v1"
OUT="$BASE/quality_anchor_v1"; JOB="${CODE%/code}"
PINNED_IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
source_identity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
root,code,revision,executing=sys.argv[1:]; root,code,executing=map(Path,(root,code,executing))
if (not root.is_dir() or not code.is_dir()
    or code!=root/'jobs'/revision/'run_tudl_anchor_evaluate'/'code'
    or executing!=code/'infra/run_tudl_anchor_evaluate.sh'):
 raise ValueError('Exact immutable CPU evaluation source namespace required')
for path in (root,code,executing):
 if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical original source paths required')
for name in ('infra/run_tudl_anchor_evaluate.sh','infra/tudl_anchor_evaluate.py',
             'infra/tudl_anchor_infer.py','infra/tudl_holdout_inputs.py','infra/tudl_evaluate.py',
             'configs/tudl_frame_holdout_input_pins.json','configs/tudl_anchor_prediction_pins.json'):
 if not (code/name).is_file(): raise ValueError('Complete source and independent actual prediction pins required')
digest=hashlib.sha256()
for name in ('revision','source-sha256'):
 path=code.parent/name
 if path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)) or not stat.S_ISREG(path.lstat().st_mode):
  raise ValueError('Canonical original dispatch marker required')
 value=path.read_bytes()
 if (name=='revision' and value!=(revision+'\n').encode()
     or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',value)):
  raise ValueError('Exact original revision/archive markers required')
 digest.update(name.encode()+b'\0'+value+b'\0')
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)) or mode&0o222:
  raise ValueError('Complete original source snapshot must remain readonly and regular')
 if stat.S_ISREG(mode):
  digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PYSOURCE
}
SOURCE_BEFORE="$(source_identity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$BASE" "$OUT" <<'PYOUTPUT'
from pathlib import Path
import sys
root,base,out=map(Path,sys.argv[1:])
if base!=root/'validation/tudl_frame_holdout_v1' or out!=base/'quality_anchor_v1':
 raise ValueError('Exact external validation namespace required')
for path in (root,base,out,base/'inputs',base/'anchor_predictions_v1',base/'eval_private'):
 if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
  raise ValueError('Canonical evaluation split paths required')
 if path!=out and not path.is_dir(): raise ValueError('Existing complete frozen validation split required')
if out.exists(): raise ValueError('Absent evaluation output required; no overwrite/retry/cleanup')
PYOUTPUT
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE" != "$PINNED_IMAGE" ]]; then echo 'Exact original VM02 CPU image differs' >&2; exit 1; fi
mkdir -m 700 "$OUT"; chown 1000:1000 "$OUT"
set +e
timeout --signal=TERM --kill-after=10s 183s docker run --rm --network none --memory 8g --cpus 4 \
 --user 1000 --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$BASE/inputs,dst=$BASE/inputs,readonly" \
 --mount "type=bind,src=$BASE/anchor_predictions_v1,dst=$BASE/anchor_predictions_v1,readonly" \
 --mount "type=bind,src=$BASE/eval_private,dst=$BASE/eval_private,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/tudl_anchor_evaluate.py"
STATUS=$?
set -e
SOURCE_AFTER="$(source_identity)"
if [[ "$SOURCE_AFTER" != "$SOURCE_BEFORE" ]]; then echo 'Original evaluation source/markers changed' >&2; exit 1; fi
IMAGE_AFTER="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE_AFTER" != "$PINNED_IMAGE" ]]; then echo 'Original CPU image changed' >&2; exit 1; fi
exit "$STATUS"
