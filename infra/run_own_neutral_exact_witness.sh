#!/usr/bin/env bash
# VM01 only via Azure job targeting; one native call, original receipt, no fit.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_own_neutral_exact_witness/code" ]]
OUT="$ROOT/validation/own_neutral_exact_witness_v1"
MODEL="$ROOT/weights/mhr/mhr_model.pt";RECEIPT="$ROOT/validation/own_grasp_capability_v2/report.json"
python3 - "$ROOT" "$CODE" "$OUT" "$MODEL" "$RECEIPT" "$REV" "${BASH_SOURCE[0]}" <<'PYSAFE'
from pathlib import Path
import hashlib,re,stat,sys
root,code,out,model,receipt=map(Path,sys.argv[1:6]);revision=sys.argv[6];executing=Path(sys.argv[7])
if (not root.is_dir() or not code.is_dir() or not out.parent.is_dir() or out.exists()
    or any(p.resolve()!=p or any(q.is_symlink() for q in (p,*p.parents)) for p in (root,code,out,model,receipt))
    or not stat.S_ISREG(model.stat().st_mode) or model.stat().st_size!=696110248
    or not stat.S_ISREG(receipt.stat().st_mode) or receipt.stat().st_mode&0o222 or receipt.stat().st_size!=18294):
 raise ValueError('Exclusive private output/exact model/original immutable v2 receipt required')
if hashlib.sha256(receipt.read_bytes()).hexdigest()!='38c56607f2abfbae7f53a1fce55e55970f97b2f4aca740773ddcc161f9ff5391':raise ValueError('Independent original v2 receipt SHA differs')
if code!=root/'jobs'/revision/'run_own_neutral_exact_witness'/'code' or executing!=code/'infra/run_own_neutral_exact_witness.sh':raise ValueError('Exact dispatched launcher namespace required')
markers={}
for name in ('revision','source-sha256'):
 path=code.parent/name
 if not stat.S_ISREG(path.lstat().st_mode) or path.resolve()!=path or any(q.is_symlink() for q in(path,*path.parents)):raise ValueError('Canonical original markers required')
 markers[name]=path.read_bytes()
if markers['revision']!=(revision+'\n').encode() or not re.fullmatch(b'[0-9a-f]{64}\n',markers['source-sha256']):raise ValueError('Exact original revision/archive markers required')
for path in (code,*code.rglob('*')):
 if path.is_symlink() or (not path.is_dir() and not stat.S_ISREG(path.lstat().st_mode)) or path.stat().st_mode&0o222:raise ValueError('Complete code snapshot must be readonly and regular')
for name in ('infra/own_neutral_exact_witness.py','infra/run_own_neutral_exact_witness.sh','infra/own_grasp_capability.py','infra/run_own_grasp_capability.sh','src/world_reward/cross_surface.py','src/world_reward/exact_triangle_witness.py','src/world_reward/__init__.py'):
 if not (code/name).is_file():raise ValueError('Complete audited source closure required')
if {name:(code.parent/name).read_bytes() for name in markers}!=markers:raise ValueError('Original markers changed')
PYSAFE
MARKERS_BEFORE="$(sha256sum "$CODE/../revision" "$CODE/../source-sha256")"
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
# Fail before reserving output when another cooperating or actual GPU job exists.
exec 9>"$ROOT/jobs/.world-reward-h100.lock"
flock --nonblock 9
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU busy; no neutral diagnostic started' >&2; exit 1; }
mkdir -m 700 "$OUT";chown scenesmith:scenesmith "$OUT"
set +e
timeout --signal=TERM --kill-after=10s 123s docker run --rm --gpus all --network none --memory 12g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$MODEL,dst=$MODEL,readonly" \
 --mount "type=bind,src=$RECEIPT,dst=$RECEIPT,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/own_neutral_exact_witness.py"
STATUS=$?
set -e
[[ "$(sha256sum "$CODE/../revision" "$CODE/../source-sha256")" == "$MARKERS_BEFORE" ]]
[[ "$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$IMAGE" ]]
exit "$STATUS"
