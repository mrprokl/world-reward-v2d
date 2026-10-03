#!/usr/bin/env bash
# One fresh private state; no public data, previous poses, models beyond MHR.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_own_grasp_capability/code" ]]
OUT="$ROOT/validation/own_grasp_capability_v2";MODEL="$ROOT/weights/mhr/mhr_model.pt"
python3 - "$ROOT" "$CODE" "$OUT" "$MODEL" "$REV" "${BASH_SOURCE[0]}" <<'PYSAFE'
from pathlib import Path
import re,stat,sys
root,code,out,model=map(Path,sys.argv[1:5]);revision=sys.argv[5];executing=Path(sys.argv[6])
if (not root.is_dir() or not code.is_dir() or not out.parent.is_dir() or out.exists()
    or any(p.resolve()!=p or any(q.is_symlink() for q in (p,*p.parents)) for p in (root,code,out,model))
    or not stat.S_ISREG(model.stat().st_mode) or model.stat().st_size!=696110248):
 raise ValueError('Exclusive canonical private output and exact original model required')
if code!=root/'jobs'/revision/'run_own_grasp_capability'/'code':raise ValueError('Exact dispatched code namespace required')
if executing!=code/'infra/run_own_grasp_capability.sh':raise ValueError('Actual immutable launcher entrypoint required')
markers={}
for name in ('revision','source-sha256'):
 path=code.parent/name
 if (not stat.S_ISREG(path.lstat().st_mode) or path.resolve()!=path
     or any(q.is_symlink() for q in (path,*path.parents))):
  raise ValueError('Canonical regular original dispatch identity markers required')
 markers[name]=path.read_bytes()
if markers['revision']!=(revision+'\n').encode() or not re.fullmatch(b'[0-9a-f]{64}\n',markers['source-sha256']):
 raise ValueError('Exact original revision/archive identity markers required')
for name in ('infra/own_grasp_capability.py','infra/run_own_grasp_capability.sh','src/world_reward/cross_surface.py','src/world_reward/__init__.py'):
 path=code/name
 if (not stat.S_ISREG(path.stat().st_mode) or path.stat().st_mode&0o222
     or path.resolve()!=path or any(q.is_symlink() for q in (path,*path.parents))):
  raise ValueError('Complete immutable original code closure required')
for path in (code,*code.rglob('*')):
 if (path.is_symlink() or (not path.is_dir() and not stat.S_ISREG(path.lstat().st_mode))
     or path.stat().st_mode&0o222):
  raise ValueError('Complete original code snapshot must remain readonly and regular')
if {name:(code.parent/name).read_bytes() for name in markers}!=markers:
 raise ValueError('Original dispatch identity markers changed during readonly validation')
PYSAFE
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')"
[[ "$IMAGE" == sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7 ]]
mkdir -m 700 "$OUT";chown scenesmith:scenesmith "$OUT"
timeout --signal=TERM --kill-after=10s 303s docker run --rm --gpus all --network none --memory 12g --cpus 4 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --env HOME=/tmp --env XDG_CACHE_HOME=/tmp/world-reward-cache \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$MODEL,dst=$MODEL,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/own_grasp_capability.py"
