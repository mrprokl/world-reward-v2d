#!/usr/bin/env bash
# Author-only procedural CPU geometry on VM02; no challenge/model assets.
# Source closure: /infra/author_human_feasibility.py
# /src/world_reward/author_human_field.py /src/world_reward/cross_surface.py
# /src/world_reward/__init__.py. Private child outputs are driver-owned, mode400.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_author_human_feasibility/code" ]] || exit 2
OUT="$ROOT/validation/author_human_feasibility_v1"
JOB="${CODE%/code}"
PINNED_IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
# Stdlib-only source/marker checks run before reservation and after the child,
# without importing geometry, packages, a model or any runtime dependency.
source_identity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
root,code,revision,executing=sys.argv[1:];root,code,executing=map(Path,(root,code,executing))
if (not root.is_dir() or not code.is_dir() or code!=root/'jobs'/revision/'run_author_human_feasibility'/'code'
    or executing!=code/'infra/run_author_human_feasibility.sh'):
 raise ValueError('Actual author-only immutable launcher namespace required')
for path in (root,code,executing):
 if not path.is_absolute() or path.resolve()!=path or any(parent.is_symlink() for parent in (path,*path.parents)):
  raise ValueError('Canonical actual source without symlinks required')
required=('infra/run_author_human_feasibility.sh','infra/author_human_feasibility.py',
          'src/world_reward/author_human_field.py','src/world_reward/cross_surface.py','src/world_reward/__init__.py')
for name in required:
 if not (code/name).is_file():raise ValueError('Complete author-only helper/driver source closure required')
digest=hashlib.sha256()
for name in ('revision','source-sha256'):
 path=code.parent/name
 if (path.resolve()!=path or any(parent.is_symlink() for parent in (path,*path.parents))
     or not stat.S_ISREG(path.lstat().st_mode)):
  raise ValueError('Canonical regular original dispatch markers required')
 value=path.read_bytes()
 if (name=='revision' and value!=(revision+'\n').encode()
     or name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',value)):
  raise ValueError('Exact original dispatch revision/archive markers required')
 digest.update(name.encode()+b'\0'+value+b'\0')
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink() or (not stat.S_ISDIR(mode) and not stat.S_ISREG(mode)) or mode&0o222:
  raise ValueError('Complete actual source snapshot must be readonly and regular')
 if stat.S_ISREG(mode):
  digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PYSOURCE
}
SOURCE_BEFORE="$(source_identity)"
env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$OUT" <<'PYOUTPUT'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:])
if (not out.is_absolute() or out.resolve()!=out or any(parent.is_symlink() for parent in (out,*out.parents))
    or out!=root/'validation/author_human_feasibility_v1' or not out.parent.is_dir() or out.exists()):
 raise ValueError('Absent private author output under existing validation parent required; no overwrite/restart/cleanup')
PYOUTPUT
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE" != "$PINNED_IMAGE" ]];then
 echo 'Exact original VM02 CPU image differs' >&2
 exit 1
fi
mkdir -m 700 "$OUT"
chown 1000:1000 "$OUT"
set +e
timeout --signal=TERM --kill-after=10s 1203s docker run --rm --network none --memory 12g --cpus 4 \
 --user 1000 --entrypoint python \
 --env "WR_ROOT=$ROOT" --env "WR_CODE=$CODE" --env "WR_CODE_REVISION=$REV" --env "WR_IMAGE_ID=$IMAGE" \
 --env "PYTHONPATH=$CODE/src:$CODE/infra" --env PYTHONDONTWRITEBYTECODE=1 \
 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$JOB/revision,dst=$JOB/revision,readonly" \
 --mount "type=bind,src=$JOB/source-sha256,dst=$JOB/source-sha256,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/author_human_feasibility.py"
STATUS=$?
set -e
SOURCE_AFTER="$(source_identity)"
if [[ "$SOURCE_AFTER" != "$SOURCE_BEFORE" ]];then
 echo 'Original author source/dispatch markers changed' >&2
 exit 1
fi
IMAGE_AFTER="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE_AFTER" != "$PINNED_IMAGE" ]];then
 echo 'Exact original VM02 CPU image changed' >&2
 exit 1
fi
exit "$STATUS"
