#!/usr/bin/env bash
# Separate author-only v2 CPU protocol on VM02; original v1 failure unchanged.
# Source closure: /infra/author_human_feasibility_v2.py /infra/author_human_feasibility.py
# /infra/run_author_human_feasibility.sh /src/world_reward/author_human_field.py
# /src/world_reward/cross_surface.py /src/world_reward/solid_queries.py
# /src/world_reward/__init__.py. No challenge/model assets or geometry repair.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_author_human_feasibility_v2/code" ]] || exit 2
OUT="$ROOT/validation/author_human_feasibility_v2"
ORIGINAL="$ROOT/validation/author_human_feasibility_v1/report.json"
JOB="${CODE%/code}"
PINNED_IMAGE=sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3
source_identity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" "$ORIGINAL" <<'PYSOURCE'
from pathlib import Path
import hashlib,re,stat,sys
root,code,revision,executing,original=sys.argv[1:];root,code,executing,original=map(Path,(root,code,executing,original))
if (not root.is_dir()or not code.is_dir()or code!=root/'jobs'/revision/'run_author_human_feasibility_v2'/'code'
    or executing!=code/'infra/run_author_human_feasibility_v2.sh'
    or original!=root/'validation/author_human_feasibility_v1/report.json'):
 raise ValueError('Actual separate immutable v2 namespace and original failure path required')
for path in (root,code,executing,original):
 if not path.is_absolute()or path.resolve()!=path or any(parent.is_symlink()for parent in(path,*path.parents)):
  raise ValueError('Canonical source and original receipt without symlinks required')
required=('infra/run_author_human_feasibility_v2.sh','infra/author_human_feasibility_v2.py',
          'infra/run_author_human_feasibility.sh','infra/author_human_feasibility.py',
          'src/world_reward/author_human_field.py','src/world_reward/cross_surface.py',
          'src/world_reward/solid_queries.py','src/world_reward/__init__.py')
for name in required:
 if not(code/name).is_file():raise ValueError('Complete original and v2 helper/driver source closure required')
digest=hashlib.sha256()
before=original.lstat()
if not stat.S_ISREG(before.st_mode)or before.st_mode&0o222 or before.st_size!=3655:
 raise ValueError('Exact immutable original failed receipt byte count required')
data=original.read_bytes();after=original.lstat()
state=lambda value:(value.st_dev,value.st_ino,value.st_mode,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
if (state(before)!=state(after)or hashlib.sha256(data).hexdigest()!='eef678c211714894934d6c0715efed2829cd809999c03e0225f59a9d6021f4c0'):
 raise ValueError('Original failed receipt SHA differs before any JSON parsing')
digest.update(hashlib.sha256(data).digest())
for name in ('revision','source-sha256'):
 path=code.parent/name
 if (path.resolve()!=path or any(parent.is_symlink()for parent in(path,*path.parents))
     or not stat.S_ISREG(path.lstat().st_mode)):
  raise ValueError('Canonical regular original dispatch markers required')
 value=path.read_bytes()
 if (name=='revision'and value!=(revision+'\n').encode()
     or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',value)):
  raise ValueError('Exact original dispatch revision/archive markers required')
 digest.update(name.encode()+b'\0'+value+b'\0')
for path in (code,*sorted(code.rglob('*'))):
 mode=path.lstat().st_mode
 if path.is_symlink()or(not stat.S_ISDIR(mode)and not stat.S_ISREG(mode))or mode&0o222:
  raise ValueError('Complete actual v2 source snapshot must be readonly and regular')
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
if (not out.is_absolute()or out.resolve()!=out or any(parent.is_symlink()for parent in(out,*out.parents))
    or out!=root/'validation/author_human_feasibility_v2'or not out.parent.is_dir()or out.exists()):
 raise ValueError('Absent private v2 output under existing validation parent required; no overwrite/retry/cleanup')
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
 --mount "type=bind,src=$ORIGINAL,dst=$ORIGINAL,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" -B "$CODE/infra/author_human_feasibility_v2.py"
STATUS=$?
set -e
SOURCE_AFTER="$(source_identity)"
if [[ "$SOURCE_AFTER" != "$SOURCE_BEFORE" ]];then
 echo 'Original v1 receipt or actual v2 source/markers changed' >&2
 exit 1
fi
IMAGE_AFTER="$(docker image inspect "$PINNED_IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE_AFTER" != "$PINNED_IMAGE" ]];then
 echo 'Exact original VM02 CPU image changed' >&2
 exit 1
fi
exit "$STATUS"
