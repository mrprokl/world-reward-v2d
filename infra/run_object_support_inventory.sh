#!/usr/bin/env bash
# CPU observation inventory only; no model, pose, GPU or private-validation mount.
# Source closure: /infra/object_support_inventory.py /infra/object_pose_smoke.py
# /infra/body_smoke.py /infra/depth_smoke.py
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2";ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_object_support_inventory/code" && "$(uname -s)" == Linux ]] || exit 2
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED";OUT="$BASE/object_support_inventory_v1"
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$BASE" "$OUT" "${BASH_SOURCE[0]}" <<'PYCONTROL'
from pathlib import Path
import hashlib,re,stat,sys
root,code,rev,base,out,entry=sys.argv[1:];root,code,base,out,entry=map(Path,(root,code,base,out,entry))
for path in(root,code,base,out,entry):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical original support-inventory paths required')
if not base.is_dir()or not code.is_dir()or entry!=code/'infra/run_object_support_inventory.sh':raise ValueError('Existing episode and actual immutable wrapper required')
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 path=code.parent/name
 if not path.is_file()or path.is_symlink()or not 1<=path.stat().st_size<=100:raise ValueError('Bounded actual dispatch markers required')
 value=path.read_bytes()
 if(name=='revision'and value!=(rev+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',value)):raise ValueError('Original dispatch marker identity differs')
 digest.update(name.encode()+b'\0'+value)
for path in(code,*sorted(code.rglob('*'))):
 before=path.lstat()
 if path.is_symlink()or before.st_mode&0o222 or not(stat.S_ISDIR(before.st_mode)or stat.S_ISREG(before.st_mode)):raise ValueError('Complete support code must remain readonly and regular')
 if path.is_file():
  if before.st_nlink!=1 or not 1<=before.st_size<=2_000_000:raise ValueError('Bounded single-link source required')
  value=path.read_bytes();after=path.stat()
  if any(getattr(before,k)!=getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Source changed during hashing')
  digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(value).digest())
for name in('object_support_inventory.py','run_object_support_inventory.sh','object_pose_smoke.py','body_smoke.py','depth_smoke.py'):
 if not(code/'infra'/name).is_file():raise ValueError('Complete genuine predictor/source closure required')
for name in('automatic_masks','body_full','depth_full','scale_smoke','object_grounded'):
 path=base/name
 if not path.is_dir()or path.resolve()!=path:raise ValueError('Existing selected public predictors required')
print(digest.hexdigest())
PYCONTROL
}
BEFORE=''
finish() {
 STATUS=$?;trap - EXIT;set +e
 if [[ -n "$BEFORE" ]];then
  AFTER="$(integrity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Original support inventory source changed' >&2;STATUS=1;fi
 fi
 exit "$STATUS"
}
trap finish EXIT
BEFORE="$(integrity)"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'Frozen support inventory target exists; no overwrite/resume' >&2;exit 1; }
export DOCKER_HOST="unix://$ROOT/docker.sock"
IMAGE=sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7
[[ "$(docker image inspect world-reward/cari4d-source:0.1 --format '{{.Id}}')" == "$IMAGE" ]] || exit 1
mkdir -m 700 "$OUT";chown "$(id -u scenesmith):$(id -g scenesmith)" "$OUT"
timeout --signal=TERM --kill-after=5s 903s docker run --rm --network none --memory 4g --cpus 2 \
 --user "$(id -u scenesmith):$(id -g scenesmith)" --entrypoint python \
 --env WR_ROOT="$ROOT" --env WR_CODE_REVISION="$REV" --env WR_IMAGE_ID="$IMAGE" --env WR_OUTPUT_RESERVED=1 \
 --env PYTHONPATH="$CODE/src" --env PYTHONDONTWRITEBYTECODE=1 --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 \
 --mount "type=bind,src=$CODE,dst=$CODE,readonly" \
 --mount "type=bind,src=$ROOT/results/input-manifest.json,dst=$ROOT/results/input-manifest.json,readonly" \
 --mount "type=bind,src=$ROOT/data/track_1/meta/episodes.jsonl,dst=$ROOT/data/track_1/meta/episodes.jsonl,readonly" \
 --mount "type=bind,src=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_$PADDED.mp4,dst=$ROOT/data/track_1/videos/chunk-000/observation.images.exo_camera/episode_$PADDED.mp4,readonly" \
 --mount "type=bind,src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" \
 --mount "type=bind,src=$BASE/body_full/report.json,dst=$BASE/body_full/report.json,readonly" \
 --mount "type=bind,src=$BASE/depth_full,dst=$BASE/depth_full,readonly" \
 --mount "type=bind,src=$BASE/scale_smoke/report.json,dst=$BASE/scale_smoke/report.json,readonly" \
 --mount "type=bind,src=$BASE/object_grounded/report.json,dst=$BASE/object_grounded/report.json,readonly" \
 --mount "type=bind,src=$BASE/object_grounded/intrinsics.json,dst=$BASE/object_grounded/intrinsics.json,readonly" \
 --mount "type=bind,src=$OUT,dst=$OUT" "$IMAGE" "$CODE/infra/object_support_inventory.py" --episode "$EPISODE"
