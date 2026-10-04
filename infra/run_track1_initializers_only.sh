#!/usr/bin/env bash
# Fixed-all16 public initializers only; cooperating lock does not cover legacy jobs.
set -euo pipefail
[[ $# == 2 && "$1" == --episode && "$2" =~ ^(0|[1-9]|[12][0-9])$ ]] || exit 2
EPISODE="$2";ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && ( "$CODE" == "$ROOT/jobs/$REV/run_track1_initializers_only/code" \
      || "$CODE" == "$ROOT/jobs/$REV/run_track1_frontends_queued/code" ) ]] || exit 2
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED";LOCK="$ROOT/jobs/.world-reward-h100.lock"
STAGE=preflight;BEFORE='';LOCK_BEFORE='';LOCK_OPEN=0
phase() { printf '{"mode":"initializers_only_fixed_all16","stage":"%s","phase":"%s","timestamp_utc":"%s"}\n' "$STAGE" "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; }
integrity() {
 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" <<'PY'
from pathlib import Path
import hashlib,re,stat,sys
root,code,rev,entry=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
def canonical(path):
 if path.resolve()!=path or any(p.is_symlink() for p in(path,*path.parents)):raise ValueError('Canonical source/runtime paths required')
def read(path):
 canonical(path);s=path.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('Regular original source/marker required')
 raw=path.read_bytes();a=path.lstat()
 if any(getattr(s,k)!=getattr(a,k) for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Source changed while hashing')
 return raw
for path in(root,code,entry):canonical(path)
if not root.is_dir() or not code.is_dir() or entry!=code/'infra/run_track1_initializers_only.sh':raise ValueError('Actual immutable initializer entrypoint required')
if code==root/'jobs'/rev/'run_track1_frontends_queued/code':
 queued=code/'infra'/'run_track1_frontends_queued.sh';raw=read(queued)
 if not raw or queued.stat().st_mode&0o222:raise ValueError('Actual readonly queued scheduling wrapper required')
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 raw=read(code.parent/name)
 if (name=='revision' and raw!=(rev+'\n').encode()) or (name=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):raise ValueError('Exact original dispatch markers required')
 digest.update(name.encode()+b'\0'+raw)
required=('run_track1_initializers_only.sh','run_automatic_masks.sh','run_episode_initializers.sh','cari_wrapper_common.sh',
 'run_body_smoke.sh','run_depth_smoke.sh','run_scale_smoke.sh','run_object_smoke.sh','run_cari_body_adapter_smoke.sh',
 'automatic_masks.py','body_smoke.py','depth_smoke.py','scale_smoke.py','object_smoke.py','cari_body_adapter_smoke.py')
for name in required:
 if not(code/'infra'/name).is_file():raise ValueError('Complete actual initializer child sources required')
for path in(code,*sorted(code.rglob('*'))):
 canonical(path);s=path.lstat()
 if s.st_mode&0o222 or not(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):raise ValueError('Full immutable code closure must be readonly and regular')
 digest.update(str(path.relative_to(code)).encode()+b'\0'+str(s.st_mode).encode()+b'\0')
 if stat.S_ISREG(s.st_mode):digest.update(hashlib.sha256(read(path)).digest())
print(digest.hexdigest())
PY
}
lock_identity() {
 python3 -I -B - "$LOCK" "${1:-path}" <<'PY'
from pathlib import Path
import os,stat,sys
p=Path(sys.argv[1]);s=p.lstat()
if p.resolve()!=p or any(a.is_symlink() for a in(p,*p.parents)) or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('Existing canonical single-link GPU lock required')
if sys.argv[2]=='fd' and(os.fstat(9).st_dev,os.fstat(9).st_ino)!=(s.st_dev,s.st_ino):raise ValueError('GPU lock inode changed')
print(str(s.st_dev)+':'+str(s.st_ino))
PY
}
gpu_idle() {
 local apps
 apps="$(timeout --signal=TERM --kill-after=2s 5s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || return 1
 [[ -z "${apps//[[:space:]]/}" ]] || { echo 'GPU compute applications present' >&2;return 1; }
}
finish() {
 local status=$? after
 trap - EXIT INT TERM;set +e
 if [[ -n "$BEFORE" ]];then after="$(integrity)";[[ $? == 0 && "$after" == "$BEFORE" ]] || status=1;fi
 if [[ "$LOCK_OPEN" == 1 ]];then
  after="$(lock_identity fd)";[[ $? == 0 && "$after" == "$LOCK_BEFORE" ]] || status=1
  gpu_idle || status=1;exec 9>&-
 fi
 if [[ "$status" != 0 ]];then phase fail;fi
 exit "$status"
}
trap finish EXIT;trap 'exit 130' INT;trap 'exit 143' TERM
phase start
BEFORE="$(integrity)"
python3 -I -B - "$ROOT" "$BASE" <<'PY'
from pathlib import Path
import stat,sys
root,base=map(Path,sys.argv[1:])
for path in(root/'outputs',base):
 if path.resolve()!=path or any(p.is_symlink() for p in(path,*path.parents)) or path.exists() and not path.is_dir():raise ValueError('Canonical output directory ancestors required')
for name in('automatic_masks','body_smoke','depth_smoke','scale_smoke','object_grounded','body_full','depth_full','body_full/cari_adapter'):
 path=base/name
 if path.exists() or path.is_symlink():raise ValueError('Frozen initializer target exists; no overwrite/resume: '+str(path))
gate=root/'results/camera-render.json'
if gate.resolve()!=gate or any(p.is_symlink() for p in(gate,*gate.parents)) or not stat.S_ISREG(gate.lstat().st_mode) or gate.stat().st_size==0:raise ValueError('Existing reference camera/depth gate required')
PY
phase pass
STAGE=gpu_preflight;phase start
LOCK_BEFORE="$(lock_identity)";exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]]
flock --nonblock 9;gpu_idle;phase pass
STAGE=automatic_masks;phase start
timeout --signal=TERM --kill-after=10s 600s bash "$CODE/infra/run_automatic_masks.sh" --episode "$EPISODE" --seed-frames 16 --actor-seed-observations 16
phase pass
STAGE=episode_initializers;phase start
gpu_idle
timeout --signal=TERM --kill-after=10s 5400s bash "$CODE/infra/run_episode_initializers.sh" --episode "$EPISODE"
phase pass
