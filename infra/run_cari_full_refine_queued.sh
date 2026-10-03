#!/usr/bin/env bash
# Scheduling only: acquire the existing cooperative GPU lock, not the preceding
# frontend unit's CPU completion. The child optimizer and all its gates stay exact.
# Source closure: /infra/run_cari_full_refine.sh /infra/cari_full_refine.py
set -euo pipefail
EPISODE='' WAIT_FOR='' episode_seen=0 wait_seen=0
while (( $# ));do
 case "$1" in
  --episode)
   if (( episode_seen || $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9]|[12][0-9])$ ]];then exit 2;fi
   EPISODE="$2";episode_seen=1;shift 2 ;;
  --wait-for)
   if (( wait_seen || $# < 2 )) || [[ ! "$2" =~ ^world-reward-[a-z0-9][a-z0-9-]{0,80}(\.service)?$ ]];then exit 2;fi
   WAIT_FOR="${2%.service}.service";wait_seen=1;shift 2 ;;
  *) exit 2 ;;
 esac
done
(( episode_seen && wait_seen )) || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_cari_full_refine_queued/code" && "$(uname -s)" == Linux ]] || exit 2
printf -v PADDED '%06d' "$EPISODE"
OUT="$ROOT/outputs/episode_$PADDED/cari_shared_refined_v1"
LOCK="$ROOT/jobs/.world-reward-h100.lock"
CURRENT_STAGE=preflight
phase() {
 printf '{"stage":"cari_full_refine_queued","phase":"%s","episode_index":%s,"timestamp_utc":"%s"}\n' \
  "$1" "$EPISODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$OUT" "$PADDED" "${BASH_SOURCE[0]}" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,re,stat,sys
root,code,rev,out,padded,entry=sys.argv[1:];root,code,out,entry=map(Path,(root,code,out,entry))
def canonical(path):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical queued refinement source required')
 return path
def raw(path,maximum=2_000_000,readonly=True):
 canonical(path);before=path.lstat()
 if not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or not 1<=before.st_size<=maximum or readonly and before.st_mode&0o222:raise ValueError('Bounded original regular readonly source required')
 value=path.read_bytes()
 after=path.lstat()
 if any(getattr(after,k)!=getattr(before,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Queue source changed during hashing')
 return value
for path in(root,code,out,entry):canonical(path)
if not root.is_dir()or not code.is_dir()or not out.parent.is_dir()or entry!=code/'infra/run_cari_full_refine_queued.sh':raise ValueError('Actual queued wrapper and existing episode required')
required=('infra/run_cari_full_refine_queued.sh','infra/run_cari_full_refine.sh','infra/cari_full_refine.py',
 'configs/cari_clip_'+padded+'_input_pins.json','configs/cari_clip_'+padded+'_shared_prepare_pins.json','configs/cari_clip_'+padded+'_shared_forward_pins.json')
if any(not(code/name).is_file()for name in required):raise ValueError('Complete child optimizer and frozen own episode pins required')
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 value=raw(code.parent/name,100,False)
 if(name=='revision'and value!=(rev+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',value)):raise ValueError('Exact original queue dispatch markers required')
 digest.update(name.encode()+b'\0'+value)
for path in(code,*sorted(code.rglob('*'))):
 canonical(path);mode=path.lstat().st_mode
 if mode&0o222 or not(stat.S_ISDIR(mode)or stat.S_ISREG(mode)):raise ValueError('Complete queue source must remain readonly and regular')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(raw(path)).digest())
print(digest.hexdigest())
PYINTEGRITY
}
lock_identity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$LOCK" "${1:-path}" <<'PYLOCK'
from pathlib import Path
import os,stat,sys
path=Path(sys.argv[1])
if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical existing cooperative lock required')
value=path.lstat()
if not stat.S_ISREG(value.st_mode)or value.st_nlink!=1:raise ValueError('Original single-link regular cooperative lock required')
if sys.argv[2]=='fd':
 actual=os.fstat(9)
 if(actual.st_dev,actual.st_ino)!=(value.st_dev,value.st_ino):raise ValueError('Opened lock is not the original cooperative inode')
print(str(value.st_dev)+':'+str(value.st_ino))
PYLOCK
}
BEFORE='' LOCK_BEFORE='' LOCK_OPEN=0
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$BEFORE" ]];then
  AFTER="$(integrity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Frozen queued refinement source/pins changed' >&2;STATUS=1;fi
 fi
 if (( LOCK_OPEN ));then
  AFTER_LOCK="$(lock_identity fd)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER_LOCK" != "$LOCK_BEFORE" ]];then echo 'Original cooperative lock changed' >&2;STATUS=1;fi
  exec 9>&-
 fi
 if [[ "$STATUS" != 0 ]];then phase fail;fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
phase preflight
BEFORE="$(integrity)"
[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'Frozen refinement target exists; no overwrite/resume' >&2;exit 1; }
# Require the independently named existing unit before waiting. Its later
# failure/PASS and CPU phase are irrelevant to this different episode's inputs.
LOAD="$(systemctl show "$WAIT_FOR" --property=LoadState --value)"
[[ "$LOAD" == loaded ]] || { echo 'Explicit preceding frontend unit is not loaded' >&2;exit 1; }
LOCK_BEFORE="$(lock_identity)"
exec 9<>"$LOCK"
LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
phase waiting_for_gpu_lock
flock --timeout 43200 9
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" && "$(integrity)" == "$BEFORE" \
 && ! -e "$OUT" && ! -L "$OUT" ]] || exit 1
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU compute applications present after lock; queued refinement aborted' >&2;exit 1; }
phase running_native_child
# FD9 remains open and locked in this parent until the unchanged child returns.
# No timeout stops another job; the child retains its original7200/7203s budget.
bash "$CODE/infra/run_cari_full_refine.sh" --episode "$EPISODE"
phase child_complete
