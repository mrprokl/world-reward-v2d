#!/usr/bin/env bash
# Scheduling only. Wait for the explicit predecessor policy and its released
# cooperative GPU lock; the original frontend child reacquires its own lock.
# Source closure: /infra/run_track1_frontends.sh /infra/run_episode_initializers.sh
# /infra/run_automatic_masks.sh /infra/run_object_pose_smoke.sh /infra/run_cari_prepare.sh
set -euo pipefail
EPISODE='' WAIT_FOR='' WAIT_MODE=success episode_seen=0 wait_seen=0
while (( $# ));do
 case "$1" in
  --episode)
   if (( episode_seen || $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9]|[12][0-9])$ ]];then exit 2;fi
   EPISODE="$2";episode_seen=1;shift 2 ;;
  --wait-for|--after-terminal)
   if (( wait_seen || $# < 2 )) || [[ ! "$2" =~ ^world-reward-[a-z0-9][a-z0-9-]{0,80}(\.service)?$ ]];then exit 2;fi
   if [[ "$1" == --after-terminal ]];then WAIT_MODE=terminal;fi
   WAIT_FOR="${2%.service}.service";wait_seen=1;shift 2 ;;
  --after-gpu-lock)
   (( ! wait_seen )) || exit 2
   WAIT_MODE=lock;wait_seen=1;shift ;;
  *) exit 2 ;;
 esac
done
(( episode_seen && wait_seen )) || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_track1_frontends_queued/code" && "$(uname -s)" == Linux ]] || exit 2
printf -v PADDED '%06d' "$EPISODE"
BASE="$ROOT/outputs/episode_$PADDED"
LOCK="$ROOT/jobs/.world-reward-h100.lock"
phase() {
 printf '{"stage":"track1_frontends_queued","phase":"%s","episode_index":%s,"timestamp_utc":"%s"}\n' \
  "$1" "$EPISODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$BASE" "${BASH_SOURCE[0]}" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,re,stat,sys
root,code,rev,base,entry=sys.argv[1:];root,code,base,entry=map(Path,(root,code,base,entry))
def canonical(path):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical queued frontend paths required')
def raw(path,maximum=2_000_000,readonly=True):
 canonical(path);before=path.lstat()
 if not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or not 1<=before.st_size<=maximum or readonly and before.st_mode&0o222:raise ValueError('Bounded original regular readonly source required')
 value=path.read_bytes();after=path.lstat()
 if any(getattr(after,k)!=getattr(before,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Queue source changed during hashing')
 return value
for path in(root,code,base,entry):canonical(path)
if not root.is_dir()or not code.is_dir()or not(root/'outputs').is_dir()or entry!=code/'infra/run_track1_frontends_queued.sh':raise ValueError('Actual queued wrapper and existing outputs root required')
if base.exists()and not base.is_dir():raise ValueError('Selected episode ancestor must be a directory')
required=('run_track1_frontends_queued.sh','run_track1_frontends.sh','run_episode_initializers.sh',
 'run_automatic_masks.sh','run_object_pose_smoke.sh','run_cari_prepare.sh')
if any(not(code/'infra'/name).is_file()for name in required):raise ValueError('Complete original frontend children required')
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
targets_absent() {
 local target
 for target in automatic_masks body_smoke depth_smoke scale_smoke object_grounded body_full depth_full body_full/cari_adapter object_pose_full cari_inputs;do
  [[ ! -e "$BASE/$target" && ! -L "$BASE/$target" ]] || { echo 'Frozen frontend target exists; no overwrite/resume' >&2;return 1; }
 done
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
unit_state() {
 local line key value load='' active='' result='' main='' count=0
 local state
 state="$(systemctl show "$WAIT_FOR" --property=LoadState --property=ActiveState --property=Result --property=ExecMainStatus)" || return 1
 while IFS= read -r line;do
  key="${line%%=*}";value="${line#*=}"
  [[ "$line" == *=* && -n "$value" ]] || return 1
  case "$key" in
   LoadState) [[ -z "$load" ]] || return 1;load="$value" ;;
   ActiveState) [[ -z "$active" ]] || return 1;active="$value" ;;
   Result) [[ -z "$result" ]] || return 1;result="$value" ;;
   ExecMainStatus) [[ -z "$main" ]] || return 1;main="$value" ;;
   *) return 1 ;;
  esac
  count=$((count+1))
 done <<< "$state"
 [[ "$count" == 4 && "$load" == loaded && "$main" =~ ^(0|[1-9][0-9]{0,2})$ ]] || return 1
 (( main<=255 )) || return 1
 case "$active" in
  inactive) [[ "$result" == success && "$main" == 0 ]] || return 1;UNIT_READY=1 ;;
  failed)
   [[ "$WAIT_MODE" == terminal ]] || { echo 'Explicit preceding unit failed; success wait aborted' >&2;return 1; }
   case "$result" in
    exit-code|signal|core-dump) (( main>0 )) || return 1 ;;
    timeout|watchdog|oom-kill|resources|protocol|start-limit-hit) ;;
    *) echo 'Unknown or inconsistent terminal predecessor result' >&2;return 1 ;;
   esac
   UNIT_READY=1 ;;
  active|activating|deactivating|reloading) UNIT_READY=0 ;;
  *) echo 'Explicit preceding unit did not complete successfully' >&2;return 1 ;;
 esac
 UNIT_RESULT="$result";UNIT_MAIN="$main";UNIT_ACTIVE="$active"
}
BEFORE='' LOCK_BEFORE='' LOCK_OPEN=0
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$BEFORE" ]];then
  AFTER="$(integrity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Frozen queued frontend source changed' >&2;STATUS=1;fi
 fi
 if [[ -n "$LOCK_BEFORE" ]];then
  if (( LOCK_OPEN ));then AFTER_LOCK="$(lock_identity fd)";else AFTER_LOCK="$(lock_identity)";fi
  CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER_LOCK" != "$LOCK_BEFORE" ]];then echo 'Original cooperative lock changed' >&2;STATUS=1;fi
 fi
 if (( LOCK_OPEN ));then exec 9>&-;fi
 if [[ "$STATUS" != 0 ]];then phase fail;fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
phase preflight
BEFORE="$(integrity)";targets_absent
if [[ "$WAIT_MODE" != lock ]];then
 LOAD="$(systemctl show "$WAIT_FOR" --property=LoadState --value)"
 [[ "$LOAD" == loaded ]] || { echo 'Explicit preceding unit is not loaded' >&2;exit 1; }
fi
LOCK_BEFORE="$(lock_identity)"
START=$SECONDS;UNIT_READY=0
if [[ "$WAIT_MODE" != lock ]];then
 if [[ "$WAIT_MODE" == terminal ]];then phase waiting_for_terminal_predecessor;else phase waiting_for_successful_predecessor;fi
 while :;do
  unit_state
  (( UNIT_READY )) && break
  REMAINING=$((43200-(SECONDS-START)))
  (( REMAINING>0 )) || { echo 'Queued frontend wait budget exceeded' >&2;exit 1; }
  DELAY=15;(( REMAINING>=DELAY )) || DELAY=$REMAINING
  sleep "$DELAY"
 done
fi
[[ "$(integrity)" == "$BEFORE" && "$(lock_identity)" == "$LOCK_BEFORE" ]] || exit 1
targets_absent
# Readonly opening neither creates nor truncates the independently checked lock.
exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
REMAINING=$((43200-(SECONDS-START)))
(( REMAINING>0 )) || exit 1
phase waiting_for_gpu_lock
flock --timeout "$REMAINING" 9
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" && "$(integrity)" == "$BEFORE" ]] || exit 1
targets_absent
if [[ "$WAIT_MODE" != lock ]];then unit_state;(( UNIT_READY )) || exit 1;fi
if [[ "$WAIT_MODE" == terminal ]];then
 printf '{"stage":"track1_frontends_queued","phase":"predecessor_terminal","episode_index":%s,"predecessor_unit":"%s","predecessor_state":"%s","predecessor_result":"%s","exec_main_status":%s}\n' \
  "$EPISODE" "$WAIT_FOR" "$UNIT_ACTIVE" "$UNIT_RESULT" "$UNIT_MAIN"
fi
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU compute applications present after lock; queued frontend aborted' >&2;exit 1; }
# The original child has its own nonblocking lock+GPU-idle preflight. A handoff
# race fails there, never runs unlocked. No fabricated child path or lock mode.
flock --unlock 9;exec 9>&-;LOCK_OPEN=0
phase running_original_frontends
bash "$CODE/infra/run_track1_frontends.sh" --episode "$EPISODE" --actor-policy fixed_all16
phase child_complete
