#!/usr/bin/env bash
# Scheduling only; unchanged children own all prediction and numerical gates.
# Source closure: /infra/run_cari_shared_prepare.sh /infra/cari_shared_prepare.py
# /infra/run_cari_full_forward.sh /infra/cari_full_forward.py
# /infra/run_cari_full_refine.sh /infra/cari_full_refine.py
# /infra/run_cari_full_export.sh /infra/cari_full_export.py
set -euo pipefail
STAGE='' EPISODE='' WAIT_FOR='' stage_seen=0 episode_seen=0 wait_seen=0
while (( $# ));do
 case "$1" in
  --stage)
   if (( stage_seen || $# < 2 ));then exit 2;fi
   case "$2" in prepare|forward|refined|export) STAGE="$2" ;; *) exit 2 ;; esac
   stage_seen=1;shift 2 ;;
  --episode)
   if (( episode_seen || $# < 2 )) || [[ ! "$2" =~ ^(0|[1-9]|[12][0-9])$ ]];then exit 2;fi
   EPISODE="$2";episode_seen=1;shift 2 ;;
  --wait-for)
   if (( wait_seen || $# < 2 )) || [[ ! "$2" =~ ^world-reward-[a-z0-9][a-z0-9-]{0,80}(\.service)?$ ]];then exit 2;fi
   WAIT_FOR="${2%.service}.service";wait_seen=1;shift 2 ;;
  *) exit 2 ;;
 esac
done
(( stage_seen && episode_seen && wait_seen )) || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_cari_shared_stage_queued/code" && "$(uname -s)" == Linux ]] || exit 2
case "$STAGE" in
 prepare) CHILD=infra/run_cari_shared_prepare.sh ;;
 forward) CHILD=infra/run_cari_full_forward.sh ;;
 refined) CHILD=infra/run_cari_full_refine.sh ;;
 export) CHILD=infra/run_cari_full_export.sh ;;
esac
printf -v PADDED '%06d' "$EPISODE"
OUT="$ROOT/outputs/episode_$PADDED/cari_shared_${STAGE}_v1"
LOCK="$ROOT/jobs/.world-reward-h100.lock"
phase() {
 printf '{"stage":"cari_shared_stage_queued","phase":"%s","native_stage":"%s","episode_index":%s,"timestamp_utc":"%s"}\n' \
  "$1" "$STAGE" "$EPISODE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
integrity() {
 env PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$OUT" "$STAGE" "$EPISODE" "${BASH_SOURCE[0]}" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,json,re,stat,sys
root,code,rev,out,stage,episode,entry=sys.argv[1:];root,code,out,entry=map(Path,(root,code,out,entry));episode=int(episode)
def canonical(path):
 if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical queued stage paths required')
def raw(path,maximum=2_000_000,readonly=True):
 canonical(path);before=path.lstat()
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<=before.st_size<=maximum or readonly and before.st_mode&0o222:raise ValueError('Bounded original regular readonly source required')
 value=path.read_bytes();after=path.lstat()
 if any(getattr(after,k)!=getattr(before,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')):raise ValueError('Queue source changed during hashing')
 return value
for path in(root,code,out,entry):canonical(path)
if not root.is_dir() or not code.is_dir() or not out.parent.is_dir() or entry!=code/'infra/run_cari_shared_stage_queued.sh':raise ValueError('Actual queued wrapper and existing episode required')
children=('run_cari_shared_prepare.sh','run_cari_full_forward.sh','run_cari_full_refine.sh','run_cari_full_export.sh')
drivers=('cari_shared_prepare.py','cari_full_forward.py','cari_full_refine.py','cari_full_export.py')
if any(not(code/'infra'/name).is_file()for name in(*children,*drivers)):raise ValueError('Complete unchanged native stage children required')
def pairs(rows):
 result={}
 for key,value in rows:
  if key in result:raise ValueError('Duplicate frozen pin key forbidden')
  result[key]=value
 return result
roles=('input','prepare','forward','refined')[:('prepare','forward','refined','export').index(stage)+1];spec=None
for role in roles:
 suffix='input'if role=='input'else 'shared_'+role
 path=code/f'configs/cari_clip_{episode:06d}_{suffix}_pins.json'
 pin=json.loads(raw(path),object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite frozen pin forbidden')))
 expected='world-reward-cari-clip-input-pins-v1'if role=='input'else f'world-reward-cari-shared-{role}-pins-v1'
 keys={'schema','clip_spec','input_report','source_files'}if role=='input'else {'schema','clip_spec',role,role+'_files'}
 if type(pin)is not dict or set(pin)!=keys or pin['schema']!=expected:raise ValueError('Exact required committed stage pin schema required')
 value=pin['clip_spec']
 if type(value)is not dict or set(value)!={'episode_index','total_frames','camera_name','height','width'} or type(value['episode_index'])is not int or value['episode_index']!=episode or type(value['total_frames'])is not int or value['total_frames']<96 or any(type(value[k])is not int or value[k]<2 for k in('height','width')) or type(value['camera_name'])is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',value['camera_name']):raise ValueError('Full original selected clip spec required')
 if spec is not None and value!=spec:raise ValueError('Required stage pins must identify the same original clip')
 spec=value;receipt=pin['input_report'if role=='input'else role]
 if type(receipt)is not dict or set(receipt)!={'sha256','bytes','producer_revision','script_sha256'} or type(receipt['bytes'])is not int or receipt['bytes']<=0 or any(type(receipt[k])is not str or not re.fullmatch('[0-9a-f]{'+str(n)+'}',receipt[k])for k,n in(('sha256',64),('producer_revision',40),('script_sha256',64))):raise ValueError('Independently committed producer identity required')
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 value=raw(code.parent/name,100,False)
 if name=='revision'and value!=(rev+'\n').encode() or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',value):raise ValueError('Exact original stage dispatch markers required')
 digest.update(name.encode()+b'\0'+value)
for path in(code,*sorted(code.rglob('*'))):
 canonical(path);mode=path.lstat().st_mode
 if mode&0o222 or not(stat.S_ISDIR(mode)or stat.S_ISREG(mode)):raise ValueError('Complete stage source must remain readonly and regular')
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
target_absent() { [[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'Frozen stage target exists; no overwrite/resume' >&2;return 1; }; }
BEFORE='' LOCK_BEFORE='' LOCK_OPEN=0
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$BEFORE" ]];then
  AFTER="$(integrity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Frozen queued stage source/pins changed' >&2;STATUS=1;fi
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
BEFORE="$(integrity)";target_absent
# This different episode needs the released GPU lock, not CPU assembly success.
LOAD="$(systemctl show "$WAIT_FOR" --property=LoadState --value)"
[[ "$LOAD" == loaded ]] || { echo 'Explicit preceding frontend unit is not loaded' >&2;exit 1; }
LOCK_BEFORE="$(lock_identity)"
exec 9<"$LOCK";LOCK_OPEN=1
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" ]] || exit 1
phase waiting_for_gpu_lock
flock --timeout 43200 9
[[ "$(lock_identity fd)" == "$LOCK_BEFORE" && "$(integrity)" == "$BEFORE" ]] || exit 1
target_absent
APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${APPS//[[:space:]]/}" ]] || { echo 'GPU compute applications present after lock; queued stage aborted' >&2;exit 1; }
phase running_native_child
# FD9 remains locked in this parent throughout the synchronous unchanged child.
bash "$CODE/$CHILD" --episode "$EPISODE"
phase child_complete
