#!/usr/bin/env bash
# Azure VM01 CPU-only archive; no acquisition, transport or image/model runtime.
# Source closure: /infra/frontend_asset_archive.py
# Current inventory helpers are retained as provenance-only source; only the
# independent old snapshot named by the frozen pin config is ever imported.
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_asset_archive/code" && "$(uname -s)" == Linux ]] || exit 2
CONFIG_BYTES=1155
CONFIG_SHA256=312a10621c02c02449d0786649eac137abc6d9d314236f851848942fb4bff72e
OLD_REV=40fdc2780076d3f83632dc20c7926a3a6763e9d7
OUT="$ROOT/results/frontend-asset-archive-$REV"
integrity() {
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - \
 "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" "$OLD_REV" "$CONFIG_BYTES" "$CONFIG_SHA256" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,json,re,stat,sys
root,code,rev,entry,old,config_bytes,config_sha=sys.argv[1:];root,code,entry=map(Path,(root,code,entry))
def canonical(path):
 if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical original archive source required')
 return path
def state(path):
 s=path.lstat();return(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
def regular(path,maximum,readonly=True):
 canonical(path);before=state(path)
 if not stat.S_ISREG(before[2])or before[6]!=1 or not 1<=before[3]<=maximum or readonly and before[2]&0o222:raise ValueError('Bounded single-link readonly source required')
 raw=path.read_bytes()
 if state(path)!=before:raise ValueError('Source changed during integrity check')
 return raw
def parse(raw):
 def pairs(rows):
  result={}
  for k,v in rows:
   if k in result:raise ValueError('Duplicate frozen config key')
   result[k]=v
  return result
 def bad(_):raise ValueError('Nonfinite frozen config')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
canonical(root);canonical(code);canonical(entry)
if not root.is_dir()or not code.is_dir()or entry!=code/'infra/run_frontend_asset_archive.sh':raise ValueError('Exact immutable archive source required')
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 raw=regular(code.parent/name,100,False)
 if(name=='revision'and raw!=(rev+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):raise ValueError('Exact original dispatch markers required')
 digest.update(name.encode()+b'\0'+raw)
required=('infra/frontend_asset_archive.py','infra/run_frontend_asset_archive.sh','infra/frontend_replica_inventory.py','infra/run_frontend_replica_inventory.sh','configs/frontend_replica_inventory_pins.json')
if any(not(code/name).is_file()for name in required):raise ValueError('Complete original and current archive source closure required')
for path in(code,*sorted(code.rglob('*'))):
 canonical(path);mode=path.lstat().st_mode
 if mode&0o222 or not(stat.S_ISDIR(mode)or stat.S_ISREG(mode)):raise ValueError('Complete archive source must remain readonly and regular')
 if stat.S_ISREG(mode):
  raw=regular(path,2_000_000);digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(raw).digest())
raw=regular(code/'configs/frontend_replica_inventory_pins.json',2_000_000)
if not re.fullmatch(r'[0-9a-f]{64}',config_sha)or int(config_bytes)!=len(raw)or hashlib.sha256(raw).hexdigest()!=config_sha:raise ValueError('Independent frozen inventory config pin differs before JSON')
config=parse(raw)
if type(config)is not dict or set(config)!={'schema','inventory_report','inventory_code','inventory_source_pins'}or config['schema']!='world_reward.frontend_replica_inventory.pins.v1':raise ValueError('Exact frozen inventory config schema required')
if config['inventory_code']!='jobs/'+old+'/run_frontend_replica_inventory/code':raise ValueError('Only original audited inventory snapshot allowed')
binding=config['inventory_report']
if type(binding)is not dict or set(binding)!={'pathrelative','pin'}or binding['pathrelative']!='results/frontend-replica-inventory-'+old+'/report.json':raise ValueError('Only exact original PASS receipt path allowed')
pin=binding['pin']
if(type(pin)is not dict or set(pin)!={'bytes','sha256','producer_revision','entries_sha256','files','links','total_bytes'}or pin['producer_revision']!=old
 or type(pin['bytes'])is not int or pin['bytes']!=149203 or pin['sha256']!='95d09454133e481324380e9253d235ba3bbdd3657e1fac25f97e8f7c4f1741f1'
 or type(pin['entries_sha256'])is not str or not re.fullmatch(r'[0-9a-f]{64}',pin['entries_sha256'])
 or any(type(pin[k])is not int for k in('files','links','total_bytes'))or(pin['files'],pin['links'],pin['total_bytes'])!=(448,3,19910803804)):
 raise ValueError('Exact independently audited receipt cohort pin required')
helperpins=config['inventory_source_pins'];names=('infra/frontend_replica_inventory.py','infra/run_frontend_replica_inventory.sh')
if type(helperpins)is not dict or set(helperpins)!=set(names):raise ValueError('Both independently pinned old source helpers required')
oldcode=canonical(root/config['inventory_code'])
for name,pin in helperpins.items():
 if(type(pin)is not dict or set(pin)!={'bytes','sha256','git_blob_sha1'}or type(pin['bytes'])is not int or not 1<=pin['bytes']<=1_000_000
  or type(pin['sha256'])is not str or not re.fullmatch(r'[0-9a-f]{64}',pin['sha256'])or type(pin['git_blob_sha1'])is not str or not re.fullmatch(r'[0-9a-f]{40}',pin['git_blob_sha1'])):raise ValueError('Independent original helper byte/SHA/blob pins required')
 raw=regular(oldcode/name,1_000_000)
 if(len(raw),hashlib.sha256(raw).hexdigest(),hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())!=(pin['bytes'],pin['sha256'],pin['git_blob_sha1']):raise ValueError('Pinned old helper changed')
 digest.update(name.encode()+b'\0'+hashlib.sha256(raw).digest())
receipt=canonical(root/binding['pathrelative']);raw=regular(receipt,2_000_000)
if receipt.stat().st_mode&0o777!=0o400 or len(raw)!=binding['pin']['bytes']or hashlib.sha256(raw).hexdigest()!=binding['pin']['sha256']:raise ValueError('Original readonly receipt byte pin differs')
digest.update(hashlib.sha256(raw).digest())
print(digest.hexdigest())
PYINTEGRITY
}
BEFORE=""
finish() {
 STATUS=$?; trap - EXIT INT TERM; set +e
 if [[ -n "$BEFORE" ]];then
  AFTER="$(integrity)"; CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Frozen archive source/config/original receipt changed' >&2; STATUS=1;fi
 fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
BEFORE="$(integrity)"
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$OUT" "$REV" <<'PYRESERVE'
from pathlib import Path
import sys
root,out=map(Path,sys.argv[1:3]);rev=sys.argv[3]
if(out!=root/'results'/('frontend-asset-archive-'+rev)or not out.parent.is_dir()
 or out.exists()or out.is_symlink()or out.resolve()!=out or any(p.is_symlink()for p in(out,*out.parents))):raise ValueError('Absent canonical owned archive result required; no overwrite/resume')
out.mkdir(mode=0o700)
if out.stat().st_mode&0o777!=0o700 or any(out.iterdir()):raise ValueError('Exclusive empty private archive directory required')
PYRESERVE
umask 077
ulimit -v 4194304
# Only bounded stdlib host CPU work. stderr is retained by the launcher; no
# credentials/environment/config/asset lists are emitted by this wrapper.
timeout --signal=TERM --kill-after=10s 1800s nice -n 15 ionice -c 3 \
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - \
 "$ROOT" "$CODE" "$REV" "$OUT" "$CONFIG_BYTES" "$CONFIG_SHA256" "$BEFORE" <<'PYCONTROL'
from pathlib import Path
import hashlib,json,os,runpy,stat,sys,time
root,code,rev,out,config_bytes,config_sha,before=sys.argv[1:];root,code,out=map(Path,(root,code,out));started=time.perf_counter()
def canonical(path):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical immutable archive control paths required')
 return path
def identity(path,maximum=2_000_000):
 canonical(path);s=path.lstat()
 if not stat.S_ISREG(s.st_mode)or s.st_nlink!=1 or not 1<=s.st_size<=maximum:raise ValueError('Bounded single-link control source required')
 raw=path.read_bytes()
 if path.lstat()!=s:raise ValueError('Control source changed while reading')
 return raw,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
raw,config_id=identity(code/'configs/frontend_replica_inventory_pins.json')
if(config_id['bytes']!=int(config_bytes)or config_id['sha256']!=config_sha):raise ValueError('Frozen config changed before isolated control')
config=json.loads(raw)
def fingerprint():
 digest=hashlib.sha256()
 for name in('revision','source-sha256'):
  raw,_=identity(code.parent/name,100);digest.update(name.encode()+b'\0'+raw)
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);mode=path.lstat().st_mode
  if mode&0o222 or not(stat.S_ISDIR(mode)or stat.S_ISREG(mode)):raise ValueError('Complete control source changed')
  if stat.S_ISREG(mode):
   raw,_=identity(path);digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(raw).digest())
 for name in config['inventory_source_pins']:
  raw,actual=identity(root/config['inventory_code']/name,1_000_000)
  if actual!=config['inventory_source_pins'][name]or(root/config['inventory_code']/name).stat().st_mode&0o222:raise ValueError('Pinned old helper changed')
  digest.update(name.encode()+b'\0'+hashlib.sha256(raw).digest())
 raw,_=identity(root/config['inventory_report']['pathrelative']);digest.update(hashlib.sha256(raw).digest())
 return digest.hexdigest()
if fingerprint()!=before:raise ValueError('Frozen source/config/provenance changed before control')
canonical(out)
if not out.is_dir()or out.stat().st_mode&0o777!=0o700 or any(out.iterdir()):raise ValueError('Only newly reserved empty private output allowed')
owned=(out.stat().st_dev,out.stat().st_ino)
source_id={}
for name in('infra/frontend_asset_archive.py','infra/run_frontend_asset_archive.sh'):
 _,source_id[name]=identity(code/name,1_000_000)
exporter={k:source_id['infra/frontend_asset_archive.py'][k]for k in('bytes','sha256')}
api=runpy.run_path(str(code/'infra/frontend_asset_archive.py'))
report=dict(schema='world_reward.frontend_asset_archive.run.v1',stage='frontend_asset_archive',status='fail',producer_revision=rev,
 source_helpers=source_id,inventory_config_identity={k:config_id[k]for k in('bytes','sha256')},budget_seconds=1800,
 license_eligibility_verified=False,training_overlap_verified=False,replica_ready=False,CUDA_verified=False,
 operational_same_run_archive_verification_is_not_independent_external_validation=True)
error=None
try:
 binding=config['inventory_report'];oldcode=root/config['inventory_code']
 receipt=api['build_archive'](root,root/binding['pathrelative'],oldcode,binding['pin'],config['inventory_source_pins'],out/'archive.tar',exporter)
 check=api['verify_archive'](out/'archive.tar',receipt['archive_identity'],receipt['manifest_identity'],oldcode,config['inventory_source_pins'])
 report.update(status='pass',archive=receipt,operational_verification=check)
except Exception as caught:
 error=caught;report.update(error_type=type(caught).__name__,error='Archive contract failed closed; inspect owned launcher diagnostic')
finally:
 try:
  if canonical(out)!=out or(out.stat().st_dev,out.stat().st_ino)!=owned or out.stat().st_mode&0o777!=0o700:raise ValueError('Owned result directory changed')
  if {n:identity(code/n,1_000_000)[1]for n in source_id}!=source_id or identity(code/'configs/frontend_replica_inventory_pins.json')[1]!=config_id:raise ValueError('Current source or config changed')
  api['_source'](root/config['inventory_code'],config['inventory_source_pins'])
  api['_read_pinned'](root/config['inventory_report']['pathrelative'],{k:config['inventory_report']['pin'][k]for k in('bytes','sha256')})
  if fingerprint()!=before:raise ValueError('Complete source/markers/provenance changed before receipt')
 except Exception as caught:
  error=caught;report.update(status='fail',error_type=type(caught).__name__,error='Immutable archive source/provenance changed')
 report['elapsed_seconds']=time.perf_counter()-started
 payload=json.dumps(report,sort_keys=True,allow_nan=False).encode()+b'\n'
 if len(payload)>12000:raise ValueError('Tiny archive receipt exceeded bound')
 # Never replace a receipt, symlink or unknown result. Only this owned output
 # may be used, and all readable checkpoint bytes are already0400.
 if canonical(out)!=out or(out.stat().st_dev,out.stat().st_ino)!=owned:raise ValueError('Owned report parent changed')
 with os.fdopen(os.open(out/'report.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400),'wb')as stream:
  stream.write(payload);stream.flush();os.fsync(stream.fileno())
 summary=dict(stage=report['stage'],status=report['status'],elapsed_seconds=report['elapsed_seconds'],replica_ready=False)
 if report['status']=='pass':summary.update(files=receipt['files'],links=receipt['links'],total_bytes=receipt['total_bytes'],archive_identity=receipt['archive_identity'],manifest_identity=receipt['manifest_identity'])
 else:summary.update(error_type=report.get('error_type','unknown'),error=report.get('error','Archive precondition failed'))
 print(json.dumps(summary,sort_keys=True,allow_nan=False),flush=True)
if error is not None:raise RuntimeError('Asset archive failed closed; immutable owned receipt retained')from None
PYCONTROL
