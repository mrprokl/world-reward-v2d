#!/usr/bin/env bash
# One pinned asset archive over a preauthenticated private Azure SSH peer only.
set -euo pipefail
[[ $# == 2 && "$1" == --server-public-key ]] || exit 2
SERVER_KEY="$2"
[[ "$SERVER_KEY" != *$'\n'* && "$SERVER_KEY" != *$'\r'* ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_peer_send/code" && "$(uname -s)" == Linux \
 && "$(hostname -s)" == scenesmith-ncc-h100-01 ]] || exit 2
OLD=c84558c673ef8af299dfa187eeb8fde8ce9fc8e1
ARCHIVE="$ROOT/results/frontend-asset-archive-$OLD/archive.tar"
KEY="$ROOT/transfer/frontend-peer-client-v1/client_ed25519"
OUT="$ROOT/results/frontend-peer-send-$REV"
integrity() {
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "${BASH_SOURCE[0]}" "$KEY" <<'PYINTEGRITY'
from pathlib import Path
import hashlib,re,stat,sys
root,code,rev,entry,key=sys.argv[1:];root,code,entry,key=map(Path,(root,code,entry,key))
def canonical(path):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical sender source/key namespace required')
def state(path):
 s=path.lstat();return(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_uid)
def source(path):
 canonical(path);before=state(path)
 if not stat.S_ISREG(before[2])or before[6]!=1 or not 1<=before[3]<=2_000_000 or before[2]&0o222:raise ValueError('Frozen sender source must remain readonly regular')
 raw=path.read_bytes()
 if state(path)!=before:raise ValueError('Sender source changed during hashing')
 return raw
for path in(root,code,entry,key,key.parent):canonical(path)
if not root.is_dir()or not code.is_dir()or entry!=code/'infra/run_frontend_peer_send.sh':raise ValueError('Actual sender snapshot required')
for path,mode,kind in((key.parent,0o700,stat.S_ISDIR),(key,0o600,stat.S_ISREG)):
 s=state(path)
 if not kind(s[2])or s[2]&0o777!=mode or s[7]!=0 or(kind==stat.S_ISREG and(s[6]!=1 or not 1<=s[3]<=10000)):raise ValueError('Existing private root-owned client key required')
# Never open or hash private key bytes. Public .pub records are separately
# checked by the control process and may legitimately be printed by its owner.
digest=hashlib.sha256()
for name in('revision','source-sha256'):
 path=code.parent/name;canonical(path);raw=path.read_bytes()
 if not stat.S_ISREG(path.lstat().st_mode)or path.lstat().st_nlink!=1 or(name=='revision'and raw!=(rev+'\n').encode()or name=='source-sha256'and not re.fullmatch(b'[0-9a-f]{64}\n',raw)):raise ValueError('Exact original dispatch markers required')
 digest.update(name.encode()+b'\0'+raw)
for path in(code,*sorted(code.rglob('*'))):
 canonical(path);mode=path.lstat().st_mode
 if mode&0o222 or not(stat.S_ISREG(mode)or stat.S_ISDIR(mode)):raise ValueError('Complete sender closure must be readonly and regular')
 if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+hashlib.sha256(source(path)).digest())
print(digest.hexdigest())
PYINTEGRITY
}
BEFORE=''
finish() {
 STATUS=$?;trap - EXIT INT TERM;set +e
 if [[ -n "$BEFORE" ]];then AFTER="$(integrity)";CHECK=$?
  if [[ "$CHECK" != 0 || "$AFTER" != "$BEFORE" ]];then echo 'Sender immutable source/key metadata changed' >&2;STATUS=1;fi
 fi
 exit "$STATUS"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
BEFORE="$(integrity)"
umask 077
ulimit -v 2097152
timeout --signal=TERM --kill-after=10s 1800s nice -n 15 ionice -c 3 \
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$ARCHIVE" "$KEY" "$OUT" "$SERVER_KEY" "$BEFORE" <<'PYCONTROL'
from pathlib import Path
import base64,hashlib,json,os,re,stat,struct,subprocess,sys,time
root,code,rev,archive,key,out,serverkey,before=sys.argv[1:];root,code,archive,key,out=map(Path,(root,code,archive,key,out));started=time.perf_counter()
def canonical(path):
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):raise ValueError('Canonical sender artifact required')
def state(path):
 s=path.lstat();return(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink,s.st_uid)
def identity(path,size=None,private=False):
 canonical(path);s=state(path)
 if not stat.S_ISREG(s[2])or s[6]!=1 or s[3]<1 or size is not None and s[3]!=size or private and s[2]&0o777!=0o400:raise ValueError('Pinned regular private sender artifact required')
 digest=hashlib.sha256()
 with path.open('rb')as h:
  for block in iter(lambda:h.read(4*1024*1024),b''):digest.update(block)
 if state(path)!=s:raise ValueError('Sender artifact changed while hashing')
 return dict(bytes=s[3],sha256=digest.hexdigest())
def public_key(value):
 if not re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: [^\x00-\x1f\x7f]+)?',value):raise ValueError('One independently supplied Ed25519 host public key required')
 encoded=value.split(' ')[1]
 raw=base64.b64decode(encoded,validate=True)
 if raw!=struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+raw[-32:]or len(raw)!=51:raise ValueError('Exact Ed25519 public-key wire format required')
 return 'ssh-ed25519 '+encoded
serverkey=public_key(serverkey)
def fingerprint():
 digest=hashlib.sha256()
 for name in('revision','source-sha256'):
  path=code.parent/name;canonical(path);raw=path.read_bytes();digest.update(name.encode()+b'\0'+raw)
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);s=state(path)
  if s[2]&0o222 or not(stat.S_ISREG(s[2])or stat.S_ISDIR(s[2])):raise ValueError('Sender complete source changed')
  if stat.S_ISREG(s[2]):digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(identity(path)['sha256']))
 return digest.hexdigest()
if fingerprint()!=before:raise ValueError('Frozen sender source/markers changed before stream')
archivepin=dict(bytes=19911464960,sha256='5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342')
reportpath=archive.with_name('report.json');reportpin=dict(bytes=3191,sha256='4c552c18ccb35e76efe40754a5c45d92703506bebc1f8bef2c355075543af0bf')
if identity(reportpath,3191,True)!=reportpin:raise ValueError('Original archive producer receipt pin differs before JSON')
producer=json.loads(reportpath.read_bytes())
if producer.get('status')!='pass'or producer.get('stage')!='frontend_asset_archive'or producer.get('producer_revision')!='c84558c673ef8af299dfa187eeb8fde8ce9fc8e1'or producer.get('archive',{}).get('archive_identity')!=archivepin:raise ValueError('Exact original published archive producer required')
if identity(archive,archivepin['bytes'],True)!=archivepin:raise ValueError('Independent source archive byte identity differs')
canonical(key);key_before=state(key);dir_before=state(key.parent)
if(key_before[2]&0o777!=0o600 or key_before[7]!=0 or key_before[6]!=1 or not stat.S_ISREG(key_before[2])or dir_before[2]&0o777!=0o700 or dir_before[7]!=0):raise ValueError('Original root-owned private key metadata required')
pub=key.with_suffix('.pub');canonical(pub)
if not stat.S_ISREG(pub.lstat().st_mode)or pub.lstat().st_nlink!=1 or pub.lstat().st_size>1000 or pub.stat().st_mode&0o777!=0o400:raise ValueError('Original client public key record required')
public_key(pub.read_text().strip())
pub_identity=identity(pub,private=True)
canonical(out)
if not out.parent.is_dir()or out.exists()or out.is_symlink():raise ValueError('Absent owned sender result required; no overwrite/resume')
out.mkdir(mode=0o700);owned=state(out)[:2]
def owned_dir():
 canonical(out)
 if state(out)[:2]!=owned or out.stat().st_mode&0o777!=0o700:raise ValueError('Owned sender result changed')
def write(name,raw):
 owned_dir();path=out/name
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400),'wb')as h:h.write(raw);h.flush();os.fsync(h.fileno())
 return path
known=write('known_hosts',('[10.0.0.9]:2222 '+serverkey+'\n').encode())
sourcepin=identity(code/'infra/run_frontend_peer_send.sh')
report=dict(schema='world_reward.frontend_peer_send.v1',stage='frontend_peer_send',status='fail',producer_revision=rev,
 source_helper=sourcepin,source_archive=archivepin,source_archive_receipt=reportpin,archive_sent=False,
 replica_ready=False,license_eligibility_verified=False,training_overlap_verified=False,private_key_bytes_recorded=False)
error=None
try:
 owned_dir();output=out/'receiver-summary.json'
 args=['ssh','-F','/dev/null','-T','-p','2222','-i',str(key),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
  '-o','UserKnownHostsFile='+str(known),'-o','GlobalKnownHostsFile=/dev/null','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3',
  'root@10.0.0.9','world-reward-frontend-assets-v1']
 environment=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',LANG='C.UTF-8')
 with archive.open('rb')as stream,os.fdopen(os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400),'wb')as response:
  result=subprocess.run(args,stdin=stream,stdout=response,stderr=subprocess.DEVNULL,env=environment,timeout=1600,check=False)
  response.flush();os.fsync(response.fileno())
 if result.returncode:raise ValueError('Private SSH sender returned failure')
 if not 1<=output.stat().st_size<4000:raise ValueError('Bounded receiver summary required')
 raw=output.read_bytes()
 def pairs(rows):
  value={}
  for k,v in rows:
   if k in value:raise ValueError('Duplicate receiver summary key')
   value[k]=v
  return value
 summary=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError('Nonfinite receiver summary')))
 if(type(summary)is not dict or summary!=dict(stage='frontend_peer_receive',status='pass',bytes_received=archivepin['bytes'],receipt_written=True,replica_ready=False)
  or type(summary['bytes_received'])is not int or summary['receipt_written']is not True or summary['replica_ready']is not False):raise ValueError('Exact complete receiver verdict required')
 if identity(archive,archivepin['bytes'],True)!=archivepin or identity(reportpath,3191,True)!=reportpin or state(key)!=key_before or state(key.parent)!=dir_before or identity(pub,private=True)!=pub_identity or identity(code/'infra/run_frontend_peer_send.sh')!=sourcepin or fingerprint()!=before:raise ValueError('Source archive/provenance/key metadata changed after stream')
 owned_dir();report.update(status='pass',archive_sent=True,receiver_summary_identity=identity(output,private=True),bytes_sent=archivepin['bytes'])
except Exception as caught:
 error=caught;report.update(error_type=type(caught).__name__,error='Private pinned asset transport failed closed; owned partial evidence retained')
finally:
 report['elapsed_seconds']=time.perf_counter()-started
 write('report.json',json.dumps(report,sort_keys=True,allow_nan=False).encode()+b'\n')
 print(json.dumps({k:report[k]for k in('stage','status','elapsed_seconds','replica_ready')},sort_keys=True),flush=True)
if error is not None:raise RuntimeError('Private asset sender failed; owned receipt retained')from None
PYCONTROL
