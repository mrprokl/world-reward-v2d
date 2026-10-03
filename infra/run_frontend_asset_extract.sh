#!/usr/bin/env bash
# Private VM02 extraction only: no promotion, network, model or image runtime.
# Source closure: /infra/frontend_asset_extract.py /infra/frontend_asset_archive.py
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}"; CODE="${WR_CODE:?}"; REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_frontend_asset_extract/code" && "$(uname -s)" == Linux ]] || exit 2
OUT="$ROOT/results/frontend-asset-extract-$REV"
env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$OUT" <<'PYPREFLIGHT'
from pathlib import Path
import stat,sys
root,code,out=map(Path,sys.argv[1:])
for p in(root,code,out):
 if not p.is_absolute() or p.resolve()!=p or any(x.is_symlink()for x in(p,*p.parents)):raise ValueError('Canonical extractor source/result paths required')
if not root.is_dir() or not code.is_dir() or not out.parent.is_dir() or out.exists() or out.is_symlink():raise ValueError('Absent isolated extraction result required')
for p in(code,*code.rglob('*')):
 s=p.lstat()
 if p.is_symlink() or s.st_mode&0o222 or not(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):raise ValueError('Immutable complete extractor source required')
PYPREFLIGHT
umask 077; mkdir -m 700 "$OUT"
ulimit -v 4194304
timeout --signal=TERM --kill-after=10s 1200s nice -n 15 ionice -c 3 \
 env -i PATH=/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 python3 -I -B - "$ROOT" "$CODE" "$REV" "$OUT" <<'PY'
from pathlib import Path
import hashlib,json,os,re,stat,sys,time,types,subprocess
root,code,revision,out=map(Path,sys.argv[1:]);rev=str(revision);start=time.perf_counter()
for p in(root,code,out):
 if not p.is_absolute() or p.resolve()!=p or any(x.is_symlink()for x in(p,*p.parents)):raise ValueError('Canonical isolated extractor paths required')
if os.geteuid()!=0 or code!=root/'jobs'/rev/'run_frontend_asset_extract/code':raise ValueError('Exact root-owned VM02 extractor required')
def identity(p,maximum=2_000_000):
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or not 1<=s.st_size<=maximum:raise ValueError('Bounded regular control file required')
 b=p.read_bytes();a=p.lstat()
 if(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=(a.st_dev,a.st_ino,a.st_mode,a.st_size,a.st_mtime_ns,a.st_ctime_ns):raise ValueError('Control file changed')
 return b,dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def closure():
 d=hashlib.sha256()
 for n in('revision','source-sha256'):
  b,_=identity(code.parent/n,100)
  if(n=='revision' and b!=(rev+'\n').encode() or n=='source-sha256' and not re.fullmatch(b'[0-9a-f]{64}\n',b)):raise ValueError('Exact dispatch markers required')
  d.update(n.encode()+b'\0'+b)
 for p in(code,*sorted(code.rglob('*'))):
  s=p.lstat()
  if p.is_symlink() or s.st_mode&0o222 or not(stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):raise ValueError('Complete immutable source closure required')
  if stat.S_ISREG(s.st_mode):d.update(str(p.relative_to(code)).encode()+b'\0'+identity(p)[1]['sha256'].encode())
 return d.hexdigest()
before=closure();disk=Path('/srv/world-reward-data');incoming=disk/'frontend-assets-v1';dest=disk/('frontend-assets-extracted-'+rev)
u=subprocess.check_output(['/usr/bin/findmnt','-n','-o','UUID','--target',str(disk)],text=True).strip()
if u!='24df126a-5f5f-41d8-801c-9ddaa7a582d8' or os.statvfs(disk).f_bavail*os.statvfs(disk).f_frsize<21_000_000_000:raise ValueError('Owned data disk and enough space required')
for p in(disk,incoming):
 if p.resolve()!=p or any(x.is_symlink()for x in(p,*p.parents)) or not p.is_dir():raise ValueError('Canonical owned data disk/transport required')
if incoming.stat().st_mode&0o777!=0o700 or incoming.stat().st_uid!=0 or {p.name for p in incoming.iterdir()}!={'archive.tar','report.json'}:raise ValueError('Exact private completed transport namespace required')
out_owned=(out.stat().st_dev,out.stat().st_ino)
for p in(incoming/'archive.tar',incoming/'report.json'):
 s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_uid!=0 or s.st_mode&0o777!=0o400:raise ValueError('Immutable private transport artifacts required')
b,_=identity(incoming/'report.json',4000);r=json.loads(b)
archive_pin={'bytes':19911464960,'sha256':'5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342'}
manifest_pin={'bytes':131666,'sha256':'16a6b5314b205c5ad526df2340ab99e46873fd5d62b1c8811f9a210184bfb0f8'}
if(r.get('schema')!='world_reward.frontend_peer_receive.v1' or r.get('status')!='pass' or r.get('archive_identity')!=archive_pin or r.get('bytes_received')!=archive_pin['bytes'] or r.get('receipt_written')is not True):raise ValueError('Genuine completed fixed transport required')
helpers={'infra/frontend_replica_inventory.py':{'bytes':28238,'sha256':'f9cbb398a53beb257c580df0959c47c707e978afb653ea411382ae8e205821ee','git_blob_sha1':'a46ae3bf244ca70d5ce74df308dbdc285229da36'},'infra/run_frontend_replica_inventory.sh':{'bytes':3088,'sha256':'db01c510368134cb0dcac6fc10cc73d8e18f689d635c7fd62248b0fd11727fc3','git_blob_sha1':'14e57c10933f13164a8fcb32bccd0e439fab674b'}}
sources={n:identity(code/n,1_000_000)[1]for n in('infra/frontend_asset_extract.py','infra/frontend_asset_archive.py')}
m=types.ModuleType('source_bound_asset_extract');m.__file__=str(code/'infra/frontend_asset_extract.py');raw=identity(Path(m.__file__),1_000_000)[0];exec(compile(raw,m.__file__,'exec'),m.__dict__)
report=m.extract_archive(incoming/'archive.tar',archive_pin,manifest_pin,code,helpers,dest,sources)
if closure()!=before or identity(incoming/'report.json',4000)[0]!=b:report.update(status='fail',error='Immutable source/transport receipt changed after extraction')
report.update(producer_revision=rev,elapsed_seconds=time.perf_counter()-start,transport_report_sha256=hashlib.sha256(b).hexdigest())
payload=(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode()
if len(payload)>12000:raise ValueError('Tiny extraction receipt bound exceeded')
if out.resolve()!=out or out.stat().st_mode&0o777!=0o700 or (out.stat().st_dev,out.stat().st_ino)!=out_owned:raise ValueError('Private result namespace changed')
with os.fdopen(os.open(out/'report.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400),'wb')as f:f.write(payload);f.flush();os.fsync(f.fileno())
print(json.dumps({k:report[k]for k in('stage','status','elapsed_seconds','replica_ready','promotion_performed')},sort_keys=True))
if report['status']!='pass':raise RuntimeError('Private extraction failed; partial tree retained')
PY
