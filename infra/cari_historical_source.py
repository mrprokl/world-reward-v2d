"""Hash-only original producer binding; never execute old code or rewrite receipts."""
from __future__ import annotations
import hashlib,json,re,stat
from pathlib import Path


def _read(path,maximum=2_000_000,readonly=True):
 path=Path(path)
 if not path.is_absolute()or path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)):
  raise ValueError('Canonical historical source required')
 before=path.lstat()
 if not stat.S_ISREG(before.st_mode)or before.st_nlink!=1 or not 0<before.st_size<=maximum or readonly and before.st_mode&0o222:
  raise ValueError('Bounded regular immutable historical source required')
 raw=path.read_bytes();after=path.lstat()
 fields=('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')
 if any(getattr(before,k)!=getattr(after,k)for k in fields):raise ValueError('Historical source changed while reading')
 return raw,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def _json(raw):
 def pairs(rows):
  result={}
  for k,v in rows:
   if k in result:raise ValueError('Duplicate historical pin key')
   result[k]=v
  return result
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError('Nonfinite historical pin')))


def verify_historical_source(root,pins_path):
 """Independent committed full Git-byte pins + original dispatch/cache audit."""
 root=Path(root);raw,pin=_read(pins_path);pins=_json(raw)
 if pins.get('schema')!='world_reward.historical_native_source_pins.v1'or pins.get('queue_status_reclassified')is not False:
  raise ValueError('Explicit non-reclassifying historical source pins required')
 rev=pins.get('producer_revision');job=pins.get('job');digest=pins.get('source_archive_sha256')
 if not re.fullmatch('[0-9a-f]{40}',str(rev))or job!='run_cari_full_refine_queued'or not re.fullmatch('[0-9a-f]{64}',str(digest)):
  raise ValueError('Original dispatched historical producer required')
 code=root/'jobs'/rev/job/'code'
 for name,wanted in(('revision',rev),('source-sha256',digest)):
  value,_=_read(code.parent/name,100,readonly=False)
  if value!=(wanted+'\n').encode():raise ValueError('Original historical dispatch marker changed')
 files=pins.get('files')
 if type(files)is not dict or not 1<=len(files)<=200:raise ValueError('Complete bounded independently frozen source map required')
 for name,row in files.items():
  if not re.fullmatch(r'(?:infra|src|configs)/[A-Za-z0-9_./-]+|pyproject\.toml',name)or any(p in('','..','.','__pycache__','.git')for p in name.split('/')):
   raise ValueError('Source-only original relative filename required')
  if type(row)is not dict or set(row)!={'bytes','sha256'}or type(row['bytes'])is not int or not re.fullmatch('[0-9a-f]{64}',str(row['sha256'])):
   raise ValueError('Independent historical file byte pins required')
 observed={}
 for path in(code,*sorted(code.rglob('*'))):
  if path.resolve()!=path or path.is_symlink()or path.stat().st_mode&0o222:
   raise ValueError('Complete historical closure must remain readonly without aliases')
  if path.is_dir():continue
  name=str(path.relative_to(code));observed[name]=_read(path)[1]
 if observed!=files:raise ValueError('Historical complete source bytes/file set differs from original Git pins')
 audit=pins.get('separate_queue_failure_receipt',{})
 if audit.get('relative_path')!='results/episode3-queued-source-cache-audit.json':raise ValueError('Separate original source-cache audit required')
 audit_raw,audit_pin=_read(root/audit['relative_path'],10000)
 if audit_pin!={k:audit[k]for k in('bytes','sha256')}:raise ValueError('Separate source-cache audit identity differs')
 if _json(audit_raw).get('status')!='pass':raise ValueError('Independent original source-cache audit must pass; queue FAIL unchanged')
 if _read(pins_path)[1]!=pin:raise ValueError('Historical consumer pins changed')
 return code,{'producer_revision':rev,'job':job,'source_archive_sha256':digest,'files_verified':len(files),'pins_identity':pin,'queue_status_reclassified':False}
