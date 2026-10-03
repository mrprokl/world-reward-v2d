"""Explicit selected-archive source binding, not a fabricated clean Git tree.

The independently frozen extraction/manifest receipts authenticate the source
whitelist. Read only selected Body/DINO source bytes, never models or the full
20GB archive. Old inference keeps its original Git binder unless opted in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

ROOT=Path('/srv/scenesmith/world-reward')
DISK=Path('/srv/world-reward-data')
PINS='configs/frontend_asset_archive_pins.json'
MANIFEST='world-reward-frontend-assets-manifest.json'
UPSTREAM='7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80'
DINO_REV='6876159a11b4df116f30f667f8c9888617df0751'
BODY='vendor/video_to_data/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body'
DINO='weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main'
BODY_DATA={'data/__init__.py','data/transforms/__init__.py','data/transforms/bbox_utils.py','data/transforms/common.py','data/utils/io.py','data/utils/prepare_batch.py'}

def require(condition,message):
 if not condition:raise ValueError(message)

def canonical(path):
 path=Path(path)
 require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical selected source path required')
 return path

def safe_name(name):
 require(type(name)is str and name and not PurePosixPath(name).is_absolute()and str(PurePosixPath(name))==name and '\\'not in name and not any(p in('','.','..','.git','.secrets','__pycache__')for p in name.split('/'))and '\0'not in name,'Canonical public manifest source name required')
 return name

def strict_json(raw):
 def pairs(rows):
  result={}
  for key,value in rows:require(key not in result,'Duplicate source binding JSON field');result[key]=value
  return result
 def invalid(_):raise ValueError('Nonfinite source binding JSON')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def read(path,maximum=1_000_000,empty=False,readonly=True):
 path=canonical(path);s=path.lstat()
 require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and(not readonly or not s.st_mode&0o222)and (0 if empty else 1)<=s.st_size<=maximum,'Readonly bounded regular selected source required')
 raw=path.read_bytes();a=path.lstat()
 require((s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(a.st_dev,a.st_ino,a.st_mode,a.st_size,a.st_mtime_ns,a.st_ctime_ns),'Selected source changed during read')
 return raw,{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}

def pinned_json(path,pin):
 require(type(pin)is dict and set(pin)=={'bytes','sha256'}and type(pin['bytes'])is int and 0<pin['bytes']<=200000 and re.fullmatch('[0-9a-f]{64}',str(pin['sha256'])),'Independent bounded metadata SHA/bytes required')
 raw,actual=read(path,200000)
 require({k:actual[k]for k in('bytes','sha256')}==pin,'Metadata bytes differ from independently frozen pins')
 return strict_json(raw),actual

def load_contract(root,selected_manifest):
 """Actual Azure paths + immutable config; pure validation below is fixtureable."""
 root=canonical(root);require(root==ROOT,'Exact selected-source runtime root required')
 helper=canonical(Path(__file__));code=helper.parent.parent
 require(helper==code/'infra/frontend_selected_assets.py','Actual selected-source helper required')
 read(helper)
 canonical(code);require(code.name=='code'and code.parent.parent.parent==root/'jobs','Immutable dispatched selected-source helper required')
 revision=code.parent.parent.name
 require(re.fullmatch('[0-9a-f]{40}',revision),'Frozen source revision required')
 marker=code.parent/'revision';raw,_=read(marker,100,readonly=False)
 require(raw==(revision+'\n').encode(),'Frozen source revision marker mismatch')
 raw,_=read(code.parent/'source-sha256',100,readonly=False);require(re.fullmatch(b'[0-9a-f]{64}\n',raw),'Frozen source archive marker required')
 raw,config_pin=read(code/PINS,200000);pins=strict_json(raw)
 require(pins.get('schema')=='world_reward.frontend_asset_archive.pins.v1'and pins.get('private_extraction_completed')is True and pins.get('promotion_performed')is False,'Independent completed non-promoting extraction pins required')
 revision=pins.get('extraction_producer_revision');require(re.fullmatch('[0-9a-f]{40}',str(revision)),'Exact extraction producer required')
 destination=DISK/('frontend-assets-extracted-'+revision)
 require(pins.get('extraction_relative_root')==destination.name and canonical(selected_manifest)==destination/MANIFEST,'Exact separately pinned extracted manifest namespace required')
 receipt=root/'results'/('frontend-asset-extract-'+revision)/'report.json'
 return validate_contract(destination,selected_manifest,receipt,pins,config_identity=config_pin)

def validate_contract(destination,manifest_path,receipt_path,pins,*,config_identity=None):
 """Validate tiny trusted receipt/manifest; no archive/model read or mutation."""
 destination=canonical(destination);manifest_path=canonical(manifest_path);receipt_path=canonical(receipt_path)
 require(destination.is_dir()and manifest_path==destination/MANIFEST,'Selected manifest must be original extracted member')
 manifest,manifest_pin=pinned_json(manifest_path,pins['manifest_identity'])
 receipt,receipt_pin=pinned_json(receipt_path,pins['extraction_receipt_identity'])
 expected={'schema':'world_reward.frontend_asset_extract.v1','stage':'frontend_asset_extract','status':'pass','producer_revision':pins['extraction_producer_revision'],
  'archive_identity':pins['archive_identity'],'manifest_identity':pins['manifest_identity'],'imported_relative_root':destination.name,'complete_payload_rehashed':True,
  'manifest_preserved_byte_identically':True,'source_helpers_rehashed':True,'archive_rehashed_after_extraction':True,'promotion_performed':False,'models_loaded':False,
  'images_imported':False,'replica_ready':False,'CUDA_verified':False,'license_eligibility_verified':False,'training_overlap_verified':False}
 require(all(receipt.get(k)==v for k,v in expected.items()),'Original genuine extraction receipt contract mismatch')
 require(manifest.get('schema')=='world_reward.frontend_asset_archive.v1'and manifest.get('original_root')==str(ROOT)and manifest.get('raw_build_receipts_exported')is False and manifest.get('images_exported')is False,'Original public selected-source archive manifest required')
 entries=manifest.get('entries');require(type(entries)is dict and 0<len(entries)<=1000,'Bounded authenticated source manifest required')
 digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 require(digest==manifest.get('inventory',{}).get('entries_sha256')==receipt.get('payload_entries_sha256'),'Extraction and source manifest payload ancestry differ')
 for name in entries:safe_name(name)
 sources=manifest.get('sources');require(type(sources)is list,'Original selected repository provenance required')
 for path,revision in(('vendor/video_to_data',UPSTREAM),(DINO,DINO_REV)):
  matches=[r for r in sources if r.get('path')==path]
  require(len(matches)==1 and matches[0].get('revision')==revision,'Selected source revision provenance differs')
 return {'destination':destination,'entries':entries,'manifest_identity':manifest_pin,'extraction_receipt_identity':receipt_pin,'config_identity':config_identity}

def selected_rows(contract,prefix,selector):
 prefix=safe_name(prefix);rows={name:row for name,row in contract['entries'].items()if name.startswith(prefix+'/')and selector(name[len(prefix)+1:])}
 require(rows,'Selected public source whitelist missing')
 records={}
 for name,row in rows.items():
  require(type(row)is dict and row.get('type')=='file'and row.get('role')=='public_source','Selected public source must be an authenticated regular source')
  raw,pin=read(contract['destination']/name,empty=True)
  require(all(pin[k]==row.get(k)for k in('bytes','sha256','git_blob_sha1')),'Selected public source byte/SHA/Git blob differs')
  records[name[len(prefix)+1:]]=pin
 return records

def python_inventory(directory):
 directory=canonical(directory);require(directory.is_dir(),'Original installed source directory required');paths={}
 for path in directory.rglob('*'):
  canonical(path);mode=path.lstat().st_mode
  require(stat.S_ISDIR(mode)or stat.S_ISREG(mode),'Installed source symlink/special file forbidden')
  if path.suffix=='.py':paths[str(path.relative_to(directory))]=path
 return paths

def bind_body(contract,installed):
 rows=selected_rows(contract,BODY,lambda n:n.endswith('.py')or n=='LICENSE')
 wanted={n:row for n,row in rows.items()if n.endswith('.py')}
 require('LICENSE'in rows and '__init__.py'in wanted and {n for n in wanted if n.startswith('data/')}==BODY_DATA,'Complete selected Body source/license closure required')
 actual=python_inventory(installed);require(set(actual)==set(wanted),'Installed Body Python file set differs from selected upstream')
 hashes={}
 for name,path in actual.items():
  _,pin=read(path,empty=True,readonly=False);require(pin==wanted[name],'Installed Body bytes/SHA/Git blob differs from authenticated source');hashes[name]=pin['sha256']
 return {'python_files':len(hashes),'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),'source_binding':'verified_selected_archive',
  'whole_checkout_verified':False,'manifest_identity':contract['manifest_identity'],'extraction_receipt_identity':contract['extraction_receipt_identity']}

def bind_dinov3(contract,repository):
 rows=selected_rows(contract,DINO,lambda n:n.endswith('.py')and(n.startswith('dinov3/')or n=='hubconf.py')or n in('LICENSE.md','MODEL_CARD.md'))
 require({'hubconf.py','LICENSE.md','MODEL_CARD.md','dinov3/__init__.py'}<=set(rows),'Selected DINOv3 source/license/card closure required')
 actual=python_inventory(repository);wanted={n for n in rows if n.endswith('.py')}
 require(set(actual)==wanted,'Local DINOv3 Python set differs from selected source')
 for name,row in rows.items():
  _,pin=read(Path(repository)/name,empty=True);require(pin==row,'Local DINOv3 differs from authenticated source')
 return {'source_binding':'verified_selected_archive','whole_checkout_verified':False,'revision':DINO_REV,'python_files':len(wanted)}
