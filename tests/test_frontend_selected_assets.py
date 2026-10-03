"""Synthetic source/provenance only; no model bytes, challenge data or Git."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frontend_selected_assets',ROOT/'infra/frontend_selected_assets.py')
assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)

def json_raw(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def pin(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():path.chmod(0o600)
 path.write_bytes(raw);path.chmod(0o400)

@pytest.fixture
def fixture(tmp_path):
 destination=tmp_path/'extracted';destination.mkdir();entries={};installed=tmp_path/'installed';installed.mkdir()
 body={'__init__.py',*assets.BODY_DATA,'model.py','LICENSE'}
 dino={'hubconf.py','dinov3/__init__.py','dinov3/model.py','LICENSE.md','MODEL_CARD.md'}
 for prefix,names in((assets.BODY,body),(assets.DINO,dino)):
  for name in names:
   raw=('procedural '+name+'\n').encode();path=destination/prefix/name;write(path,raw)
   entries[prefix+'/'+name]={'type':'file','role':'public_source',**pin(raw),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
   if prefix==assets.BODY and name.endswith('.py'):write(installed/name,raw)
 digest=hashlib.sha256(json.dumps(entries,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 manifest={'schema':'world_reward.frontend_asset_archive.v1','original_root':str(assets.ROOT),'entries':entries,'inventory':{'entries_sha256':digest},'raw_build_receipts_exported':False,'images_exported':False,'sources':[{'path':'vendor/video_to_data','revision':assets.UPSTREAM},{'path':assets.DINO,'revision':assets.DINO_REV}]}
 manifest_path=destination/assets.MANIFEST;raw=json_raw(manifest);write(manifest_path,raw)
 pins={'manifest_identity':pin(raw),'archive_identity':{'bytes':999,'sha256':'a'*64},'extraction_producer_revision':'b'*40}
 receipt={'schema':'world_reward.frontend_asset_extract.v1','stage':'frontend_asset_extract','status':'pass','producer_revision':pins['extraction_producer_revision'],'archive_identity':pins['archive_identity'],'manifest_identity':pins['manifest_identity'],'imported_relative_root':destination.name,'complete_payload_rehashed':True,'manifest_preserved_byte_identically':True,'source_helpers_rehashed':True,'archive_rehashed_after_extraction':True,'promotion_performed':False,'models_loaded':False,'images_imported':False,'replica_ready':False,'CUDA_verified':False,'license_eligibility_verified':False,'training_overlap_verified':False,'payload_entries_sha256':digest}
 receipt_path=tmp_path/'receipt.json';raw=json_raw(receipt);write(receipt_path,raw);pins['extraction_receipt_identity']=pin(raw)
 return destination,manifest_path,receipt_path,pins,installed,manifest,receipt

def contract(fixture):
 destination,manifest,receipt,pins,*_=fixture
 return assets.validate_contract(destination,manifest,receipt,pins)

def test_exact_selected_source_body_and_dino_without_git(fixture):
 selected=contract(fixture);report=assets.bind_body(selected,fixture[4])
 assert report['source_binding']=='verified_selected_archive' and report['whole_checkout_verified']is False
 assert report['python_files']==8
 report=assets.bind_dinov3(selected,fixture[0]/assets.DINO)
 assert report['revision']==assets.DINO_REV and report['whole_checkout_verified']is False
 assert not (fixture[0]/'.git').exists()

@pytest.mark.parametrize('kind',['manifest_sha','receipt_sha','receipt_fail','receipt_archive','payload','producer','private','source_revision','duplicate_repo','manifest_namespace','promotion'])
def test_provenance_mismatch_fails_closed(fixture,kind):
 destination,manifest_path,receipt_path,pins,installed,manifest,receipt=fixture
 if kind=='manifest_sha':pins['manifest_identity']['sha256']='0'*64
 elif kind=='receipt_sha':pins['extraction_receipt_identity']['sha256']='0'*64
 elif kind=='receipt_fail':receipt['status']='fail'
 elif kind=='receipt_archive':receipt['archive_identity']={'bytes':999,'sha256':'0'*64}
 elif kind=='payload':receipt['payload_entries_sha256']='0'*64
 elif kind=='producer':receipt['producer_revision']='c'*40
 elif kind=='private':manifest['entries']['../private']=copy.deepcopy(next(iter(manifest['entries'].values())))
 elif kind=='source_revision':manifest['sources'][1]['revision']='0'*40
 elif kind=='duplicate_repo':manifest['sources'].append(copy.deepcopy(manifest['sources'][1]))
 elif kind=='manifest_namespace':manifest_path=destination/'foreign.json'
 elif kind=='promotion':receipt['promotion_performed']=True
 if kind not in('manifest_sha','receipt_sha'):
  raw=json_raw(manifest);write(manifest_path,raw);pins['manifest_identity']=pin(raw)
  receipt['manifest_identity']=pins['manifest_identity'];raw=json_raw(receipt);write(receipt_path,raw);pins['extraction_receipt_identity']=pin(raw)
 with pytest.raises(ValueError):assets.validate_contract(destination,manifest_path,receipt_path,pins)

@pytest.mark.parametrize('kind',['changed','missing','extra','symlink','hardlink'])
def test_installed_body_exact_source_contract(fixture,tmp_path,kind):
 selected=contract(fixture);installed=fixture[4];path=installed/'model.py'
 if kind=='changed':path.chmod(0o600);path.write_bytes(b'foreign code')
 elif kind=='missing':path.unlink()
 elif kind=='extra':write(installed/'extra.py',b'extra')
 elif kind=='symlink':path.unlink();path.symlink_to(fixture[0]/assets.BODY/'model.py')
 elif kind=='hardlink':(tmp_path/'alias.py').hardlink_to(path)
 with pytest.raises(ValueError):assets.bind_body(selected,installed)

@pytest.mark.parametrize('kind',['blob','sha','role','type','missing_license','missing_data'])
def test_selected_source_pins_not_recomputed_from_corrupted_bytes(fixture,kind):
 selected=contract(fixture);name=assets.BODY+'/model.py'
 if kind=='blob':selected['entries'][name]['git_blob_sha1']='0'*40
 elif kind=='sha':selected['entries'][name]['sha256']='0'*64
 elif kind=='role':selected['entries'][name]['role']='asset'
 elif kind=='type':selected['entries'][name]['type']='symlink'
 elif kind=='missing_license':del selected['entries'][assets.BODY+'/LICENSE']
 elif kind=='missing_data':del selected['entries'][assets.BODY+'/data/__init__.py']
 with pytest.raises(ValueError):assets.bind_body(selected,fixture[4])

def test_model_entries_never_read_by_source_binding(fixture,monkeypatch):
 selected=contract(fixture);selected['entries']['weights/do-not-read-model.ckpt']={'type':'file','bytes':10**10}
 seen=[];original=assets.read
 def read(path,*args,**kwargs):seen.append(str(path));return original(path,*args,**kwargs)
 monkeypatch.setattr(assets,'read',read)
 assets.bind_body(selected,fixture[4]);assets.bind_dinov3(selected,fixture[0]/assets.DINO)
 assert seen and not any('ckpt'in path for path in seen)

def test_dino_extra_or_changed_source_fails(fixture):
 selected=contract(fixture);repository=fixture[0]/assets.DINO
 write(repository/'foreign.py',b'foreign')
 with pytest.raises(ValueError):assets.bind_dinov3(selected,repository)

def test_metadata_is_independently_byte_pinned(fixture):
 path=fixture[1];original=path.read_bytes();path.chmod(0o600);path.write_bytes(original+b' ');path.chmod(0o400)
 with pytest.raises(ValueError,match='independently frozen'):contract(fixture)

def test_default_body_parser_and_git_branch_unchanged():
 source=(ROOT/'infra/body_smoke.py').read_text()
 assert 'def _source_identity(root: Path, *, selected_manifest=None)'in source
 assert 'def _install_local_dinov3_loader(torch, repository: Path, *, selected_manifest=None)'in source
 assert 'if selected_manifest is None:\n        _pinned_checkout(repository, DINOV3_REVISION)'in source
 assert 'parser.add_argument("--selected-source-manifest", type=Path, default=None'in source
 assert '_pinned_checkout(vendor, UPSTREAM_REVISION)'in source

def test_explicit_loader_uses_genuine_local_hub_only(fixture,monkeypatch):
 spec=importlib.util.spec_from_file_location('body_selected_loader_test',ROOT/'infra/body_smoke.py')
 body=importlib.util.module_from_spec(spec);spec.loader.exec_module(body)
 selected=contract(fixture);bound=[];hub_calls=[]
 def original(repo,*args,**kwargs):hub_calls.append((repo,args,kwargs));return 'genuine backend'
 monkeypatch.setitem(sys.modules,'frontend_selected_assets',SimpleNamespace(load_contract=lambda root,manifest:bound.append((root,manifest))or selected,bind_dinov3=assets.bind_dinov3))
 torch=SimpleNamespace(hub=SimpleNamespace(load=original));repository=fixture[0]/assets.DINO
 saved,calls=body._install_local_dinov3_loader(torch,repository,selected_manifest=fixture[1])
 assert saved is original and bound==[(fixture[0],fixture[1])]
 assert torch.hub.load('facebookresearch/dinov3','test_model',source='github',pretrained=False)=='genuine backend'
 assert calls==['test_model']and hub_calls==[(str(repository),('test_model',),{'source':'local','pretrained':False})]
 with pytest.raises(RuntimeError):torch.hub.load('foreign','test_model',source='github',pretrained=False)
 with pytest.raises(RuntimeError):torch.hub.load('facebookresearch/dinov3','test_model',source='github',pretrained=True)
