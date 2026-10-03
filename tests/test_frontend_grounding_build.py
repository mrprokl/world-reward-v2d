"""Tiny public-pin/recipe/control tests. No network, Docker, models or Azure."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frontend_grounding_build',ROOT/'infra/frontend_grounding_build.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)

@pytest.fixture
def pins():return json.loads((ROOT/build.CONFIG).read_text())

def test_actual_complete_public_source_and_wheel_pins(pins):
 assert build.validate_pins(pins)==pins
 sam,nv=pins['repositories']
 assert len(sam['files'])==32 and sum(r['path'].endswith('.py')for r in sam['files'])==26
 assert len(nv['files'])==12
 assert next(r for r in sam['files']if r['path']=='LICENSE')['sha256']=='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
 assert next(r for r in nv['files']if r['path']=='LICENSE')['sha256']=='e87dc2a40b553c5f52acb3909f479afbeac8c35a85328fceaa0cab15ed50b9fc'
 assert not any('/data/'in r['path']or 'checkpoint'in r['path']or 'recording'in r['path']for repo in pins['repositories']for r in repo['files'])

@pytest.mark.parametrize('kind',['base','target','revision','source_missing','source_duplicate','private','blob','url','wheelsha','wheelbytes','wheelversion','wheelurl','wheelpolicy'])
def test_contradictory_or_broad_inputs_rejected(pins,kind):
 if kind=='base':pins['base_image_id']='sha256:'+'0'*64
 elif kind=='target':pins['target_image']='world-reward/grounding:0.1'
 elif kind=='revision':pins['repositories'][0]['revision']='0'*40
 elif kind=='source_missing':pins['repositories'][0]['files'].pop()
 elif kind=='source_duplicate':pins['repositories'][0]['files'].append(copy.deepcopy(pins['repositories'][0]['files'][0]))
 elif kind=='private':pins['repositories'][1]['files'][0]['path']='track_2/private'
 elif kind=='blob':pins['repositories'][0]['files'][0]['git_blob_sha1']='bad'
 elif kind=='url':pins['repositories'][0]['files'][0]['url']='https://example.com/source'
 elif kind=='wheelsha':pins['wheels'][0]['sha256']='0'*64
 elif kind=='wheelbytes':pins['wheels'][0]['bytes']+=1
 elif kind=='wheelversion':pins['wheels'][0]['version']='5.3.0'
 elif kind=='wheelurl':pins['wheels'][0]['url']='https://files.pythonhosted.org/a?token=private'
 elif kind=='wheelpolicy':pins['wheels'][0]['install_if_absent_only']=True
 with pytest.raises(ValueError):build.validate_pins(pins)

@pytest.mark.parametrize('name',['/absolute','../private','a/../b','a//b','./a','a\\b','a\x00b','a?token=x',''])
def test_unsafe_public_relative_names(name):
 with pytest.raises(ValueError):build.safe_name(name)

def test_git_blob_exact_bytes_not_just_file_name():
 raw=b'procedural public code\n';row={'bytes':len(raw),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
 assert build.verify_source(raw,row)['sha256']==hashlib.sha256(raw).hexdigest()
 with pytest.raises(ValueError):build.verify_source(b'X'+raw[1:],row)
 with pytest.raises(ValueError):build.verify_source(raw,{**row,'sha256':'0'*64})

def test_duplicate_or_nonfinite_json_rejected():
 for raw in ('{"x":1,"x":2}','{"x":NaN}'):
  with pytest.raises(ValueError):build.strict_json(raw)

def test_recipe_offline_no_resolver_and_new_source_only(pins):
 recipe=build.dockerfile(pins,'a'*64,True).decode()
 assert recipe.startswith('FROM '+build.BASE+'\n')
 assert '--no-deps' in recipe and '--no-index' in recipe and '--no-build-isolation'in recipe
 assert 'PIP_CONSTRAINT=/dev/null'in recipe and 'SAM2_BUILD_ALLOW_ERRORS=0'in recipe
 assert 'TORCH_CUDA_ARCH_LIST=9.0'in recipe and 'SAM2_BUILD_CUDA=1'in recipe
 assert '--gpus'not in recipe and 'git+'not in recipe and 'pip install numpy'not in recipe
 assert 'COPY nvidia /opt/world-reward-grounding/nvidia'in recipe
 assert pins['wheels'][-1]['filename']in recipe
 assert pins['wheels'][-1]['filename']not in build.dockerfile(pins,'a'*64,False).decode()

def test_exact_pypi_record_and_license_classifier(pins):
 row=pins['wheels'][0]
 meta={'info':{'name':row['name'],'version':row['version'],'classifiers':['License :: OSI Approved :: Apache Software License']},'urls':[{'filename':row['filename'],'url':row['url'],'size':row['bytes'],'digests':{'sha256':row['sha256']},'yanked':False}]}
 raw=json.dumps(meta).encode();assert build.wheel_metadata(raw,row)['sha256']==hashlib.sha256(raw).hexdigest()
 meta['urls'][0]['yanked']=True
 with pytest.raises(ValueError):build.wheel_metadata(json.dumps(meta).encode(),row)

def test_packaged_wheel_license_identity(tmp_path):
 path=tmp_path/'procedural.whl'
 with zipfile.ZipFile(path,'w')as archive:archive.writestr('example.dist-info/licenses/LICENSE.txt',b'procedural Apache notice')
 notice=build.wheel_notices(path)
 assert len(notice)==1 and next(iter(notice.values()))['bytes']==24
 bad=tmp_path/'missing.whl'
 with zipfile.ZipFile(bad,'w')as archive:archive.writestr('module.py',b'code')
 with pytest.raises(ValueError):build.wheel_notices(bad)
 assert build.wheel_notices(bad,external_notice_verified=True)=={}

@pytest.mark.parametrize('kind',['commit','version','license','url','permission','missing'])
def test_external_notice_requires_exact_frozen_publisher_closure(pins,kind):
 row=pins['wheels'][1];notice=row['publisher_notice']
 if kind=='commit':notice['revision']='0'*40
 elif kind=='version':notice['version_literal']='version = "0.0.0"'
 elif kind=='license':notice['license']['sha256']='0'*64
 elif kind=='url':notice['license']['url']='https://example.com/LICENSE'
 elif kind=='permission':notice['packaged_notice_absence_permitted_with_external_pinned_publisher_notice']=False
 elif kind=='missing':del row['publisher_notice']
 with pytest.raises(ValueError):build.validate_pins(pins)

def test_publisher_tag_version_and_license_evidence(pins,tmp_path,monkeypatch):
 row=pins['wheels'][1];notice=row['publisher_notice'];captured=[]
 responses={notice['release_ref_url']:json.dumps({'ref':'refs/tags/'+notice['release_tag'],'object':{'type':'commit','sha':notice['revision']}}).encode(),notice['license']['url']:b'procedural publisher LICENSE',notice['version_source']['url']:b'version = "0.21.4"'}
 monkeypatch.setattr(build,'fetch',lambda url,maximum:responses[url])
 monkeypatch.setattr(build,'verify_source',lambda raw,source:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':source['git_blob_sha1']})
 monkeypatch.setattr(build,'exclusive',lambda path,raw:captured.append((path,raw)))
 evidence=build.publisher_notice(row,tmp_path)
 assert evidence['runtime_tag_resolution_performed']is False
 assert evidence['immutable_publisher_sources_verified']is True
 assert evidence['wheel_bytes_equivalent_to_publisher_source_proven']is False
 assert len(captured)==2 and all('publisher-notices/tokenizers'in str(path)for path,raw in captured)
 del responses[notice['release_ref_url']]
 assert build.publisher_notice(row,tmp_path)['runtime_tag_resolution_performed']is False

def test_original_failure_and_image_not_reused(pins):
 source=(ROOT/'infra/frontend_grounding_build.py').read_text()
 assert build.TARGET=='world-reward/frontend-grounding-v4:0.1'
 assert "root/'results/frontend-grounding-build-v4'"in source
 assert 'frontend-grounding-build-v1'not in source and 'frontend-grounding-build-v2'not in source
 recipe=build.dockerfile(pins,'a'*64,True).decode()
 assert 'COPY publisher-notices /opt/world-reward-grounding/publisher-notices'in recipe

def test_bounded_isolated_control_never_logs_process_output(monkeypatch):
 calls=[]
 def run(args,**kwargs):calls.append((args,kwargs));return subprocess.CompletedProcess(args,1,'hf_PRIVATE','hf_PRIVATE')
 monkeypatch.setattr(build.subprocess,'run',run)
 with pytest.raises(ValueError)as caught:build.command(['docker','build'])
 assert str(caught.value)=='Runtime control failed: docker'
 args,kwargs=calls[0];assert args[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=5s','30s']
 assert kwargs['env']==build.SAFE_ENV and kwargs['timeout']==37
 assert 'HF_TOKEN'not in kwargs['env']

def test_cleanup_only_owned_ids_and_survivor_gate(monkeypatch):
 calls=[];answers=iter(['a'*64+'\n','',''])
 def command(args,*a,**kw):calls.append(args);return next(answers)
 monkeypatch.setattr(build,'command',command)
 assert build.cleanup('b'*64)['survivor_check_passed']is True
 assert calls[1]==['docker','rm','--force','a'*64]
 assert calls[0][-1]=='label='+build.LABEL+'='+'b'*64
 monkeypatch.setattr(build,'command',lambda *args,**kwargs:'foreign')
 with pytest.raises(ValueError):build.cleanup('b'*64)

def test_cpu_probe_has_no_gpu_or_model_mounts(monkeypatch):
 calls=[]
 monkeypatch.setattr(build,'command',lambda args,*a,**kw:calls.append(args)or '{"CPU":true}\n')
 assert build.probe(build.BASE,'wr-grounding-test','c'*64)==[{'CPU':True}]
 args=calls[0]
 assert '--gpus'not in args and '--mount'not in args and '--volume'not in args
 assert args[args.index('--network')+1]=='none'and '--read-only'in args
 assert args[args.index('--entrypoint')+1]=='/usr/bin/env' and '-i'in args
 assert 'SAM2ImagePredictor('not in build.CHILD_PROBE and '.cuda('not in build.CHILD_PROBE
 assert 'not torch.cuda.is_initialized()'in build.CHILD_PROBE

def test_receipt_no_overwrite_and_readonly(tmp_path):
 target=tmp_path/'output/report.json';build.exclusive(target,b'{}\n')
 assert target.stat().st_mode&0o777==0o400
 with pytest.raises(FileExistsError):build.exclusive(target,b'new')

@pytest.mark.parametrize('status',[0,1,124])
def test_offline_build_private_log_not_terminal_or_error(tmp_path,monkeypatch,status):
 monkeypatch.setattr(build,'ROOT',tmp_path);calls=[]
 def run(args,**kwargs):
  calls.append((args,kwargs));kwargs['stdout'].write(b'procedural public compiler diagnostic\n')
  return subprocess.CompletedProcess(args,status)
 monkeypatch.setattr(build.subprocess,'run',run)
 report={};path=tmp_path/'private.log'
 if status:
  with pytest.raises(ValueError,match='inspect owned private Azure build log'):build.offline_build(['docker','build'],path,report)
 else:build.offline_build(['docker','build'],path,report)
 assert path.stat().st_mode&0o777==0o400
 assert report['offline_build_exit_code']==status
 assert report['private_build_log']=={'relative_path':'private.log','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'terminal_output_disclosed':False}
 args,kwargs=calls[0];assert args[:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=5s','1000s']
 assert kwargs['env']==build.SAFE_ENV and kwargs['stderr']==subprocess.STDOUT and kwargs['timeout']==1007
 with pytest.raises(FileExistsError):build.offline_build(['docker','build'],path,{})
