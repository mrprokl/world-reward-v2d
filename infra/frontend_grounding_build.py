"""Azure-only CPU build of a NEW pinned Grounding child; no models or data.

Acquire only frozen public source bytes and five possible dependency wheels.
All installs are offline/no-deps. Original Body image/versions remain unchanged.
This is not the original Grounding image, CUDA execution or leakage clearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

ROOT=Path('/srv/scenesmith/world-reward')
BASE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
TARGET='world-reward/frontend-grounding-v2:0.1'
CONFIG='configs/frontend_grounding_source_pins.json'
HELPERS=('infra/frontend_grounding_build.py','infra/run_frontend_grounding_build.sh',CONFIG)
SAM_REV='2b90b9f5ceec907a1c18123530e92e794ad901a4'
NV_REV='7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80'
BUDGET=1800
WHEEL_EXPECTED={
 'transformers':('4.53.3',10826382,'5aba81c92095806b6baf12df35d756cf23b66c356975fb2a7fa9e536138d7c75'),
 'tokenizers':('0.21.4',3115426,'51b7eabb104f46c1c50b486520555715457ae833d5aee9ff6ae853d1130506ff'),
 'safetensors':('0.6.2',485835,'8045db2c872db8f4cbe3faa0495932d89c38c899c603f21e9b6486951a5ecb8f'),
 'huggingface_hub':('0.36.2',566395,'48f0c8eac16145dfce371e9d2d7772854a4f591bcb56c9cf548accf531d54270'),
 'decord':('0.6.0',13602299,'51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976')}
COMMON=('__init__.py','broadcast.py','utils.py','datatypes.py','video.py','pyproject.toml')
THIN=('__init__.py','datatypes.py','sam2_utils.py','video_to_masks.py','pyproject.toml')
LABEL='world_reward_frontend_grounding_owner'
NOTICE_RELEASES={
 'tokenizers':('huggingface/tokenizers','v0.21.4','e892882fd4608b468dcf9dc33ea95283882b8e6d'),
 'safetensors':('huggingface/safetensors','v0.6.2','aa6c43d729868fc43918e862d42bfeaf60485d1d'),
 'decord':('dmlc/decord','v0.6.0','6a3617cef035535193f390f86399dde139fa2a53')}
SAFE_ENV={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','LANG':'C.UTF-8',
 'DOCKER_HOST':'unix://'+str(ROOT/'docker.sock'),'DOCKER_BUILDKIT':'0'}

def require(value,message):
 if not value:raise ValueError(message)

def safe_name(value):
 require(type(value)is str and re.fullmatch(r'[A-Za-z0-9_+./-]+',value) and not PurePosixPath(value).is_absolute() and str(PurePosixPath(value))==value and not any(p in('','..','.')for p in value.split('/')),'Canonical public relative source name required')
 return value

def canonical(path):
 path=Path(path)
 require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical owned path required')
 return path

def identity(path,readonly=False):
 path=canonical(path);s=path.lstat()
 require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and 0<s.st_size<=2_000_000 and(not readonly or not s.st_mode&0o222),'Bounded regular immutable source required')
 raw=path.read_bytes();a=path.lstat()
 require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns,a.st_mode),'Source changed while reading')
 return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def strict_json(raw):
 def pairs(rows):
  result={}
  for key,value in rows:require(key not in result,'Duplicate public metadata key');result[key]=value
  return result
 def nonfinite(_):raise ValueError('Nonfinite public metadata')
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=nonfinite)

def source_binding(code,revision):
 code=canonical(code)
 require(re.fullmatch('[0-9a-f]{40}',revision) and code==ROOT/'jobs'/revision/'run_frontend_grounding_build/code','Actual immutable builder namespace required')
 markers={}
 for name in('revision','source-sha256'):
  identity(code.parent/name);raw=(code.parent/name).read_bytes()
  require(raw==(revision+'\n').encode() if name=='revision' else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Original dispatch markers required');markers[name]=raw.decode()
 sha=hashlib.sha256()
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);s=path.lstat()
  require(not s.st_mode&0o222 and(stat.S_ISDIR(s.st_mode)or stat.S_ISREG(s.st_mode)),'Complete readonly source closure required')
  if stat.S_ISREG(s.st_mode):pin=identity(path,True);sha.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(pin['sha256']))
 return {'markers':markers,'closure_sha256':sha.hexdigest(),'helpers':{n:identity(code/n,True)for n in HELPERS}}

def validate_pins(pins):
 require(type(pins)is dict and pins.get('schema')=='world_reward.frontend_grounding_source_pins.v2' and pins.get('base_image_id')==BASE and pins.get('target_image')==TARGET,'Exact new child identity contract required')
 repositories=pins.get('repositories');require(type(repositories)is list and len(repositories)==2,'Two public pinned source repositories required')
 for row,(repo,revision)in zip(repositories,(('facebookresearch/sam2',SAM_REV),('nvidia-isaac/video_to_data',NV_REV))):
  require(row.get('repo')==repo and row.get('revision')==revision,'Exact historical public source commit required')
  records=row.get('files');require(type(records)is list and 0<len(records)<=40,'Bounded source whitelist required');names=[]
  for item in records:
   name=safe_name(item['path']);names.append(name)
   require(type(item.get('bytes'))is int and 0<item['bytes']<=100000 and re.fullmatch('[0-9a-f]{40}',str(item.get('git_blob_sha1',''))) and item.get('url')==f'https://raw.githubusercontent.com/{repo}/{revision}/{name}','Exact public source blob URL/size required')
  require(len(names)==len(set(names)) and 'LICENSE'in names,'Unique source whitelist and public licenses required')
  if repo=='facebookresearch/sam2':
   required={'LICENSE','README.md','setup.py','pyproject.toml','MANIFEST.in','sam2/csrc/connected_components.cu','sam2/configs/sam2.1/sam2.1_hiera_l.yaml'}
   require(required<=set(names) and len(names)==32 and all(n in required or n.startswith('sam2/')and n.endswith('.py')for n in names),'Exact 26-Python SAM2 source/config/CUDA closure required')
  else:
   require(set(names)=={'LICENSE',*['reconstruction/modules/v2d_common/'+n for n in COMMON],*['reconstruction/modules/v2d_sam2/lib/'+n for n in THIN]},'Only thin public NVIDIA mask/common source allowed')
 wheels=pins.get('wheels');require(type(wheels)is list and len(wheels)==5 and {r.get('name')for r in wheels}==set(WHEEL_EXPECTED),'Five explicit possible dependency wheels required')
 for row in wheels:
  name=row['name'];require((row.get('version'),row.get('bytes'),row.get('sha256'))==WHEEL_EXPECTED[name],'Independent exact wheel version/SHA/size required')
  safe_name(row['filename']);parsed=urllib.parse.urlsplit(row['url'])
  require(parsed.scheme=='https' and parsed.netloc=='files.pythonhosted.org' and not parsed.query and not parsed.fragment and parsed.path.endswith('/'+row['filename']) and row.get('pypi_url')==f'https://pypi.org/pypi/{name}/{row["version"]}/json' and row.get('install_if_absent_only')is(name=='decord'),'Pinned primary wheel URL and install policy required')
  if name in NOTICE_RELEASES:
   notice=row.get('publisher_notice',{});repo,tag,revision=NOTICE_RELEASES[name]
   require((notice.get('repo'),notice.get('release_tag'),notice.get('revision'))==(repo,tag,revision) and notice.get('release_ref_url')==f'https://api.github.com/repos/{repo}/git/ref/tags/{tag}' and notice.get('packaged_notice_absence_permitted_with_external_pinned_publisher_notice')is True,'Exact independently pinned publisher release required')
   for key in('license','version_source'):
    source=notice.get(key,{});path=safe_name(source.get('path'))
    require(source.get('url')==f'https://raw.githubusercontent.com/{repo}/{revision}/{path}' and type(source.get('bytes'))is int and 0<source['bytes']<=15000 and re.fullmatch('[0-9a-f]{64}',str(source.get('sha256','')))and re.fullmatch('[0-9a-f]{40}',str(source.get('git_blob_sha1',''))),'Frozen publisher source notice/version evidence required')
   license=notice['license'];require(license['path']=='LICENSE' and license['bytes']==11357 and license['sha256']=='c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4','Primary Apache publisher LICENSE required')
   require(notice.get('version_literal')==(f'__version__ = "{row["version"]}"'if name=='decord'else f'version = "{row["version"]}"'),'Publisher version literal must match exact wheel release')
  else:require('publisher_notice'not in row,'No unpinned external notice fallback permitted')
 return pins

class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):raise ValueError('Redirect outside frozen public endpoint forbidden')

def fetch(url,maximum):
 require(type(maximum)is int and 0<maximum<=15_000_000,'Bounded public acquisition required')
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
 try:
  with opener.open(urllib.request.Request(url,headers={'User-Agent':'WorldReward-pinned-runtime'}),timeout=25)as stream:
   raw=stream.read(maximum+1)
 except Exception as error:raise ValueError('Pinned public HTTPS acquisition failed')from None
 require(len(raw)<=maximum,'Public endpoint exceeded declared byte bound');return raw

def verify_source(raw,row):
 require(len(raw)==row['bytes'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1'],'Downloaded public source differs from frozen Git blob')
 sha=hashlib.sha256(raw).hexdigest()
 if'sha256'in row:require(sha==row['sha256'],'Pinned public license SHA differs')
 return {'bytes':len(raw),'sha256':sha,'git_blob_sha1':row['git_blob_sha1']}

def wheel_metadata(raw,row):
 metadata=strict_json(raw);info=metadata.get('info',{})
 require(info.get('name','').lower().replace('-','_')==row['name'] and info.get('version')==row['version'] and 'License :: OSI Approved :: Apache Software License'in info.get('classifiers',[]),'Primary exact Apache package metadata required')
 matches=[r for r in metadata.get('urls',[])if r.get('filename')==row['filename']]
 require(len(matches)==1,'Unique pinned public wheel required');actual=matches[0]
 require(actual.get('url')==row['url'] and actual.get('size')==row['bytes'] and actual.get('digests',{}).get('sha256')==row['sha256'] and actual.get('yanked')is False,'Primary wheel identity contradicts frozen pins')
 return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'license_assurance':'primary_PyPI_classifier_plus_pinned_wheel_notices'}

def wheel_notices(path,external_notice_verified=False):
 with zipfile.ZipFile(path)as archive:
  names=[n for n in archive.namelist()if re.search(r'(?i)(?:^|/)(?:license|notice)(?:[._-][^/]*)?$',n)and '.dist-info/'in n]
  require(names or external_notice_verified,'Pinned wheel must retain packaged notice or independently verified exact publisher LICENSE');result={}
  for name in names:
   member=archive.getinfo(name);require(0<member.file_size<=200000,'Bounded packaged public notice required');raw=archive.read(name)
   result[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  return result

def publisher_notice(row,context):
 notice=row['publisher_notice'];repo,tag,revision=NOTICE_RELEASES[row['name']]
 ref=strict_json(fetch(notice['release_ref_url'],20000));require(ref.get('ref')=='refs/tags/'+tag,'Exact publisher release tag required')
 obj=ref.get('object',{})
 if obj.get('type')=='tag':
  sha=obj.get('sha');require(re.fullmatch('[0-9a-f]{40}',str(sha)),'Exact annotated publisher tag required')
  obj=strict_json(fetch(f'https://api.github.com/repos/{repo}/git/tags/{sha}',20000)).get('object',{})
 require(obj.get('type')=='commit'and obj.get('sha')==revision,'Publisher tag changed from independently pinned release commit')
 evidence={}
 for key in('license','version_source'):
  source=notice[key];raw=fetch(source['url'],source['bytes']);pin=verify_source(raw,source)
  if key=='version_source':require(notice['version_literal'].encode()in raw,'Publisher source version disagrees with exact wheel version')
  path='publisher-notices/'+row['name']+'/'+source['path'];exclusive(context/path,raw);evidence[key]={'context_path':path,**pin}
 return {'repo':repo,'release_tag':tag,'revision':revision,'release_tag_binding_verified':True,'wheel_bytes_equivalent_to_publisher_source_proven':False,**evidence}

def exclusive(path,raw):
 canonical(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
 with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_NOFOLLOW',0),0o400),'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

def command(arguments,seconds=30):
 try:result=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(seconds)+'s',*arguments],env=SAFE_ENV,capture_output=True,text=True,timeout=seconds+7,check=False)
 except(OSError,subprocess.TimeoutExpired):raise ValueError('Bounded runtime control unavailable or timed out')from None
 require(result.returncode==0,'Runtime control failed: '+Path(arguments[0]).name)
 return result.stdout

def inspect_image(image):
 fmt='{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'
 row=strict_json(command(['docker','image','inspect',image,'--format',fmt]))
 require(row.get('Id')==image if image.startswith('sha256:')else bool(re.fullmatch('sha256:[0-9a-f]{64}',str(row.get('Id','')))),'Actual image ID required')
 require(row.get('Architecture')=='amd64' and row.get('Os')=='linux' and row.get('RootFS',{}).get('Type')=='layers','Linux AMD64 layer image required')
 layers=row['RootFS'].get('Layers');require(type(layers)is list and layers and all(re.fullmatch('sha256:[0-9a-f]{64}',n)for n in layers),'Ordered rootfs identities required')
 return row

PROBE=r'''import importlib,importlib.metadata as m,json,os,shutil,subprocess,torch,torchvision,numpy
names=('torch','torchvision','numpy','hydra-core','iopath','tqdm','Pillow','scipy','imageio','av','omegaconf','transformers','tokenizers','safetensors','huggingface-hub')
versions={name:m.version(name)for name in names}
assert torch.__version__=='2.5.1+cu124' and torchvision.__version__.split('+')[0]=='0.20.1' and numpy.__version__=='1.26.3'
for name in('cv2','av','imageio','hydra','iopath','scipy','PIL'):importlib.import_module(name)
try:decord=m.version('decord');importlib.import_module('decord')
except m.PackageNotFoundError:decord=None
nvcc=shutil.which('nvcc');cxx=shutil.which('g++');assert nvcc and cxx
cuda=subprocess.run([nvcc,'--version'],capture_output=True,text=True,timeout=5,check=True).stdout
assert 'release 12.4' in cuda and not torch.cuda.is_initialized()
print(json.dumps(dict(versions=versions,decord=decord,nvcc=nvcc,cxx=cxx,CUDA_initialized=False),sort_keys=True))
'''

CHILD_PROBE=PROBE+r'''
from sam2 import _C
from v2d.sam2.lib.video_to_masks import video_to_masks
from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor
assert callable(_C.get_connected_componnets) and callable(video_to_masks) and callable(build_sam2_video_predictor_low_mem)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(extension_import_verified=True,CUDA_execution_verified=False,model_loaded=False,source_path=__import__('sam2').__file__),sort_keys=True))
'''

def probe(image,name,owner,child=False):
 require(re.fullmatch('[0-9a-f]{64}',owner)and re.fullmatch('wr-grounding-[a-z0-9-]+',name),'Exact owned probe identity required')
 args=['docker','run','--rm','--name',name,'--label',LABEL+'='+owner,'--network','none','--memory','6g','--cpus','2','--read-only','--tmpfs','/tmp:rw,noexec,nosuid,size=128m','--entrypoint','/usr/bin/env',image,'-i','PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin','HOME=/tmp','PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','/opt/conda/bin/python','-I','-B','-c',CHILD_PROBE if child else PROBE]
 lines=command(args,90).splitlines();require(len(lines)==(2 if child else 1),'Bounded exact CPU package probe required')
 return [strict_json(line)for line in lines]

def dockerfile(pins,owner,decord_missing):
 require(re.fullmatch('[0-9a-f]{64}',owner),'Exact owned build label required')
 wheels=[r['filename']for r in pins['wheels']if r['name']!='decord'or decord_missing]
 common='/opt/world-reward-grounding/nvidia/reconstruction/modules/v2d_common'
 thin='/opt/world-reward-grounding/nvidia/reconstruction/modules/v2d_sam2/lib'
 sam='/opt/world-reward-grounding/sam2'
 env='env -i PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin HOME=/tmp CUDA_HOME=/usr/local/cuda PIP_CONSTRAINT=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=9.0 SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 MAX_JOBS=4 PYTHONDONTWRITEBYTECODE=1'
 install='python -m pip install --no-index --no-deps --no-build-isolation --disable-pip-version-check'
 return(f'FROM {BASE}\nLABEL {LABEL}="{owner}"\nCOPY sam2 {sam}\nCOPY nvidia /opt/world-reward-grounding/nvidia\nCOPY wheels /opt/world-reward-grounding/wheels\nCOPY publisher-notices /opt/world-reward-grounding/publisher-notices\n'
  +f'RUN {env} {install} '+' '.join('/opt/world-reward-grounding/wheels/'+n for n in wheels)+'\n'
  +f'RUN {env} {install} {sam} {common} {thin} && rm -r /opt/world-reward-grounding/wheels\n'
  +'ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1\n').encode()

def cleanup(owner):
 ids=command(['docker','ps','--all','--quiet','--no-trunc','--filter','label='+LABEL+'='+owner],8).split()
 require(all(re.fullmatch('[0-9a-f]{64}',n)for n in ids),'Only exact owned container IDs permitted')
 for value in ids:command(['docker','rm','--force',value],8)
 survivors=command(['docker','ps','--all','--quiet','--no-trunc','--filter','label='+LABEL+'='+owner],8).strip()
 require(not survivors,'Owned build/probe container survivor');return {'containers_removed':len(ids),'survivor_check_passed':True,'daemon_build_termination_proven':False}

def build(code,revision,output,report):
 before=source_binding(code,revision);pins=validate_pins(strict_json((code/CONFIG).read_bytes()))
 parent=inspect_image(BASE);require(len(parent['RootFS']['Layers'])==44,'Original Body 44-layer parent required')
 require(command(['docker','info','--format','{{.DockerRootDir}}']).strip()==str(ROOT/'docker'),'Existing task-private Docker store required')
 require(not command(['docker','image','ls','--quiet','--no-trunc',TARGET]).strip(),'New Grounding image must be absent; no rebuild or alias')
 owner=hashlib.sha256((revision+before['closure_sha256']).encode()).hexdigest();report['owner']=owner
 prior=probe(BASE,'wr-grounding-parent-'+revision[:12],owner)[0]
 require(prior['decord']in(None,'0.6.0'),'Explicit existing decord version required')
 report.update(phase='public_acquisition',source_binding=before,parent_image=parent,parent_packages=prior)
 context=output/'context';context.mkdir(mode=0o700);files={};wheels={}
 for repo in pins['repositories']:
  folder='sam2'if repo['repo']=='facebookresearch/sam2'else'nvidia'
  for row in repo['files']:
   raw=fetch(row['url'],row['bytes']);pin=verify_source(raw,row);relative=folder+'/'+row['path'];exclusive(context/relative,raw);files[relative]=pin
 for row in pins['wheels']:
  if row['name']=='decord'and prior['decord']is not None:continue
  metadata=wheel_metadata(fetch(row['pypi_url'],200000),row);raw=fetch(row['url'],row['bytes'])
  require(len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Downloaded wheel differs from independent primary SHA/bytes')
  path=context/'wheels'/row['filename'];exclusive(path,raw)
  external=publisher_notice(row,context)if'publisher_notice'in row else None
  notices=wheel_notices(path,external_notice_verified=external is not None)
  wheels[row['name']]={'version':row['version'],'bytes':row['bytes'],'sha256':row['sha256'],'primary_metadata':metadata,'packaged_notice_identities':notices,'packaged_notices_present':bool(notices),'external_publisher_notice':external}
 exclusive(context/'Dockerfile',dockerfile(pins,owner,prior['decord']is None))
 report.update(phase='offline_CPU_build',source_files=files,wheels=wheels,build_recipe_identity=identity(context/'Dockerfile'))
 require(source_binding(code,revision)==before,'Frozen builder source changed before build')
 command(['docker','build','--network','none','--pull=false','--force-rm','--memory','12g','--cpu-period','100000','--cpu-quota','400000','--tag',TARGET,'--file',str(context/'Dockerfile'),str(context)],1000)
 child=inspect_image(TARGET);require(child['Id']!=BASE and child['RootFS']['Layers'][:44]==parent['RootFS']['Layers'],'New child must preserve exact ordered parent rootfs')
 label=command(['docker','image','inspect',child['Id'],'--format','{{index .Config.Labels "'+LABEL+'"}}']).strip();require(label==owner,'Owned new child label required')
 after=probe(child['Id'],'wr-grounding-child-'+revision[:12],owner,True)
 expected=dict(prior['versions']);expected.update({'transformers':'4.53.3','tokenizers':'0.21.4','safetensors':'0.6.2','huggingface-hub':'0.36.2'})
 require(after[0]['versions']==expected and after[0]['decord']=='0.6.0','Only four explicit dependency replacements and optional pinned decord permitted')
 require(inspect_image(BASE)==parent and probe(BASE,'wr-grounding-recheck-'+revision[:12],owner)[0]==prior and source_binding(code,revision)==before,'Original parent/packages or frozen source changed')
 report.update(status='pass',phase='complete',child_image=child,child_probe=after,parent_unchanged_verified=True,source_rechecked_before_and_after=True,extension_import_verified=True,
  imported_source_is_new_pinned_child=True,original_grounding_image_parity_claimed=False,CUDA_execution_verified=False)

def main(argv=None):
 argparse.ArgumentParser(allow_abbrev=False).parse_args(argv)
 require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02','Exact Azure VM02 root CPU builder required')
 root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
 require(root==ROOT and Path(__file__).resolve()==code/'infra/frontend_grounding_build.py','Actual frozen builder file required')
 source_binding(code,revision)
 output=canonical(root/'results/frontend-grounding-build-v2');require(output.parent.is_dir()and not output.exists(),'Fresh owned Grounding build namespace required; no overwrite or retry')
 os.umask(0o077);output.mkdir(mode=0o700)
 report={'schema':'world_reward.frontend_grounding_build.v2','stage':'frontend_grounding_build','status':'fail','phase':'preflight','producer_revision':revision,
  'budget_seconds':BUDGET,'GPU_used':False,'model_loaded':False,'challenge_data_read':False,'private_validation_read':False,'credential_material_read':False,
  'build_network':'none','base_pull_performed':False,'replica_ready':False,'license_eligibility_verified':False,'training_overlap_verified':False}
 started=time.monotonic()
 def expired(*_):raise TimeoutError('Frozen Grounding build deadline')
 signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
 status=1
 try:build(code,revision,output,report);status=0
 except BaseException as error:
  report.update(status='fail',error=str(error)[:180]if isinstance(error,ValueError)else type(error).__name__)
  if'owner'in report:
   try:report['owned_cleanup']=cleanup(report['owner'])
   except BaseException:report['owned_cleanup']={'survivor_check_passed':False,'daemon_build_termination_proven':False}
 finally:
  signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started
  exclusive(output/'report.json',(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n').encode())
 summary={k:report[k]for k in('stage','status','phase','elapsed_seconds','replica_ready','license_eligibility_verified','training_overlap_verified')}
 if'error'in report:summary['error']=report['error']
 if'child_image'in report:summary['child_image_id']=report['child_image']['Id']
 summary['report_identity']=identity(output/'report.json');print(json.dumps(summary,sort_keys=True));return status

if __name__=='__main__':sys.exit(main())
