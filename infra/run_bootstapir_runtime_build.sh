#!/usr/bin/env bash
# CPU dependencies only; no TAPIR/model/data imports or GPU execution.
# Source closure: /configs/bootstapir_runtime_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_bootstapir_runtime_build/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_bootstapir_runtime_build.sh" && "$(uname -s)" == Linux ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 603s /usr/bin/python3 -I -B - <<'PYBUILD'
from pathlib import Path,PurePosixPath
import email,hashlib,json,os,re,stat,subprocess,sys,time,urllib.request,zipfile
ROOT=Path('/srv/scenesmith/world-reward');CONFIG='configs/bootstapir_runtime_pins.json';HELPER='infra/run_bootstapir_runtime_build.sh'
CONFIG_PIN={'bytes':4852,'sha256':'ea01a5593d63572d0890e48b49e31a757c5d4acc2442cd0bc32e420722e6c489'}
BASE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3';TARGET='world-reward/bootstapir:0.1'
SAFE_ENV={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','LANG':'C.UTF-8','DOCKER_HOST':'unix://'+str(ROOT/'docker.sock'),'DOCKER_BUILDKIT':'0'}
PROBE="""import importlib.metadata as m,json,sys,numpy as np,torch,tree
from einshape.torch import einshape
assert sys.version_info[:2]==(3,11) and torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.3'
assert not torch.cuda.is_initialized()
assert tree.map_structure(lambda x:x+1,{'x':[1,2]})=={'x':[2,3]}
x=torch.arange(6,dtype=torch.float32).reshape(2,3);y=einshape('ab->ba',x)
assert y.device.type=='cpu' and torch.equal(y,x.T) and not torch.cuda.is_initialized()
print(json.dumps({'versions':{n:m.version(n)for n in('dm-tree','einshape','absl-py','attrs','wrapt')},'python':'3.11','torch':torch.__version__,'numpy':np.__version__,'tree_cpu_verified':True,'einshape_torch_cpu_verified':True,'cuda_initialized':False}))
"""
def require(value,message):
 if not value:raise ValueError(message)
def canonical(path):
 require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical owned path required');return path
def identity(path,readonly=False):
 canonical(path);s=path.lstat();require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and 0<=s.st_size<=2_000_000 and(not readonly or not s.st_mode&0o222),'Bounded immutable regular file required');raw=path.read_bytes();a=path.lstat()
 require((s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(a.st_dev,a.st_ino,a.st_mode,a.st_size,a.st_mtime_ns,a.st_ctime_ns),'Source changed during hashing');return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def exclusive(path,raw):
 with path.open('xb')as stream:os.fchmod(stream.fileno(),0o400);stream.write(raw)
def binding(code,rev):
 canonical(code);require(code==ROOT/'jobs'/rev/'run_bootstapir_runtime_build/code'and re.fullmatch('[0-9a-f]{40}',rev),'Exact immutable builder namespace required');digest=hashlib.sha256()
 for name in('revision','source-sha256'):
  path=code.parent/name;identity(path);raw=path.read_bytes();require(raw==(rev+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Original dispatch markers required');digest.update(name.encode()+raw)
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);s=path.lstat();require(not s.st_mode&0o222 and(stat.S_ISDIR(s.st_mode)or stat.S_ISREG(s.st_mode)),'Complete readonly source closure required')
  if stat.S_ISREG(s.st_mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(identity(path,True)['sha256']))
 return {'closure_sha256':digest.hexdigest(),'helpers':{name:identity(code/name,True)for name in(HELPER,CONFIG)}}
def validate(pins):
 require(pins['schema']=='world_reward.bootstapir_runtime.pins.v1'and pins['base_image_id']==BASE and pins['target_image']==TARGET and pins['budget_seconds']==600,'Exact CPU child pins required')
 require([r['name']for r in pins['wheels']]==['dm-tree','einshape','absl-py','attrs','wrapt']and sum(r['bytes']for r in pins['wheels'])==493734,'Exact five dependency wheels required')
 for r in pins['wheels']:
  require(r['publication_date']<=pins['publication_cutoff'] and r['license']in('Apache-2.0','MIT','BSD-2-Clause')and r['url'].startswith('https://files.pythonhosted.org/packages/')and r['url'].endswith('/'+r['filename'])and re.fullmatch(r'[A-Za-z0-9_.-]+\.whl',r['filename'])and re.fullmatch('[0-9a-f]{64}',r['sha256'])and 0<r['bytes']<200000,'Exact bounded public wheel/license required')
 return pins
def fetch(record,deadline):
 require(time.monotonic()<deadline,'CPU build budget exceeded');url=record['url'];require('?'not in url and '@'not in url,'Unsigned public source only')
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=min(30,max(1,deadline-time.monotonic())))as response:
  require(response.status==200 and response.geturl()==url,'Exact public publisher URL required');raw=response.read(record['bytes']+1)
 require(len(raw)==record['bytes']and hashlib.sha256(raw).hexdigest()==record['sha256'],'Publisher byte count/SHA mismatch');return raw
def normalized(raw):return raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n').strip()
def wheel_notice(path,row,publisher):
 with zipfile.ZipFile(path)as archive:
  members=archive.infolist();names=[m.filename for m in members]
  require(len(names)==len(set(names))and len(names)<=1000 and sum(m.file_size for m in members)<=5_000_000,'Bounded unique wheel ZIP required')
  require(all(not PurePosixPath(n).is_absolute()and '..'not in PurePosixPath(n).parts and '\\'not in n for n in names)and all(not stat.S_ISLNK(m.external_attr>>16)for m in members),'Canonical regular wheel members required')
  metadata=[n for n in names if n.endswith('.dist-info/METADATA')];require(len(metadata)==1,'Unique wheel metadata required');m=email.message_from_bytes(archive.read(metadata[0]));require(m['Name'].lower().replace('_','-')==row['name']and m['Version']==row['version'],'Exact wheel package/version required')
  notices=[n for n in names if '.dist-info/'in n and PurePosixPath(n).name.upper()in('LICENSE','LICENSE.TXT','LICENSE.MD','COPYING')];require(notices,'Mandatory embedded OSI license required');result={};matched=False
  for n in notices:
   require(0<archive.getinfo(n).file_size<=65536,'Bounded embedded notice required');raw=archive.read(n);matched|=normalized(raw)==normalized(publisher);result[n]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  require(matched,'Embedded permission text differs from pinned publisher license');return result
def command(args,log,deadline):
 remaining=int(deadline-time.monotonic());require(remaining>0,'CPU build budget exceeded')
 with log.open('xb')as stream:os.fchmod(stream.fileno(),0o400);result=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5s',str(remaining)+'s',*args],env=SAFE_ENV,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining+7)
 require(result.returncode==0,'CPU runtime operation failed; inspect owned Azure build log')
def inspect(image,deadline,absent=False):
 result=subprocess.run(['docker','image','inspect',image,'--format','{{json .Id}} {{json .Architecture}} {{json .Os}} {{json .RootFS}}'],env=SAFE_ENV,capture_output=True,timeout=min(30,max(1,deadline-time.monotonic())))
 if absent:require(result.returncode==1 and b'No such image' in result.stderr,'Target tag must be absent, never retagged');return None
 require(result.returncode==0 and len(result.stdout)<16384,'Bounded actual image identity required');decoder=json.JSONDecoder();text=result.stdout.decode().strip();values=[]
 while text:value,end=decoder.raw_decode(text);values.append(value);text=text[end:].lstrip()
 require(len(values)==4 and re.fullmatch('sha256:[0-9a-f]{64}',values[0])and values[1:3]==['amd64','linux']and values[3]['Type']=='layers'and all(re.fullmatch('sha256:[0-9a-f]{64}',v)for v in values[3]['Layers']),'Actual Linux AMD64 layered image required');return {'image_id':values[0],'layers':values[3]['Layers']}
def recipe(pins):
 wheels=' '.join('/opt/bootstapir-runtime/wheels/'+r['filename']for r in pins['wheels'])
 return ('FROM '+BASE+'\nCOPY wheels /opt/bootstapir-runtime/wheels\nCOPY publisher-licenses /opt/bootstapir-runtime/publisher-licenses\nRUN /usr/bin/env -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp PIP_CONSTRAINT=/dev/null python -m pip --isolated install --no-index --no-deps --no-cache-dir --disable-pip-version-check --force-reinstall '+wheels+'\n').encode()
def run():
 code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION'];require(os.environ['WR_ROOT']==str(ROOT)and sys.platform=='linux','Owned Azure Linux only');before=binding(code,rev);require(before['helpers'][CONFIG]==CONFIG_PIN,'Frozen runtime config SHA required');pins=validate(json.loads((code/CONFIG).read_text()));out=canonical(ROOT/'results'/('bootstapir-runtime-build-'+rev));require(out.parent.is_dir()and not out.exists(),'Fresh exclusive runtime result namespace required');out.mkdir(mode=0o700)
 report={'stage':'bootstapir_runtime_build','status':'fail','phase':'preflight','producer_revision':rev,'source_helpers':before['helpers'],'pins':CONFIG_PIN,'model_or_data_read':False,'gpu_execution':False,'benchmark_ready':False,'quality_verified':False,'license_closure_verified':False};deadline=time.monotonic()+600;start=time.monotonic();name='world-reward-boots-build-'+rev[:12];owned=False
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(urllib.request.Request('http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01',headers={'Metadata':'true'}),timeout=5)as response:raw=response.read(65537)
  require(len(raw)<=65536,'Bounded VM metadata required');vm=json.loads(raw);require(vm.get('name')==pins['azure_vm_name']and vm.get('resourceGroupName','').lower()==pins['azure_resource_group'].lower(),'Actual VM02 required');report['azure_vm02_verified']=True
  base=inspect(BASE,deadline);require(base['image_id']==BASE and len(base['layers'])==44,'Original selected base identity/44 layers required');inspect(TARGET,deadline,True);report['base_image_id']=BASE
  existing=subprocess.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],env=SAFE_ENV,capture_output=True,timeout=15);require(existing.returncode==0 and not existing.stdout.strip(),'Owned CPU container namespace must be absent')
  report['phase']='public_dependencies';wheels=out/'wheels';licenses=out/'publisher-licenses';wheels.mkdir(mode=0o700);licenses.mkdir(mode=0o700);evidence={}
  for row in pins['wheels']:
   publisher=fetch(row['publisher_license'],deadline);exclusive(licenses/(row['name']+'.LICENSE'),publisher)
   if 'publisher_version_source'in row:version=fetch(row['publisher_version_source'],deadline);exclusive(licenses/(row['name']+'.setup.py'),version);require('1.0'in version.decode()and 'Apache'in version.decode(),'Pinned publisher release evidence required')
   path=wheels/row['filename'];exclusive(path,fetch(row,deadline));evidence[row['name']]={'wheel':identity(path),'publisher_license':identity(licenses/(row['name']+'.LICENSE')),'embedded_notices':wheel_notice(path,row,publisher),'license':row['license']}
  report['dependencies']=evidence;report['phase']='offline_build';exclusive(out/'Dockerfile',recipe(pins));exclusive(out/'.dockerignore',b'*\n!Dockerfile\n!wheels/\n!wheels/**\n!publisher-licenses/\n!publisher-licenses/**\n');log=out/'build.log'
  inspect(TARGET,deadline,True);require(binding(code,rev)==before,'Frozen source changed before build');command(['nice','-n','10','docker','build','--network','none','--memory','4g','--cpu-period','100000','--cpu-quota','200000','--label','world_reward_bootstapir_owner='+rev,'--tag',TARGET,'--file',str(out/'Dockerfile'),str(out)],log,deadline)
  child=inspect(TARGET,deadline);require(child['image_id']!=BASE and child['layers'][:44]==base['layers']and len(child['layers'])>44,'New child must preserve every ordered base layer');report.update(child_image_id=child['image_id'],base_layers=44,child_layers=len(child['layers']),ordered_rootfs_sha256=hashlib.sha256(json.dumps(child['layers'],separators=(',',':')).encode()).hexdigest(),platform={'os':'linux','architecture':'amd64'})
  report['phase']='cpu_import';probe=out/'cpu-import.log';owned=True
  command(['docker','run','--rm','--cidfile',str(out/'cpu-container.id'),'--name',name,'--label','world_reward_bootstapir_owner='+rev,'--network','none','--read-only','--memory','4g','--cpus','2','--tmpfs','/tmp:rw,noexec,nosuid,size=64m','--entrypoint','/usr/bin/env',child['image_id'],'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','python','-B','-c',PROBE],probe,deadline);owned=False
  require(probe.stat().st_size<=8192,'Bounded CPU import output required');native=json.loads(probe.read_text().splitlines()[-1]);require(native['versions']=={r['name']:r['version']for r in pins['wheels']}and native['tree_cpu_verified']is True and native['einshape_torch_cpu_verified']is True and native['cuda_initialized']is False,'Every native CPU dependency gate required');report['cpu_import']=native
  require(inspect(BASE,deadline)==base and inspect(TARGET,deadline)==child and binding(code,rev)==before,'Source/base/child changed after build');require(all(identity(wheels/r['filename'])==evidence[r['name']]['wheel']and identity(licenses/(r['name']+'.LICENSE'))==evidence[r['name']]['publisher_license']for r in pins['wheels']),'Dependency bytes changed after build');report.update(status='pass',phase='complete',install_verified=True,source_rehashed_after=True,wheel_source_equivalence_proven=False)
 except Exception:
  report['error']='CPU dependency/runtime gate failed; inspect owned Azure log'
 finally:
  try:
   cid=out/'cpu-container.id'
   if cid.exists():
    canonical(cid);cid.chmod(0o400);identity(cid);value=cid.read_text().strip();require(re.fullmatch('[0-9a-f]{64}',value),'Exact owned container ID required')
    query=['docker','ps','-aq','--no-trunc','--filter','id='+value,'--filter','name=^/'+name+'$','--filter','label=world_reward_bootstapir_owner='+rev]
    found=subprocess.run(query,env=SAFE_ENV,capture_output=True,timeout=10);ids=found.stdout.decode().split();require(found.returncode==0 and ids in([],[value]),'Bounded exact owned container query required')
    if owned and ids:require(subprocess.run(['docker','rm','--force',value],env=SAFE_ENV,capture_output=True,timeout=15).returncode==0,'Owned CPU cleanup failed')
    after=subprocess.run(query,env=SAFE_ENV,capture_output=True,timeout=10);require(after.returncode==0 and not after.stdout.strip(),'Owned CPU container survives');report['owned_container_survivors_verified_empty']=True
   require(binding(code,rev)==before,'Frozen source changed after runtime');report['source_rehashed_after']=True
  except Exception:report.update(status='fail',error='Post-run source/owned-container gate failed; inspect owned Azure log',source_rehashed_after=False)
  report['elapsed_seconds']=time.monotonic()-start
  for file in('build.log','cpu-import.log'):
   path=out/file
   if path.exists():report[file.replace('.','_')]=identity(path)
  exclusive(out/'report.json',(json.dumps(report,indent=2,allow_nan=False)+'\n').encode());print(json.dumps({'stage':report['stage'],'status':report['status'],'phase':report['phase'],'report':identity(out/'report.json'),'gpu_execution':False}));return 0 if report['status']=='pass'else 1
if __name__=='__main__':
 try:sys.exit(run())
 except Exception:print('BootsTAPIR CPU runtime preflight failed',file=sys.stderr);sys.exit(1)
PYBUILD
