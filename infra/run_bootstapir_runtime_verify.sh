#!/usr/bin/env bash
# Independent CPU source-native gate; original failed build remains immutable.
# Source closure: /configs/bootstapir_runtime_verify_pins.json
# /configs/robotap_boots_protocol.json /configs/robotap_boots_acquisition_pins.json
set +x
set -euo pipefail
[[ $# == 0 ]] || exit 2
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_bootstapir_runtime_verify/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_bootstapir_runtime_verify.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 && "$(uname -s)" == Linux ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin HOME=/nonexistent LANG=C.UTF-8 \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/timeout --signal=TERM --kill-after=10s 123s /usr/bin/python3 -I -B - <<'PYVERIFY'
from pathlib import Path
import hashlib,json,os,re,stat,subprocess,sys,time
ROOT=Path('/srv/scenesmith/world-reward');CONFIG='configs/bootstapir_runtime_verify_pins.json';HELPER='infra/run_bootstapir_runtime_verify.sh'
PIN={'bytes':2097,'sha256':'7ec0b87fbf62426b2df4fe9f2dbc788b8aa83ddfe41ad2a0b871299a376015c2'}
SAFE_ENV={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','LANG':'C.UTF-8','DOCKER_HOST':'unix://'+str(ROOT/'docker.sock')}
PROBE="""import hashlib,importlib.metadata as m,json,sys
from pathlib import Path
source=Path('/opt/bootstapir-source');expected=json.loads(sys.argv[1])
def verify():
 for name,row in expected.items():
  p=source/name;raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
verify();sys.path.insert(0,str(source))
import numpy as np,torch,tree
from tapnet.torch import utils,nets,tapir_model
assert sys.version_info[:2]==(3,11) and torch.__version__=='2.5.1+cu124' and np.__version__=='1.26.3' and not torch.cuda.is_initialized()
assert tree.map_structure(lambda x:x+1,{'x':[1,2]})=={'x':[2,3]}
x=torch.arange(6,dtype=torch.float32).reshape(2,3);y=utils.einshape('ab->ba',x)
assert y.device.type=='cpu' and torch.equal(y,x.T)
z=utils.bilinear(torch.ones((1,1,2,2,1),dtype=torch.float32),(3,3));assert z.shape==(1,1,3,3,1) and torch.equal(z,torch.ones_like(z))
assert tapir_model.TAPIR.__module__=='tapnet.torch.tapir_model' and nets.ResNet.__module__=='tapnet.torch.nets'
for module in(utils,nets,tapir_model):assert Path(module.__file__).resolve()==source/('tapnet/torch/'+module.__name__.split('.')[-1]+'.py')
assert not torch.cuda.is_initialized();verify()
print(json.dumps({'versions':{n:m.version(n)for n in('dm-tree','einshape','absl-py','attrs','wrapt')},'python':'3.11','torch':torch.__version__,'numpy':np.__version__,'tree_cpu_verified':True,'source_native_einshape_cpu_verified':True,'native_bilinear_cpu_verified':True,'native_tapir_modules_imported':True,'models_instantiated':False,'cuda_initialized':False}))
"""
def require(value,message):
 if not value:raise ValueError(message)
def canonical(path):
 require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),'Canonical existing path required');return path
def identity(path,readonly=True):
 canonical(path);before=path.lstat();require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and 0<=before.st_size<=2_000_000 and(not readonly or not before.st_mode&0o222),'Bounded immutable regular bytes required');raw=path.read_bytes();after=path.lstat()
 require(all(getattr(before,k)==getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')),'Source changed while hashing');return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def pinned(path,row):
 require(identity(path)=={k:row[k]for k in('bytes','sha256')},'Independent immutable SHA/bytes mismatch');return path
def exclusive(path,raw):
 with path.open('xb')as stream:os.fchmod(stream.fileno(),0o400);stream.write(raw)
def strict_json(raw):
 def pairs(rows):
  result={}
  for key,value in rows:require(key not in result,'Duplicate receipt key forbidden');result[key]=value
  return result
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite receipt forbidden')))
def binding(code,rev):
 canonical(code);require(code==ROOT/'jobs'/rev/'run_bootstapir_runtime_verify/code'and re.fullmatch('[0-9a-f]{40}',rev),'Actual immutable verification namespace required');digest=hashlib.sha256()
 for name in('revision','source-sha256'):
  path=code.parent/name;identity(path,False);raw=path.read_bytes();require(raw==(rev+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Original dispatch marker required');digest.update(name.encode()+raw)
 for path in(code,*sorted(code.rglob('*'))):
  canonical(path);mode=path.lstat().st_mode;require(not mode&0o222 and(stat.S_ISDIR(mode)or stat.S_ISREG(mode)),'Complete readonly source closure required')
  if stat.S_ISREG(mode):digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(identity(path)['sha256']))
 names=(HELPER,CONFIG,'configs/robotap_boots_protocol.json','configs/robotap_boots_acquisition_pins.json')
 return {'closure_sha256':digest.hexdigest(),'helpers':{name:identity(code/name)for name in names}}
def originals(root,code,pins):
 original=root/pins['build_output'];bound={};reportpath=pinned(original/'report.json',pins['original_build_report']);bound[reportpath]=identity(reportpath);report=strict_json(reportpath.read_bytes())
 require(report.get('stage')=='bootstapir_runtime_build'and report.get('status')=='fail'and report.get('phase')=='cpu_import'and report.get('producer_revision')==pins['original_build_report']['producer_revision']and report.get('source_rehashed_after')is True and report.get('model_or_data_read')is False and report.get('gpu_execution')is False,'Original pre-model CPU failure must remain FAIL')
 require(report.get('child_image_id')==pins['child_image_id']and report.get('ordered_rootfs_sha256')==pins['ordered_rootfs_sha256']and report.get('base_layers')==44 and report.get('child_layers')==47 and report.get('source_helpers')==pins['original_build_helpers'],'Actual original child/source identity required')
 for file,key in(('cpu-import.log','original_cpu_import_log'),('build.log','original_build_log')):
  path=pinned(original/file,pins[key]);bound[path]=identity(path)
 buildcode=root/'jobs'/pins['original_build_report']['producer_revision']/'run_bootstapir_runtime_build/code'
 for name,row in pins['original_build_helpers'].items():path=pinned(buildcode/name,row);bound[path]=identity(path)
 marker=buildcode.parent/'revision';identity(marker,False);require(marker.read_bytes()==(pins['original_build_report']['producer_revision']+'\n').encode(),'Actual original build marker required');bound[marker]=identity(marker,False)
 dependencies=strict_json((buildcode/'configs/bootstapir_runtime_pins.json').read_bytes())
 for row in dependencies['wheels']:
  wheel=pinned(original/'wheels'/row['filename'],row);publisher=pinned(original/'publisher-licenses'/(row['name']+'.LICENSE'),row['publisher_license']);bound[wheel]=identity(wheel);bound[publisher]=identity(publisher)
  evidence=report.get('dependencies',{}).get(row['name'],{});require(evidence.get('wheel')==identity(wheel)and evidence.get('publisher_license')==identity(publisher)and evidence.get('license')==row['license']and evidence.get('embedded_notices'),'Every originally checked dependency and licence required')
 for key in('source_protocol','acquisition_pins'):path=pinned(code/pins[key]['path'],pins[key]);bound[path]=identity(path)
 protocol=strict_json((code/pins['source_protocol']['path']).read_bytes());acq=strict_json((code/pins['acquisition_pins']['path']).read_bytes());acqpath=pinned(root/protocol['namespace']/'report.json',acq['report']);bound[acqpath]=identity(acqpath);acqreport=strict_json(acqpath.read_bytes())
 require(acqreport.get('status')=='pass'and acqreport.get('producer_revision')==acq['report']['producer_revision']and acqreport.get('script_sha256')==acq['report']['script_sha256'],'Actual source acquisition PASS required before import')
 source=root/protocol['namespace']/'assets/tapnet_source';expected={name:{k:row[k]for k in('bytes','sha256')}for name,row in protocol['source']['files'].items()}
 require(len(expected)==7 and protocol['source']['revision']=='730cda1c730877cfedbe01bf87fb1cadb78a565d','Exact seven native source files required')
 for name,row in expected.items():path=pinned(source/name,row);bound[path]=identity(path)
 return bound,source,expected,{r['name']:r['version']for r in dependencies['wheels']}
def control(args,deadline):
 result=subprocess.run(args,env=SAFE_ENV,capture_output=True,timeout=min(15,max(1,deadline-time.monotonic())));require(result.returncode==0 and len(result.stdout)<=16384,'Bounded runtime metadata control failed');return result.stdout
def image(value,deadline):
 raw=control(['docker','image','inspect',value,'--format','{{json .Id}} {{json .Architecture}} {{json .Os}} {{json .RootFS}}'],deadline).decode().strip();decoder=json.JSONDecoder();values=[]
 while raw:value,end=decoder.raw_decode(raw);values.append(value);raw=raw[end:].lstrip()
 require(len(values)==4 and values[1:3]==['amd64','linux']and values[3]['Type']=='layers','Actual Linux AMD64 child required');return {'image_id':values[0],'layers':values[3]['Layers']}
def run():
 code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION'];before=binding(code,rev);pins=strict_json(pinned(code/CONFIG,PIN).read_bytes());require(pins['schema']=='world_reward.bootstapir_runtime_verify.pins.v1'and pins['budget_seconds']==120,'Frozen new verification contract required');out=canonical(ROOT/'results'/('bootstapir-runtime-verify-'+rev));require(out.parent.is_dir()and not out.exists(),'Fresh CPU verification namespace required');out.mkdir(mode=0o700);deadline=time.monotonic()+120;start=time.monotonic();name='world-reward-boots-verify-'+rev[:12];owned=False;bound={}
 report={'stage':'bootstapir_runtime_verify','status':'fail','phase':'original_provenance','producer_revision':rev,'source_helpers':before['helpers'],'pins':PIN,'original_build_status':'fail','original_build_failure_preserved':False,'gpu_execution':False,'checkpoint_read':False,'rgb_or_labels_read':False,'quality_verified':False,'benchmark_ready':False,'license_closure_verified':False}
 try:
  bound,source,expected,versions=originals(ROOT,code,pins);base=image(pins['base_image_id'],deadline);child=image(pins['child_image_id'],deadline);require(base['image_id']==pins['base_image_id']and child['image_id']==pins['child_image_id']and image(pins['target_image'],deadline)==child and len(base['layers'])==44 and len(child['layers'])==47 and child['layers'][:44]==base['layers']and hashlib.sha256(json.dumps(child['layers'],separators=(',',':')).encode()).hexdigest()==pins['ordered_rootfs_sha256'],'Original full child/base layer identity required')
  require(not control(['docker','ps','-aq','--filter','name=^/'+name+'$'],deadline).strip(),'Owned CPU namespace occupied');report.update(child_image_id=child['image_id'],native_source_revision='730cda1c730877cfedbe01bf87fb1cadb78a565d',native_sources=expected,phase='cpu_native_import')
  mounts=[]
  for file in sorted(expected):mounts+=['--mount','type=bind,src='+str(source/file)+',dst=/opt/bootstapir-source/'+file+',readonly']
  args=['docker','run','--rm','--name',name,'--cidfile',str(out/'container.id'),'--label','world_reward_bootstapir_verify_owner='+rev,'--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','4g','--cpus','2','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',*mounts,'--entrypoint','/usr/bin/env',child['image_id'],'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','python','-B','-c',PROBE,json.dumps(expected,separators=(',',':'))];log=out/'cpu-native-import.log';owned=True
  with log.open('xb')as stream:os.fchmod(stream.fileno(),0o400);result=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=5s','100s',*args],env=SAFE_ENV,stdout=stream,stderr=subprocess.STDOUT,timeout=107)
  require(result.returncode==0 and log.stat().st_size<=8192,'Native CPU gate failed; inspect owned Azure log');native=strict_json(log.read_text().splitlines()[-1]);require(native['versions']==versions and all(native[k]is True for k in('tree_cpu_verified','source_native_einshape_cpu_verified','native_bilinear_cpu_verified','native_tapir_modules_imported'))and native['models_instantiated']is False and native['cuda_initialized']is False,'Native CPU operators/modules required');owned=False;report['cpu_import']=native
  require(image(pins['base_image_id'],deadline)==base and image(pins['child_image_id'],deadline)==child and image(pins['target_image'],deadline)==child,'Image changed after verification');report.update(status='pass',phase='complete')
 except Exception:report['error']='Source-native CPU verification failed; inspect owned Azure log'
 finally:
  try:
   cid=out/'container.id'
   if cid.exists():
    canonical(cid);cid.chmod(0o400);identity(cid);value=cid.read_text().strip();require(re.fullmatch('[0-9a-f]{64}',value),'Owned container ID required');query=['docker','ps','-aq','--no-trunc','--filter','id='+value,'--filter','name=^/'+name+'$','--filter','label=world_reward_bootstapir_verify_owner='+rev];ids=control(query,deadline).decode().split();require(ids in([],[value]),'Exact owned container query required')
    if owned and ids:control(['docker','rm','--force',value],deadline)
    require(not control(query,deadline).strip(),'Owned CPU survivor remains')
   require(binding(code,rev)==before and all(identity(path,path.name!='revision')==row for path,row in bound.items()),'Immutable source/original receipts changed after CPU gate');report['source_rehashed_after']=True;report['original_build_failure_preserved']=bool(bound)
  except Exception:report.update(status='fail',source_rehashed_after=False,error='Post-run immutable/source/owned-container gate failed')
  report['elapsed_seconds']=time.monotonic()-start
  if (out/'cpu-native-import.log').exists():report['cpu_native_import_log']=identity(out/'cpu-native-import.log')
  exclusive(out/'report.json',(json.dumps(report,indent=2,allow_nan=False)+'\n').encode());print(json.dumps({'stage':report['stage'],'status':report['status'],'phase':report['phase'],'report':identity(out/'report.json'),'original_build_status':'fail','gpu_execution':False}));return 0 if report['status']=='pass'else 1
if __name__=='__main__':
 try:sys.exit(run())
 except Exception:print('BootsTAPIR CPU verification preflight failed',file=sys.stderr);sys.exit(1)
PYVERIFY
