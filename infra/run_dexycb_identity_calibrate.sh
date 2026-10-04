#!/usr/bin/env bash
# Source closure: /infra/dexycb_identity_calibrate.py /infra/dexycb_identity_infer.py
# CPU-only external validation; never mount challenge data or a private directory.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_dexycb_identity_calibrate/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_dexycb_identity_calibrate.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="unix://$ROOT/docker.sock" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$@" <<'PYWRAPPER'
import hashlib,json,os,re,signal,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path('/srv/scenesmith/world-reward');code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION']
sys.path[:0]=[str(code/'infra'),str(code/'src')];import dexycb_identity_infer as infer
binding=infer.binding;require=binding.require
ENTRY='run_dexycb_identity_calibrate';IMAGE=infer.IMAGES['masks'];BASE=ROOT/infer.BASE
FOLDERS={'public_features':'public_features_v1','private_calibration':'private_calibration_v1'}
HELPERS=(*infer.HELPERS,'infra/dexycb_identity_calibrate.py','src/world_reward/identity_calibration.py')
def parse(args):
 require(len(args)%2==0,'Exact paired independent pins required');values={}
 allowed={'--stage'}|{'--'+n+'-'+k for n in('manifest','acquisition','masks','tracks','features')
                    for k in(('bytes','sha256')if n in('manifest','acquisition')else('bytes','sha256','producer-revision','script-sha256'))}
 for flag,value in zip(args[::2],args[1::2]):
  require(flag in allowed and flag not in values,'Unknown or duplicate argument');values[flag]=value
 stage=values.get('--stage');require(stage in FOLDERS,'Explicit public/private stage required')
 roles=('manifest','acquisition','masks','tracks')+(('features',)if stage=='private_calibration'else())
 pins={};expected={'--stage'}
 for name in roles:
  keys=('bytes','sha256')if name in('manifest','acquisition')else('bytes','sha256','producer-revision','script-sha256')
  item={}
  for key in keys:
   flag='--'+name+'-'+key;expected.add(flag);value=values.get(flag,'')
   if key=='bytes':require(re.fullmatch('[1-9][0-9]{0,8}',value)and int(value)<=16<<20,'Bounded positive bytes required');value=int(value)
   else:require(re.fullmatch('[0-9a-f]{40}'if key=='producer-revision'else'[0-9a-f]{64}',value),'Exact independent source/hash pin required')
   item[key.replace('-','_')]=value
  pins[name]=item
 require(set(values)==expected,'Exact stage-specific pin roles required')
 require(pins['masks']['producer_revision']==pins['tracks']['producer_revision']
         and pins['masks']['script_sha256']==pins['tracks']['script_sha256'],'Same frozen native masks/tracks producer required')
 return stage,pins
def snapshot(folder,revision,entry,helpers):
 binding.canonical(folder);require(folder==ROOT/'jobs'/revision/entry/'code','Original immutable source namespace required')
 markers={}
 for name in('revision','source-sha256'):
  raw,_=binding.selected.read(folder.parent/name,100,readonly=False)
  require(raw==(revision+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Original dispatch markers required');markers[name]=raw.decode()
 digest=hashlib.sha256();pins={}
 for path in(folder,*sorted(folder.rglob('*'))):
  binding.canonical(path);s=path.lstat();require(not s.st_mode&0o222 and(stat.S_ISREG(s.st_mode)or stat.S_ISDIR(s.st_mode)),'Readonly whole source required')
  if path.is_file():
   _,pin=binding.selected.read(path,2_000_000,empty=True);name=str(path.relative_to(folder));pins[name]={k:pin[k]for k in('bytes','sha256')}
   digest.update(name.encode()+b'\0'+bytes.fromhex(pin['sha256']))
 require(set(helpers)<=set(pins),'Complete source closure required')
 return dict(markers=markers,closure_sha256=digest.hexdigest(),helpers={n:pins[n]for n in helpers})
def exact(receipt,fields):
 require(all(type(receipt.get(k))is type(v)and receipt[k]==v for k,v in fields.items()),'Actual immutable passing producer required')
def proof(stage,pins):
 own=snapshot(code,rev,ENTRY,HELPERS);require(Path(infer.__file__).resolve()==code/'infra/dexycb_identity_infer.py','Actual helper origin required')
 clips,rgb,manifest=infer.public_inputs(BASE/'inputs',pins['manifest'])
 infer.acquisition_proof(pins['acquisition'],pins['manifest'],manifest,code)
 acquisition=binding.pinned(BASE/'report.json',pins['acquisition'],16<<20)
 models={'masks':infer.frontend_proof(code,live=True)[1],'tracks':infer.boots_proof(code,live=True)}
 paths={p for method in('masks','tracks')for p in infer.host_mounts(method,code,pins[method]['producer_revision'])}
 frozen={str(p):v for p,v in rgb.items()};sources={};receipts={}
 for method in('masks','tracks'):
  item=pins[method];old=ROOT/'jobs'/item['producer_revision']/infer.ENTRY/'code'
  source=snapshot(old,item['producer_revision'],infer.ENTRY,infer.HELPERS);sources[method]=source
  require(all({k:binding.selected.read(code/n,2_000_000,empty=True)[1][k]for k in('bytes','sha256')}==p
              for n,p in source['helpers'].items()),'Native helper bytes differ')
  folder=infer.output_path(method,item['producer_revision']);pin={k:item[k]for k in('bytes','sha256')}
  receipt=binding.pinned(folder/'report.json',pin,2<<20);receipts[method]=pin
  exact(receipt,dict(schema='world-reward-dexycb-identity-infer-v1',stage=method,status='pass',phase='complete',
   producer_revision=item['producer_revision'],script_sha256=item['script_sha256'],source_binding=source,
   public_manifest=pins['manifest'],image_id=infer.IMAGES[method],original_rehashed_after=True,private_annotations_read=False))
  require(receipt.get('model_assets')==models[method]and item['script_sha256']==source['helpers'][infer.HELPERS[0]]['sha256'],'Original model/source proof differs')
  require(len(receipt.get('clips',[]))==12 and {p.name for p in folder.iterdir()}=={'report.json','.container.cid',*[f'clip_{i:03d}.npz'for i in range(12)]},'Exclusive all-twelve outputs required')
  paths.update((folder,old,old.parent/'revision',old.parent/'source-sha256'))
  for clip,row in zip(clips,receipt['clips']):
   exact(row,dict(clip_index=clip['clip_index'],frames=clip['frames'],file=f"clip_{clip['clip_index']:03d}.npz"))
   file=folder/row['file'];wanted={k:row[k]for k in('bytes','sha256')};require(binding.identity(file)==wanted,'Original native output differs');frozen[str(file)]=wanted
 paths.add(BASE/'report.json')
 acquired=ROOT/'jobs'/acquisition['producer_revision']/'run_dexycb_acquire/code';paths.add(acquired)
 for name in acquisition['source_before']:
  path=binding.canonical(Path(name))
  if not path.is_relative_to(acquired):paths.add(path)
 if stage=='private_calibration':
  item=pins['features'];folder=BASE/FOLDERS['public_features'];old=ROOT/'jobs'/item['producer_revision']/ENTRY/'code'
  source=snapshot(old,item['producer_revision'],ENTRY,HELPERS);wanted={k:item[k]for k in('bytes','sha256')}
  receipt=binding.pinned(folder/'report.json',wanted,2<<20)
  exact(receipt,dict(stage='public_features',status='pass',phase='complete',producer_revision=item['producer_revision'],
   script_sha256=item['script_sha256'],source_binding=source,pins={k:v for k,v in pins.items()if k!='features'},
   original_rehashed_after=True,private_annotations_read=False,all_twelve_features_frozen=True))
  require(item['script_sha256']==source['helpers']['infra/dexycb_identity_calibrate.py']['sha256'],'Frozen feature source differs')
  paths.update((folder,old,old.parent/'revision',old.parent/'source-sha256'));frozen[str(folder/'report.json')]=wanted
  require(len(receipt.get('clips',[]))==12 and {p.name for p in folder.iterdir()}=={'report.json',*[f'clip_{i:03d}.npz'for i in range(12)]},'Exactly twelve frozen features required')
  for clip,row in zip(clips,receipt['clips']):
   exact(row,dict(clip_index=clip['clip_index'],file=f"clip_{clip['clip_index']:03d}.npz"));p=folder/row['file'];wanted={k:row[k]for k in('bytes','sha256')};require(binding.identity(p)==wanted,'Frozen feature differs');frozen[str(p)]=wanted
  for clip in clips:
   prefix=clip['subject']+'/'+clip['sequence']
   for suffix in('/meta.yml','/'+infer.CAMERA+'/labels_000000.npz'):
    name=prefix+suffix;path=binding.canonical(BASE/'eval_private'/name);wanted=acquisition['retained_files'].get(name)
    require(binding.identity(path,64<<20)==wanted,'Pinned external initial annotation differs');paths.add(path);frozen[str(path)]=wanted
 paths={binding.canonical(p)for p in paths}
 require(not any(p==BASE/'eval_private'or p.is_relative_to(ROOT/'data')or p==ROOT/'vendor'for p in paths),'No broad private/challenge mounts')
 require(stage=='private_calibration'or not any(p.is_relative_to(BASE/'eval_private')for p in paths),'Public stage cannot read private bytes')
 value=dict(source=own,pins=pins,sources=sources,models=models,frozen=frozen)
 return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),sorted(paths)
def call(args,timeout=10,**kwargs):
 result=subprocess.run(args,timeout=timeout,check=False,**kwargs);require(result.returncode==0,'Bounded CPU control failed');return result
def query(name):return call(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],capture_output=True).stdout.strip()
def cleanup(cidfile,name):
 raw,_=binding.selected.read(cidfile,65,readonly=False);require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned CID required');cid=raw.decode().strip()
 ids=call(['docker','ps','-aq','--no-trunc','--filter','id='+cid],capture_output=True).stdout.decode().split();require(ids in([],[cid]),'Ambiguous cleanup query')
 if ids:
  actual=call(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],capture_output=True).stdout.decode().strip()
  require(actual==IMAGE+'|/'+name+'|'+ENTRY+'|'+rev,'Foreign container cannot be removed');call(['docker','rm','-f',cid],timeout=15,capture_output=True)
 require(not query(name),'Owned cleanup incomplete');cidfile.chmod(0o400)
def run(args):
 stage,pins=parse(args);out=binding.canonical(BASE/FOLDERS[stage]);control=binding.canonical(ROOT/'results'/('dexycb-identity-cpu-'+stage+'-'+rev))
 require(not out.exists()and not control.exists()and out.parent.is_dir()and control.parent.is_dir(),'Fresh CPU targets required')
 before,paths=proof(stage,pins);name='world-reward-dexycb-cpu-'+stage.replace('_','-')+'-'+rev[:12]
 require(not query(name),'Container name occupied');require(not out.exists()and not control.exists(),'Target changed during proof')
 control.mkdir(mode=0o700);control.chmod(0o700);out.mkdir(mode=0o700);out.chmod(0o700);cid=control/'.container.cid'
 report=dict(stage='dexycb_identity_cpu_host_wrapper',status='fail',prediction_stage=stage,producer_revision=rev,image_id=IMAGE,
  host_proof_sha256=before,budget_seconds=120,gpu_used=False,private_values_read=False,quality_verified=False,
  owned_cleanup_verified=False,source_rehashed_after=False,
  read_capability='DAC_READ_SEARCH'if stage=='private_calibration'else None,
  read_capability_basis='readonly_external_initial_annotations_acquired_as_UID1000_mode0400'if stage=='private_calibration'else None)
 failure=None;started=time.monotonic()
 def interrupted(*_):raise TimeoutError('CPU wrapper interrupted; only owned cleanup permitted')
 handlers={s:signal.signal(s,interrupted)for s in(signal.SIGINT,signal.SIGTERM)}
 try:
  mounts=[v for path in paths for v in('--mount','type=bind,src='+str(path)+',dst='+str(path)+',readonly')]
  command=['docker','run','--rm','--name',name,'--cidfile',str(cid),'--label','world-reward.job='+ENTRY,
   '--label','world-reward.revision='+rev,'--network','none','--user','0:0','--memory','32g','--cpus','4',
   '--read-only','--cap-drop','ALL',*(['--cap-add','DAC_READ_SEARCH']if stage=='private_calibration'else[]),
   '--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=512m',
   '--entrypoint','/usr/bin/env',*mounts,'--mount','type=bind,src='+str(out)+',dst='+str(out),IMAGE,'-i',
   'PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','PYTHONPATH='+str(code/'infra')+':'+str(code/'src'),
   'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+rev,'WR_CPU_IMAGE_ID='+IMAGE,
   'WR_DEXYCB_CPU_OUTPUT_RESERVED=1','WR_HOST_PROOF_SHA256='+before,'CUDA_VISIBLE_DEVICES=-1',
   'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','OMP_NUM_THREADS=4',
   'OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','/opt/conda/bin/python','-B',str(code/'infra/dexycb_identity_calibrate.py'),*args]
  with(control/'native.log').open('xb')as log:
   os.fchmod(log.fileno(),0o400);result=subprocess.run(command,timeout=150,check=False,stdout=log,stderr=log)
  report['native_exit_status']=result.returncode;require(result.returncode==0,'CPU stage failed; immutable log retained')
  pin=binding.identity(out/'report.json',2<<20);native=binding.pinned(out/'report.json',pin,2<<20)
  own=snapshot(code,rev,ENTRY,HELPERS)
  exact(native,dict(stage=stage,status='pass',phase='complete',producer_revision=rev,pins=pins,source_binding=own,
   script_sha256=own['helpers']['infra/dexycb_identity_calibrate.py']['sha256'],original_rehashed_after=True,
   device='cpu',network='none',budget_seconds=120,adoption=False,private_annotations_read=stage=='private_calibration'))
  report['native_report_identity']=pin
 except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
 finally:
  for s,handler in handlers.items():signal.signal(s,handler)
  try:
   cleanup(cid,name);report['owned_cleanup_verified']=True
   require(proof(stage,pins)[0]==before,'Post-run source/public/model proofs differ');report['source_rehashed_after']=True
  except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
  report.update(status='fail'if failure else'pass',elapsed_seconds=time.monotonic()-started)
  with(control/'report.json').open('x')as stream:
   os.fchmod(stream.fileno(),0o444);json.dump(report,stream,sort_keys=True);stream.write('\n')
 if failure:raise ValueError('CPU stage failed; inspect owned receipts')from None
 print(json.dumps({k:report[k]for k in('stage','status','prediction_stage','gpu_used','quality_verified')}))
try:run(sys.argv[1:])
except Exception as exc:print(json.dumps(dict(stage='dexycb_identity_cpu_host_wrapper',status='fail',error_type=type(exc).__name__)));sys.exit(1)
PYWRAPPER
