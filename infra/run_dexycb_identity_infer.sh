#!/usr/bin/env bash
# Immutable source closure: /infra/dexycb_identity_infer.py
# Two independent native stages; only authenticated public/model mounts.
set +x
set -euo pipefail
ROOT="${WR_ROOT:?}";CODE="${WR_CODE:?}";REV="${WR_CODE_REVISION:?}"
[[ "$ROOT" == /srv/scenesmith/world-reward && "$REV" =~ ^[0-9a-f]{40}$ \
 && "$CODE" == "$ROOT/jobs/$REV/run_dexycb_identity_infer/code" \
 && "${BASH_SOURCE[0]}" == "$CODE/infra/run_dexycb_identity_infer.sh" \
 && "$(hostname)" == world-reward-ncc-h100-02 && "$(id -u)" == 0 ]] || exit 2
export DOCKER_HOST="unix://$ROOT/docker.sock"
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent DOCKER_HOST="$DOCKER_HOST" \
 WR_ROOT="$ROOT" WR_CODE="$CODE" WR_CODE_REVISION="$REV" PYTHONDONTWRITEBYTECODE=1 \
 /usr/bin/python3 -I -B - "$@" <<'PYWRAPPER'
import hashlib,json,os,re,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path('/srv/scenesmith/world-reward');code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION']
driver=code/'infra/dexycb_identity_infer.py';entry='run_dexycb_identity_infer'
images={'masks':'sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252',
        'tracks':'sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'}
budgets={'masks':300,'tracks':7200}
def require(value,message):
 if not value:raise ValueError(message)
def canonical(path):
 require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),
         'Canonical owned paths required');return path
def parse(args):
 values={};allowed={'stage','manifest-sha256','manifest-bytes','acquisition-report-sha256','acquisition-report-bytes',
                   'masks-report-sha256','masks-report-bytes'}
 require(len(args)%2==0,'Exact paired stage/pin arguments required')
 for flag,value in zip(args[::2],args[1::2]):
  require(flag.startswith('--')and flag[2:]in allowed and flag not in values,'Unknown/duplicate argument');values[flag]=value
 stage=values.get('--stage');require(stage in images,'Explicit masks or tracks stage required')
 expected={'--stage','--manifest-sha256','--manifest-bytes','--acquisition-report-sha256','--acquisition-report-bytes'}
 if stage=='tracks':expected.update(('--masks-report-sha256','--masks-report-bytes'))
 require(set(values)==expected,'Exact stage-specific independent pins required')
 for flag,value in values.items():
  if flag.endswith('-sha256'):require(re.fullmatch('[0-9a-f]{64}',value),'Explicit exact SHA256 pin required')
  if flag.endswith('-bytes'):require(re.fullmatch('[1-9][0-9]{0,8}',value)and int(value)<=16<<20,'Bounded positive byte pin required')
 return stage
def call(args,timeout=60,**kwargs):
 result=subprocess.run(args,timeout=timeout,check=False,**kwargs)
 require(result.returncode==0,'Bounded native control failed');return result
def host(args,mode):
 result=call(['/usr/bin/python3','-I','-B',str(driver),*args,mode],timeout=180,capture_output=True)
 return result.stdout.decode().strip()
def idle():
 require(not call(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],timeout=5,
                  capture_output=True).stdout.strip(),'Other GPU compute present')
def lock_state(path,fd=None):
 s=canonical(path).lstat();require(stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Existing cooperative lock required')
 if fd is not None:require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(s.st_dev,s.st_ino),'Lock inode changed')
 return s.st_dev,s.st_ino
def query(name):
 return call(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],timeout=5,capture_output=True).stdout.strip()
def cleanup(cidfile,name,image):
 require(cidfile.exists(),'Owned container ID receipt missing');s=canonical(cidfile).lstat()
 require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==0 and 0<s.st_size<=65,'Owned regular CID required')
 raw=cidfile.read_bytes()
 require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned container ID required');cid=raw.decode().strip()
 ids=call(['docker','ps','-aq','--no-trunc','--filter','id='+cid],timeout=5,capture_output=True).stdout.decode().split()
 require(ids in([],[cid]),'Container ID query ambiguous')
 if ids:
  found=call(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],timeout=5,capture_output=True).stdout.decode().strip()
  require(found==image+'|/'+name+'|'+entry+'|'+rev,'Foreign container cannot be removed')
  call(['docker','rm','-f',cid],timeout=15,capture_output=True)
 require(not query(name),'Owned container cleanup incomplete');cidfile.chmod(0o400)
def mounts(text):
 result=[];seen=set();inputs=ROOT/'validation/dexycb_identity_v1/inputs'
 require(text,'Authenticated readonly mount whitelist missing')
 for line in text.splitlines():
  pair=line.split('\t');require(len(pair)==2 and pair[0]==pair[1]and pair[0]not in seen,'Exact nonduplicate source mounts required')
  path=canonical(Path(pair[0]));require(path.exists()and ','not in pair[0]and '\n'not in pair[0],'Existing unambiguous mount required')
  require(not path.is_relative_to(ROOT/'data')and not path.is_relative_to(ROOT/'vendor')
          and 'eval_private'not in path.parts and path!=ROOT/'validation/dexycb_identity_v1/report.json',
          'No private acquisition/challenge mount allowed')
  seen.add(pair[0]);result.extend(('--mount','type=bind,src='+pair[0]+',dst='+pair[1]+',readonly'))
 require(str(code)in seen and str(inputs)in seen,'Actual source closure/public inputs required')
 return result
def run(args):
 stage=parse(args);canonical(code);require(code==ROOT/'jobs'/rev/entry/'code','Exact immutable entry required')
 out=canonical(ROOT/'validation/dexycb_identity_v1'/('identity_'+stage+'_'+rev))
 control=canonical(ROOT/'results'/('dexycb-identity-'+stage+'-'+rev));lock=ROOT/'jobs/.world-reward-h100.lock'
 require(not out.exists()and not control.exists()and out.parent.is_dir()and control.parent.is_dir(),'Fresh stage targets required')
 before=host(args,'--host-proof');require(re.fullmatch('[0-9a-f]{64}',before),'Exact complete host proof required')
 name='world-reward-dexycb-'+stage+'-'+rev[:12];require(not query(name),'Container name already occupied')
 initial=lock_state(lock);fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);os.dup2(fd,9)
 if fd!=9:os.close(fd)
 report={'stage':'dexycb_identity_host_wrapper','status':'fail','prediction_stage':stage,'producer_revision':rev,
         'image_id':images[stage],'host_proof_sha256':before,'budget_seconds':budgets[stage],
         'private_values_read':False,'quality_verified':False,'owned_cleanup_verified':False,'source_rehashed_after':False}
 owned=False;started=time.monotonic();failure=None
 try:
  require(lock_state(lock,9)==initial,'Lock changed before acquisition')
  call(['flock','--nonblock','9'],timeout=5,pass_fds=(9,),capture_output=True);idle()
  mount_args=mounts(host(args,'--mounts'));require(host(args,'--host-proof')==before,'Host proofs changed before native call')
  require(not out.exists()and not control.exists()and lock_state(lock,9)==initial,'Target/lock changed before native call')
  control.mkdir(mode=0o700);control.chmod(0o700);out.mkdir(mode=0o700);out.chmod(0o700);owned=True
  cid=out/'.container.cid';log=control/'native.log'
  command=['docker','run','--rm','--name',name,'--cidfile',str(cid),'--label','world-reward.job='+entry,
   '--label','world-reward.revision='+rev,'--gpus','all','--network','none','--user','0:0','--memory','32g','--cpus','4',
   '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,noexec,nosuid,size=512m',
   '--entrypoint','/usr/bin/env',*mount_args,'--mount','type=bind,src='+str(out)+',dst='+str(out),images[stage],
   '-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','XDG_CACHE_HOME=/tmp','PYTHONPATH='+str(code/'infra')+':'+str(code/'src'),
   'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+rev,'WR_IMAGE_ID='+images[stage],
   'WR_AZURE_VM02_VERIFIED=1','WR_HOST_PROOF_SHA256='+before,'PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1',
   'TRANSFORMERS_OFFLINE=1','WANDB_MODE=disabled','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
   '/opt/conda/bin/python','-B',str(driver),*args]
  with log.open('xb')as stream:
   os.fchmod(stream.fileno(),0o400);result=subprocess.run(command,timeout=budgets[stage]+30,check=False,stdout=stream,stderr=stream)
  report['native_exit_status']=result.returncode;require(result.returncode==0,'Native stage failed; immutable log retained')
  native=canonical(out/'report.json');require(native.is_file()and not native.stat().st_mode&0o222
            and native.stat().st_size<=2<<20,'Sealed native receipt required')
  raw=native.read_bytes();value=json.loads(raw)
  require(all(type(value.get(k))is type(v)and value[k]==v for k,v in {
   'schema':'world-reward-dexycb-identity-infer-v1','stage':stage,'status':'pass','phase':'complete','producer_revision':rev,
   'image_id':images[stage],'host_proof_sha256':before,'original_rehashed_after':True,'private_annotations_read':False,
   'quality_verified':False}.items()),'Native passing/source-bound receipt required')
  report['native_report_identity']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
 finally:
  try:
   if owned:cleanup(out/'.container.cid',name,images[stage]);report['owned_cleanup_verified']=True;idle()
   require(host(args,'--host-proof')==before and lock_state(lock,9)==initial,'Post-run source/public/lock proof differs')
   report['source_rehashed_after']=True
  except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
  finally:os.close(9)
  if owned:
   report.update(status='fail'if failure else'pass',elapsed_seconds=time.monotonic()-started)
   with(control/'report.json').open('x')as stream:
    os.fchmod(stream.fileno(),0o444);json.dump(report,stream,sort_keys=True);stream.write('\n')
 if failure:raise ValueError('DexYCB stage wrapper failed; inspect owned receipts')from None
 print(json.dumps({'stage':report['stage'],'status':'pass','prediction_stage':stage,'quality_verified':False}))
try:run(sys.argv[1:])
except Exception as exc:
 print(json.dumps({'stage':'dexycb_identity_host_wrapper','status':'fail','error_type':type(exc).__name__}));sys.exit(1)
PYWRAPPER
