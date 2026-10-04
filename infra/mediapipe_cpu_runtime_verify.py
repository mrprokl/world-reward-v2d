"""CPU-only offline runtime construction/graph loading; never image inference."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

ROOT=Path('/srv/scenesmith/world-reward')
ENTRY='run_mediapipe_cpu_runtime_verify'
BASE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
TARGET='world-reward-mediapipe-cpu-v1'
MANIFEST='configs/mediapipe_cpu_dependencies_v1.json'
MANIFEST_PIN=dict(bytes=59268,sha256='efb697fd458156a93520db3add265b75b3506f84e62b681920ce6f7c75375c81')
DEP_PINS='configs/mediapipe_cpu_dependencies_acquire_pins.json'
MP_PINS='configs/mediapipe_hands_acquire_pins.json'
DOWNLOAD_PINS='configs/mediapipe_cpu_dependencies_download_pins.json'
HELPERS=('infra/mediapipe_cpu_runtime_verify.py','infra/run_mediapipe_cpu_runtime_verify.sh',
 'infra/Dockerfile.mediapipe_cpu','infra/mediapipe_cpu_dependencies_acquire.py','infra/mediapipe_hands_acquire.py',
 MANIFEST,DEP_PINS,MP_PINS,DOWNLOAD_PINS)
VENV=Path('/opt/world-reward-mediapipe')


def require(value,message):
    if not value:raise ValueError(message)


def canonical(path):
    path=Path(path);require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),
                          'Canonical immutable/control path required');return path


def identity(path,maximum=1_000_000_000,*,readonly=True,empty=False):
    p=canonical(path);before=p.lstat()
    require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and(not readonly or not before.st_mode&0o222)
            and (0 if empty else 1)<=before.st_size<=maximum,'Bounded single-link regular artifact required')
    digest=hashlib.sha256()
    with p.open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    after=p.lstat()
    require(all(getattr(before,k)==getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_uid','st_gid')), 'Artifact changed while hashing')
    return dict(bytes=before.st_size,sha256=digest.hexdigest())


def strict(raw):
    def pairs(rows):
        value={}
        for k,v in rows:require(k not in value,'Duplicate receipt/config field');value[k]=v
        return value
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError('Nonfinite metadata')))


def pinned(path,pin,maximum=1<<20):
    require(type(pin)is dict and set(pin)=={'bytes','sha256'}and type(pin['bytes'])is int
            and 0<pin['bytes']<=maximum and re.fullmatch('[0-9a-f]{64}',str(pin['sha256'])),'Exact independent SHA/byte pin required')
    require(identity(path,maximum)==pin,'Independent artifact identity differs');return strict(Path(path).read_bytes())


def source(root,code,revision,entry,helpers):
    require(canonical(code)==root/'jobs'/revision/entry/'code'and re.fullmatch('[0-9a-f]{40}',revision),'Original immutable source namespace required')
    markers={n:identity(code.parent/n,100)for n in('revision','source-sha256')}
    require((code.parent/'revision').read_bytes()==(revision+'\n').encode()
            and re.fullmatch(b'[0-9a-f]{64}\n',(code.parent/'source-sha256').read_bytes()),'Actual dispatch markers required')
    entries={}
    for p in(code,*sorted(code.rglob('*'))):
        canonical(p);s=p.lstat();require(not s.st_mode&0o222 and(stat.S_ISDIR(s.st_mode)or stat.S_ISREG(s.st_mode)),'Readonly complete source closure required')
        entries[str(p.relative_to(code))]={'directory':True}if p.is_dir()else identity(p,2_000_000,empty=True)
    require(set(helpers)<=set(entries),'Required immutable source/config closure missing')
    return dict(producer_revision=revision,markers=markers,entries=len(entries),helpers={n:entries[n]for n in helpers},
                closure_sha256=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest())


def acquire_pins(code,name,schema):
    path=code/name;pin=identity(path,16<<10);value=strict(path.read_bytes())
    require(type(value)is dict and set(value)=={'schema','producer_revision','report','helper'}and value['schema']==schema
            and re.fullmatch('[0-9a-f]{40}',str(value['producer_revision'])),'Independent acquisition pin schema differs')
    for key in('report','helper'):
        row=value[key];require(type(row)is dict and set(row)=={'bytes','sha256'}and type(row['bytes'])is int
                              and 0<row['bytes']<=4<<20 and re.fullmatch('[0-9a-f]{64}',str(row['sha256'])),'Independent acquisition identity missing')
    return value,pin


def prerequisites(root,code):
    mp_pins,mp_pin=acquire_pins(code,MP_PINS,'world_reward.mediapipe_hands_acquire_pins.v1')
    dp,dp_pin=acquire_pins(code,DEP_PINS,'world_reward.mediapipe_cpu_dependencies_acquire_pins.v1')
    require(identity(code/'infra/mediapipe_hands_acquire.py',2_000_000)==mp_pins['helper']
            and identity(code/'infra/mediapipe_cpu_dependencies_acquire.py',2_000_000)==dp['helper'],'Independent imported acquisition helper differs')
    sys.path.insert(0,str(code/'infra'));dep=importlib.import_module('mediapipe_cpu_dependencies_acquire');mp=dep.mp
    require(Path(dep.__file__).resolve()==code/'infra/mediapipe_cpu_dependencies_acquire.py'
            and Path(mp.__file__).resolve()==code/'infra/mediapipe_hands_acquire.py','Loaded acquisition helper origin differs')
    prior=dep.verify_prior(root,code/MP_PINS);manifest=dep.load_manifest(code/MANIFEST);downloads=dep.verify_downloads(root,code)
    require(dep.DOWNLOAD_PINS==DOWNLOAD_PINS,'Exact independently pinned download lineage required')
    old=root/'jobs'/dp['producer_revision']/dep.JOB/'code';snapshot=dep.dependency_source(root,old,dp['producer_revision'],verify_acquired=True)
    require(snapshot['helpers']['infra/mediapipe_cpu_dependencies_acquire.py']==dp['helper'],'Original dependency source differs')
    receipt=pinned(root/dep.RESULT/'report.json',dp['report'],4<<20)
    expected=dict(schema='world_reward.mediapipe_cpu_dependencies_acquisition.v1',stage='mediapipe_cpu_dependencies_acquisition',
     status='pass',source_binding=snapshot,prior_acquisition=prior,manifest_identity=MANIFEST_PIN,
     mode='verify_acquired_bytes_and_notices',prior_verified_downloads=downloads,network_used=False,new_downloads=0,stage_adoption=False,
     source_rehashed_after=True,prior_rehashed_after=True,artifacts_rehashed_after=True,downloads_rehashed_after=True,owned_partials_removed=True,
     models_loaded=False,packages_installed=False,gpu_used=False,dataset_read=False,private_values_read=False,
     license_eligibility_verified=False,training_overlap_verified=False,challenge_overlap_verified=False)
    require(all(type(receipt.get(k))is type(v)and receipt[k]==v for k,v in expected.items()),'Actual complete dependency acquisition PASS required')
    require(type(receipt.get('elapsed_seconds'))in(int,float)and 0<receipt['elapsed_seconds']<=300,'Original acquisition budget differs')
    rows=manifest['packages'];wanted=dep.assets(rows);artifacts=receipt.get('artifacts',[])
    require(len(artifacts)==len(wanted)and len(receipt.get('wheels',[]))==26,'Complete exact dependency inventory required')
    expected_files={r['folder']+'/'+r['name']:r for r in wanted};records={r['file']:r for r in artifacts}
    require(set(records)==set(expected_files)and len(records)==len(artifacts),'No foreign dependency artifact allowed')
    frozen={code/MP_PINS:mp_pin,code/DEP_PINS:dp_pin,code/DOWNLOAD_PINS:downloads['pins_identity'],
            code/MANIFEST:MANIFEST_PIN,root/dep.RESULT/'report.json':dp['report'],root/dep.DOWNLOAD_RESULT/'report.json':downloads['report_identity']}
    for name,row in records.items():
        expected=expected_files[name];pin={k:row[k]for k in('bytes','sha256')}
        require(pin=={k:expected[k]for k in('bytes','sha256')}and row.get('publisher_sha256_verified')is True
                and identity(root/name)==pin,'Original exact dependency/metadata bytes differ');frozen[root/name]=pin
    require({p.name for p in(root/dep.EVIDENCE).iterdir()}=={Path(n).name for n in expected_files},'Exclusive dependency directory required')
    for name,pin in prior['artifacts'].items():frozen[root/name]=pin
    frozen[root/mp.RESULT/'report.json']=mp_pins['report'];wheels={};notices={}
    wheel_records={r['name']:r for r in receipt['wheels']};require(set(wheel_records)=={r['name']for r in rows},'All26 wheel qualifications required')
    for row in rows:
        path=root/(mp.EVIDENCE if row['name']=='mediapipe'else dep.EVIDENCE)/row['filename'];record=wheel_records[row['name']]
        require(record['file']==str(path)and record['bytes']==row['bytes']and record['sha256']==row['sha256']
                and record.get('unresolved_license_files')==[]and record.get('license_metadata_files'),'Declared wheel notices/identity required')
        require(identity(path)=={k:row[k]for k in('bytes','sha256')},'Wheel bytes differ');wheels[path]=row;frozen[path]={k:row[k]for k in('bytes','sha256')}
        notices[row['name']]=record['license_metadata_files']
    task=root/mp.WEIGHTS/'hand_landmarker.task';require(task in frozen,'Original task acquisition pin missing')
    return dict(manifest=manifest,wheels=wheels,task=task,task_pin=frozen[task],frozen=frozen,
                prior=prior,verified_downloads=downloads,dependency_source=snapshot,dependency_pins=dp,notices=notices)


def write(path,raw,mode=0o400):
    with Path(path).open('xb')as stream:os.fchmod(stream.fileno(),mode);stream.write(raw);stream.flush();os.fsync(stream.fileno())


def make_context(context,code,proof):
    require(canonical(context).is_dir()and not tuple(context.iterdir()),'Exclusive empty build context required');(context/'wheels').mkdir(mode=0o700)
    for path,row in proof['wheels'].items():
        require(identity(path)=={k:row[k]for k in('bytes','sha256')},'Wheel changed before CPU copy')
        destination=context/'wheels'/row['filename']
        with path.open('rb')as source_file,destination.open('xb')as target:
            os.fchmod(target.fileno(),0o400);shutil.copyfileobj(source_file,target,1<<20);target.flush();os.fsync(target.fileno())
        require(identity(destination)=={k:row[k]for k in('bytes','sha256')}and identity(path)==identity(destination),'Copy is not original pinned wheel')
    write(context/'Dockerfile',(code/'infra/Dockerfile.mediapipe_cpu').read_bytes())
    write(context/'manifest.json',(code/MANIFEST).read_bytes());write(context/'wheel-notices.json',(json.dumps(proof['notices'],sort_keys=True)+'\n').encode())


def control(args,timeout=15):
    result=subprocess.run(args,capture_output=True,timeout=timeout,check=False)
    require(result.returncode==0 and len(result.stdout)<=32<<10,'Bounded image/container control failed');return result.stdout


def image(value):
    raw=control(['docker','image','inspect',value,'--format','{{json .Id}} {{json .Architecture}} {{json .Os}} {{json .RootFS}} {{json .Config.Labels}}']).decode()
    decoder=json.JSONDecoder();values=[]
    while raw.strip():v,end=decoder.raw_decode(raw.lstrip());values.append(v);raw=raw.lstrip()[end:]
    require(len(values)==5 and re.fullmatch('sha256:[0-9a-f]{64}',values[0])and values[1:3]==['amd64','linux']
            and values[3]['Type']=='layers'and all(re.fullmatch('sha256:[0-9a-f]{64}',s)for s in values[3]['Layers']), 'Actual LinuxAMD64 image projection required')
    labels={k:v for k,v in(values[4]or{}).items()if k.startswith('world_reward.mediapipe_cpu.')}
    return dict(Id=values[0],Architecture=values[1],Os=values[2],RootFS=values[3],Labels=labels)


def qualify_child(base,child,revision,binding):
    require(base['Id']==BASE and child['RootFS']['Layers'][:len(base['RootFS']['Layers'])]==base['RootFS']['Layers']
            and len(child['RootFS']['Layers'])>len(base['RootFS']['Layers']), 'Original base image/layers must remain unchanged')
    labels={'revision':revision,'source':binding['closure_sha256'],'manifest':MANIFEST_PIN['sha256']}
    require(all(child['Labels'].get('world_reward.mediapipe_cpu.'+k)==v for k,v in labels.items()),'New child source ownership differs')


def cleanup(cidfile,name,child,revision):
    raw=cidfile.read_bytes()if cidfile.exists()else b''
    require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Owned container ID receipt missing');cid=raw.decode().strip()
    identity(cidfile,65,readonly=False)
    ids=control(['docker','ps','-aq','--no-trunc','--filter','id='+cid]).decode().split();require(ids in([],[cid]),'Owned container query ambiguous')
    if ids:
        found=control(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.mediapipe_cpu.owner"}}']).decode().strip()
        require(found==child['Id']+'|/'+name+'|'+revision,'Cannot remove foreign container');control(['docker','rm','-f',cid],15)
    require(not control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Owned CPU container survives');cidfile.chmod(0o400)


def smoke(manifest_path,task_path,task_bytes,task_sha):
    started=time.monotonic();manifest=pinned(manifest_path,MANIFEST_PIN);task_pin=dict(bytes=int(task_bytes),sha256=task_sha)
    require(identity(task_path)==task_pin,'Original task differs before CPU graph load')
    require(sys.platform=='linux'and sys.version_info[:2]==(3,11)and Path(sys.prefix)==VENV
            and sys.prefix!=sys.base_prefix and 'include-system-site-packages = false'in(VENV/'pyvenv.cfg').read_text().lower(), 'Isolated CP311 venv without base site packages required')
    require(os.geteuid()==0 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('JAX_PLATFORMS')=='cpu'
            and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Restricted offlineCPU native runtime required')
    from importlib import metadata
    expected={r['name']:r['version']for r in manifest['packages']}
    require({n:metadata.version(n)for n in expected}==expected,'Exact installed dependency versions required')
    result=subprocess.run([sys.executable,'-I','-B','-m','pip','--isolated','check'],capture_output=True,timeout=20)
    require(result.returncode==0,'Native installed dependency consistency failed')
    import mediapipe as mp
    import numpy as np
    import cv2
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker,HandLandmarkerOptions,RunningMode
    require(Path(mp.__file__).is_relative_to(VENV)and Path(np.__file__).is_relative_to(VENV)
            and Path(cv2.__file__).is_relative_to(VENV),'No packages inherited from base runtime')
    options=HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(task_path),delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.IMAGE,num_hands=4,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    with HandLandmarker.create_from_options(options):pass
    require('torch'not in sys.modules and 'jax_cuda12_plugin'not in sys.modules,'No GPU/model-stack import permitted')
    require(identity(task_path)==task_pin and identity(manifest_path)==MANIFEST_PIN and time.monotonic()-started<=120,'Inclusive CPU posthash/budget failed')
    value=dict(status='pass',versions=expected,python='3.11',numpy=np.__version__,opencv=cv2.__version__,
               venv=str(VENV),system_site_packages=False,native_graph_loaded=True,native_graph_closed=True,
               detect_calls=0,gpu_used=False,dataset_read=False,task_identity=task_pin,elapsed_seconds=time.monotonic()-started)
    print(json.dumps(value,sort_keys=True));return value


def run(root,code,revision):
    require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'
            and canonical(root)==ROOT and Path(__file__).resolve()==code/HELPERS[0],'Actual VM02 CPU source-bound driver required')
    before=source(root,code,revision,ENTRY,HELPERS);proof=prerequisites(root,code)
    out=canonical(root/'results'/('mediapipe-cpu-runtime-verify-'+revision));require(not out.exists()and out.parent.is_dir(),'Fresh result namespace required')
    out.mkdir(mode=0o700);out.chmod(0o700);context=out/'build-context';name='world-reward-mediapipe-verify-'+revision[:12]
    report=dict(stage='mediapipe_cpu_runtime_verify',status='fail',phase='preflight',producer_revision=revision,
      source_binding=before,manifest_identity=MANIFEST_PIN,dependency_acquisition_pins=proof['dependency_pins'],
      prior_acquisition=proof['prior'],prior_verified_downloads=proof['verified_downloads'],dependency_network_used=False,
      gpu_used=False,dataset_read=False,private_values_read=False,detect_calls=0,
      quality_verified=False,license_eligibility_verified=False,task_constituent_license_verified=False,
      training_overlap_verified=False,challenge_overlap_verified=False,source_rehashed_after=False,owned_cleanup_verified=False)
    base=child=None;owned=False;failure=None;started=time.monotonic();created=None
    def interrupted(*_):raise TimeoutError('CPU qualification interrupted; owned cleanup only')
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        base=image(BASE);require(base['Id']==BASE,'Actual existing base image required')
        absent=subprocess.run(['docker','image','inspect',TARGET],capture_output=True,timeout=15)
        require(absent.returncode==1 and b'No such image'in absent.stderr,'New target tag must be absent, never retagged')
        require(not control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'CPU container namespace occupied')
        report['phase']='offline_build';build_started=time.monotonic();deadline=build_started+600
        context.mkdir(mode=0o700);created=context.stat().st_dev,context.stat().st_ino
        make_context(context,code,proof)
        absent=subprocess.run(['docker','image','inspect',TARGET],capture_output=True,timeout=15)
        require(absent.returncode==1 and b'No such image'in absent.stderr and time.monotonic()<deadline,'Target appeared or inclusive build budget exceeded')
        command=['docker','build','--network','none','--memory','8g','--cpu-period','100000','--cpu-quota','400000','--tag',TARGET,
                 '--build-arg','WR_REVISION='+revision,'--build-arg','WR_SOURCE_SHA256='+before['closure_sha256'],
                 '--build-arg','WR_MANIFEST_SHA256='+MANIFEST_PIN['sha256'],str(context)]
        with(out/'build.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400);result=subprocess.run(command,stdout=log,stderr=log,timeout=max(1,deadline-time.monotonic()),check=False)
        require(result.returncode==0 and time.monotonic()<=deadline,'Offline build failed or budget exceeded')
        child=image(TARGET);qualify_child(base,child,revision,before)
        require(time.monotonic()<=deadline,'Inclusive context/build/child qualification budget exceeded')
        report.update(base_image=base,child_image=child,build_elapsed_seconds=time.monotonic()-build_started,
                      build_budget_seconds=600,build_budget_scope='context_copy_offline_build_child_qualification',phase='cpu_graph_load')
        cid=out/'.container.cid';command=['docker','run','--rm','--name',name,'--cidfile',str(cid),
          '--label','world_reward.mediapipe_cpu.owner='+revision,'--network','none','--read-only','--user','0:0',
          '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','8g','--cpus','4','--tmpfs','/tmp:rw,nosuid,size=512m',
          '--mount','type=bind,src='+str(code)+',dst='+str(code)+',readonly',
          '--mount','type=bind,src='+str(proof['task'])+',dst=/opt/mediapipe-task/hand_landmarker.task,readonly',
          '--entrypoint','/usr/bin/env',child['Id'],'-i','PATH='+str(VENV/'bin')+':/usr/bin:/bin','HOME=/tmp',
          'MPLBACKEND=Agg','XDG_CACHE_HOME=/tmp','JAX_PLATFORMS=cpu','CUDA_VISIBLE_DEVICES=-1','PYTHONDONTWRITEBYTECODE=1',
          'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','WR_ROOT='+str(root),'WR_CODE='+str(code),
          'WR_CODE_REVISION='+revision,str(VENV/'bin/python'),'-I','-B',str(code/HELPERS[0]),'--smoke',
          str(code/MANIFEST),'/opt/mediapipe-task/hand_landmarker.task',str(proof['task_pin']['bytes']),proof['task_pin']['sha256']]
        owned=True
        with(out/'cpu-load.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400);result=subprocess.run(command,stdout=log,stderr=log,timeout=120,check=False)
        require(result.returncode==0 and(out/'cpu-load.log').stat().st_size<=32<<10,'Native CPU graph qualification failed')
        native=strict((out/'cpu-load.log').read_bytes().splitlines()[-1]);require(native.get('status')=='pass'
            and type(native.get('detect_calls'))is int and native['detect_calls']==0 and native.get('native_graph_loaded')is True
            and native.get('native_graph_closed')is True and native.get('gpu_used')is False and native.get('dataset_read')is False
            and native.get('versions')=={r['name']:r['version']for r in proof['manifest']['packages']}
            and native.get('task_identity')==proof['task_pin'],'Native create/close/package/task gate required')
        report['cpu_smoke']=native;require(image(BASE)==base and image(TARGET)==child,'Original base/child changed')
        report.update(status='pass',phase='complete')
    except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
        try:
            if owned:cleanup(out/'.container.cid',name,child,revision)
            report['owned_cleanup_verified']=True
            require((base is None or image(BASE)==base)and(child is None or image(TARGET)==child),'Original image projection changed')
            require(source(root,code,revision,ENTRY,HELPERS)==before and prerequisites(root,code)==proof,'Original source/receipt/assets changed')
            report['source_rehashed_after']=True
        except Exception as exc:failure=failure or exc;report.update(status='fail',post_error_type=type(exc).__name__)
        try:
            if created and context.exists():
                require(not context.is_symlink()and(context.stat().st_dev,context.stat().st_ino)==created,'Owned build context replaced; retained')
                shutil.rmtree(context);report['owned_build_context_removed']=True
        except Exception as exc:failure=failure or exc;report.update(status='fail',context_error_type=type(exc).__name__)
        report.update(status='fail'if failure else report['status'],elapsed_seconds=time.monotonic()-started)
        write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('CPU runtime qualification failed; inspect owned receipts')from None
    print(json.dumps({k:report[k]for k in('stage','status','gpu_used','dataset_read','detect_calls','quality_verified')}));return report


def main(argv=None):
    args=sys.argv[1:]if argv is None else argv
    if args and args[0]=='--smoke':
        require(len(args)==5,'Exact internal smoke arguments required');return smoke(*args[1:])
    require(not args,'No arbitrary runtime build arguments')
    return run(ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps(dict(stage='mediapipe_cpu_runtime_verify',status='fail',error_type=type(exc).__name__)));sys.exit(1)
