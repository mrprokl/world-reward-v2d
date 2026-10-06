"""One blind full48 original HOI run; public banks, no roles/FIT/selection."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.request

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_public_observation_context as public
import vcoco_cardinality_observation_adapters as adapter
import hoi_detr_model_qualify as original
import hoi_detr_runtime_verify as runtime
import hoi_detr_acquire as acquisition
import mediapipe_hands_acquire as atomic

rt,ROOT=public.rt,public.rt.ROOT
ENTRY='run_vcoco_full_hoi_observations'
OUTPUT=ROOT/'results/vcoco-full-hoi-observations-v1'
DATA=Path('/srv/world-reward-data/vcoco_fit_cal_v1/inputs')
BANKS=Path('/srv/world-reward-data/vcoco_full48_endpoint_banks') # individual Docker leaf remapping only
ENDPOINT=ROOT/'results/vcoco-fit-cal-endpoint-observations-v1'
WORK=OUTPUT/'.work'
IMAGE='sha256:5fa124a3c7c92c0482b6be0d1477f43da2d246d550411170e01a8f9d4fb8523a'
SCHEMA='world_reward.vcoco_full_hoi_observations.v1'
BUDGET=1800
MAX_BANK=8 << 20
FLAGS=('reference_metadata_read','split_metadata_read','FIT_performed','ground_truth_used','challenge_inputs_used',
       'ownership_verified','quality_verified','adoption','AMP_used','TF32_used')
ENDPOINT_REV='6837a74a36a29c7c0cc79dfecfb98fc03d569a3f'
ENDPOINT_PINS={'report.json':dict(bytes=280261,sha256='0ff65853cc5ecec8060b7206d3cfe3960923c8b2ebef8b11eda3eae8ed75da62'),
 'native.json':dict(bytes=177044,sha256='83cb64b09e026332481fb5227b618acd036edff1d220257c7c3a97470280fe56'),
 'proof.json':dict(bytes=12350,sha256='71f296fc005ffb12191bef5d98dfd7864bce30b7925e7be08d573b60b75df1c3')}
NATIVE_FILES=tuple(dict.fromkeys(('infra/vcoco_full_hoi_run.py','infra/hoi_detr_native_model.py',*public.HELPERS,
 'infra/hoi_detr_model_qualify.py',original.PROTOCOL,'infra/hoi_detr_runtime_verify.py','infra/hoi_detr_acquire.py',
 'infra/mediapipe_hands_acquire.py')))
HELPERS=tuple(dict.fromkeys((*NATIVE_FILES,'infra/run_vcoco_full_hoi_observations.sh','infra/sealed_callback_publication.py',
 'infra/vcoco_fit_cal_endpoint_run.py','infra/vcoco_hoi_observations.py','infra/coco_endpoint_evaluate.py',*original.HELPERS,*runtime.HELPERS,*acquisition.HELPERS,
 *public.endpoint.HELPERS)))
CONTEXT=public.PublicObservationContext(DATA,BANKS,OUTPUT)
encode,error=public.endpoint.original.encode,public.endpoint.original.error
snapshot=public._state
check=public.endpoint.check


def source(code,revision):
    value=rt.source(ROOT,code,revision,ENTRY,HELPERS);public.source_identity(code,value)
    for module,name in ((original,'infra/hoi_detr_model_qualify.py'),(runtime,'infra/hoi_detr_runtime_verify.py'),
            (acquisition,'infra/hoi_detr_acquire.py'),(atomic,'infra/mediapipe_hands_acquire.py'),(adapter,'infra/vcoco_cardinality_observation_adapters.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Actual qualified helper origin required')
    import vcoco_hoi_observations as qualified
    rt.require(Path(qualified.__file__).resolve()==code/'infra/vcoco_hoi_observations.py'
        and set(qualified.FROZEN)<=set(value['helpers'])and all(value['helpers'][n]==pin for n,pin in qualified.FROZEN.items()), 'Qualified original numerical source differs')
    return dict(binding=value,states={str(p.relative_to(code)):snapshot(p)for p in(code,*sorted(code.rglob('*')))})


def endpoint_inputs(code,current,deadline):
    """HOST ONLY metadata/byte authentication; never NumPy on the system host."""
    import vcoco_fit_cal_endpoint_run as endpoint
    host=rt.pinned(ENDPOINT/'report.json',ENDPOINT_PINS['report.json'],2 << 20)
    native=rt.pinned(ENDPOINT/'native.json',ENDPOINT_PINS['native.json'],2 << 20)
    proof=rt.pinned(ENDPOINT/'proof.json',ENDPOINT_PINS['proof.json'],2 << 20)
    old=ROOT/'jobs'/ENDPOINT_REV/endpoint.ENTRY/'code';binding=host['source_binding']
    rt.require(Path(endpoint.__file__).resolve()==code/'infra/vcoco_fit_cal_endpoint_run.py'
        and rt.source(ROOT,old,ENDPOINT_REV,endpoint.ENTRY,tuple(binding['helpers']))==binding==proof['source']
        and host['producer_revision']==ENDPOINT_REV and host['schema']==public.endpoint.SCHEMA
        and host['stage']=='full48_endpoint_observations_host'and host['status']=='pass'
        and host['native_report_identity']==ENDPOINT_PINS['native.json']and native['proof_identity']==ENDPOINT_PINS['proof.json']
        and host['native_exit_status']==0 and host['native_images']==native['images']
        and all(host[k]is True for k in('outputs_sealed','owned_cleanup_verified','source_inputs_assets_runtime_rehashed_after'))
        and all(host[k]is False for k in('reference_metadata_read','split_metadata_read','FIT_performed','ground_truth_used',
            'challenge_inputs_used','actor_selection_performed','ownership_verified','quality_verified','adoption')), 'Actual independently pinned48 endpoint PASS required')
    inputs=public.endpoint.public_inputs(public.endpoint.EndpointContext(DATA,ENDPOINT),host['public_inputs_identity'])
    endpoint.validate_native(native,proof,ENDPOINT_REV,ENDPOINT_PINS['proof.json'],inputs)
    expected=set(ENDPOINT_PINS)|{'.container.cid'}|{f'image_{i:06d}.npz'for i in range(48)}
    owner=ENDPOINT.lstat();rt.require(stat.S_ISDIR(owner.st_mode)and stat.S_IMODE(owner.st_mode)==0o500
        and {p.name for p in ENDPOINT.iterdir()}==expected,'Exact sealed original endpoint namespace')
    files={str(p):dict(pin=rt.identity(p,16 << 20),state=snapshot(p))for p in ENDPOINT.iterdir()}
    rt.require(all(stat.S_IMODE(Path(n).stat().st_mode)==0o400 and Path(n).stat().st_uid==owner.st_uid for n in files),'Original400 endpoint leaves')
    projection=dict(schema='world_reward.public_endpoint_bank_reference.v1',inputs=inputs,
        manifest_identity=host['public_inputs_identity'],banks=[{k:r[k]for k in public.BANK_FIELDS}for r in native['images']])
    return projection,dict(source=binding,source_states={str(p.relative_to(old)):snapshot(p)for p in(old,*sorted(old.rglob('*')))},files=files)


def qualified_inputs(code,deadline):
    policy=rt.pinned(code/original.PROTOCOL,original.PROTOCOL_PIN,16 << 10)
    runtime_proof=original.authenticate_runtime(rt,runtime,policy,deadline)
    assets,manifest=original.authenticate_acquisition(rt,acquisition,policy)
    import vcoco_hoi_observations as qualified
    model=qualified.qualified_model();rt.require(runtime_proof['image']['Id']==IMAGE,'Exact original HOI image')
    return policy,runtime_proof,assets,manifest,model


def write(path,value):
    raw=encode(value);rt.require(len(raw)<=2 << 20,'Bounded current public proof/receipt')
    rt.write(path,raw,0o400);return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def native_source(code,revision,proof):
    rt.require(proof['revision']==revision and proof['image_id']==IMAGE
        and {str(p.relative_to(code))for p in code.rglob('*')if p.is_file()}==set(NATIVE_FILES)
        and set(proof['native_files'])==set(NATIVE_FILES),'Exact current native source whitelist')
    for n,pin in proof['markers'].items():rt.require(rt.identity(code.parent/n,100)==pin,'Current marker differs')
    rt.require((code.parent/'revision').read_bytes()==(revision+'\n').encode(),'Exact revision marker newline')
    rows={n:dict(pin=rt.identity(code/n,2 << 20,empty=True),state=snapshot(code/n))for n in NATIVE_FILES}
    rt.require({n:r['pin']for n,r in rows.items()}==proof['native_files'],'Every native helper changed')
    public.source_identity(code,dict(helpers={n:proof['native_files'][n]for n in public.HELPERS}));return rows


def overlay(policy,proof,deadline):
    """Unchanged original pure-wheel recipe in a new caller-owned namespace."""
    root=WORK/'scratch';fair=root/'fairscale';fair.mkdir();build=root/'build';wheels=root/'wheels';wheels.mkdir();site=WORK/'site';site.mkdir()
    original.unpack_fairscale(rt,acquisition,root/policy['fairscale']['file'],fair,policy['fairscale'],deadline,check)
    original.seal_tree(fair);before=original.inventory(rt,fair);shutil.copytree(fair,build)
    for p in(build,*build.rglob('*')):p.chmod(0o755 if p.is_dir()else 0o644)
    torch,_,versions=runtime.versions(proof['runtime_configuration']);rt.require(not torch.cuda.is_initialized(),'CPU overlay cannot initialize CUDA')
    env=dict(PATH='/opt/conda/bin:/usr/bin:/bin',HOME='/tmp',CUDA_VISIBLE_DEVICES='-1',BUILD_CUDA_EXTENSIONS='0',PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1')
    def pip(args):
        r=subprocess.run([sys.executable,'-I','-B','-m','pip','--isolated',*args],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            timeout=max(.001,deadline-time.monotonic()),check=False);rt.require(r.returncode==0,'Original offline pure-wheel command failed');check(deadline)
    pip(['wheel','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',str(wheels),str(build)])
    candidates=list(wheels.glob('*.whl'));rt.require(len(candidates)==1 and candidates[0].name=='fairscale-0.4.13-py3-none-any.whl','One original FairScale wheel')
    notices={'fairscale':runtime.wheel_notice(candidates[0],policy['fairscale'],(root/'FairScale.LICENSE').read_bytes())};extra=[]
    for row in policy['import_wheels']:
        notices[row['name']]=runtime.wheel_notice(root/row['file'],row,(root/row['publisher_license']['file']).read_bytes());extra.append(str(root/row['file']))
    pip(['install','--no-index','--no-deps','--no-cache-dir','--no-compile','--target',str(site),str(candidates[0]),*extra])
    original.check_inventory(rt,fair,before);rt.require(not any(p.suffix in('.so','.pyd','.dll')for p in site.rglob('*')),'Pure original overlay only')
    original.seal_tree(site);original.seal_tree(wheels);base=Path('/opt/world-reward-hoi-mmcv')
    sys.path[:0]=[str(site),str(base/'site'),str(base/'source'),str(WORK/'source')]
    for name in('fairscale.nn.checkpoint','mmdet.models.builder','projects.models','mmdet.datasets.pipelines'):__import__(name)
    from mmcv import Config
    cfg=Config.fromfile(str(WORK/'source'/policy['native_config']['path']),import_custom_modules=False);original.configuration_policy(cfg._cfg_dict,policy['native_config'])
    rt.require(not torch.cuda.is_initialized()and not any(n=='mmdet.apis'or n.startswith('mmdet.apis.')for n in sys.modules)
        and not any(x.startswith(('/gpfs','/lus'))for x in sys.path),'Original CPU-only import/config closure')
    return dict(site_inventory=original.inventory(rt,site),wheel_identity=rt.identity(candidates[0],1 << 20),notices=notices,versions=versions,cuda_initialized=False,base_installed=False)


def reference(code,proof,deadline):
    return public.public_bank_reference(CONTEXT,projection_path=OUTPUT/'public_bank.json',projection_pin=proof['projection_identity'],
        code=code,native_source=dict(helpers={n:proof['native_files'][n]for n in public.HELPERS}),deadline=deadline,native_mounts=True)


def forward(code,proof,policy,deadline,report):
    import hoi_detr_native_model as neutral
    from world_reward.hoi_detr_observations import infer_hoi_detr_frame
    ref=reference(code,proof,deadline) # ALL48 full17 banks/JPEGs before checkpoint/model
    original.check_inventory(rt,WORK/'site',proof['overlay']['site_inventory']);base=Path('/opt/world-reward-hoi-mmcv')
    sys.path[:0]=[str(WORK/'site'),str(base/'site'),str(base/'source'),str(WORK/'source')]
    torch,np,versions=runtime.versions(proof['runtime_configuration'])
    rt.require({n:importlib.metadata.version(n)for n in policy['extra_base_distributions']}==policy['extra_base_distributions']
        and importlib.metadata.version('fairscale')=='0.4.13'
        and all(importlib.metadata.version(r['name'])==r['version']for r in policy['import_wheels']),'Original full dependency versions')
    import mmcv,mmcv._ext as extension
    rt.require(Path(mmcv.__file__).is_relative_to(base/'source')and rt.identity(Path(extension.__file__),1 << 30)==proof['extension_identity'],'Original qualified MMCV extension')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);built=None
    try:
        report['phase']='model';built=neutral.build_original_hoi(torch=torch,np=np,runtime=runtime,deadline=deadline,source_root=WORK/'source',
            checkpoint_path=original.DATA/'weights/epoch_5.pth',native_config=policy['native_config'],checkpoint_buffers=policy['checkpoint_buffers'])
        report.update(models_loaded=1,model=dict(built.provenance),versions=versions,phase='all48_original_forwards')
        def infer(rgb):
            report['native_forward_calls']=report.get('native_forward_calls',0)+1
            return infer_hoi_detr_frame(built.model,rgb,0,built.operations)
        adapter.observe_hoi(ref,infer,deadline,records=report['images'])
        torch.cuda.synchronize()
    finally:
        try:ref.verify(deadline)
        finally:built=None;torch.cuda.empty_cache();torch.cuda.synchronize();report['models_released']=True


def native(code,revision,phase,pin,deadline):
    proof=rt.pinned(OUTPUT/(phase+'_proof.json'),pin,2 << 20);before=None;states={};owner=snapshot(OUTPUT)[:2]+(OUTPUT.stat().st_uid,)
    rt.require(stat.S_ISDIR(OUTPUT.stat().st_mode)and stat.S_IMODE(OUTPUT.stat().st_mode)==0o700 and OUTPUT.stat().st_uid==os.geteuid(),'Owned native output700 namespace')
    report=dict(schema=SCHEMA,stage='native_full48_hoi_'+phase,
        status='fail',phase='authentication',producer_revision=revision,image_id=IMAGE,proof_identity=pin,images=[],models_loaded=0,models_released=False,native_forward_calls=0,completed_bank_count=0,
        source_inputs_runtime_assets_rehashed_after=False,**{k:False for k in FLAGS})
    try:
        rt.require(set(proof)=={'revision','markers','native_files','image_id','runtime_configuration','runtime_artifacts','extension_identity',
            'derived_source_inventory','files','projection_identity','started_monotonic'}|({'overlay'}if phase=='model'else set()),
            'Sanitized current native proof only')
        rt.require(os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==IMAGE and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Exact offline qualified image')
        report['phase']='source';before=native_source(code,revision,proof);report['phase']='assets';states={n:dict(pin=rt.identity(n,6_000_000_000,empty=p['bytes']==0),state=snapshot(n))for n,p in proof['files'].items()}
        rt.require({n:r['pin']for n,r in states.items()}==proof['files'],'Original current model/public assets changed')
        original.check_inventory(rt,Path('/opt/world-reward-hoi-mmcv'),proof['runtime_artifacts']);original.check_inventory(rt,WORK/'source',proof['derived_source_inventory'])
        policy=rt.pinned(code/original.PROTOCOL,original.PROTOCOL_PIN,16 << 10)
        if phase=='overlay':report['phase']='overlay';report.update(overlay(policy,proof,deadline));report['models_released']=True
        else:report['phase']='banks';forward(code,proof,policy,deadline,report)
        check(deadline);report.update(status='pass',phase='complete')
    except BaseException as exc:report.update(error_type=error(exc),failure_stage=report['phase'])
    finally:
        try:
            rt.require(before is not None and native_source(code,revision,proof)==before
                and {n:dict(pin=rt.identity(n,6_000_000_000,empty=p['bytes']==0),state=snapshot(n))for n,p in proof['files'].items()}==states,
                'Current source/input/model bytes or states changed')
            original.check_inventory(rt,Path('/opt/world-reward-hoi-mmcv'),proof['runtime_artifacts']);original.check_inventory(rt,WORK/'source',proof['derived_source_inventory'])
            if phase=='model':
                original.check_inventory(rt,WORK/'site',proof['overlay']['site_inventory'])
                ref=reference(code,proof,deadline);adapter.validate_saved_records(ref,'hoi',report['images'],deadline,complete=report['status']=='pass')
            report['source_inputs_runtime_assets_rehashed_after']=True;check(deadline)
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc),post_failure_stage='posthash')
        report['completed_bank_count']=len(report['images']);report['elapsed_seconds']=time.monotonic()-proof['started_monotonic']
        rt.require(snapshot(OUTPUT)[:2]+(OUTPUT.stat().st_uid,)==owner and stat.S_IMODE(OUTPUT.stat().st_mode)==0o700,'Original native output namespace changed')
        path=OUTPUT/(phase+'.json')
        with path.open('x+b')as stream:
            os.fchmod(stream.fileno(),0o400);opened=os.fstat(stream.fileno())
            def update():
                s=path.lstat();raw=encode(report);rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and stat.S_IMODE(s.st_mode)==0o400
                    and(s.st_dev,s.st_ino,s.st_uid)==(opened.st_dev,opened.st_ino,opened.st_uid)and len(raw)<=2 << 20,'Owned native receipt')
                stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
            try:update();check(deadline)
            except BaseException:report.update(status='fail',publication_failed=True);update()
    return report


def validate_native(value,proof,revision,phase,pin,projection):
    rt.require(value['schema']==SCHEMA and value['stage']=='native_full48_hoi_'+phase and value['status']=='pass'and value['phase']=='complete'
        and value['producer_revision']==revision and value['image_id']==IMAGE and value['proof_identity']==pin
        and value['models_released']is value['source_inputs_runtime_assets_rehashed_after']is True
        and all(value[k]is False for k in FLAGS),'Actual complete48 native original HOI receipt')
    if phase=='overlay':
        rt.require(type(value['models_loaded'])is int and type(value['native_forward_calls'])is int
            and value['models_loaded']==value['native_forward_calls']==value['completed_bank_count']==0 and value['cuda_initialized']is value['base_installed']is False,'Original CPU overlay');return
    rt.require(len(projection['inputs']['images'])==len(projection['banks'])==48 and type(value['models_loaded'])is int and value['models_loaded']==1 and type(value['native_forward_calls'])is int and value['native_forward_calls']==value['completed_bank_count']==len(value['images'])==48
        and value['model']['strict_checkpoint']['keys']==1796 and value['model']['strict_checkpoint']['strict']is value['model']['strict_checkpoint']['weights_only']is True
        and value['model']['checkpoint_buffer_schema']['ema_swapped']is False,'One strict original model/all48')
    ints={'native_nms_keep','retained_nms_positions','query_ids','class_ids','hand_object_pairs','object_target_pairs','image_size','original_frame_index','original_slot','acquired_ordinal'}
    for i,(r,image,bank)in enumerate(zip(value['images'],projection['inputs']['images'],projection['banks'])):
        n,k,t=r['native_detections'],r['hand_object_pairs'],r['object_target_pairs'];a=r['arrays']
        rt.require(all(type(v)is int and 0<=v<=1_000_000 for v in(n,k,t))and n<=1000,'Full native pair counts')
        shapes=dict(query_logits=[1500,3],query_boxes_cxcywh=[1500,4],query_tokens=[1500,256],native_nms_detections=a['native_nms_detections']['shape'],
            native_nms_keep=[a['native_nms_detections']['shape'][0]],retained_nms_positions=[n],query_ids=[n],class_ids=[n],boxes_original_xyxy=[n,4],raw_scores=[n],decayed_scores=[n],
            hand_object_pairs=[k,2],hand_object_logits=[k,2],object_target_pairs=[t,2],object_target_logits=[t,2],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        rt.require(r['image_id']==image['image_id']and r['original_slot']==r['acquired_ordinal']==i and r['original_frame_index']==0
            and r['image_size']==bank['image_size']and r['endpoint_bank_identity']==bank['identity']and r['source_person_ids']==bank['person_ids']and r['owl_patches']==3600
            and r['file']==f'image_{i:06d}.npz'and rt.identity(OUTPUT/r['file'],MAX_BANK)==r['identity']and set(a)==set(adapter.HOI_FIELDS)
            and len(shapes['native_nms_detections'])==2 and shapes['native_nms_detections'][1]==5 and shapes['native_nms_detections'][0]<=1000
            and all(a[name]['shape']==shape and a[name]['dtype']==('<i8'if name in ints else '<f4')and re.fullmatch('[0-9a-f]{64}',str(a[name]['sha256']))for name,shape in shapes.items()),
            'Full original19 arrays/IDs/grid required')


def cleanup(phase,name,revision,deadline):
    path=OUTPUT/(phase+'.cid')
    if path.exists():
        rt.identity(path,65,readonly=False);raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Original owned CID')
        cid=raw.decode().strip();found=runtime.command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline)
        rt.require(found in('',cid),'Ambiguous CID')
        if found:
            actual=runtime.command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)
            rt.require(actual==IMAGE+'|/'+name+'|'+ENTRY+'|'+revision,'Foreign CID/name cannot be removed');runtime.command(['docker','rm','-f',cid],deadline)
        rt.require(not runtime.command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline),'CID survives');path.chmod(0o400)
    rt.require(not runtime.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Name survives')


def launch(code,revision,phase,proof,projection,deadline):
    pin=write(OUTPUT/(phase+'_proof.json'),proof);name='world-reward-vcoco-full-hoi-'+phase+'-'+revision[:12]
    mounts=[(code/n,code/n)for n in NATIVE_FILES]+[(code.parent/n,code.parent/n)for n in('revision','source-sha256')]
    if phase=='model':
        mounts +=[(DATA/'manifest.json',DATA/'manifest.json')]+[(DATA/r['file'],DATA/r['file'])for r in projection['inputs']['images']]
        mounts +=[(ENDPOINT/r['file'],BANKS/r['file'])for r in projection['banks']]
    mounts +=[(Path(n),Path(n))for n in proof['files']]
    args=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/(phase+'.cid')),'--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,
        '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges','--pids-limit','512','--cpus','4',
        '--memory','64g'if phase=='model'else '16g','--tmpfs','/tmp:rw,exec,nosuid,size=2g','--entrypoint','/usr/bin/env']
    if phase=='model':args+=['--gpus','device=0']
    for src,dst in dict.fromkeys(mounts):
        rt.canonical(src);rt.require(src.is_file()and all(c not in str(src)+str(dst)for c in(',','\n')), 'Literal individual readonly leaf mount')
        args+=['--mount',f'type=bind,src={src},dst={dst},readonly']
    args+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}']
    if phase=='model':args+=['--mount',f'type=bind,src={WORK},dst={WORK},readonly']
    args +=[IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin','HOME=/tmp','PYTHONDONTWRITEBYTECODE=1',
        'CUDA_VISIBLE_DEVICES=0'if phase=='model'else 'CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4',
        'WR_IMAGE_ID='+IMAGE,'/opt/conda/bin/python','-I','-B',str(code/NATIVE_FILES[0]),'--phase',phase,'--proof-bytes',str(pin['bytes']),
        '--proof-sha256',pin['sha256'],'--deadline',format(deadline-60,'.17g'),'--code',str(code),'--revision',revision]
    rt.require(not runtime.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Occupied own namespace')
    try:
        r=subprocess.run(args,env=runtime.SAFE_ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.001,deadline-time.monotonic()-60),check=False)
        saved_pin=rt.identity(OUTPUT/(phase+'.json'),2 << 20);saved=rt.pinned(OUTPUT/(phase+'.json'),saved_pin,2 << 20)
        rt.require(r.returncode==0,'Original native child failed');validate_native(saved,proof,revision,phase,pin,projection);return saved,pin
    finally:cleanup(phase,name,revision,min(deadline+60,time.monotonic()+30))


def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def host(code,revision,started,deadline):
    import sealed_callback_publication as publication
    before=projection=inputs=prior=asset_states=None;owners=[];partials=[];directories=set();lock_fd=None;artifacts={}
    rt.require(not OUTPUT.exists()and not OUTPUT.is_symlink(),'Fresh full48 output');OUTPUT.mkdir(mode=0o700);owner=snapshot(OUTPUT)[:2]+(OUTPUT.stat().st_uid,)
    report=dict(schema=SCHEMA,stage='full48_hoi_observations_host',phase='source',status='fail',producer_revision=revision,image_id=IMAGE,images=[],
        owned_cleanup_verified=False,source_inputs_runtime_assets_rehashed_after=False,outputs_sealed=False,host_network_used=False,**{k:False for k in FLAGS})
    try:
        before=source(code,revision);report['phase']='endpoint';projection,inputs=endpoint_inputs(code,before['binding'],deadline)
        report['phase']='runtime_assets';policy,qualified,assets,manifest,prior=qualified_inputs(code,deadline)
        frozen={**assets['frozen'],**qualified['frozen'],**prior['files']};asset_states={n:snapshot(n)for n in frozen}
        report.update(source_binding=before['binding'],endpoint_receipt_pins=ENDPOINT_PINS,endpoint_source=inputs['source'],
            model_qualification=prior,runtime_report_identity=qualified['report_identity'],acquisition_report_identity=assets['report_identity'])
        report['phase']='overlay_assets';WORK.mkdir(mode=0o700);s=WORK.lstat();owners.append((WORK,(s.st_dev,s.st_ino,s.st_uid)))
        scratch=WORK/'scratch';scratch.mkdir();rows=[policy['fairscale'],policy['fairscale']['publisher_license'],*[r for w in policy['import_wheels']for r in(w,w['publisher_license'])]]
        rt.require(len(rows)==4 and sum(r['bytes']for r in rows)==284220,'Exact original4 publisher overlay assets')
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),runtime.NoRedirect());report['host_network_used']=True
        for row in rows:runtime.fetch(atomic,acquisition,scratch,row,deadline,partials,directories,opener)
        for row in rows:
            if row['file'].endswith('.LICENSE'):rt.write(OUTPUT/row['file'],(scratch/row['file']).read_bytes(),0o400)
        patch=original.derive_source(rt,acquisition,manifest,WORK/'source',policy)
        report['phase']='projection';projection_pin=write(OUTPUT/'public_bank.json',projection)
        proof=dict(revision=revision,markers=before['binding']['markers'],native_files={n:before['binding']['helpers'][n]for n in NATIVE_FILES},image_id=IMAGE,
            runtime_configuration=qualified['configuration'],runtime_artifacts=qualified['manifest']['artifacts'],extension_identity=qualified['report']['native_extension_identity'],
            derived_source_inventory=original.inventory(rt,WORK/'source'),files={str(scratch/r['file']):original.row_pin(r)for r in rows},
            projection_identity=projection_pin,started_monotonic=started)
        report['phase']='overlay';cpu,cpu_pin=launch(code,revision,'overlay',proof,projection,deadline);report['overlay_identity']=rt.identity(OUTPUT/'overlay.json',2 << 20)
        proof['files']={str(original.DATA/'weights/epoch_5.pth'):policy['acquisition']['checkpoint']};proof['overlay']={k:cpu[k]for k in('site_inventory','wheel_identity','notices')}
        report['phase']='gpu_lock';lock=rt.canonical(ROOT/'jobs/.world-reward-h100.lock');s=lock.lstat();rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Original cooperative GPU lock')
        lock_fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);rt.require((os.fstat(lock_fd).st_dev,os.fstat(lock_fd).st_ino)==(s.st_dev,s.st_ino),'Lock inode changed');fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt.require(not runtime.command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline),'GPU occupied')
        report['phase']='native';saved,model_pin=launch(code,revision,'model',proof,projection,deadline)
        report.update(status='pass',phase='complete',images=saved['images'],native_report_identity=rt.identity(OUTPUT/'model.json',2 << 20),native_proof_identity=model_pin)
    except BaseException as exc:report.update(error_type=error(exc),failure_stage=report['phase'])
    finally:
        try:
            for phase in('overlay','model'):
                path=OUTPUT/(phase+'.json')
                if path.exists():
                    pin=rt.identity(path,2 << 20);value=rt.pinned(path,pin,2 << 20)
                    report[phase+'_summary']={k:value[k]for k in('status','phase','failure_stage','error_type','post_error_type','models_loaded','native_forward_calls','completed_bank_count')if k in value}
                    report[phase+'_identity']=pin
            artifacts={p.name:rt.identity(p,16 << 20,readonly=not p.name.endswith('.cid'))for p in OUTPUT.iterdir()if p.is_file()}
        except BaseException as exc:report.update(status='fail',receipt_error_type=error(exc),receipt_failure_stage='saved_receipts')
        try:
            rt.require(before is not None and projection is not None and prior is not None and source(code,revision)==before
                and endpoint_inputs(code,before['binding'],deadline)==(projection,inputs)and qualified_inputs(code,deadline)==(policy,qualified,assets,manifest,prior)
                and {n:snapshot(n)for n in asset_states}==asset_states,'Current/original source/public/model/runtime drift')
            rt.require(not runtime.command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline),'GPU process survives')
            for name,pin in artifacts.items():rt.require(rt.identity(OUTPUT/name,16 << 20,readonly=not name.endswith('.cid'))==pin,'Saved native file changed')
            for r in report['images']:rt.require(rt.identity(OUTPUT/r['file'],MAX_BANK)==r['identity'],'Saved48 model files changed')
            report['source_inputs_runtime_assets_rehashed_after']=True;check(deadline)
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc),post_failure_stage='posthash')
        try:
            for phase in('overlay','model'):cleanup(phase,'world-reward-vcoco-full-hoi-'+phase+'-'+revision[:12],revision,min(deadline+60,time.monotonic()+30))
            for folder,inode in owners:
                if folder.exists():runtime.remove_owned_folder(rt,folder,inode,OUTPUT)
            report['owned_cleanup_verified']=True
        except BaseException as exc:report.update(status='fail',cleanup_error_type=error(exc),cleanup_failure_stage='cleanup')
        finally:
            if lock_fd is not None:os.close(lock_fd)
        names={'public_bank.json','FairScale.LICENSE','terminaltables.LICENSE','overlay_proof.json','model_proof.json','overlay.json','model.json','overlay.cid','model.cid'}|{f'image_{i:06d}.npz'for i in range(48)}
        allowed={p.name for p in OUTPUT.iterdir()}&names
        rt.require(sum(p.stat().st_size for p in OUTPUT.iterdir()if p.is_file())<=400 << 20,'Bounded full48 output bytes')
        report['saved_output_identities']={name:rt.identity(OUTPUT/name,16 << 20)for name in allowed}
        publication.publish(OUTPUT,report,deadline,started,owner,allowed,encode=encode,identity=rt.identity,snapshot=snapshot,require=rt.require,check=check,
            sync=sync,error=error,maximum=16 << 20,report_maximum=2 << 20)
    return report


def arguments(argv):
    rt.require(all(argv.count(n)<=1 for n in('--phase','--proof-bytes','--proof-sha256','--deadline','--code','--revision')),'No duplicate native args')
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--phase',choices=('overlay','model'));p.add_argument('--proof-bytes',type=int);p.add_argument('--proof-sha256');p.add_argument('--deadline',type=float);p.add_argument('--code');p.add_argument('--revision');a=p.parse_args(argv)
    if a.phase:
        rt.require(type(a.deadline)is float and math.isfinite(a.deadline)and type(a.code)is str and re.fullmatch('[0-9a-f]{40}',str(a.revision)), 'Explicit native context required')
        pin=dict(bytes=a.proof_bytes,sha256=a.proof_sha256);setattr(a,'pin',pin);rt.require(type(a.proof_bytes)is int and 0<a.proof_bytes<=2 << 20 and re.fullmatch('[0-9a-f]{64}',str(a.proof_sha256)), 'Independent native proof required')
    else:rt.require(all(getattr(a,n)is None for n in('proof_bytes','proof_sha256','deadline','code','revision')),'Fixed host context only')
    return a


def main():
    a=arguments(sys.argv[1:]);code=Path(a.code if a.phase else os.environ['WR_CODE']);revision=a.revision if a.phase else os.environ['WR_CODE_REVISION']
    rt.require(code==ROOT/'jobs'/revision/ENTRY/'code'and Path(__file__).resolve()==code/NATIVE_FILES[0]and re.fullmatch('[0-9a-f]{40}',revision),'Exact current caller source')
    started=time.monotonic();deadline=a.deadline if a.phase else started+BUDGET
    rt.require(math.isfinite(deadline)and 0<deadline-time.monotonic()<=BUDGET,'Shared inclusive budget')
    def expired(*_):raise TimeoutError('Inclusive complete48 HOI budget')
    handlers={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    try:
        if a.phase:return native(code,revision,a.phase,a.pin,deadline)
        rt.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02','Exact root VM02');return host(code,revision,started,deadline)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)


if __name__=='__main__':
    result=main();print(encode({k:result[k]for k in('stage','status')}).decode(),end='');raise SystemExit(0 if result['status']=='pass'else 1)
