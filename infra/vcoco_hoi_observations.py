"""Fresh V-COCO16 original HOI banks; no references, fitting or ownership claim."""
import argparse
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
import vcoco_observation_replica as public
import hoi_detr_model_qualify as original
import hoi_detr_runtime_verify as runtime
import hoi_detr_acquire as acquisition
import mediapipe_hands_acquire as atomic

rt,ROOT=public.rt,public.ROOT
ENTRY='run_vcoco_hoi_observations'
OUTPUT=ROOT/'results/vcoco-hoi-observations-v1'
IMAGE='sha256:5fa124a3c7c92c0482b6be0d1477f43da2d246d550411170e01a8f9d4fb8523a'
SCHEMA='world_reward.vcoco_hoi_observations.v1'
BUDGET=1800
MAX_BANK=8 << 20
FIELDS=('query_logits','query_boxes_cxcywh','query_tokens','native_nms_detections','native_nms_keep',
    'retained_nms_positions','query_ids','class_ids','boxes_original_xyxy','raw_scores','decayed_scores',
    'hand_object_pairs','hand_object_logits','object_target_pairs','object_target_logits',
    'image_size','original_frame_index','original_slot','acquired_ordinal')
FROZEN={'infra/hoi_detr_model_qualify.py':dict(bytes=76004,sha256='80f75eecfdac7e3a4f6f71a246145eeb94bf58c4e6030fcfcb22c29e497df263'),
 'infra/hoi_detr_runtime_verify.py':dict(bytes=55165,sha256='cb1d444478d95cb3a0cc416d1268381e24051956c9ac89e4a7f1bbb92c66c44b'),
 'infra/hoi_detr_acquire.py':runtime.ACQUIRE_PIN,
 'infra/hoi_detr_native_model.py':dict(bytes=4624,sha256='a267c19695616d70b2dc6e051da74b20746e93124e849ab3713b92877d871896'),
 'src/world_reward/hoi_detr_observations.py':dict(bytes=14388,sha256='047a80cb618fb98da19b233c199fb927c9b9930b1a6573f62abb2e07ac6ebd8f'),
 'infra/coco_endpoint_evaluate.py':dict(bytes=38477,sha256='f1bf582b704b06891efbda4f13d643c214379e92a997b38506e7d33b13085685')}
QUALIFIED=dict(path='results/hoi-detr-model-qualify-v4/report.json',producer_revision='b9571478c734c41531204f8c1a974556be6802dc',
 report=dict(bytes=2967,sha256='65eb67cf953a698e4c5decada37a2c051d3af23e34967610d7a40fca9413c3e5'),
 native=dict(bytes=3572,sha256='51c30b02ba9127b3a6208e6e4f2ce77469f2e6434188742ed00d72b54bdd3dcb'))
NATIVE_FILES=tuple(dict.fromkeys(('infra/vcoco_hoi_observations.py','infra/hoi_detr_native_model.py',
    'infra/coco_endpoint_evaluate.py',*original.HELPERS,*runtime.HELPERS,*acquisition.HELPERS,*public.HELPERS)))
HELPERS=tuple(dict.fromkeys((*NATIVE_FILES,'infra/run_vcoco_hoi_observations.sh')))
encode,error=public.encode,public.bank.error


def check(deadline):
    if not math.isfinite(deadline)or time.monotonic()>=deadline:raise TimeoutError('Inclusive original HOI observation budget')


def source(code,revision):
    value=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    for module,name in ((public,'infra/vcoco_observation_replica.py'),(original,'infra/hoi_detr_model_qualify.py'),
        (runtime,'infra/hoi_detr_runtime_verify.py'),(acquisition,'infra/hoi_detr_acquire.py'),(atomic,'infra/mediapipe_hands_acquire.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Actual unchanged helper origin required')
    rt.require(Path(__file__).resolve()==code/NATIVE_FILES[0]and all(value['helpers'][n]==p for n,p in FROZEN.items()),
        'Frozen numerical helper/source changed')
    rt.pinned(code/original.PROTOCOL,original.PROTOCOL_PIN,16 << 10)
    return dict(source=value,states=public.bank.source_state(code),markers={n:public.snapshot(code.parent/n)for n in ('revision','source-sha256')})


def qualified_model():
    paths={'host':ROOT/QUALIFIED['path'],'native':(ROOT/QUALIFIED['path']).parent/'native.json'}
    h=rt.pinned(paths['host'],QUALIFIED['report'],32 << 10);n=rt.pinned(paths['native'],QUALIFIED['native'],32 << 10)
    rt.require(h['status']==n['status']=='pass'and h['producer_revision']==QUALIFIED['producer_revision']and h['phase']=='complete'
        and h['actual_model_qualified']is True and h['native_report_identity']==QUALIFIED['native']and n['native_forward_calls']==1
        and n['strict_checkpoint']['keys']==1796 and n['strict_checkpoint']['strict']is n['strict_checkpoint']['weights_only']is True
        and all(h[k]is True for k in ('source_rehashed_after','inputs_rehashed_after','image_unchanged','owned_containers_removed','owned_overlay_removed'))
        and all(h[k]is n[k]is False for k in original.FLAGS),'Original full data-free model qualification required')
    old=ROOT/'jobs'/QUALIFIED['producer_revision']/original.ENTRY/'code';sb=h['source_binding']
    rt.require(set(original.HELPERS)<=set(sb['helpers'])and rt.source(ROOT,old,QUALIFIED['producer_revision'],original.ENTRY,tuple(sb['helpers']))==sb==n['source_binding'],
        'Complete original qualification source required')
    return dict(files={str(paths['host']):QUALIFIED['report'],str(paths['native']):QUALIFIED['native']},source=sb,
        states=public.bank.source_state(old))


def native_source(code,revision,proof):
    rt.require(proof['source']['producer_revision']==revision and Path(__file__).resolve()==code/NATIVE_FILES[0]
        and {str(p.relative_to(code))for p in code.rglob('*')if p.is_file()}==set(NATIVE_FILES),'Exact native literal source whitelist required')
    for n,p in proof['source']['markers'].items():rt.require(rt.identity(code.parent/n,100)==p,'Actual marker differs')
    rt.require((code.parent/'revision').read_bytes()==(revision+'\n').encode(),'Original revision newline required')
    rows={}
    for n in NATIVE_FILES:
        pin=rt.identity(code/n,2 << 20,empty=True)
        rt.require(pin==proof['source']['helpers'][n],'Native mounted source changed')
        rows[n]=dict(pin=pin,state=public.snapshot(code/n))
    return rows


def overlay_cpu(proof,policy,deadline):
    """Recreate the original pure private wheel overlay, never install in image."""
    root=OUTPUT/'.scratch';fairscale=root/'fairscale';fairscale.mkdir();wheels=root/'wheels';wheels.mkdir();site=OUTPUT/'.overlay/site';site.mkdir()
    original.unpack_fairscale(rt,acquisition,root/policy['fairscale']['file'],fairscale,policy['fairscale'],deadline,check)
    original.seal_tree(fairscale);before=original.inventory(rt,fairscale);build=root/'build';shutil.copytree(fairscale,build)
    for p in (build,*build.rglob('*')):p.chmod(0o755 if p.is_dir()else 0o644)
    torch,_,versions=runtime.versions(proof['runtime']['configuration']);rt.require(not torch.cuda.is_initialized(),'CPU overlay cannot initialize CUDA')
    env=dict(PATH='/opt/conda/bin:/usr/bin:/bin',HOME='/tmp',CUDA_VISIBLE_DEVICES='-1',BUILD_CUDA_EXTENSIONS='0',PIP_NO_INDEX='1',
        PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1')
    def pip(args):
        result=subprocess.run([sys.executable,'-I','-B','-m','pip','--isolated',*args],env=env,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,timeout=max(.001,deadline-time.monotonic()),check=False)
        rt.require(result.returncode==0,'Original offline wheel operation failed');check(deadline)
    pip(['wheel','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',str(wheels),str(build)])
    candidates=list(wheels.glob('*.whl'));rt.require(len(candidates)==1 and candidates[0].name=='fairscale-0.4.13-py3-none-any.whl','Exact pure original wheel required')
    notices={'fairscale':runtime.wheel_notice(candidates[0],policy['fairscale'],(root/'FairScale.LICENSE').read_bytes())}
    extras=[]
    for row in policy['import_wheels']:
        notices[row['name']]=runtime.wheel_notice(root/row['file'],row,(root/row['publisher_license']['file']).read_bytes());extras.append(str(root/row['file']))
    pip(['install','--no-index','--no-deps','--no-cache-dir','--no-compile','--target',str(site),str(candidates[0]),*extras])
    original.check_inventory(rt,fairscale,before);rt.require(not any(p.suffix in ('.so','.pyd','.dll')for p in site.rglob('*')),'Only pure Python overlay')
    original.seal_tree(site);original.seal_tree(wheels);mmcv=Path('/opt/world-reward-hoi-mmcv');sys.path[:0]=[str(site),str(mmcv/'site'),str(mmcv/'source'),str(OUTPUT/'.overlay/source')]
    for name in ('fairscale.nn.checkpoint','mmdet.models.builder','projects.models','mmdet.datasets.pipelines'):__import__(name)
    from mmcv import Config
    cfg=Config.fromfile(str(OUTPUT/'.overlay/source'/policy['native_config']['path']),import_custom_modules=False)
    original.configuration_policy(cfg._cfg_dict,policy['native_config'])
    rt.require(not torch.cuda.is_initialized()and not any(n=='mmdet.apis'or n.startswith('mmdet.apis.')for n in sys.modules)
        and not any(x.startswith(('/gpfs','/lus'))for x in sys.path),'Original CPU-only import/config closure required')
    return dict(cuda_initialized=False,base_installed=False,full_import_closure_qualified=True,versions=versions,notices=notices,
        wheel_identity=rt.identity(candidates[0],1 << 20),site_inventory=original.inventory(rt,site))


def observe(images,banks,load,decode,infer,save,check_now,records):
    rt.require(type(images)is list and type(banks)is list and len(images)==len(banks)==16,'All16 original aligned RGB/bank records required')
    for row in banks:check_now();load(row)
    for ordinal,(image,bank)in enumerate(zip(images,banks)):
        check_now();rgb=decode(image);digest=hashlib.sha256(rgb.tobytes()).hexdigest();result=infer(rgb)
        rt.require(result.original_frame_index==0 and result.image_size==(image['height'],image['width'])and hashlib.sha256(rgb.tobytes()).hexdigest()==digest,
            'Original RGB/grid/frame changed');records.append(save(result,image,bank,ordinal));check_now()
    return records


def native(code,revision,phase,proof_pin,deadline):
    import numpy as np
    proof=rt.pinned(OUTPUT/(phase+'_proof.json'),proof_pin,2 << 20);before_source=native_source(code,revision,proof)
    rt.require(os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==IMAGE and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Exact offline original image required')
    policy=rt.pinned(code/original.PROTOCOL,original.PROTOCOL_PIN,16 << 10)
    report=dict(schema=SCHEMA,stage='native_vcoco_hoi_'+phase,status='fail',phase='preflight',producer_revision=revision,image_id=IMAGE,
        proof_identity=proof_pin,images=[],models_loaded=0,native_forward_calls=0,reference_metadata_read=False,FIT_performed=False,
        ownership_verified=False,quality_verified=False,adoption=False,AMP_used=False,TF32_used=False,source_inputs_runtime_assets_rehashed_after=False)
    files={n:rt.identity(n,6_000_000_000,empty=p['bytes']==0)for n,p in proof['files'].items()};states={n:public.snapshot(n)for n in files}
    artifacts={};owner=OUTPUT.lstat();previous={s:signal.signal(s,lambda *_:(_ for _ in()).throw(TimeoutError('Native deadline')))for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    try:
        rt.require(files==proof['files'],'Every original input/asset differs');runtime_root=Path('/opt/world-reward-hoi-mmcv')
        original.check_inventory(rt,runtime_root,proof['runtime']['manifest']['artifacts'])
        original.check_inventory(rt,OUTPUT/'.overlay/source',proof['derived_source_inventory'])
        if phase=='overlay':report.update(overlay_cpu(proof,policy,deadline))
        else:
            import hoi_detr_native_model as neutral
            import coco_endpoint_evaluate as saved
            from world_reward.hoi_detr_observations import infer_hoi_detr_frame
            original.check_inventory(rt,OUTPUT/'.overlay/site',proof['overlay']['site_inventory'])
            sys.path[:0]=[str(OUTPUT/'.overlay/site'),str(runtime_root/'site'),str(runtime_root/'source'),str(OUTPUT/'.overlay/source')]
            torch,np,versions=runtime.versions(proof['runtime']['configuration'])
            rt.require({n:importlib.metadata.version(n)for n in policy['extra_base_distributions']}==policy['extra_base_distributions']
                and importlib.metadata.version('fairscale')=='0.4.13'and all(importlib.metadata.version(r['name'])==r['version']for r in policy['import_wheels']),
                'Complete original model dependencies required')
            import mmcv,mmcv._ext as extension
            rt.require(Path(mmcv.__file__).is_relative_to(runtime_root/'source')and rt.identity(Path(extension.__file__),1 << 30)==proof['runtime']['report']['native_extension_identity'],
                'Exactly qualified original MMCV extension required')
            def validate(row):saved.validate_npz(public.bank.OUTPUT/row['file'],row)
            for row in proof['banks']:check(deadline);validate(row)
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            built=neutral.build_original_hoi(torch=torch,np=np,runtime=runtime,deadline=deadline,source_root=OUTPUT/'.overlay/source',
                checkpoint_path=original.DATA/'weights/epoch_5.pth',native_config=policy['native_config'],checkpoint_buffers=policy['checkpoint_buffers'])
            report['models_loaded']=1;report['model']=dict(built.provenance);report['versions']=versions;report['phase']='all16_original_forwards'
            def save(result,image,bank,ordinal):
                payload={n:getattr(result,n)for n in FIELDS[:15]};payload.update(image_size=np.asarray(result.image_size,np.int64),original_frame_index=np.asarray(0,np.int64),
                    original_slot=np.asarray(bank['original_slot'],np.int64),acquired_ordinal=np.asarray(ordinal,np.int64))
                identities={n:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())for n,a in payload.items()}
                path=OUTPUT/f'image_{bank["original_slot"]:06d}.npz'
                with path.open('xb')as f:np.savez(f,**payload);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o400)
                artifacts[path.name]=rt.identity(path,MAX_BANK)
                with np.load(path,allow_pickle=False)as z:
                    rt.require(set(z.files)==set(payload)and identities=={n:dict(shape=list(z[n].shape),dtype=z[n].dtype.str,sha256=hashlib.sha256(z[n].tobytes()).hexdigest())for n in z.files},'Full native bank roundtrip differs')
                rt.require(identities=={n:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())for n,a in payload.items()},'Raw native observations mutated while saving')
                report['native_forward_calls']+=1
                return dict(image_id=image['image_id'],original_slot=bank['original_slot'],acquired_ordinal=ordinal,original_frame_index=0,image_size=list(result.image_size),
                    file=path.name,identity=artifacts[path.name],arrays=identities,native_detections=len(result.query_ids),hand_object_pairs=len(result.hand_object_pairs),
                    object_target_pairs=len(result.object_target_pairs),endpoint_bank_identity=bank['identity'],source_person_ids=bank['person_ids'],owl_patches=3600)
            observe(proof['images'],proof['banks'],lambda r:None,lambda r:public.bank.rgb_inputs.decode_rgb(public.bank.DATA,r,identity=rt.identity),
                lambda rgb:infer_hoi_detr_frame(built.model,rgb,0,built.operations),save,lambda:check(deadline),report['images'])
            torch.cuda.synchronize();del built;torch.cuda.empty_cache();torch.cuda.synchronize()
        check(deadline);report.update(status='pass',phase='complete')
    except BaseException as exc:report['error_type']=error(exc)
    finally:
        try:
            rt.require(native_source(code,revision,proof)==before_source and {n:rt.identity(n,6_000_000_000,empty=p['bytes']==0)for n,p in proof['files'].items()}==files==proof['files']
                and {n:public.snapshot(n)for n in files}==states,'Original complete source/input/asset changed')
            original.check_inventory(rt,Path('/opt/world-reward-hoi-mmcv'),proof['runtime']['manifest']['artifacts'])
            original.check_inventory(rt,OUTPUT/'.overlay/source',proof['derived_source_inventory'])
            if phase=='model':original.check_inventory(rt,OUTPUT/'.overlay/site',proof['overlay']['site_inventory'])
            report['source_inputs_runtime_assets_rehashed_after']=True
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc))
        report['elapsed_seconds']=time.monotonic()-proof['started_monotonic']
        with(OUTPUT/(phase+'.json')).open('x+b')as f:
            os.fchmod(f.fileno(),0o400)
            def update():f.seek(0);f.write(encode(report));f.truncate();f.flush();os.fsync(f.fileno())
            try:
                rt.require((OUTPUT.lstat().st_dev,OUTPUT.lstat().st_ino,OUTPUT.lstat().st_uid)==(owner.st_dev,owner.st_ino,owner.st_uid),'Original native output replaced')
                update();rt.require(rt.identity(OUTPUT/(phase+'.json'),2 << 20)==public.pin(encode(report))
                    and all(rt.identity(OUTPUT/n,MAX_BANK)==p for n,p in artifacts.items()),'Native artifacts changed');check(deadline)
            except BaseException:report.update(status='fail',publication_failed=True);update()
        signal.alarm(0)
        for s,h in previous.items():signal.signal(s,h)
    return report


def validate_native(report,proof,revision,phase,proof_pin):
    rt.require(report['schema']==SCHEMA and report['stage']=='native_vcoco_hoi_'+phase and report['status']=='pass'and report['phase']=='complete'
        and report['producer_revision']==revision and report['image_id']==IMAGE and report['proof_identity']==proof_pin
        and report['source_inputs_runtime_assets_rehashed_after']is True and all(report[k]is False for k in
        ('reference_metadata_read','FIT_performed','ownership_verified','quality_verified','adoption','AMP_used','TF32_used')),'Actual complete native receipt required')
    if phase=='overlay':
        rt.require(report['models_loaded']==report['native_forward_calls']==0 and report['cuda_initialized']is report['base_installed']is False
            and report['full_import_closure_qualified']is True,'Original CPU-only overlay required');return
    rt.require(type(report['models_loaded'])is int and report['models_loaded']==1 and type(report['native_forward_calls'])is int
        and report['native_forward_calls']==len(report['images'])==16 and report['model']['strict_checkpoint']['keys']==1796
        and report['model']['strict_checkpoint']['strict']is report['model']['strict_checkpoint']['weights_only']is True
        and report['model']['checkpoint_buffer_schema']['ema_swapped']is False
,'One full original model/all16 forwards required')
    for ordinal,(r,image,bank)in enumerate(zip(report['images'],proof['images'],proof['banks'])):
        n,k,t=r['native_detections'],r['hand_object_pairs'],r['object_target_pairs'];rt.require(all(type(x)is int and 0<=x<=1_000_000 for x in (n,k,t))and n<=1000,'Exact native counts required')
        shapes=dict(query_logits=[1500,3],query_boxes_cxcywh=[1500,4],query_tokens=[1500,256],native_nms_keep=[r['arrays']['native_nms_detections']['shape'][0]],
            native_nms_detections=r['arrays']['native_nms_detections']['shape'],retained_nms_positions=[n],query_ids=[n],class_ids=[n],boxes_original_xyxy=[n,4],raw_scores=[n],decayed_scores=[n],
            hand_object_pairs=[k,2],hand_object_logits=[k,2],object_target_pairs=[t,2],object_target_logits=[t,2],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        ints={'native_nms_keep','retained_nms_positions','query_ids','class_ids','hand_object_pairs','object_target_pairs','image_size','original_frame_index','original_slot','acquired_ordinal'}
        rt.require(r['image_id']==image['image_id']and r['original_slot']==bank['original_slot']and r['acquired_ordinal']==ordinal and r['original_frame_index']==0
            and r['image_size']==bank['image_size']==[image['height'],image['width']]and r['endpoint_bank_identity']==bank['identity']
            and r['source_person_ids']==bank['person_ids']and r['owl_patches']==3600 and r['file']==f'image_{bank["original_slot"]:06d}.npz'
            and rt.identity(OUTPUT/r['file'],MAX_BANK)==r['identity']and set(r['arrays'])==set(FIELDS)
            and len(shapes['native_nms_detections'])==2 and shapes['native_nms_detections'][1]==5 and shapes['native_nms_detections'][0]<=1000
            and all(r['arrays'][name]['shape']==shape and r['arrays'][name]['dtype']==('<i8'if name in ints else '<f4')
                and re.fullmatch('[0-9a-f]{64}',str(r['arrays'][name]['sha256']))for name,shape in shapes.items()),'All native queries/pairs/aliases/grid fields required')


def cleanup(phase,name,revision,deadline):
    path=OUTPUT/(phase+'.cid')
    if path.exists():
        raw=path.read_bytes();rt.identity(path,65,readonly=False);rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Original owned CID required');cid=raw.decode().strip()
        present=public.bank.command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline);rt.require(present in ('',cid),'Ambiguous exact CID')
        if present:
            actual=public.bank.command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)
            rt.require(actual==IMAGE+'|/'+name+'|'+ENTRY+'|'+revision,'Foreign container never removed')
            public.bank.command(['docker','rm','-f',cid],deadline)
        rt.require(not public.bank.command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline),'Owned exact CID remains');path.chmod(0o400)
    rt.require(not public.bank.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Owned name remains')


def launch(code,revision,phase,proof,deadline):
    pin=public.bank.write(OUTPUT/(phase+'_proof.json'),proof);name='world-reward-vcoco-hoi-'+phase+'-'+revision[:12]
    files=[code/n for n in NATIVE_FILES]+[code.parent/n for n in ('revision','source-sha256')]+[Path(n)for n in proof['files']]
    args=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/(phase+'.cid')),'--label','world-reward.job='+ENTRY,
        '--label','world-reward.revision='+revision,'--network','none','--read-only','--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges',
        '--pids-limit','512','--cpus','4','--memory','64g'if phase=='model'else '16g','--tmpfs','/tmp:rw,exec,nosuid,size=2g','--entrypoint','/usr/bin/env']
    if phase=='model':args+=['--gpus','driver=nvidia,count=all']
    for path in files:
        rt.canonical(path);rt.require(path.is_file()and ','not in str(path)and '\n'not in str(path),'Literal readonly native leaf mount');args+=['--mount',f'type=bind,src={path},dst={path},readonly']
    args+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}']
    if phase=='model':args+=['--mount',f'type=bind,src={OUTPUT}/.overlay,dst={OUTPUT}/.overlay,readonly']
    args+=[IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/bin:/bin',
        'HOME=/tmp','PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=0'if phase=='model'else 'CUDA_VISIBLE_DEVICES=-1',
        'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','WR_IMAGE_ID='+IMAGE,'/opt/conda/bin/python','-I','-B',str(code/NATIVE_FILES[0]),
        '--phase',phase,'--proof-bytes',str(pin['bytes']),'--proof-sha256',pin['sha256'],'--deadline',format(deadline-60,'.17g'),
        '--code',str(code),'--revision',revision]
    rt.require(not public.bank.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Occupied original model namespace')
    try:
        result=subprocess.run(args,env=runtime.SAFE_ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=max(.001,deadline-time.monotonic()-60),check=False)
        rt.require(result.returncode==0,'Native child failed');report=rt.pinned(OUTPUT/(phase+'.json'),rt.identity(OUTPUT/(phase+'.json'),2 << 20),2 << 20)
        validate_native(report,proof,revision,phase,pin);return report
    finally:cleanup(phase,name,revision,min(deadline+60,time.monotonic()+30))


def run(code,revision):
    started=time.monotonic();deadline=started+BUDGET
    def expired(*_):raise TimeoutError('Inclusive original HOI budget')
    handlers={s:signal.signal(s,expired)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    before=inputs=qualified=models=manifest=prior=asset_states=None;lock_fd=None;owners=[];partials=[];directories=set()
    rt.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02','Exact original Azure host required')
    rt.require(not OUTPUT.exists()and not OUTPUT.is_symlink(),'Fresh observation namespace required');OUTPUT.mkdir(mode=0o700);s=OUTPUT.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
    report=dict(schema=SCHEMA,stage='vcoco_hoi_observations_host',status='fail',producer_revision=revision,image_id=IMAGE,images=[],
        source_inputs_runtime_assets_rehashed_after=False,owned_cleanup_verified=False,reference_metadata_read=False,FIT_performed=False,
        ownership_verified=False,quality_verified=False,adoption=False,host_network_used=False)
    try:
        before=source(code,revision);policy=rt.pinned(code/original.PROTOCOL,original.PROTOCOL_PIN,16 << 10)
        input_manifest,paths,inputs=public.export_inputs(code,before['source'])
        qualified=original.authenticate_runtime(rt,runtime,policy,deadline);rt.require(qualified['image']['Id']==IMAGE,'Exact original native image required')
        models,manifest=original.authenticate_acquisition(rt,acquisition,policy);prior=qualified_model()
        asset_states={n:public.snapshot(n)for n in {**models['frozen'],**qualified['frozen'],**prior['files']}}
        report.update(source_binding=before['source'],public_inputs_identity=public.PUBLIC,endpoint_producer_revision=public.REV,
            endpoint_receipt_pins=public.PINS,runtime_report_identity=qualified['report_identity'],acquisition_report_identity=models['report_identity'],qualified_model=QUALIFIED)
        for name in ('.overlay','.scratch'):
            folder=OUTPUT/name;folder.mkdir(mode=0o700);s=folder.lstat();owners.append((folder,(s.st_dev,s.st_ino,s.st_uid)))
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),runtime.NoRedirect())
        rows=[policy['fairscale'],policy['fairscale']['publisher_license'],*[r for w in policy['import_wheels']for r in (w,w['publisher_license'])]]
        rt.require(len(rows)==4 and sum(r['bytes']for r in rows)==284220,'Exact minimal original publisher assets required')
        report['downloaded_overlay_assets']=[];report['host_network_used']=True
        for row in rows:
            report['downloaded_overlay_assets'].append(runtime.fetch(atomic,acquisition,OUTPUT/'.scratch',row,deadline,partials,directories,opener))
        for row in (policy['fairscale']['publisher_license'],*[w['publisher_license']for w in policy['import_wheels']]):
            rt.write(OUTPUT/row['file'],(OUTPUT/'.scratch'/row['file']).read_bytes())
        patch=original.derive_source(rt,acquisition,manifest,OUTPUT/'.overlay/source',policy)
        public_inputs=public.bank.public_inputs(public.PUBLIC,16);native_bank=rt.pinned(public.bank.OUTPUT/'native.json',public.PINS['native.json'],2 << 20)
        native_files={str(paths[n]):p for n,p in input_manifest['files'].items()}
        proof=dict(source=before['source'],runtime=qualified,images=public_inputs['images'],banks=native_bank['images'],files={**native_files,
            **{str(original.DATA/'weights/epoch_5.pth'):policy['acquisition']['checkpoint']}},derived_source_inventory=original.inventory(rt,OUTPUT/'.overlay/source'),
            derived_source_patch=patch,started_monotonic=started,image_id=IMAGE)
        for row in rows:proof['files'][str(OUTPUT/'.scratch'/row['file'])]=original.row_pin(row)
        overlay=launch(code,revision,'overlay',proof,deadline);report['overlay_identity']=rt.identity(OUTPUT/'overlay.json',2 << 20)
        for folder,inode in list(owners):
            if folder.name=='.scratch':runtime.remove_owned_folder(rt,folder,inode,OUTPUT);owners.remove((folder,inode))
        proof['files']={n:p for n,p in proof['files'].items()if not Path(n).is_relative_to(OUTPUT/'.scratch')};proof['overlay']=overlay
        import fcntl
        lock=ROOT/'jobs/.world-reward-h100.lock';rt.canonical(lock);s=lock.lstat();rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Original cooperative GPU lock required')
        lock_fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);rt.require((os.fstat(lock_fd).st_dev,os.fstat(lock_fd).st_ino)==(s.st_dev,s.st_ino),'Original lock inode');fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt.require(not runtime.command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline),'GPU not idle after owned lock')
        native_report=launch(code,revision,'model',proof,deadline);report.update(native_identity=rt.identity(OUTPUT/'model.json',2 << 20),images=native_report['images'],status='pass')
    except BaseException as exc:report['error_type']=error(exc)
    finally:
        try:
            rt.require(before is not None and inputs is not None and models is not None and qualified is not None and prior is not None,'Incomplete authenticated preflight cannot PASS')
            rt.require(source(code,revision)==before and public.export_inputs(code,before['source'])[2]==inputs and qualified_model()==prior,
                'Whole original source/RGB/endpoint/model proof changed')
            rt.require(original.authenticate_acquisition(rt,acquisition,policy)[0]==models
                and original.authenticate_runtime(rt,runtime,policy,deadline)==qualified
                and {n:public.snapshot(n)for n in asset_states}==asset_states,'Original acquisition/runtime/stat identity changed')
            report['source_inputs_runtime_assets_rehashed_after']=True
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc))
        try:
            for phase in ('overlay','model'):cleanup(phase,'world-reward-vcoco-hoi-'+phase+'-'+revision[:12],revision,min(deadline+60,time.monotonic()+30))
            for folder,inode in owners:
                if folder.exists():runtime.remove_owned_folder(rt,folder,inode,OUTPUT)
            report['owned_cleanup_verified']=True
        except BaseException as exc:report.update(status='fail',cleanup_error_type=error(exc))
        finally:
            if lock_fd is not None:os.close(lock_fd)
        names={'FairScale.LICENSE','terminaltables.LICENSE','overlay_proof.json','model_proof.json','overlay.json','model.json','overlay.cid','model.cid'}|{f'image_{i:06d}.npz'for i in range(16)}
        allowed={p.name for p in OUTPUT.iterdir()}&names
        public.publish(OUTPUT,report,deadline,started,owner,allowed)
        signal.alarm(0)
        for s,h in handlers.items():signal.signal(s,h)
    return report


def arguments(argv):
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--phase',choices=('overlay','model'));p.add_argument('--proof-bytes',type=int)
    p.add_argument('--proof-sha256');p.add_argument('--deadline',type=float);p.add_argument('--code');p.add_argument('--revision');a=p.parse_args(argv)
    if a.phase:
        rt.require(type(a.deadline)is float and math.isfinite(a.deadline)and type(a.code)is str and re.fullmatch('[0-9a-f]{40}',str(a.revision)), 'Explicit native source/deadline required')
        a.pin=public.fixed_pin(dict(bytes=a.proof_bytes,sha256=a.proof_sha256),2 << 20)
    else:rt.require(all(getattr(a,n)is None for n in ('proof_bytes','proof_sha256','deadline','code','revision')),'Host cannot override frozen inputs')
    return a


if __name__=='__main__':
    args=arguments(sys.argv[1:]);value=native(Path(args.code),args.revision,args.phase,args.pin,args.deadline)if args.phase else run(Path(os.environ.get('WR_CODE','/invalid')),os.environ.get('WR_CODE_REVISION',''))
    print(encode({k:value[k]for k in ('stage','status')}).decode(),end='');raise SystemExit(0 if value['status']=='pass'else 1)
