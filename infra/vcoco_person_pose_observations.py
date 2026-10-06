"""One qualified CPU DWPose session on ALL retained V-COCO replica persons."""
import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_observation_replica as replica
import vcoco_replica_completion as completion
import sealed_callback_publication as publication
import dwpose_acquire as acquisition
import dwpose_wheel_audit as notices
rt,ROOT=replica.rt,replica.ROOT
ENTRY='run_vcoco_person_pose_observations'
OUTPUT=ROOT/'results/vcoco-person-pose-observations-v1'
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
SCHEMA='world_reward.vcoco_person_pose_observations.v1'
BUDGET=600
FIELDS=('person_ids','boxes_original_xyxy','detector_scores','keypoints_original_xy','raw_scores',
    'native_valid','in_original_image','image_size','original_frame_index','original_slot','acquired_ordinal')
FROZEN={'infra/dwpose_smoke.py':dict(bytes=28479,sha256='ea0beb43dd698261a5b2accdbd0432dcc8de82b56958b0e93cec8062089ce72c'),
 'infra/keypoint_rgb_dwpose.py':dict(bytes=16781,sha256='f6fbf3e578e2a3c603a9644f341401576cc6360694ec52cf909907ba00771121'),
 'src/world_reward/person_pose_observations.py':dict(bytes=7614,sha256='c93fa24f4ce4bd6b5f4c53f4e48943b2c61d0c20a2211b36ef6d0ccb6b3ce375'),
 'infra/coco_endpoint_evaluate.py':dict(bytes=38477,sha256='f1bf582b704b06891efbda4f13d643c214379e92a997b38506e7d33b13085685')}
NATIVE_FILES=tuple(dict.fromkeys(('infra/vcoco_person_pose_observations.py','infra/coco_endpoint_evaluate.py',
 'infra/dwpose_smoke.py','infra/run_dwpose_smoke.sh','infra/keypoint_rgb_dwpose.py','infra/dwpose_acquire.py','infra/dwpose_wheel_audit.py',
 'src/world_reward/person_pose_observations.py',*replica.HELPERS,*completion.HELPERS)))
HELPERS=(*NATIVE_FILES,'infra/run_vcoco_person_pose_observations.sh')
encode,error=replica.encode,replica.bank.error


def source(code,revision):
    proof=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    rt.require(Path(__file__).resolve()==code/'infra/vcoco_person_pose_observations.py'
        and Path(replica.__file__).resolve()==code/'infra/vcoco_observation_replica.py'
        and Path(acquisition.__file__).resolve()==code/'infra/dwpose_acquire.py'
        and Path(notices.__file__).resolve()==code/'infra/dwpose_wheel_audit.py'
        and Path(completion.__file__).resolve()==code/'infra/vcoco_replica_completion.py'
        and Path(publication.__file__).resolve()==code/'infra/sealed_callback_publication.py','Actual unchanged helper origins required')
    rt.require(all(proof['helpers'][n]==p for n,p in FROZEN.items()),'Frozen qualified numerical helper changed')
    return dict(source=proof,states=replica.bank.source_state(code),markers={n:replica.snapshot(code.parent/n)for n in ('revision','source-sha256')})


def assets():
    notices.validate_previous(ROOT,acquisition);notices.validate_failed(ROOT,acquisition)
    files={str(ROOT/acquisition.BASE/n):rt.identity(ROOT/acquisition.BASE/n,200 << 20)for n,*_ in acquisition.ASSETS}
    for name in (acquisition.REPORT,notices.PREVIOUS_REPORT,notices.OUT+'/report.json',
        'validation/dwpose_smoke_v1/report.json','validation/dwpose_smoke_v2/report.json'):
        files[str(ROOT/name)]=rt.identity(ROOT/name,2 << 20)
    audit=rt.pinned(ROOT/notices.OUT/'report.json',dict(bytes=rt.identity(ROOT/notices.OUT/'report.json')['bytes'],
        sha256='e7fa6fce0397654ec5d1d2c07c49bd6f2a50655cda185d4297bfb8f0f4aae3e2'),2 << 20)
    rt.require(audit['status']=='pass'and audit['stage']=='pinned_dwpose_wheels_notice_audit_v3','Original completed wheel audit required')
    for row in audit['wheel_audits']:
        for item in row['retained_texts']:
            path=ROOT/notices.OUT/row['package']/str(acquisition.safe_relative(item['file']))
            rt.require(rt.identity(path,2 << 20)=={k:item[k]for k in ('bytes','sha256')},'Original primary notices changed');files[str(path)]=rt.identity(path,2 << 20)
    smoke=ROOT/'validation/dwpose_smoke_v2/report.json'
    rt.require(files[str(smoke)]['sha256']=='d96eb5c8c4030bf2e16924093aef03a9636f12cc2f3d3a8b7475731fe49d9a79','Original qualified DWPose receipt differs')
    return dict(files=files,states={n:replica.snapshot(n)for n in files})


def replica_inputs(code,replica_revision,pins):
    old=ROOT/'jobs'/replica_revision/replica.ENTRY/'code';oldsource=rt.source(ROOT,old,replica_revision,replica.ENTRY,replica.HELPERS)
    rt.require(all(rt.identity(code/n,2 << 20,empty=True)==oldsource['helpers'][n]for n in replica.HELPERS),
        'All current replica helpers must equal original imported source')
    receipt_dir=ROOT/f'results/vcoco-observation-replica-import-{replica_revision}'
    values={n:rt.pinned(receipt_dir/name,pins[n],256 << 10)for n,name in
        (('import','report.json'),('manifest','manifest.json'),('export','export-receipt.json'))}
    report,manifest,export=(values[n]for n in ('import','manifest','export'))
    replica_snapshot=replica.replica_state()
    replica.validate_manifest(manifest,export['producer_revision'])
    rt.require(report['schema']==replica.SCHEMA and report['phase']=='import'and report['status']=='pass'
        and report['producer_revision']==replica_revision and report['source_binding']==oldsource
        and report['source_inputs_rehashed_after']is report['outputs_sealed']is report['blob_cleanup_verified']is True
        and report['manifest_identity']==pins['manifest']and report['export_receipt_identity']==pins['export']
        and report['archive_identity']==export['archive_identity']and report['blob_etag']==export['blob_etag']
        and report['replica_directory']==str(replica.DEST)and report['original_source_on_receiver_live_verified']is False,
        'Actual sealed imported replica required')
    replica.receipt_from_base64(__import__('base64').b64encode((receipt_dir/'export-receipt.json').read_bytes()).decode(),pins['export'],export['producer_revision'])
    rt.require(export['manifest_identity']==pins['manifest']and {p.name for p in receipt_dir.iterdir()}=={'report.json','manifest.json','export-receipt.json'}
        and stat.S_IMODE(receipt_dir.lstat().st_mode)==0o500 and receipt_dir.lstat().st_uid==0,'Exclusive readonly imported evidence required')
    files=replica.replica_identity(manifest)
    public=replica.bank.rgb_inputs.read_inputs(replica.DEST/'inputs',replica.PUBLIC,16,identity=rt.identity,pinned=rt.pinned,maximum_slots=16)
    native=rt.pinned(replica.DEST/'banks/native.json',replica.PINS['native.json'],256 << 10)
    rt.require(len(native['images'])==16 and all(r['original_slot']==i and r['original_frame_index']==0
        and r['image_id']==public['images'][i]['image_id']and r['identity']==files['banks/'+r['file']]
        for i,r in enumerate(native['images'])),'Original all16 slot/person-bank mapping required')
    paths={str(replica.DEST/n):p for n,p in files.items()}
    paths.update({str(receipt_dir/name):pins[n]for n,name in (('import','report.json'),('manifest','manifest.json'),('export','export-receipt.json'))})
    return dict(manifest=manifest,images=public['images'],banks=native['images'],files=paths,
        states={str(p):replica.snapshot(p)for folder in (replica.DEST,receipt_dir)for p in (folder,*sorted(folder.rglob('*')))},
        original_replica_source=oldsource,original_source_states=replica.bank.source_state(old),replica_snapshot=replica_snapshot)


def completed_replica_inputs(code,revision,pin):
    value=completion.authenticate_completion(code,revision,pin);path=completion.OUTPUT/'report.json'
    rt.require(rt.identity(path,512 << 10)==pin,'Actual completion receipt changed')
    return dict(value,files={**value['files'],str(path):pin},
        states={**value['states'],str(completion.OUTPUT):replica.snapshot(completion.OUTPUT),str(path):replica.snapshot(path)},
        original_source_states=value['original_sources'][completion.IMPORT]['states'])


def native_source(code,revision,proof):
    rt.require(proof['source']['producer_revision']==revision and Path(__file__).resolve()==code/'infra/vcoco_person_pose_observations.py',
        'Original native source declaration required')
    for n in ('revision','source-sha256'):
        rt.require(rt.identity(code.parent/n,100)==proof['source']['markers'][n],'Original readonly source markers differ')
    rt.require((code.parent/'revision').read_bytes()==(revision+'\n').encode(),'Exact native revision marker')
    rows={}
    for n in NATIVE_FILES:
        pin=rt.identity(code/n,2 << 20,empty=True);rt.require(pin==proof['source']['helpers'][n],'Every actual mounted helper differs')
        rows[n]=dict(pin=pin,state=replica.snapshot(code/n))
    return rows


def observe(images,banks,load_bank,decode,infer,save,check,records=None):
    """Validate EVERY complete bank before the first pose, preserve all person rows."""
    rt.require(type(images)is list and type(banks)is list and len(images)==len(banks), 'Aligned complete original RGB/bank rows required')
    rows=[]if records is None else records
    for bank in banks:check();load_bank(bank,validate=True)
    expected=0
    for ordinal,(image,bank)in enumerate(zip(images,banks)):
        check();a=load_bank(bank,validate=False);rgb=decode(image)
        ids=tuple(a['person_retained_ids'].tolist());expected+=len(ids)
        rt.require(list(ids)==bank['person_ids']and len(ids)==bank['person_retained_rows'], 'Every recorded person ID required')
        fingerprints={n:(a[n].dtype.str,a[n].shape,hashlib.sha256(a[n].tobytes()).hexdigest())for n in a}
        result=infer(rgb,ids,a['person_retained_boxes'],a['person_retained_scores'])
        rt.require(result.person_ids==ids and result.original_frame_index==0 and result.image_size==tuple(bank['image_size']),
            'Every original person/grid in original order required')
        rt.require(fingerprints=={n:(a[n].dtype.str,a[n].shape,hashlib.sha256(a[n].tobytes()).hexdigest())for n in a},'Supplied original person bank mutated')
        rows.append(save(result,image,bank,ordinal));check()
    return rows,expected


def cpu(code,revision,proof_pin,deadline):
    import numpy as np
    import dwpose_smoke as dw
    import keypoint_rgb_dwpose as capability
    import coco_endpoint_evaluate as saved
    from world_reward.person_pose_observations import infer_person_pose_frame
    for module,name in ((dw,'infra/dwpose_smoke.py'),(capability,'infra/keypoint_rgb_dwpose.py'),(saved,'infra/coco_endpoint_evaluate.py')):
        rt.require(Path(module.__file__).resolve()==code/name,'Qualified native helper origin required')
    proof=rt.pinned(OUTPUT/'proof.json',proof_pin,2 << 20);before_source=native_source(code,revision,proof)
    rt.require(os.environ.get('WR_IMAGE_ID')==IMAGE and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Offline exactB47 CPU required')
    rt.require(proof['image_id']==IMAGE,'Native host image differs')
    owner=OUTPUT.lstat();report=dict(schema=SCHEMA,stage='native_vcoco_person_pose_observations',status='fail',phase='preflight',
        producer_revision=revision,image_id=IMAGE,proof_identity=proof_pin,images=[],native_calls=[],
        models_loaded=0,GPU_used=False,reference_metadata_read=False,FIT_performed=False,ownership_verified=False,
        quality_verified=False,adoption=False,full_image_fallback=False,source_inputs_assets_rehashed_after=False)
    before_files={n:rt.identity(n,200 << 20)for n in proof['files']};prefix=None
    input_states={n:replica.snapshot(n)for n in proof['files']};artifact_pins={}
    previous={s:signal.signal(s,lambda *_:(_ for _ in()).throw(TimeoutError('Native inclusive CPU deadline')))for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    try:
        rt.require(before_files==proof['files'],'Actual public/assets byte identities differ')
        dw.source_identity();report['capability']=capability.validate_smoke(ROOT);report['asset_qualification']=dw.validate_assets(ROOT)
        report['dependencies']=dw.dependency_identity();rt.require(not any(n in sys.modules or importlib.util.find_spec(n)is not None for n in ('onnxruntime','flatbuffers')),'Only original disposable runtime overlay')
        def load(row,validate):
            path=replica.DEST/'banks'/row['file']
            if validate:saved.validate_npz(path,row);return None
            with np.load(path,allow_pickle=False)as z:return {n:z[n]for n in ('person_retained_ids','person_retained_boxes','person_retained_scores')}
        def save(result,image,bank,ordinal):
            rt.require(result.keypoints_original_xy.dtype==np.float64 and result.raw_scores.dtype==np.float32,
                'Original native decoded float64 coordinates/float32 scores required')
            arrays={n:getattr(result,n)for n in FIELDS[:7]if n!='person_ids'}
            arrays['person_ids']=np.array(result.person_ids,dtype=load(bank,False)['person_retained_ids'].dtype)
            arrays.update(image_size=np.asarray(result.image_size,np.int64),original_frame_index=np.asarray(0,np.int64),
                original_slot=np.asarray(bank['original_slot'],np.int64),acquired_ordinal=np.asarray(ordinal,np.int64))
            path=OUTPUT/f'image_{bank["original_slot"]:06d}.npz'
            with path.open('xb')as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o400)
            artifact_pins[path.name]=rt.identity(path,2 << 20)
            return dict(image_id=image['image_id'],original_slot=bank['original_slot'],acquired_ordinal=ordinal,original_frame_index=0,
                image_size=list(result.image_size),person_ids=list(result.person_ids),persons=len(result.person_ids),
                file=path.name,identity=rt.identity(path,2 << 20),arrays={n:dict(shape=list(a.shape),dtype=a.dtype.str,
                    sha256=hashlib.sha256(a.tobytes()).hexdigest())for n,a in arrays.items()})
        with dw.private_prefix(report,lambda:None)as prefix:
            report['phase']='private_overlay';subprocess.run(dw.pip_argv(prefix,ROOT),check=True,capture_output=True,
                timeout=max(.001,deadline-time.monotonic()))
            ort,imports=dw.import_runtime(prefix);report['runtime_imports']=imports
            spec=importlib.util.spec_from_file_location('world_reward_vcoco_original_dwpose',ROOT/acquisition.BASE/'source/onnxpose.py')
            original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            session=ort.InferenceSession(str(ROOT/acquisition.BASE/acquisition.ASSETS[0][0]),sess_options=options,providers=['CPUExecutionProvider'])
            session.disable_fallback();dw.validate_session(session);dw.validate_options(session,ort)
            proxy=dw.SessionProxy(session,report['native_calls'],lambda:None);report['models_loaded']=1;report['phase']='all_person_observations'
            rows,expected=observe(proof['images'],proof['banks'],load,
                lambda r:replica.bank.rgb_inputs.decode_rgb(replica.DEST/'inputs',r,identity=rt.identity),
                lambda rgb,ids,boxes,scores:infer_person_pose_frame(original.inference_pose,proxy,rgb,0,ids,boxes,scores),
                save,lambda:replica.check(deadline),report['images'])
            rt.require(len(rows)==16 and len(report['native_calls'])==expected and all(r['run_completed']for r in report['native_calls']),
                'One original batch-one native crop for every retained person')
            report['persons']=expected;del proxy,session
        report['phase']='complete';replica.check(deadline);report['status']='pass'
    except BaseException as exc:report['error_type']=error(exc)
    finally:
        try:
            rt.require(native_source(code,revision,proof)==before_source and {n:rt.identity(n,200 << 20)for n in proof['files']}==before_files==proof['files'],
                'Complete source/public/asset changed')
            rt.require({n:replica.snapshot(n)for n in proof['files']}==input_states,'Original file inode/modes changed')
            rt.require(prefix is None or not prefix.exists(),'Disposable original CPU overlay survives');report['source_inputs_assets_rehashed_after']=True
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc))
        allowed={'proof.json'}|set(artifact_pins)|({'.container.cid'}if(OUTPUT/'.container.cid').exists()else set())
        with(OUTPUT/'native.json').open('x+b')as f:
            os.fchmod(f.fileno(),0o400);opened=os.fstat(f.fileno())
            def update():f.seek(0);f.write(encode(report));f.truncate();f.flush();os.fsync(f.fileno())
            try:
                rt.require({p.name for p in OUTPUT.iterdir()}==allowed|{'native.json'}
                    and (OUTPUT.lstat().st_dev,OUTPUT.lstat().st_ino,OUTPUT.lstat().st_uid)==(owner.st_dev,owner.st_ino,owner.st_uid)
                    and (OUTPUT/'native.json').lstat().st_ino==opened.st_ino,'Unknown/replaced native artifact')
                update();rt.require(rt.identity(OUTPUT/'native.json',2 << 20)==replica.pin(encode(report))
                    and all(rt.identity(OUTPUT/n,2 << 20)==p for n,p in artifact_pins.items()),'Native publication differs');replica.check(deadline)
            except BaseException:report.update(status='fail',publication_failed=True);update()
        signal.alarm(0)
        for s,handler in previous.items():signal.signal(s,handler)
    return report


def validate_native(report,proof,revision,proof_pin):
    rt.require(report['schema']==SCHEMA and report['stage']=='native_vcoco_person_pose_observations'and report['status']=='pass'
        and report['phase']=='complete'and report['producer_revision']==revision and report['image_id']==IMAGE
        and report['proof_identity']==proof_pin and type(report['models_loaded'])is int and report['models_loaded']==1 and len(report['images'])==16
        and report['source_inputs_assets_rehashed_after']is True and report['private_prefix_removed']is True
        and all(report[k]is False for k in ('GPU_used','reference_metadata_read','FIT_performed','ownership_verified','quality_verified','adoption','full_image_fallback')),
        'Complete native all-person CPU observation required')
    total=0
    for ordinal,(row,image,bank)in enumerate(zip(report['images'],proof['images'],proof['banks'])):
        n=bank['person_retained_rows'];total+=n
        rt.require(row['image_id']==image['image_id']and row['original_slot']==bank['original_slot']and row['acquired_ordinal']==ordinal
            and row['person_ids']==bank['person_ids']and row['persons']==n and row['original_frame_index']==0 and row['image_size']==bank['image_size']
            and row['file']==f'image_{bank["original_slot"]:06d}.npz'and rt.identity(OUTPUT/row['file'],2 << 20)==row['identity']
            and set(row['arrays'])==set(FIELDS),'All original IDs/rows/arrays retained')
        shapes=dict(person_ids=[n],boxes_original_xyxy=[n,4],detector_scores=[n],keypoints_original_xy=[n,133,2],raw_scores=[n,133],
            native_valid=[n,133],in_original_image=[n,133],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        rt.require(type(row['persons'])is int and type(row['original_slot'])is int and type(row['acquired_ordinal'])is int
            and type(row['original_frame_index'])is int,'Exact original integer slots/counts required')
        rt.require(row['arrays']['boxes_original_xyxy']['sha256']==bank['arrays']['person_retained_boxes']['sha256']
            and row['arrays']['detector_scores']['sha256']==bank['arrays']['person_retained_scores']['sha256']
            and row['arrays']['person_ids']['sha256']==bank['arrays']['person_retained_ids']['sha256'],'Original person input arrays unchanged')
        rt.require(all(type(row['arrays'][k])is dict and set(row['arrays'][k])=={'shape','dtype','sha256'}
            and row['arrays'][k]['shape']==s and re.fullmatch('[0-9a-f]{64}',str(row['arrays'][k]['sha256']))for k,s in shapes.items()),'Complete133 native ABI required')
        types=dict(boxes_original_xyxy=bank['arrays']['person_retained_boxes']['dtype'],detector_scores=bank['arrays']['person_retained_scores']['dtype'],
            person_ids=bank['arrays']['person_retained_ids']['dtype'],keypoints_original_xy='<f8',raw_scores='<f4',native_valid='|b1',
            in_original_image='|b1',image_size='<i8',original_frame_index='<i8',original_slot='<i8',acquired_ordinal='<i8')
        rt.require(all(row['arrays'][k]['dtype']==t for k,t in types.items()),'Native original dtype preserved')
    rt.require(type(report['persons'])is int and report['persons']==total and len(report['native_calls'])==total and all(x['run_completed']for x in report['native_calls']),
        'One session and all person calls required')
    for call in report['native_calls']:
        rt.require(call['supplied_container']=='list'and call['effective_feed_shape']==[1,3,384,288]
            and call['delegated_unmodified']is True and call['effective_runtime_conversion_observed']is False
            and call['supplied_array']['dtype']=='float64'and call['supplied_array']['shape']==[3,384,288]
            and [(x['dtype'],x['shape'])for x in call['raw_simcc']]==[('float32',[1,133,576]),('float32',[1,133,768])],
            'Original native feed/SimCC ABI required')


def command(args,deadline):return replica.bank.command(args,deadline)
def image_state(deadline):
    raw=command(['docker','image','inspect',IMAGE,'--format','{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'],deadline)
    v=rt.strict(raw);rt.require(v['Id']==IMAGE and v['Architecture']=='amd64'and v['Os']=='linux'and v['RootFS']['Type']=='layers','ExactB47 CPU image required');return v

def cleanup(name,revision,deadline):
    path=OUTPUT/'.container.cid'
    if path.exists():
        rt.identity(path,65,readonly=False);raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned CPU CID required');cid=raw.decode().strip()
        found=command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline);rt.require(found in('',cid),'Ambiguous CID')
        if found:
            expected=IMAGE+'|/'+name+'|'+ENTRY+'|'+revision
            actual=command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)
            rt.require(actual==expected,'Refuse foreign container removal');command(['docker','rm','-f',cid],deadline)
        rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline),'Owned exactCID survives');path.chmod(0o400)
    rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Owned exactname survives')


def dispatch(code,revision,replica_revision,pins,completion_context=None):
    started=time.monotonic();deadline=started+BUDGET
    def expired(*_):raise TimeoutError('Inclusive host all-person budget')
    previous={s:signal.signal(s,expired)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    try:return dispatch_work(code,revision,replica_revision,pins,started,deadline,completion_context)
    finally:
        signal.alarm(0)
        for s,handler in previous.items():signal.signal(s,handler)


def dispatch_work(code,revision,replica_revision,pins,started,deadline,completion_context=None):
    rt.require(completion_context is None or replica_revision is None and pins=={},'No mixed completion/import context')
    load_inputs=lambda:completed_replica_inputs(code,*completion_context)if completion_context is not None else replica_inputs(code,replica_revision,pins)
    replica.transport.verify_azure_peer('import');before=source(code,revision);inputs=load_inputs();models=assets();image=image_state(deadline)
    rt.require(not OUTPUT.exists()and not OUTPUT.is_symlink(),'Fresh all-person observations required');OUTPUT.mkdir(mode=0o700);s=OUTPUT.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
    report=dict(schema=SCHEMA,stage='vcoco_person_pose_observations_host',status='fail',producer_revision=revision,image_id=IMAGE,
        source_binding=before['source'],source_stat_identity=before['states'],replica_revision=replica_revision,replica_pins=pins,
        ownership_verified=False,quality_verified=False,adoption=False,FIT_performed=False,GPU_used=False,reference_metadata_read=False,
        owned_cleanup_verified=False,source_inputs_assets_rehashed_after=False)
    lineage=dict(input_mode='completed_replica'if completion_context is not None else'original_import',
        completion_revision=completion_context[0]if completion_context is not None else None,
        completion_identity=completion_context[1]if completion_context is not None else None,
        completion_source=inputs['completion_source']if completion_context is not None else None)
    report.update(lineage)
    name='world-reward-vcoco-person-pose-'+revision[:12]
    try:
        rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Occupied owned CPU name')
        proof=dict(source=before['source'],replica_pins=pins,original_replica_source=inputs['original_replica_source'],
            images=inputs['images'],banks=inputs['banks'],files={**inputs['files'],**models['files']},image_id=IMAGE)
        proof.update(lineage)
        proof_pin=replica.bank.write(OUTPUT/'proof.json',proof)
        mounts=[code/n for n in NATIVE_FILES]+[code.parent/n for n in ('revision','source-sha256')]+[Path(n)for n in proof['files']]
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/'.container.cid'),
            '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,'--network','none','--user','0:0',
            '--memory','8g','--cpus','4','--pids-limit','256','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges',
            '--tmpfs','/tmp:rw,exec,nosuid,size=512m','--entrypoint','/usr/bin/env']
        for path in mounts:
            rt.canonical(path);rt.require(path.is_file()and ','not in str(path)and '\n'not in str(path),'Individual readonly file mounts')
            cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp',
            'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','WR_IMAGE_ID='+IMAGE,
            '/opt/conda/bin/python','-I','-B',str(code/'infra/vcoco_person_pose_observations.py'),'--native',
            '--proof-bytes',str(proof_pin['bytes']),'--proof-sha256',proof_pin['sha256'],'--deadline',format(deadline-20,'.17g'),
            '--code',str(code),'--revision',revision]
        result=subprocess.run(cmd,env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,timeout=max(.001,deadline-time.monotonic()-20))
        report['native_exit_status']=result.returncode;report['native_identity']=rt.identity(OUTPUT/'native.json',2 << 20)
        native=rt.pinned(OUTPUT/'native.json',report['native_identity'],2 << 20)
        rt.require(result.returncode==0,'NativeCPU failed');validate_native(native,proof,revision,proof_pin);report['images']=native['images']
        report['persons']=native['persons'];report['status']='pass'
    except BaseException as exc:report['error_type']=error(exc)
    finally:
        try:cleanup(name,revision,min(deadline+15,time.monotonic()+10));report['owned_cleanup_verified']=True
        except BaseException as exc:report.update(status='fail',cleanup_error_type=error(exc))
        try:
            rt.require(source(code,revision)==before and load_inputs()==inputs and assets()==models
                and image_state(deadline)==image,'Original source/replica/assets/image changed');report['source_inputs_assets_rehashed_after']=True
            if 'images'in report:validate_native(native,proof,revision,proof_pin)
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc))
        s=OUTPUT.lstat();rt.require((s.st_dev,s.st_ino,s.st_uid)==owner,'Original output directory replaced')
        allowed={'proof.json','native.json'}|({'.container.cid'}if(OUTPUT/'.container.cid').exists()else set())|{f'image_{i:06d}.npz'for i in range(16)if(OUTPUT/f'image_{i:06d}.npz').exists()}
        rt.require({p.name for p in OUTPUT.iterdir()}<=allowed,'Foreign observation outputs')
        allowed={p.name for p in OUTPUT.iterdir()}
        publication.publish(OUTPUT,report,deadline,started,owner,allowed,
            encode=encode,identity=rt.identity,snapshot=replica.snapshot,require=rt.require,
            check=replica.check,sync=replica.sync,error=error,report_maximum=2 << 20)
    return report


def arguments(argv):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('--native',action='store_true')
    p.add_argument('--replica-revision');p.add_argument('--proof-bytes',type=int);p.add_argument('--proof-sha256');p.add_argument('--deadline',type=float);p.add_argument('--code');p.add_argument('--revision')
    p.add_argument('--completion-revision');p.add_argument('--completion-bytes',type=int);p.add_argument('--completion-sha256')
    for n in ('import','manifest','export'):p.add_argument('--'+n+'-bytes',type=int);p.add_argument('--'+n+'-sha256')
    a=p.parse_args(argv);a.completion_context=None
    completion_keys=('completion_revision','completion_bytes','completion_sha256')
    original_keys=('replica_revision','import_bytes','import_sha256','manifest_bytes','manifest_sha256','export_bytes','export_sha256')
    if a.native:
        rt.require(all(getattr(a,k)is None for k in (*original_keys,*completion_keys)),
            'Native cannot accept alternate input pins')
        rt.require(type(a.deadline)is float and math.isfinite(a.deadline)and type(a.code)is str and re.fullmatch('[0-9a-f]{40}',str(a.revision)), 'Explicit native origin/deadline')
        a.proof_pin=replica.fixed_pin(dict(bytes=a.proof_bytes,sha256=a.proof_sha256),2 << 20)
    else:
        rt.require(all(getattr(a,k)is None for k in ('proof_bytes','proof_sha256','deadline','code','revision')), 'Host cannot override native origin')
        if any(getattr(a,k)is not None for k in completion_keys):
            rt.require(all(getattr(a,k)is None for k in original_keys)and re.fullmatch('[0-9a-f]{40}',str(a.completion_revision)),
                'Completion and original import inputs are mutually exclusive')
            a.completion_context=(a.completion_revision,replica.fixed_pin(dict(bytes=a.completion_bytes,sha256=a.completion_sha256),512 << 10));a.pins={}
        else:
            rt.require(re.fullmatch('[0-9a-f]{40}',str(a.replica_revision)), 'Actual imported producer revision required')
            a.pins={n:replica.fixed_pin(dict(bytes=getattr(a,n+'_bytes'),sha256=getattr(a,n+'_sha256')),256 << 10)for n in ('import','manifest','export')}
    return a


if __name__=='__main__':
    a=arguments(sys.argv[1:]);rt.require(sys.platform=='linux'and os.geteuid()==0,'Azure rootCPU-only caller')
    result=cpu(Path(a.code),a.revision,a.proof_pin,a.deadline)if a.native else dispatch(Path(os.environ.get('WR_CODE','/invalid')),
        os.environ.get('WR_CODE_REVISION',''),a.replica_revision,a.pins,a.completion_context)
    print(encode({k:result[k]for k in ('stage','status')}).decode(),end='');raise SystemExit(0 if result['status']=='pass'else 1)
