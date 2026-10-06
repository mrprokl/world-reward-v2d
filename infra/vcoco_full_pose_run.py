"""One original CPU DWPose session on the complete public48 endpoint replica."""
import argparse
import ast
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
from types import SimpleNamespace

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import mediapipe_cpu_runtime_verify as rt
import vcoco_public_observation_context as public
import vcoco_cardinality_observation_adapters as adapters
import vcoco_public_pose_core as core

ROOT=rt.ROOT
ENTRY='run_vcoco_full_pose_run'
OUTPUT=ROOT/'results/vcoco-full-person-pose-v2'
DEST=Path('/srv/world-reward-data/vcoco_full_public_replica_v1')
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
SCHEMA='world_reward.vcoco_full_person_pose.v1'
BUDGET=600
SMOKE_PIN=dict(bytes=28479,sha256='ea0beb43dd698261a5b2accdbd0432dcc8de82b56958b0e93cec8062089ce72c')
FROZEN={'infra/dwpose_smoke.py':SMOKE_PIN,
 'infra/vcoco_public_pose_core.py':dict(bytes=2117,sha256='01dde489499461d3a4c62cc00e49efa941635a516ac4ba3c58ef36d24fef0f10'),
 'src/world_reward/person_pose_observations.py':dict(bytes=7614,sha256='c93fa24f4ce4bd6b5f4c53f4e48943b2c61d0c20a2211b36ef6d0ccb6b3ce375')}
NATIVE_FILES=tuple(dict.fromkeys(('infra/vcoco_full_pose_run.py','infra/vcoco_public_pose_core.py','infra/dwpose_smoke.py',*public.HELPERS)))
RUNTIME_NAMES=('private_prefix','dependency_identity','import_runtime','array_identity','session_metadata','validate_session','validate_options','SessionProxy')
POSE_REVISION='df7db36998df454d022ff8f593b8d59036c6d771'
POSE_CLOSURE='ae1b70b2fb2caa62432382ae001de65bf74ea25356bea0286c26f9351fc52de1'
POSE_XZ='fa9fb811b359963e159f0fd30e6acfe34b6fa73d7be8518c50b4f1560c630613'
NATIVE_ASSETS=tuple(dict(path=str(ROOT/'weights/dwpose_native_v1'/name),identity=dict(bytes=size,sha256=sha))for name,size,sha in(
    ('dw-ll_ucoco_384.onnx',134399116,'724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843'),
    ('source/onnxpose.py',11608,'16fb69ab54f5e1ce8a5ad186e92f357da0162fc2ca2eeec4ccf1db72949291a2'),
    ('wheels/onnxruntime-1.30.0-cp311-cp311-manylinux_2_28_x86_64.whl',23561046,'fd54b314ea385bcecac69ab431f020ba503e3878dad4ebb645fec5a24b041242'),
    ('wheels/flatbuffers-25.12.19-py2.py3-none-any.whl',26661,'7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4')))
FLAGS=('reference_metadata_read','split_metadata_read','FIT_performed','GPU_used','ground_truth_used','challenge_inputs_used',
       'actor_selection_performed','anatomical_ownership_verified','quality_verified','adoption','full_image_fallback')
encode=public.endpoint.original.encode
error=public.endpoint.original.error
check=public.endpoint.check


def host_modules():
    import vcoco_full_public_replica as transfer
    import vcoco_person_pose_observations as original
    import sealed_callback_publication as publisher
    return transfer,original,publisher


def helpers():
    transfer,original,_=host_modules()
    import dwpose_metadata_context as metadata
    return tuple(dict.fromkeys((*NATIVE_FILES,'infra/run_vcoco_full_pose_run.sh','infra/sealed_callback_publication.py',*metadata.HELPERS,*transfer.HELPERS,*original.HELPERS)))


def state(path):return public._state(path)

def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def source(code,revision):
    value=rt.source(ROOT,code,revision,ENTRY,helpers())
    rt.require(Path(__file__).resolve()==code/'infra/vcoco_full_pose_run.py','Current caller origin required')
    rt.require(all(value['helpers'][n]==pin for n,pin in FROZEN.items()),'Original numerical helper changed')
    return dict(proof=value,states={str(p):state(p)for p in(code,*sorted(code.rglob('*')))},
                markers={n:state(code.parent/n)for n in('revision','source-sha256')})


def asset_context(code):
    _,original,_=host_modules()
    old=ROOT/'jobs'/POSE_REVISION/original.ENTRY/'code'
    binding=rt.source(ROOT,old,POSE_REVISION,original.ENTRY,original.HELPERS)
    rt.require(binding['closure_sha256']==POSE_CLOSURE and binding['entries']==352
        and (old.parent/'source-sha256').read_bytes()==(POSE_XZ+'\n').encode()
        and all(rt.identity(code/n,2<<20,empty=True)==rt.identity(old/n,2<<20,empty=True)for n in
            ('infra/vcoco_person_pose_observations.py','infra/dwpose_smoke.py','infra/run_dwpose_smoke.sh','infra/dwpose_acquire.py','infra/dwpose_wheel_audit.py',
             'infra/keypoint_rgb_dwpose.py','infra/run_keypoint_rgb_dwpose.sh','src/world_reward/person_pose_observations.py')),
        'Original qualified DWPose helper/source differs')
    import dwpose_metadata_context as metadata
    smoke,capability=metadata.metadata_delegate(code)
    smoke.source_identity();capability.validate_smoke(ROOT)
    models=original.assets();names=('dw-ll_ucoco_384.onnx','source/onnxpose.py',
        'wheels/onnxruntime-1.30.0-cp311-cp311-manylinux_2_28_x86_64.whl','wheels/flatbuffers-25.12.19-py2.py3-none-any.whl')
    assets=[]
    for name in names:
        path=ROOT/original.acquisition.BASE/name;spec=next(r for r in original.acquisition.ASSETS if r[0]==name)
        pin=dict(bytes=spec[1],sha256=spec[2]);rt.require(models['files'][str(path)]==pin,'Original native asset differs')
        assets.append(dict(path=str(path),identity=pin))
    rt.require(assets==list(NATIVE_ASSETS),'Only four original native asset paths/pins required')
    return dict(assets=assets,models=models,source=binding,source_states={str(p):state(p)for p in(old,*sorted(old.rglob('*')))})


def runtime_delegate(code):
    """Execute only eight source-pinned original definitions, not smoke imports/main."""
    import numpy as np
    import tempfile
    import platform
    from contextlib import contextmanager
    from importlib import metadata
    path=code/'infra/dwpose_smoke.py';rt.require(rt.identity(path,2<<20)==SMOKE_PIN,'Original runtime helper changed')
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in RUNTIME_NAMES]
    rt.require(tuple(n.name for n in nodes)==RUNTIME_NAMES,'Exact original runtime definition whitelist required')
    def regular(p):
        p=Path(p)
        if not p.is_file()or any(q.is_symlink()for q in(p,*p.parents)):
            raise ValueError('Require regular nonsymlink audit input')
        return p
    def digest(p):
        h=hashlib.sha256()
        with regular(p).open('rb')as stream:
            for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
        return h.hexdigest()
    namespace=dict(np=np,Path=Path,hashlib=hashlib,sys=sys,time=time,tempfile=tempfile,platform=platform,
        metadata=metadata,contextmanager=contextmanager,audit=SimpleNamespace(regular=regular),acquisition=SimpleNamespace(digest=digest))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**{name:namespace[name]for name in RUNTIME_NAMES})


def native_source(code,revision,proof):
    rt.require(set(proof['helpers'])==set(NATIVE_FILES) and Path(__file__).resolve()==code/'infra/vcoco_full_pose_run.py'
        and (code.parent/'revision').read_bytes()==(revision+'\n').encode(),'Exact native source declaration required')
    rt.require({str(p.relative_to(code))for p in code.rglob('*')if p.is_file()}==set(NATIVE_FILES),
        'Exact native visible file whitelist required')
    actual={n:rt.identity(code/n,2<<20,empty=True)for n in NATIVE_FILES}
    rt.require(actual==proof['helpers'] and all(actual[n]==p for n,p in FROZEN.items()),'Original mounted helper differs')
    rt.require({n:rt.identity(code.parent/n,100)for n in('revision','source-sha256')}==proof['markers'],'Original source markers differ')
    return dict(helpers=actual,states={n:state(code/n)for n in NATIVE_FILES},markers={n:state(code.parent/n)for n in proof['markers']})


def native_publish(report,deadline,started,owner,allowed):
    directory=OUTPUT.lstat()
    rt.require(set(p.name for p in OUTPUT.iterdir())==set(allowed) and stat.S_ISDIR(directory.st_mode)
        and (directory.st_dev,directory.st_ino,directory.st_uid)==owner and stat.S_IMODE(directory.st_mode)==0o700,
        'Foreign native namespace')
    leaves={n:(rt.identity(OUTPUT/n,2<<20,readonly=n!='.container.cid'),state(OUTPUT/n))for n in allowed}
    with(OUTPUT/'native.json').open('x+b')as stream:
        os.fchmod(stream.fileno(),0o400);opened=os.fstat(stream.fileno())
        def write():
            current=OUTPUT.lstat();now=(OUTPUT/'native.json').lstat()
            rt.require((current.st_dev,current.st_ino,current.st_uid)==owner and current.st_gid==directory.st_gid
                and stat.S_ISDIR(current.st_mode)and stat.S_IMODE(current.st_mode)==0o700
                and stat.S_ISREG(now.st_mode)and now.st_nlink==1 and stat.S_IMODE(now.st_mode)==0o400
                and (now.st_dev,now.st_ino,now.st_uid,now.st_gid)==(opened.st_dev,opened.st_ino,opened.st_uid,opened.st_gid),
                'Foreign native report inode')
            report['elapsed_seconds']=time.monotonic()-started;raw=encode(report);rt.require(len(raw)<=2<<20,'Bounded native report')
            stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno());sync(OUTPUT)
        try:
            write();rt.require(rt.identity(OUTPUT/'native.json',2<<20)==dict(bytes=len(encode(report)),sha256=hashlib.sha256(encode(report)).hexdigest())
                and set(p.name for p in OUTPUT.iterdir())==set(allowed)|{'native.json'}
                and {n:(rt.identity(OUTPUT/n,2<<20,readonly=n!='.container.cid'),state(OUTPUT/n))for n in allowed}==leaves,
                'Native report/input/partial outputs differ');check(deadline)
        except BaseException as exc:report.update(status='fail',publication_failed=True,publication_error_type=error(exc));write()


def cpu(code,revision,proof_pin,deadline):
    proof=rt.pinned(OUTPUT/'proof.json',proof_pin,2<<20);started=time.monotonic();s=OUTPUT.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
    report=dict(schema=SCHEMA,stage='native_public48_person_pose',status='fail',phase='preflight',producer_revision=revision,
        image_id=IMAGE,proof_identity=proof_pin,images=[],native_calls=[],models_loaded=0,models_released=False,
        source_inputs_assets_runtime_rehashed_after=False,**{k:False for k in FLAGS})
    previous={s:signal.signal(s,lambda *_:(_ for _ in()).throw(TimeoutError('Inclusive native pose deadline')))for s in(signal.SIGALRM,signal.SIGTERM)}
    signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    reference=None;before=None;assets=None;prefix=None
    try:
        rt.require(set(proof)=={'schema','producer_revision','image_id','helpers','markers','public','assets'}
            and proof['schema']==SCHEMA and proof['producer_revision']==revision and proof['image_id']==IMAGE
            and os.environ.get('WR_IMAGE_ID')==IMAGE and os.environ.get('CUDA_VISIBLE_DEVICES')==''
            and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
            and proof['assets']==list(NATIVE_ASSETS) and proof['public']['path']==str(DEST/'public-reference.json'),
            'Exact offline B47 CPU proof required')
        before=native_source(code,revision,proof)
        assets={r['path']:rt.identity(r['path'],200<<20)for r in proof['assets']}
        rt.require(len(assets)==4 and assets=={r['path']:r['identity']for r in proof['assets']},'Original four native assets differ')
        asset_states={n:state(n)for n in assets}
        context=public.PublicObservationContext(DEST/'inputs',DEST/'banks',OUTPUT)
        reference=public.public_bank_reference(context,projection_path=Path(proof['public']['path']),projection_pin=proof['public']['identity'],
            code=code,native_source={'helpers':{n:proof['helpers'][n]for n in public.HELPERS}},deadline=deadline,native_mounts=True)
        dw=runtime_delegate(code);report['dependencies']=dw.dependency_identity()
        rt.require(not any(n in sys.modules or importlib.util.find_spec(n)is not None for n in('onnxruntime','flatbuffers')),'No preexisting runtime overlay')
        with dw.private_prefix(report,lambda:None)as prefix:
            wheels=[r['path']for r in proof['assets']if '/wheels/'in r['path']]
            rt.require(len(wheels)==2,'Original two runtime wheels required');report['phase']='private_overlay'
            command=[sys.executable,'-I','-m','pip','--isolated','--disable-pip-version-check','install','--no-index','--no-deps',
                '--no-compile','--no-warn-script-location','--target',str(prefix),*wheels]
            subprocess.run(command,check=True,capture_output=True,timeout=max(.001,deadline-time.monotonic()))
            ort,imports=dw.import_runtime(prefix);report['runtime_imports']=imports
            src=next(Path(r['path'])for r in proof['assets']if r['path'].endswith('/source/onnxpose.py'))
            spec=importlib.util.spec_from_file_location('world_reward_original_dwpose_public48',src)
            original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
            model=next(r['path']for r in proof['assets']if r['path'].endswith('.onnx'))
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            session=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider']);session.disable_fallback()
            dw.validate_session(session);dw.validate_options(session,ort)
            proxy=dw.SessionProxy(session,report['native_calls'],lambda:None);report['models_loaded']=1;report['phase']='all_person_observations'
            from world_reward.person_pose_observations import infer_person_pose_frame
            try:
                adapters.observe_pose(reference,core.observe,lambda rgb,ids,boxes,scores:
                    infer_person_pose_frame(original.inference_pose,proxy,rgb,0,ids,boxes,scores),deadline,records=report['images'])
                report['persons']=sum(r['person_retained_rows']for r in reference.banks)
                rt.require(len(report['images'])==48 and len(report['native_calls'])==report['persons'],'Every native crop required')
            finally:del proxy,session;report['models_released']=True
        report['phase']='complete';check(deadline);report['status']='pass'
    except BaseException as exc:report.update(error_type=error(exc),failure_stage=report['phase'])
    finally:
        try:
            post_stage='source'
            if before is not None:rt.require(native_source(code,revision,proof)==before,'Native source changed')
            post_stage='assets'
            if assets is not None:rt.require({n:rt.identity(n,200<<20)for n in assets}==assets and {n:state(n)for n in assets}==asset_states,'Native assets changed')
            post_stage='public_inputs_outputs'
            if reference is not None:reference.verify(deadline);adapters.validate_saved_records(reference,'pose',report['images'],deadline,complete=False)
            post_stage='overlay_cleanup'
            rt.require(prefix is None or not prefix.exists(),'Original CPU overlay survives')
            report['source_inputs_assets_runtime_rehashed_after']=before is not None and assets is not None and reference is not None
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc),post_failure_stage=post_stage)
        allowed={'proof.json'}|{f'image_{i:06d}.npz'for i in range(48)if(OUTPUT/f'image_{i:06d}.npz').exists()}|({'.container.cid'}if(OUTPUT/'.container.cid').exists()else set())
        try:native_publish(report,deadline,started,owner,allowed)
        finally:
            signal.alarm(0)
            for s,handler in previous.items():signal.signal(s,handler)
    return report


def validate_native(report,projection,revision,proof_pin):
    rt.require(report['schema']==SCHEMA and report['stage']=='native_public48_person_pose'and report['status']=='pass'
        and report['phase']=='complete'and report['producer_revision']==revision and report['image_id']==IMAGE
        and report['proof_identity']==proof_pin and report['models_loaded']==1 and report['models_released']is True
        and report['source_inputs_assets_runtime_rehashed_after']is True and report['private_prefix_removed']is True
        and all(report[k]is False for k in FLAGS)and len(report['images'])==48,'Complete public48 pose receipt required')
    total=sum(r['person_retained_rows']for r in projection['banks'])
    rt.require(type(report['persons'])is int and report['persons']==total and len(report['native_calls'])==total,'All derived person crops required')
    for ordinal,(row,bank,image)in enumerate(zip(report['images'],projection['banks'],projection['inputs']['images'])):
        n=bank['person_retained_rows'];shapes=dict(person_ids=[n],boxes_original_xyxy=[n,4],detector_scores=[n],keypoints_original_xy=[n,133,2],
            raw_scores=[n,133],native_valid=[n,133],in_original_image=[n,133],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        rt.require(row['image_id']==image['image_id']and row['file']==f'image_{ordinal:06d}.npz'and row['original_slot']==row['acquired_ordinal']==ordinal
            and row['original_frame_index']==0 and row['image_size']==bank['image_size']and row['person_ids']==row['source_person_ids']==bank['person_ids']
            and row['persons']==n and row['endpoint_bank_identity']==bank['identity']and row['owl_patches']==3600
            and rt.identity(OUTPUT/row['file'],2<<20)==row['identity']and set(row['arrays'])==set(shapes),'Every public pose identity/slot required')
        for name,shape in shapes.items():
            a=row['arrays'][name];rt.require(set(a)=={'shape','dtype','sha256'}and a['shape']==shape and re.fullmatch('[0-9a-f]{64}',a['sha256']),'Full133-array metadata required')
        for target,origin in(('person_ids','person_retained_ids'),('boxes_original_xyxy','person_retained_boxes'),('detector_scores','person_retained_scores')):
            rt.require(row['arrays'][target]==bank['arrays'][origin],'Original person arrays changed')
        types=dict(keypoints_original_xy='<f8',raw_scores='<f4',native_valid='|b1',in_original_image='|b1',image_size='<i8',
            original_frame_index='<i8',original_slot='<i8',acquired_ordinal='<i8')
        rt.require(all(row['arrays'][n]['dtype']==t for n,t in types.items()),'Original native dtypes required')
    for call in report['native_calls']:
        rt.require(call['run_completed']is True and call['supplied_container']=='list'and call['supplied_array']['dtype']=='float64'
            and call['supplied_array']['shape']==[3,384,288]and call['effective_feed_shape']==[1,3,384,288]
            and call['delegated_unmodified']is True and call['effective_runtime_conversion_observed']is False
            and [(x['dtype'],x['shape'])for x in call['raw_simcc']]==[('float32',[1,133,576]),('float32',[1,133,768])],'Original SimCC feed ABI required')


def cleanup(name,revision,deadline):
    path=OUTPUT/'.container.cid';command=public.endpoint.original.command
    if path.exists():
        rt.identity(path,65,readonly=False);raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned CID required');cid=raw.decode().strip()
        found=command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline);rt.require(found in('',cid),'Ambiguous CID')
        if found:
            wanted=IMAGE+'|/'+name+'|'+ENTRY+'|'+revision
            rt.require(command(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline)==wanted,'Foreign container removal forbidden')
            command(['docker','rm','-f',cid],deadline)
        rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline),'Exact CID survives');path.chmod(0o400)
    rt.require(not command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Exact name survives')


def image_state(deadline):
    value=rt.strict(public.endpoint.original.command(['docker','image','inspect',IMAGE,'--format','{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'],deadline))
    rt.require(value['Id']==IMAGE and value['Architecture']=='amd64'and value['Os']=='linux'and value['RootFS']['Type']=='layers','Exact B47 CPU image required');return value


def run(code,revision,replica_revision,replica_pin):
    started=time.monotonic();deadline=started+BUDGET
    def expired(*_):raise TimeoutError('Inclusive public48 pose budget')
    previous={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    try:return work(code,revision,replica_revision,replica_pin,started,deadline)
    finally:
        signal.alarm(0)
        for s,handler in previous.items():signal.signal(s,handler)


def work(code,revision,replica_revision,replica_pin,started,deadline):
    transfer,_,publisher=host_modules()
    rt.canonical(OUTPUT);rt.require(not OUTPUT.exists()and not OUTPUT.is_symlink(),'Fresh public48 pose output required')
    OUTPUT.mkdir(mode=0o700);s=OUTPUT.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
    before=inputs=models=image=None
    report=dict(schema=SCHEMA,stage='public48_person_pose_host',status='fail',phase='host_peer',producer_revision=revision,image_id=IMAGE,
        replica_revision=replica_revision,replica_identity=replica_pin,owned_cleanup_verified=False,
        source_inputs_assets_runtime_rehashed_after=False,**{k:False for k in FLAGS})
    name='world-reward-public48-pose-'+revision[:12];owned=set()
    try:
        transfer.transport.verify_azure_peer('import');report['phase']='source';before=source(code,revision)
        report['source_binding']=before['proof'];report['phase']='receiver';inputs=transfer.authenticate_receiver(code,replica_revision,replica_pin)
        report['input_import_source']=inputs['import_source'];report['phase']='assets';models=asset_context(code)
        report['dwpose_source']=models['source'];report['phase']='image';image=image_state(deadline);report['phase']='container'
        rt.require(not public.endpoint.original.command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline),'Occupied CPU name')
        proof=dict(schema=SCHEMA,producer_revision=revision,image_id=IMAGE,helpers={n:before['proof']['helpers'][n]for n in NATIVE_FILES},
            markers=before['proof']['markers'],public=dict(path=str(inputs['projection_path']),identity=inputs['projection_identity']),assets=models['assets'])
        proof_pin=public.endpoint.original.write(OUTPUT/'proof.json',proof);owned.add('proof.json')
        mounts=[code/n for n in NATIVE_FILES]+[code.parent/n for n in('revision','source-sha256')]
        mounts+=[Path(n)for n in inputs['files']if str(n).startswith(str(DEST)+'/')]+[Path(r['path'])for r in models['assets']]
        rt.require(sum(p.parent==DEST/'banks'for p in mounts)==48 and sum(p.parent==DEST/'inputs'for p in mounts)==49,'Only complete public replica mounted')
        cmd=['docker','run','--rm','--name',name,'--cidfile',str(OUTPUT/'.container.cid'),'--label','world-reward.job='+ENTRY,
            '--label','world-reward.revision='+revision,'--network','none','--user','0:0','--memory','8g','--cpus','4','--pids-limit','256',
            '--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--tmpfs','/tmp:rw,exec,nosuid,size=512m','--entrypoint','/usr/bin/env']
        for path in mounts:
            rt.canonical(path);rt.require(path.is_file()and ','not in str(path)and '\n'not in str(path),'Individual readonly native file mount')
            cmd+=['--mount',f'type=bind,src={path},dst={path},readonly']
        cmd+=['--mount',f'type=bind,src={OUTPUT},dst={OUTPUT}',IMAGE,'-i','PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','WR_IMAGE_ID='+IMAGE,
            'CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','/opt/conda/bin/python','-I','-B',
            str(code/'infra/vcoco_full_pose_run.py'),'--native','--code',str(code),'--revision',revision,'--proof-bytes',str(proof_pin['bytes']),
            '--proof-sha256',proof_pin['sha256'],'--deadline',format(deadline-20,'.17g')]
        result=subprocess.run(cmd,env=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,timeout=max(.001,deadline-time.monotonic()-20))
        for n in('native.json','.container.cid'):
            if(OUTPUT/n).exists():owned.add(n)
        report['phase']='native_receipt';report['native_exit_status']=result.returncode;report['native_identity']=rt.identity(OUTPUT/'native.json',2<<20)
        native=rt.pinned(OUTPUT/'native.json',report['native_identity'],2<<20)
        rt.require(result.returncode==0,'Native full48 CPU failed')
        projection=rt.pinned(inputs['projection_path'],inputs['projection_identity'],2<<20);validate_native(native,projection,revision,proof_pin)
        report.update(images=native['images'],persons=native['persons'],phase='complete',status='pass')
    except BaseException as exc:report.update(error_type=error(exc),failure_stage=report['phase'])
    finally:
        try:cleanup(name,revision,min(deadline+15,time.monotonic()+10));report['owned_cleanup_verified']=True
        except BaseException as exc:report.update(status='fail',cleanup_error_type=error(exc))
        try:
            post_stage='source'
            if before is not None:rt.require(source(code,revision)==before,'Full source changed')
            post_stage='receiver'
            if inputs is not None:rt.require(transfer.authenticate_receiver(code,replica_revision,replica_pin)==inputs,'Public replica changed')
            post_stage='assets'
            if models is not None:rt.require(asset_context(code)==models,'Original source/assets changed')
            post_stage='image'
            if image is not None:rt.require(image_state(deadline)==image,'Original runtime changed')
            report['source_inputs_assets_runtime_rehashed_after']=all(v is not None for v in(before,inputs,models,image))
        except BaseException as exc:report.update(status='fail',post_error_type=error(exc),post_failure_stage=post_stage)
        owned|={f'image_{i:06d}.npz'for i in range(48)if(OUTPUT/f'image_{i:06d}.npz').exists()}
        for n in('native.json','.container.cid'):
            if(OUTPUT/n).exists():owned.add(n)
        publisher.publish(OUTPUT,report,deadline,started,owner,owned,encode=encode,identity=rt.identity,snapshot=state,
            require=rt.require,check=check,sync=sync,error=error,report_maximum=2<<20)
    return report


def arguments(argv):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument('--native',action='store_true')
    for name in('replica-revision','replica-sha256','code','revision','proof-sha256'):parser.add_argument('--'+name)
    for name in('replica-bytes','proof-bytes'):parser.add_argument('--'+name,type=int)
    parser.add_argument('--deadline',type=float);args=parser.parse_args(argv)
    if args.native:
        rt.require(all(getattr(args,n)is None for n in('replica_revision','replica_sha256','replica_bytes'))
            and type(args.deadline)is float and math.isfinite(args.deadline)and type(args.code)is str
            and re.fullmatch('[0-9a-f]{40}',str(args.revision)),'Only explicit native origin/deadline allowed')
        args.pin=dict(bytes=args.proof_bytes,sha256=args.proof_sha256)
    else:
        rt.require(all(getattr(args,n)is None for n in('code','revision','proof_sha256','proof_bytes','deadline'))
            and re.fullmatch('[0-9a-f]{40}',str(args.replica_revision)),'Host requires independent imported replica identity')
        args.pin=dict(bytes=args.replica_bytes,sha256=args.replica_sha256)
    rt.require(type(args.pin['bytes'])is int and 0<args.pin['bytes']<=2<<20 and re.fullmatch('[0-9a-f]{64}',str(args.pin['sha256'])),'Bounded explicit receipt pin')
    return args


if __name__=='__main__':
    a=arguments(sys.argv[1:]);rt.require(sys.platform=='linux'and os.geteuid()==0,'Azure root CPU required')
    result=cpu(Path(a.code),a.revision,a.pin,a.deadline)if a.native else run(Path(os.environ.get('WR_CODE','/invalid')),
        os.environ.get('WR_CODE_REVISION',''),a.replica_revision,a.pin)
    print(encode(dict(stage=result['stage'],status=result['status'])).decode(),end='');raise SystemExit(0 if result['status']=='pass'else 1)
