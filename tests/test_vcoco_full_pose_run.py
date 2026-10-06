"""Authored public-only controls; no Azure, ORT/model/real pixels or references."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time

import numpy as np
import pytest
import vcoco_full_pose_run as p

ROOT=Path(__file__).resolve().parents[1]


def pinned_runtime(tmp_path):
    code=tmp_path/'code';(code/'infra').mkdir(parents=True)
    leaf=code/'infra/dwpose_smoke.py';leaf.write_bytes((ROOT/'infra/dwpose_smoke.py').read_bytes());leaf.chmod(0o400)
    return code,p.runtime_delegate(code)


def test_original_runtime_ASTs_are_exact_and_whitelist_dependency_graph_closed(tmp_path):
    code,delegate=pinned_runtime(tmp_path)
    tree=ast.parse((code/'infra/dwpose_smoke.py').read_text())
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in p.RUNTIME_NAMES]
    assert tuple(n.name for n in definitions)==p.RUNTIME_NAMES
    assert all(callable(getattr(delegate,n))for n in p.RUNTIME_NAMES)
    # Actual compiled function code retains its original module source location.
    assert all(getattr(getattr(delegate,n),'__wrapped__',getattr(delegate,n)).__code__.co_filename==str(code/'infra/dwpose_smoke.py')for n in p.RUNTIME_NAMES[:-1])
    allowed={'audit':{'regular'},'acquisition':{'digest'}}
    for node in ast.walk(ast.Module(body=definitions,type_ignores=[])):
        if isinstance(node,ast.Attribute)and isinstance(node.value,ast.Name)and node.value.id in allowed:
            assert node.attr in allowed[node.value.id]
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(a.name in('dwpose_acquire','dwpose_wheel_audit')for a in n.names)for n in definitions)
    with delegate.private_prefix({},lambda:None)as prefix:assert prefix.is_dir()
    assert not prefix.exists()


def test_delegate_rejects_changed_smoke_without_executing_it(tmp_path):
    code,d=pinned_runtime(tmp_path);path=code/'infra/dwpose_smoke.py';path.chmod(0o600);path.write_bytes(b'raise AssertionError("not executed")');path.chmod(0o400)
    with pytest.raises(ValueError):p.runtime_delegate(code)


class Session:
    def get_inputs(self):return [SimpleNamespace(name='input',shape=['batch',3,384,288],type='tensor(float)')]
    def get_outputs(self):return [SimpleNamespace(name='simcc_'+axis,shape=['batch','MatMulsimcc_'+axis+'_dim_1','MatMulsimcc_'+axis+'_dim_2'],type='tensor(float)')for axis in('x','y')]
    def get_providers(self):return ['CPUExecutionProvider']
    def get_modelmeta(self):return SimpleNamespace(custom_metadata_map={})
    def get_session_options(self):return SimpleNamespace(intra_op_num_threads=4,inter_op_num_threads=1,execution_mode='seq')
    def run(self,names,feed):
        self.original_feed=feed;return [np.zeros((1,133,576),np.float32),np.zeros((1,133,768),np.float32)]


def test_original_session_and_float64_list_feed_delegate_exact_no_cast(tmp_path):
    code,d=pinned_runtime(tmp_path);session=Session();calls=[]
    d.validate_session(session);d.validate_options(session,SimpleNamespace(ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='seq')))
    proxy=d.SessionProxy(session,calls,lambda:None);feed={'input':[np.zeros((3,384,288),np.float64)]}
    outputs=proxy.run(['simcc_x','simcc_y'],feed)
    assert session.original_feed is feed and outputs[0].dtype==np.float32 and calls[0]['run_completed']is True
    assert calls[0]['supplied_array']['dtype']=='float64'and calls[0]['effective_runtime_conversion_observed']is False
    with pytest.raises(ValueError):proxy.run(['simcc_x','simcc_y'],{'input':[np.zeros((3,384,288),np.float32)]})
    session.get_session_options=lambda:SimpleNamespace(intra_op_num_threads=1,inter_op_num_threads=1,execution_mode='seq')
    with pytest.raises(ValueError):d.validate_options(session,SimpleNamespace(ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='seq')))


def test_native_imports_never_load_old_private_producers_or_readers():
    script="""import sys
sys.path[:0]=["%s","%s"]
import vcoco_full_pose_run as p
assert not set(('vcoco_person_pose_observations','dwpose_acquire','dwpose_wheel_audit','vcoco_full_public_replica','vcoco_replica_completion'))&set(sys.modules)
assert not set(('numpy','torch','onnxruntime','transformers'))&set(sys.modules)
"""%(ROOT/'infra',ROOT/'src')
    subprocess.run([sys.executable,'-I','-B','-c',script],check=True,capture_output=True)
    assert not set(p.NATIVE_FILES)&{'infra/vcoco_person_pose_observations.py','infra/dwpose_acquire.py','infra/dwpose_wheel_audit.py','infra/vcoco_full_public_replica.py'}
    assert set(p.public.HELPERS)<=set(p.NATIVE_FILES)
    assert 'infra/vcoco_public_pose_core.py'in p.NATIVE_FILES


@pytest.mark.parametrize('native',[True,False])
def test_cli_explicit_pin_and_origin_forbidden_cross_mode(native):
    args=['--native','--code','/code','--revision','a'*40,'--deadline','2','--proof-bytes','3','--proof-sha256','b'*64]if native else ['--replica-revision','a'*40,'--replica-bytes','3','--replica-sha256','b'*64]
    assert p.arguments(args).pin==dict(bytes=3,sha256='b'*64)
    with pytest.raises(ValueError):p.arguments(args+(['--replica-revision','c'*40]if native else['--deadline','2']))
    with pytest.raises(ValueError):p.arguments(args[:-1]+['bad'])


def report_fixture(tmp_path,monkeypatch):
    output=tmp_path/'out';output.mkdir();monkeypatch.setattr(p,'OUTPUT',output)
    banks=[];images=[];rows=[]
    for i in range(48):
        n=i%2;counts={'person_retained_ids':([n],'<U1'),'person_retained_boxes':([n,4],'<f8'),'person_retained_scores':([n],'<f8')}
        arrays={name:dict(shape=shape,dtype=dtype,sha256=hashlib.sha256((name+str(i)).encode()).hexdigest())for name,(shape,dtype)in counts.items()}
        bank=dict(person_retained_rows=n,person_ids=['p']if n else[],arrays=arrays,image_size=[3,4],identity=dict(bytes=1,sha256='d'*64));banks.append(bank)
        images.append(dict(image_id=str(i)))
        shapes=dict(person_ids=[n],boxes_original_xyxy=[n,4],detector_scores=[n],keypoints_original_xy=[n,133,2],raw_scores=[n,133],
            native_valid=[n,133],in_original_image=[n,133],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        dtypes=dict(person_ids='<U1',boxes_original_xyxy='<f8',detector_scores='<f8',keypoints_original_xy='<f8',raw_scores='<f4',native_valid='|b1',
            in_original_image='|b1',image_size='<i8',original_frame_index='<i8',original_slot='<i8',acquired_ordinal='<i8')
        metadata={k:dict(shape=s,dtype=dtypes[k],sha256='c'*64)for k,s in shapes.items()}
        for target,origin in(('person_ids','person_retained_ids'),('boxes_original_xyxy','person_retained_boxes'),('detector_scores','person_retained_scores')):metadata[target]=arrays[origin].copy()
        path=output/f'image_{i:06d}.npz';path.write_bytes(b'authored');path.chmod(0o400)
        rows.append(dict(image_id=str(i),file=path.name,original_slot=i,acquired_ordinal=i,original_frame_index=0,image_size=[3,4],person_ids=bank['person_ids'],
            source_person_ids=bank['person_ids'],persons=n,endpoint_bank_identity=bank['identity'],owl_patches=3600,identity=p.rt.identity(path),arrays=metadata))
    call=dict(run_completed=True,supplied_container='list',supplied_array=dict(dtype='float64',shape=[3,384,288]),effective_feed_shape=[1,3,384,288],
        delegated_unmodified=True,effective_runtime_conversion_observed=False,raw_simcc=[dict(dtype='float32',shape=[1,133,576]),dict(dtype='float32',shape=[1,133,768])])
    report=dict(schema=p.SCHEMA,stage='native_public48_person_pose',status='pass',phase='complete',producer_revision='a'*40,image_id=p.IMAGE,
        proof_identity={'bytes':1,'sha256':'b'*64},models_loaded=1,models_released=True,source_inputs_assets_runtime_rehashed_after=True,private_prefix_removed=True,
        images=rows,persons=24,native_calls=[deepcopy(call)for _ in range(24)],**{k:False for k in p.FLAGS})
    return report,dict(banks=banks,inputs=dict(images=images))


def test_full48_metadata_validation_zero_person_all11_original_inputs(tmp_path,monkeypatch):
    report,projection=report_fixture(tmp_path,monkeypatch);p.validate_native(report,projection,'a'*40,report['proof_identity'])
    assert report['persons']==24 and report['images'][0]['persons']==0 and len(report['images'])==48


@pytest.mark.parametrize('fault',['rows','calls','ids','dtype','boxes','off','slot','fallback','shape','model'])
def test_invalid_or_incomplete_native_metadata_fails(tmp_path,monkeypatch,fault):
    report,projection=report_fixture(tmp_path,monkeypatch)
    if fault=='rows':report['images'].pop()
    elif fault=='calls':report['native_calls'].pop()
    elif fault=='ids':report['images'][1]['person_ids']=[]
    elif fault=='dtype':report['images'][1]['arrays']['raw_scores']['dtype']='<f8'
    elif fault=='boxes':report['images'][1]['arrays']['boxes_original_xyxy']['sha256']='e'*64
    elif fault=='off':report['native_calls'][0]['effective_runtime_conversion_observed']=True
    elif fault=='slot':report['images'][1]['original_slot']=0
    elif fault=='fallback':report['full_image_fallback']=True
    elif fault=='shape':report['images'][1]['arrays']['keypoints_original_xy']['shape']=[1,134,2]
    else:report['models_loaded']=2
    with pytest.raises(ValueError):p.validate_native(report,projection,'a'*40,report['proof_identity'])


@pytest.mark.parametrize('late',[False,True])
def test_native_sameFD_publication_persists_lateFAIL(tmp_path,monkeypatch,late):
    monkeypatch.setattr(p,'OUTPUT',tmp_path);s=tmp_path.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
    report=dict(status='pass');p.native_publish(report,time.monotonic()+(-1 if late else 10),time.monotonic(),owner,set())
    saved=p.rt.strict((tmp_path/'native.json').read_bytes())
    assert saved['status']==('fail'if late else'pass')and (tmp_path/'native.json').stat().st_mode&0o777==0o400
    assert saved.get('publication_failed',False)is late


def test_native_foreign_namespace_not_admitted(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUTPUT',tmp_path);(tmp_path/'foreign').write_bytes(b'foreign');s=tmp_path.lstat()
    with pytest.raises(ValueError):p.native_publish(dict(status='pass'),time.monotonic()+10,time.monotonic(),(s.st_dev,s.st_ino,s.st_uid),set())
    assert not(tmp_path/'native.json').exists()and(tmp_path/'foreign').read_bytes()==b'foreign'


@pytest.mark.parametrize('newline',[False,True])
def test_exactCID_and_name_cleanup_after_removal(tmp_path,monkeypatch,newline):
    monkeypatch.setattr(p,'OUTPUT',tmp_path);cid='d'*64;(tmp_path/'.container.cid').write_text(cid+('\n'if newline else''));present=True
    def command(args,deadline):
        nonlocal present
        if args[:2]==['docker','inspect']:return p.IMAGE+'|/owned|'+p.ENTRY+'|'+'a'*40
        if args[:3]==['docker','rm','-f']:present=False;return cid
        return cid if present else ''
    monkeypatch.setattr(p.public.endpoint.original,'command',command);p.cleanup('owned','a'*40,time.monotonic()+10)
    assert not present and(tmp_path/'.container.cid').stat().st_mode&0o777==0o400


def test_surviving_or_renamedCID_fails_not_falsePASS(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'OUTPUT',tmp_path);cid='d'*64;(tmp_path/'.container.cid').write_text(cid)
    def command(args,deadline):
        if args[:2]==['docker','inspect']:return p.IMAGE+'|/owned|'+p.ENTRY+'|'+'a'*40
        return cid
    monkeypatch.setattr(p.public.endpoint.original,'command',command)
    with pytest.raises(ValueError):p.cleanup('owned','a'*40,time.monotonic()+10)


@pytest.mark.parametrize('fail',[False,True])
def test_cpu_single_session_all48_native_callbacks_cleanup_and_safe_failure(tmp_path,monkeypatch,fail):
    from contextlib import contextmanager
    code,d=pinned_runtime(tmp_path);out=tmp_path/'out';out.mkdir(mode=0o700);monkeypatch.setattr(p,'OUTPUT',out)
    files=[]
    for name in('model.onnx','source/onnxpose.py','wheels/ort.whl','wheels/flat.whl'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'authored');path.chmod(0o400)
        files.append(dict(path=str(path),identity=p.rt.identity(path)))
    proof=dict(schema=p.SCHEMA,producer_revision='a'*40,image_id=p.IMAGE,helpers={n:dict(bytes=1,sha256='c'*64)for n in p.public.HELPERS},markers={},public=dict(path=str(tmp_path/'public.json'),identity={}),assets=files)
    monkeypatch.setattr(p,'NATIVE_ASSETS',tuple(files));monkeypatch.setattr(p,'DEST',tmp_path);proof['public']['path']=str(tmp_path/'public-reference.json')
    raw=p.encode(proof);(out/'proof.json').write_bytes(raw);(out/'proof.json').chmod(0o400);pin=p.rt.identity(out/'proof.json')
    monkeypatch.setenv('WR_IMAGE_ID',p.IMAGE);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    actual_iter=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([Path('/sys/class/net/lo')])if str(path)=='/sys/class/net'else actual_iter(path))
    monkeypatch.setattr(p,'native_source',lambda *args:{'stable':'original_source'})
    counts=[];session=Session();options=[]
    def make_session(model,sess_options,providers):
        counts.append(1);assert providers==['CPUExecutionProvider'];return session
    session.disable_fallback=lambda:None
    def option():value=SimpleNamespace();options.append(value);return value
    ort=SimpleNamespace(SessionOptions=option,InferenceSession=make_session,ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='seq'))
    d.import_runtime=lambda prefix:(ort,{'origin':'owned overlay'})
    d.dependency_identity=lambda:{'ABI':'authored'}
    monkeypatch.setattr(p,'runtime_delegate',lambda code:d)
    calls=[]
    monkeypatch.setattr(p.subprocess,'run',lambda argv,**kwargs:calls.append(argv)or SimpleNamespace(returncode=0))
    original=SimpleNamespace()
    def native(proxy,boxes,rgb):
        if fail:raise RuntimeError('SECRET_NOT_TO_SERIALIZE')
        proxy.run(['simcc_x','simcc_y'],{'input':[np.zeros((3,384,288),np.float64)]})
        return np.zeros((len(boxes),133,2),np.float64),np.ones((len(boxes),133),np.float32)
    original.inference_pose=native
    monkeypatch.setattr(p.importlib.util,'spec_from_file_location',lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None)))
    monkeypatch.setattr(p.importlib.util,'module_from_spec',lambda spec:original)
    monkeypatch.setattr(p.importlib.util,'find_spec',lambda name:None)
    banks=[dict(person_retained_rows=1)for _ in range(48)]
    class Reference:
        def __init__(self):self.banks=banks
        def verify(self,deadline):p.check(deadline)
    reference=Reference();monkeypatch.setattr(p.public,'public_bank_reference',lambda *a,**k:reference)
    monkeypatch.setattr(p.adapters,'validate_saved_records',lambda *a,**k:None)
    def observe(reference,original_observe,infer,deadline,records):
        assert original_observe is p.core.observe
        for i in range(48):
            result=infer(np.zeros((3,4,3),np.uint8),('p',),np.array([[0,0,3,3]],np.float64),np.ones(1,np.float64))
            assert result.raw_scores.dtype==np.float32 and result.keypoints_original_xy.dtype==np.float64
            path=out/f'image_{i:06d}.npz';path.write_bytes(b'authored-pose');path.chmod(0o400)
            records.append(dict(file=path.name,persons=1,identity=p.rt.identity(path)))
    monkeypatch.setattr(p.adapters,'observe_pose',observe)
    report=p.cpu(code,'a'*40,pin,time.monotonic()+20)
    saved=p.rt.strict((out/'native.json').read_bytes());assert saved==report
    assert len(counts)==1 and len(calls)==1 and '--no-index'in calls[0]and '--no-deps'in calls[0]
    assert options[0].intra_op_num_threads==4 and options[0].inter_op_num_threads==1
    assert report['models_released']is True and report['private_prefix_removed']is True and report['source_inputs_assets_runtime_rehashed_after']is True
    assert 'SECRET_NOT_TO_SERIALIZE'not in(out/'native.json').read_text()
    if fail:assert report['status']=='fail'and report['failure_stage']=='all_person_observations'and report['error_type']=='RuntimeError'
    else:assert report['status']=='pass'and len(report['images'])==report['persons']==len(report['native_calls'])==48


def test_host_signal_handlers_restored_on_preflight_failure(tmp_path,monkeypatch):
    import signal
    before={s:signal.getsignal(s)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    monkeypatch.setattr(p,'work',lambda *a:(_ for _ in()).throw(ValueError('safe preflight')))
    with pytest.raises(ValueError):p.run(tmp_path,'a'*40,'b'*40,dict(bytes=1,sha256='c'*64))
    assert before=={s:signal.getsignal(s)for s in before}


@pytest.mark.parametrize('failure',[None,'receiver','native'])
def test_host_complete_mock_lifecycle_or_bounded_failure_receipt(tmp_path,monkeypatch,failure):
    report,projection=report_fixture(tmp_path,monkeypatch)
    out=tmp_path/'newout';monkeypatch.setattr(p,'OUTPUT',out);dest=tmp_path/'replica';(dest/'inputs').mkdir(parents=True);(dest/'banks').mkdir()
    monkeypatch.setattr(p,'DEST',dest);files={}
    for i in range(48):
        for folder,suffix in(('inputs','jpg'),('banks','npz')):
            path=dest/folder/f'image_{i:06d}.{suffix}';path.write_bytes(b'authored');path.chmod(0o400);files[str(path)]=p.rt.identity(path)
    path=dest/'inputs/manifest.json';path.write_bytes(b'public');path.chmod(0o400);files[str(path)]=p.rt.identity(path)
    projection_path=dest/'public-reference.json';projection_path.write_bytes(p.encode(projection));projection_path.chmod(0o400);files[str(projection_path)]=p.rt.identity(projection_path)
    inputs=dict(images=projection['inputs']['images'],banks=projection['banks'],files=files,import_source={'original':'binding'},
        projection_path=str(projection_path),projection_identity=files[str(projection_path)])
    assets=[]
    for n in range(4):
        path=tmp_path/f'asset{n}';path.write_bytes(b'asset');path.chmod(0o400);assets.append(dict(path=str(path),identity=p.rt.identity(path)))
    models=dict(assets=assets,source={'original':'pose'})
    code=tmp_path/'code'
    for n in p.NATIVE_FILES:
        leaf=code/n;leaf.parent.mkdir(parents=True,exist_ok=True);leaf.write_bytes(b'code');leaf.chmod(0o400)
    for n in('revision','source-sha256'):
        leaf=code.parent/n;leaf.write_bytes(b'marker');leaf.chmod(0o400)
    before=dict(proof=dict(helpers={n:dict(bytes=1,sha256='a'*64)for n in p.NATIVE_FILES},markers={}),states={},markers={})
    def receiver(*args):
        if failure=='receiver':raise ValueError('SECRET_RECEIVER_TOKEN')
        return inputs
    transfer=SimpleNamespace(transport=SimpleNamespace(verify_azure_peer=lambda phase:None),authenticate_receiver=receiver)
    import sealed_callback_publication as publisher
    monkeypatch.setattr(p,'host_modules',lambda:(transfer,None,publisher));monkeypatch.setattr(p,'source',lambda *a:before)
    monkeypatch.setattr(p,'asset_context',lambda code:models);monkeypatch.setattr(p,'image_state',lambda deadline:{'image':p.IMAGE})
    monkeypatch.setattr(p.public.endpoint.original,'command',lambda *a:'');monkeypatch.setattr(p,'cleanup',lambda *a:None)
    def run(argv,**kwargs):
        assert '--gpus'not in argv and '--network'in argv and argv[argv.index('--network')+1]=='none'
        mounts=[argv[i+1]for i,n in enumerate(argv[:-1])if n=='--mount']
        assert sum('/replica/inputs/'in n for n in mounts)==49 and sum('/replica/banks/'in n for n in mounts)==48
        assert not any('/eval_private/'in n or '/cohort' in n for n in mounts)
        if failure=='native':raise RuntimeError('SECRET_CONTAINER_TOKEN')
        proof=p.rt.identity(out/'proof.json');value=dict(report,proof_identity=proof)
        for row in value['images']:
            leaf=out/row['file'];leaf.write_bytes(b'authored');leaf.chmod(0o400);row['identity']=p.rt.identity(leaf)
        path=out/'native.json';path.write_bytes(p.encode(value));path.chmod(0o400)
        path=out/'.container.cid';path.write_text('c'*64);path.chmod(0o400)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run)
    result=p.work(code,'a'*40,'b'*40,dict(bytes=1,sha256='c'*64),time.monotonic(),time.monotonic()+20)
    saved=p.rt.strict((out/'report.json').read_bytes());assert result==saved
    assert 'SECRET_'not in(out/'report.json').read_text()and out.stat().st_mode&0o777==0o500
    assert result['owned_cleanup_verified']is True
    if failure:assert result['status']=='fail'and result['failure_stage']==('receiver'if failure=='receiver'else'container')
    else:assert result['status']=='pass'and len(result['images'])==48 and result['persons']==24


def test_native_whitelist_rejects_even_readonly_unknown_codefile(tmp_path,monkeypatch):
    code=tmp_path/'code';code.mkdir();monkeypatch.setattr(p,'__file__',str(code/'infra/vcoco_full_pose_run.py'))
    for n in p.NATIVE_FILES:
        leaf=code/n;leaf.parent.mkdir(parents=True,exist_ok=True);leaf.write_bytes(b'authored');leaf.chmod(0o400)
    markers={}
    for n,data in(('revision',b'a'*40+b'\n'),('source-sha256',b'b'*64+b'\n')):
        leaf=code.parent/n;leaf.write_bytes(data);leaf.chmod(0o400);markers[n]=p.rt.identity(leaf)
    helpers={n:p.rt.identity(code/n)for n in p.NATIVE_FILES};monkeypatch.setattr(p,'FROZEN',{})
    proof=dict(helpers=helpers,markers=markers);p.native_source(code,'a'*40,proof)
    leaf=code/'infra/private_unknown.py';leaf.write_bytes(b'never imported');leaf.chmod(0o400)
    with pytest.raises(ValueError):p.native_source(code,'a'*40,proof)
