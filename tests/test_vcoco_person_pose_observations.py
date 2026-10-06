"""Authored tiny all-person adapters only; no real RGB, references or models."""
import copy
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import vcoco_person_pose_observations as p
from world_reward.person_pose_observations import infer_person_pose_frame


def arrays(n):return dict(person_retained_ids=np.asarray(['person-%d'%i for i in range(n)],dtype='<U12'),
    person_retained_boxes=np.tile(np.asarray([[1.,1.,3.,3.]],np.float32),(n,1)),person_retained_scores=np.full(n,.7,np.float32))

def rows(ns):
    images=[dict(image_id='%032x'%i,file='image_%06d.jpg'%i)for i in range(len(ns))]
    banks=[dict(file='image_%06d.npz'%i,image_size=[4,4],person_retained_rows=n,person_ids=arrays(n)['person_retained_ids'].tolist(),original_slot=i)for i,n in enumerate(ns)]
    return images,banks


def test_all_banks_validate_before_any_pose_full_rows_and_n0():
    images,banks=rows([2,0,3]);events=[];calls=[]
    def load(row,validate):
        events.append(('validate'if validate else 'load',row['original_slot']));return arrays(row['person_retained_rows'])
    def native(session,boxes,rgb):
        calls.append(len(boxes));n=len(boxes)
        points=np.full((n,133,2),-2.,np.float64);scores=np.zeros((n,133),np.float32)
        scores[:,9]=.4;scores[:,91]=-.3
        return points,scores
    def infer(rgb,ids,boxes,scores):return infer_person_pose_frame(native,None,rgb,0,ids,boxes,scores)
    def save(result,image,bank,ordinal):
        assert result.person_ids==tuple(bank['person_ids'])
        assert result.keypoints_original_xy.shape==(bank['person_retained_rows'],133,2)
        assert not result.in_original_image.any()
        assert np.array_equal(result.native_valid,result.raw_scores>0)
        return dict(slot=bank['original_slot'],ordinal=ordinal)
    out,total=p.observe(images,banks,load,lambda _:np.zeros((4,4,3),np.uint8),infer,save,lambda:None)
    assert events[:3]==[('validate',i)for i in range(3)]
    assert calls==[2,3]and total==5 and len(out)==3


def test_invalid_bank_stops_before_pose():
    images,banks=rows([2,3]);called=[]
    def load(row,validate):
        if validate and row['original_slot']==1:raise ValueError('invalid fullbank')
        return arrays(row['person_retained_rows'])
    with pytest.raises(ValueError):p.observe(images,banks,load,lambda _:None,lambda *a:called.append(a),lambda *a:None,lambda:None)
    assert not called


def test_abi_mismatch_and_mutation_fail_closed():
    images,banks=rows([2]);a=arrays(2)
    with pytest.raises(ValueError):p.observe(images,[],lambda *a:None,lambda *a:None,lambda *a:None,lambda *a:None,lambda:None)
    banks[0]['person_ids']=['invented']
    with pytest.raises(ValueError):p.observe(images,banks,lambda *unused,**kw:a,lambda *a:None,lambda *a:None,lambda *a:None,lambda:None)


def test_native_call_mutation_is_rejected():
    images,banks=rows([2]);a=arrays(2)
    def load(*args,**kw):return a
    def infer(rgb,ids,boxes,scores):
        result=infer_person_pose_frame(lambda session,b,r:(np.zeros((2,133,2),np.float64),np.ones((2,133),np.float32)),None,rgb,0,ids,boxes,scores)
        a['person_retained_scores'][0]=.2;return result
    with pytest.raises(ValueError):p.observe(images,banks,load,lambda _:np.zeros((4,4,3),np.uint8),infer,lambda *a:None,lambda:None)


def test_host_import_stdlib_only_and_no_old_probe_paths():
    code="""import sys,importlib.abc
sys.path[:0]=['infra','src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in ('numpy','torch','transformers','onnxruntime','PIL','cv2'):raise AssertionError('ML import forbidden')
sys.meta_path.insert(0,Deny())
import vcoco_person_pose_observations
print('stdlib')
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout=='stdlib\n'
    source=Path(p.__file__).read_text()
    assert 'person_pose_bank_probe'not in source and 'actor_bbox'not in source
    assert 'capability.validate_smoke(ROOT)'in source and 'dw.validate_assets(ROOT)'in source
    assert "providers=['CPUExecutionProvider']"in source
    assert 'session.disable_fallback()'in source and 'dw.private_prefix' in source
    assert '--gpus'not in source and 'eval_private'not in source and 'cohort'not in source


def test_native_whitelist_actual_pins_and_markers(monkeypatch,tmp_path):
    code=tmp_path/'code';code.mkdir();helpers={}
    monkeypatch.setattr(p,'NATIVE_FILES',('one.py','two.py'))
    for n in p.NATIVE_FILES:(code/n).write_bytes(n.encode());(code/n).chmod(0o400);helpers[n]=p.replica.pin(n.encode())
    revision='1'*40
    for n,raw in [('revision',(revision+'\n').encode()),('source-sha256',('a'*64+'\n').encode())]:
        (code.parent/n).write_bytes(raw);(code.parent/n).chmod(0o400)
    monkeypatch.setattr(p,'__file__',str(code/'infra/vcoco_person_pose_observations.py'))
    (code/'infra').mkdir();(code/'infra/vcoco_person_pose_observations.py').write_bytes(b'x')
    proof=dict(source=dict(producer_revision=revision,helpers=helpers,markers={n:p.rt.identity(code.parent/n)for n in ('revision','source-sha256')}))
    a=p.native_source(code,revision,proof);assert set(a)==set(p.NATIVE_FILES)
    (code/'one.py').chmod(0o600);(code/'one.py').write_bytes(b'changed');(code/'one.py').chmod(0o400)
    with pytest.raises(ValueError):p.native_source(code,revision,proof)


def fixture_report(monkeypatch,tmp_path):
    output=tmp_path/'out';output.mkdir();monkeypatch.setattr(p,'OUTPUT',output)
    images,banks=rows([1]*16);observations=[]
    for i,bank in enumerate(banks):
        a=arrays(1);bank['arrays']={n:dict(dtype=v.dtype.str,sha256=hashlib.sha256(v.tobytes()).hexdigest())for n,v in a.items()}
        name='image_%06d.npz'%i;(output/name).write_bytes(b'authored-%d'%i);(output/name).chmod(0o400)
        shapes=dict(person_ids=[1],boxes_original_xyxy=[1,4],detector_scores=[1],keypoints_original_xy=[1,133,2],raw_scores=[1,133],
            native_valid=[1,133],in_original_image=[1,133],image_size=[2],original_frame_index=[],original_slot=[],acquired_ordinal=[])
        types=dict(person_ids='<U12',boxes_original_xyxy='<f4',detector_scores='<f4',keypoints_original_xy='<f8',raw_scores='<f4',
            native_valid='|b1',in_original_image='|b1',image_size='<i8',original_frame_index='<i8',original_slot='<i8',acquired_ordinal='<i8')
        aa={k:dict(shape=s,dtype=types[k],sha256='0'*64)for k,s in shapes.items()}
        for x,y in [('person_ids','person_retained_ids'),('boxes_original_xyxy','person_retained_boxes'),('detector_scores','person_retained_scores')]:aa[x]['sha256']=bank['arrays'][y]['sha256']
        observations.append(dict(image_id=images[i]['image_id'],original_slot=i,acquired_ordinal=i,person_ids=bank['person_ids'],persons=1,
            original_frame_index=0,image_size=[4,4],file=name,identity=p.rt.identity(output/name),arrays=aa))
    proof=dict(images=images,banks=banks);pin=p.replica.pin(b'proof');revision='1'*40
    report=dict(schema=p.SCHEMA,stage='native_vcoco_person_pose_observations',status='pass',phase='complete',producer_revision=revision,
        image_id=p.IMAGE,proof_identity=pin,models_loaded=1,images=observations,source_inputs_assets_rehashed_after=True,private_prefix_removed=True,
        GPU_used=False,reference_metadata_read=False,FIT_performed=False,ownership_verified=False,quality_verified=False,adoption=False,full_image_fallback=False,
        persons=16,native_calls=[dict(run_completed=True,supplied_container='list',effective_feed_shape=[1,3,384,288],delegated_unmodified=True,
            effective_runtime_conversion_observed=False,supplied_array=dict(dtype='float64',shape=[3,384,288]),
            raw_simcc=[dict(dtype='float32',shape=[1,133,576]),dict(dtype='float32',shape=[1,133,768])])for _ in range(16)])
    return report,proof,revision,pin


def test_complete_native_validator(monkeypatch,tmp_path):
    r,proof,revision,pin=fixture_report(monkeypatch,tmp_path);p.validate_native(r,proof,revision,pin)


@pytest.mark.parametrize('field',('id','dtype','shape','calls','missing','owner','input_hash','feed'))
def test_native_validator_rejects_partial_or_changed(monkeypatch,tmp_path,field):
    r,proof,revision,pin=fixture_report(monkeypatch,tmp_path)
    if field=='id':r['images'][0]['person_ids']=['fake']
    elif field=='dtype':r['images'][0]['arrays']['raw_scores']['dtype']='<f8'
    elif field=='shape':r['images'][0]['arrays']['keypoints_original_xy']['shape']=[1,134,2]
    elif field=='calls':r['native_calls'].pop()
    elif field=='missing':r['images'].pop()
    elif field=='owner':r['ownership_verified']=True
    elif field=='input_hash':r['images'][0]['arrays']['detector_scores']['sha256']='f'*64
    elif field=='feed':r['native_calls'][0]['supplied_array']['dtype']='float32'
    with pytest.raises(ValueError):p.validate_native(r,proof,revision,pin)


def test_cleanup_exact_cid_name_and_renamed_survivor(monkeypatch,tmp_path):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);cid='a'*64;(out/'.container.cid').write_bytes((cid+'\n').encode())
    live={'active':True};name='owned';revision='1'*40
    def command(args,deadline):
        if args[1:3]==['inspect',cid]:return p.IMAGE+'|/'+name+'|'+p.ENTRY+'|'+revision
        if args[1:3]==['rm','-f']:live['active']=False;return ''
        if 'id='+cid in args:return cid if live['active']else ''
        return ''
    monkeypatch.setattr(p,'command',command);p.cleanup(name,revision,5)
    assert(out/'.container.cid').stat().st_mode&0o777==0o400
    live['active']=True
    def survivor(args,deadline):
        if args[1:3]==['inspect',cid]:return p.IMAGE+'|/'+name+'|'+p.ENTRY+'|'+revision
        return cid if 'id='+cid in args else ''
    monkeypatch.setattr(p,'command',survivor)
    with pytest.raises(ValueError):p.cleanup(name,revision,5)


def test_arguments_require_real_import_pins():
    with pytest.raises(ValueError):p.arguments([])
    r=p.arguments(['--replica-revision','1'*40,*[v for n in ('import','manifest','export')for v in
        ('--'+n+'-bytes','10','--'+n+'-sha256','a'*64)]])
    assert set(r.pins)=={'import','manifest','export'}
    with pytest.raises(ValueError):p.arguments(['--native','--code','/absolute','--revision','1'*40,'--deadline','nan'])


@pytest.mark.parametrize('completed',[False,True])
def test_host_dispatch_complete_binding_mounts_and_seal(monkeypatch,tmp_path,completed):
    root=tmp_path/'root';(root/'results').mkdir(parents=True);output=root/'results/pose';monkeypatch.setattr(p,'ROOT',root);monkeypatch.setattr(p,'OUTPUT',output)
    code=tmp_path/'code';code.mkdir();native_files=('infra/vcoco_person_pose_observations.py',)
    monkeypatch.setattr(p,'NATIVE_FILES',native_files)
    for n in native_files:
        path=code/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'authored source');path.chmod(0o400)
    for n in ('revision','source-sha256'):(code.parent/n).write_bytes(b'authored marker');(code.parent/n).chmod(0o400)
    before=dict(source={'authored':True},states='fixed')
    monkeypatch.setattr(p,'source',lambda *a:before);monkeypatch.setattr(p.replica.transport,'verify_azure_peer',lambda _:None)
    origin={'saved_status':'fail'if completed else'pass'};inputs=dict(images=[],banks=[],files={},original_replica_source=origin)
    completion_context=None
    if completed:
        completion_dir=tmp_path/'completion';completion_dir.mkdir();receipt=completion_dir/'report.json'
        receipt.write_bytes(b'authored sealed completion');receipt.chmod(0o400)
        completion_context=('3'*40,p.rt.identity(receipt));inputs.update(files={str(receipt):completion_context[1]},completion_source={'qualified_completion':True})
        monkeypatch.setattr(p,'completed_replica_inputs',lambda *a:inputs)
        monkeypatch.setattr(p,'replica_inputs',lambda *a:pytest.fail('Original PASS-only loader must not run'))
    else:monkeypatch.setattr(p,'replica_inputs',lambda *a:inputs)
    monkeypatch.setattr(p,'assets',lambda:dict(files={}));monkeypatch.setattr(p,'image_state',lambda _:dict(image=p.IMAGE))
    monkeypatch.setattr(p,'command',lambda *a:'');monkeypatch.setattr(p,'cleanup',lambda *a:None)
    seen=[]
    def launch(cmd,**kwargs):
        seen.extend(cmd)
        assert '--gpus'not in cmd and '--network'in cmd and cmd[cmd.index('--network')+1]=='none'
        assert kwargs['env']['DOCKER_HOST']=='unix://'+str(root/'docker.sock')
        (output/'native.json').write_bytes(p.encode(dict(status='pass',images=[],persons=0)));(output/'native.json').chmod(0o400)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',launch);monkeypatch.setattr(p,'validate_native',lambda *a:None)
    result=p.dispatch(code,'1'*40,None if completed else'2'*40,{},completion_context)
    saved=p.rt.strict((output/'report.json').read_bytes());assert result==saved and saved['status']=='pass'
    assert saved['source_inputs_assets_rehashed_after']is saved['owned_cleanup_verified']is True
    assert output.stat().st_mode&0o777==0o500
    assert all(x.stat().st_mode&0o777==0o400 for x in output.iterdir())
    assert not any('reference' in str(x)for x in seen)
    proof=p.rt.strict((output/'proof.json').read_bytes())
    assert proof['original_replica_source']==origin
    assert proof['input_mode']==saved['input_mode']==('completed_replica'if completed else'original_import')
    assert proof['completion_identity']==saved['completion_identity']==(completion_context[1]if completed else None)
    if completed:assert f'type=bind,src={receipt},dst={receipt},readonly'in seen


def test_fake_cpu_original_session_overlay_and_all16(monkeypatch,tmp_path):
    import contextlib,types
    import dwpose_smoke as dw
    import keypoint_rgb_dwpose as capability
    import coco_endpoint_evaluate as saved
    code=Path(p.__file__).resolve().parents[1];output=tmp_path/'out';output.mkdir(mode=0o700);monkeypatch.setattr(p,'OUTPUT',output)
    images,banks=rows([1]*16);proof=dict(source={'authored':True},files={},images=images,banks=banks,image_id=p.IMAGE)
    proofraw=p.encode(proof);(output/'proof.json').write_bytes(proofraw);(output/'proof.json').chmod(0o400)
    actual_iterdir=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda self:iter([Path('lo')])if str(self)=='/sys/class/net'else actual_iterdir(self))
    monkeypatch.setenv('WR_IMAGE_ID',p.IMAGE);monkeypatch.setattr(p,'native_source',lambda *a: {'sealedfixture':True})
    monkeypatch.setattr(dw,'source_identity',lambda:None);monkeypatch.setattr(dw,'validate_assets',lambda *a:{})
    monkeypatch.setattr(capability,'validate_smoke',lambda *a:{'actual_pass_receipt':'authored_mock'})
    monkeypatch.setattr(dw,'dependency_identity',lambda:{'native':'authored_mock'})
    monkeypatch.setattr(p.importlib.util,'find_spec',lambda *a:None)
    prefix=tmp_path/'overlay'
    @contextlib.contextmanager
    def overlay(report,persist):
        prefix.mkdir();report['private_prefix_removed']=False
        try:yield prefix
        finally:prefix.rmdir();report['private_prefix_removed']=True
    monkeypatch.setattr(dw,'private_prefix',overlay);monkeypatch.setattr(dw,'pip_argv',lambda *a:['fake_offline_install'])
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0))
    sessions=[]
    class Options:pass
    class Node:
        def __init__(self,name,shape):self.name,self.shape,self.type=name,shape,'tensor(float)'
    class Session:
        def __init__(self,path,sess_options,providers):
            self.options=sess_options;self.providers=providers;self.disabled=False;sessions.append(self)
        def disable_fallback(self):self.disabled=True
        def get_session_options(self):return self.options
        def get_inputs(self):return [Node('input',['batch',3,384,288])]
        def get_outputs(self):return [Node('simcc_'+a,['batch','MatMulsimcc_'+a+'_dim_1','MatMulsimcc_'+a+'_dim_2'])for a in ('x','y')]
        def get_providers(self):return self.providers
        def get_modelmeta(self):return types.SimpleNamespace(custom_metadata_map={})
        def run(self,names,feed):return [np.zeros((1,133,576),np.float32),np.zeros((1,133,768),np.float32)]
    ort=types.SimpleNamespace(SessionOptions=Options,ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL='sequential'),InferenceSession=Session)
    monkeypatch.setattr(dw,'import_runtime',lambda *a:(ort,{'authored_overlay':True}))
    def native_inference(session,boxes,rgb):
        points=[];scores=[]
        for _ in boxes:
            session.run(['simcc_x','simcc_y'],{'input':[np.zeros((3,384,288),np.float64)]})
            points.append(np.full((133,2),-2.,np.float64));scores.append(np.zeros(133,np.float32))
        return np.asarray(points,np.float64),np.asarray(scores,np.float32)
    original=types.SimpleNamespace(inference_pose=native_inference)
    class Loader:
        def exec_module(self,module):module.inference_pose=native_inference
    monkeypatch.setattr(p.importlib.util,'spec_from_file_location',lambda *a:types.SimpleNamespace(loader=Loader()))
    monkeypatch.setattr(p.importlib.util,'module_from_spec',lambda *a:original)
    validated=[];monkeypatch.setattr(saved,'validate_npz',lambda path,row:validated.append(row['original_slot']))
    @contextlib.contextmanager
    def fake_npz(*a,**kw):yield arrays(1)
    monkeypatch.setattr(np,'load',fake_npz)
    monkeypatch.setattr(p.replica.bank.rgb_inputs,'decode_rgb',lambda *a,**kw:np.zeros((4,4,3),np.uint8))
    r=p.cpu(code,'1'*40,p.replica.pin(proofraw),__import__('time').monotonic()+5)
    assert r['status']=='pass'and r['source_inputs_assets_rehashed_after']is True and r['persons']==16
    assert validated==list(range(16))and len(sessions)==1 and sessions[0].disabled is True
    assert sessions[0].options.intra_op_num_threads==4 and sessions[0].options.inter_op_num_threads==1
    assert r['private_prefix_removed']is True and len(r['native_calls'])==16
    assert len(list(output.glob('image_*.npz')))==16
    assert p.rt.strict((output/'native.json').read_bytes())==r


def test_host_deadline_signal_installed_and_restored(monkeypatch):
    import signal
    before={s:signal.getsignal(s)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    def work(*a):
        assert all(signal.getsignal(s)!=before[s]for s in before)
        raise TimeoutError('manufactured')
    monkeypatch.setattr(p,'dispatch_work',work)
    with pytest.raises(TimeoutError):p.dispatch(Path('/source'),'1'*40,'2'*40,{})
    assert all(signal.getsignal(s)==v for s,v in before.items())


def test_completed_inputs_preserve_failed_lineage_and_authenticate_real_pin_state(monkeypatch,tmp_path):
    folder=tmp_path/'completion';folder.mkdir();receipt=folder/'report.json';receipt.write_bytes(b'authored');receipt.chmod(0o400)
    monkeypatch.setattr(p.completion,'OUTPUT',folder);pin=p.rt.identity(receipt);calls=[]
    failed={'actual_status':'fail'};value=dict(images=[],banks=[],files={},states={},manifest={'original':True},
        original_replica_source=failed,original_sources={p.completion.IMPORT:{'states':'old-failed-source'}},
        completion_source={'new_status':'pass'},completion_identity=pin)
    original=copy.deepcopy(value)
    def auth(code,revision,identity):calls.append((code,revision,identity));return value
    monkeypatch.setattr(p.completion,'authenticate_completion',auth)
    monkeypatch.setattr(p,'replica_inputs',lambda *a:pytest.fail('Old loader unchanged, never fallback'))
    result=p.completed_replica_inputs(tmp_path,'3'*40,pin)
    assert calls==[(tmp_path,'3'*40,pin)] and value==original
    assert result['original_replica_source']==failed and result['original_source_states']=='old-failed-source'
    assert result['files'][str(receipt)]==pin and result['states'][str(receipt)]==p.replica.snapshot(receipt)
    receipt.chmod(0o600);receipt.write_bytes(b'changed');receipt.chmod(0o400)
    with pytest.raises(ValueError):p.completed_replica_inputs(tmp_path,'3'*40,pin)


@pytest.mark.parametrize('fault',['revision','bytes','sha','import','native','native_partial'])
def test_completion_cli_mutually_exclusive_all_or_none_and_native_forbidden(fault):
    args=['--completion-revision','3'*40,'--completion-bytes','10','--completion-sha256','a'*64]
    good=p.arguments(args);assert good.pins=={}and good.replica_revision is None
    assert good.completion_context==('3'*40,dict(bytes=10,sha256='a'*64))
    if fault in('revision','bytes','sha'):
        at={'revision':0,'bytes':2,'sha':4}[fault];del args[at:at+2]
    elif fault=='import':args+=['--replica-revision','2'*40]
    else:
        args+=['--native','--code','/absolute','--revision','1'*40,'--deadline','10','--proof-bytes','10','--proof-sha256','b'*64]
        if fault=='native_partial':args=args[2:]
    with pytest.raises(ValueError):p.arguments(args)


def test_completion_false_claim_has_no_original_fallback_or_model(monkeypatch,tmp_path):
    monkeypatch.setattr(p.replica.transport,'verify_azure_peer',lambda _:None)
    monkeypatch.setattr(p,'source',lambda *a:{'authored':True})
    monkeypatch.setattr(p,'completed_replica_inputs',lambda *a:(_ for _ in ()).throw(ValueError('Unsealed completion')))
    monkeypatch.setattr(p,'replica_inputs',lambda *a:pytest.fail('No legacy fallback'))
    monkeypatch.setattr(p,'assets',lambda:pytest.fail('No model asset gate after invalid completion'))
    with pytest.raises(ValueError):p.dispatch(tmp_path,'1'*40,None,{},('3'*40,dict(bytes=10,sha256='a'*64)))


def test_completion_helpers_whitelisted_and_old_globals_not_overridden():
    assert set(p.completion.HELPERS)<=set(p.NATIVE_FILES)
    assert p.publication.__file__.endswith('/infra/sealed_callback_publication.py')
    source=Path(p.__file__).read_text()
    assert 'replica.publish('not in source and 'publication.publish('in source
    assert 'completion.run('not in source and 'completion.authenticate_completion('in source
