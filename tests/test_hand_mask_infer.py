"""Real pure proposals, numeric roundtrips and offline native/control spies only."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).parents[1]/'infra'),str(Path(__file__).parents[1]/'src')]
import hand_mask_infer as adapter
import mediapipe_cpu_runtime_verify as rt
from world_reward.hand_observations import HandInstances,HandObservations,LandmarkEvidence
from world_reward.hand_mask_proposals import propose_hand_masks


def hands(n=1,invalid=False):
    xy=np.empty((n,21,2),np.float64)
    for i in range(n):xy[i,:,0]=np.linspace(1+i,5+i,21);xy[i,:,1]=np.linspace(2,7,21)
    if invalid and n:xy[0,5,0]=640
    return HandInstances(LandmarkEvidence(xy,np.isfinite(xy).all(axis=-1),'original_image','pixel'))


class Predictor:
    def __init__(self):self.images=[];self.calls=[]
    def set_image(self,rgb):self.images.append(rgb)
    def predict(self,**kw):
        self.calls.append({k:v.copy()if isinstance(v,np.ndarray)else v for k,v in kw.items()})
        n=len(kw['box']);masks=np.zeros((n,1,480,640),bool)
        if 'point_coords'in kw:masks[:,:,2:5,1:4]=True
        scores=np.linspace(-.4,1.4,n,dtype=np.float32)
        return (masks[0],scores,None)if n==1 else(masks,scores[:,None],None)


@pytest.mark.parametrize('count,invalid',[(0,False),(1,False),(2,False),(2,True)])
def test_actual_proposals_all_arrays_lossless_decoder_and_native_singleton_batches(tmp_path,count,invalid):
    bank=hands(count,invalid);predictor=Predictor();rgb=np.zeros((480,640,3),np.uint8)
    expected=propose_hand_masks(rgb,bank,predictor,frame_index=17)
    path=tmp_path/'frame.npz';pin=adapter.save_frame(rt,np,path,expected)
    decoded=adapter.load_frame(rt,np,path,17)
    assert rt.identity(path)==pin and path.stat().st_mode&0o777==0o400
    for field in adapter.fields(expected):
        a=getattr(expected,field.name);b=getattr(decoded,field.name)
        if isinstance(a,np.ndarray):assert a.tobytes()==b.tobytes()and not b.flags.writeable
        else:assert a==b
    assert len(predictor.images)==int(count>0)
    if count>0:
        assert len(predictor.calls)==2 and predictor.calls[0]['box'].dtype==np.float32
        eligible=np.flatnonzero(~expected.b_reuses_a)
        np.testing.assert_array_equal(predictor.calls[1]['box'],predictor.calls[0]['box'][eligible])
        assert (predictor.calls[1]['point_labels']==1).all()
    assert decoded.local_ids==tuple(f'hand:{i:06d}'for i in range(count))


@pytest.mark.parametrize('fault',['extra','dtype','grid','frame','score','reuse','mask','points','ids'])
def test_decoder_rejects_corruption_no_repair(tmp_path,fault):
    result=propose_hand_masks(np.zeros((480,640,3),np.uint8),hands(2,True),Predictor(),frame_index=0)
    path=tmp_path/'frame.npz';adapter.save_frame(rt,np,path,result)
    with np.load(path,allow_pickle=False)as archive:arrays={k:archive[k]for k in archive.files}
    if fault=='extra':arrays['unknown']=np.array(1)
    elif fault=='dtype':arrays['native_boxes']=arrays['native_boxes'].astype(np.float64)
    elif fault=='grid':arrays['image_size'][0]=479
    elif fault=='frame':arrays['frame_index']=np.array(1,np.int64)
    elif fault=='score':arrays['raw_scores_b'][0]=17
    elif fault=='reuse':arrays['b_reuses_a'][0]=False
    elif fault=='mask':arrays['masks_b'][0,0,0]=True
    elif fault=='points':arrays['points17'][0,1,0]+=1
    else:arrays['local_ids'][0]='identity:wrong'
    path.chmod(0o600)
    with path.open('wb')as stream:np.savez_compressed(stream,**arrays)
    path.chmod(0o400)
    with pytest.raises(ValueError):adapter.load_frame(rt,np,path,0)


def manifest_and_observations():
    seqs=[];rows=[];observations=[]
    for index in(4,39,74):
        seqs.append(dict(sequence_lex_index=index,frames=3))
        frames=(hands(0),hands(1),hands(2,True))
        observations.append(HandObservations(np.arange(3,dtype=np.int64),(480,640),frames,'tiny automatic',{'source':'manufactured test'}))
        for t in range(3):rows.append(dict(sequence_lex_index=index,frame_position=t,source_frame_id=t,
            file=f'sequence_{index:03d}_frame_{t:06d}.jpg',sha256='a'*64))
    return dict(sequences=seqs,images=rows),observations


def test_full_three_timeline_stream_empty_slots_native_counts_and_saved_original_index(tmp_path):
    manifest,observations=manifest_and_observations();model=Predictor();reads=[];checks=[]
    def read(row):reads.append((row['sequence_lex_index'],row['frame_position']));return np.zeros((480,640,3),np.uint8)
    rows=adapter.infer_frames(rt,np,manifest,observations,model,tmp_path,read,lambda:checks.append(1))
    assert len(rows)==9 and reads==[(i,t)for i in(4,39,74)for t in range(3)]
    assert len(model.images)==6 and len(model.calls)==12 and len(checks)==18
    assert [r['slots']for r in rows]==[0,1,2]*3
    for row in rows:
        loaded=adapter.load_frame(rt,np,tmp_path/row['file'],row['frame_index'])
        assert len(loaded.local_ids)==row['slots']and rt.identity(tmp_path/row['file'])=={k:row[k]for k in('bytes','sha256')}


def test_native_error_propagates_no_retry_and_retains_only_completed_frame_records(tmp_path):
    manifest,observations=manifest_and_observations();calls=[];outputs=[]
    class Error(Predictor):
        def predict(self,**kw):calls.append(1);raise RuntimeError('native failure')
    with pytest.raises(RuntimeError):adapter.infer_frames(rt,np,manifest,observations,Error(),tmp_path,
        lambda row:np.zeros((480,640,3),np.uint8),lambda:None,outputs)
    assert len(calls)==1 and len(outputs)==1 and outputs[0]['slots']==0
    assert [p.name for p in tmp_path.iterdir()]==[outputs[0]['file']]


@pytest.mark.parametrize('fault',['count','indices','rgbdtype','deadline'])
def test_full_timeline_bad_inputs_fail_instead_of_skipping(tmp_path,fault):
    manifest,observations=manifest_and_observations()
    if fault=='count':observations=observations[:-1]
    if fault=='indices':manifest['images'][0]['source_frame_id']=1
    dtype=np.float32 if fault=='rgbdtype'else np.uint8
    def check():
        if fault=='deadline':raise TimeoutError('budget')
    with pytest.raises((ValueError,TimeoutError)):adapter.infer_frames(rt,np,manifest,observations,Predictor(),tmp_path,
        lambda row:np.zeros((480,640,3),dtype),check)


def test_rich_frozen_protocol_uses_actual_root_bytes_and_corefields(tmp_path):
    repo=Path(__file__).parents[1];(tmp_path/'configs').mkdir();path=tmp_path/adapter.PROTOCOL
    raw=(repo/adapter.PROTOCOL).read_bytes();path.write_bytes(raw);path.chmod(0o444)
    value,pin=adapter.protocol(rt,tmp_path)
    assert pin==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert value['point_indices']==[0,*range(5,21)]and value['sam2']['apply_postprocessing']is True
    value['sam2']['multimask_output']=True;path.chmod(0o644);path.write_text(json.dumps(value));path.chmod(0o444)
    with pytest.raises(ValueError):adapter.protocol(rt,tmp_path)


def scan_fixture(tmp_path,monkeypatch):
    revision='b'*40;code=tmp_path/'code';(code/'configs').mkdir(parents=True)
    source={'producer_revision':revision,'helpers':{}};acq={'protocol':{'bytes':1,'sha256':'a'*64}}
    evidence=dict(acquisition=acq,runtime={'proof':'tiny'},manifest={'images':[{}]*9,'sequences':[dict(sequence_lex_index=i,frames=3)for i in(4,39,74)]})
    native=dict(stage='external_dexycb_full_t_mediapipe_hand_scan',status='pass',phase='complete',source_binding=source,
        producer_revision=revision,native_calls=9,native_graphs=1,cohort='v2',public_base=adapter.scan.profile('v2')['base'],protocol_identity=acq['protocol'],
        **{k:False for k in('gpu_used','private_values_read','accuracy_verified','availability_is_visibility','handedness_is_detection_confidence','actor_identity_inferred')},
        outputs=[dict(sequence_lex_index=i,frames=3,file=f'sequence_{i:03d}.npz')for i in(4,39,74)])
    control=adapter.scan.control_path(tmp_path,revision,'v2');out=control/'predictions';out.mkdir(parents=True)
    for name in {'report.json',*(f'sequence_{i:03d}'+suffix for i in(4,39,74)for suffix in('.npz','.npz.json'))}:
        rt.write(out/name,(json.dumps(native).encode()if name=='report.json'else b'opaque fixture'))
    outputs={p.name:rt.identity(p)for p in out.iterdir()}
    host=dict(stage='mediapipe_hand_scan_host_seal',status='pass',producer_revision=revision,source_binding=source,
        acquisition_pins=acq,runtime_pins=evidence['runtime'],source_rehashed_after=True,owned_cleanup_verified=True,gpu_used=False,
        private_values_read=False,accuracy_verified=False,cohort='v2',public_base=adapter.scan.profile('v2')['base'],
        protocol_identity=acq['protocol'],elapsed_seconds=1,outputs=outputs,native_report=native)
    rt.write(control/'report.json',json.dumps(host).encode())
    pins=dict(schema='world_reward.mediapipe_hand_scan_pins.v1',producer_revision=revision,report=rt.identity(control/'report.json'),outputs=outputs)
    rt.write(code/adapter.SCAN_PINS,json.dumps(pins).encode())
    monkeypatch.setattr(adapter.scan,'authenticate',lambda *args:evidence)
    fake=types.SimpleNamespace(**{name:getattr(rt,name)for name in('require','strict','identity','pinned')},source=lambda *args:source)
    return fake,code,pins,host,control


def test_scan_seals_actual_typed_seven_artifact_contract_and_byte_tamper(tmp_path,monkeypatch):
    fake,code,pins,host,control=scan_fixture(tmp_path,monkeypatch)
    _,actual,out,frozen=adapter.authenticate_scan(fake,tmp_path,code)
    assert actual==pins and len(frozen)==9 and len(list(out.iterdir()))==7
    path=out/'sequence_004.npz';path.chmod(0o600);path.write_bytes(b'changed');path.chmod(0o400)
    with pytest.raises(ValueError):adapter.authenticate_scan(fake,tmp_path,code)


@pytest.mark.parametrize('field,value',[('status','fail'),('private_values_read',True),('elapsed_seconds',601),('cohort','v1'),('source_rehashed_after',False)])
def test_scan_invalid_completion_contract_not_accepted_even_with_new_receipt_hash(tmp_path,monkeypatch,field,value):
    fake,code,pins,host,control=scan_fixture(tmp_path,monkeypatch);host[field]=value
    path=control/'report.json';path.chmod(0o600);path.write_text(json.dumps(host));path.chmod(0o400)
    pins['report']=rt.identity(path);config=code/adapter.SCAN_PINS;config.chmod(0o600);config.write_text(json.dumps(pins));config.chmod(0o400)
    with pytest.raises(ValueError):adapter.authenticate_scan(fake,tmp_path,code)


def test_narrow_mounts_drop_old_dataset_code_and_never_include_private_modeltask(monkeypatch):
    import dexycb_identity_infer as frontend
    code=adapter.ROOT/'jobs'/('b'*40)/adapter.ENTRY/'code'
    paths=[code,code.parent/'revision',code.parent/'source-sha256',adapter.ROOT/frontend.BASE/'inputs',
           frontend.binding.BUILD_CODE,frontend.binding.KERNEL_CODE,frontend.binding.DEST/'weights/sam2/sam2.1_hiera_large.pt']
    monkeypatch.setattr(frontend,'host_mounts',lambda *args:paths)
    predictions=[adapter.ROOT/'results/scan/predictions/sequence_004.npz']
    actual=adapter.native_mounts(code,'b'*40,predictions)
    assert code.parent in actual and code not in actual and adapter.ROOT/frontend.BASE/'inputs'not in actual
    assert adapter.ROOT/adapter.scan.profile('v2')['base']/'inputs'in actual
    assert all('eval_private'not in p.parts and 'hand_landmarker.task'not in str(p)for p in actual)
    with pytest.raises(ValueError):adapter.native_mounts(code,'b'*40,[adapter.ROOT/'validation/dexycb_hand_v2/eval_private/labels.npz'])


def test_static_closure_shell_nativeprecision_and_no_evaluator_values_or_detector_load():
    from azure_job import runtime_bundle_paths
    repo=Path(__file__).parents[1];files={str(p.relative_to(repo)):p.read_bytes()for top in('infra','src','configs')for p in(repo/top).rglob('*')if p.is_file()}
    paths=runtime_bundle_paths(files,'infra/run_hand_mask_infer.sh')
    assert all(name in paths for name in('infra/hand_mask_infer.py','infra/mediapipe_hand_evaluate.py','src/world_reward/hand_mask_proposals.py','infra/frontend_sam2_kernel_gate.py'))
    source=(repo/'infra/hand_mask_infer.py').read_text();shell=(repo/'infra/run_hand_mask_infer.sh').read_text()
    assert 'apply_postprocessing=True'in source and "dtype=torch.bfloat16"in source and '--memory\',\'16g'in source
    assert 'AutoModelForZeroShotObjectDetection'not in source and 'private_field('not in source and 'private_inventory('not in source
    assert 'pass_fds=(9,)'in source and source.index("subprocess.run(['flock'")<source.index("lock_check();idle()",source.index("subprocess.run(['flock'"))
    assert 'WR_CODE_REVISION' in shell and '960s'in shell and 'set +x'in shell


def test_host_lock_lifecycle_native_labels_posthash_and_cleanup_before_hash(tmp_path,monkeypatch):
    import dexycb_identity_infer as frontend
    revision='b'*40;root=tmp_path.resolve();code=root/'jobs'/revision/adapter.ENTRY/'code'
    (code/'infra').mkdir(parents=True);driver=code/'infra/hand_mask_infer.py';driver.write_bytes(b'ownedsource')
    (root/'results').mkdir();lock=root/'jobs/.world-reward-h100.lock';lock.write_bytes(b'lock')
    monkeypatch.setattr(adapter,'ROOT',root);monkeypatch.setattr(adapter,'__file__',str(driver))
    monkeypatch.setattr(adapter.sys,'platform','linux');monkeypatch.setattr(adapter.os,'geteuid',lambda:0)
    monkeypatch.setattr(adapter.os,'uname',lambda:types.SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    source={'source':'bound'};events=[];model={'child_image':{'Id':frontend.binding.IMAGE}};assets={'weights':'sealed'}
    frozen={root/'sealed.npz':{'bytes':1,'sha256':'a'*64}};predictions=root/'oldscan/predictions'
    evidence={'public':{},'acquisition':{'manifest':{'bytes':1,'sha256':'b'*64}}}
    pins={'outputs':{},'producer_revision':'c'*40}
    monkeypatch.setattr(adapter,'authenticate_scan',lambda *args:(evidence,pins,predictions,frozen))
    monkeypatch.setattr(adapter,'protocol',lambda *args:({},dict(bytes=1,sha256='c'*64)))
    monkeypatch.setattr(adapter,'native_mounts',lambda *args:[code.parent,root/'inputs'])
    def frontend_proof(*args,**kwargs):events.append('frontend');return model,assets
    monkeypatch.setattr(frontend,'frontend_proof',frontend_proof)
    def control(args,*unused):
        if args[0]=='nvidia-smi':events.append('idle');return b''
        assert args[:3]==['docker','ps','-aq'];return b''
    def source_call(*args):events.append('source');return source
    def cleanup(cid,name,child,rev):
        assert os.fstat(9).st_ino==lock.stat().st_ino and child==model['child_image']and rev==revision
        events.append('cleanup')
    fake=types.SimpleNamespace(**{k:getattr(rt,k)for k in('require','strict','identity','pinned','write','canonical')},
        source=source_call,control=control,cleanup=cleanup)
    monkeypatch.setattr(adapter.scan,'runtime',lambda code:fake)
    def run(args,**kw):
        if args[0]=='flock':
            assert kw['pass_fds']==(9,)and os.fstat(9).st_ino==lock.stat().st_ino
            events.append('lock');return types.SimpleNamespace(returncode=0)
        assert args[:2]==['docker','run']and '--network'in args and args[args.index('--network')+1]=='none'
        assert '--cap-drop'in args and args[args.index('--cap-drop')+1]=='ALL'
        assert args[args.index('--memory')+1]=='16g'and frontend.binding.IMAGE in args
        events.append('native')
        out=root/'results'/('hand-mask-infer-'+revision)/'predictions'
        native=dict(status='pass',phase='complete',source_binding=source,source_rehashed_after=True)
        rt.write(out/'report.json',json.dumps(native).encode());return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(adapter.subprocess,'run',run)
    try:original_fd=os.dup(9)
    except OSError:original_fd=None
    try:result=adapter.run(root,code,revision)
    finally:
        if original_fd is not None:os.dup2(original_fd,9);os.close(original_fd)
    assert result['status']=='pass'and result['owned_cleanup_verified']and result['source_rehashed_after']
    assert events.index('lock')<events.index('idle',events.index('lock'))<events.index('native')<events.index('cleanup')
    assert events.index('cleanup')<len(events)-1 and events[-1]=='frontend'
    if original_fd is None:
        with pytest.raises(OSError):os.fstat(9)


def test_native_proof_run_rejects_wrong_output_before_model_import(tmp_path,monkeypatch):
    revision='b'*40;monkeypatch.setenv('WR_CODE_REVISION',revision)
    proof_path=tmp_path/'proof.json';rt.write(proof_path,json.dumps({'remaining_seconds':1}).encode())
    with pytest.raises(ValueError):adapter.run_native(rt,tmp_path,tmp_path/'code',tmp_path/'output',proof_path,rt.identity(proof_path))
