"""Pure adapter serialization, full-T spies and isolated source tests only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).parents[1]/'infra'),str(Path(__file__).parents[1]/'src')]
import hand_mask_infer as adapter
import mediapipe_cpu_runtime_verify as rt
from world_reward.hand_observations import HandInstances, HandObservations, LandmarkEvidence
from world_reward.hand_temporal_masks import TemporalMaskFrame


def hands(count):
    xy=np.empty((count,21,2),np.float64)
    for i in range(count):xy[i,:,0]=np.linspace(1+i,5+i,21);xy[i,:,1]=np.linspace(2,7,21)
    return HandInstances(LandmarkEvidence(xy,np.isfinite(xy).all(axis=-1),'original_image','pixel'))


def frame(branch='A',count=2,unusable=True):
    raw=np.tile([1.,2.,5.,7.],(count,1));native=raw.astype(np.float32);usable=np.ones(count,bool)
    if count and unusable:raw[-1]=np.nan;native[-1]=np.nan;usable[-1]=False
    masks=np.zeros((count,480,640),bool);scores=np.where(usable,1.4,np.nan).astype(np.float64)if branch=='A'else None
    return TemporalMaskFrame(branch,2,2,tuple(range(count)),raw,native,usable,masks,usable.copy(),scores,
        'image_prompt'if branch=='A'else'video_inferred'if count else'no_anchor_abstention')


@pytest.mark.parametrize('branch,count',[('A',0),('A',1),('A',3),('B',0),('B',1),('B',3)])
def test_lossless_typed_temporal_roundtrip_without_fake_paired_record(tmp_path,branch,count):
    expected=frame(branch,count);path=tmp_path/'frame.npz';pin=adapter.save_temporal_frame(rt,np,path,expected)
    actual=adapter.load_temporal_frame(rt,np,path,2,branch)
    assert type(actual)is TemporalMaskFrame and rt.identity(path)==pin and path.stat().st_mode&0o777==0o400
    for name in expected.__dataclass_fields__:
        left,right=getattr(expected,name),getattr(actual,name)
        if isinstance(left,np.ndarray):assert left.dtype==right.dtype and left.tobytes()==right.tobytes()and not right.flags.writeable
        else:assert left==right


@pytest.mark.parametrize('fault',['extra','schema','branch','frame','grid','ids','dtype','box','mask','support','score','absent_score','evidence'])
def test_temporal_decoder_rejects_corruption_without_repairs(tmp_path,fault):
    path=tmp_path/'frame.npz';adapter.save_temporal_frame(rt,np,path,frame())
    with np.load(path,allow_pickle=False)as archive:values={n:archive[n]for n in archive.files}
    if fault=='extra':values['unknown']=np.array(1)
    elif fault=='schema':values['schema']=np.array('wrong',dtype='U64')
    elif fault=='branch':values['branch']=np.array('B',dtype='U1')
    elif fault=='frame':values['position']=np.array(1,np.int64)
    elif fault=='grid':values['image_size'][0]=479
    elif fault=='ids':values['proposal_ids'][0]=10
    elif fault=='dtype':values['masks']=values['masks'].astype(np.uint8)
    elif fault=='box':values['native_boxes'][0,0]+=1
    elif fault=='mask':values['masks'][-1,0,0]=True
    elif fault=='support':values['supported'][-1]=True
    elif fault=='score':values['raw_scores'][-1]=0
    elif fault=='absent_score':values['raw_scores_available']=np.array(False,np.bool_)
    else:values['evidence']=np.array('no_anchor_abstention',dtype='U32')
    path.chmod(0o600)
    with path.open('wb')as stream:np.savez_compressed(stream,**values)
    path.chmod(0o400)
    with pytest.raises(ValueError):adapter.load_temporal_frame(rt,np,path,2,'A')


class Image:
    def set_image(self,image):pass
    def predict(self,**kw):
        assert set(kw)=={'box','multimask_output'}and kw['multimask_output']is False
        n=len(kw['box']);m=np.zeros((n,1,480,640),bool);s=np.zeros(n,np.float32)
        return(m[0],s,None)if n==1 else(m,s[:,None],None)


def inputs(counts=(0,2,0)):
    manifest={'sequences':[],'images':[]};observations=[]
    for index in(4,39,74):
        manifest['sequences'].append(dict(sequence_lex_index=index,frames=len(counts)))
        observations.append(HandObservations(np.arange(len(counts),dtype=np.int64),(480,640),tuple(hands(n)for n in counts),'tiny',{'source':'fixture'}))
        for t in range(len(counts)):manifest['images'].append(dict(sequence_lex_index=index,frame_position=t,source_frame_id=t,
            file=f'{index}_{t}.jpg',sha256='a'*64))
    return manifest,observations


def test_temporal_full_three_stream_sorted_all_branches_summary_and_native_counts(tmp_path):
    manifest,observations=inputs();calls=[]
    def callbacks(original):
        def init():return {'ids':[]}
        def seed(state,**kw):state['ids'].append(kw['obj_id']);calls.append(('seed',kw['frame_idx'],kw['obj_id']))
        def propagate(state,**kw):
            a=kw['start_frame_idx'];positions=range(a,-1,-1)if kw['reverse']else range(a,len(original))
            for t in positions:yield t,state['ids'],np.zeros((len(state['ids']),1,480,640),np.float32)
        return dict(init_state=init,seed_box=seed,propagate=propagate,release_state=lambda s:s.clear())
    rows,summaries=adapter.infer_temporal_frames(rt,np,manifest,observations,Image(),tmp_path,
        lambda row:np.zeros((480,640,3),np.uint8),lambda:None,callbacks)
    assert len(rows)==18 and len(calls)==12 and len(summaries)==3
    assert [(r['sequence_lex_index'],r['frame_index'],r['branch'])for r in rows]==[(i,t,b)for i in(4,39,74)for t in range(3)for b in('A','B')]
    assert all(s['anchor_position']==1 and s['seeded_proposal_ids']==[0,1]and s['native_forward_frames']==2 and s['native_reverse_frames']==2 for s in summaries)
    for row in rows:
        decoded=adapter.load_temporal_frame(rt,np,tmp_path/row['file'],row['frame_index'],row['branch'])
        assert len(decoded.proposal_ids)==row['slots']and int(decoded.supported.sum())==row['supported']


def test_temporal_no_anchor_has_full_abstention_no_native_init_and_no_nan_json_summary(tmp_path):
    manifest,observations=inputs((0,0));calls=[]
    def callbacks(original):
        def fail(*args,**kw):calls.append(1);raise AssertionError('no anchor')
        return dict(init_state=fail,seed_box=fail,propagate=fail)
    rows,summary=adapter.infer_temporal_frames(rt,np,manifest,observations,Image(),tmp_path,
        lambda r:np.zeros((480,640,3),np.uint8),lambda:None,callbacks)
    assert len(rows)==12 and not calls and all(s['anchor_position']is None for s in summary)
    json.dumps(summary,allow_nan=False)


def test_numeric_stage_exact_bytes_modes_hashes_and_duplicate_refusal(tmp_path):
    source=tmp_path/'source';target=tmp_path/'stage';source.mkdir();target.mkdir()
    raw=b'procedural opaque JPEG bytes not decoded';rt.write(source/'native.jpg',raw)
    row=dict(file='native.jpg',**rt.identity(source/'native.jpg'))
    frozen=adapter.stage_temporal_jpegs(rt,source,[row],target,lambda:None)
    assert frozen==[(source/'native.jpg',target/'000000.jpg',{k:row[k]for k in('bytes','sha256')})]
    assert(target/'000000.jpg').read_bytes()==raw and(target/'000000.jpg').stat().st_mode&0o777==0o400
    with pytest.raises(FileExistsError):adapter.stage_temporal_jpegs(rt,source,[row],target,lambda:None)


def test_temporal_protocol_exact_profile_and_frozen_native_defaults(tmp_path):
    repo=Path(__file__).parents[1];(tmp_path/'configs').mkdir();path=tmp_path/adapter.TEMPORAL_PROTOCOL
    rt.write(path,(repo/adapter.TEMPORAL_PROTOCOL).read_bytes());value,pin=adapter.protocol(rt,tmp_path,'v3')
    assert value['sam2']['image_multimask_output']is False and value['sam2']['video_head_policy'].startswith('unchanged_native')
    assert adapter.profile()['cohort']=='v2'and adapter.profile('v3')['cohort']=='v3'
    assert adapter.SCAN_PINS not in adapter.source_helpers('v3')and adapter.TEMPORAL_SCAN_PINS in adapter.source_helpers('v3')
    assert adapter.scan.profile('v3')['protocol']in adapter.source_helpers('v3')
    with pytest.raises(ValueError):adapter.profile('v2')
    value['reseed']=True;path.chmod(0o600);path.write_text(json.dumps(value));path.chmod(0o400)
    with pytest.raises(ValueError):adapter.protocol(rt,tmp_path,'v3')


def test_native_video_state_exact_qualified_kwargs_grid_and_original_count(tmp_path):
    calls=[];state=dict(num_frames=5,video_height=480,video_width=640,
        offload_video_to_cpu=True,offload_state_to_cpu=False)
    class Native:
        def init_state(self,*args,**kwargs):calls.append((args,kwargs));return state
    result=adapter.native_video_state(rt,Native(),tmp_path,5)
    assert result is state
    assert calls==[((str(tmp_path),),dict(offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False))]


@pytest.mark.parametrize('field,value',[('num_frames',4),('num_frames',True),('video_height',479),
    ('video_width',641),('offload_video_to_cpu',False),('offload_state_to_cpu',True)])
def test_native_video_state_invalid_count_grid_or_storage_abstains_by_failure_not_fill(tmp_path,field,value):
    state=dict(num_frames=5,video_height=480,video_width=640,offload_video_to_cpu=True,offload_state_to_cpu=False)
    state[field]=value
    class Native:
        def init_state(self,*args,**kwargs):return state
    with pytest.raises(ValueError):adapter.native_video_state(rt,Native(),tmp_path,5)
    assert state=={}  # Failed owned state is released before any seed.


def test_isolated_host_import_stays_stdlib_without_numpy_and_shell_closure():
    repo=Path(__file__).parents[1]
    source=f"import sys;sys.path[:0]=[{str(repo/'infra')!r},{str(repo/'src')!r}];import hand_mask_infer;assert 'numpy'not in sys.modules"
    result=subprocess.run([sys.executable,'-I','-B','-S','-c',source],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run(['bash','-n',str(repo/'infra/run_hand_mask_infer.sh')],check=True)
    for args in(('--temporal-cohort',),('--temporal-cohort','v2'),('--temporal-cohort','v3','--other')):
        result=subprocess.run(['bash',str(repo/'infra/run_hand_mask_infer.sh'),*args],capture_output=True,
            env={'PATH':os.environ.get('PATH','/usr/bin:/bin')})
        assert result.returncode==2


def test_temporal_host_reuses_lock_offline_control_cohort_proof_and_independent_seal(tmp_path,monkeypatch):
    import dexycb_identity_infer as frontend
    revision='c'*40;root=tmp_path.resolve();code=root/'jobs'/revision/adapter.ENTRY/'code'
    (code/'infra').mkdir(parents=True);driver=code/'infra/hand_mask_infer.py';driver.write_bytes(b'fixture source')
    (root/'results').mkdir();lock=root/'jobs/.world-reward-h100.lock';lock.write_bytes(b'lock')
    monkeypatch.setattr(adapter,'ROOT',root);monkeypatch.setattr(adapter,'__file__',str(driver))
    monkeypatch.setattr(adapter.sys,'platform','linux');monkeypatch.setattr(adapter.os,'geteuid',lambda:0)
    monkeypatch.setattr(adapter.os,'uname',lambda:types.SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    pin=dict(bytes=1,sha256='a'*64);model=dict(child_image={'Id':frontend.binding.IMAGE},parent_image={},owner='b'*64,
        source_files={},selected_contract=dict(destination=root/'assets',entries={},manifest_identity=pin,extraction_receipt_identity=pin,config_identity=pin))
    images=[dict(sequence_lex_index=4,frame_position=t)for t in range(2)]
    evidence=dict(public={},acquisition={'manifest':pin},manifest={'images':images});pins={'outputs':{}}
    predictions=root/'scan/predictions';source={'source':'manufactured'};events=[]
    monkeypatch.setattr(adapter,'protocol',lambda *args:({},pin))
    def auth(*args):assert args[-1]=='v3';events.append('auth');return evidence,pins,predictions,{}
    monkeypatch.setattr(adapter,'authenticate_scan',auth)
    monkeypatch.setattr(adapter,'native_mounts',lambda *args:[code.parent])
    monkeypatch.setattr(frontend,'frontend_proof',lambda *args,**kw:(model,{}))
    def cleanup(*args):events.append('cleanup');assert os.fstat(9).st_ino==lock.stat().st_ino
    def control(args):events.append(args[0]);return b''
    fake=types.SimpleNamespace(**{n:getattr(rt,n)for n in('require','identity','pinned','write','canonical')},
        source=lambda *args:source,control=control,cleanup=cleanup)
    monkeypatch.setattr(adapter.scan,'runtime',lambda code:fake)
    def execute(args,**kw):
        if args[0]=='flock':assert kw['pass_fds']==(9,);events.append('flock');return types.SimpleNamespace(returncode=0)
        events.append('native');assert args[-2:]==['--temporal-cohort','v3']and args[args.index('--tmpfs')+1].endswith('2g')
        assert '--network'in args and args[args.index('--network')+1]=='none'and '--gpus'in args
        control=root/'results'/('hand-temporal-mask-infer-'+revision);proof=json.loads((control/'native-proof.json').read_bytes())
        assert proof['temporal_cohort']=='v3'
        outputs=[]
        for image in images:
            for branch in('A','B'):
                name=f"sequence_004_frame_{image['frame_position']:06d}_{branch}.npz"
                rt.write(control/'predictions'/name,b'opaque pinned fixture')
                outputs.append(dict(sequence_lex_index=4,frame_index=image['frame_position'],branch=branch,file=name))
        native=dict(stage='public_full_t_temporal_hand_sam2',temporal_cohort='v3',status='pass',phase='complete',
            source_binding=source,source_rehashed_after=True,a_frames=2,b_frames=2,outputs=outputs)
        rt.write(control/'predictions/report.json',json.dumps(native).encode());return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(adapter.subprocess,'run',execute)
    try:oldfd=os.dup(9)
    except OSError:oldfd=None
    try:result=adapter.run(root,code,revision,'v3')
    finally:
        if oldfd is not None:os.dup2(oldfd,9);os.close(oldfd)
    assert result['stage']=='hand_temporal_mask_infer_host_seal'and result['status']=='pass'
    assert result['owned_cleanup_verified']and result['source_rehashed_after']
    assert events.index('flock')<events.index('native')<events.index('cleanup')<len(events)-1
