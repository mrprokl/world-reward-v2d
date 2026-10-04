"""Frozen full-T hand slots -> paired native SAM2 prompts; no identity/GT claims."""
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time

if __name__=='__main__':
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    if not(re.fullmatch('[0-9a-f]{40}',revision)and code==Path('/srv/scenesmith/world-reward/jobs')/revision/'run_hand_mask_infer/code'
        and code.resolve()==code and Path(__file__).resolve()==code/'infra/hand_mask_infer.py'):
        raise ValueError('Actual immutable adapter required before imports')
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
import mediapipe_hand_scan as scan

ROOT=scan.ROOT;ENTRY='run_hand_mask_infer';BUDGET=900
PROTOCOL='configs/hand_mask_protocol_v1.json';SCAN_PINS='configs/mediapipe_hand_scan_v2_pins.json'
HELPERS=('infra/hand_mask_infer.py','infra/run_hand_mask_infer.sh','infra/mediapipe_hand_scan.py',
 'infra/mediapipe_hand_evaluate.py','infra/mediapipe_cpu_runtime_verify.py','infra/dexycb_identity_infer.py',
 'infra/bridge_frontend_bindings.py','src/world_reward/hand_mask_proposals.py',
 'src/world_reward/hand_observations.py','src/world_reward/automatic_candidate_bank.py',PROTOCOL,SCAN_PINS,
 *scan.source_helpers('v2'))
HELPERS=tuple(dict.fromkeys(HELPERS))
TEMPORAL_PROTOCOL='configs/hand_temporal_mask_protocol_v1.json'
TEMPORAL_SCAN_PINS='configs/mediapipe_hand_scan_v3_pins.json'


def profile(temporal_cohort=None):
    if temporal_cohort not in(None,'v3'):raise ValueError('Only explicit temporal v3 opt-in permitted')
    return dict(cohort='v3'if temporal_cohort else'v2',protocol=TEMPORAL_PROTOCOL if temporal_cohort else PROTOCOL,
        scan_pins=TEMPORAL_SCAN_PINS if temporal_cohort else SCAN_PINS,
        prefix='hand-temporal-mask-infer-'if temporal_cohort else'hand-mask-infer-')


def source_helpers(temporal_cohort=None):
    if temporal_cohort is None:return HELPERS
    profile(temporal_cohort)
    selected=tuple(TEMPORAL_PROTOCOL if n==PROTOCOL else TEMPORAL_SCAN_PINS if n==SCAN_PINS else n
        for n in HELPERS if n not in scan.source_helpers('v2'))
    return tuple(dict.fromkeys((*selected,*scan.source_helpers('v3'),'src/world_reward/hand_temporal_masks.py')))


def frontend_json(rt,model):
    """Project only the authenticated selected-contract destination into JSON."""
    rt.require(type(model)is dict and set(model)=={'child_image','parent_image','owner','source_files','selected_contract'},
        'Exact original frontend proof schema required')
    contract=model['selected_contract']
    rt.require(type(contract)is dict and set(contract)=={'destination','entries','manifest_identity','extraction_receipt_identity','config_identity'}
        and isinstance(contract['destination'],Path),'Exact selected-contract path schema required')
    destination=rt.canonical(contract['destination'])
    projected=dict(model,selected_contract=dict(contract,destination=str(destination)))
    json.dumps(projected,allow_nan=False)  # No other Path/unknown object is coerced.
    return projected


def protocol(rt,code,temporal_cohort=None):
    path=code/profile(temporal_cohort)['protocol'];pin=rt.identity(path)
    value=rt.strict(path.read_bytes())
    if temporal_cohort:
        expected=dict(schema='world-reward-hand-temporal-mask-protocol-v1',cohort='v3',subject=scan.profile('v3')['subject'],
            camera='836212060125',sequence_lex_indices=[4,39,74],original_image_size=[480,640],all_original_frames=True,
            sam2=dict(model='sam2.1_hiera_large',native_source_revision='2b90b9f5ceec907a1c18123530e92e794ad901a4',
                apply_postprocessing=True,autocast='bfloat16',tf32=False,seed=0,image_multimask_output=False,
                video_head_policy='unchanged_native_build_sam2_video_predictor_defaults_no_extra_overrides'),
            anchor='first_original_frame_with_at_least_one_usable_automatic_box',
            seed_slots='all_usable_original_anchor_slots_in_original_order_in_both_states',
            branch_ownership='reverse_before_anchor_forward_anchor_and_after',native_foreground='logits_strictly_greater_than_zero',
            frame_conversion='exact_original_JPEG_byte_copy_to_numeric_position_names_no_recompression',
            video_storage=dict(offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False),
            points=False,reseed=False,manual_prompts=False,late_birth_recovery=False,state_fusion=False)
        rt.require(all(type(value.get(k))is type(v)and value[k]==v for k,v in expected.items())
            and value.get('budgets_seconds',{}).get('all_three_clip_inference')==BUDGET,'Frozen native temporal protocol required')
        rt.require(rt.identity(path)==pin,'Temporal protocol changed');return value,pin
    expected=dict(schema='world-reward-hand-mask-protocol-v1',cohort='v2',subject='20200903-subject-04',
        camera='836212060125',sequence_lex_indices=[4,39,74],original_image_size=[480,640],
        point_indices=[0,*range(5,21)],all_original_frames=True,
        sam2=dict(model='sam2.1_hiera_large',multimask_output=False,apply_postprocessing=True,
                  autocast='bfloat16',tf32=False,seed=0))
    rt.require(all(type(value.get(k))is type(v)and value[k]==v for k,v in expected.items())
        and value.get('budgets_seconds',{}).get('sam2_inference')==BUDGET,'Frozen native paired protocol required')
    rt.require(rt.identity(path)==pin,'Protocol changed during read')
    return value,pin


def authenticate_scan(rt,root,code,temporal_cohort=None):
    selected=profile(temporal_cohort);cohort=selected['cohort']
    evidence=scan.authenticate(rt,root,code,cohort)
    pins,pin=scan.pins(rt,code,selected['scan_pins'],'world_reward.mediapipe_hand_scan_pins.v1',('report','outputs'))
    revision=pins['producer_revision'];old=root/'jobs'/revision/scan.ENTRY/'code'
    source=rt.source(root,old,revision,scan.ENTRY,scan.source_helpers(cohort))
    control=scan.control_path(root,revision,cohort);out=control/'predictions'
    host=rt.pinned(control/'report.json',pins['report'],4<<20)
    expected=dict(stage='mediapipe_hand_scan_host_seal',status='pass',producer_revision=revision,
        source_binding=source,acquisition_pins=evidence['acquisition'],runtime_pins=evidence['runtime'],
        source_rehashed_after=True,owned_cleanup_verified=True,gpu_used=False,private_values_read=False,
        accuracy_verified=False,cohort=cohort,public_base=scan.profile(cohort)['base'],protocol_identity=evidence['acquisition']['protocol'])
    rt.require(all(type(host.get(k))is type(v)and host[k]==v for k,v in expected.items()),'Genuine original scan host PASS required')
    rt.require(type(host.get('elapsed_seconds'))in(int,float)and 0<host['elapsed_seconds']<=scan.BUDGET,'Original full-scan budget differs')
    names={'report.json',*(f'sequence_{i:03d}'+s for i in(4,39,74)for s in('.npz','.npz.json'))}
    rt.require(set(pins['outputs'])==names and host.get('outputs')==pins['outputs']
        and {p.name for p in out.iterdir()}==names,'Exactly seven frozen scan artifacts required')
    frozen={control/'report.json':pins['report'],code/selected['scan_pins']:pin}
    for name,identity in pins['outputs'].items():
        path=out/name;rt.require(rt.identity(path,32<<20)==identity,'Frozen native scan artifact changed');frozen[path]=identity
    native=rt.strict((out/'report.json').read_bytes())
    rt.require(native==host.get('native_report')and native.get('stage')=='external_dexycb_full_t_mediapipe_hand_scan'
        and native.get('status')=='pass'and native.get('phase')=='complete'and native.get('source_binding')==source
        and native.get('producer_revision')==revision and type(native.get('native_calls'))is int
        and native['native_calls']==len(evidence['manifest']['images'])and native.get('native_graphs')==1
        and native.get('cohort')==cohort and native.get('public_base')==expected['public_base']
        and native.get('protocol_identity')==expected['protocol_identity']
        and all(native.get(k)is False for k in('gpu_used','private_values_read','accuracy_verified',
            'availability_is_visibility','handedness_is_detection_confidence','actor_identity_inferred')),
        'Complete lossless native full-T scan required')
    rt.require([(r['sequence_lex_index'],r['frames'],r['file'])for r in native.get('outputs',[])]==[
        (s['sequence_lex_index'],s['frames'],f"sequence_{s['sequence_lex_index']:03d}.npz")for s in evidence['manifest']['sequences']],
        'All original scan sequence outputs required')
    return evidence,pins,out,frozen


def native_mounts(code,revision,predictions,temporal_cohort=None):
    import dexycb_identity_infer as frontend
    paths=[p for p in frontend.host_mounts('masks',code,revision)
        if p!=code and p not in(code.parent/'revision',code.parent/'source-sha256',ROOT/frontend.BASE/'inputs')]
    paths += [code.parent,ROOT/scan.profile(profile(temporal_cohort)['cohort'])['base']/'inputs',*predictions]
    paths=sorted(set(paths))
    frontend.require(all(p.resolve()==p and ','not in str(p)and 'eval_private'not in p.parts
        and not p.is_relative_to(ROOT/'data')and not p.is_relative_to(ROOT/'vendor')for p in paths),
        'Readonly source/public/frontend leaves only')
    return paths


def save_frame(rt,np,path,result):
    arrays={f.name:getattr(result,f.name)for f in fields(result)if isinstance(getattr(result,f.name),np.ndarray)}
    arrays.update(frame_index=np.array(result.frame_index,np.int64),image_size=np.array(result.image_size,np.int64),
        point_indices=np.array(result.point_indices,np.int64),local_ids=np.array(result.local_ids,dtype='U32'),
        diagnostics=np.array(result.diagnostics,dtype='U64'))
    rt.require(not any(v.dtype.hasobject for v in arrays.values()),'No pickle or dropped proposal field')
    with path.open('xb')as stream:
        os.fchmod(stream.fileno(),0o400);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return rt.identity(path,16<<20)


def load_frame(rt,np,path,frame_index):
    """Lossless CPU-only restoration; unavailable raw floats are not repaired."""
    from world_reward.hand_mask_proposals import HandMaskProposals,POINT_INDICES
    from world_reward.hand_observations import HandInstances,LandmarkEvidence
    arrays=('original_xy','original_supported','raw_boxes','clipped_boxes','native_boxes','box_usable',
        'points17','native_points17','points_usable','points_outside','masks_a','masks_b',
        'mask_supported_a','mask_supported_b','raw_scores_a','raw_scores_b','b_reuses_a')
    with np.load(path,allow_pickle=False)as archive:
        rt.require(set(archive.files)==set(arrays)|{'frame_index','image_size','point_indices','local_ids','diagnostics'},'Exact mask evidence keys required')
        a={k:archive[k]for k in archive.files}
    rt.require(all(not x.dtype.hasobject for x in a.values())and a['frame_index'].dtype==np.int64
        and a['frame_index'].shape==()and int(a['frame_index'])==frame_index
        and a['image_size'].dtype==np.int64 and a['image_size'].tolist()==[480,640]
        and a['point_indices'].dtype==np.int64 and a['point_indices'].tolist()==list(POINT_INDICES),'Original mask grid/indices required')
    n=len(a['original_xy']);shapes=dict(original_xy=(n,21,2),original_supported=(n,21),
        raw_boxes=(n,4),clipped_boxes=(n,4),native_boxes=(n,4),box_usable=(n,),points17=(n,17,2),
        native_points17=(n,17,2),points_usable=(n,17),points_outside=(n,17),masks_a=(n,480,640),masks_b=(n,480,640),
        mask_supported_a=(n,),mask_supported_b=(n,),raw_scores_a=(n,),raw_scores_b=(n,),b_reuses_a=(n,))
    floats={'original_xy','raw_boxes','clipped_boxes','native_boxes','points17','native_points17','raw_scores_a','raw_scores_b'}
    for name,shape in shapes.items():
        dtype=np.float32 if name.startswith('native_')else np.float64 if name in floats else np.bool_
        rt.require(a[name].shape==shape and a[name].dtype==dtype,'Original array dtype/shape differs')
    ids=tuple(f'hand:{i:06d}'for i in range(n))
    rt.require(a['local_ids'].dtype==np.dtype('U32')and a['local_ids'].shape==(n,)and tuple(a['local_ids'])==ids
        and a['diagnostics'].dtype==np.dtype('U64')and a['diagnostics'].shape==(n,),'All local slots retained')
    class Replay:
        def set_image(self,rgb):pass
        def predict(self,**kw):
            arm='b'if 'point_coords'in kw else'a';eligible=a['box_usable']&~a['b_reuses_a']if arm=='b'else a['box_usable']
            masks,scores=a['masks_'+arm][eligible],a['raw_scores_'+arm][eligible]
            return (masks,scores,None)if len(masks)==1 else(masks[:,None],scores[:,None],None)
    from world_reward.hand_mask_proposals import propose_hand_masks
    restored=propose_hand_masks(np.zeros((480,640,3),np.uint8),HandInstances(LandmarkEvidence(
        a['original_xy'],a['original_supported'],'original_image','pixel')),Replay(),frame_index=frame_index)
    rt.require(all(getattr(restored,k).tobytes()==a[k].tobytes()for k in arrays)
        and restored.diagnostics==tuple(a['diagnostics']),'Lossless evidence/reuse/score reconstruction differs')
    return restored


def infer_frames(rt,np,manifest,observations,predictor,out,read_rgb,check,outputs=None):
    from world_reward.hand_mask_proposals import propose_hand_masks
    rt.require(len(manifest['sequences'])==len(observations)==3,'Exactly three lossless original scans required')
    rows=[] if outputs is None else outputs
    for sequence,obs in zip(manifest['sequences'],observations):
        frames=sequence['frames'];index=sequence['sequence_lex_index']
        rt.require(obs.image_size==(480,640)and len(obs.frames)==frames
            and np.array_equal(obs.frame_index,np.arange(frames,dtype=np.int64)),'Exact full-T restored slots required')
        original=[r for r in manifest['images']if r['sequence_lex_index']==index]
        rt.require(len(original)==frames,'No omitted original RGB frames')
        for t,(row,hands)in enumerate(zip(original,obs.frames)):
            check();rt.require(row['frame_position']==row['source_frame_id']==t,'Original indices differ')
            result=propose_hand_masks(read_rgb(row),hands,predictor,frame_index=t)
            name=f'sequence_{index:03d}_frame_{t:06d}.npz';pin=save_frame(rt,np,out/name,result)
            rows.append(dict(file=name,sequence_lex_index=index,frame_index=t,rgb_file=row['file'],
                rgb_sha256=row['sha256'],**pin,slots=len(result.local_ids),usable_boxes=int(result.box_usable.sum()),
                b_interventions=int((~result.b_reuses_a).sum()),a_supported=int(result.mask_supported_a.sum()),
                b_supported=int(result.mask_supported_b.sum())))
            check()
    rt.require(len(rows)==len(manifest['images']),'All original frames retained');return rows


TEMPORAL_SCHEMA='world_reward.temporal_hand_mask_frame.v1'


def save_temporal_frame(rt,np,path,result):
    arrays={name:getattr(result,name)for name in('raw_boxes','native_boxes','box_usable','masks','supported')}
    arrays.update(schema=np.array(TEMPORAL_SCHEMA,dtype='U64'),branch=np.array(result.branch,dtype='U1'),
        position=np.array(result.position,np.int64),original_frame_id=np.array(result.original_frame_id,np.int64),
        image_size=np.array(result.masks.shape[1:],np.int64),proposal_ids=np.array(result.proposal_ids,np.int64),
        evidence=np.array(result.evidence,dtype='U32'),raw_scores=np.empty(0,np.float64)if result.raw_scores is None else result.raw_scores,
        raw_scores_available=np.array(result.raw_scores is not None,np.bool_))
    rt.require(not any(v.dtype.hasobject for v in arrays.values()),'No dropped temporal fields or pickle')
    with path.open('xb')as stream:
        os.fchmod(stream.fileno(),0o400);np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    return rt.identity(path,16<<20)


def load_temporal_frame(rt,np,path,frame_index,branch):
    """Pure CPU lossless decoder, never interprets anchor IDs as physical hands."""
    from world_reward.hand_temporal_masks import TemporalMaskFrame
    from world_reward.hand_observations import _copy
    names={'schema','branch','position','original_frame_id','image_size','proposal_ids','evidence','raw_scores',
        'raw_scores_available','raw_boxes','native_boxes','box_usable','masks','supported'}
    with np.load(path,allow_pickle=False)as archive:
        rt.require(set(archive.files)==names,'Exactly original temporal evidence fields required');a={n:archive[n]for n in names}
    def scalar(name,dtype):return a[name].shape==()and a[name].dtype==np.dtype(dtype)
    rt.require(branch in('A','B')and scalar('schema','U64')and str(a['schema'])==TEMPORAL_SCHEMA
        and scalar('branch','U1')and str(a['branch'])==branch and scalar('position',np.int64)
        and int(a['position'])==frame_index and scalar('original_frame_id',np.int64)and int(a['original_frame_id'])>=0
        and a['image_size'].dtype==np.int64 and a['image_size'].shape==(2,)and a['image_size'].tolist()==[480,640]
        and scalar('evidence','U32')and scalar('raw_scores_available',np.bool_),'Original temporal metadata/grid required')
    n=len(a['proposal_ids']);rt.require(a['proposal_ids'].dtype==np.int64 and a['proposal_ids'].shape==(n,)
        and np.array_equal(a['proposal_ids'],np.arange(n,dtype=np.int64))and n<=4,'Every native proposal slot retained')
    for name,shape,dtype in (('raw_boxes',(n,4),np.float64),('native_boxes',(n,4),np.float32),
        ('box_usable',(n,),np.bool_),('masks',(n,480,640),np.bool_),('supported',(n,),np.bool_)):
        rt.require(a[name].dtype==dtype and a[name].shape==shape,'Temporal original array dtype/shape differs')
    raw=a['raw_boxes'];native=a['native_boxes'];usable=a['box_usable'];finite=np.isfinite(raw).all(axis=1)
    rt.require(np.all(finite|np.isnan(raw).all(axis=1))and np.all(~finite|(raw[:,:2]<=raw[:,2:]).all(axis=1)),
        'Raw box is finite min/max or explicit unavailable NaN')
    clipped=np.clip(raw,[0,0,0,0],[640,480,640,480]).astype(np.float32)
    rt.require(native.tobytes()==clipped.tobytes()and np.all(~usable|(finite&(native[:,0]<native[:,2])&(native[:,1]<native[:,3]))),
        'Original clipped FP32 boxes changed')
    rt.require(np.array_equal(a['supported'],usable)and not a['masks'][~usable].any(),'Unusable slots cannot be dropped or filled')
    available=bool(a['raw_scores_available']);score=a['raw_scores'];evidence=str(a['evidence'])
    rt.require(score.dtype==np.float64 and score.shape==((n,)if available else(0,))
        and ((branch=='A'and available and evidence=='image_prompt'and np.isfinite(score[usable]).all()and np.isnan(score[~usable]).all())
            or(branch=='B'and not available and evidence in('anchor_prompt','video_inferred','no_anchor_abstention'))),
        'Native IMAGE scores and VIDEO availability must remain distinct')
    rt.require(evidence!='no_anchor_abstention'or n==0,'No-anchor abstention has no fabricated proposals')
    return TemporalMaskFrame(branch,int(a['position']),int(a['original_frame_id']),tuple(map(int,a['proposal_ids'])),
        *(_copy(a[name],'readonly temporal evidence')for name in('raw_boxes','native_boxes','box_usable','masks','supported')),
        _copy(score,'raw IMAGE scores')if available else None,evidence)


def stage_temporal_jpegs(rt,root,rows,directory,check):
    """Owned /tmp byte copies preserve JPEG decoding; no recompression/ID alias."""
    frozen=[]
    for position,row in enumerate(rows):
        check();source=root/row['file'];wanted={k:row[k]for k in('bytes','sha256')}
        rt.require(rt.identity(source,16<<20)==wanted,'Original JPEG changed before numeric staging')
        raw=source.read_bytes();rt.require(len(raw)==wanted['bytes']and hashlib.sha256(raw).hexdigest()==wanted['sha256'],'Original JPEG read changed')
        target=directory/f'{position:06d}.jpg'
        with target.open('xb')as stream:os.fchmod(stream.fileno(),0o400);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        rt.require(rt.identity(source,16<<20)==wanted and rt.identity(target,16<<20)==wanted,'Numeric JPEG stage differs')
        frozen.append((source,target,wanted));check()
    return frozen


def native_video_state(rt,predictor,directory,total_frames):
    """Match the qualified stock SAM2 storage policy and full original grid."""
    rt.require(type(total_frames)is int and total_frames>0,'Original positive frame count required')
    state=predictor.init_state(str(directory),offload_video_to_cpu=True,
        offload_state_to_cpu=False,async_loading_frames=False)
    try:
        rt.require(type(state)is dict and all(type(state.get(name))is int and state[name]==expected
            for name,expected in(('num_frames',total_frames),('video_height',480),('video_width',640)))
            and state.get('offload_video_to_cpu')is True and state.get('offload_state_to_cpu')is False,
            'Native VIDEO state must preserve qualified storage/full frame count/grid')
    except BaseException:
        if type(state)is dict:state.clear()
        raise
    return state


def infer_temporal_frames(rt,np,manifest,observations,predictor,out,read_rgb,check,video_callbacks,outputs=None):
    from world_reward.hand_temporal_masks import stream_temporal_hand_masks
    rows=[]if outputs is None else outputs;summaries=[]
    rt.require(len(manifest['sequences'])==len(observations)==3,'Exactly three original scans required')
    for sequence,obs in zip(manifest['sequences'],observations):
        count=sequence['frames'];index=sequence['sequence_lex_index']
        original=[r for r in manifest['images']if r['sequence_lex_index']==index]
        rt.require(obs.image_size==(480,640)and len(obs.frames)==count and len(original)==count
            and np.array_equal(obs.frame_index,np.arange(count,dtype=np.int64))and all(
                r['frame_position']==r['source_frame_id']==t for t,r in enumerate(original)),'Full original temporal scan/RGB required')
        callbacks=video_callbacks(original)
        def frames():
            for row,hands in zip(original,obs.frames):check();yield row['source_frame_id'],read_rgb(row),hands
        def emit(result):
            check();t=result.position;row=original[t];name=f'sequence_{index:03d}_frame_{t:06d}_{result.branch}.npz'
            pin=save_temporal_frame(rt,np,out/name,result)
            rows.append(dict(file=name,sequence_lex_index=index,frame_index=t,original_frame_id=result.original_frame_id,
                branch=result.branch,rgb_file=row['file'],rgb_sha256=row['sha256'],slots=len(result.proposal_ids),
                usable_boxes=int(result.box_usable.sum()),supported=int(result.supported.sum()),evidence=result.evidence,**pin));check()
        summary=stream_temporal_hand_masks(frames(),frame_ids=np.array([r['source_frame_id']for r in original],np.int64),
            image_size=(480,640),image_predictor=predictor,emit=emit,**callbacks)
        summaries.append(dict(sequence_lex_index=index,frames=count,anchor_position=summary.anchor_position,
            anchor_frame_id=summary.anchor_frame_id,anchor_raw_boxes=np.where(np.isfinite(summary.anchor_raw_boxes),summary.anchor_raw_boxes,None).tolist(),
            anchor_native_boxes=np.where(np.isfinite(summary.anchor_native_boxes),summary.anchor_native_boxes,None).tolist(),anchor_box_usable=summary.anchor_box_usable.tolist(),
            seeded_proposal_ids=list(summary.seeded_proposal_ids),native_forward_frames=summary.native_forward_frames,
            native_reverse_frames=summary.native_reverse_frames))
    rows.sort(key=lambda r:(r['sequence_lex_index'],r['frame_index'],r['branch']))
    rt.require(len(rows)==2*len(manifest['images'])and len({r['file']for r in rows})==len(rows),'Exact paired full-T temporal outputs required')
    return rows,summaries


def run_native(rt,root,code,out,proof_path,proof_pin,temporal_cohort=None):
    import dexycb_identity_infer as frontend
    import mediapipe_hand_evaluate as restoration
    selected=profile(temporal_cohort);cohort=selected['cohort'];helpers=source_helpers(temporal_cohort)
    start=time.monotonic();revision=os.environ['WR_CODE_REVISION'];proof=rt.pinned(proof_path,proof_pin,4<<20)
    deadline=start+proof['remaining_seconds'];check=lambda:rt.require(time.monotonic()<=deadline,'Inclusive all3 budget exceeded')
    rt.require(root==ROOT and code==root/'jobs'/revision/ENTRY/'code'and Path(__file__).resolve()==code/HELPERS[0]
        and out==root/'results'/(selected['prefix']+revision)/'predictions'
        and (proof.get('temporal_cohort')==temporal_cohort if temporal_cohort else 'temporal_cohort'not in proof),
        'Exact source/native namespace/cohort required')
    rt.require(sys.platform=='linux'and os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==frontend.binding.IMAGE
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
        and int(next(s.split()[1]for s in Path('/proc/self/status').read_text().splitlines()if s.startswith('CapEff:')),16)==0,
        'Actual cap-drop offline native CUDA container required')
    rt.require(not any((root/p).exists()for p in('data','vendor',scan.profile(cohort)['base']+'/eval_private',
        scan.profile(cohort)['base']+'/report.json'))and not Path('/opt/mediapipe-task').exists(),'No private/task/challenge mounts')
    source=rt.source(root,code,revision,ENTRY,helpers);rt.require(source==proof['source_binding'],'Own source changed')
    rt.require(out.is_dir()and out.stat().st_uid==0 and stat.S_IMODE(out.stat().st_mode)==0o700 and not tuple(out.iterdir()),'Fresh native leaf required')
    frozen={Path(p):pin for p,pin in proof['frozen'].items()}
    report=dict(stage='public_full_t_paired_hand_sam2',status='fail',phase='preflight',producer_revision=revision,
        source_binding=source,protocol_identity=proof['protocol_identity'],manifest_identity=proof['manifest'],
        scan_pins=proof['scan_pins'],image_id=frontend.binding.IMAGE,budget_seconds=BUDGET,
        private_values_read=False,quality_verified=False,identity_accepted=False,contacts_inferred=False,
        geometry_inferred=False,all_original_frames=True,network='none',device='cuda',sam2_native_postprocessing=True,
        encoder_attempts=0,encoder_completed=0,a_attempts=0,a_completed=0,b_attempts=0,b_completed=0,outputs=[])
    if temporal_cohort:report.update(stage='public_full_t_temporal_hand_sam2',temporal_cohort=temporal_cohort,
        a_image_multimask_output=False,video_head_policy='unchanged_native_build_sam2_video_predictor_defaults_no_extra_overrides',
        video_storage=dict(offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False),
        video_logits_conversion='native_BF16_or_FP32_to_FP32_exact_values_for_numpy_no_threshold_change',
        video_states_attempted=0,video_states_completed=0,video_seed_attempts=0,video_seed_completed=0,
        video_frame_attempts=0,video_frame_completed=0,sequence_summaries=[])
    model=None;installed=None;failure=None
    try:
        rt.require(all(rt.identity(p,32<<20)==pin for p,pin in frozen.items()),'ALL scans/RGB frozen before values')
        _,pin=protocol(rt,code,temporal_cohort);rt.require(pin==proof['protocol_identity'],'Protocol changed')
        manifest,_=scan.public_inputs(rt,root,code,proof['acquisition'],cohort)
        model,assets=frontend.frontend_proof(code);rt.require(frontend_json(rt,model)==proof['frontend']and assets==proof['assets'],'Frontend source/model proof differs');check()
        import numpy as np
        import torch
        from PIL import Image
        from importlib import metadata
        rt.require(metadata.version('numpy')=='1.26.3'and metadata.version('torch')=='2.5.1+cu124','Measured native runtime required')
        rt.require(torch.cuda.is_available()and 'H100'in torch.cuda.get_device_name(),'Native H100 required')
        torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        installed=frontend.binding.installed_sam2(model)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
            str(frontend.binding.DEST/'weights/sam2/sam2.1_hiera_large.pt'),device='cuda',mode='eval',apply_postprocessing=True))
        class Native:
            def set_image(self,rgb):
                report['encoder_attempts']+=1;check()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):predictor.set_image(rgb)
                torch.cuda.synchronize();report['encoder_completed']+=1;check()
            def predict(self,**kw):
                arm='b'if 'point_coords'in kw else 'a';report[arm+'_attempts']+=1;check()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):value=predictor.predict(**kw)
                torch.cuda.synchronize();report[arm+'_completed']+=1;check();return value
        observations=[]
        for s in manifest['sequences']:
            path=Path(proof['predictions'])/f"sequence_{s['sequence_lex_index']:03d}.npz"
            observations.append(restoration.reload_observations(rt,np,path,Path(str(path)+'.json'),s['frames'],{'scan':proof['scan_pins']}));check()
        def read_rgb(row):
            path=root/scan.profile(cohort)['base']/'inputs'/row['file']
            rt.require(rt.identity(path,16<<20)=={k:row[k]for k in('bytes','sha256')},'JPEG changed before decode')
            with Image.open(path)as image:
                rt.require(image.format=='JPEG'and image.mode=='RGB'and image.size==(640,480),'Original RGB required')
                return np.asarray(image).copy()
        if temporal_cohort:
            from sam2.build_sam import build_sam2_video_predictor
            video_model=[None];stages=[]
            with tempfile.TemporaryDirectory(prefix='world-reward-hand-video-')as temporary:
                def video_callbacks(original):
                    directory=Path(temporary)/f'sequence_{original[0]["sequence_lex_index"]:03d}'
                    directory.mkdir(mode=0o700)
                    staged=stage_temporal_jpegs(rt,root/scan.profile(cohort)['base']/'inputs',original,directory,check);stages.extend(staged)
                    def init():
                        check();report['video_states_attempted']+=1
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            if video_model[0]is None:video_model[0]=build_sam2_video_predictor(
                                'configs/sam2.1/sam2.1_hiera_l.yaml',str(frontend.binding.DEST/'weights/sam2/sam2.1_hiera_large.pt'),
                                device='cuda',mode='eval',apply_postprocessing=True,vos_optimized=False)
                            value=native_video_state(rt,video_model[0],directory,len(original))
                        torch.cuda.synchronize();report['video_states_completed']+=1;check();return value
                    def seed(state,**kwargs):
                        check();report['video_seed_attempts']+=1
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):video_model[0].add_new_points_or_box(state,**kwargs)
                        torch.cuda.synchronize();report['video_seed_completed']+=1;check()
                    def propagate(state,**kwargs):
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            for position,ids,logits in video_model[0].propagate_in_video(state,**kwargs):
                                check();report['video_frame_attempts']+=1
                                value=logits.detach().to(torch.float32).cpu().numpy()
                                torch.cuda.synchronize();report['video_frame_completed']+=1;check();yield position,ids,value
                    def release(state):state.clear();torch.cuda.synchronize();check()
                    return dict(init_state=init,seed_box=seed,propagate=propagate,release_state=release)
                report['phase']='native_temporal_masks'
                _,summaries=infer_temporal_frames(rt,np,manifest,observations,Native(),out,read_rgb,check,video_callbacks,report['outputs'])
                report['sequence_summaries']=summaries
                rt.require(all(rt.identity(source,16<<20)==pin==rt.identity(target,16<<20)for source,target,pin in stages),
                    'Exact original/numeric JPEG copies changed after video inference');check()
            active=sum(r['usable_boxes']>0 for r in report['outputs']if r['branch']=='A')
            states=2*sum(s['anchor_position']is not None for s in summaries)
            seeds=2*sum(len(s['seeded_proposal_ids'])for s in summaries)
            native_frames=sum(s['native_forward_frames']+s['native_reverse_frames']for s in summaries)
            rt.require(report['encoder_attempts']==report['encoder_completed']==report['a_attempts']==report['a_completed']==active
                and report['b_attempts']==report['b_completed']==0 and report['video_states_attempted']==report['video_states_completed']==states
                and report['video_seed_attempts']==report['video_seed_completed']==seeds
                and report['video_frame_attempts']==report['video_frame_completed']==native_frames,'Exact full-T native temporal counters differ')
            report.update(a_frames=len(manifest['images']),b_frames=len(manifest['images']),video_model_loads=int(video_model[0]is not None))
        else:
            report['phase']='native_paired_masks';infer_frames(rt,np,manifest,observations,Native(),out,read_rgb,check,report['outputs'])
            active=sum(r['usable_boxes']>0 for r in report['outputs']);positive=sum(r['b_interventions']>0 for r in report['outputs'])
            rt.require(report['encoder_attempts']==report['encoder_completed']==report['a_attempts']==report['a_completed']==active
                and report['b_attempts']==report['b_completed']==positive,'Exact shared-encode native paired call counts')
        rt.require({p.name for p in out.iterdir()}=={r['file']for r in report['outputs']},'No omitted/extra output');report.update(status='pass',phase='complete')
    except BaseException as error:failure=error;report['error_type']=type(error).__name__
    finally:
        try:
            rt.require(rt.source(root,code,revision,ENTRY,helpers)==source and rt.identity(proof_path,4<<20)==proof_pin
                and all(rt.identity(p,32<<20)==pin for p,pin in frozen.items()),'Public/source changed after inference')
            if model is not None:rt.require(frontend.frontend_proof(code)==(model,proof['assets']),'Assets changed')
            if installed is not None:rt.require(frontend.binding.installed_sam2(model)==installed,'Installed SAM2 changed')
            rt.require(all(rt.identity(out/r['file'],16<<20)=={k:r[k]for k in('bytes','sha256')}for r in report['outputs']),'Output changed');check()
            report['source_rehashed_after']=True
        except BaseException as error:failure=failure or error;report['post_error_type']=type(error).__name__
        report.update(status='fail'if failure else'pass',elapsed_seconds=time.monotonic()-start)
        rt.write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('Paired native inference failed; retained own receipt')from None
    return report


def run(root,code,revision,temporal_cohort=None):
    import dexycb_identity_infer as frontend
    selected=profile(temporal_cohort);helpers=source_helpers(temporal_cohort)
    rt=scan.runtime(code);start=time.monotonic();deadline=start+BUDGET
    rt.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'
        and root==ROOT and Path(__file__).resolve()==code/HELPERS[0],'Actual immutable VM02 host required')
    source=rt.source(root,code,revision,ENTRY,helpers)
    _,protocol_pin=protocol(rt,code,temporal_cohort)if temporal_cohort else protocol(rt,code)
    evidence,pins,predictions,frozen=authenticate_scan(rt,root,code,temporal_cohort)if temporal_cohort else authenticate_scan(rt,root,code)
    model,assets=frontend.frontend_proof(code,live=True);remaining=deadline-time.monotonic()
    rt.require(remaining>0,'Inclusive source prehash exhausted budget')
    name='world-reward-'+selected['prefix']+revision[:12]
    rt.require(not rt.control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Owned container name occupied')
    control=root/'results'/(selected['prefix']+revision);rt.canonical(control);rt.require(not control.exists(),'Fresh owned inference namespace required')
    out=control/'predictions'
    proof=dict(source_binding=source,protocol_identity=protocol_pin,manifest=evidence['acquisition']['manifest'],
        acquisition=evidence['acquisition'],scan_pins=pins,predictions=str(predictions),frontend=frontend_json(rt,model),assets=assets,
        frozen={str(p):pin for p,pin in{**evidence['public'],**frozen}.items()if p!=code/selected['scan_pins']
                and p!=predictions.parent/'report.json'},remaining_seconds=remaining)
    if temporal_cohort:proof['temporal_cohort']=temporal_cohort
    raw_proof=(json.dumps(proof,sort_keys=True,allow_nan=False)+'\n').encode()
    control.mkdir(mode=0o700);out.mkdir(mode=0o700)
    rt.write(control/'native-proof.json',raw_proof);proof_pin=rt.identity(control/'native-proof.json')
    report=dict(stage='hand_mask_infer_host_seal',status='fail',
        producer_revision=revision,source_binding=source,protocol_identity=protocol_pin,scan_pins=pins,
        image_id=frontend.binding.IMAGE,private_values_read=False,quality_verified=False,owned_cleanup_verified=False,source_rehashed_after=False)
    if temporal_cohort:report.update(stage='hand_temporal_mask_infer_host_seal',temporal_cohort=temporal_cohort)
    lock=root/'jobs/.world-reward-h100.lock';state=rt.canonical(lock).lstat()
    rt.require(stat.S_ISREG(state.st_mode)and state.st_nlink==1,'Existing cooperative lock required')
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);os.dup2(fd,9)
    if fd!=9:os.close(fd)
    owned=False;failure=None
    def idle():rt.require(not rt.control(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(),'GPU compute active')
    def lock_check():rt.require((os.fstat(9).st_dev,os.fstat(9).st_ino)==(state.st_dev,state.st_ino)==(lock.lstat().st_dev,lock.lstat().st_ino),'Lock changed')
    try:
        lock_check();idle()
        # flock must inherit this original descriptor; control() deliberately does not.
        subprocess.run(['flock','--nonblock','9'],check=True,pass_fds=(9,),timeout=5,capture_output=True)
        lock_check();idle()
        mounts=native_mounts(code,revision,[predictions/n for n in pins['outputs']],temporal_cohort)if temporal_cohort else native_mounts(code,revision,[predictions/n for n in pins['outputs']])
        args=['docker','run','--rm','--name',name,'--cidfile',str(control/'.container.cid'),
            '--label','world_reward.mediapipe_cpu.owner='+revision,'--gpus','all','--network','none','--read-only',
            '--user','0:0','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4','--memory','16g',
            '--tmpfs','/tmp:rw,nosuid,size='+('2g'if temporal_cohort else'512m')]
        for p in mounts:args+=['--mount',f'type=bind,src={p},dst={p},readonly']
        args+=['--mount',f'type=bind,src={control}/native-proof.json,dst=/opt/hand-mask-proof.json,readonly',
            '--mount',f'type=bind,src={out},dst={out}','--entrypoint','/usr/bin/env',frontend.binding.IMAGE,'-i',
            'PATH=/opt/conda/bin:/usr/bin:/bin','HOME=/tmp','XDG_CACHE_HOME=/tmp','HF_HUB_OFFLINE=1',
            'TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',
            'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_IMAGE_ID='+frontend.binding.IMAGE,
            '/opt/conda/bin/python','-I','-B',str(code/HELPERS[0]),'--native',str(out),str(proof_pin['bytes']),proof_pin['sha256']]
        if temporal_cohort:args+=['--temporal-cohort',temporal_cohort]
        owned=True
        with(control/'native.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400);result=subprocess.run(args,stdout=log,stderr=log,timeout=max(.001,deadline-time.monotonic()))
        rt.require(result.returncode==0,'Native paired SAM2 stage failed')
        native=rt.pinned(out/'report.json',rt.identity(out/'report.json'),4<<20)
        rt.require(native['status']=='pass'and native['phase']=='complete'and native['source_binding']==source
            and native.get('source_rehashed_after')is True,'Complete native source-bound PASS required')
        if temporal_cohort:
            expected=[(r['sequence_lex_index'],r['frame_position'],b,
                f"sequence_{r['sequence_lex_index']:03d}_frame_{r['frame_position']:06d}_{b}.npz")
                for r in evidence['manifest']['images']for b in('A','B')]
            rt.require(native.get('stage')=='public_full_t_temporal_hand_sam2'and native.get('temporal_cohort')==temporal_cohort
                and native.get('a_frames')==native.get('b_frames')==len(evidence['manifest']['images'])
                and [(r['sequence_lex_index'],r['frame_index'],r['branch'],r['file'])for r in native['outputs']]==expected
                and {p.name for p in out.iterdir()}=={'report.json',*(r[-1]for r in expected)},
                'All paired temporal outputs required before independent host seal')
        report['native_report']=native;report['outputs']={p.name:rt.identity(p,16<<20)for p in out.iterdir()}
    except BaseException as error:failure=error;report['error_type']=type(error).__name__
    finally:
        try:
            if owned:rt.cleanup(control/'.container.cid',name,model['child_image'],revision);idle()
            report['owned_cleanup_verified']=True;lock_check()
        except BaseException as error:failure=failure or error;report['cleanup_error_type']=type(error).__name__
        finally:os.close(9)
        try:
            after=authenticate_scan(rt,root,code,temporal_cohort)if temporal_cohort else authenticate_scan(rt,root,code)
            rt.require(rt.source(root,code,revision,ENTRY,helpers)==source and after==(evidence,pins,predictions,frozen)
                and frontend.frontend_proof(code,live=True)==(model,assets),'Posthash original inputs/sources/assets changed')
            if 'outputs'in report:rt.require(all(rt.identity(out/n,16<<20)==p for n,p in report['outputs'].items()),'Outputs changed before host seal')
            rt.require(time.monotonic()<=deadline,'Inclusive all3 posthash budget exceeded');report['source_rehashed_after']=True
        except BaseException as error:failure=failure or error;report['post_error_type']=type(error).__name__
        report.update(status='fail'if failure else'pass',elapsed_seconds=time.monotonic()-start)
        rt.write(control/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('Owned hand-mask stage failed; inspect sealed receipt')from None
    return report


def main():
    args=sys.argv[1:];rt=scan.runtime(Path(os.environ['WR_CODE']))
    temporal_cohort=None
    if len(args)>=2 and args[-2:]==['--temporal-cohort','v3']:temporal_cohort='v3';args=args[:-2]
    if args[:1]==['--native']and len(args)==4:
        return run_native(rt,ROOT,Path(os.environ['WR_CODE']),Path(args[1]),Path('/opt/hand-mask-proof.json'),dict(bytes=int(args[2]),sha256=args[3]),temporal_cohort)
    if args:raise ValueError('No arbitrary cohort, prompt or threshold arguments')
    return run(ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'],temporal_cohort)


if __name__=='__main__':
    try:main()
    except Exception as error:print(json.dumps(dict(stage='hand_mask_infer',status='fail',error_type=type(error).__name__)));sys.exit(1)
