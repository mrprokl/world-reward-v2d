"""Manufactured full-T temporal serialization/authentication; no model/data jobs."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from test_mediapipe_hand_evaluate import REPO,gate,rt,scan,seal,source,native,host
from world_reward.hand_mask_evaluation import evaluate_temporal_hand_masks
from world_reward.hand_temporal_masks import TemporalMaskFrame


def readonly(value):
    a=np.array(value,copy=True);a.flags.writeable=False;return a


def temporal_files(folder,index):
    import hand_mask_infer as masks
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    boxes=np.array([[110.,120.,210.,210.],[250.,250.,260.,260.]],np.float64);usable=np.array([True,False])
    for t in range(2):
        for branch in('A','B'):
            n=0 if branch=='A'and t==0 else 2;raw=boxes[:n];supported=usable[:n]
            union=np.zeros((n,480,640),bool)
            if branch=='B':union[0,120:210,110:210]=True
            result=TemporalMaskFrame(branch,t,t,tuple(range(n)),readonly(raw),readonly(raw.astype(np.float32)),readonly(supported),
                readonly(union),readonly(supported),readonly(np.where(supported,.7,np.nan))if branch=='A'else None,
                'image_prompt'if branch=='A'else'anchor_prompt'if t==1 else'video_inferred')
            name=f'sequence_{index:03d}_frame_{t:06d}_{branch}.npz';pin=masks.save_temporal_frame(rt,np,folder/name,result)
            rows.append(dict(file=name,sequence_lex_index=index,frame_index=t,original_frame_id=t,branch=branch,
                rgb_file=f'{index}_{t}.jpg',rgb_sha256='e'*64,slots=n,usable_boxes=int(supported.sum()),supported=int(supported.sum()),evidence=result.evidence,**pin))
    summary=dict(sequence_lex_index=index,frames=2,anchor_position=1,anchor_frame_id=1,anchor_raw_boxes=boxes.tolist(),
        anchor_native_boxes=boxes.tolist(),anchor_box_usable=usable.tolist(),seeded_proposal_ids=[0],native_forward_frames=1,native_reverse_frames=2)
    return rows,summary


@pytest.fixture
def temporal_native(native,monkeypatch):
    import hand_mask_infer as masks
    root=native['root'];revision='e'*40;code=source(root,revision,gate.ENTRY,gate.source_helpers('v3'));folder=root/'temporal_predictions'
    rows=[];summaries=[]
    for i in(4,39,74):
        frames,summary=temporal_files(folder,i);rows.extend(frames);summaries.append(summary)
    seal(folder/'report.json',b'{}');frozen={p:rt.identity(p,16<<20)for p in folder.iterdir()}
    frozen.update({p:pin for p,pin in native['frozen'].items()if '/private/'in str(p)or p.name=='manifest.json'})
    out=gate.control_path(root,revision,'v3')/'diagnostics';out.mkdir(parents=True,mode=0o700)
    proof=json.loads(native['proof_path'].read_bytes());proof.update(source_binding=rt.source(root,code,revision,gate.ENTRY,gate.source_helpers('v3')),
        mask_cohort='v3',mask_infer_pins={'producer_revision':'c'*40},mask_rows=rows,sequence_summaries=summaries,
        predictions=str(folder),frozen={str(p):pin for p,pin in frozen.items()})
    native['proof_path'].chmod(0o600);pin=seal(native['proof_path'],json.dumps(proof).encode())
    original=Path.stat
    def owned(p,*a,**kw):
        s=original(p,*a,**kw);return types.SimpleNamespace(st_uid=0,st_mode=s.st_mode)if p==out else s
    monkeypatch.setattr(Path,'stat',owned);monkeypatch.setattr(masks,'__file__',str(code/'infra/hand_mask_infer.py'))
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]));monkeypatch.setenv('WR_CODE_REVISION',revision)
    return dict(native,code=code,out=out,proof_pin=pin,frozen=frozen,rows=rows,summaries=summaries)


def test_fullT_all_branches_and_anchor_provenance_before_seg_no_RGB_models_or_fake_interventions(temporal_native,monkeypatch):
    import hand_mask_infer as masks
    loads=[];load=masks.load_temporal_frame;reads=[];private=gate.private_field
    def decode(rt,np,path,t,branch):loads.append(path.name);return load(rt,np,path,t,branch)
    def labels(rt,np,path,name):
        assert len(set(loads))==12 and name=='seg';reads.append(name);return private(rt,np,path,name)
    monkeypatch.setattr(masks,'load_temporal_frame',decode);monkeypatch.setattr(gate,'private_field',labels)
    before={p:p.read_bytes()for p in temporal_native['frozen']}
    r=gate.run_native(rt,temporal_native['root'],temporal_native['code'],temporal_native['out'],temporal_native['proof_path'],temporal_native['proof_pin'],'v3')
    assert reads==['seg']*6 and len(loads)==36 and r['mask_forward_gate']=='pass'
    assert r['pooled']['original_frames']==6 and r['pooled']['positive_frames']==3 and r['pooled']['unlabelled_frames']==3
    assert r['pooled']['a']['positive_empty_union_frames']==3 and r['pooled']['b']['positive_empty_union_frames']==0
    assert r['pooled']['a']['mean_positive_dice']==0 and r['pooled']['b']['mean_positive_dice']==1
    assert 'positive_b_intervention_frames'not in r['pooled']and all(p.read_bytes()==v for p,v in before.items())
    for i in(4,39,74):
        diagnostics=json.loads((temporal_native['out']/f'sequence_{i:03d}.json').read_bytes())
        assert diagnostics['proposal_count_a']==[0,2]and diagnostics['proposal_count_b']==[2,2]
        assert diagnostics['anchor_position']==1 and diagnostics['seeded_proposal_ids']==[0]


@pytest.mark.parametrize('fault',['changed_bytes','missing','duplicate','support_count','original_id','anchor','summary_boxes','seed',
    'directions','anchor_bank','fixed_bank','evidence','dtype'])
def test_temporal_alloutputs_lossless_prepass_blocks_fault_before_any_private_value(temporal_native,monkeypatch,fault):
    proof=json.loads(temporal_native['proof_path'].read_bytes());rows=proof['mask_rows'];summary=proof['sequence_summaries'][0]
    row=rows[1];path=Path(proof['predictions'])/row['file']
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]=rows[-2]
    elif fault=='support_count':row['supported']=0
    elif fault=='original_id':row['original_frame_id']=99
    elif fault=='anchor':summary['anchor_position']=0
    elif fault=='summary_boxes':summary['anchor_raw_boxes'][0][0]+=1
    elif fault=='seed':summary['seeded_proposal_ids']=[1]
    elif fault=='directions':summary['native_reverse_frames']=1
    else:
        if fault=='anchor_bank':path=Path(proof['predictions'])/rows[2]['file']
        with np.load(path,allow_pickle=False)as archive:a={k:archive[k]for k in archive.files}
        if fault in('fixed_bank','anchor_bank'):a['raw_boxes'][0,0]+=1;a['native_boxes'][0,0]+=1
        elif fault=='evidence':a['evidence']=np.array('anchor_prompt',dtype='U32')
        elif fault=='dtype':a['masks']=a['masks'].astype(np.uint8)
        else:a['masks'][0,0,0]=True
        path.chmod(0o600)
        with path.open('wb')as f:np.savez_compressed(f,**a)
        path.chmod(0o400)
        if fault!='changed_bytes':proof['frozen'][str(path)]=rt.identity(path,16<<20)
    temporal_native['proof_path'].chmod(0o600);pin=seal(temporal_native['proof_path'],json.dumps(proof).encode())
    monkeypatch.setattr(gate,'private_field',lambda *_:pytest.fail('No private values before complete frozen temporal provenance'))
    with pytest.raises((ValueError,KeyError)):gate.run_native(rt,temporal_native['root'],temporal_native['code'],temporal_native['out'],temporal_native['proof_path'],pin,'v3')


def result(delta=(1.,),empty_a=1,empty_b=0,objects_a=0,objects_b=0):
    n=len(delta);pairs=[]
    for t in range(n or 1):
        count=int(n>0);raw=np.tile([0.,0.,1.,1.],(count,1));usable=np.ones(count,bool)
        a=TemporalMaskFrame('A',t,t,tuple(range(count)),readonly(raw),readonly(raw.astype(np.float32)),readonly(usable),
            readonly(np.zeros((count,1,1),bool)),readonly(usable),readonly(np.full(count,.7)),'image_prompt')
        b=replace(a,branch='B',masks=readonly(np.ones((count,1,1),bool)),raw_scores=None,
            evidence='no_anchor_abstention'if not n else'anchor_prompt'if t==0 else'video_inferred')
        pairs.append((a,b))
    r=evaluate_temporal_hand_masks(np.arange(n or 1,dtype=np.int64),pairs,[np.full((1,1),255 if n else 0,np.uint8)]*(n or 1))
    return replace(r,paired_dice_delta=tuple(delta)or(None,),mean_positive_paired_dice_delta=sum(delta)/n if n else None,
        a=replace(r.a,positive_empty_union_frames=empty_a,positive_object_label_pixels=objects_a),
        b=replace(r.b,positive_empty_union_frames=empty_b,positive_object_label_pixels=objects_b))


def test_temporal_preregistered_fourgate_conditions_nullable_equalframe_pool_no_drop():
    pooled,status=gate.mask_pool([result((.2,)),result((.1,)*3),result((0.,))],True)
    assert status=='pass'and pooled['mean_positive_paired_dice_delta']==pytest.approx(.1)
    assert gate.mask_pool([result((.9,)),result((-.1,)),result((.9,))],True)[1]=='fail'
    assert gate.mask_pool([result((0.,))]*3,True)[1]=='fail'
    assert gate.mask_pool([result(empty_b=1)]*3,True)[1]=='fail'
    assert gate.mask_pool([result(objects_b=1)]*3,True)[1]=='fail'
    assert gate.mask_pool([result(objects_a=1,objects_b=1)]*3,True)[1]=='pass'
    assert gate.mask_pool([result(),result(()),result()],True)[1]=='inconclusive'
    assert gate.mask_pool([result(())]*3,True)[0]['mean_positive_paired_dice_delta']is None


@pytest.mark.parametrize('fault',['','host_fail','source','missing','branch','counter','anchor','native_private','summary_type','core','changed','video_storage'])
def test_genuine_temporal_receipt_source_ALL_output_bytes_before_private_inventory(tmp_path,monkeypatch,fault):
    import hand_mask_infer as masks
    root=tmp_path/'root';revision='b'*40;old=source(root,revision,masks.ENTRY,masks.source_helpers('v3'))
    binding=rt.source(root,old,revision,masks.ENTRY,masks.source_helpers('v3'));control=root/'results'/('hand-temporal-mask-infer-'+revision);out=control/'predictions'
    rows=[];summaries=[];sequences=[];images=[]
    for i in(4,39,74):
        frames,s=temporal_files(out,i);rows.extend(frames);summaries.append(s)
        sequences.append(dict(subject='20200908-subject-05',sequence=f'20200908_{i:06d}',camera='836212060125',sequence_lex_index=i,frames=2))
        for t in range(2):images.append(dict(sequence_lex_index=i,frame_position=t,source_frame_id=t,file=f'{i}_{t}.jpg',sha256='e'*64))
    code=tmp_path/'code';protocol_pin=seal(code/masks.TEMPORAL_PROTOCOL,(REPO/masks.TEMPORAL_PROTOCOL).read_bytes(),0o444)
    manifest=dict(sequences=sequences,images=images);base=root/scan.profile('v3')['base'];manifest_pin=seal(base/'inputs/manifest.json',json.dumps(manifest).encode())
    evidence=dict(manifest=manifest,acquisition=dict(manifest=manifest_pin,report=seal(base/'report.json',b'{"retained_files":{}}')),image={'Id':'sha256:'+'c'*64});scan_pins={'producer_revision':'f'*40}
    native=dict(stage='public_full_t_temporal_hand_sam2',status='pass',phase='complete',producer_revision=revision,source_binding=binding,
        protocol_identity=protocol_pin,manifest_identity=manifest_pin,scan_pins=scan_pins,image_id=gate.SAM2_IMAGE,budget_seconds=900,
        source_rehashed_after=True,all_original_frames=True,network='none',device='cuda',sam2_native_postprocessing=True,
        private_values_read=False,quality_verified=False,identity_accepted=False,contacts_inferred=False,geometry_inferred=False,
        encoder_attempts=3,encoder_completed=3,a_attempts=3,a_completed=3,b_attempts=0,b_completed=0,outputs=rows,sequence_summaries=summaries,
        temporal_cohort='v3',a_frames=6,b_frames=6,video_model_loads=1,a_image_multimask_output=False,
        video_head_policy='unchanged_native_build_sam2_video_predictor_defaults_no_extra_overrides',
        video_storage=dict(offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False),
        video_logits_conversion='native_BF16_or_FP32_to_FP32_exact_values_for_numpy_no_threshold_change',
        video_states_attempted=6,video_states_completed=6,video_seed_attempts=6,video_seed_completed=6,video_frame_attempts=9,video_frame_completed=9)
    if fault=='branch':rows[0]['branch']='B'
    elif fault=='counter':native['video_frame_completed']=8
    elif fault=='anchor':summaries[0]['anchor_position']=0
    elif fault=='native_private':native['private_values_read']=True
    elif fault=='video_storage':native['video_storage']['offload_video_to_cpu']=False
    elif fault=='summary_type':summaries[0]['anchor_position']=True
    elif fault=='core':binding=json.loads(json.dumps(binding));binding['helpers'].pop('src/world_reward/hand_temporal_masks.py');native['source_binding']=binding
    seal(out/'report.json',json.dumps(native).encode());outputs={p.name:rt.identity(p,16<<20)for p in out.iterdir()}
    host=dict(stage='hand_temporal_mask_infer_host_seal',status='fail'if fault=='host_fail'else'pass',producer_revision=revision,source_binding=binding,
        protocol_identity=protocol_pin,scan_pins=scan_pins,image_id=gate.SAM2_IMAGE,private_values_read=False,quality_verified=False,
        owned_cleanup_verified=True,source_rehashed_after=True,elapsed_seconds=1.,temporal_cohort='v3',native_report=native,outputs=outputs)
    pins=dict(schema='world_reward.hand_temporal_mask_infer_pins.v1',producer_revision=revision,report=seal(control/'report.json',json.dumps(host).encode()))
    seal(code/gate.TEMPORAL_MASK_PINS,json.dumps(pins).encode(),0o444)
    if fault=='missing':(out/rows[0]['file']).unlink()
    elif fault=='changed':p=out/rows[0]['file'];p.chmod(0o600);p.write_bytes(b'changed');p.chmod(0o400)
    elif fault=='source':p=old/'infra/hand_mask_infer.py';p.chmod(0o600);p.write_bytes(b'changed');p.chmod(0o444)
    monkeypatch.setattr(masks,'__file__',str(code/'infra/hand_mask_infer.py'))
    def authentic(*args,**kwargs):assert kwargs==dict(temporal_cohort='v3');return evidence,scan_pins,root/'oldscan',{}
    monkeypatch.setattr(masks,'authenticate_scan',authentic);private=[]
    monkeypatch.setattr(gate,'private_inventory',lambda *args:(private.append(args[-1])or({},[])))
    if fault:
        with pytest.raises((ValueError,FileNotFoundError)):gate.authenticate_masks(scan,rt,root,code,'v3')
        assert not private
    else:
        e=gate.authenticate_masks(scan,rt,root,code,'v3');assert private==[scan.profile('v3')['base']]
        assert e['mask_infer_pins']==pins and e['sequence_summaries']==summaries and len(e['mask_rows'])==12


def test_temporal_host_keeps_exact_CPU_narrow_readonly_boundary_and_distinct_namespace(host,monkeypatch):
    root,_,_,evidence,cleanups=host;revision='e'*40;code=source(root,revision,gate.ENTRY,gate.source_helpers('v3'))
    e=dict(evidence,mask_infer_pins={'producer_revision':'c'*40},mask_rows=[],sequence_summaries=[])
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    def auth(*args,**kwargs):assert kwargs==dict(mask_cohort='v3');return e
    monkeypatch.setattr(gate,'authenticate_masks',auth);command=[]
    def fake(args,**kwargs):
        command.extend(args);out=gate.control_path(root,revision,'v3')/'diagnostics'
        report=dict(stage='mediapipe_hand_mask_private_evaluation',status='pass',phase='complete',producer_revision=revision,
            source_binding=rt.source(root,code,revision,gate.ENTRY,gate.source_helpers('v3')),private_fields_decoded=['seg'],private_values_read=True,
            native_graphs=0,prediction_values_modified=False,models_loaded=False,gpu_used=False,accuracy_threshold_calibrated=False,
            adoption=False,quality_claim=False,cohort='v3',mask_infer_pins=e['mask_infer_pins'],physical_identity_inferred=False,
            false_positive_truth_certified=False,challenge_adoption=False)
        seal(out/'report.json',json.dumps(report).encode())
        for i in(4,39,74):seal(out/f'sequence_{i:03d}.json',b'{}')
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(gate.subprocess,'run',fake);r=gate.run(root,code,revision,'v3')
    mounts=[command[i+1]for i,v in enumerate(command)if v=='--mount']
    assert r['status']=='pass'and r['cohort']=='v3'and cleanups==[True]and command[-2:]==['--mask-cohort','v3']
    assert 'mediapipe-temporal-mask-evaluate-'in str(gate.control_path(root,revision,'v3'))
    assert not any(word in '|'.join(mounts)for word in('weights','vendor','hand_landmarker.task','inputs','RGB','wheels'))
    assert command[command.index('--cap-add')+1]=='DAC_READ_SEARCH'and '--gpus'not in command
