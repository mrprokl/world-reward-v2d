"""Private CPU diagnostic only, after ALL original hand predictions are sealed."""
import importlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import zipfile

ROOT=Path('/srv/scenesmith/world-reward');ENTRY='run_mediapipe_hand_evaluate'
SCAN_PINS='configs/mediapipe_hand_scan_pins.json';BUDGET=300
MASK_PINS='configs/hand_mask_infer_pins.json'
TEMPORAL_MASK_PINS='configs/hand_temporal_mask_infer_pins.json'
SAM2_IMAGE='sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252'
HELPERS=('infra/mediapipe_hand_evaluate.py','infra/run_mediapipe_hand_evaluate.sh',
 'infra/mediapipe_hand_scan.py','infra/mediapipe_cpu_runtime_verify.py',
 'src/world_reward/hand_evaluation.py','src/world_reward/hand_observations.py',
 'configs/mediapipe_hand_scan_pins.json','configs/mediapipe_cpu_runtime_pins.json',
 'configs/dexycb_hand_acquire_pins.json','configs/dexycb_hand_protocol_v1.json')
# Publisher README 64551b001d360ad83bc383157a559ec248fb9100 L157/161,
# 40820 B SHA e19797f352bb5615b43b5a3ed4a6a193cee4eee2cd1cd1f5859615165081bb7c:
# seg uint8[H,W]; joint_2d float32[1,21,2]. Only axis0[0] is projected.


def source_helpers(mask_cohort=None):
    if mask_cohort is None:return HELPERS
    if mask_cohort not in('v2','v3'):raise ValueError('Only the preregistered mask cohorts v2/v3 are permitted')
    import mediapipe_hand_scan as scan
    temporal=mask_cohort=='v3'
    return tuple(dict.fromkeys((*HELPERS[:6],*scan.source_helpers(mask_cohort),'infra/hand_mask_infer.py',
        'src/world_reward/hand_mask_evaluation.py','src/world_reward/hand_mask_proposals.py',
        'src/world_reward/hand_temporal_masks.py','src/world_reward/automatic_candidate_bank.py',
        'configs/hand_temporal_mask_protocol_v1.json'if temporal else'configs/hand_mask_protocol_v1.json',
        'configs/mediapipe_hand_scan_v3_pins.json'if temporal else'configs/mediapipe_hand_scan_v2_pins.json',
        TEMPORAL_MASK_PINS if temporal else MASK_PINS)))


def control_path(root,revision,mask_cohort=None):
    prefix='mediapipe-temporal-mask-evaluate-'if mask_cohort=='v3'else'mediapipe-mask-evaluate-'if mask_cohort else'mediapipe-hand-evaluate-'
    return root/'results'/(prefix+revision)


def modules(code):
    sys.path[:0]=[str(code/'infra'),str(code/'src')]
    scan=importlib.import_module('mediapipe_hand_scan');rt=scan.runtime(code)
    rt.require(Path(scan.__file__).resolve()==code/'infra/mediapipe_hand_scan.py','Actual scan helper origin required')
    return scan,rt


def private_inventory(rt,root,manifest,retained,base='validation/dexycb_hand_v1'):
    files={};folders=[];base=root/base/'eval_private'
    for sequence in manifest['sequences']:
        relative=f"{sequence['subject']}/{sequence['sequence']}/{sequence['camera']}"
        folder=rt.canonical(base/relative);folders.append(folder)
        rt.require(stat.S_IMODE(folder.stat().st_mode)==0o700 and folder.stat().st_uid>0,'Original private directory owner/mode required')
        names={f'labels_{t:06d}.npz'for t in range(sequence['frames'])}
        rt.require({p.name for p in folder.iterdir()}==names,'Only selected complete original label files allowed')
        for name in sorted(names):
            path=folder/name;pin=retained.get(relative+'/'+name)
            rt.require(pin is not None and stat.S_IMODE(path.stat().st_mode)==0o400 and path.stat().st_uid==folder.stat().st_uid
                and rt.identity(path,16<<20)==pin,'Original opaque private label bytes/permissions differ')
            files[path]=pin
    expected={str(p.relative_to(base))for p in files}
    rt.require({n for n in retained if n.endswith('.npz')}==expected,'No private cohort substitution allowed')
    return files,folders


def authenticate(scan,rt,root,code):
    evidence=scan.authenticate(rt,root,code)  # stdlib hashes only; no models loaded.
    pins,pins_pin=scan.pins(rt,code,SCAN_PINS,'world_reward.mediapipe_hand_scan_pins.v1',('report',))
    revision=pins['producer_revision'];original=root/'jobs'/revision/scan.ENTRY/'code'
    source=rt.source(root,original,revision,scan.ENTRY,scan.HELPERS)
    control=root/'results'/('mediapipe-hand-scan-'+revision)
    host=rt.pinned(control/'report.json',pins['report'],4<<20)
    expected=dict(stage='mediapipe_hand_scan_host_seal',status='pass',producer_revision=revision,source_binding=source,
        acquisition_pins=evidence['acquisition'],runtime_pins=evidence['runtime'],source_rehashed_after=True,
        owned_cleanup_verified=True,gpu_used=False,private_values_read=False,accuracy_verified=False)
    rt.require(all(type(host.get(k))is type(v)and host[k]==v for k,v in expected.items()),'ALL original scans must have genuine sealed host PASS')
    rt.require(type(host.get('elapsed_seconds'))in(int,float)and 0<host['elapsed_seconds']<=600,'Original all-three scan budget differs')
    outputs=host.get('outputs',{});names={'report.json',*(f'sequence_{i:03d}'+s for i in(4,39,74)for s in('.npz','.npz.json'))}
    out=control/'predictions';rt.require(set(outputs)==names and {p.name for p in out.iterdir()}==names,'All seven prediction artifacts required')
    frozen={control/'report.json':pins['report'],code/SCAN_PINS:pins_pin}
    for name,pin in outputs.items():
        path=out/name;rt.require(rt.identity(path,32<<20)==pin,'Frozen native prediction changed');frozen[path]=pin
    native=rt.strict((out/'report.json').read_bytes())
    rt.require(native==host.get('native_report')and native.get('status')=='pass'and native.get('phase')=='complete'
        and native.get('producer_revision')==revision and native.get('source_binding')==source
        and native.get('native_calls')==len(evidence['manifest']['images'])and native.get('native_graphs')==1
        and all(native.get(k)is False for k in('gpu_used','private_values_read','accuracy_verified','availability_is_visibility',
            'handedness_is_detection_confidence','actor_identity_inferred')),'Original complete native scan report differs')
    declared=native.get('outputs',[])
    rt.require(len(declared)==3 and [(r['sequence_lex_index'],r['frames'],r['file'])for r in declared]==[
        (s['sequence_lex_index'],s['frames'],f"sequence_{s['sequence_lex_index']:03d}.npz")for s in evidence['manifest']['sequences']],
        'Native output original sequence/frame declaration differs')
    acq=rt.pinned(root/scan.BASE/'report.json',evidence['acquisition']['report'],4<<20)
    private,folders=private_inventory(rt,root,evidence['manifest'],acq['retained_files']);frozen.update(private)
    manifest_path=root/scan.BASE/'inputs/manifest.json';frozen[manifest_path]=evidence['acquisition']['manifest']
    return dict(evidence=evidence,scan_pins=pins,predictions=out,frozen=frozen,private_folders=folders,manifest_path=manifest_path)


def authenticate_masks(scan,rt,root,code,mask_cohort='v2'):
    import hand_mask_infer as masks
    rt.require(Path(masks.__file__).resolve()==code/'infra/hand_mask_infer.py','Actual model-free mask decoder origin required')
    rt.require(mask_cohort in('v2','v3'),'Fixed preregistered mask cohort required');temporal=mask_cohort=='v3'
    selected=masks.profile('v3'if temporal else None);pins_path=TEMPORAL_MASK_PINS if temporal else MASK_PINS
    evidence,scan_pins,_,frozen=masks.authenticate_scan(rt,root,code,temporal_cohort='v3')if temporal else masks.authenticate_scan(rt,root,code)
    protocol,protocol_pin=masks.protocol(rt,code,temporal_cohort='v3')if temporal else masks.protocol(rt,code)
    evaluation=protocol.get('evaluation',{})
    gate_ok=(evaluation.get('gate')=='pooled_B_minus_A_Dice_positive_each_clip_delta_nonnegative_positive_empty_union_count_strictly_lower_and_object_label_pixel_count_not_higher'
        and evaluation.get('no_positive_clip')=='inconclusive_stop_no_drop'and evaluation.get('contact_3D_identity_adoption')is False)if temporal else(
        evaluation.get('mask_forward_gate')=='pooled_positive_frame_B_minus_A_Dice_strictly_positive_and_each_clip_delta_nonnegative_and_each_clip_positive_B_intervention_present'
        and evaluation.get('no_positive_clip_policy')=='inconclusive_stop_no_drop_no_zero_substitution')
    rt.require(gate_ok
        and protocol.get('budgets_seconds',{}).get('private_mask_evaluation')==BUDGET,'Unchanged preregistered diagnostic gate required')
    pins,pin=scan.pins(rt,code,pins_path,'world_reward.hand_temporal_mask_infer_pins.v1'if temporal else'world_reward.hand_mask_infer_pins.v1',('report',))
    revision=pins['producer_revision'];old=root/'jobs'/revision/masks.ENTRY/'code'
    control=root/'results'/(selected['prefix']+revision);out=control/'predictions'
    host=rt.pinned(control/'report.json',pins['report'],4<<20)
    helpers=host.get('source_binding',{}).get('helpers',{})
    required={'infra/hand_mask_infer.py','infra/run_hand_mask_infer.sh','src/world_reward/hand_mask_proposals.py',
        'src/world_reward/hand_observations.py','src/world_reward/automatic_candidate_bank.py',selected['protocol'],selected['scan_pins']}
    if temporal:required.add('src/world_reward/hand_temporal_masks.py')
    rt.require(type(helpers)is dict and required<=set(helpers),'Original native producer/core source required')
    original=rt.source(root,old,revision,masks.ENTRY,tuple(helpers))
    expected=dict(stage='hand_temporal_mask_infer_host_seal'if temporal else'hand_mask_infer_host_seal',status='pass',producer_revision=revision,source_binding=original,
        protocol_identity=protocol_pin,scan_pins=scan_pins,image_id=SAM2_IMAGE,private_values_read=False,
        quality_verified=False,owned_cleanup_verified=True,source_rehashed_after=True)
    if temporal:expected['temporal_cohort']='v3'
    rt.require(all(type(host.get(k))is type(v)and host[k]==v for k,v in expected.items())
        and type(host.get('elapsed_seconds'))in(int,float)and 0<host['elapsed_seconds']<=masks.BUDGET,'Genuine frozen original mask host PASS required')
    native=host.get('native_report',{});rows=native.get('outputs',[]);images=evidence['manifest']['images']
    rt.require(type(rows)is list and len(rows)==len(images)*(2 if temporal else 1),'All original mask frames required')
    expected_frames=[(image,branch)for image in images for branch in(('A','B')if temporal else('',))]
    for row,(image,branch)in zip(rows,expected_frames):
        rt.require(all(row.get(k)==v for k,v in dict(sequence_lex_index=image['sequence_lex_index'],frame_index=image['frame_position'],
            file=f"sequence_{image['sequence_lex_index']:03d}_frame_{image['frame_position']:06d}"+(f'_{branch}'if temporal else'')+'.npz',rgb_file=image['file'],rgb_sha256=image['sha256']).items())
            and image['source_frame_id']==image['frame_position'],'No mask/RGB frame substitution or omission')
        keys=('slots','usable_boxes','supported')if temporal else('slots','usable_boxes','b_interventions','a_supported','b_supported')
        rt.require(all(type(row.get(k))is int and 0<=row[k]<=row['slots'] for k in keys)
            and row['slots']<=4,'Native per-frame slot/count declaration differs')
        if temporal:rt.require(row.get('branch')==branch and type(row.get('original_frame_id'))is int
            and row['original_frame_id']==image['source_frame_id']and row['supported']==row['usable_boxes'],'Temporal branch/support provenance differs')
    outputs=host.get('outputs',{});names={'report.json',*(r['file']for r in rows)}
    rt.require(set(outputs)==names and {p.name for p in out.iterdir()}==names,'Exactly all frozen native mask artifacts required')
    native_frozen={control/'report.json':pins['report'],code/pins_path:pin,code/selected['protocol']:protocol_pin}
    for name,identity in outputs.items():
        path=out/name;rt.require(rt.identity(path,16<<20)==identity,'Frozen native masks changed');native_frozen[path]=identity
    active=sum(r['usable_boxes']>0 for r in rows if not temporal or r['branch']=='A');intervened=0 if temporal else sum(r['b_interventions']>0 for r in rows)
    expected=dict(stage='public_full_t_temporal_hand_sam2'if temporal else'public_full_t_paired_hand_sam2',status='pass',phase='complete',producer_revision=revision,
        source_binding=original,protocol_identity=protocol_pin,manifest_identity=evidence['acquisition']['manifest'],scan_pins=scan_pins,
        image_id=SAM2_IMAGE,budget_seconds=masks.BUDGET,source_rehashed_after=True,all_original_frames=True,network='none',device='cuda',
        sam2_native_postprocessing=True,private_values_read=False,quality_verified=False,identity_accepted=False,contacts_inferred=False,
        geometry_inferred=False,encoder_attempts=active,encoder_completed=active,a_attempts=active,a_completed=active,b_attempts=intervened,b_completed=intervened)
    summaries=[]
    if temporal:
        summaries=temporal_summaries(rt,native,rows,evidence['manifest'])
        states=2*sum(s['anchor_position']is not None for s in summaries);seeds=2*sum(len(s['seeded_proposal_ids'])for s in summaries)
        frames=sum(s['native_forward_frames']+s['native_reverse_frames']for s in summaries)
        expected.update(temporal_cohort='v3',a_frames=len(images),b_frames=len(images),video_model_loads=int(states>0),
            a_image_multimask_output=False,video_head_policy='unchanged_native_build_sam2_video_predictor_defaults_no_extra_overrides',
            video_storage=dict(offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False),
            video_logits_conversion='native_BF16_or_FP32_to_FP32_exact_values_for_numpy_no_threshold_change',
            video_states_attempted=states,video_states_completed=states,video_seed_attempts=seeds,video_seed_completed=seeds,
            video_frame_attempts=frames,video_frame_completed=frames)
    rt.require(all(type(native.get(k))is type(v)and native[k]==v for k,v in expected.items())and rt.strict((out/'report.json').read_bytes())==native
        and all(outputs[r['file']]=={k:r[k]for k in('bytes','sha256')}for r in rows),'Complete original paired native mask report required')
    frozen.update(native_frozen);base=scan.profile(mask_cohort)['base']
    acquisition=rt.pinned(root/base/'report.json',evidence['acquisition']['report'],4<<20)
    private,folders=private_inventory(rt,root,evidence['manifest'],acquisition['retained_files'],base)
    manifest_path=root/base/'inputs/manifest.json';native_frozen.update(private);native_frozen[manifest_path]=evidence['acquisition']['manifest']
    frozen.update(native_frozen)
    result=dict(evidence=evidence,scan_pins=scan_pins,mask_infer_pins=pins,predictions=out,frozen=frozen,
        native_frozen=native_frozen,private_folders=folders,manifest_path=manifest_path,mask_rows=rows)
    if temporal:result['sequence_summaries']=summaries
    return result


def temporal_summaries(rt,native,rows,manifest):
    summaries=native.get('sequence_summaries');rt.require(type(summaries)is list and len(summaries)==len(manifest['sequences']),'All original temporal summaries required')
    for summary,sequence in zip(summaries,manifest['sequences']):
        index=sequence['sequence_lex_index'];frames=sequence['frames'];a=[r for r in rows if r['sequence_lex_index']==index and r['branch']=='A']
        b=[r for r in rows if r['sequence_lex_index']==index and r['branch']=='B'];anchor=next((r['frame_index']for r in a if r['usable_boxes']),None)
        rt.require(type(summary.get('sequence_lex_index'))is int and summary['sequence_lex_index']==index
            and type(summary.get('frames'))is int and summary['frames']==frames
            and type(summary.get('anchor_position'))is type(anchor)and summary['anchor_position']==anchor
            and type(summary.get('anchor_frame_id'))is type(anchor)and summary['anchor_frame_id']==anchor,'First automatic anchor/original frame provenance differs')
        usable=summary.get('anchor_box_usable');raw=summary.get('anchor_raw_boxes');boxes=summary.get('anchor_native_boxes');seeded=summary.get('seeded_proposal_ids')
        n=0 if anchor is None else a[anchor]['slots']
        rt.require(type(usable)is list and len(usable)==n and all(type(v)is bool for v in usable)
            and type(raw)is list and type(boxes)is list and len(raw)==len(boxes)==n
            and all(type(row)is list and len(row)==4 and all(v is None or type(v)in(int,float)and math.isfinite(v)for v in row)for row in raw+boxes)
            and type(seeded)is list and all(type(i)is int for i in seeded)and seeded==[i for i,v in enumerate(usable)if v],
            'All anchor slots/boxes/seed provenance required')
        forward=0 if anchor is None else frames-anchor;reverse=anchor+1 if anchor else 0
        rt.require(type(summary.get('native_forward_frames'))is int and summary['native_forward_frames']==forward
            and type(summary.get('native_reverse_frames'))is int and summary['native_reverse_frames']==reverse
            and all(r['slots']==n and r['usable_boxes']==len(seeded)
                and r.get('evidence')==('no_anchor_abstention'if anchor is None else'anchor_prompt'if r['frame_index']==anchor else'video_inferred')for r in b),
            'Complete independent native directional ownership/counts required')
    return summaries


def reload_observations(rt,np,path,categories_path,frames,refs):
    from world_reward.hand_observations import HandObservations,normalized_hand_instances
    with np.load(path,allow_pickle=False)as archive:
        keys={'frame_index','frame_offsets','capacity_saturation','normalized_xyz','native_world_xyz','pixel_xy','image_z',
              'xy_supported','image_z_supported','world_supported','handedness_scores'}
        rt.require(set(archive.files)==keys,'Exact native numeric artifacts required');a={k:archive[k]for k in keys}
    rt.require(all(not v.dtype.hasobject for v in a.values())and a['frame_index'].dtype==np.int64
        and np.array_equal(a['frame_index'],np.arange(frames,dtype=np.int64))and a['frame_offsets'].dtype==np.int64
        and a['frame_offsets'].shape==(frames+1,)and a['frame_offsets'][0]==0,'Native full-T indices/offsets differ')
    counts=np.diff(a['frame_offsets']);n=int(a['frame_offsets'][-1])
    rt.require(np.all((counts>=0)&(counts<=4))and n>=0,'Native ragged capacity differs')
    shapes={'normalized_xyz':(n,21,3),'native_world_xyz':(n,21,3),'pixel_xy':(n,21,2),'image_z':(n,21,1),
            'xy_supported':(n,21),'image_z_supported':(n,21),'world_supported':(n,21),'handedness_scores':(n,),
            'capacity_saturation':(frames,)}
    for name,shape in shapes.items():
        rt.require(a[name].shape==shape and a[name].dtype==(np.bool_ if name.endswith('supported')or name=='capacity_saturation'else np.float64),
                   'Native array shape/dtype changed')
    rt.require(np.array_equal(a['capacity_saturation'],counts==4),'Native capacity flags differ')
    category=rt.strict(categories_path.read_bytes());rt.require(set(category)=={'native_categories'}and len(category['native_categories'])==frames,'Native category timeline differs')
    bank=[]
    for t in range(frames):
        lo,hi=map(int,a['frame_offsets'][t:t+2]);rows=category['native_categories'][t]
        rt.require(type(rows)is list and len(rows)==hi-lo and all(type(r)is dict and set(r)=={'category_name','display_name','index'}
            and r['category_name']in('Left','Right')and type(r['index'])is int and r['index']in(0,1)
            and type(r['display_name'])is str and len(r['display_name'])<=256 for r in rows),'Native categories invalid')
        f=normalized_hand_instances(a['normalized_xyz'][lo:hi],(480,640),native_world_xyz=a['native_world_xyz'][lo:hi],
            native_handedness_labels=tuple(r['category_name']for r in rows),native_handedness_scores=a['handedness_scores'][lo:hi])
        for name,value in(('pixel_xy',f.original_xy.values),('image_z',f.image_z.values),('xy_supported',f.original_xy.supported),
                           ('image_z_supported',f.image_z.supported),('world_supported',f.hand_centred_world_xyz.supported)):
            rt.require(value.tobytes()==a[name][lo:hi].tobytes(),'Lossless native coordinate/support restoration failed')
        bank.append(f)
    return HandObservations(a['frame_index'],(480,640),tuple(bank),'MediaPipe IMAGE CPU frozen native outputs',refs)


def summary(result):
    from dataclasses import asdict
    return dict(counts=asdict(result.counts),joint_weighted_epe_pixels=result.joint_weighted_epe_pixels,
                frame_mean_epe_pixels=result.frame_mean_epe_pixels,diagnostic_status=result.diagnostic_status)


def private_field(rt,np,path,name):
    """Bound only the two permitted NPY fields before allocating their arrays."""
    rt.require(name in('seg','joint_2d'),'Only permitted private diagnostic fields may be decoded')
    shape,dtype=((480,640),np.dtype('uint8'))if name=='seg'else((1,21,2),np.dtype('float32'))
    with zipfile.ZipFile(path)as archive:
        member=name+'.npy';rt.require(archive.namelist().count(member)==1,'Unique permitted private NPY field required')
        info=archive.getinfo(member);rt.require(info.file_size<=dtype.itemsize*int(np.prod(shape))+4096,'Private NPY expanded-byte cap')
        with archive.open(member)as stream:
            version=np.lib.format.read_magic(stream)
            rt.require(version in((1,0),(2,0)),'Known non-pickle NPY header required')
            header=np.lib.format.read_array_header_1_0 if version==(1,0)else np.lib.format.read_array_header_2_0
            actual,fortran,kind=header(stream);rt.require(actual==shape and kind==dtype and not kind.hasobject,'Publisher private NPY shape/dtype required')
    with np.load(path,allow_pickle=False)as archive:return archive[name]


def mask_summary(result,temporal=False):
    keys=('positive_frames','mean_positive_paired_dice_delta')if temporal else('positive_frames','no_proposal_frames','positive_no_proposal_frames','b_intervention_frames','positive_b_intervention_frames',
        'mean_positive_paired_dice_delta')
    arm_keys=('mean_positive_iou','mean_positive_dice','positive_empty_union_frames','total_predicted_pixels','total_object_label_pixels',
        'total_background_label_pixels','positive_predicted_pixels','positive_object_label_pixels','positive_background_label_pixels')
    value=dict(original_frames=len(result.frame_index),unlabelled_frames=int(result.unlabelled.sum()),
        **{k:getattr(result,k)for k in keys},**{arm:{k:getattr(getattr(result,arm),k)for k in arm_keys}for arm in('a','b')})
    if temporal:
        value.update(anchor_position=result.anchor_position,seeded_proposal_ids=list(result.seeded_proposal_ids))
        for arm in('a','b'):
            counts=getattr(result,'proposal_count_'+arm)
            value[arm].update(no_proposal_frames=int((counts==0).sum()),positive_no_proposal_frames=int(((counts==0)&result.annotated_positive).sum()))
    return value


def mask_pool(results,temporal=False):
    def mean(values):return math.fsum(values)/len(values)if values else None
    keys=('original_frames','positive_frames','unlabelled_frames')if temporal else('original_frames','positive_frames','unlabelled_frames','no_proposal_frames',
        'positive_no_proposal_frames','b_intervention_frames','positive_b_intervention_frames')
    pooled={k:sum(mask_summary(r,temporal)[k]for r in results)for k in keys}
    for arm in('a','b'):
        pooled[arm]={k:sum(getattr(getattr(r,arm),k)for r in results)for k in('positive_empty_union_frames','total_predicted_pixels',
            'total_object_label_pixels','total_background_label_pixels','positive_predicted_pixels','positive_object_label_pixels','positive_background_label_pixels')}
        for metric in('iou','dice'):pooled[arm]['mean_positive_'+metric]=mean([v for r in results for v in getattr(getattr(r,arm),metric)if v is not None])
        if temporal:
            for k in('no_proposal_frames','positive_no_proposal_frames'):pooled[arm][k]=sum(mask_summary(r,True)[arm][k]for r in results)
    pooled['mean_positive_paired_dice_delta']=mean([v for r in results for v in r.paired_dice_delta if v is not None])
    improved=(pooled['mean_positive_paired_dice_delta']is not None and pooled['mean_positive_paired_dice_delta']>0
        and all(r.mean_positive_paired_dice_delta is not None and r.mean_positive_paired_dice_delta>=0 for r in results))
    if temporal:
        improved=improved and pooled['b']['positive_empty_union_frames']<pooled['a']['positive_empty_union_frames']\
            and pooled['b']['positive_object_label_pixels']<=pooled['a']['positive_object_label_pixels']
    else:improved=improved and all(r.positive_b_intervention_frames>0 for r in results)
    gate='inconclusive'if any(r.positive_frames==0 for r in results)else'pass'if improved else'fail'
    return pooled,gate


def native_masks(rt,np,code,out,proof,manifest,frozen,check):
    import hand_mask_infer as masks
    from dataclasses import asdict
    from world_reward.hand_mask_evaluation import evaluate_hand_masks,evaluate_temporal_hand_masks,validate_temporal_hand_frames
    rt.require(Path(masks.__file__).resolve()==code/'infra/hand_mask_infer.py','Actual pure mask decoder required')
    temporal=proof.get('mask_cohort')=='v3';rows=proof['mask_rows']
    by_frame={(r['sequence_lex_index'],r['frame_index'],r.get('branch','')):r for r in rows}
    rt.require(len(by_frame)==len(rows)==sum(s['frames']for s in manifest['sequences'])*(2 if temporal else 1),'Unique complete original mask timeline required')
    if temporal:temporal_summaries(rt,dict(sequence_summaries=proof['sequence_summaries']),rows,manifest)
    def proposals(s):
        for t in range(s['frames']):
            pair=[]
            for branch in(('A','B')if temporal else('',)):
                check();row=by_frame[(s['sequence_lex_index'],t,branch)];path=Path(proof['predictions'])/row['file']
                rt.require(row['file']==f"sequence_{s['sequence_lex_index']:03d}_frame_{t:06d}"+(f'_{branch}'if temporal else'')+'.npz'
                    and rt.identity(path,16<<20)==frozen[path],'Original mask frame bytes changed')
                p=masks.load_temporal_frame(rt,np,path,t,branch)if temporal else masks.load_frame(rt,np,path,t)
                counts=dict(slots=len(p.proposal_ids),usable_boxes=int(p.box_usable.sum()),supported=int(p.supported.sum()),
                    original_frame_id=p.original_frame_id,branch=p.branch,evidence=p.evidence)if temporal else dict(
                    slots=len(p.local_ids),usable_boxes=int(p.box_usable.sum()),b_interventions=int((~p.b_reuses_a).sum()),
                    a_supported=int(p.mask_supported_a.sum()),b_supported=int(p.mask_supported_b.sum()))
                rt.require(all(type(row.get(k))is type(v)and row[k]==v for k,v in counts.items()),'Native mask support/provenance counts changed');pair.append(p)
            yield tuple(pair)if temporal else pair[0]
    # Restore EVERY frame's immutable schema/reuse/support before the first seg value.
    for s in manifest['sequences']:
        if temporal:
            metadata=validate_temporal_hand_frames(np.arange(s['frames'],dtype=np.int64),proposals(s))
            summaries=proof['sequence_summaries'];summary=next(v for v in summaries if v['sequence_lex_index']==s['sequence_lex_index'])
            rt.require(summary['anchor_position']==metadata['anchor_position']and summary['seeded_proposal_ids']==list(metadata['seeded_proposal_ids']),
                'Serialized first-anchor/seed provenance changed before private values')
            # Compare the complete original bank to its immutable JSON summary, no GT.
            for _,b in proposals(s):
                def finite(value):return [[float(v)if np.isfinite(v)else None for v in row]for row in value]
                rt.require(summary['anchor_raw_boxes']==finite(b.raw_boxes)and summary['anchor_native_boxes']==finite(b.native_boxes)
                    and summary['anchor_box_usable']==b.box_usable.tolist(),'Serialized anchor bank summary differs')
        else:
            for _ in proposals(s):pass
    results=[]
    for s in manifest['sequences']:
        folder=Path(proof['private_folders'][str(s['sequence_lex_index'])])
        def segments():
            for t in range(s['frames']):
                check();path=folder/f'labels_{t:06d}.npz';rt.require(rt.identity(path,16<<20)==frozen[path],'Private segmentation changed')
                yield private_field(rt,np,path,'seg')
        evaluate=evaluate_temporal_hand_masks if temporal else evaluate_hand_masks
        r=evaluate(np.arange(s['frames'],dtype=np.int64),proposals(s),segments());results.append(r)
        def numeric(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if type(value)is dict:return{k:numeric(v)for k,v in value.items()}
            return value
        rt.write(out/f"sequence_{s['sequence_lex_index']:03d}.json",(json.dumps(numeric(asdict(r)),allow_nan=False)+'\n').encode());check()
    pooled,gate=mask_pool(results,temporal)
    return dict(per_clip=[dict(clip=s['sequence'],**mask_summary(r,temporal))for s,r in zip(manifest['sequences'],results)],pooled=pooled,
        mask_forward_gate=gate,mask_infer_pins=proof['mask_infer_pins'],private_fields_decoded=['seg'],physical_identity_inferred=False,
        false_positive_truth_certified=False,challenge_adoption=False)


def run_native(rt,root,code,out,proof_path,proof_pin,mask_cohort=None):
    revision=os.environ['WR_CODE_REVISION']
    helpers=source_helpers(mask_cohort)
    rt.require(root==ROOT and code==root/'jobs'/revision/ENTRY/'code'
        and out==control_path(root,revision,mask_cohort)/'diagnostics'
        and Path(__file__).resolve()==code/HELPERS[0],'Exact native evaluator source/output namespace required')
    rt.require(sys.platform=='linux'and os.geteuid()==0 and Path(sys.prefix)==rt.VENV and sys.version_info[:2]==(3,11)
        and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'},'Qualified offline CP311 CPU required')
    caps=int(next(line.split()[1]for line in Path('/proc/self/status').read_text().splitlines()if line.startswith('CapEff:')),16)
    rt.require(caps==4,'Only explicitly necessary DAC_READ_SEARCH capability permitted')
    proof=rt.pinned(proof_path,proof_pin,4<<20);started=time.monotonic();deadline=started+proof['remaining_seconds']
    def check():rt.require(time.monotonic()<=deadline,'Inclusive private evaluation budget exceeded')
    rt.require(rt.source(root,code,os.environ['WR_CODE_REVISION'],ENTRY,helpers)==proof['source_binding'],'Native original current source differs')
    rt.require(out.is_dir()and stat.S_IMODE(out.stat().st_mode)==0o700 and out.stat().st_uid==0 and not tuple(out.iterdir()),'Fresh owned private diagnostics required')
    frozen={Path(p):pin for p,pin in proof['frozen'].items()}
    rt.require(all(rt.identity(p,32<<20)==pin for p,pin in frozen.items()),'ALL frozen predictions/labels before any value read required');check()
    import numpy as np
    from world_reward.hand_evaluation import evaluate_hand_clip,pool_hand_evaluations
    rt.require(Path(np.__file__).is_relative_to(rt.VENV)and not({'mediapipe','torch'}&sys.modules.keys())and sys.prefix!=sys.base_prefix
        and 'include-system-site-packages = false'in(rt.VENV/'pyvenv.cfg').read_text().lower(),'No model import or base-site numpy permitted')
    manifest=rt.pinned(Path(proof['manifest_path']),proof['manifest_pin'],16<<20)
    if mask_cohort:
        rt.require(proof.get('mask_cohort')==mask_cohort,'Native mask profile/proof differs')
        report=dict(stage='mediapipe_hand_mask_private_evaluation',status='pass',phase='complete',producer_revision=revision,
            source_binding=proof['source_binding'],cohort=mask_cohort,private_values_read=True,prediction_values_modified=False,
            models_loaded=False,native_graphs=0,gpu_used=False,accuracy_threshold_calibrated=False,adoption=False,quality_claim=False,
            **native_masks(rt,np,code,out,proof,manifest,frozen,check))
        rt.require(not({'mediapipe','torch'}&sys.modules.keys())and all(rt.identity(p,32<<20)==pin for p,pin in frozen.items())
            and rt.identity(proof_path,4<<20)==proof_pin and rt.source(root,code,revision,ENTRY,helpers)==proof['source_binding'],
            'Model import or original source/mask/private bytes changed')
        check();rt.write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());check();return report
    bank=[]
    for s in manifest['sequences']:
        base=Path(proof['predictions'])/f"sequence_{s['sequence_lex_index']:03d}.npz"
        bank.append(reload_observations(rt,np,base,Path(str(base)+'.json'),s['frames'],{'scan':proof['scan_pins']}));check()
    results=[]
    for s,obs in zip(manifest['sequences'],bank):
        folder=Path(proof['private_folders'][str(s['sequence_lex_index'])]);joints=[]
        for t in range(s['frames']):
            check();path=folder/f'labels_{t:06d}.npz';rt.require(rt.identity(path,16<<20)==frozen[path],'Private label bytes changed')
            # Only these two NPY members are decoded. No joint_3d/pose/MANO/K.
            xy=private_field(rt,np,path,'joint_2d')
            rt.require(xy.dtype==np.float32 and xy.shape==(1,21,2),'Publisher joint_2d native shape/dtype required')
            joints.append(xy[0])
        def segments():
            for t in range(s['frames']):
                check();path=folder/f'labels_{t:06d}.npz';rt.require(rt.identity(path,16<<20)==frozen[path],'Private segmentation bytes changed')
                seg=private_field(rt,np,path,'seg')
                rt.require(seg.dtype==np.uint8 and seg.shape==(480,640),'Publisher seg native shape/dtype required');yield seg
        r=evaluate_hand_clip(s['sequence'],obs,obs.frame_index,segments(),np.stack(joints));results.append(r)
        diagnostics=dict(frame_index=r.frame_index.tolist(),positive=r.annotated_positive.tolist(),unlabelled=r.unlabelled.tolist(),
          unique=r.unique_association.tolist(),missed=r.missed.tolist(),ambiguous=r.ambiguous.tolist(),matched_prediction=r.matched_prediction.tolist(),
          prediction_count=r.prediction_count.tolist(),valid_gt_joint_count=r.valid_gt_joints.sum(axis=1).tolist(),
          frame_epe_pixels=r.frame_epe_pixels,joint_error_pixels=r.joint_error_pixels)
        rt.write(out/f"sequence_{s['sequence_lex_index']:03d}.json",(json.dumps(diagnostics,allow_nan=False)+'\n').encode());check()
    pooled=pool_hand_evaluations(tuple(results));report=dict(stage='mediapipe_hand_private_evaluation',status='pass',phase='complete',producer_revision=revision,
      source_binding=proof['source_binding'],per_clip=[dict(clip=r.clip,**summary(r))for r in results],pooled=summary(pooled),
      verified_joint_indices=[0,*range(5,21)],private_fields_decoded=['seg','joint_2d'],private_values_read=True,
      prediction_values_modified=False,models_loaded=False,native_graphs=0,gpu_used=False,accuracy_threshold_calibrated=False,
      adoption=False,quality_claim=False)
    rt.require(all(rt.identity(p,32<<20)==pin for p,pin in frozen.items())and rt.identity(proof_path,4<<20)==proof_pin
        and rt.source(root,code,os.environ['WR_CODE_REVISION'],ENTRY,HELPERS)==proof['source_binding'],'Source/predictions/private bytes changed after evaluation')
    check();rt.write(out/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());check();return report


def run(root,code,revision,mask_cohort=None):
    scan,rt=modules(code);rt.require(root==ROOT and sys.platform=='linux'and os.geteuid()==0
        and os.uname().nodename=='world-reward-ncc-h100-02'and Path(__file__).resolve()==code/HELPERS[0],'Actual immutable VM02 adapter required')
    helpers=source_helpers(mask_cohort)
    def auth(scan,rt,root,code):return authenticate_masks(scan,rt,root,code,mask_cohort='v3')if mask_cohort=='v3'else authenticate_masks(scan,rt,root,code)if mask_cohort else authenticate(scan,rt,root,code)
    started=time.monotonic();deadline=started+BUDGET;source=rt.source(root,code,revision,ENTRY,helpers)
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact readonly snapshot parent inventory required')
    evidence=auth(scan,rt,root,code);rt.require(time.monotonic()<deadline,'Inclusive source/ALL prediction prehash deadline exceeded')
    control=control_path(root,revision,mask_cohort);rt.canonical(control);rt.require(not control.exists(),'Fresh evaluator namespace required')
    control.mkdir(mode=0o700);out=control/'diagnostics';out.mkdir(mode=0o700)
    private={str(s['sequence_lex_index']):str(folder)for s,folder in zip(evidence['evidence']['manifest']['sequences'],evidence['private_folders'])}
    proof=dict(source_binding=source,remaining_seconds=deadline-time.monotonic(),scan_pins=evidence['scan_pins'],
        predictions=str(evidence['predictions']),private_folders=private,manifest_path=str(evidence['manifest_path']),
        manifest_pin=evidence['evidence']['acquisition']['manifest'],frozen={str(p):pin for p,pin in evidence.get('native_frozen',evidence['frozen']).items()})
    if mask_cohort:proof.update(mask_cohort=mask_cohort,mask_infer_pins=evidence['mask_infer_pins'],mask_rows=evidence['mask_rows'])
    if mask_cohort=='v3':proof['sequence_summaries']=evidence['sequence_summaries']
    rt.write(control/'proof.json',(json.dumps(proof,sort_keys=True)+'\n').encode());proof_pin=rt.identity(control/'proof.json',4<<20)
    image=evidence['evidence']['image'];name=('world-reward-mediapipe-mask-evaluate-'if mask_cohort else'world-reward-mediapipe-hand-evaluate-')+revision[:12];owned=False;failure=None;log_inode=None
    report=dict(stage='mediapipe_mask_evaluation_host_seal'if mask_cohort else'mediapipe_hand_evaluation_host_seal',status='fail',producer_revision=revision,source_binding=source,
        scan_pins=evidence['scan_pins'],budget_seconds=BUDGET,budget_scope='source_all_prediction_private_hash_evaluate_serialize_posthash',
        private_values_read=None,private_value_reads_certified=False,models_loaded=False,gpu_used=False,adoption=False,quality_claim=False)
    if mask_cohort:report.update(cohort=mask_cohort,mask_infer_pins=evidence['mask_infer_pins'])
    def interrupted(*_):raise TimeoutError('Private evaluation interrupted; owned cleanup only')
    handlers={s:signal.signal(s,interrupted)for s in(signal.SIGTERM,signal.SIGINT)}
    try:
        rt.require(not rt.control(['docker','ps','-aq','--filter','name=^/'+name+'$']).strip(),'Evaluator container namespace occupied')
        command=['docker','run','--rm','--name',name,'--cidfile',str(control/'.container.cid'),'--label','world_reward.mediapipe_cpu.owner='+revision,
          '--network','none','--read-only','--user','0:0','--cap-drop','ALL','--cap-add','DAC_READ_SEARCH','--security-opt','no-new-privileges',
          '--cpus','4','--memory','8g','--tmpfs','/tmp:rw,nosuid,size=64m']
        mounts=[(code.parent,code.parent),(control/'proof.json',Path('/opt/hand-evaluation-proof.json')),
                (evidence['manifest_path'],evidence['manifest_path']),*[(p,p)for p in evidence['private_folders']],
                *[(p,p)for p in evidence['frozen']if p.parent==evidence['predictions']],
                (evidence['predictions'].parent/'report.json',evidence['predictions'].parent/'report.json')]
        for src,dst in mounts:command+=['--mount',f'type=bind,src={src},dst={dst},readonly']
        command+=['--mount',f'type=bind,src={out},dst={out}','--entrypoint','/usr/bin/env',image['Id'],'-i',
          f'PATH={rt.VENV}/bin:/usr/bin:/bin','HOME=/nonexistent','CUDA_VISIBLE_DEVICES=-1','JAX_PLATFORMS=cpu',
          'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',f'WR_CODE_REVISION={revision}',
          f'{rt.VENV}/bin/python','-I','-B',str(code/HELPERS[0]),'--native',str(root),str(code),str(out),str(proof_pin['bytes']),proof_pin['sha256']]
        if mask_cohort:command+=['--mask-cohort',mask_cohort]
        owned=True;remaining=deadline-time.monotonic();rt.require(remaining>0,'No private inference budget remains')
        with (control/'.native-output').open('xb')as log:
            os.fchmod(log.fileno(),0o400);s=os.fstat(log.fileno());log_inode=(s.st_dev,s.st_ino)
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining,check=False)
        rt.require(result.returncode==0,'Private native evaluator failed');(control/'.native-output').unlink()
        native=rt.strict((out/'report.json').read_bytes())
        expected=dict(stage='mediapipe_hand_mask_private_evaluation'if mask_cohort else'mediapipe_hand_private_evaluation',status='pass',phase='complete',producer_revision=revision,
            source_binding=source,private_fields_decoded=['seg']if mask_cohort else['seg','joint_2d'],private_values_read=True,native_graphs=0,
            prediction_values_modified=False,models_loaded=False,gpu_used=False,accuracy_threshold_calibrated=False,adoption=False,quality_claim=False)
        if mask_cohort:expected.update(cohort=mask_cohort,mask_infer_pins=evidence['mask_infer_pins'],physical_identity_inferred=False,
            false_positive_truth_certified=False,challenge_adoption=False)
        rt.require(all(type(native.get(k))is type(v)and native[k]==v for k,v in expected.items()),'Private evaluation receipt differs')
        report['evaluation']=native;report['private_values_read']=True;report['private_value_reads_certified']=True
        rt.require({p.name for p in out.iterdir()}=={'report.json',*(f'sequence_{i:03d}.json'for i in(4,39,74))},'Exact private diagnostic outputs required')
        report['diagnostics']={p.name:rt.identity(p,4<<20)for p in out.iterdir()}
    except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
    finally:
        for s,handler in handlers.items():signal.signal(s,handler)
        try:
            if owned:rt.cleanup(control/'.container.cid',name,image,revision)
            log=control/'.native-output'
            if log.exists():
                rt.canonical(log);s=log.lstat();rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1
                    and (s.st_dev,s.st_ino)==log_inode,'Refuse foreign native-log cleanup');log.unlink()
            report['owned_cleanup_verified']=True
            rt.require(rt.source(root,code,revision,ENTRY,helpers)==source and auth(scan,rt,root,code)==evidence,'Original source/scan/private bytes changed')
            rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Snapshot parent changed after evaluation')
            rt.require('diagnostics'in report and all(rt.identity(out/n,4<<20)==pin for n,pin in report['diagnostics'].items()),'Diagnostics changed before seal')
            rt.require(time.monotonic()<=deadline,'Inclusive private evaluation posthash budget exceeded');report['source_rehashed_after']=True
        except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
        report.update(status='fail'if failure else 'pass',elapsed_seconds=time.monotonic()-started)
        rt.write(control/'report.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
    if failure:raise ValueError('Private hand diagnostic failed; sealed historical evidence retained')from None
    return report


def main():
    args=sys.argv[1:];mask_cohort=None
    if len(args)>=2 and args[-2]=='--mask-cohort'and args[-1]in('v2','v3'):mask_cohort=args[-1];args=args[:-2]
    if args[:1]==['--native']:
        if len(args)!=6:raise ValueError('Exact internal arguments required')
        root,code,out=map(Path,args[1:4]);_,rt=modules(code)
        return run_native(rt,root,code,out,Path('/opt/hand-evaluation-proof.json'),dict(bytes=int(args[4]),sha256=args[5]),mask_cohort)
    if args:raise ValueError('Only fixed --mask-cohort v2/v3 is permitted')
    return run(ROOT,Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'],mask_cohort)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps(dict(stage='mediapipe_hand_private_evaluation',status='fail',error_type=type(exc).__name__)));sys.exit(1)
