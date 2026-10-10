"""Saved-only CPU diagnostics: no fits, labels, model calls or production writes."""
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from world_reward.sequence_pose import SequenceContactEvidence
import sequence_pose_contact_probe as contact
from mediapipe_cpu_runtime_verify import source, strict

saved = contact.saved
old = saved.old
ROOT = old.ROOT
ENTRY = 'run_pose_gain_probe'
RGBD_SOURCE = '9d31ed96edd24ed3bae13df0627b1fef26164325'
CONTACT_SOURCE = '40183b3a83ba59080c192c3cdf2db9e1d76ef021'
RGBD_REPORT_PIN = dict(bytes=319383,sha256='87f96748a6bb529cc81d5179b03c4fc62ce71a91f591e15cf1e22053a544892a')
RGBD_PIN = dict(bytes=233604,sha256='8340fba63e5393c0a22f6ffa08de6f49d080e4c52692a2bd0661aac7e1d2993f')
CONTACT_REPORT_PIN = dict(bytes=18236,sha256='2f4c602e666fcea91914eb55530836f094343daba85b95a5e5ffb5e076d013c4')
CONTACT_PIN = dict(bytes=176637,sha256='4e8060965215a120148109bcf392ed25462c9c48bc80a262b84528529570bdbc')
HELPERS = ('infra/pose_gain_probe.py','infra/run_pose_gain_probe.sh',*contact.HELPERS)


def same_bank(original, bank, reference=None):
    n = len(original['object_translation']); grid = np.arange(n)
    if (not np.array_equal(original['frame_index'],grid) or not np.array_equal(bank['frame_index'],grid)
            or float(original['object_scale']) != 1. or any(not np.array_equal(original[k],bank[k])
                for k in ('object_vertices','object_faces','object_scale','camera_K'))
            or bank['rotation'].shape != (n,3,3) or bank['translation'].shape != (n,3)
            or bank['points'].shape != (33,3) or bank['tracks_xy'].shape != (n,33,2)
            or bank['RGB_visible'].dtype != bool or bank['RGB_visible'].shape != (n,33)
            or not np.isfinite(bank['points']).all()
            or not np.isfinite(bank['tracks_xy'][bank['RGB_visible']]).all()
            or not np.isnan(bank['tracks_xy'][~bank['RGB_visible']]).all()):
        raise ValueError('Fixed full-T geometry/gauge/material observation bank required')
    contact._rigid(bank['rotation'],bank['translation'])
    if reference is not None:
        for key in ('points','tracks_xy','RGB_visible'):
            if not np.array_equal(reference[key],bank[key],equal_nan=True):
                raise ValueError('Material witnesses changed between saved candidates')


def saved_report(ledger,path,pin,producer,output_pin,status):
    row = strict(ledger.read(path,pin))
    expected = dict(producer_revision=producer,numerical_source=old.SOURCE,status=status,
        ground_truth_used=False,manual_labels=False,model_calls=0,full_4D_export_replaced=False,output=output_pin)
    if any(type(row.get(k)) is not type(v) or row[k] != v for k,v in expected.items()):
        raise ValueError('Independent no-GT saved candidate lineage differs')
    return row


def acceleration(vertices,r,t,fps):
    centroid = np.einsum('tij,j->ti',r,vertices.mean(0))+t
    a = np.linalg.norm(np.diff(centroid,n=2,axis=0),axis=1)*fps**2
    spin = Rotation.from_matrix(r[1:]@r[:-1].swapaxes(-1,-2)).as_rotvec()*fps
    return a,np.linalg.norm(np.diff(spin,axis=0),axis=1)*fps


def summary(values):
    a = np.asarray(values); a = a[np.isfinite(a)]
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,p95=float(np.quantile(a,.95)) if len(a) else None)


def correlation(a,b):
    a,b = np.asarray(a),np.asarray(b); good = np.isfinite(a)&np.isfinite(b)
    if good.sum()<3 or np.std(a[good])==0 or np.std(b[good])==0: return None
    return float(np.corrcoef(a[good],b[good])[0,1])


def support_overlap(ledger,base,inventory,xy,visible,scale):
    """Descriptive native-mask pixel membership, not hand/occlusion truth."""
    from PIL import Image
    person = np.zeros(visible.shape,bool); object_only = person.copy(); object_any = person.copy()
    for i in range(len(xy)):
        masks = []
        for role in (0,1):
            path = base/f'automatic_masks/masks/{role}/{i:06d}.png'; ledger.record(path,inventory[f'{role}/{i:06d}.png'])
            with Image.open(path) as image:
                if image.mode != 'L': raise ValueError('Original binary native mask required')
                mask = np.asarray(image)
            if mask.ndim != 2 or not np.isin(mask,[0,255]).all(): raise ValueError('Invalid original native mask')
            masks.append(mask>0)
        if masks[0].shape != masks[1].shape: raise ValueError('One original mask grid required')
        h,w = masks[0].shape
        if not np.array_equal(scale,np.array([640/w,480/h])): raise ValueError('Original RGB/mask scaling differs')
        ids = np.flatnonzero(visible[i]); p = np.floor((xy[i,ids]+.5)/scale).astype(int)
        inside = (p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h); ids,p = ids[inside],p[inside]
        person[i,ids] = masks[0][p[:,1],p[:,0]]; object_any[i,ids] = masks[1][p[:,1],p[:,0]]
        object_only[i] = object_any[i]&~person[i]
    denominator = visible.sum(1)
    fractions = {name:np.divide(value.sum(1),denominator,out=np.full(len(xy),np.nan),where=denominator>0)
        for name,value in (('person',person),('object_only',object_only),('object_any',object_any))}
    return dict(RGB_supported_queries=int(visible.sum()),person_queries=int(person.sum()),
        object_only_queries=int(object_only.sum()),object_any_queries=int(object_any.sum()),
        perframe_fraction={key:[float(x) if np.isfinite(x) else None for x in value] for key,value in fractions.items()}),fractions


def projection_jacobian(points,vertices,r,t,K):
    """SE3 derivative at physical centroid; translation normalized by diameter."""
    diameter = float(np.linalg.norm(np.ptp(vertices,axis=0))); centred = (points-vertices.mean(0))@r.T
    xyz = points@r.T+t
    if diameter<=0 or not np.isfinite(xyz).all() or (xyz[:,2]<=0).any() or not np.array_equal(K[2],np.array([0.,0.,1.])) or K[0,1]!=0 or K[1,0]!=0:
        raise ValueError('Positive camera depth and canonical projection required')
    x,y,z = xyz.T; j = np.zeros((len(points),2,3))
    j[:,0,0]=K[0,0]/z; j[:,0,1]=K[0,1]/z; j[:,0,2]=-(K[0,0]*x+K[0,1]*y)/z**2
    j[:,1,0]=K[1,0]/z; j[:,1,1]=K[1,1]/z; j[:,1,2]=-(K[1,0]*x+K[1,1]*y)/z**2
    skew = np.zeros((len(points),3,3)); u,v,w=centred.T
    skew[:,0,1]=-w;skew[:,0,2]=v;skew[:,1,0]=w;skew[:,1,2]=-u;skew[:,2,0]=-v;skew[:,2,1]=u
    return np.concatenate((j@(-skew),j*diameter),axis=2).reshape(-1,6)


def concentration(a,period=96):
    centre = np.arange(1,len(a)+1); boundary = (centre%period<=1)|(centre%period>=period-1)
    total = float(a.sum()); order = np.argsort(-a)[:10]
    return dict(top10_frame_indices=centre[order].tolist(),top10_values=a[order].tolist(),
        top10_sum_fraction=float(a[order].sum()/total) if total else 0.,
        nominal_96_window_boundary=summary(a[boundary]),other_frames=summary(a[~boundary]),
        boundary_scope='descriptive_nominal_window_not_verified_cause')


def depth_commonmode(points,r,t,depth,support,fps):
    if depth.shape!=support.shape or support.dtype!=bool or not np.isfinite(depth[support]).all() or not np.isnan(depth[~support]).all():
        raise ValueError('Explicit untouched inferred axial depth support required')
    z = (points[None]@r.swapaxes(-1,-2)+t[:,None])[...,2]
    common = np.full(len(t),np.nan); independent=[]
    for i in range(len(t)):
        if support[i].sum():
            e=depth[i,support[i]]-z[i,support[i]];common[i]=np.median(e);independent.extend(np.abs(e-common[i]))
    good = np.isfinite(common); diff=np.diff(common); pair=good[1:]&good[:-1]
    return dict(scope='inferred_depth_minus_original_pose_prediction_not_truth',frame_median_residual_m=[float(x) if np.isfinite(x) else None for x in common],
        commonmode_absolute=summary(np.abs(common)),withinframe_absolute_about_median=summary(independent),frame_median_absolute_deviation_m=[float(np.median(np.abs(depth[i,support[i]]-z[i,support[i]]-common[i]))) if np.isfinite(common[i]) else None for i in range(len(t))],
        commonmode_velocity_m_s=summary(np.abs(diff[pair])*fps),lag1_correlation=correlation(common[:-1],common[1:]),
        depth_supported_per_point=support.sum(0).tolist(),depth_supported_per_frame=support.sum(1).tolist())


def stage_runtime(rows):
    out=[]
    for row in rows:
        stages=[]
        for value in row.get('stages',[]):
            elapsed=value.get('elapsed_seconds')
            if type(elapsed) in (int,float) and np.isfinite(elapsed) and elapsed>=0:
                stages.append(dict(stage=value['stage'],seconds=float(elapsed),reused=value.get('reused')))
        out.append(dict(episode=row['episode'],stages=sorted(stages,key=lambda r:-r['seconds']),
            measured_sum_seconds=sum(s['seconds'] for s in stages),scope='recorded_stage_wall_times_not_additive_parallel_total'))
    return out


def run():
    started=time.monotonic(); revision=os.environ['WR_CODE_REVISION']; code=Path(os.environ['WR_CODE'])
    if ROOT!=Path(os.environ['WR_ROOT']) or code!=ROOT/'jobs'/revision/ENTRY/'code': raise ValueError('Exact Azure source required')
    out=ROOT/'results'/('pose-gain-'+revision);old.fresh_runtime_output(out);ledger=old.ArtifactLedger()
    report=dict(status='fail',producer_revision=revision,ground_truth_used=False,manual_labels=False,model_calls=0,RGB_scope='same_fit_tracks_in_sample_consistency_not_accuracy',
        new_fits=0,production_adopted=False,heldout_accuracy_verified=False,full_4D_export_replaced=False)
    try:
        binding=source(ROOT,code,revision,ENTRY,HELPERS); report['source_binding']=binding
        for name in HELPERS:ledger.record(code/name)
        front,rows=old.saved_frontend(ledger);experiment=ROOT/'experiments'/('full4d-v1-'+old.SOURCE)
        old.saved_numerical_frontend(ledger,experiment,rows);base=experiment/'outputs/episode_000009';directory=base/'cari_shared_export_v1'
        pins=strict(ledger.read(experiment/'pins/cari_clip_000009_shared_export_pins.json'))['export_files']
        export=strict(ledger.read(directory/'report.json',pins['report.json']))
        if export['status']!='pass' or export['producer_revision']!=old.SOURCE or export['ground_truth_used'] is not False or export['oracle_modes']!=[]:raise ValueError('Original export lineage differs')
        original=saved.load_npz(ledger,directory/'trajectory.npz',pins['trajectory.npz'])
        track=ROOT/'results'/('sequence-pose-probe-'+saved.TRACK_SOURCE+'-v2');saved.source_track_report(ledger,track/'report.json')
        rgb=saved.load_npz(ledger,track/'episode_000009.npz',saved.TRACK_PIN)
        dp=ROOT/'results'/('sequence-pose-rgbd-'+RGBD_SOURCE); dr=saved_report(ledger,dp/'report.json',RGBD_REPORT_PIN,RGBD_SOURCE,RGBD_PIN,'complete_saved_RGBD_diagnostic_not_quality_pass')
        rgbd=saved.load_npz(ledger,dp/'episode_000009.npz',RGBD_PIN)
        cp=ROOT/'results'/('sequence-pose-contact-'+CONTACT_SOURCE);cr=saved_report(ledger,cp/'report.json',CONTACT_REPORT_PIN,CONTACT_SOURCE,CONTACT_PIN,'complete_saved_RGB_contact_diagnostic_not_quality_pass')
        coupled=saved.load_npz(ledger,cp/'episode_000009.npz',CONTACT_PIN)
        for bank in (rgb,rgbd,coupled):same_bank(original,bank,rgb)
        n=len(original['frame_index']);fps=dr['fps']
        if n!=415 or export['frames']!=n or fps!=30. or cr['fps']!=fps:raise ValueError('Original 415-frame 30FPS comparison required')
        inventory,_=old.authenticated_masks(ledger,front,rows[9],base,9,n);spec=export['clip_spec'];scale=np.array([640/spec['width'],480/spec['height']])
        K=original['camera_K'].copy();K[0]*=scale[0];K[1]*=scale[1];K[:2,2]-=.5
        v=original['object_vertices'];p=rgb['points'];xy=rgb['tracks_xy'];visible=rgb['RGB_visible']
        evidence=SequenceContactEvidence(coupled['contact_activations'],coupled['hand_points_camera'],coupled['hand_geometry_supported'],original['object_faces'],'same_pinned_frozen_052_prior_contacts')
        if int(evidence.activations.sum())!=271:raise ValueError('Original 271 frozen contact witnesses required')
        overlap,fraction=support_overlap(ledger,base,inventory,xy,visible,scale);report['mask_overlap']=overlap
        selected=coupled['selected_vertex_indices'];active=evidence.activations
        if selected.shape!=(n,2) or selected.dtype.kind not in 'iu' or active.dtype!=bool:raise ValueError('Frozen original contact witness selection required')
        supported_pair=active[1:]&active[:-1];switch=(selected[1:]!=selected[:-1])&supported_pair
        switch_centred=np.any(switch[:-1]|switch[1:],axis=1)
        report['contact_witness_switches']=dict(adjacent_active_hand_pairs=int(supported_pair.sum()),switches=int(switch.sum()),fraction=float(switch.sum()/supported_pair.sum()) if supported_pair.sum() else None,scope='predicted_contact_witness_switching_not_motion_truth')
        depth_diagnostic=depth_commonmode(p,original['object_rotation'],original['object_translation'],rgbd['tracks_depth_m'],rgbd['depth_visible'],fps)
        common=np.array(depth_diagnostic['frame_median_residual_m'],float);common_accel=np.diff(common,n=2)*fps**2
        candidates={'original':(original['object_rotation'],original['object_translation']),
            'RGB':(rgb['rotation'],rgb['translation']),'RGBD':(rgbd['rotation'],rgbd['translation']),
            'RGB_contact':(coupled['rotation'],coupled['translation'])};metrics={}
        for name,(r,t) in candidates.items():
            a,angular=acceleration(v,r,t,fps);metrics[name]=contact.residuals(v,p,r,t,K,xy,visible,evidence,fps)
            centroid=np.einsum('tij,j->ti',r,v.mean(0))+t
            metrics[name].update(centroid_spikes=concentration(a),angular_spikes=concentration(angular),
                centroid_acceleration_near_contact_witness_switch=summary(a[switch_centred]),centroid_acceleration_without_contact_witness_switch=summary(a[~switch_centred]),
                inferred_depth_commonmode_second_difference_centroidZ_accel_correlation=correlation(common_accel,np.diff(centroid[:,2],n=2)*fps**2),
                mask_overlap_acceleration_correlation={key:correlation(value[1:-1],a) for key,value in fraction.items()})
        r,t=candidates['original'];diameter=float(np.linalg.norm(np.ptp(v,axis=0)));singular=np.linalg.svd((p-p.mean(0))/diameter,compute_uv=False)
        svds=[np.linalg.svd(projection_jacobian(p[visible[i]],v,r[i],t[i],K),compute_uv=False) for i in range(n)]
        report.update(status='complete_saved_diagnostic_not_quality_pass',frames=n,points=len(p),fps=fps,metrics=metrics,
            fixed_geometry_gauge_and_observation_bank=True,canonical_points=dict(singular_values=singular.tolist(),rank=int(np.linalg.matrix_rank(p-p.mean(0))),RGB_support_per_point=visible.sum(0).tolist()),
            original_projection_observability=dict(translation_unit='mesh_diameter',sigma_min_per_frame=[float(s[-1]) if len(s)==6 else None for s in svds],sigma_max_per_frame=[float(s[0]) if len(s) else None for s in svds],scope='local_saved_pose_geometry_not_confidence'),
            depth_commonmode=depth_diagnostic,
            original_stage_runtime=stage_runtime(strict(ledger.read(experiment/'report.json',maximum=8<<20))['episodes']))
        ledger.verify()
        if source(ROOT,code,revision,ENTRY,HELPERS)!=binding:raise ValueError('Immutable source closure changed')
        report['sources']=ledger.records
    except Exception as exc:report.update(status='fail',error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds']=time.monotonic()-started
    path=out/'report.json';path.write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n');path.chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds')},allow_nan=False))
    if report['status']=='fail':raise SystemExit(1)


if __name__=='__main__':
    if len(os.sys.argv)!=1:raise SystemExit('No arbitrary inputs accepted')
    run()
