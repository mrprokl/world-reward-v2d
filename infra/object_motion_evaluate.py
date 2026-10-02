"""Evaluate all frozen temporal proposals; private truth never enters inference.

Absolute camera surface CD contains inherited anchor scale/shape/layout error.
Known-motion-on-predicted-anchor CD is a separate diagnostic, NOT actual GT shape
CD or per-frame alignment. Both original fixed geometries/trajectories are kept.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import time

import numpy as np
from scipy.spatial import cKDTree
import object_synthetic_generate as native
import object_synthetic_evaluate as surfaces
from world_reward.data import sha256

STAGE='private_object_motion_fixed_mesh_quality'


def cd(a,b):return float((cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean())*100)


def rotation_error(a,b):
    r=a@b.T
    return float(np.arctan2(np.linalg.norm([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])/2,(np.trace(r)-1)/2)*180/np.pi)


def decision(cases):
    before=np.array([c['baseline_mean_camera_cd_cm'] for c in cases]);after=np.array([c['candidate_mean_camera_cd_cm'] for c in cases])
    if len(cases)!=3 or not np.isfinite(np.r_[before,after]).all() or np.any(before<=0) or np.any(after<0):raise ValueError('Complete three-object paired evaluation required')
    gain=float(1-np.median(after)/np.median(before));reg=after/before-1
    fast=all(c['candidate_fast_camera_cd_cm']<=c['baseline_fast_camera_cd_cm'] for c in cases)
    closed=all(c['anchor_closed_oriented_positive_volume'] for c in cases)
    return {'median_camera_cd_gain':gain,'per_object_camera_cd_regression':reg.tolist(),'fast_step_nonregression':fast,
            'anchor_closed_oriented_positive_volume_all':closed,
            'synthetic_hypothesis_supported':bool(gain>=.05 and np.max(reg)<=.05 and fast and closed),
            'adoption_performed':False,'full_track1_score_verified':False}


def run(root,report):
    base=root/'validation/object_motion_v1';p=base/'tracking/report.json';ph=sha256(p);native.require_hash(p,ph);r=json.loads(p.read_text())
    if (r.get('stage')!='public_fixed_mesh_rgb_motion_proposals' or r.get('status')!='pass' or r.get('private_truth_read') is not False
            or r.get('original_frame_coverage_verified') is not True or r.get('fixed_shape_preserved') is not True or r.get('fixed_scale_preserved') is not True):raise ValueError('Require frozen actual public trajectory producer')
    pred=base/'tracking/predictions.npz';native.require_hash(pred,r.get('predictions_sha256'))
    with np.load(pred,allow_pickle=False) as d:R,t=d['rotations'].copy(),d['translations'].copy();fi=d['frame_index'];oi=d['object_index']
    if (R.shape!=(2,3,8,3,3) or t.shape!=(2,3,8,3) or not np.isfinite(np.r_[R.ravel(),t.ravel()]).all()
            or not np.allclose(R@R.swapaxes(-1,-2),np.eye(3),atol=1e-5,rtol=0) or not np.allclose(np.linalg.det(R),1,atol=1e-5)
            or not np.array_equal(fi,np.arange(8)) or not np.array_equal(oi,np.arange(3))):raise ValueError('All finite original rigid trajectories required')
    ap=base/'anchors/report.json';op=base/'observations/report.json'
    native.require_hash(ap,r['anchor_report_sha256']);native.require_hash(op,r['observation_report_sha256']);ar=json.loads(ap.read_text())
    # Freeze all three predicted surfaces BEFORE the first private truth read.
    frozen=[]
    for obj,record in enumerate(ar['proposals']):
        artifact=base/'anchors'/f'object_{obj:02d}_anchor.npz';native.require_hash(artifact,record['sha256'])
        with np.load(artifact,allow_pickle=False) as d:frozen.append(surfaces.surface_samples(d['vertices_camera_m'],d['faces']))
    private=base/'eval_private';rp=private/'render-report.json';rh=sha256(rp);native.require_hash(rp,rh);render=json.loads(rp.read_text())
    if (render.get('stage')!='own_procedural_object_motion_rgb' or render.get('status')!='pass' or render.get('inference_performed') is not False
            or [(x['object_index'],x['frame_index']) for x in render['cases']]!=[(o,f) for o in range(3) for f in range(8)]):raise ValueError('Private renderer complete identities required')
    observer=json.loads(op.read_text());native.require_hash(base/'inputs/manifest.json',observer['input_manifest']['sha256'])
    if observer['input_manifest']['sha256']!=render['public_manifest_sha256']:raise ValueError('RGB evidence identity differs')
    report.update(prediction_report_sha256=ph,render_report_sha256=rh,predictions_frozen_before_private_truth_read=True,cases=[])
    for obj in range(3):
        mesh=private/f'object_{obj:02d}_mesh.npz';native.require_hash(mesh,render['meshes'][obj]['sha256'])
        with np.load(mesh,allow_pickle=False) as d:truth=surfaces.surface_samples(d['vertices_m'],d['faces'])
        known=[]
        for frame in range(8):
            a=private/f'object_{obj:02d}_frame_{frame:03d}.npz';native.require_hash(a,render['cases'][obj*8+frame]['truth_sha256'])
        with np.load(a,allow_pickle=False) as d:known.append((d['camera_R'].copy(),d['camera_t'].copy()))
        if any(tr.shape!=(3,3) or tt.shape!=(3,) or not np.isfinite(np.r_[tr.ravel(),tt]).all()
               or not np.allclose(tr@tr.T,np.eye(3),atol=1e-12,rtol=0) or not np.isclose(np.linalg.det(tr),1) for tr,tt in known):
            raise ValueError('Private renderer rigid transforms invalid')
        proof=ar['proposals'][obj]
        case={'object_index':obj,'frames':[], 'anchor_closed_oriented_positive_volume':bool(
            proof['watertight'] and proof['winding_consistent'] and proof['signed_volume_camera_m3']>0)}
        for frame,(tr,tt) in enumerate(known):
            rr=tr@known[0][0].T;rt=tt-rr@known[0][1]
            target=truth@tr.T+tt;motion_reference=frozen[obj]@rr.T+rt
            record={'frame_index':frame,'modes':{}}
            for mode,label in enumerate(('baseline','candidate')):
                sample=frozen[obj]@R[mode,obj,frame].T+t[mode,obj,frame]
                record['modes'][label]={'camera_cd_cm':cd(sample,target),'known_motion_on_predicted_anchor_cd_cm_diagnostic':cd(sample,motion_reference),
                    'relative_rotation_error_deg':rotation_error(R[mode,obj,frame],rr),'relative_camera_translation_error_cm':float(np.linalg.norm(t[mode,obj,frame]-rt)*100)}
            case['frames'].append(record)
        for label in ('baseline','candidate'):
            case[label+'_mean_camera_cd_cm']=float(np.mean([f['modes'][label]['camera_cd_cm'] for f in case['frames'][1:]]))
            case[label+'_fast_camera_cd_cm']=case['frames'][5]['modes'][label]['camera_cd_cm']
            case[label+'_mean_relative_motion_cd_cm_diagnostic']=float(np.mean([f['modes'][label]['known_motion_on_predicted_anchor_cd_cm_diagnostic'] for f in case['frames'][1:]]))
        report['cases'].append(case)
    report['decision']=decision(report['cases']);native.require_hash(p,ph);native.require_hash(rp,rh)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Require isolated remote CPU evaluation')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/object_motion_v1/quality';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require exclusive motion quality output')
    report={'stage':STAGE,'status':'fail','code_revision':os.environ['WR_CODE_REVISION'],'image_id':os.environ['WR_IMAGE_ID'],
            'script_sha256':sha256(Path(__file__)),'gpu_used':False,'evaluation_truth_used':True,'challenge_inputs_used':False,
            'adoption_performed':False,'alignment_performed':False,'pose_fitting_performed':False,'all_original_frames_evaluated':True,
            'budget_seconds':120,'preregistered_gate':{'median_camera_cd_gain':.05,'max_object_regression':.05,'fast_step_nonregression':True},
            'relative_motion_reference_scope':'private true motion applied to frozen predicted anchor ONLY evaluation diagnostic, not GT-shape CD'}
    started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Frozen120s private eval deadline')));signal.alarm(120)
    try:run(root,report);report['status']='pass'
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-started;native.persist(path,report)
    print(json.dumps({k:report[k] for k in ('stage','status','elapsed_seconds','decision')}))

if __name__=='__main__':main()
