"""Evaluate frozen object camera geometry against private synthesis truth.

No fitting/alignment, regeneration, repair or adoption. Anchor-camera surface
Chamfer combines predicted pose and geometry; it is not the full Track1 score.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
from scipy.spatial import cKDTree
from world_reward.data import sha256

STAGE='private_object_rgb_anchor_geometry_quality'
SAMPLES=18439


def require_hash(path,digest):
    if (not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest)
            or path.is_symlink() or not path.is_file() or path.resolve()!=path.absolute() or sha256(path)!=digest):
        raise ValueError('Frozen regular artifact mismatch')


def mesh_arrays(v,f):
    v,f=np.asarray(v),np.asarray(f)
    if (v.ndim!=2 or v.shape[1]!=3 or v.dtype.kind!='f' or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu' or not len(f)
            or np.any(f<0) or np.any(f>=len(v))):raise ValueError('Finite mesh with valid integer triangles required')
    return v.astype(np.float64),f


def surface_samples(vertices,faces,seed=0):
    v,f=mesh_arrays(vertices,faces);tri=v[f]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('Finite noncollapsed surface required')
    rng=np.random.default_rng(seed);selected=tri[rng.choice(len(tri),SAMPLES,p=area/area.sum())]
    r=rng.random((SAMPLES,2));a=np.sqrt(r[:,0]);b=r[:,1]
    return selected[:,0]*(1-a[:,None])+selected[:,1]*(a*(1-b))[:,None]+selected[:,2]*(a*b)[:,None]


def metrics(pred_v,pred_f,true_v,true_f):
    prediction=surface_samples(pred_v,pred_f);truth=surface_samples(true_v,true_f)
    a=cKDTree(truth).query(prediction,workers=1)[0].mean();b=cKDTree(prediction).query(truth,workers=1)[0].mean()
    return {'chamfer_sum_cm':float((a+b)*100),'pred_to_truth_cm':float(a*100),'truth_to_pred_cm':float(b*100),
            'surface_samples_each':SAMPLES,'metric_scope':'two directed surface means, not exact Track1 role sample/RNG'}


def decision(cases):
    before=np.asarray([c['single']['chamfer_sum_cm'] for c in cases]);after=np.asarray([c['three_view']['chamfer_sum_cm'] for c in cases])
    if len(cases)!=2 or not np.isfinite(np.r_[before,after]).all() or np.any(before<=0) or np.any(after<0):
        raise ValueError('Require complete two-object finite nonzero baseline quality')
    gain=float(1-np.median(after)/np.median(before));regression=(after/before-1).tolist()
    valid=all(c[m]['watertight'] and c[m]['winding_consistent'] and c[m]['signed_volume_camera_m3']>0 for c in cases for m in ('single','three_view'))
    return {'median_camera_chamfer_relative_gain':gain,'per_object_relative_regression':regression,
            'closed_oriented_positive_volume_all':valid,'synthetic_hypothesis_supported':bool(gain>=.05 and max(regression)<=.05 and valid),
            'embedding_intersections_verified':False,'full_v2d_score_verified':False,'adoption_performed':False}


def run(root,report):
    base=root/'validation/objects_rgb_v1';proposal=base/'proposals';private=base/'eval_private'
    p=proposal/'report.json';digest=sha256(p);require_hash(p,digest);receipt=json.loads(p.read_text())
    expected={'stage':'public_object_rgb_single_and_three_view_proposals','status':'pass','private_truth_read':False,
              'challenge_inputs_used':False,'adoption_performed':False,'actual_raw_decoder_parity_verified':True,
              'native_camera_transform_parity_verified':True,'train_views':[0,2,4]}
    if any(type(receipt.get(k)) is not type(v) or receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Require actual frozen public camera-parity proposals')
    records=receipt.get('proposals',[])
    if [(r.get('object_index'),r.get('mode')) for r in records]!=[(o,m) for o in range(2) for m in ('single','three_view')]:
        raise ValueError('Require every original paired object proposal')
    # Load and hash every prediction before opening private evaluation files.
    frozen={}
    for r in records:
        name=f"object_{r['object_index']:02d}_{r['mode']}.npz"
        if r.get('file')!=name:raise ValueError('Proposal file identity differs')
        require_hash(proposal/name,r['sha256'])
        with np.load(proposal/name,allow_pickle=False) as data:frozen[r['object_index'],r['mode']]={k:data[k].copy() for k in data.files}
    observations=base/'observations-v2/report.json';require_hash(observations,receipt.get('observation_report_sha256'))
    observer=json.loads(observations.read_text());require_hash(base/'inputs/manifest.json',observer.get('input_manifest',{}).get('sha256'))
    render_path=private/'render-report.json';render_sha=sha256(render_path);require_hash(render_path,render_sha)
    render=json.loads(render_path.read_text())
    if render.get('stage')!='own_procedural_object_rgb_validation' or render.get('status')!='pass' or render.get('inference_performed') is not False:
        raise ValueError('Require complete private own renderer')
    if render.get('public_manifest_sha256')!=observer['input_manifest']['sha256']:raise ValueError('Synthesis/public data identity differs')
    report.update(proposal_report_sha256=digest,render_report_sha256=render_sha,observation_report_sha256=sha256(observations),
                  predictions_frozen_before_private_truth_read=True,cases=[])
    for obj in range(2):
        mesh_path=private/f'object_{obj:02d}_mesh.npz';entry=render['meshes'][obj]
        if entry.get('object_index')!=obj:raise ValueError('Private mesh identity differs')
        require_hash(mesh_path,entry['sha256'])
        with np.load(mesh_path,allow_pickle=False) as data:true_v,true_f=mesh_arrays(data['vertices_m'],data['faces'])
        anchor=private/f'object_{obj:02d}_view_00.npz';record=render['cases'][obj*6]
        if record.get('object_index')!=obj or record.get('view_index')!=0:raise ValueError('Private anchor identity differs')
        require_hash(anchor,record['truth_sha256'])
        with np.load(anchor,allow_pickle=False) as data:R,t=data['camera_R'],data['camera_t']
        if (R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(np.r_[R.ravel(),t]).all()
                or not np.allclose(R@R.T,np.eye(3),rtol=0,atol=1e-12) or not np.isclose(np.linalg.det(R),1)):
            raise ValueError('Private synthesis rotation/translation invalid')
        truth=true_v@R.T+t;case={'object_index':obj,'truth_anchor_sha256':record['truth_sha256']}
        for mode in ('single','three_view'):
            data=frozen[obj,mode];v,f=mesh_arrays(data['vertices_camera_m'],data['faces'])
            proof=next(r for r in records if r['object_index']==obj and r['mode']==mode)
            case[mode]=metrics(v,f,truth,true_f)|{k:proof[k] for k in ('watertight','winding_consistent','signed_volume_camera_m3')}
        report['cases'].append(case)
    report['decision']=decision(report['cases']);require_hash(p,digest);require_hash(render_path,render_sha)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Require isolated remote CPU evaluation')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/objects_rgb_v1/quality';path=out/'report.json'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Require exclusively reserved output')
    revision,image=os.environ.get('WR_CODE_REVISION',''),os.environ.get('WR_IMAGE_ID','')
    if not re.fullmatch(r'[0-9a-f]{40}',revision) or not re.fullmatch(r'sha256:[0-9a-f]{64}',image):raise ValueError('Immutable evaluation source/image required')
    report={'stage':STAGE,'status':'fail','code_revision':revision,'image_id':image,'script_sha256':sha256(Path(__file__)),
            'network':'none','gpu_used':False,'evaluation_truth_used':True,'challenge_inputs_used':False,
            'adoption_performed':False,'alignment_performed':False,'fitting_performed':False,'full_v2d_score_verified':False,
            'heldout_pose_evaluation_performed':False,'photorealistic_domain_validation':False,
            'preregistered_gate':{'median_camera_chamfer_gain':.05,'per_object_max_regression':.05,
                                  'closed_oriented_positive_volume_required':True,'surface_samples':SAMPLES}}
    started=time.perf_counter()
    try:run(root,report);report['status']='pass'
    except Exception as exc:report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        with path.open('x') as stream:json.dump(report,stream,allow_nan=False,indent=2)
    print(json.dumps({'stage':STAGE,'status':report['status'],'decision':report['decision']}))


if __name__=='__main__':main()
