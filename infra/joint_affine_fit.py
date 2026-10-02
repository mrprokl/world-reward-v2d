"""Freeze predicted-human affine/alpha-only depth before any private J3 truth.

Public spatial holdout selects the extra offset model. Underconstrained slopes
abstain to the unchanged alpha-only comparator; invalid inputs remain fatal.
No camera, canonical mesh, human geometry or source validity is changed.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import joint_affine_infer as observations
from world_reward.affine_depth_alignment import (UnderconstrainedDepthAlignment,
    fit_shared_affine_depth_alignment, spatial_split_mask)
from world_reward.data import sha256
from world_reward.metric_alignment import fit_shared_depth_scale
from world_reward.pointmap import validate_camera_pointmap

STAGE = 'public_joint_affine_depth_alignment_predictions'
CLIPS, FRAMES, HEIGHT, WIDTH, BUDGET = 3, 6, 768, 1024, 90
K = np.array([[1280.,0.,512.],[0.,1280.,384.],[0.,0.,1.]])
PACKED = {'baseline_points','selected_points','validity','clip_index','frame_index','K'}


def validate_producer(report, records):
    """Verify saved native provenance without importing models or reading GT."""
    expected = {'network': 'none', 'affine_fit_performed': False, 'scale_fit_performed': False,
                'camera_source': 'fixed_RGB_size_prior', 'body_inference_type': 'body',
                'geometry_filled': False, 'apply_mask': False, 'force_projection': True,
                'frames_completed': 18, 'adoption_performed': False}
    body, depth = observations.human, observations.depth_model
    source, model = report.get('MoGe_source', {}), report.get('body_model', {})
    if (any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items())
            or not re.fullmatch('[0-9a-f]{40}', str(report.get('producer_revision', '')))
            or not re.fullmatch('sha256:[0-9a-f]{64}', str(report.get('image_id', '')))
            or report.get('script_sha256') != sha256(Path(observations.__file__))
            or report.get('helper_source_sha256') != observations.helper_identities()
            or report.get('camera_K') != observations.CAMERA_K.tolist()
            or report.get('focal_geometry_source_sha256') != observations.GEOMETRY_SHA
            or report.get('MoGe_model_revision') != depth.MODEL_REVISION
            or report.get('MoGe_model_asset') != {'sha256': depth.MODEL_SHA, 'bytes': depth.MODEL_BYTES}
            or source.get('revision') != depth.SOURCE_REVISION or source.get('v2_source_sha256') != depth.SOURCE_V2_SHA
            or model.get('body_revision') != body.body.BODY_REVISION
            or model.get('upstream_revision') != body.body.UPSTREAM_REVISION
            or model.get('dinov3_revision') != body.body.DINOV3_REVISION
            or model.get('body_assets', {}).get('model.ckpt') != {'sha256': body.BODY_SHA, 'bytes': body.BODY_BYTES}):
        raise ValueError('Exact native public producer, model pins and no prior fit required')
    calls = report.get('native_focal_solver_calls')
    if not isinstance(calls, list) or len(calls) != len(records):
        raise ValueError('Every original image requires its actual native focal call')
    for record, call in zip(records, calls):
        if (not isinstance(call, dict) or call.get('file') != record['file']
                or any(type(call.get(k)) is not int or call[k] != record[k] for k in ('clip_index', 'frame_index'))
                or type(call.get('native_nearest64_valid_pixels')) is not int
                or not 2 <= call['native_nearest64_valid_pixels'] <= 4096
                or call.get('focal_prior_supplied') is not True or call.get('original_solver_returned') is not True):
            raise ValueError('Native focal call identity/support/return proof incomplete')


def frozen_observations(root):
    records, public_hashes = observations.public_inputs(root)
    base = root/'validation/joint_affine_rgb_v1/predictions_v1'; path = base/'report.json'
    report_hash=observations.depth_model.identity(path)['sha256']; report=json.loads(path.read_text())
    if (report.get('stage')!=observations.STAGE or report.get('status')!='pass' or report.get('phase')!='complete'
            or type(report.get('body_calls_completed')) is not int or report.get('body_calls_completed')!=18
            or type(report.get('MoGe_calls_completed')) is not int or report.get('MoGe_calls_completed')!=18
            or any(report.get(k) is not True for k in ['actual_body_inference','actual_MoGe_inference','actual_predicted_human_render','inputs_assets_sources_rechecked'])
            or report.get('public_inputs_sha')!=public_hashes['public_inputs_sha']
            or report.get('mask_report_sha')!=public_hashes['mask_report_sha']
            or report.get('public_records')!=[{k:v for k,v in r.items() if not k.endswith('_path')} for r in records]
            or any(report.get(k) is not False for k in ['private_truth_read','ground_truth_used','challenge_inputs_used','hand_labeled_test'])
            or report.get('oracle_modes')!=[] or len(report.get('outputs',[]))!=18):
        raise ValueError('Complete actual eighteen-image RGB-only observations required')
    validate_producer(report, records)
    from PIL import Image
    data=[]; frozen=[(path,report_hash)]
    for record,row in zip(records,report['outputs']):
        clip,frame=record['clip_index'],record['frame_index'];filename=f'clip_{clip:02d}_frame_{frame:03d}.npz'
        if (row.get('file')!=filename or type(row.get('clip_index')) is not int or row.get('clip_index')!=clip
                or type(row.get('frame_index')) is not int or row.get('frame_index')!=frame
                or row.get('rgb_sha256')!=record['rgb_sha256']):
            raise ValueError('Original RGB/observation ordered identity mismatch')
        p=base/filename
        if observations.depth_model.identity(p)['sha256']!=row.get('sha256'):
            raise ValueError('Frozen original observation changed')
        with np.load(p,allow_pickle=False) as file: value={k:file[k].copy() for k in file.files}
        observations.validate_observation(value,clip,frame)
        for kind in ('human', 'object'):
            if not np.array_equal(value[kind+'_mask'], observations.read_mask(record[kind+'_mask_path'], Image)>0):
                raise ValueError('Observation masks differ from frozen automatic PNG masks')
        data.append(value);frozen.append((p,row['sha256']))
    if {p.name for p in base.iterdir()}!={'report.json',*[p.name for p,_ in frozen[1:]]}:
        raise ValueError('Unexpected observation files')
    return data,records,public_hashes,frozen


def unproject(depth,validity,alpha,beta,camera=None):
    camera=K if camera is None else camera
    z=np.asarray(depth);v=np.asarray(validity)
    if (z.shape!=(HEIGHT,WIDTH) or z.dtype!=np.float32 or v.dtype!=np.bool_ or v.shape!=z.shape
            or not v.any() or not np.isfinite(z[v]).all() or np.any(z[v]<=0)
            or type(alpha) not in (int,float) or not np.isfinite(alpha) or alpha<=0
            or type(beta) not in (int,float) or not np.isfinite(beta) or not np.array_equal(camera,K)):
        raise ValueError('Original valid camera-Z, fixed K and positive finite slope required')
    with np.errstate(over='ignore',invalid='ignore'):
        aligned=(alpha*z.astype(np.float64)+beta).astype(np.float32)
        yy,xx=np.indices(z.shape)
        p=np.stack(((xx+.5-K[0,2])*aligned/K[0,0],(yy+.5-K[1,2])*aligned/K[1,1],aligned),axis=-1).astype(np.float32)
    validate_camera_pointmap(aligned,p,v,np.diag([1/WIDTH,1/HEIGHT,1.])@K,K)
    return p


def fit_clip(values):
    if len(values)!=FRAMES: raise ValueError('Every original clip frame required')
    z=np.stack([v['raw_depth'] for v in values]);h=np.stack([v['rendered_depth'] for v in values])
    support=np.stack([v['silhouette']&v['human_mask']&~v['object_mask']&v['validity'] for v in values])
    train=support&spatial_split_mask(HEIGHT,WIDTH)
    heldout=support&~spatial_split_mask(HEIGHT,WIDTH)
    if (np.any(support.sum((1,2))<64) or np.any(train.sum((1,2))<32) or np.any(heldout.sum((1,2))<32)
            or not np.isfinite(z[support]).all() or not np.isfinite(h[support]).all()
            or np.any(z[support]<=0) or np.any(h[support]<=0)):
        raise ValueError('Every frame requires full finite positive human support and both spatial splits')
    baseline=fit_shared_depth_scale(z,h,train,list(range(FRAMES)),min_correspondences_per_frame=32,min_supported_frames=FRAMES)
    try:
        result=fit_shared_affine_depth_alignment(z,h,support,list(range(FRAMES)))
        if result.alpha_only_scale!=baseline.shared_scale: raise ValueError('Paired alpha-only train pixels differ')
        alpha,betas=result.shared_scale,result.frame_offsets;diagnostic=result.to_dict()
    except UnderconstrainedDepthAlignment as error:
        # This fallback is not a rank waiver or offset fit. Raw comparator remains.
        alpha,betas=baseline.shared_scale,(0.,)*FRAMES
        diagnostic={'selection':'alpha_only','reason':'within_frame_slope_underconstrained',
                    'error':str(error),'affine_fit_performed':False,'adoption_authorized':False}
    before=[];after=[]
    for i,value in enumerate(values):
        before.append(unproject(value['raw_depth'],value['validity'],baseline.shared_scale,0.,value['K']))
        after.append(unproject(value['raw_depth'],value['validity'],float(alpha),float(betas[i]),value['K']))
    return before,after,{'alpha_only':baseline.to_dict(),'selected':diagnostic,
                        'selected_alpha':float(alpha),'selected_offsets':list(betas)}


def main(argv=None):
    argparse.ArgumentParser(allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Remote network-none CPU public fit required')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/joint_affine_rgb_v1/alignment_v1'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()): raise FileExistsError('Exclusive new alignment directory required')
    rev,image=os.environ['WR_CODE_REVISION'],os.environ['WR_IMAGE_ID']
    if not re.fullmatch('[0-9a-f]{40}',rev) or not re.fullmatch('sha256:[0-9a-f]{64}',image):raise ValueError('Immutable producing source/image required')
    report={'stage':STAGE,'status':'fail','phase':'integrity','producer_revision':rev,'image_id':image,
        'script_sha256':sha256(Path(__file__)),'budget_seconds':BUDGET,'network':'none','private_truth_read':False,
        'challenge_inputs_used':False,'ground_truth_used':False,'hand_labeled_test':False,'oracle_modes':[],
        'adoption_performed':False,'human_geometry_modified':False,'canonical_object_mesh_modified':False,
        'camera_modified':False,'source_validity_modified':False,'clips':[],'outputs':[]}
    start=time.perf_counter()
    def expired(*_):raise TimeoutError('Whole public affine fit exceeded90s')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
    try:
        values,records,hashes,frozen=frozen_observations(root)
        report.update(public_hashes=hashes,observation_report_sha256=frozen[0][1],phase='fit')
        for clip in range(CLIPS):
            before,after,diagnostic=fit_clip(values[clip*FRAMES:(clip+1)*FRAMES])
            report['clips'].append({'clip_index':clip,**diagnostic})
            for frame in range(FRAMES):
                v=values[clip*FRAMES+frame];p=out/f'clip_{clip:02d}_frame_{frame:03d}.npz'
                with p.open('xb') as f:np.savez_compressed(f,baseline_points=before[frame],selected_points=after[frame],
                    validity=v['validity'],clip_index=np.array(clip,np.int64),frame_index=np.array(frame,np.int64),K=v['K'])
                report['outputs'].append({'file':p.name,'sha256':sha256(p),'clip_index':clip,'frame_index':frame})
        if any(p.is_symlink() or sha256(p)!=h for p,h in frozen) or observations.public_inputs(root)!=(records,hashes):
            raise ValueError('Frozen observation/RGB/mask changed during fit')
        report.update(status='pass',phase='complete',frames=18,outputs_completed=18)
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-start
        with (out/'report.json').open('x') as f:json.dump(report,f,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['stage','status','outputs_completed','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
