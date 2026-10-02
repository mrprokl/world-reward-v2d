"""Independent private J3 visible-surface quality after both methods freeze.

No GT alignment, fitting, prediction selection, intrinsic changes or exclusion
policy. Predicted and true supports are disclosed; complete whole-object mesh
and full V2D/HOI scores are NOT certified by this visible-pointmap experiment.
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
import joint_affine_fit as fit
import joint_rgb_evaluate as metric
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

BUDGET=90


def coefficients(clip, index):
    if (not isinstance(clip, dict) or type(clip.get('clip_index')) is not int or clip['clip_index'] != index
            or clip.get('alpha_only', {}).get('schema') != 'world-reward-shared-depth-scale-v1'):
        raise ValueError('Every original clip requires its frozen paired fit coefficients')
    baseline = clip['alpha_only'].get('shared_scale'); selected = clip.get('selected_alpha'); offsets = clip.get('selected_offsets')
    if (any(type(a) not in (int, float) or not np.isfinite(a) or a <= 0 for a in (baseline, selected))
            or not isinstance(offsets, list) or len(offsets) != fit.FRAMES
            or any(type(b) not in (int, float) or not np.isfinite(b) for b in offsets)):
        raise ValueError('One positive finite slope per clip and every finite frame offset required')
    diagnostic = clip.get('selected', {})
    if diagnostic.get('selection') == 'alpha_only':
        if selected != baseline or offsets != [0.]*fit.FRAMES:
            raise ValueError('Alpha-only selection must retain the exact unchanged comparator')
    elif diagnostic.get('selection') == 'affine':
        if (diagnostic.get('schema') != 'world-reward-shared-affine-depth-alignment-v1'
                or diagnostic.get('shared_scale') != selected or diagnostic.get('frame_offsets') != offsets
                or diagnostic.get('alpha_only_scale') != baseline or diagnostic.get('heldout_pixels_disjoint') is not True):
            raise ValueError('Selected affine coefficients differ from the frozen public fit')
    else:
        raise ValueError('Explicit public-heldout affine or alpha-only selection required')
    return float(baseline), float(selected), [float(b) for b in offsets]


def aligned_public(base,values):
    out=base/'alignment_v1';path=out/'report.json';digest=fit.observations.depth_model.identity(path)['sha256'];r=json.loads(path.read_text())
    expected={'stage':fit.STAGE,'status':'pass','phase':'complete','private_truth_read':False,
        'ground_truth_used':False,'challenge_inputs_used':False,'hand_labeled_test':False,'oracle_modes':[],
        'outputs_completed':18,'frames':18,'adoption_performed':False,'human_geometry_modified':False,
        'canonical_object_mesh_modified':False,'camera_modified':False,'source_validity_modified':False,'network':'none'}
    if (any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in expected.items()) or len(r.get('outputs',[]))!=18
            or len(r.get('clips',[]))!=3 or len(values)!=18
            or not re.fullmatch('[0-9a-f]{40}',str(r.get('producer_revision','')))
            or not re.fullmatch('sha256:[0-9a-f]{64}',str(r.get('image_id','')))
            or r.get('script_sha256')!=sha256(Path(fit.__file__))):
        raise ValueError('Complete paired public-only alignment receipt required')
    all_coefficients=[coefficients(clip,index) for index,clip in enumerate(r['clips'])]
    arrays=[];frozen=[(path,digest)]
    for index,(value,row) in enumerate(zip(values,r['outputs'])):
        clip,frame=divmod(index,6);name=f'clip_{clip:02d}_frame_{frame:03d}.npz'
        if (row.get('file')!=name or type(row.get('clip_index')) is not int or row.get('clip_index')!=clip
                or type(row.get('frame_index')) is not int or row.get('frame_index')!=frame):raise ValueError('All original fitted records required')
        p=out/name;metric.require_hash(p,row.get('sha256'));frozen.append((p,row['sha256']))
        with np.load(p,allow_pickle=False) as file:data={k:file[k].copy() for k in file.files}
        if (set(data)!=fit.PACKED or data['K'].dtype!=np.float64 or not np.array_equal(data['K'],fit.K)
                or data['validity'].dtype!=np.bool_ or not np.array_equal(data['validity'],value['validity'])):
            raise ValueError('No camera/support changes are permitted between paired methods')
        for key,v in [('clip_index',clip),('frame_index',frame)]:
            if data[key].dtype!=np.int64 or data[key].shape!=() or int(data[key])!=v:raise ValueError('Original fitted identity mismatch')
        for key in ['baseline_points','selected_points']:
            points=data[key]
            if points.shape!=(fit.HEIGHT,fit.WIDTH,3) or points.dtype!=np.float32:raise ValueError('Original pointmap shape/dtype required')
            validate_camera_pointmap(points[...,2],points,data['validity'],np.diag([1/fit.WIDTH,1/fit.HEIGHT,1.])@fit.K,fit.K)
        alpha,selected,offsets=all_coefficients[clip]
        for key,a,b in [('baseline_points',alpha,0.),('selected_points',selected,offsets[frame])]:
            replay=fit.unproject(value['raw_depth'],value['validity'],a,b,value['K'])
            if not np.array_equal(data[key],replay,equal_nan=True):
                raise ValueError('Frozen pointmap does not apply its declared public coefficients exactly once')
        arrays.append(data)
    if {p.name for p in out.iterdir()}!={'report.json',*[r['file'] for r in r['outputs']]}:raise ValueError('Unexpected fitted files')
    return arrays,r,frozen


def decision(scenes,coverage):
    if len(scenes)!=3:raise ValueError('Every predeclared clip required')
    before=np.array([s['baseline_visible_chamfer_half_cm'] for s in scenes]);after=np.array([s['selected_visible_chamfer_half_cm'] for s in scenes])
    if not np.isfinite(np.r_[before,after]).all() or np.any(before<=0) or np.any(after<0):raise ValueError('Finite complete positive-baseline metrics required')
    gain=(before-after)/before
    return {'median_paired_clip_relative_gain':float(np.median(gain)),'per_clip_relative_gain':gain.tolist(),
        'coverage_gate_pass':bool(coverage),'synthetic_affine_grounding_hypothesis_supported':bool(coverage and np.median(gain)>=.05 and gain.min()>=-.05),
        'adoption_performed':False,'full_v2d_score_verified':False,'full_object_geometry_quality_verified':False}


def run(root,report):
    base=root/'validation/joint_affine_rgb_v1'
    values,records,hashes,frozen=fit.frozen_observations(root)
    paired,alignment,more=aligned_public(base,values);frozen+=more
    if alignment.get('observation_report_sha256')!=frozen[0][1] or alignment.get('public_hashes')!=hashes:
        raise ValueError('Alignment does not address this frozen inference cohort')
    # First evaluation-private read occurs only after all eighteen public pairs.
    private=base/'eval_private';rp=private/'render-report.json';rh=fit.observations.depth_model.identity(rp)['sha256'];r=json.loads(rp.read_text())
    if (r.get('stage')!='own_joint_affine_human_object_rgb_render' or r.get('status')!='pass'
            or type(r.get('frames')) is not int or r.get('frames')!=18 or r.get('phase')!='complete'
            or r.get('public_manifest_sha256')!=sha256(base/'inputs/manifest.json')
            or r.get('challenge_inputs_used') is not False or r.get('inference_performed') is not False
            or r.get('network')!='none' or r.get('all_truth_private') is not True
            or r.get('generated_GT_masks_supplied_to_inference') is not False
            or not re.fullmatch('[0-9a-f]{40}',str(r.get('code_revision','')))
            or not re.fullmatch('sha256:[0-9a-f]{64}',str(r.get('image_id','')))
            or len(r.get('cases',[]))!=18):raise ValueError('Independent complete new private render provenance required')
    report.update(phase='private_quality',public_frozen_before_private=True,observation_report_sha256=frozen[0][1],
        alignment_report_sha256=more[0][1],private_render_report_sha256=rh,frames=[])
    frozen.append((rp,rh));coverage=True;allmetrics=[]
    truth_keys={'human_vertices_camera_m','human_faces','object_vertices_camera_m','object_faces','camera_K',
                'scene_depth_m','visible_face_indices','clip_index','frame_index'}
    for i,(value,data,record,case) in enumerate(zip(values,paired,records,r['cases'])):
        clip,frame=divmod(i,6);stem=f'clip_{clip:02d}_frame_{frame:03d}'
        if case.get('file')!=stem+'.png' or case.get('rgb_sha256')!=record['rgb_sha256'] or case.get('clip_index')!=clip or case.get('frame_index')!=frame:
            raise ValueError('Private original record identity differs')
        p=private/(stem+'.npz');metric.require_hash(p,case.get('truth_sha256'));frozen.append((p,case['truth_sha256']))
        with np.load(p,allow_pickle=False) as file:truth={k:file[k].copy() for k in file.files}
        if (set(truth)!=truth_keys or any(truth[k].dtype!=np.int64 or truth[k].shape!=() or int(truth[k])!=v
                for k,v in [('clip_index',clip),('frame_index',frame)])
                or truth['scene_depth_m'].shape!=(fit.HEIGHT,fit.WIDTH)
                or truth['visible_face_indices'].shape!=(fit.HEIGHT,fit.WIDTH)):
            raise ValueError('Exact original-grid new private truth arrays required')
        human=metric.points(truth['human_vertices_camera_m']);ov=metric.points(truth['object_vertices_camera_m'])
        hf=metric.faces(truth['human_faces'],len(human));of=metric.faces(truth['object_faces'],len(ov))
        truepoints,truecount=metric.visible_object_points(truth['scene_depth_m'],truth['visible_face_indices'],truth['camera_K'],len(hf),len(of))
        valid=data['validity'];mask=value['object_mask'];selection=valid&mask
        true_visible=truth['visible_face_indices']>=len(hf)
        pixel_coverage=float(np.count_nonzero(selection&true_visible)/truecount)
        coverage &= pixel_coverage>=.95
        if not selection.any():raise ValueError('No automatically predicted object coverage')
        before=metric.sample_points(data['baseline_points'][selection]);after=metric.sample_points(data['selected_points'][selection])
        # Index sampling is identical because validity/object support are identical.
        row={'clip_index':clip,'frame_index':frame,'visible_object_pixels':truecount,'predicted_object_pixels':int(selection.sum()),
             'true_visible_object_pixel_coverage':pixel_coverage,'baseline':metric.metrics(before,truepoints,value['human_vertices_camera_m'],human),
             'selected':metric.metrics(after,truepoints,value['human_vertices_camera_m'],human)}
        report['frames'].append(row);allmetrics.append(row)
    scenes=[]
    for clip in range(3):
        rows=allmetrics[clip*6:(clip+1)*6]
        scenes.append({'clip_index':clip,'baseline_visible_chamfer_half_cm':float(np.mean([r['baseline']['visible_chamfer_half_cm'] for r in rows])),
            'selected_visible_chamfer_half_cm':float(np.mean([r['selected']['visible_chamfer_half_cm'] for r in rows])),
            'baseline_relative_centroid_vector_error_cm':float(np.mean([r['baseline']['relative_visible_centroid_vector_error_cm'] for r in rows])),
            'selected_relative_centroid_vector_error_cm':float(np.mean([r['selected']['relative_visible_centroid_vector_error_cm'] for r in rows]))})
    if any(p.is_symlink() or sha256(p)!=h for p,h in frozen) or fit.observations.public_inputs(root)!=(records,hashes):
        raise ValueError('Frozen inference/fit/private/RGB source changed during quality')
    report.update(status='pass',phase='complete',clips=scenes,decision=decision(scenes,coverage))


def main(argv=None):
    argparse.ArgumentParser(allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:raise RuntimeError('Remote offline CPU quality required')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/joint_affine_rgb_v1/quality_v1'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Exclusive new quality directory required')
    if (not re.fullmatch('[0-9a-f]{40}',os.environ['WR_CODE_REVISION'])
            or not re.fullmatch('sha256:[0-9a-f]{64}',os.environ['WR_IMAGE_ID'])):
        raise ValueError('Immutable quality source/image required')
    report={'stage':'private_joint_affine_depth_grounding_quality','status':'fail','phase':'integrity',
        'producer_revision':os.environ['WR_CODE_REVISION'],'image_id':os.environ['WR_IMAGE_ID'],'script_sha256':sha256(Path(__file__)),
        'budget_seconds':BUDGET,'external_private_synthetic_truth_used':True,'challenge_inputs_used':False,'adoption_performed':False,
        'full_v2d_score_verified':False,'evaluation_alignment_performed':False,'GT_camera_used_for_truth_unprojection_only':True,
        'predicted_camera_changed':False,'methods_share_identical_predicted_validity_and_object_masks':True}
    start=time.perf_counter()
    def expired(*_):raise TimeoutError('Whole private J3 quality exceeded90s')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
    try:run(root,report)
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.perf_counter()-start
        with (out/'report.json').open('x') as f:json.dump(report,f,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['stage','status','decision','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
