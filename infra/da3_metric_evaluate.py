"""Paired private RGB-D depth evaluation, only after both RGB predictions freeze.

Compare old MoGe fixed-K800 to DA3 depth at identical original K/rays. Sensor
truth is private external TUD-L only. No alignment, GT ray/scale calibration,
sample dropping, human/HOI score or challenge-performance claim.
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
import tudl_evaluate as prior
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

MODEL_SOURCE='3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
MODEL_REV='4010e39f3634a45bc60553321fb49fb760bd594e'
MODEL_SHA='bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776'
STAGE='public_tudl_rgb_da3_metric_depth_predictions'
ARRAY_KEYS={'depth','points','validity','K','scene_id','frame_id'}
NOMINAL_K=np.array([[647.5,0.,259.],[0.,653.3333333333334,196.],[0.,0.,1.]])
BUDGET=90
FIXED_FOCAL=800.


def public_da3(base, records):
    """Entire nine-image candidate hash/geometry/counters before private truth."""
    output=base/'da3_predictions_v1'; rp=output/'report.json'
    rh=prior.regular_hash(rp); report=json.loads(rp.read_text())
    expected={'stage':STAGE,'status':'pass','phase':'complete','network':'none',
              'private_truth_read':False,'challenge_inputs_used':False,'oracle_modes':[],
              'ground_truth_used':False,'hand_labeled_test':False,'actual_DA3_inference':True,
              'scale_fit':False,'adoption_performed':False,'original_frame_coverage_verified':True,
              'DA3_calls_completed':9,'outputs_completed':9,'metric_depth_scaling_applied_once':True,
              'checkpoint_states_loaded':406,'checkpoint_load_strict':True,
              'native_sky_correction_unchanged':True,'native_sky_threshold':.3,'native_sky_quantile':.99,
              'confidence_validity_exclusion':False,'pointmap_geometry_filled':False,
              'native_forward_arguments':dict(extrinsics=None,intrinsics=None,export_feat_layers=[],infer_gs=False,use_ray_pose=False,ref_view_strategy='saddle_balanced'),
              'source_revision':MODEL_SOURCE,'model_revision':MODEL_REV,'model_sha256':MODEL_SHA,
              'processed_grid':[392,518]}
    if not isinstance(report,dict) or any(type(report.get(k)) is not type(v) or report[k]!=v for k,v in expected.items()):
        raise ValueError('Complete pinned GT-free DA3 prediction receipt required')
    if not re.fullmatch('[0-9a-f]{40}',report.get('producer_revision','')) or not re.fullmatch('sha256:[0-9a-f]{64}',report.get('image_id','')):
        raise ValueError('Immutable actual candidate source/image required')
    if (report.get('public_input_manifest_sha256')!=prior.regular_hash(base/'inputs/manifest.json')
            or report.get('public_records')!=records
            or not re.fullmatch('[0-9a-f]{64}',report.get('source_manifest_sha256',''))
            or not re.fullmatch('[0-9a-f]{64}',report.get('acquisition_report',{}).get('sha256',''))):
        raise ValueError('Candidate public/model source identity unbound')
    Kp=prior.camera(np.asarray(report.get('processed_camera_K'),dtype=np.float64))
    factor=report.get('metric_depth_factor')
    if (not np.allclose(Kp,NOMINAL_K,atol=1e-3,rtol=0) or type(factor) not in (int,float)
            or not np.isfinite(factor) or abs(factor-(Kp[0,0]+Kp[1,1])/600)>1e-12):
        raise ValueError('Native canonical depth must scale once by actual processed mean focal/300')
    outputs=report.get('outputs')
    if not isinstance(outputs,list) or len(outputs)!=9:raise ValueError('All candidate outputs required')
    frozen=[(rp,rh)]; arrays=[]
    for record,row,(scene,frame) in zip(records,outputs,prior.FRAMES):
        stem=f'scene_{scene:06d}_frame_{frame:06d}'
        if (not isinstance(row,dict) or type(row.get('scene_id')) is not int or row['scene_id']!=scene
                or type(row.get('frame_id')) is not int or row['frame_id']!=frame
                or row.get('file')!=stem+'.npz' or row.get('rgb_sha256')!=record['sha256']):
            raise ValueError('Candidate original ordered RGB/frame identity differs')
        path=output/row['file']; digest=prior.regular_hash(path,row.get('sha256'));frozen.append((path,digest))
        with np.load(path,allow_pickle=False) as file:data={k:file[k].copy() for k in file.files}
        if set(data)!=ARRAY_KEYS:raise ValueError('Exact depth-only candidate arrays required')
        for key,value in (('scene_id',scene),('frame_id',frame)):
            if data[key].shape!=() or data[key].dtype!=np.int64 or int(data[key])!=value:raise ValueError('Original scalar identity required')
        z,p,v,K=(data[k] for k in ('depth','points','validity','K'))
        expectedK=np.array([[FIXED_FOCAL,0.,prior.WIDTH/2],[0.,FIXED_FOCAL,prior.HEIGHT/2],[0.,0.,1.]])
        if (z.dtype!=np.float32 or p.dtype!=np.float32 or v.dtype!=np.bool_ or K.dtype not in (np.float32,np.float64)
                or z.shape!=(prior.HEIGHT,prior.WIDTH) or not v.all() or not np.array_equal(K,expectedK)):
            raise ValueError('Full original positive depth and same fixed camera required, no validity filtering')
        validate_camera_pointmap(z,p,v,np.diag([1/prior.WIDTH,1/prior.HEIGHT,1.])@K,K)
        arrays.append(data)
    if {p.name for p in output.iterdir()}!={'report.json',*[r['file'] for r in outputs]}:raise ValueError('Unexpected candidate artifacts')
    return arrays,frozen,{'DA3_prediction_report_sha256':rh,'DA3_image_id':report['image_id']}


def paired_predictions(base):
    baseline,records,frozen,hashes=prior.public_predictions(base)
    candidate,newfrozen,newhashes=public_da3(base,records)
    pairs=[]
    for before,after in zip(baseline,candidate):
        if not np.array_equal(before['fixed_K'],after['K']):raise ValueError('Depth methods must use identical original RGB-only K')
        pairs.append({**{k:before[k] for k in before if k.startswith('fixed_')},
                      **{'learned_'+k:after[k] for k in ('depth','points','validity','K')}})
    return pairs,records,frozen+newfrozen,{**hashes,**newhashes}


def run(root,report):
    # Reuse exactly the frozen real-sensor evaluator and coverage/decision gates.
    # Explicit loader injection avoids global hooks and leaves default intact.
    prior.run(root,report,prediction_loader=paired_predictions)
    report.update(candidate_and_baseline_frozen_before_private=True,comparison='MoGe2_fixedK800_vs_DA3Metric_sameK800',
                  camera_hypothesis_tested=False,depth_hypothesis_tested=True,
                  full_v2d_score_verified=False,
                  candidate_pointmap_native_DA3=False,candidate_XYZ_from_depth_with_shared_original_RGB_K=True)
    report['decision']['real_object_depth_hypothesis_supported']=report['decision'].pop('real_object_camera_hypothesis_supported')


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!='Linux' or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError('Remote network-none CPU evaluation required')
    root=Path(os.environ['WR_ROOT']);out=root/'validation/tudl_rgb_v1/da3_quality_v1'
    if out.is_symlink() or not out.is_dir() or any(out.iterdir()):raise FileExistsError('Exclusive fresh quality directory required')
    revision,image=os.environ['WR_CODE_REVISION'],os.environ['WR_IMAGE_ID']
    if not re.fullmatch('[0-9a-f]{40}',revision) or not re.fullmatch('sha256:[0-9a-f]{64}',image):raise ValueError('Immutable evaluator source/image required')
    report={'stage':'private_tudl_real_metric_depth_quality','status':'fail','phase':'integrity',
            'producer_revision':revision,'image_id':image,'script_sha256':sha256(Path(__file__)),
            'external_private_sensor_truth_used':True,'challenge_inputs_used':False,'adoption_performed':False,
            'full_v2d_score_verified':False,'budget_seconds':BUDGET}
    start=time.perf_counter()
    def expired(*_):raise TimeoutError('Paired metric-depth evaluation exceeded90s')
    alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
    try:run(root,report)
    except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term)
        report['elapsed_seconds']=time.perf_counter()-start
        with (out/'report.json').open('x') as f:json.dump(report,f,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('stage','status','decision','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
