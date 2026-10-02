"""Paired numeric sensor tests and complete public-before-private firewall."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest
from test_tudl_evaluate import evaluate, frozen, write_json


@pytest.fixture
def depth_evaluator(evaluate,monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'tudl_evaluate',evaluate)
    path=Path(__file__).resolve().parents[1]/'infra/da3_metric_evaluate.py'
    spec=importlib.util.spec_from_file_location('paired_depth_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'prior',evaluate)
    monkeypatch.setattr(module,'FIXED_FOCAL',10.)
    return module


@pytest.fixture
def paired(depth_evaluator,frozen):
    root,data,baseline,acquisition,sync,ap=frozen
    base=root/'validation/tudl_rgb_v1';out=base/'da3_predictions_v1';out.mkdir()
    outputs=[];candidates=[]
    # Preserve prior's actual producer validation on tiny K10 own arrays.
    # Only the candidate's fixed focal constant is scaled for this tiny test.
    for record,arrays in zip(baseline['public_records'],data):
        candidate={k:arrays['learned_'+k].copy() for k in ('depth','points','validity','K')}
        candidate.update(scene_id=arrays['scene_id'].copy(),frame_id=arrays['frame_id'].copy())
        name=Path(record['file']).stem+'.npz';path=out/name;np.savez(path,**candidate);candidates.append(candidate)
        outputs.append({'file':name,'scene_id':record['scene_id'],'frame_id':record['frame_id'],
                        'rgb_sha256':record['sha256'],'sha256':depth_evaluator.prior.sha256(path)})
    kp=depth_evaluator.NOMINAL_K
    report=dict(stage=depth_evaluator.STAGE,status='pass',phase='complete',network='none',producer_revision='a'*40,
                image_id='sha256:'+'b'*64,private_truth_read=False,challenge_inputs_used=False,oracle_modes=[],scale_fit=False,
                ground_truth_used=False,hand_labeled_test=False,actual_DA3_inference=True,checkpoint_states_loaded=406,
                checkpoint_load_strict=True,native_sky_correction_unchanged=True,native_sky_threshold=.3,native_sky_quantile=.99,
                asset_source_reverified_after_inference=True,RNG_seed_reset_before_each_forward=True,
                confidence_validity_exclusion=False,pointmap_geometry_filled=False,public_records=baseline['public_records'],
                native_forward_arguments=dict(extrinsics=None,intrinsics=None,export_feat_layers=[],infer_gs=False,use_ray_pose=False,ref_view_strategy='saddle_balanced'),
                adoption_performed=False,original_frame_coverage_verified=True,DA3_calls_completed=9,outputs_completed=9,
                metric_depth_scaling_applied_once=True,source_revision=depth_evaluator.MODEL_SOURCE,
                model_revision=depth_evaluator.MODEL_REV,model_sha256=depth_evaluator.MODEL_SHA,
                processed_grid=[392,518],processed_camera_K=kp.tolist(),metric_depth_factor=float((kp[0,0]+kp[1,1])/600),
                public_input_manifest_sha256=depth_evaluator.prior.sha256(base/'inputs/manifest.json'),
                source_manifest_sha256='c'*64,acquisition_report={'sha256':'d'*64},outputs=outputs)
    path=out/'report.json'
    def update():
        for row,arrays in zip(outputs,candidates):
            p=out/row['file'];np.savez(p,**arrays);row['sha256']=depth_evaluator.prior.sha256(p)
        write_json(path,report)
    update()
    return root,report,candidates,update


def test_complete_paired_numeric_flow_sameK_no_private_ray_fitting(depth_evaluator,paired):
    root,_,_,_=paired;report={};depth_evaluator.run(root,report)
    assert report['status']=='pass' and len(report['frames'])==9
    assert report['candidate_and_baseline_frozen_before_private']
    assert report['decision']['real_object_depth_hypothesis_supported']
    assert report['decision']['median_paired_scene_relative_gain']>.99999
    assert not report['alignment_performed'] and not report['predicted_XYZ_camera_or_scale_changed']
    assert not report['human_quality_verified'] and not report['full_v2d_score_verified']


@pytest.mark.parametrize('fault',['private','partial','scale','processedK','model','source','manifest','SHA','missing','validity','NaN','K','indices','extra'])
def test_bad_candidate_rejected_before_any_private_annotation(depth_evaluator,paired,monkeypatch,fault):
    root,receipt,arrays,sync=paired
    if fault=='private':receipt['private_truth_read']=True
    elif fault=='partial':receipt['DA3_calls_completed']=8
    elif fault=='scale':receipt['metric_depth_factor']=800/300
    elif fault=='processedK':receipt['processed_camera_K'][0][0]=800.
    elif fault=='model':receipt['model_revision']='c'*40
    elif fault=='source':receipt['source_manifest_sha256']='bad'
    elif fault=='manifest':receipt['public_input_manifest_sha256']='c'*64
    elif fault=='SHA':receipt['outputs'][-1]['rgb_sha256']='c'*64
    elif fault=='missing':receipt['outputs']=receipt['outputs'][:-1]
    elif fault=='validity':arrays[-1]['validity'][0,0]=False
    elif fault=='NaN':arrays[-1]['points'][0,0,0]=np.nan
    elif fault=='K':arrays[-1]['K'][0,0]=700.
    elif fault=='indices':arrays[-1]['frame_id']=np.array(0,np.int64)
    else:arrays[-1]['gt_camera']=np.eye(3)
    sync();original=depth_evaluator.prior.regular_hash
    def guard(p,d=None):
        assert 'eval_private' not in str(p),'Private truth read before all predictions checked'
        return original(p,d)
    monkeypatch.setattr(depth_evaluator.prior,'regular_hash',guard)
    with pytest.raises(ValueError):depth_evaluator.run(root,{})


def test_wrapper_private_CPU_only_no_models_or_parentmount():
    path=Path(__file__).resolve().parents[1]/'infra/run_da3_metric_evaluate.sh';text=path.read_text()
    assert '--gpus' not in text and 'weights' not in text and '--network none' in text
    assert 'src=$BASE,dst=$BASE' not in text and 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in text
    subprocess.run(['bash','-n',str(path)],check=True)
