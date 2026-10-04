"""Independent analytical CPU fixtures; no real labels, files, models or cloud."""
import copy

import numpy as np
import pytest

from world_reward.point_motion_evaluation import (evaluate_sequence,cohort_gate,CHUNK_POINTS)


def poses(translation=None,rotation=None,frames=3):
    result=np.tile(np.eye(4),(frames,1,1))
    if translation is not None:result[:,:3,3]=translation
    if rotation is not None:result[:,:3,:3]=rotation
    return result


def rz(angle):
    c,s=np.cos(angle),np.sin(angle);return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def case(height=3,width=4):
    return dict(frame_indices=np.arange(3),baseline_poses=poses(),candidate_poses=poses(),gt_poses=poses()[None],
        automatic_initial_mask=np.ones((height,width),bool),gt_initial_masks=np.ones((1,height,width),bool),
        depth_z_metres=np.ones((height,width)),private_K=np.array([[2.,0.,1.],[0.,3.,1.],[0.,0.,1.]]),sequence_id='synthetic_a')


def test_exact_motion_zero_all_original_frames_and_no_private_geometry_return():
    value=case();result=evaluate_sequence(**value)
    assert result['candidate']['displacement_m']==[0.,0.,0.]and result['candidate']['rotation_degrees']==[0.,0.,0.]
    assert result['frame_indices']==[0,1,2]and result['coverage']==1. and result['initial_material_points']==12
    assert result['truth_pixel_convention']=='integer_BOP'and result['reacquisition_supported']is False
    assert not any(isinstance(v,np.ndarray)for v in result.values())
    assert not any(k in result for k in('points','GT','poses','private_K','depth_z_metres'))


def test_translation_error_analytic_uses_original_frame0_relative_motion():
    value=case();gt_t=np.array([[5.,2.,3.],[5.3,2.4,3.],[5.6,2.8,3.]])
    value['gt_poses']=poses(gt_t)[None];value['candidate_poses']=poses(np.array([[99.,-20.,7.],[99.3,-19.6,7.],[99.6,-19.2,7.]]))
    report=evaluate_sequence(**value)
    assert report['candidate']['displacement_m']==pytest.approx([0.,0.,0.],abs=1e-13)
    assert report['baseline']['displacement_m']==pytest.approx([0.,.5,1.])
    assert report['baseline']['mean_displacement_m']==pytest.approx(.5)


def test_rotation_error_and_integer_sensor_backprojection_analytic():
    value=case();value['gt_poses']=poses(rotation=np.stack([np.eye(3),rz(np.pi/2),rz(np.pi)]))[None]
    report=evaluate_sequence(**value)
    yy,xx=np.indices((3,4));radius=np.sqrt(((xx-1)/2)**2+((yy-1)/3)**2)
    expected=[0.,float(np.sqrt(2)*radius.mean()),float(2*radius.mean())]
    assert report['baseline']['displacement_m']==pytest.approx(expected)
    assert report['baseline']['rotation_degrees']==pytest.approx([0.,90.,180.])


def test_unique_mask_match_not_minimum_pose_error():
    value=case();value['automatic_initial_mask'][:]=False;value['automatic_initial_mask'][:2]=True
    masks=np.zeros((2,3,4),bool);masks[0,:2]=True;masks[1,2]=True;value['gt_initial_masks']=masks
    moving=poses(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]));value['gt_poses']=np.stack([moving,poses()])
    report=evaluate_sequence(**value)
    assert report['matched_gt_mask_index']==0 and report['baseline']['displacement_m']==pytest.approx([0.,1.,2.])
    # Reordered instances preserve the mask choice, not a hardcoded instance0.
    value['gt_initial_masks']=masks[::-1];value['gt_poses']=value['gt_poses'][::-1]
    report=evaluate_sequence(**value);assert report['matched_gt_mask_index']==1
    assert report['baseline']['displacement_m']==pytest.approx([0.,1.,2.])


@pytest.mark.parametrize('fault',['emptyauto','nomatch','multiple','invaliddepth_denominator','under8'])
def test_initial_matching_or_material_support_fails(fault):
    value=case()
    if fault=='emptyauto':value['automatic_initial_mask'][:]=False
    elif fault=='nomatch':value['gt_initial_masks'][0,0,0]=False
    elif fault=='multiple':value['gt_initial_masks']=np.repeat(value['gt_initial_masks'],2,axis=0);value['gt_poses']=np.repeat(value['gt_poses'],2,axis=0)
    elif fault=='invaliddepth_denominator':
        value['gt_initial_masks'][0,0,0]=False;value['depth_z_metres'][0,0]=np.nan
    else:value['depth_z_metres'].flat[:5]=0.
    with pytest.raises(ValueError):evaluate_sequence(**value)


def test_fixed95_percent_boundary_and_all_valid_visible_pixels_not_auto_subset():
    value=case(5,4);value['gt_initial_masks'][0,0,0]=False
    report=evaluate_sequence(**value);assert report['initial_mask_purity']==.95 and report['initial_material_points']==19
    value['automatic_initial_mask'][0]=False;value['gt_initial_masks'][:]=True
    value['depth_z_metres'][0,0]=np.nan;value['depth_z_metres'][0,1]=-1.
    report=evaluate_sequence(**value);assert report['initial_material_points']==18


@pytest.mark.parametrize('target',['baseline_poses','candidate_poses','gt_poses'])
@pytest.mark.parametrize('fault',['nan','scaled','reflection','lastrow','missingframe'])
def test_every_proper_full_trajectory_required(target,fault):
    value=case();array=value[target]
    flat=array.reshape(-1,4,4)
    if fault=='nan':flat[1,0,3]=np.nan
    elif fault=='scaled':flat[1,:3,:3]*=1.01
    elif fault=='reflection':flat[1,0,0]=-1.
    elif fault=='lastrow':flat[1,3,0]=1e-9
    else:value[target]=array[:,:2]if target=='gt_poses'else array[:2]
    with pytest.raises(ValueError):evaluate_sequence(**value)


@pytest.mark.parametrize('indices',[np.arange(2),np.array([0,2,3]),np.array([0,1,1]),np.array([1,2,3]),np.array([0.,1.,2.]),np.array([False,True,True])])
def test_no_frame_deletion_reindex_or_float_indices(indices):
    value=case();value['frame_indices']=indices
    with pytest.raises(ValueError):evaluate_sequence(**value)


@pytest.mark.parametrize('fault',['K_timevarying','K_negative','K_scaled','K_skew','depth_integer','mask_integer','depth_masked'])
def test_sensor_units_grid_and_fixed_K_strict(fault):
    value=case()
    if fault=='K_timevarying':value['private_K']=np.repeat(value['private_K'][None],3,axis=0)
    elif fault=='K_negative':value['private_K'][0,0]=-2.
    elif fault=='K_scaled':value['private_K']*=2.
    elif fault=='K_skew':value['private_K'][0,1]=.1
    elif fault=='depth_integer':value['depth_z_metres']=np.ones((3,4),np.uint16)
    elif fault=='mask_integer':value['automatic_initial_mask']=np.ones((3,4),np.uint8)
    else:value['depth_z_metres']=np.ma.array(value['depth_z_metres'])
    with pytest.raises(ValueError):evaluate_sequence(**value)


def test_chunks_use_every_pixel_equal_weight_and_do_not_mutate_inputs():
    value=case(73,61);assert value['depth_z_metres'].size>CHUNK_POINTS
    value['depth_z_metres']=np.linspace(.1,3.,73*61).reshape(73,61)
    rotations=np.stack([np.eye(3),rz(.4),rz(.8)]);value['gt_poses']=poses(rotation=rotations)[None]
    before={k:v.copy()for k,v in value.items()if isinstance(v,np.ndarray)}
    result=evaluate_sequence(**value);yy,xx=np.indices((73,61))
    radius=np.hypot((xx-1)*value['depth_z_metres']/2,(yy-1)*value['depth_z_metres']/3)
    assert result['baseline']['displacement_m']==pytest.approx([0.,2*np.sin(.2)*radius.mean(),2*np.sin(.4)*radius.mean()])
    assert result['initial_material_points']==73*61
    for k,v in before.items():np.testing.assert_array_equal(value[k],v)


def gain_case(gain,identity):
    value=case();value['sequence_id']=identity
    truth=poses(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]));value['gt_poses']=truth[None]
    value['candidate_poses']=poses(np.array([[0.,0.,0.],[gain,0.,0.],[2*gain,0.,0.]]))
    return evaluate_sequence(**value)


def test_exact3_gate_accepts10_percent_median_and_no_more5_percent_regression():
    rows=[gain_case(.10,'a'),gain_case(.10,'b'),gain_case(-.05,'c')]
    gate=cohort_gate(rows);assert gate['status']=='pass'and gate['coverage']==1.
    assert gate['median_relative_mean_error_gain']==pytest.approx(.10)
    assert gate['reacquisition_supported']is False and gate['challenge_accuracy_verified']is False


@pytest.mark.parametrize('gains',[(.09,.09,.8),(.2,.2,-.051)])
def test_predeclared_accuracy_gate_failures(gains):
    gate=cohort_gate([gain_case(g,name)for g,name in zip(gains,'abc')])
    assert gate['status']=='fail'


def test_zero_baseline_no_identified_improvement_even_zero_candidate():
    rows=[evaluate_sequence(**{**case(),'sequence_id':name})for name in 'abc']
    gate=cohort_gate(rows);assert gate['status']=='fail'and gate['relative_mean_error_gains']==[None]*3
    assert gate['median_relative_mean_error_gain']is None and 'zero_baseline_gain_unidentified'in gate['reasons']


@pytest.mark.parametrize('fault',['two','four','duplicate','coverage','missingerror','tamperedmean','nan'])
def test_cohort_cannot_filter_repeat_or_spoof_records(fault):
    rows=[gain_case(.2,name)for name in 'abc']
    if fault=='two':rows.pop()
    elif fault=='four':rows.append(gain_case(.2,'d'))
    elif fault=='duplicate':rows[2]=copy.deepcopy(rows[0])
    elif fault=='coverage':rows[1]['coverage']=.5
    elif fault=='missingerror':rows[1]['candidate']['displacement_m'].pop()
    elif fault=='tamperedmean':rows[1]['candidate']['mean_displacement_m']=0.
    else:rows[1]['candidate']['displacement_m'][1]=float('nan')
    with pytest.raises(ValueError):cohort_gate(rows)


def test_nonzero_initial_pose_uses_SE3_inverse_not_rotation_translation_subtraction():
    value=case();initial=poses()[0];initial[:3,:3]=rz(np.pi/2);initial[:3,3]=[3.,-2.,1.]
    motion=poses(np.array([[0.,0.,0.],[.3,.1,.2],[.5,-.2,.4]]),rotation=np.stack([np.eye(3),rz(.2),rz(.7)]))
    gt=np.stack([m@initial for m in motion]);value['gt_poses']=gt[None]
    other=initial.copy();other[:3,:3]=rz(-.8);other[:3,3]=[-7.,5.,4.]
    value['candidate_poses']=np.stack([m@other for m in motion])
    result=evaluate_sequence(**value)
    assert result['candidate']['displacement_m']==pytest.approx([0.,0.,0.],abs=2e-14)
    assert result['candidate']['rotation_degrees']==pytest.approx([0.,0.,0.],abs=2e-6)
    yy,xx=np.indices((3,4));points=np.column_stack(((xx.ravel()-1)/2,(yy.ravel()-1)/3,np.ones(12)))
    expected=[]
    for m in motion:
        moved=points@m[:3,:3].T+m[:3,3]
        expected.append(float(np.linalg.norm(moved-points,axis=1).mean()))
    assert result['baseline']['displacement_m']==pytest.approx(expected)


def test_no_scale_rescue_of_smaller_candidate_motion():
    row=gain_case(.5,'scale_not_fitted')
    assert row['candidate']['displacement_m']==pytest.approx([0.,.5,1.])
    assert row['candidate']['mean_displacement_m']==pytest.approx(.5)and row['scale_fitted']is False


def test_unmatched_instance_improper_pose_not_silently_ignored():
    value=case();value['gt_initial_masks']=np.concatenate([value['gt_initial_masks'],np.zeros_like(value['gt_initial_masks'])])
    value['gt_poses']=np.repeat(value['gt_poses'],2,axis=0);value['gt_poses'][1,2,0,0]=-1.
    with pytest.raises(ValueError):evaluate_sequence(**value)


def test_gate_tolerance_is_only_single_arithmetic_ULP_not_threshold_retuning():
    rows=[gain_case(.2,name)for name in 'abc'];rows[2]=gain_case(-.050000001,'c')
    assert cohort_gate(rows)['status']=='fail'
    rows=[gain_case(.099999999,name)for name in 'abc'];assert cohort_gate(rows)['status']=='fail'
