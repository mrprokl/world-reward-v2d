"""New tiny CPU cohort; synthetic truth stays outside the solver API."""
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation,Slerp

from world_reward.occlusion_bridge import (ANCHORS_PER_SIDE,MAX_ROTATION_DISPERSION_RAD,
    MAX_TRANSLATION_DISPERSION_M,initialize_occluded_gaps)

SEED=310427
FRAMES=48
SCENARIOS=('carry_accelerating','carry_curved','free','slip','regrasp','noisy_hand')
GAP_LENGTHS=(3,12)


def new_cohort():
    """Twelve predeclared new pose-observation fixtures, not RGB/model validation."""
    rng=np.random.default_rng(SEED);u=np.arange(FRAMES)/47.
    hand_r=Rotation.from_rotvec(np.column_stack((.2*np.sin(2.7*u),.8*u*u,.12*np.cos(3.2*u)))).as_matrix()
    hand_t=np.column_stack((.35*u*u,.10*np.sin(4*u),2.+.12*np.cos(2.6*u)))
    relative_r=Rotation.from_rotvec([.13,-.08,.09]).as_matrix();relative_t=np.array([.07,.02,.04])
    for scenario in SCENARIOS:
        truth_r=hand_r@relative_r;truth_t=np.einsum('tij,j->ti',hand_r,relative_t)+hand_t
        measured_hr=hand_r.copy();measured_ht=hand_t.copy()
        if scenario=='carry_curved':
            measured_ht[:,0]+=.10*np.sin(7*u);measured_ht[:,1]+=.07*np.cos(6*u)
            truth_t=np.einsum('tij,j->ti',hand_r,relative_t)+measured_ht
        elif scenario=='free':
            truth_r=Rotation.from_rotvec(np.column_stack((-.3*u,.15*np.sin(3*u),.5*u))).as_matrix()
            truth_t=np.column_stack((-.3+.22*u,.15*np.cos(4*u),2.2+.07*u*u))
        elif scenario=='slip':
            truth_t[:,0]+=.35*np.clip((u-.25)/.4,0,1)
            truth_r=truth_r@Rotation.from_rotvec(np.column_stack((np.zeros(FRAMES),.7*u,np.zeros(FRAMES)))).as_matrix()
        elif scenario=='regrasp':
            truth_t[u>=.5]+=[.13,-.06,.03]
            truth_r[u>=.5]=truth_r[u>=.5]@Rotation.from_rotvec([0.,.5,.1]).as_matrix()
        elif scenario=='noisy_hand':
            measured_ht+=rng.normal(0,.018,(FRAMES,3))
            measured_hr=measured_hr@Rotation.from_rotvec(rng.normal(0,.035,(FRAMES,3))).as_matrix()
        for length in GAP_LENGTHS:
            first=(FRAMES-length)//2;observed=np.ones(FRAMES,bool);observed[first:first+length]=False
            object_r=truth_r.copy();object_t=truth_t.copy();object_r[~observed]=np.nan;object_t[~observed]=np.nan
            yield dict(name=f'{scenario}_gap{length}',scenario=scenario,length=length,truth_r=truth_r.copy(),
                truth_t=truth_t.copy(),object_r=object_r,object_t=object_t,observed=observed,
                hand_r=measured_hr.copy(),hand_t=measured_ht.copy(),hand_valid=np.ones(FRAMES,bool))


def infer(case):
    return initialize_occluded_gaps(np.arange(FRAMES),case['object_r'],case['object_t'],case['observed'],
        case['hand_r'],case['hand_t'],case['hand_valid'])


def endpoint_baseline(case):
    missing=np.flatnonzero(~case['observed']);left=missing[0]-1;right=missing[-1]+1
    alpha=(missing-left)/(right-left)
    r=Slerp([0.,1.],Rotation.from_matrix(case['object_r'][[left,right]]))(alpha).as_matrix()
    t=(1-alpha[:,None])*case['object_t'][left]+alpha[:,None]*case['object_t'][right]
    return r,t


def pose_surface_error(r,t,truth_r,truth_t):
    # Fixed asymmetric surface witnesses, no geometry fit or evaluation alignment.
    surface=np.array([[.10,.03,.04],[-.08,.02,.01],[.01,.09,-.02],[.04,-.07,.03],[-.02,.01,-.06]])
    predicted=np.einsum('tij,vj->tvi',r,surface)+t[:,None]
    actual=np.einsum('tij,vj->tvi',truth_r,surface)+truth_t[:,None]
    return float(np.mean(np.linalg.norm(predicted-actual,axis=-1)))


def evaluate_frozen_cohort():
    results=[];signatures=[]
    for case in new_cohort():
        result=infer(case);missing=~case['observed'];br,bt=endpoint_baseline(case)
        baseline=pose_surface_error(br,bt,case['truth_r'][missing],case['truth_t'][missing])
        proposed=pose_surface_error(result.rotations[missing],result.translations[missing],case['truth_r'][missing],case['truth_t'][missing])
        left=np.flatnonzero(missing)[0]-1
        last=pose_surface_error(np.repeat(case['object_r'][left][None],case['length'],0),
            np.repeat(case['object_t'][left][None],case['length'],0),case['truth_r'][missing],case['truth_t'][missing])
        rows=dict(name=case['name'],method=result.gaps[0]['method'],baseline_error_m=baseline,
                  proposed_error_m=proposed,last_pose_error_m=last,gain=1-proposed/baseline,
                  complete=result.complete,visible_pose_bytes_unchanged=(result.rotations[case['observed']].tobytes()==case['object_r'][case['observed']].tobytes()
                    and result.translations[case['observed']].tobytes()==case['object_t'][case['observed']].tobytes()))
        results.append(rows)
        signatures.append(hashlib.sha256(case['object_r'].tobytes()+case['object_t'].tobytes()+case['hand_r'].tobytes()+case['hand_t'].tobytes()).hexdigest())
    carry=[r for r in results if r['name'].startswith('carry_')]
    failed=[]
    if not all(r['complete']and r['visible_pose_bytes_unchanged']for r in results):failed.append('coverage_or_visible_pose_mutation')
    if not all(r['gain']>=.10 for r in carry):failed.append('carry_gap_gain_below10percent')
    if not all(r['proposed_error_m']<=1.05*r['baseline_error_m']+1e-12 for r in results if not r['name'].startswith('carry_')):
        failed.append('noncarry_stratum_error_regression_over5percent')
    return dict(stage='new_procedural_occlusion_bridge_initializer_gate',status='pass'if not failed else'fail',
        seed=SEED,scenarios=12,frames_each=FRAMES,cohort_sha256=hashlib.sha256(''.join(signatures).encode()).hexdigest(),
        failed_gates=failed,carry_minimum_gain=min(r['gain']for r in carry),
        initializer_only=True,rgb_or_model_inference_validated=False,challenge_performance_verified=False,results=results)


def test_frozen12_case_cohort_gate_keeps_truth_out_solver_and_does_not_assert_adoption():
    report=evaluate_frozen_cohort()
    assert len(report['results'])==12 and report['seed']==310427
    assert report['status']in('pass','fail')
    assert report['initializer_only']and not report['challenge_performance_verified']
    assert all(r['complete']and r['visible_pose_bytes_unchanged']for r in report['results'])


def test_stable_carry_conditions_on_measured_hand_with_proper_rotations():
    case=next(new_cohort());result=infer(case)
    assert result.complete and result.gaps[0]['method']=='hand_conditioned_endpoint_constrained'
    assert result.gaps[0]['contact_asserted']is False
    np.testing.assert_allclose(result.translations,case['truth_t'],atol=1e-14)
    np.testing.assert_allclose(result.rotations,case['truth_r'],atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(result.rotations),1,atol=1e-14)
    assert np.isnan(case['object_r'][~case['observed']]).all()


@pytest.mark.parametrize('scenario',['free','slip','regrasp'])
def test_unstable_relative_pose_selects_free_not_forced_contact(scenario):
    case=next(c for c in new_cohort()if c['scenario']==scenario and c['length']==12)
    result=infer(case);br,bt=endpoint_baseline(case)
    assert result.gaps[0]['method']=='free_endpoint_interpolation'
    np.testing.assert_array_equal(result.rotations[~case['observed']],br)
    np.testing.assert_array_equal(result.translations[~case['observed']],bt)


def test_missing_hand_is_explicit_endpoint_fallback():
    case=next(new_cohort());missing=np.flatnonzero(~case['observed'])[1]
    case['hand_valid'][missing]=False;case['hand_r'][missing]=np.nan;case['hand_t'][missing]=np.nan
    result=infer(case)
    assert result.complete and result.gaps[0]['reason']=='missing_hand_evidence'


def test_leading_trailing_abstain_not_static_pose_extrapolation():
    case=next(new_cohort());case['observed'][:2]=False;case['observed'][-3:]=False
    case['object_r'][~case['observed']]=np.nan;case['object_t'][~case['observed']]=np.nan
    result=infer(case)
    assert not result.complete and np.isnan(result.translations[:2]).all()and np.isnan(result.rotations[-3:]).all()
    assert result.gaps[0]['method']==result.gaps[-1]['method']=='abstain'
    with pytest.raises(ValueError,match='bilateral'):result.require_complete()


def test_visible_inputs_bit_identical_and_no_mutation():
    case=next(new_cohort());r=case['object_r'].copy();t=case['object_t'].copy()
    result=infer(case)
    np.testing.assert_array_equal(case['object_r'],r);np.testing.assert_array_equal(case['object_t'],t)
    assert result.rotations[case['observed']].tobytes()==r[case['observed']].tobytes()
    assert result.translations[case['observed']].tobytes()==t[case['observed']].tobytes()


@pytest.mark.parametrize('fault',['reflection','nonorthogonal','filledunknown','handfilled','indices','nonfinite','flags'])
def test_invalid_or_fabricated_observations_rejected(fault):
    case=next(new_cohort());frames=np.arange(FRAMES)
    if fault=='reflection':case['object_r'][0,:,0]*=-1
    elif fault=='nonorthogonal':case['hand_r'][0]*=2
    elif fault=='filledunknown':case['object_t'][~case['observed']]=0
    elif fault=='handfilled':case['hand_valid'][0]=False
    elif fault=='indices':frames[10]=11
    elif fault=='nonfinite':case['object_t'][0]=np.inf
    else:case['observed']=case['observed'].astype(int)
    with pytest.raises(ValueError):initialize_occluded_gaps(frames,case['object_r'],case['object_t'],case['observed'],case['hand_r'],case['hand_t'],case['hand_valid'])


def test_fixed_thresholds_no_tunable_per_record_contract():
    assert ANCHORS_PER_SIDE==3 and MAX_TRANSLATION_DISPERSION_M==.03 and MAX_ROTATION_DISPERSION_RAD==.15


if __name__=='__main__':
    report=evaluate_frozen_cohort()
    # Only aggregate useful decision/reproducibility output; no pose arrays/files.
    print(json.dumps({k:v for k,v in report.items()if k!='results'},sort_keys=True))
