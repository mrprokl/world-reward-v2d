"""Tiny numerical private evaluation tests; no assets, torch, rendering or GPU."""
import importlib.util
from pathlib import Path
import copy
import numpy as np
import pytest

@pytest.fixture
def evaluate():
    path=Path(__file__).resolve().parents[1]/'infra/hand_synthetic_evaluate.py'
    spec=importlib.util.spec_from_file_location('wr_hand_eval_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def records(before=100., after=90., neutral_delta=0.):
    return [{'hands':{s:{'baseline':{'hand_pve_mm':before},'candidate':{'hand_pve_mm':before+neutral_delta if i==0 else after}} for s in ('left','right')}} for i in range(6)]

def test_preregistered_gain_and_neutral_control(evaluate):
    assert evaluate.decision(records())['synthetic_hypothesis_supported']
    assert not evaluate.decision(records(after=96))['synthetic_hypothesis_supported']
    assert not evaluate.decision(records(neutral_delta=.02))['synthetic_hypothesis_supported']
    assert not evaluate.decision(records(before=0,after=0))['synthetic_hypothesis_supported']

def test_every_nonneutral_hand_counts_no_cherrypick(evaluate):
    r=records();r[5]['hands']['right']['candidate']['hand_pve_mm']=200
    assert not evaluate.decision(r)['synthetic_hypothesis_supported']
    assert evaluate.decision(r)['mean_nonneutral_hand_pve_candidate_mm']==101

def test_no_alignment_and_wrist_relative_diagnostic_distinct(evaluate):
    v=np.array([[0.,0.,1.],[.1,0.,1.],[.2,0.,1.]])
    j=np.array([[0.,0.,1.],[.1,0.,1.],[.2,0.,1.]])
    region={'mask':np.array([True,True,False]),'wrist':0,'fingers':np.array([1,2])}
    result,_=evaluate.errors(v+[.1,0,0],j+[.1,0,0],v,j,region)
    assert result['hand_pve_mm']==pytest.approx(100)
    assert result['finger_mpjpe_mm']==pytest.approx(100)
    assert result['wrist_relative_hand_pve_mm_diagnostic']<1e-10

def test_visible_face_support_explicit_not_exact_visibility(evaluate):
    face=np.full((768,1024),-1,np.int32);face[0,0]=0;face[1,1]=99
    mask=np.zeros(18439,np.bool_);mask[:4]=True
    assert evaluate.visible_face_support(face,np.array([[0,1,2],[1,2,3]]),mask).tolist()==[True,True,True,False]

@pytest.mark.parametrize('value',[np.zeros((2,3)),np.full((3,3),np.nan),np.zeros((3,3),int)])
def test_finite_shape_fail_closed(evaluate,value):
    with pytest.raises(ValueError):evaluate.finite(value,(3,3))

def report(evaluate):
    return {'stage':'public_synthetic_rgb_official_finger_proposals','status':'pass','cases':6,
            'private_truth_read':False,'challenge_inputs_used':False,'adoption_performed':False,
            'nonhand_controls_bit_identical':True,'shared_identity_preserved':True,
            'conversion_fidelity_verified':True,'model_sha256':evaluate.MODEL_SHA,'converter_sha256':evaluate.CONVERTER_SHA}

@pytest.mark.parametrize('key,value',[('private_truth_read',True),('status','fail'),('cases',True),
    ('conversion_fidelity_verified',1),('model_sha256','a'*64),('adoption_performed',True)])
def test_public_chain_required_before_private_eval(evaluate,key,value):
    r=report(evaluate);evaluate.proposal_contract(r);r[key]=value
    with pytest.raises(ValueError):evaluate.proposal_contract(r)

def test_sha_no_symlink_or_frozen_mutation(evaluate,tmp_path):
    p=tmp_path/'p';p.write_text('tiny');digest=evaluate.sha256(p);evaluate.require_hash(p,digest)
    q=tmp_path/'q';q.symlink_to(p)
    with pytest.raises(ValueError):evaluate.require_hash(q,digest)
    p.write_text('changed')
    with pytest.raises(ValueError):evaluate.require_hash(p,digest)

def test_wrapper_has_no_gpu_models_challenge_or_writable_truth(evaluate):
    wrapper=Path(evaluate.__file__).with_name('run_hand_synthetic_evaluate.sh').read_text()
    assert '--gpus' not in wrapper and 'src=$ROOT/weights' not in wrapper
    assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in wrapper
    assert 'src=$BASE,dst=$BASE' not in wrapper
    assert '--network none' in wrapper and '60s docker run' in wrapper
    assert 'chown -R' not in wrapper


def test_incomplete_quality_cohort_cannot_pass(evaluate):
    with pytest.raises(ValueError):evaluate.decision(records()[:-1])
