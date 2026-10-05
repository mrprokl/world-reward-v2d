"""Tiny FP64 arithmetic controls only; no FIT, full-bank cost or GPU claim."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from world_reward import coherent_pair_learning as old
from world_reward import coherent_pair_marginal as oracle
from world_reward.coherent_pair_cache import prepare_pair_cache
from world_reward.coherent_pair_packed import prepare_marginal_packed
from world_reward.coherent_pair_packed_score import score_pair_marginal_packed

spec=importlib.util.spec_from_file_location('tiny_packed_score_fixtures',Path(__file__).with_name('test_coherent_pair_marginal.py'))
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
RTOL,ATOL,FD_STEP,FD_TOL=1e-12,1e-12,1e-6,1e-7


def prepared(bank=None,variable=True):
    if bank is None:bank=fixtures.bank()
    cache=prepare_pair_cache(bank,old.PairScale(np.arange(1.,13.),np.full(12,variable,bool)))
    return prepare_marginal_packed(cache),oracle.MarginalPairReference(cache)


def score(packed,theta=None,temperature=.7,alpha=0.):
    if theta is None:theta=np.linspace(-.2,.3,17,dtype=np.float64)
    return score_pair_marginal_packed(packed,theta,temperature=temperature,alpha=alpha)


def bits(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


@pytest.mark.parametrize('temperature',[.1,.7,3.])
@pytest.mark.parametrize('alpha',[0.,.2,1.])
@pytest.mark.parametrize('aliases,missing,nohoi',[(False,False,False),(True,True,False),(False,True,True)])
def test_scores_native_support_identity_and_all17_vjp_against_enumeration(temperature,alpha,aliases,missing,nohoi):
    p,r=prepared(fixtures.bank(aliases=aliases,missing=missing,no_hoi=nohoi));before=old.core._fingerprint(p)
    actual=score(p,temperature=temperature,alpha=alpha)
    expected=oracle.score_pair_marginal(r,np.linspace(-.2,.3,17),temperature=temperature,alpha=alpha)
    for name in ('native_scores_a','native_scores_b','scores_a','scores_b','geometry_derivatives_a','geometry_derivatives_b','alpha_derivatives_b'):
        np.testing.assert_allclose(getattr(actual,name),getattr(expected,name),rtol=RTOL,atol=ATOL,equal_nan=True)
        assert not getattr(actual,name).flags.writeable
    for name in ('native_supported','native_route_supported','supported'):bits(getattr(actual,name),getattr(expected,name))
    assert old.core._fingerprint(actual.identity)==old.core._fingerprint(expected.identity)
    assert actual.parameter_fingerprint==expected.parameter_fingerprint
    assert actual.distribution_fingerprint==p.packed_fingerprint
    assert old.core._fingerprint(p)==before


@pytest.mark.parametrize('alpha',[0.,.37])
def test_all17_geometry_analytic_vjp_and_right_alpha_derivative_fd(alpha):
    p,_=prepared(fixtures.with_margins(fixtures.bank(n=1,o=2),np.array([-2.,1.,1.,3.])))
    theta=np.linspace(-.2,.3,17);base=score(p,theta,alpha=alpha)
    for j in range(17):
        delta=np.zeros(17);delta[j]=FD_STEP
        plus,minus=score(p,theta+delta,alpha=alpha),score(p,theta-delta,alpha=alpha)
        for name,derivative in (('scores_a',base.geometry_derivatives_a),('scores_b',base.geometry_derivatives_b)):
            numerical=(getattr(plus,name)-getattr(minus,name))/(2*FD_STEP)
            np.testing.assert_allclose(numerical,derivative[...,j],rtol=FD_TOL,atol=FD_TOL)
    # Second-order right FD respects alpha>=0 without its O(h) curvature bias.
    numerical=(-3*base.scores_b+4*score(p,theta,alpha=FD_STEP).scores_b-score(p,theta,alpha=2*FD_STEP).scores_b)/(2*FD_STEP) if alpha==0. else \
        (score(p,theta,alpha=alpha+FD_STEP).scores_b-score(p,theta,alpha=alpha-FD_STEP).scores_b)/(2*FD_STEP)
    np.testing.assert_allclose(numerical,base.alpha_derivatives_b,rtol=FD_TOL,atol=FD_TOL)


def test_alpha0_shared_path_bitexact_and_margin_has_no_a_mass():
    p,_=prepared(fixtures.bank(n=1,o=1));z,_=prepared(fixtures.with_margins(fixtures.bank(n=1,o=1),np.array([-2.,3.,4.,4.])))
    a,b=score(p),score(z)
    for result in (a,b):
        bits(result.native_scores_a,result.native_scores_b);bits(result.scores_a,result.scores_b)
        bits(result.geometry_derivatives_a,result.geometry_derivatives_b)
    bits(a.native_scores_a,b.native_scores_a);bits(a.scores_a,b.scores_a)
    assert np.any(a.alpha_derivatives_b!=b.alpha_derivatives_b)


def test_full_hierarchy_same_complete_geometry_and_conditional_margin_not_flatmass():
    b=fixtures.bank(n=1,o=1);h=b.evidence.hoi_evidence;x=h.features.copy();x[:,0]=np.tile([.1,.1,.3,.3],2)
    b=replace(b,evidence=replace(b.evidence,hoi_evidence=replace(h,features=x)))
    b=fixtures.with_margins(b,np.array([-2.,3.,4.,4.]));p,r=prepared(b)
    theta=np.linspace(-.2,.3,17);tau,alpha=.8,.3;actual=score(p,theta,temperature=tau,alpha=alpha)
    values=[];features=[];margins=[];mass=[];sides=[s for s in r.group_sides[0] if s]
    for side in sides:
        for state in side:
            for m in state.margins:
                values.append(state.values@theta+alpha*m);features.append(state.values);margins.append(m)
                mass.append(1/len(sides)/len(side)/len(state.margins))
    values=np.array(values);mass=np.array(mass);offset=values.max();w=mass*np.exp((values-offset)/tau);posterior=w/w.sum()
    np.testing.assert_allclose(actual.scores_b[0,0],offset+tau*np.log(w.sum()),rtol=RTOL,atol=ATOL)
    np.testing.assert_allclose(actual.geometry_derivatives_b[0,0],posterior@features,rtol=RTOL,atol=ATOL)
    np.testing.assert_allclose(actual.alpha_derivatives_b[0,0],posterior@margins,rtol=RTOL,atol=ATOL)


def test_equal_side_prior_not_flat_geometry_count_prior():
    b=fixtures.bank(n=1,o=1);h=b.evidence.hoi_evidence;x=h.features.copy();x[:4,0]=[.1,.2,.3,.4]
    b=replace(b,evidence=replace(b.evidence,hoi_evidence=replace(h,features=x)));p,r=prepared(b)
    theta=np.zeros(17);theta[6]=1.;tau=.8;out=score(p,theta,temperature=tau)
    assert [len(side) for side in r.group_sides[0]]==[4,1]
    side_values=[np.array([state.values@theta for state in side]) for side in r.group_sides[0]]
    expected=tau*np.log(sum(np.exp(v/tau).mean() for v in side_values)/2)
    flat=tau*np.log(np.exp(np.concatenate(side_values)/tau).mean())
    np.testing.assert_allclose(out.scores_a[0,0],expected,rtol=RTOL,atol=ATOL)
    assert abs(expected-flat)>1e-4


def test_incompatible_local_bridge_never_make_cross_route_max():
    b=fixtures.bank(n=1,o=1);e=b.evidence;h=e.hoi_evidence
    x=h.features.copy();x[:,0]=np.tile([0.,10.,0.,10.],2)
    bridge=e.route_features.copy();bridge[:,0]=[10.,0.,10.,0.]
    b=replace(b,evidence=replace(e,route_features=bridge,hoi_evidence=replace(h,features=x)));p,r=prepared(b)
    theta=np.zeros(17);theta[6],theta[10]=p.cache.scales.scale[6],p.cache.scales.scale[10]
    result=score(p,theta,temperature=1.)
    np.testing.assert_allclose(result.scores_a,[[10.]],rtol=RTOL,atol=ATOL)
    assert result.scores_a[0,0]!=20.
    bits(result.native_supported,p.arrays['native_supported'])


def test_parameters_copied_fingerprint_and_output_stable_after_caller_mutation():
    p,_=prepared();theta=np.linspace(-.2,.3,17);before=old.core._fingerprint(p)
    result=score(p,theta);saved=result.scores_a.copy();parameter=result.parameter_fingerprint
    theta[:]=999.
    bits(result.scores_a,saved);assert result.parameter_fingerprint==parameter and old.core._fingerprint(p)==before


def test_unsupported_anchors_remain_nan_and_zero_vjp_not_off():
    b=fixtures.bank(n=1,o=2,no_hoi=True,missing=True);e=b.evidence
    boxes=e.objects.boxes_original_xyxy.copy();boxes[0,2:]=boxes[0,:2]
    person=fixtures.PersonPoseObservations(2,(20,30),('p0',),np.array([[0.,0.,25.,18.]]),np.array([.9]),
        np.full((1,133,2),5.),np.zeros((1,133),np.float32))
    bank=old.pair_route_bank(person,fixtures.build_interaction_candidate_evidence(person,replace(e.objects,boxes_original_xyxy=boxes)))
    p,_=prepared(bank);out=score(p,alpha=.7);unsupported=~out.supported
    assert unsupported.any() and np.isnan(out.scores_a[unsupported]).all() and np.isnan(out.scores_b[unsupported]).all()
    assert np.all(out.geometry_derivatives_a[unsupported]==0.) and np.all(out.geometry_derivatives_b[unsupported]==0.)
    assert np.all(out.alpha_derivatives_b==0.) and not out.native_route_supported.any()


def test_output_and_identity_allarrays_cannot_reenable_writes():
    p,_=prepared();out=score(p,alpha=.3)
    for name in ('native_scores_a','native_scores_b','native_supported','native_route_supported','scores_a','scores_b','supported',
                 'geometry_derivatives_a','geometry_derivatives_b','alpha_derivatives_b'):
        with pytest.raises(ValueError):getattr(out,name).flags.writeable=True
    for name in ('native_pair_slots','native_query_ids','person_to_group','object_to_group'):
        with pytest.raises(ValueError):out.identity[name].flags.writeable=True
    with pytest.raises(TypeError):out.identity['source_person_ids']=()


@pytest.mark.parametrize('n,o',[(0,0),(0,2),(1,0)])
def test_empty_banks_preserve_native_ids_and_no_unsupported_filler(n,o):
    p,r=prepared(fixtures.bank(n=n,o=o));out=score(p)
    assert out.native_scores_a.shape==(n,2,o) and out.scores_a.shape==(n,o) and not out.supported.any()
    assert out.identity['native_pair_slots'].shape==(4,)
    bits(out.scores_a,out.scores_b);assert out.geometry_derivatives_a.shape==(n,o,17)


def test_duplicate_routes_aliases_raw_signedzero_and_inactive_scales():
    p,_=prepared(fixtures.bank(n=1,o=1,copies=1));z,_=prepared(fixtures.bank(n=3,o=2,copies=3,aliases=True))
    for alpha in (0.,.3):
        a,b=score(p,alpha=alpha),score(z,alpha=alpha)
        for name in ('scores_a','scores_b','geometry_derivatives_a','geometry_derivatives_b','alpha_derivatives_b'):
            np.testing.assert_allclose(getattr(a,name),getattr(b,name),rtol=RTOL,atol=ATOL)
    b=fixtures.bank(n=1,o=1);h=b.evidence.hoi_evidence;x=h.features.copy();x[:,0]=np.tile([0.,-0.,.25,.25],2)
    b=replace(b,evidence=replace(b.evidence,hoi_evidence=replace(h,features=x)));p,r=prepared(b,False)
    actual=score(p,alpha=.3);expected=oracle.score_pair_marginal(r,np.linspace(-.2,.3,17),temperature=.7,alpha=.3)
    np.testing.assert_allclose(actual.scores_b,expected.scores_b,rtol=RTOL,atol=ATOL)
    assert p.arrays['geometry_components'].shape[0]>1


@pytest.mark.parametrize('kwargs',[{'temperature':0.},{'temperature':True},{'temperature':np.nan},{'alpha':-.1},{'alpha':True},{'alpha':np.inf}])
def test_invalid_parameters_rejected(kwargs):
    p,_=prepared()
    with pytest.raises(ValueError):score(p,**kwargs)


def test_nonfinite_dtype_shape_coefficients_and_overflow_rejected():
    p,_=prepared()
    for theta in (np.zeros(17,np.float32),np.zeros(16),np.full(17,np.nan)):
        with pytest.raises(ValueError):score(p,theta)
    with pytest.raises((ValueError,FloatingPointError)):score(p,np.full(17,1e308),alpha=1e308)
