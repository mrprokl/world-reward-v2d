"""Tiny geometry certification tests only; full real audit stays on Azure."""
from dataclasses import replace
import time

import numpy as np
import pytest

import contact_real_geometry_audit as audit
from world_reward.sequence_pose import _ContactTriangleSurface
from test_sequence_contact_patch_real import tiny_bank,protocol
from test_sequence_pose_contact import config


def test_certified_full_hand_minimum_matches_exhaustive_continuous_surface():
    rng=np.random.default_rng(33)
    vertices=rng.normal(size=(90,3));faces=np.arange(90).reshape(-1,3)
    surface=_ContactTriangleSurface(vertices,faces)
    points=rng.normal(size=(2318,3))
    row=audit.whole_hand_minimum(surface,points)
    expected=surface.distances(points,batch_size=32)
    assert row['distance_m']==expected.min()
    assert expected[row['point_index']]==expected.min()
    assert row['points_total']==2318
    assert row['points_exactly_evaluated']+row['points_proven_nonwinning']==2318


def test_lower_bounds_never_exceed_full_exact_distance_and_real_pruning():
    vertices=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]])
    surface=_ContactTriangleSurface(vertices,np.array([[0,1,2]]))
    points=np.vstack(([0.,0.,.002],np.column_stack((np.arange(1,102),np.zeros(101),np.ones(101)))))
    exact=surface.distances(points)
    assert np.all(audit.lower_bounds(surface,points)<=exact)
    result=audit.whole_hand_minimum(surface,points)
    assert result['distance_m']==exact.min() and result['points_proven_nonwinning']>80


def test_degenerate_and_unreferenced_geometry_not_silently_changed():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[99.,99.,99.]])
    surface=_ContactTriangleSurface(vertices,np.array([[0,1,2],[1,1,1]]))
    points=np.array([[99.,99.,99.],[1.,.2,0.]])
    result=audit.whole_hand_minimum(surface,points)
    assert result['point_index']==1 and result['distance_m']==.2
    surface._centre_tree=None
    assert audit.whole_hand_minimum(surface,points)['distance_m']==.2


def test_complete_hands_cannot_drop_invalid_points_or_overrun_deadline():
    surface=_ContactTriangleSurface(np.eye(3),np.array([[0,1,2]]))
    with pytest.raises(ValueError):audit.whole_hand_minimum(surface,np.array([[1.,2.,np.nan]]))
    with pytest.raises(TimeoutError):audit.whole_hand_minimum(surface,np.ones((3,3)),deadline=time.monotonic()-1)


def test_distance_gradient_ray_alignment_is_not_claimed_surface_normal():
    surface=_ContactTriangleSurface(np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]]),np.array([[0,1,2]]))
    row=audit.ray_gradient_alignment(surface,np.array([.5,.5,.3]),np.array([0.,0.,1.]))
    assert row['absolute_ray_alignment']==pytest.approx(1.,abs=1e-10)
    assert row['distance_gradient_norm']==pytest.approx(1.,abs=1e-10)
    assert 'not_oriented_surface_normal' in row['scope']


def test_factor_costs_match_unchanged_fitter_initial_objective():
    bank=tiny_bank();pool,_=audit.real.bounded_pool(bank,protocol())
    cfg=replace(config(),max_nfev=300)
    measured=audit.factor_costs(bank,bank['rotations'],bank['translations'],pool,protocol(),cfg,'J1')
    fitted=audit.real.fit_worker(('J1',bank,bank['evidence'],protocol(),cfg))
    assert fitted['status']=='complete'
    assert measured['total_cost']==pytest.approx(fitted['fit']['initial_cost'],abs=1e-10)
    assert measured['blocks']['RGB']['count']>measured['blocks']['contact']['count']
    assert measured['contact_scale_m']>0


def test_variant_audit_whole_hand_all_active_rows_and_proxy_disclosure():
    bank=tiny_bank();pool,_=audit.real.bounded_pool(bank,protocol())
    row=audit.variant_audit('original',bank,bank['rotations'],bank['translations'],pool,
        protocol(),config(),time.monotonic()+10)
    assert row['active_hand_rows']==int(bank['evidence'].activations.sum())
    assert row['possible_points']==row['active_hand_rows']*12
    assert row['whole_hand_gap_m']['mean']<=row['pool_gap_m']['mean']+1e-12
    assert row['frames']==6
