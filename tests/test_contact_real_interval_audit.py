"""Tiny proof of interval containment; no full challenge computation locally."""
import numpy as np
import pytest

import contact_real_interval_audit as audit
from world_reward.sequence_pose import _ContactTriangleSurface
from test_sequence_contact_patch_real import tiny_bank,protocol
from test_sequence_pose_contact import config


def test_intervals_contain_exhaustive_truth_and_count_all_2318_points():
    rng=np.random.default_rng(4)
    vertices=rng.normal(size=(60,3));faces=np.arange(60).reshape(-1,3)
    surface=_ContactTriangleSurface(vertices,faces);points=rng.normal(size=(2318,3))
    truth=float(surface.distances(points,batch_size=32).min())
    row=audit.whole_hand_interval(surface,points)
    assert row['lower_m']<=truth<=row['upper_m']
    assert row['points_total']==row['points_bound_evaluated']==2318
    assert row['exact_witness_points']==1
    assert row['upper_m']<=row['vertex_upper_m']
    assert not row['exact_surface_minimum_claimed']


def test_cheap_mode_has_no_continuous_triangle_calls_and_tight_mode_exactly_one(monkeypatch):
    surface=_ContactTriangleSurface(np.eye(3),np.array([[0,1,2]]));calls=[]
    original=surface.distances
    def record(points,**kw):calls.append(len(points));return original(points,**kw)
    monkeypatch.setattr(surface,'distances',record)
    points=np.random.default_rng(9).normal(size=(2318,3))
    row=audit.whole_hand_interval(surface,points,tighten=False)
    assert calls==[] and row['exact_witness_points']==0
    audit.whole_hand_interval(surface,points)
    assert calls==[1]


def test_aabb_contains_only_referenced_mesh_and_degenerate_faces_supported():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[99.,99.,99.]])
    surface=_ContactTriangleSurface(vertices,np.array([[0,1,2],[1,1,1]]))
    points=np.array([[99.,99.,99.],[1.,.2,0.]])
    row=audit.whole_hand_interval(surface,points)
    assert row['lower_m']<=.2<=row['upper_m']
    assert row['nearest_vertex_witness_anatomical_index']==1
    assert row['lower_m']>.19


def test_overlapping_intervals_explicitly_inconclusive_no_contact_threshold():
    original=dict(lower_m=.01,upper_m=.05,numerical_guard_m=1e-12)
    candidate=dict(lower_m=.04,upper_m=.06,numerical_guard_m=1e-12)
    assert audit.strict_interval_comparison(candidate,original)['outcome']=='inconclusive_overlapping_intervals'
    candidate['lower_m']=.051
    assert audit.strict_interval_comparison(candidate,original)['outcome']=='certified_larger_gap'
    candidate.update(lower_m=0.,upper_m=.009)
    assert audit.strict_interval_comparison(candidate,original)['outcome']=='certified_smaller_gap'


def test_invalid_anatomy_does_not_drop_bad_points():
    surface=_ContactTriangleSurface(np.eye(3),np.array([[0,1,2]]))
    with pytest.raises(ValueError):audit.whole_hand_interval(surface,np.array([[1.,2.,np.nan]]))


def test_full_variant_intervals_cover_all_active_anatomy_and_rows():
    bank=tiny_bank();row=audit.variant_intervals(bank,bank['rotations'],bank['translations'])
    assert row['full_active_hand_rows']==int(bank['evidence'].activations.sum())
    assert row['points_total']==row['points_bound_evaluated']==12*row['full_active_hand_rows']
    assert row['exact_surface_minimum_claimed'] is False


def test_cost_receipt_marks_geometry_certification_not_accuracy():
    bank=tiny_bank();pool,_=audit.real.bounded_pool(bank,protocol())
    intervals=audit.variant_intervals(bank,bank['rotations'],bank['translations'])
    receipt=audit.coverage_cost_receipt(bank,bank['rotations'],bank['translations'],pool,
        protocol(),config(),'original',intervals)
    assert len(receipt['pool_coverage_rows'])==int(bank['evidence'].activations.sum())
    assert receipt['certified_better_proximity_outside_pool']+receipt['uncertified_pool_coverage']==len(receipt['pool_coverage_rows'])
    assert receipt['scope']=='prediction_geometry_proximity_not_physical_contact_truth'
