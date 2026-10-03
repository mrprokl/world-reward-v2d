"""Own analytic geometry only; no MHR/model/record/media reads."""
import copy
import time

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from world_reward.cross_surface import (audit_closed_surface, audit_cross_surfaces,
    triangle_relation, validate_closed_mesh, _segment_distance, _shared_simplex_only)


def box(center=(0., 0., 0.), scale=1.):
    v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float) * scale + center
    f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                  [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]], np.int64)
    return v, f


def merge(first, second):
    a, f = first; b, g = second
    return np.r_[a,b], np.r_[f,g+len(a)]


def tetrahedron():
    v = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], np.int64)
    return v, f


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_closed_embedded_outward_cube_and_tetrahedron(dtype):
    for v, f in (box(), tetrahedron()):
        report = audit_closed_surface(v.astype(dtype), f)
        assert report["status"] == "pass"
        assert report["mesh"]["original_face_coverage"] == len(f)
        assert report["mesh"]["embedding_verified"] is True
        assert report["full_original_faces_retained"] is True
        assert report["exact_arithmetic_proof"] is False


def test_two_disjoint_multicomponents_are_valid_closed_solid():
    v, f = merge(box(), box((5,0,0)))
    report = audit_closed_surface(v, f)
    assert report["status"] == "pass" and report["mesh"]["components"] == 2


def test_nested_components_cannot_be_positive_same_solid():
    v, f = merge(box(), box(scale=.1))
    report = audit_closed_surface(v, f)
    assert report["status"] == "fail"
    assert report["embedding"]["nested_or_ambiguous_components"] == [1]


def test_closed_outward_crossing_components_rejected_by_full_embedding():
    v, f = merge(box(), box((.5,.6,.7)))
    report = audit_closed_surface(v, f)
    assert report["status"] == "fail"
    assert report["embedding"]["forbidden_intersections"]


@pytest.mark.parametrize("mutation", ["nan", "inf", "float_faces", "bool_faces", "negative", "past_end",
    "repeated", "duplicate_face", "unused_vertex", "boundary", "flipped_one", "all_inward", "collapsed"])
def test_invalid_meshes_fail_before_certificate(mutation):
    v, f = box()
    if mutation == "nan": v[0,0] = np.nan
    elif mutation == "inf": v[0,0] = np.inf
    elif mutation == "float_faces": f = f.astype(float)
    elif mutation == "bool_faces": f = f.astype(bool)
    elif mutation == "negative": f[0,0] = -1
    elif mutation == "past_end": f[0,0] = len(v)
    elif mutation == "repeated": f[0,0] = f[0,1]
    elif mutation == "duplicate_face": f = np.r_[f, f[:1]]
    elif mutation == "unused_vertex": v = np.r_[v, [[4.,4.,4.]]]
    elif mutation == "boundary": f = f[1:]
    elif mutation == "flipped_one": f[0] = f[0,::-1]
    elif mutation == "all_inward": f = f[:,::-1]
    else: v[0] = v[1]
    with pytest.raises(ValueError):
        audit_closed_surface(v, f)


def test_masked_or_bad_precision_geometry_rejected():
    v, f = box()
    with pytest.raises(ValueError): validate_closed_mesh(np.ma.array(v), f)
    with pytest.raises(ValueError): validate_closed_mesh(v, np.ma.array(f))
    with pytest.raises(ValueError, match="conditioned"): validate_closed_mesh(v + 1e10, f)
    with pytest.raises(ValueError): validate_closed_mesh(v.astype(np.int64), f)


def test_pinched_vertex_link_rejected_despite_closed_oriented_edges():
    v, f = tetrahedron()
    b, g = tetrahedron(); b = -b
    vv = np.r_[v, b[1:]]
    gg = g[:,::-1].copy()
    gg = np.where(gg == 0, 0, gg + 3)
    with pytest.raises(ValueError, match="Pinched"):
        validate_closed_mesh(vv, np.r_[f,gg])


def test_separation_and_two_mm_proximity_are_not_intersection_or_forceclosure():
    v, f = box()
    distant = audit_cross_surfaces(v, f, v+[4,0,0], f)
    close = audit_cross_surfaces(v, f, v+[2.001,0,0], f)
    assert distant["status"] == close["status"] == "pass"
    assert distant["minimum_surface_distance_m"] is None
    assert distant["minimum_distance_lower_bound_m"] == .002
    assert close["minimum_surface_distance_m"] == pytest.approx(.001)
    assert close["proximity_pairs"] > 0
    assert not any(close["intersection_counts"].values())
    assert close["force_closure_verified"] is close["biomechanics_verified"] is False
    assert close["sampled_collision_test_used"] is False


@pytest.mark.parametrize("offset", [[2,0,0],[2,2,0],[2,2,2]])
def test_exact_face_edge_and_vertex_contact_always_fail_closed(offset):
    v, f = box()
    r = audit_cross_surfaces(v, f, v+offset, f)
    assert r["status"] == "fail" and r["touching_certified"] is False
    assert r["intersection_counts"]["boundary_contact"] + r["intersection_counts"]["coplanar_overlap"] > 0


def test_tolerance_gap_fails_and_safely_positive_gap_passes():
    v, f = box()
    assert audit_cross_surfaces(v, f, v+[2+5e-9,0,0], f)["status"] == "fail"
    assert audit_cross_surfaces(v, f, v+[2+1e-6,0,0], f)["status"] == "pass"


def test_full_containment_without_surface_intersections_is_penetration():
    v, f = box()
    r = audit_cross_surfaces(v, f, v*.1, f)
    assert r["status"] == "fail"
    assert not any(r["intersection_counts"].values())
    assert r["containment"][1][0]["state"] == "inside"
    assert r["containment"][1][0]["winding"] == pytest.approx(1.)


def test_partial_overlapping_solids_with_transverse_surface_intersections():
    v, f = box()
    r = audit_cross_surfaces(v, f, v+[1.2,.3,.4], f)
    assert r["status"] == "fail"
    assert r["intersection_counts"]["proper_intersection"] > 0


@pytest.mark.parametrize("offset,kind", [([0,0,0],"coplanar_overlap"), ([1,0,0],"boundary_contact"),
    ([2,0,0],"separated"), ([0,0,.001],"separated")])
def test_coplanar_area_boundary_separation_and_parallel_proximity(offset, kind):
    a = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    r = triangle_relation(a, a+offset)
    assert r["kind"] == kind
    if offset == [0,0,.001]: assert r["distance_m"] == pytest.approx(.001)


def test_transverse_triangle_intersection_has_no_nearby_vertex():
    a = np.array([[-2.,-1.,0.],[2.,-1.,0.],[0.,2.,0.]])
    b = np.array([[0.,-2.,-1.],[0.,2.,-1.],[0.,0.,2.]])
    r = triangle_relation(a,b)
    assert r["kind"] == "proper_intersection" and r["distance_m"] == 0
    assert len(r["points"]) == 2


def test_skew_edge_edge_distance_not_vertex_to_surface_proxy():
    a = np.array([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.]])
    b = np.array([[0.,-1.,.001],[0.,1.,.001],[1.,0.,1.]])
    assert triangle_relation(a,b)["distance_m"] == pytest.approx(.001)
    # Nearly parallel interior crossings must not be lost by dot-product cancellation.
    assert _segment_distance(np.array([0.,0.,0.]), np.array([1.,0.,0.]),
        np.array([0.,-1e-9,1e-4]), np.array([1.,1e-9,1e-4])) == pytest.approx(1e-8, rel=1e-12)


def test_common_global_rigid_transform_and_face_order_do_not_change_certificate():
    v, f = box(scale=.1); b = v + [.2005,0,0]
    original = audit_cross_surfaces(v,f,b,f)
    rotation = Rotation.from_rotvec([.2,-.3,.1]).as_matrix()
    translation = np.array([.03,-.04,1.])
    r = audit_cross_surfaces(v@rotation.T+translation,f[::-1],b@rotation.T+translation,f[::-1])
    assert r["status"] == original["status"] == "pass"
    assert r["minimum_surface_distance_m"] == pytest.approx(original["minimum_surface_distance_m"], abs=1e-14)


def test_input_array_bytes_never_mutated_or_repaired():
    v, f = box(); vb, fb = v.tobytes(), f.tobytes()
    r = audit_cross_surfaces(v,f,v+[2.001,0,0],f)
    assert v.tobytes() == vb and f.tobytes() == fb
    assert r["meshes"][0]["original_vertices_sha256"] == validate_closed_mesh(v,f)["report"]["original_vertices_sha256"]


@pytest.mark.parametrize("kwargs", [dict(tolerance_m=0),dict(tolerance_m=True),dict(proximity_m=np.nan),
    dict(proximity_m=1e-9),dict(max_candidate_pairs=False),dict(max_candidate_pairs=0),dict(budget_seconds=-1)])
def test_strict_budgets_and_tolerances(kwargs):
    v, f = box()
    with pytest.raises(ValueError): audit_cross_surfaces(v,f,v+[4,0,0],f,**kwargs)


def test_bounded_candidate_and_whole_deadline_never_silent_pass():
    v,f = box()
    with pytest.raises(RuntimeError, match="candidate"):
        audit_closed_surface(v,f,max_candidate_pairs=1)
    with pytest.raises(TimeoutError):
        audit_cross_surfaces(v,f,v+[4,0,0],f,budget_seconds=1e-12)


def test_disconnected_component_containment_each_side_not_one_arbitrary_mesh_vertex():
    v,f = merge(box((0,0,0), .1), box((4,0,0), .1))
    outer,g = box((4,0,0), .5)
    r = audit_cross_surfaces(v,f,outer,g)
    assert r["status"] == "fail"
    assert [item["state"] for item in r["containment"][0]] == ["outside","inside"]


def test_random_transverse_triangles_known_shared_interior_point():
    rng = np.random.default_rng(821)
    base = np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,2.,0.]])
    for _ in range(16):
        first = base @ Rotation.random(random_state=rng).as_matrix()
        last = base @ Rotation.random(random_state=rng).as_matrix()
        relation = triangle_relation(first,last)
        assert relation["kind"] == "proper_intersection"


@pytest.mark.parametrize("scale", [1e-5,1e-3,1.,1e3])
def test_known_interior_intersection_metric_scale_and_rigid_transform(scale):
    rng = np.random.default_rng(881)
    base = np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,2.,0.]]) * scale
    shift = np.array([.2,-.3,1.])
    for _ in range(8):
        first = base @ Rotation.random(random_state=rng).as_matrix() + shift
        last = base @ Rotation.random(random_state=rng).as_matrix() + shift
        assert triangle_relation(first,last)["kind"] == "proper_intersection"


def test_skinny_valid_known_interior_intersection_and_subtolerance_fail_closed():
    a = np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,2.,0.]]) * .001
    b = a.copy(); b[:,1] *= .001
    b = b @ Rotation.from_rotvec([.2,.3,.1]).as_matrix()
    assert triangle_relation(a,b)["kind"] == "proper_intersection"
    with pytest.raises(ValueError, match="conditioned"):
        triangle_relation(a*1e-6,b*1e-6)


def test_single_component_self_intersection_not_only_disconnected_component_overlap():
    rng = np.random.default_rng(221)
    v = rng.normal(size=(30,3)); v /= np.linalg.norm(v,axis=1)[:,None]
    f = ConvexHull(v).simplices.copy()
    normals = np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    inward = np.sum(normals*v[f].mean(1),axis=1) < 0
    f[inward] = f[inward,::-1]
    v[0] = [0.,0.,-1.5]
    r = audit_closed_surface(v,f)
    assert r["status"] == "fail" and r["mesh"]["components"] == 1
    assert any(pair["kind"] == "proper_intersection" for pair in r["embedding"]["forbidden_intersections"])


def test_adjacent_coplanar_shared_simplex_does_not_excuse_sliver_overlap():
    a = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    below = np.array([[0.,0.,0.],[1.,0.,0.],[.5,-1.,0.]])
    sliver = below.copy(); sliver[2,1] = 1e-9
    ids_a, ids_b = np.array([0,1,2]), np.array([0,1,3])
    assert _shared_simplex_only(a,below,ids_a,ids_b,1e-8)
    assert not _shared_simplex_only(a,sliver,ids_a,ids_b,1e-8)


def test_adjacent_shared_vertex_requires_strict_separating_support_proof():
    a = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    disjoint = np.array([[0.,0.,0.],[-1.,0.,0.],[0.,-1.,0.]])
    crossing = np.array([[0.,0.,0.],[.2,.3,0.],[.4,.2,0.]])
    ids_a,ids_b = np.array([0,1,2]),np.array([0,3,4])
    assert _shared_simplex_only(a,disjoint,ids_a,ids_b,1e-8)
    assert not _shared_simplex_only(a,crossing,ids_a,ids_b,1e-8)


def test_convex_cloud_original_face_coverage_and_broadphase_smoke():
    rng = np.random.default_rng(922)
    v = rng.normal(size=(80,3)); v /= np.linalg.norm(v,axis=1)[:,None]
    hull = ConvexHull(v); f = hull.simplices.copy()
    for i,row in enumerate(f):
        if np.dot(np.cross(v[row[1]]-v[row[0]],v[row[2]]-v[row[0]]),v[row].mean(0)) < 0:
            f[i] = row[::-1]
    started = time.monotonic()
    r = audit_cross_surfaces(v,f,v+[4,0,0],f,budget_seconds=10)
    assert r["status"] == "pass"
    assert r["meshes"][0]["original_face_coverage"] == len(f)
    assert time.monotonic()-started < 10
    assert r["intermesh_candidate_pairs"] == 0
