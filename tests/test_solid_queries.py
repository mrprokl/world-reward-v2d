"""Tiny own analytic meshes; no data/model/grid/manufacture or embedding claim."""
import time

import numpy as np
import pytest

from world_reward import solid_queries as q
from world_reward.cross_surface import _point_triangle, _winding


def tetrahedron(dtype=np.float64):
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=dtype)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return v, f


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inside_outside_full_surface_and_wrong_expected_state(dtype):
    v, f = tetrahedron(dtype)
    for point, state, winding in (([.1, .1, .1], "inside", 1.), ([2., 2., 2.], "outside", 0.)):
        p = np.array(point, dtype=dtype)
        report = q.point_surface_query(p, v, f, expected_state=state)
        assert report["status"] == "pass" and report["winding"] == pytest.approx(winding, abs=1e-6)
        assert report["triangles_examined"] == 4 and report["full_original_faces_retained"] is True
        assert report["exact_arithmetic_proof"] is report["embedding_verified"] is False
        assert report["requires_prior_embedding_certificate"] is True
        other = "outside" if state == "inside" else "inside"
        assert q.point_surface_query(p, v, f, expected_state=other)["status"] == "fail"


@pytest.mark.parametrize("point", [[.25, .25, 0.], [.5, .5, 0.], [0., 0., 0.]])
@pytest.mark.parametrize("state", ["inside", "outside"])
def test_face_interior_edge_and_vertex_hits_fail_for_both_states(point, state):
    v, f = tetrahedron()
    report = q.point_surface_query(np.array(point), v, f, expected_state=state)
    assert report["status"] == "fail" and report["continuous_surface_distance_m"] == 0.
    assert report["triangles_examined"] == 4


def test_face_interior_hit_catches_original_winding_vertex_only_gap():
    v, f = tetrahedron(); point = np.array([.25, .25, 0.])
    assert np.linalg.norm(v - point, axis=1).min() > .3
    report = q.point_surface_query(point, v, f, expected_state="inside")
    assert report["winding"] == _winding(point, v[f], 1e-8)
    assert report["continuous_surface_distance_m"] == 0 and report["status"] == "fail"


@pytest.mark.parametrize("height,passed", [(5e-9, False), (1e-8, False), (2e-8, True), (.1, True)])
def test_continuous_face_distance_strict_metric_tolerance(height, passed):
    v, f = tetrahedron(); point = np.array([.25, .25, -height])
    report = q.point_surface_query(point, v, f, expected_state="outside")
    assert report["continuous_surface_distance_m"] == pytest.approx(height, abs=1e-17)
    assert (report["status"] == "pass") == passed


@pytest.mark.parametrize("scale", [.001, 1., 100.])
def test_vectorized_continuous_distance_parity_all_triangle_regions(scale):
    v, f = tetrahedron(); triangles = v[f] * scale
    points = np.array([[.2, .2, -.3], [.7, .7, -.1], [-.2, 0., .4], [1.2, -.2, -.3], [.1, .1, .1]]) * scale
    for p in points:
        actual = q._triangle_distances(p, triangles, 1e-8)
        expected = np.array([np.sqrt(_point_triangle(p, tri)[0]) for tri in triangles])
        assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14 * scale)


def test_random_tiny_distance_parity_without_mesh_sampling_claim():
    rng = np.random.default_rng(8798)
    triangles = rng.normal(size=(32, 3, 3))
    for point in rng.normal(size=(8, 3)):
        actual = q._triangle_distances(point, triangles, 1e-8)
        expected = np.array([np.sqrt(_point_triangle(point, tri)[0]) for tri in triangles])
        assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_all_chunks_examined_original_winding_once_and_arrays_unchanged(monkeypatch):
    v, f = tetrahedron(); before = (v.tobytes(), f.tobytes()); calls = []
    monkeypatch.setattr(q, "MAX_TRIANGLES_PER_CHUNK", 2)
    original_distance = q._triangle_distances
    def distance(p, triangles, tolerance):
        calls.append(("distance", len(triangles)))
        return original_distance(p, triangles, tolerance)
    def winding(p, triangles, tolerance):
        calls.append(("winding", len(triangles)))
        return _winding(p, triangles, tolerance)
    monkeypatch.setattr(q, "_triangle_distances", distance)
    monkeypatch.setattr(q, "_winding", winding)
    report = q.point_surface_query(np.array([.1, .1, .1]), v, f, expected_state="inside")
    assert report["status"] == "pass" and calls == [("distance", 2), ("distance", 2), ("winding", 4)]
    assert before == (v.tobytes(), f.tobytes())


@pytest.mark.parametrize("winding", [None, .5, -1., 2., 1. + 2e-6])
def test_ambiguous_or_wrong_winding_never_passes(monkeypatch, winding):
    v, f = tetrahedron(); monkeypatch.setattr(q, "_winding", lambda *args: winding)
    assert q.point_surface_query(np.array([.1, .1, .1]), v, f, expected_state="inside")["status"] == "fail"


def test_original_winding_integer_guard_is_unchanged(monkeypatch):
    v, f = tetrahedron(); monkeypatch.setattr(q, "_winding", lambda *args: 1. + .5e-6)
    assert q.point_surface_query(np.array([.1, .1, .1]), v, f, expected_state="inside")["status"] == "pass"


@pytest.mark.parametrize("phase", ["first", "second_chunk", "before_winding", "after_winding"])
def test_deadline_checked_before_each_chunk_and_around_winding(monkeypatch, phase):
    v, f = tetrahedron(); monkeypatch.setattr(q, "MAX_TRIANGLES_PER_CHUNK", 2)
    expiry = {"first": 0, "second_chunk": 2, "before_winding": 3, "after_winding": 4}[phase]
    calls = []
    def clock():
        index = len(calls); calls.append(index)
        return 2. if index >= expiry else 0.
    monkeypatch.setattr(q.time, "monotonic", clock)
    with pytest.raises(TimeoutError): q.point_surface_query(np.array([.1, .1, .1]), v, f, expected_state="inside", deadline=1.)


@pytest.mark.parametrize("fault", ["masked_point", "masked_vertices", "masked_faces", "point_nan", "vertex_inf",
    "point_integer", "vertex_integer", "point_shape", "face_float", "face_bool", "negative", "large",
    "repeated", "zero_area", "tiny_altitude", "huge_coordinates", "empty"])
def test_bad_geometry_rejected_without_dropping_any_faces(fault):
    v, f = tetrahedron(); p = np.array([.1, .1, .1])
    if fault == "masked_point": p = np.ma.array(p)
    elif fault == "masked_vertices": v = np.ma.array(v)
    elif fault == "masked_faces": f = np.ma.array(f)
    elif fault == "point_nan": p[0] = np.nan
    elif fault == "vertex_inf": v[0, 0] = np.inf
    elif fault == "point_integer": p = p.astype(np.int64)
    elif fault == "vertex_integer": v = v.astype(np.int64)
    elif fault == "point_shape": p = p[None]
    elif fault == "face_float": f = f.astype(float)
    elif fault == "face_bool": f = f.astype(bool)
    elif fault == "negative": f[0, 0] = -1
    elif fault == "large": f[0, 0] = 4
    elif fault == "repeated": f[0, 0] = f[0, 1]
    elif fault == "zero_area": v[2] = [.5, 0., 0.]
    elif fault == "tiny_altitude": v[2] = [.5, 1e-9, 0.]
    elif fault == "huge_coordinates": p += 1e10
    else: f = f[:0]
    with np.errstate(invalid="ignore", divide="ignore"):
        with pytest.raises(ValueError): q.point_surface_query(p, v, f, expected_state="inside")


@pytest.mark.parametrize("tolerance", [0., -1., np.nan, np.inf, True])
def test_invalid_tolerance(tolerance):
    v, f = tetrahedron()
    with pytest.raises(ValueError): q.point_surface_query(np.array([.1, .1, .1]), v, f, expected_state="inside", tolerance_m=tolerance)


def test_bad_expected_state_or_deadline():
    v, f = tetrahedron(); p = np.array([.1, .1, .1])
    with pytest.raises(ValueError): q.point_surface_query(p, v, f, expected_state="maybe")
    with pytest.raises(ValueError): q.point_surface_query(p, v, f, expected_state="inside", deadline=np.nan)
    with pytest.raises(TimeoutError): q.point_surface_query(p, v, f, expected_state="inside", deadline=time.monotonic() - 1)
