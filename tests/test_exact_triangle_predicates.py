"""Data-free tests of a precision-only gate, never challenge predictions."""

from fractions import Fraction
import json
import math

import numpy as np
import pytest

from world_reward.exact_triangle_predicates import (
    ExactTriangleDegeneracyError,
    validate_exact_triangle_non_degeneracy,
)
import world_reward.exact_triangle_predicates as predicates


def _triangle():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]])


def _exact_oracle(triangle):
    points = [[Fraction.from_float(float(x)) for x in row] for row in triangle]
    a = [points[1][i] - points[0][i] for i in range(3)]
    b = [points[2][i] - points[0][i] for i in range(3)]
    return any(a[i] * b[j] != a[j] * b[i] for i, j in ((0, 1), (1, 2), (2, 0)))


def _unchanged(v, f, v_before, f_before):
    assert v.dtype == v_before.dtype and f.dtype == f_before.dtype
    assert v.shape == v_before.shape and f.shape == f_before.shape
    assert v.tobytes() == v_before.tobytes() and f.tobytes() == f_before.tobytes()


def test_regular_triangle_scalar_only_certificate_and_immutable_arrays():
    v, f = _triangle()
    before = v.copy(), f.copy()
    v.flags.writeable = f.flags.writeable = False
    report = validate_exact_triangle_non_degeneracy(v, f)
    assert report["status"] == "pass" and report["exact_noncollinearity_proven"] is True
    assert report["interval_certified_faces"] == 1
    assert report["exact_dyadic_fallback_faces"] == 0
    assert all(type(value) in (str, int, bool) for value in report.values())
    json.dumps(report, allow_nan=False)
    assert all(report[field] is False for field in (
        "area_or_length_tolerance_used", "input_arrays_modified",
        "faces_deleted_or_reordered", "components_selected_or_removed",
        "original_coordinate_cast_or_normalization", "pre_export_geometry_recovered",
        "topology_verified", "embedding_verified", "volume_or_fidelity_verified",
        "backend_numerical_readiness_verified", "challenge_performance_verified"))
    _unchanged(v, f, *before)


@pytest.mark.parametrize("plane", [(0, 1), (1, 2), (2, 0)])
def test_all_three_projected_planes(plane):
    v = np.zeros((3, 3))
    v[1, plane[0]] = v[2, plane[1]] = 1.
    report = validate_exact_triangle_non_degeneracy(v, [[0, 1, 2]])
    assert report["interval_certified_faces"] == 1


@pytest.mark.parametrize("scale", [2.**-600, 2.**-300, 2.**-30, 1., 2.**300, 2.**600])
def test_scale_invariance_including_norm_underflow_and_product_overflow(scale):
    v, f = _triangle()
    v *= scale
    before = v.copy(), f.copy()
    report = validate_exact_triangle_non_degeneracy(v, f)
    assert report["exact_noncollinearity_proven"] is True
    if scale == 2.**-600:
        assert report["exact_dyadic_fallback_faces"] == 1
    _unchanged(v, f, *before)


def test_true_coordinate_subtraction_overflow_is_resolved_exactly():
    maximum = np.finfo(np.float64).max
    v = np.array([[-maximum, 0., 0.], [maximum, maximum, 0.], [maximum, 0., maximum]])
    report = validate_exact_triangle_non_degeneracy(v, [[0, 1, 2]])
    assert report["exact_noncollinearity_proven"] is True
    assert _exact_oracle(v)


def test_positive_near_collinear_triangle_is_not_mislabeled_exactly_degenerate():
    v = np.array([[0., 0., 0.], [1., 1., 0.], [1., np.nextafter(1., 2.), 0.]])
    report = validate_exact_triangle_non_degeneracy(v, [[0, 1, 2]])
    assert report["exact_dyadic_fallback_faces"] == 1
    assert report["backend_numerical_readiness_verified"] is False


@pytest.mark.parametrize("v", [
    [[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]],
    [[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
    [[1e200, 1e200, 1e200], [2e200, 2e200, 2e200], [3e200, 3e200, 3e200]],
    [[0., 0., 0.], [1e-200, 1e-200, 1e-200], [2e-200, 2e-200, 2e-200]],
])
def test_exactly_collinear_or_coincident_positions_fail_whole_geometry(v):
    v, f = np.asarray(v), np.array([[0, 1, 2]])
    before = v.copy(), f.copy()
    with pytest.raises(ExactTriangleDegeneracyError, match="Exactly collinear"):
        validate_exact_triangle_non_degeneracy(v, f)
    _unchanged(v, f, *before)


@pytest.mark.parametrize("face", [[0, 0, 2], [0, 1, 0], [0, 1, 1]])
def test_repeated_indices_fail_without_returning_filtered_geometry(face):
    v, _ = _triangle()
    with pytest.raises(ExactTriangleDegeneracyError, match="Repeated face indices"):
        validate_exact_triangle_non_degeneracy(v, [face])


def test_quantization_loss_is_not_recovered_by_exact_predicates():
    v = np.array([[1e6, 1e6, 1e6], [1e6 + .01, 1e6, 1e6], [1e6, 1e6 + .01, 1e6]])
    assert validate_exact_triangle_non_degeneracy(v, [[0, 1, 2]])["status"] == "pass"
    # Simulate serialization independently, not as a repair in the gate.
    stored = v.astype(np.float32).astype(np.float64)
    before = stored.copy()
    with pytest.raises(ExactTriangleDegeneracyError, match="Exactly collinear"):
        validate_exact_triangle_non_degeneracy(stored, [[0, 1, 2]])
    assert stored.tobytes() == before.tobytes()


def test_translation_preserves_represented_exact_geometry_not_unrecoverable_values():
    v, f = _triangle()
    translated = v + [2.**40, -(2.**40), 2.**39]
    assert validate_exact_triangle_non_degeneracy(translated, f)["status"] == "pass"
    collapsed = v + [2.**60, 2.**60, 2.**60]
    with pytest.raises(ExactTriangleDegeneracyError):
        validate_exact_triangle_non_degeneracy(collapsed, f)


def _tetrahedron(scale=1., offset=(0., 0., 0.), inward=False):
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if inward:
        f = f[:, ::-1]
    return v * scale + offset, f


def test_disconnected_tiny_component_retains_every_face_without_global_area_cutoff():
    large_v, large_f = _tetrahedron()
    small_v, small_f = _tetrahedron(2.**-30, (2., 0., 0.))
    v, f = np.r_[large_v, small_v], np.r_[large_f, small_f + 4]
    before = v.copy(), f.copy()
    report = validate_exact_triangle_non_degeneracy(v, f)
    assert report["input_faces"] == 8 and report["components_selected_or_removed"] is False
    _unchanged(v, f, *before)


def test_inward_cavity_winding_is_untouched_but_containment_not_certified():
    outer_v, outer_f = _tetrahedron()
    inner_v, inner_f = _tetrahedron(.1, (.1, .1, .1), inward=True)
    v, f = np.r_[outer_v, inner_v], np.r_[outer_f, inner_f + 4]
    before = v.copy(), f.copy()
    report = validate_exact_triangle_non_degeneracy(v, f)
    assert report["embedding_verified"] is report["volume_or_fidelity_verified"] is False
    _unchanged(v, f, *before)


def test_intersecting_or_duplicate_surfaces_do_not_invent_topology_embedding_pass():
    v, f = _tetrahedron()
    duplicate_f = np.r_[f, f]
    report = validate_exact_triangle_non_degeneracy(v, duplicate_f)
    assert report["input_faces"] == 8
    assert report["topology_verified"] is report["embedding_verified"] is False


def test_one_degenerate_among_valid_triangles_rejects_whole_geometry():
    v, _ = _triangle()
    v = np.r_[v, [[2., 0., 0.]]]
    with pytest.raises(ExactTriangleDegeneracyError):
        validate_exact_triangle_non_degeneracy(v, [[0, 1, 2], [0, 1, 3]])


def test_seeded_adversarial_10k_faces_agree_with_independent_exact_oracle():
    # This fixture is entirely synthetic. Powers of two preserve intended
    # represented geometry across a wide range, including arithmetic zeros.
    rng = np.random.default_rng(7601)
    scales = np.ldexp(np.ones(10_000), rng.integers(-650, 651, size=10_000))
    points = np.zeros((10_000, 3, 3))
    points[:, 1, 0] = scales
    points[:, 2, 1] = scales
    vertices = points.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    before = vertices.copy(), faces.copy()
    assert all(_exact_oracle(triangle) for triangle in points)
    report = validate_exact_triangle_non_degeneracy(vertices, faces)
    assert report["input_faces"] == 10_000
    assert report["interval_certified_faces"] + report["exact_dyadic_fallback_faces"] == 10_000
    assert report["exact_dyadic_fallback_faces"] > 0
    _unchanged(vertices, faces, *before)


def test_random_cancellation_cases_agree_with_exact_oracle():
    rng = np.random.default_rng(7602)
    for _ in range(192):
        values = rng.integers(-8, 9, size=(3, 3)).astype(np.float64)
        exponent = int(rng.integers(-600, 601))
        values *= 2.**exponent
        if rng.integers(2):
            values[2] = values[0] + 2. * (values[1] - values[0])
        expected = _exact_oracle(values)
        if expected:
            assert validate_exact_triangle_non_degeneracy(values, [[0, 1, 2]])["status"] == "pass"
        else:
            with pytest.raises(ExactTriangleDegeneracyError):
                validate_exact_triangle_non_degeneracy(values, [[0, 1, 2]])


def test_outward_projection_intervals_enclose_exact_dyadic_determinants():
    # Verify the filter itself against exact values, not only final decisions.
    rng = np.random.default_rng(7603)
    values = np.ldexp(rng.normal(size=(2048, 3, 3)),
                      rng.integers(-1050, 1024, size=(2048, 1, 1)))
    with np.errstate(all="ignore"):
        edge1 = predicates._difference(values[:, 1], values[:, 0])
        edge2 = predicates._difference(values[:, 2], values[:, 0])
        intervals = [predicates._orientation_interval(edge1, edge2, plane)
                     for plane in predicates._PLANES]
    for triangle_index, triangle in enumerate(values):
        points = [[Fraction.from_float(float(x)) for x in point]
                  for point in triangle]
        for (i, j), (lower, upper) in zip(predicates._PLANES, intervals):
            exact = ((points[1][i] - points[0][i]) * (points[2][j] - points[0][j])
                     - (points[1][j] - points[0][j]) * (points[2][i] - points[0][i]))
            low, high = float(lower[triangle_index]), float(upper[triangle_index])
            if math.isnan(low) or math.isnan(high):
                continue  # NaN cannot certify a sign; exact fallback is required.
            assert (math.isinf(low) and low < 0) or Fraction.from_float(low) <= exact
            assert (math.isinf(high) and high > 0) or Fraction.from_float(high) >= exact


def test_nan_filter_intervals_fail_over_to_exact_not_unchecked_acceptance(monkeypatch):
    monkeypatch.setattr(predicates, "_orientation_interval", lambda e1, e2, plane:
                        (np.full(len(e1[0]), np.nan), np.full(len(e1[0]), np.nan)))
    v, f = _triangle()
    report = validate_exact_triangle_non_degeneracy(v, f)
    assert report["interval_certified_faces"] == 0
    assert report["exact_dyadic_fallback_faces"] == 1
    with pytest.raises(ExactTriangleDegeneracyError):
        validate_exact_triangle_non_degeneracy(v, [[0, 1, 1]])
    with pytest.raises(ExactTriangleDegeneracyError):
        validate_exact_triangle_non_degeneracy(np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), f)


def test_filter_fails_closed_when_gradual_underflow_precondition_is_missing(monkeypatch):
    original = predicates.np.multiply

    def flush(a, b):
        value = original(a, b)
        if np.ndim(value) == 0 and abs(value) < np.finfo(float).tiny:
            return np.float64(0.)
        return value

    monkeypatch.setattr(predicates.np, "multiply", flush)
    v, f = _triangle()
    with pytest.raises(RuntimeError, match="gradual underflow"):
        validate_exact_triangle_non_degeneracy(v, f)


def test_invalid_one_sided_nan_or_reversed_interval_cannot_certify_sign(monkeypatch):
    v = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    for low, high in ((np.nan, -1.), (1., np.nan), (1., -1.)):
        monkeypatch.setattr(predicates, "_orientation_interval", lambda a, b, p:
                            (np.full(len(a[0]), low), np.full(len(a[0]), high)))
        with pytest.raises(ExactTriangleDegeneracyError):
            validate_exact_triangle_non_degeneracy(v, [[0, 1, 2]])


@pytest.mark.parametrize("vertices", [
    [], np.zeros((3, 2)), np.zeros((2, 3)), np.zeros((3, 3), dtype=np.float32),
    np.zeros((3, 3), dtype=np.int64), np.zeros((3, 3), dtype=bool),
    np.zeros((3, 3), dtype=complex), [[0., 0., np.inf]] * 3,
    [[0., 0., np.nan]] * 3, np.ma.array(np.zeros((3, 3)), mask=False),
])
def test_invalid_or_implicitly_cast_positions_rejected(vertices):
    with pytest.raises(ValueError):
        validate_exact_triangle_non_degeneracy(vertices, [[0, 1, 2]])


@pytest.mark.parametrize("faces", [
    [], np.zeros((0, 3), dtype=int), [[0, 1]], [[0., 1., 2.]],
    [[False, True, True]], [[-1, 1, 2]], [[0, 1, 3]],
    np.array([[0, 1, 2**64 - 1]], dtype=np.uint64),
    np.ma.array([[0, 1, 2]], mask=False),
])
def test_invalid_indices_rejected(faces):
    v, _ = _triangle()
    with pytest.raises(ValueError):
        validate_exact_triangle_non_degeneracy(v, faces)


def test_non_native_endian_float64_and_unsigned_indices_preserved():
    v, f = _triangle()
    v, f = v.astype(">f8"), f.astype(np.uint32)
    before = v.copy(), f.copy()
    assert validate_exact_triangle_non_degeneracy(v, f)["status"] == "pass"
    _unchanged(v, f, *before)
