"""Own tiny analytic NumPy geometry and API contracts; no Torch/GPU execution."""

import builtins
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward.continuous_surface import (
    _torch_distances, numpy_reference_distance_squared, observed_to_triangle_distance_squared, validate_numpy_geometry,
)


def fixture(dtype=np.float64):
    triangles = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]], dtype=dtype)
    points = np.array([[.0025, .0025, .02], [.005, .005, .02], [-.005, -.004, .02],
                       [.0075, .0075, .02]], dtype=dtype)
    closest = np.array([[.0025, .0025, 0.], [.005, .005, 0.], [0., 0., 0.], [.005, .005, 0.]], dtype=dtype)
    return points, triangles, closest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interior_boundary_vertex_and_outside_hypotenuse_true_continuous_distance(dtype):
    points, triangles, closest = fixture(dtype)
    expected = np.sum((points.astype(float) - closest.astype(float))**2, axis=-1)
    actual = numpy_reference_distance_squared(points, triangles)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-18)
    assert actual[-1] > actual[0]  # Does not misclassify outside hypotenuse as face interior.


@pytest.mark.parametrize("scale", [1e-5, 1., 1e3])
def test_tiny_and_large_metric_triangle_preserve_squared_units_without_regularizer(scale):
    points, triangles, closest = fixture()
    np.testing.assert_allclose(numpy_reference_distance_squared(points * scale, triangles * scale),
                               np.sum((points - closest)**2, axis=-1) * scale**2, rtol=1e-14, atol=1e-30)


def test_faces_winding_duplication_and_singleton_do_not_change_geometry():
    points, triangles, _ = fixture()
    expected = numpy_reference_distance_squared(points, triangles)
    np.testing.assert_array_equal(numpy_reference_distance_squared(points, np.repeat(triangles, 2, axis=0)), expected)
    np.testing.assert_array_equal(numpy_reference_distance_squared(points, triangles[:, ::-1]), expected)
    assert numpy_reference_distance_squared(points[:1], triangles)[0] == expected[0]


def test_nearest_face_really_selected_and_closed_segment_endpoints():
    triangles = np.array([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]],
                          [[0., 0., .03], [.01, 0., .03], [0., .01, .03]]])
    points = np.array([[.0025, .0025, .02], [.015, 0., .03], [0., .015, .03]])
    np.testing.assert_allclose(numpy_reference_distance_squared(points, triangles), [.0001, .000025, .000025], rtol=1e-14)


def test_rigid_equivariance_camera_metres_no_independent_alignment():
    points, triangles, _ = fixture()
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
    translation = np.array([.03, -.04, 1.])
    np.testing.assert_allclose(numpy_reference_distance_squared(points @ rotation.T + translation, triangles @ rotation.T + translation),
                               numpy_reference_distance_squared(points, triangles), atol=1e-17, rtol=1e-12)


def test_independent_analytic_point_gradient_and_interior_triangle_gradient_finite_difference():
    points, triangles, closest = fixture()
    expected_point = 2 * (points - closest)
    h = 1e-7
    for row in range(len(points)):
        for axis in range(3):
            offset = np.eye(3)[axis] * h
            fd = (numpy_reference_distance_squared(points[row:row + 1] + offset, triangles)[0]
                  - numpy_reference_distance_squared(points[row:row + 1] - offset, triangles)[0]) / (2 * h)
            # Exact boundary crosses regions: central difference has O(h)
            # truncation although the squared-distance first derivative agrees.
            assert fd == pytest.approx(expected_point[row, axis], abs=.26 * h if row == 1 else 2e-11)
    expected_tri = -np.array([.5, .25, .25])[:, None] * expected_point[0]
    for vertex in range(3):
        for axis in range(3):
            offset = np.zeros_like(triangles); offset[0, vertex, axis] = h
            fd = (numpy_reference_distance_squared(points[:1], triangles + offset)[0]
                  - numpy_reference_distance_squared(points[:1], triangles - offset)[0]) / (2 * h)
            assert fd == pytest.approx(expected_tri[vertex, axis], abs=2e-10)


@pytest.mark.parametrize("field", [0, 1])
@pytest.mark.parametrize("failure", ["empty", "shape", "integer", "bool", "float16", "nan", "inf", "masked"])
def test_invalid_numpy_inputs_fail_not_filtered(field, failure):
    values = list(fixture()[:2])
    value = values[field]
    if failure == "empty": value = value[:0]
    if failure == "shape": value = value[..., :2]
    if failure == "integer": value = value.astype(int)
    if failure == "bool": value = value.astype(bool)
    if failure == "float16": value = value.astype(np.float16)
    if failure in ("nan", "inf"): value = value.copy(); value.flat[0] = np.nan if failure == "nan" else np.inf
    if failure == "masked": value = np.ma.array(value, mask=False)
    values[field] = value
    with pytest.raises(ValueError): validate_numpy_geometry(*values)


@pytest.mark.parametrize("triangle", [np.zeros((1, 3, 3)),
                                       np.array([[[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]]]),
                                       np.array([[[0., 0., 0.], [1e-100, 0., 0.], [0., 1e-100, 0.]]]),
                                       np.array([[[0., 0., 0.], [1e100, 0., 0.], [0., 1e100, 0.]]])])
def test_degenerate_underflow_and_overflow_geometries_fail_without_epsilon_repair(triangle):
    with pytest.raises(ValueError, match="nonzero edges/area"):
        validate_numpy_geometry(fixture()[0], triangle)


def test_numpy_copies_no_mutation_and_dtype_mismatch_rejected():
    points, triangles, _ = fixture(np.float32)
    converted = validate_numpy_geometry(points, triangles)
    for before, after in zip((points, triangles), converted, strict=True):
        assert after.dtype == np.float64 and not np.shares_memory(before, after)
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="same input dtype"):
        validate_numpy_geometry(points, triangles.astype(np.float64))


@pytest.mark.parametrize("chunk", [False, True, 0, -1, 1.5, "64", None])
def test_invalid_chunk_fails_before_any_torch_import(chunk, monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == "torch": pytest.fail("No heavy import for invalid chunk")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ValueError, match="positive integer"):
        observed_to_triangle_distance_squared(None, None, point_chunk_size=chunk)
    with pytest.raises(ValueError): validate_numpy_geometry(*fixture()[:2], point_chunk_size=chunk)


@pytest.mark.parametrize("chunk", [1, 64, 100, np.int64(2)])
def test_numpy_reference_chunk_contract_does_not_change_results(chunk):
    values = fixture()[:2]
    np.testing.assert_array_equal(numpy_reference_distance_squared(*values, point_chunk_size=chunk), numpy_reference_distance_squared(*values))


def test_lazy_runtime_has_fixed_default_and_no_vendor_import_or_geometry_regularizer():
    assert inspect.signature(observed_to_triangle_distance_squared).parameters["point_chunk_size"].default == 64
    source = inspect.getsource(observed_to_triangle_distance_squared)
    assert "import torch" in source and "torch.no_grad()" in source
    assert "argmin(dim=1)" in source and "faces[indices]" in source
    assert "pytorch3d" not in source and "kaolin" not in source


class NumpyTensor(np.ndarray):
    """Only arithmetic ops needed to check production geometry without Torch."""
    def __new__(cls, value): return np.asarray(value).view(cls)
    def unbind(self, dim): return tuple(NumpyTensor(value) for value in np.moveaxis(self, dim, 0))
    def sum(self, dim): return NumpyTensor(np.sum(np.asarray(self), axis=dim))
    def min(self, dim): return SimpleNamespace(values=NumpyTensor(np.min(np.asarray(self), axis=dim)))
    def square(self): return NumpyTensor(np.square(np.asarray(self)))
    def clamp(self, low, high): return NumpyTensor(np.clip(np.asarray(self), low, high))


def numpy_torch():
    return SimpleNamespace(
        isfinite=lambda value: np.isfinite(value),
        cross=lambda a, b, dim: NumpyTensor(np.cross(np.asarray(a), np.asarray(b), axis=dim)),
        stack=lambda values, dim: NumpyTensor(np.stack(values, axis=dim)),
        where=lambda condition, first, second: NumpyTensor(np.where(condition, first, second)))


def test_production_piecewise_broadcast_and_paired_formulas_match_independent_reference():
    points, triangles, _ = fixture()
    second_face = triangles + [0, 0, .03]
    faces = np.concatenate((triangles, second_face))
    p, t = NumpyTensor(points), NumpyTensor(faces)
    all_distances = np.asarray(_torch_distances(numpy_torch(), p[:, None], t[None]))
    assert all_distances.shape == (len(p), len(t))
    selected = np.argmin(all_distances, axis=1)
    paired = np.asarray(_torch_distances(numpy_torch(), p, t[selected]))
    np.testing.assert_allclose(paired, numpy_reference_distance_squared(points, faces), atol=1e-18, rtol=1e-14)
    duplicated = NumpyTensor(np.repeat(triangles, 2, axis=0))
    assert np.argmin(_torch_distances(numpy_torch(), p[:, None], duplicated[None]), axis=1).tolist() == [0] * len(p)


def test_production_formula_point_and_triangle_derivatives_match_independent_analytic():
    points, triangles, closest = fixture()
    p, t = NumpyTensor(points[:1]), NumpyTensor(triangles)
    expected = 2 * (points[0] - closest[0])
    expected_tri = -np.array([.5, .25, .25])[:, None] * expected
    h = 1e-7
    for axis in range(3):
        delta = np.eye(3)[axis] * h
        fd = (_torch_distances(numpy_torch(), p + delta, t)[0] - _torch_distances(numpy_torch(), p - delta, t)[0]) / (2 * h)
        assert fd == pytest.approx(expected[axis], abs=2e-11)
    for vertex in range(3):
        for axis in range(3):
            delta = np.zeros_like(t); delta[0, vertex, axis] = h
            fd = (_torch_distances(numpy_torch(), p, t + delta)[0] - _torch_distances(numpy_torch(), p, t - delta)[0]) / (2 * h)
            assert fd == pytest.approx(expected_tri[vertex, axis], abs=2e-10)


def test_thin_nondegenerate_triangle_avoids_gram_determinant_cancellation():
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [1., 1e-10, 0.]]])
    point = np.array([[.75, .25e-10, .02]])
    # Float64 Gram subtraction collapses although cross-product area is nonzero.
    a, b = triangle[0, 1], triangle[0, 2]
    assert (a @ a) * (b @ b) - (a @ b)**2 == 0.
    actual = _torch_distances(numpy_torch(), NumpyTensor(point), NumpyTensor(triangle))
    assert actual[0] == pytest.approx(.0004, rel=1e-14)
    assert numpy_reference_distance_squared(point, triangle)[0] == pytest.approx(.0004, rel=1e-14)
