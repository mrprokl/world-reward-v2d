"""Data-free analytic shapes; no fitting or challenge-accuracy assertions."""

import numpy as np
import pytest
from scipy.linalg import logm
from scipy.spatial.transform import Rotation

from world_reward.shape_model import (
    DEFAULT_MAX_ABS_LOG_STRETCH,
    apply_fixed_shape,
    deformation,
)


def asymmetric_box():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    vertices = vertices * [.4, .7, 1.2] + [.31, -.27, .83]
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                      [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    return vertices, faces


def signed_volume(vertices, faces):
    # Translation-insensitive arithmetic, with no mesh library dependency.
    triangles = (vertices - vertices.mean(axis=0))[faces]
    return np.einsum("fc,fc->f", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6


def test_identity_and_exact_diagonal_proportions_preserve_metric_volume():
    np.testing.assert_array_equal(deformation(np.zeros(5)), np.eye(3))
    parameters = np.array([np.log(1.2), np.log(.9), 0, 0, 0])
    expected = np.diag([1.2, .9, 1 / (1.2 * .9)])
    np.testing.assert_allclose(deformation(parameters), expected, atol=1e-15, rtol=1e-15)
    vertices, faces = asymmetric_box()
    pivot = vertices.mean(axis=0)
    shaped = apply_fixed_shape(vertices, parameters, centroid=pivot)
    assert signed_volume(vertices, faces) == pytest.approx(.4 * .7 * 1.2)
    assert signed_volume(shaped, faces) == pytest.approx(signed_volume(vertices, faces), rel=1e-14)
    np.testing.assert_allclose(np.ptp(shaped, axis=0), [.4 * 1.2, .7 * .9, 1.2 / 1.08], atol=1e-14)
    np.testing.assert_allclose(shaped.mean(axis=0), pivot, atol=1e-15)


def test_symmetric_exponential_spd_det_one_not_coordinatewise_stretch():
    parameters = np.array([.1, -.07, .06, -.04, .05])
    matrix = deformation(parameters)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-15)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert np.linalg.det(matrix) == pytest.approx(1., rel=1e-14)
    expected_log = np.array([[.1, .06, -.04], [.06, -.07, .05], [-.04, .05, -.03]])
    np.testing.assert_allclose(logm(matrix), expected_log, atol=1e-14)
    assert np.linalg.eigvalsh(matrix).max() <= 1.5
    assert np.linalg.eigvalsh(matrix).min() >= 1 / 1.5


def test_inverse_and_non_origin_supplied_pivot_are_exact_contracts():
    vertices, _ = asymmetric_box()
    parameters = np.array([.1, -.07, .06, -.04, .05])
    pivot = np.array([.17, -.13, .23])  # Supplied, not silently recomputed.
    shaped = apply_fixed_shape(vertices, parameters, centroid=pivot)
    expected = (vertices - pivot) @ deformation(parameters).T + pivot
    np.testing.assert_allclose(shaped, expected, atol=1e-15, rtol=0)
    recovered = apply_fixed_shape(shaped, -parameters, centroid=pivot)
    np.testing.assert_allclose(recovered, vertices, atol=1e-15, rtol=1e-15)
    np.testing.assert_allclose(deformation(-parameters) @ deformation(parameters), np.eye(3), atol=1e-15)
    point = apply_fixed_shape(pivot[None], parameters, centroid=pivot)
    np.testing.assert_array_equal(point, pivot[None])
    assert not np.allclose(shaped, apply_fixed_shape(vertices, parameters, centroid=vertices.mean(axis=0)))


def test_closed_oriented_box_and_inward_cavity_keep_faces_and_signed_volumes():
    outer, faces = asymmetric_box()
    pivot = outer.mean(axis=0)
    cavity = (outer - pivot) * .4 + pivot
    vertices = np.concatenate([outer, cavity, outer[:1]])  # Harmless vertex padding.
    all_faces = np.concatenate([faces, faces[:, ::-1] + 8, [[0, 0, 0]]])
    face_copy, vertex_copy = all_faces.copy(), vertices.copy()
    params = np.array([.16, -.1, .03, -.05, .02])
    shaped = apply_fixed_shape(vertices, params, centroid=pivot)
    assert signed_volume(vertices, all_faces[:12]) > 0
    assert signed_volume(vertices, all_faces[12:24]) < 0
    for selection in (slice(0, 12), slice(12, 24), slice(None)):
        assert signed_volume(shaped, all_faces[selection]) == pytest.approx(
            signed_volume(vertices, all_faces[selection]), rel=1e-14)
    triangles = shaped[all_faces[:24]]
    assert (np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) > 0).all()
    # The closed simplicial complex is unchanged; no cavity deletion/inversion.
    edges = np.sort(np.concatenate([all_faces[:24, [0, 1]], all_faces[:24, [1, 2]], all_faces[:24, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    np.testing.assert_array_equal(all_faces, face_copy)
    np.testing.assert_array_equal(vertices, vertex_copy)
    np.testing.assert_array_equal(shaped[-1], shaped[0])


def test_inputs_not_changed_and_output_does_not_alias_vertices_or_parameters():
    vertices, _ = asymmetric_box()
    params = np.array([.1, -.05, .02, .01, -.01])
    pivot = vertices.mean(axis=0)
    before = [array.copy() for array in (vertices, params, pivot)]
    shaped = apply_fixed_shape(vertices, params, centroid=pivot)
    shaped[:] = 0
    for original, copy in zip((vertices, params, pivot), before):
        np.testing.assert_array_equal(original, copy)
    unchanged = apply_fixed_shape(vertices, np.zeros(5), centroid=pivot)
    assert unchanged is not vertices and not np.shares_memory(unchanged, vertices)


def test_fixed_shape_translation_equivariance_and_canonical_basis_dependence():
    vertices, _ = asymmetric_box()
    pivot = vertices.mean(axis=0)
    params = np.array([.13, -.07, .02, -.01, .03])
    translation = np.array([.7, -.2, .6])
    shaped = apply_fixed_shape(vertices, params, centroid=pivot)
    np.testing.assert_allclose(apply_fixed_shape(vertices + translation, params, centroid=pivot + translation),
                               shaped + translation, atol=1e-15)
    rotation = Rotation.from_euler("z", .7).as_matrix()
    # Changing the canonical basis without also conjugating the log-stretch
    # changes this proposal; an optimizer cannot treat those gauges as separate.
    rotated_input = apply_fixed_shape(vertices @ rotation.T, params, centroid=pivot @ rotation.T)
    assert not np.allclose(rotated_input, shaped @ rotation.T)


def test_spectral_bound_not_parameterwise_clipping_and_boundary_is_allowed():
    limit = DEFAULT_MAX_ABS_LOG_STRETCH
    parameters = np.array([limit, 0, 0, 0, 0])
    np.testing.assert_allclose(np.linalg.eigvalsh(deformation(parameters)), [1 / 1.5, 1, 1.5])
    np.testing.assert_array_equal(deformation(np.zeros(5), max_abs_log_stretch=0), np.eye(3))
    with pytest.raises(ValueError, match="no clipping"):
        deformation([.3, .3, 0, 0, 0])  # Each param<limit, but third eigenvalue=-.6.
    with pytest.raises(ValueError, match="no clipping"):
        deformation([0, 0, .3, .3, .3])  # Offdiagonal coupling eigenvalue=.6.
    with pytest.raises(ValueError, match="no clipping"):
        deformation([limit + 1e-8, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="no clipping"):
        deformation([.01, 0, 0, 0, 0], max_abs_log_stretch=0)
    assert np.linalg.det(deformation([.5, 0, 0, 0, 0], max_abs_log_stretch=.6)) == pytest.approx(1.)


@pytest.mark.parametrize("parameters", [np.eye(3), np.diag([-1., 1, 1]), np.zeros((1, 5)),
                                      [], [0] * 6, [0, 0, 0, 0, np.nan], [0, 0, 0, 0, np.inf],
                                      [True] * 5, [1j] * 5, ["0"] * 5, np.ma.array(np.zeros(5), mask=False)])
def test_bad_shape_parameter_arrays_and_reflection_inputs_fail(parameters):
    with pytest.raises(ValueError):
        deformation(parameters)


@pytest.mark.parametrize("limit", [-1, np.nan, np.inf, True, "1", [1], np.ma.array(.3, mask=False)])
def test_bad_spectral_limits_fail(limit):
    with pytest.raises(ValueError):
        deformation(np.zeros(5), max_abs_log_stretch=limit)


@pytest.mark.parametrize("vertices", [np.empty((0, 3)), np.zeros((3,)), np.zeros((2, 2)),
                                     np.zeros((2, 3, 1)), [[0, 0, np.nan]], [[0, 0, np.inf]],
                                     [[True, False, True]], [[1j, 0, 0]], [["0", "0", "0"]],
                                     np.ma.array(np.zeros((3, 3)), mask=False)])
def test_bad_vertex_arrays_fail(vertices):
    with pytest.raises(ValueError):
        apply_fixed_shape(vertices, np.zeros(5), centroid=np.zeros(3))


@pytest.mark.parametrize("pivot", [np.zeros((1, 3)), [0, 0], [0, 0, np.nan], [0, 0, np.inf],
                                  [True] * 3, ["0"] * 3, np.ma.array(np.zeros(3), mask=False)])
def test_bad_or_masked_supplied_centroids_fail(pivot):
    with pytest.raises(ValueError):
        apply_fixed_shape(np.zeros((3, 3)), np.zeros(5), centroid=pivot)


def test_finite_inputs_with_overflowing_trace_or_pivot_difference_fail():
    with pytest.raises(ValueError, match="finite float64"):
        deformation([1e308, 1e308, 0, 0, 0])
    with pytest.raises(ValueError, match="finite float64"):
        apply_fixed_shape([[1e308, 0, 0]], np.zeros(5), centroid=[-1e308, 0, 0])
