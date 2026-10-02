"""Tiny synthetic surfaces; preserve cavity faces and meaningful thin geometry."""

import json

import numpy as np
import pytest

from world_reward.mesh_geometry import normalize_degenerate_faces


def tetrahedron():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return vertices, faces


def signed_volume(vertices, faces):
    triangles = vertices[faces]
    return np.einsum("fc,fc->f", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6


def test_normal_surface_and_original_face_order_are_preserved():
    vertices, faces = tetrahedron()
    original_vertices, original_faces = vertices.copy(), faces.copy()
    active, report = normalize_degenerate_faces(vertices, faces)
    np.testing.assert_array_equal(active, np.arange(4))
    np.testing.assert_array_equal(vertices, original_vertices)
    np.testing.assert_array_equal(faces, original_faces)
    assert report["excluded_faces"] == 0 and report["closure_winding_volume_verified"] is False
    json.dumps(report, allow_nan=False)


def test_padding_and_exact_collinear_faces_are_render_only_exclusions():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 0, 0], [0, 1, 3], [1, 1, 2]])
    original = faces.copy()
    active, report = normalize_degenerate_faces(vertices, faces)
    np.testing.assert_array_equal(active, [0])
    np.testing.assert_array_equal(faces, original)
    assert report["repeated_index_faces"] == 2 and report["exact_zero_area_faces"] == 1
    assert report["excluded_original_face_indices"] == [1, 2, 3]


def test_roundoff_level_sliver_is_excluded_but_resolvable_sliver_is_preserved():
    eps = np.finfo(np.float64).eps
    vertices = np.array([[0., 0, 0], [1., 0, 0], [1., 8 * eps, 0],
                         [1., 128 * eps, 0], [0., 1., 0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    active, report = normalize_degenerate_faces(vertices, faces)
    np.testing.assert_array_equal(active, [1, 2])
    assert report["numerically_collapsed_faces"] == 1
    assert report["numerically_collapsed_original_face_indices"] == [0]


def test_meaningful_thin_triangle_below_float32_precision_threshold_is_retained():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1e-10, 0]], dtype=float)
    active, report = normalize_degenerate_faces(vertices, [[0, 1, 2]])
    np.testing.assert_array_equal(active, [0])
    assert report["excluded_faces"] == 0


def test_small_well_shaped_component_is_not_removed_by_global_area_threshold():
    outer_vertices, outer_faces = tetrahedron()
    tiny_vertices = np.array([[.2, .2, .2], [.2 + 1e-10, .2, .2], [.2, .2 + 1e-10, .2]])
    vertices = np.concatenate((outer_vertices, tiny_vertices))
    faces = np.concatenate((outer_faces, [[4, 5, 6]]))
    active, report = normalize_degenerate_faces(vertices, faces)
    np.testing.assert_array_equal(active, np.arange(5))
    assert report["excluded_faces"] == 0


def test_closed_outer_and_inward_cavity_shell_preserve_signed_volumes():
    outer, faces = tetrahedron()
    inner = outer * .2 + [.1, .1, .1]
    all_vertices = np.concatenate((outer, inner))
    all_faces = np.concatenate((faces, faces[:, ::-1] + 4, [[0, 0, 0]]))
    expected_inner = signed_volume(all_vertices, all_faces[4:8])
    assert expected_inner < 0
    active, report = normalize_degenerate_faces(all_vertices, all_faces)
    np.testing.assert_array_equal(active, np.arange(8))
    assert signed_volume(all_vertices, all_faces[active][:4]) > 0
    assert signed_volume(all_vertices, all_faces[active][4:]) == expected_inner
    assert report["repeated_index_faces"] == 1 and report["components_selected_or_removed"] is False


def test_tiny_closed_cavity_is_not_deleted_due_to_small_volume():
    outer, faces = tetrahedron()
    tiny = outer * 1e-10 + [.1, .1, .1]
    vertices = np.concatenate((outer, tiny))
    combined_faces = np.concatenate((faces, faces[:, ::-1] + 4))
    active, _ = normalize_degenerate_faces(vertices, combined_faces)
    np.testing.assert_array_equal(active, np.arange(8))


@pytest.mark.parametrize("scale", [1e-50, 1e-6, 100., 1e50])
def test_index_selection_is_uniform_scale_invariant(scale):
    eps = np.finfo(np.float64).eps
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 8 * eps, 0], [1, 128 * eps, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    active, _ = normalize_degenerate_faces(vertices * scale, faces)
    np.testing.assert_array_equal(active, [1, 2])


def test_regular_translated_geometry_is_not_normalized_in_place():
    vertices, faces = tetrahedron()
    vertices += [1e6, -1e6, 1e6]
    original = vertices.copy()
    active, _ = normalize_degenerate_faces(vertices, faces)
    np.testing.assert_array_equal(active, np.arange(4))
    np.testing.assert_array_equal(vertices, original)


def test_all_collapsed_faces_return_empty_view_not_invented_geometry():
    vertices = np.ones((3, 3))
    active, report = normalize_degenerate_faces(vertices, [[0, 1, 2], [0, 0, 0]])
    assert active.dtype == np.int64 and len(active) == 0
    assert report["exact_zero_area_faces"] == 1 and report["repeated_index_faces"] == 1
    assert report["minimum_active_relative_cross_product"] is None
    json.dumps(report, allow_nan=False)


def test_empty_face_array_is_an_empty_view():
    active, report = normalize_degenerate_faces(np.ones((1, 3)), np.zeros((0, 3), dtype=int))
    assert len(active) == 0 and report["input_faces"] == 0


@pytest.mark.parametrize("vertices", [[], np.ones((3, 2)), np.ones((3, 3), dtype=bool),
                                      np.ones((3, 3), dtype=complex), [[0, 0, np.inf]],
                                      [[0, 0, np.nan]], np.ma.array(np.ones((3, 3)), mask=False)])
def test_invalid_positions_fail(vertices):
    with pytest.raises(ValueError):
        normalize_degenerate_faces(vertices, [[0, 1, 2]])


@pytest.mark.parametrize("faces", [[[0, 1]], [[0., 1., 2.]], [[False, True, True]],
                                   [[-1, 1, 2]], [[0, 1, 3]],
                                   np.ma.array([[0, 1, 2]], mask=False)])
def test_invalid_indices_fail(faces):
    vertices, _ = tetrahedron()
    with pytest.raises(ValueError):
        normalize_degenerate_faces(vertices[:3], faces)


def test_extent_overflow_is_not_silently_repaired():
    with pytest.raises(ValueError, match="extent"):
        normalize_degenerate_faces([[-1e308, 0, 0], [1e308, 0, 0], [0, 1, 0]], [[0, 1, 2]])
