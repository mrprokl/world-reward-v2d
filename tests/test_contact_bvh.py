"""Tiny synthetic all-pairs references; no challenge inputs or GPU dependencies."""

import numpy as np
import pytest

from world_reward.contact_bvh import (
    ContactSearchBudgetExceeded, ContactSearchLimits, NativeContactBVH,
    native_point_triangle_squared_distance,
)


def exhaustive(points, vertices, faces, area=.005):
    values = np.asarray([[native_point_triangle_squared_distance(p, vertices[f], min_triangle_area=area)
                          for f in faces] for p in points])
    return float(values.min())


def test_large_triangle_interior_uses_native_regularized_normal():
    triangle = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    distance = native_point_triangle_squared_distance([.2, .2, 1.], triangle)
    assert distance == pytest.approx((1 / (1 + 1e-8)) ** 2, abs=1e-15)
    assert distance < 1.


def test_small_triangle_is_segments_not_surface_projection():
    triangle = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .02, 0.]])
    assert native_point_triangle_squared_distance([.005, .005, .01], triangle) == pytest.approx(.000125)
    assert native_point_triangle_squared_distance([.005, .005, .01], triangle, min_triangle_area=0) < .0001


@pytest.mark.parametrize('triangle', [np.zeros((3, 3)), [[0., 0., 0.], [1., 0., 0.], [.5, 0., 0.]]])
def test_degenerate_triangles_remain_eligible(triangle):
    vertices = np.array(triangle, dtype=float)
    faces = np.array([[0, 1, 2]])
    points = np.array([[.1, .2, .3], [0., 0., .2]])
    result = NativeContactBVH(vertices, faces).minimum(points)
    assert result.squared_distance == exhaustive(points, vertices, faces)
    assert result.eligible_points == 2 and result.eligible_faces == 1


@pytest.mark.parametrize('area', [0., .005, .1])
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_dual_tree_matches_every_pair_reference(area, seed):
    rng = np.random.default_rng(seed)
    vertices = rng.normal(size=(45, 3))
    faces = np.arange(45).reshape(-1, 3)
    points = rng.normal(size=(19, 3))
    limits = ContactSearchLimits(face_leaf_size=2, point_leaf_size=2)
    result = NativeContactBVH(vertices, faces, min_triangle_area=area, limits=limits).minimum(points)
    assert result.squared_distance == exhaustive(points, vertices, faces, area)
    assert result.search_complete and result.whole_input_hand_minimum
    assert not result.native_parity_verified


def test_regularized_barycentric_expansion_is_conservatively_covered():
    vertices = np.array([[0., 0., 0.], [.11, 0., 0.], [0., .11, 0.]])
    faces = np.array([[0, 1, 2]])
    # Slightly beyond the Euclidean triangle, but within +epsilon native barycentrics.
    points = np.array([[.0550001, .0550001, .01]])
    result = NativeContactBVH(vertices, faces).minimum(points)
    assert result.squared_distance == exhaustive(points, vertices, faces)
    assert result.squared_distance < .0001


def test_far_geometry_pruned_not_deleted_and_every_point_remains_eligible():
    vertices, faces = [], []
    for index in range(256):
        offset = float(index) * 10
        faces.append(np.arange(len(vertices), len(vertices) + 3))
        vertices.extend([[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]])
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    points = np.array([[.2, .2, .2], [2500.2, .2, .1], [9999., 1., 1.]])
    result = NativeContactBVH(vertices, faces, limits=ContactSearchLimits(face_leaf_size=2, point_leaf_size=1)).minimum(points)
    assert result.squared_distance == exhaustive(points, vertices, faces)
    assert result.point_index == 1 and result.face_index == 250
    assert result.pair_evaluations < len(points) * len(faces) // 8
    assert result.eligible_faces == 256 and result.eligible_points == 3


def test_seed_is_recomputed_not_frozen_and_cannot_override_new_winner():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [10., 0., 0.], [11., 0., 0.], [10., 1., 0.]])
    faces = np.arange(6).reshape(-1, 3)
    search = NativeContactBVH(vertices, faces, limits=ContactSearchLimits(face_leaf_size=1, point_leaf_size=1))
    first = search.minimum(np.array([[.2, .2, .1], [10.2, .2, 1.]]))
    second_points = np.array([[.2, .2, 2.], [10.2, .2, .01]])
    second = search.minimum(second_points, seed=(first.point_index, first.face_index))
    assert second.point_index == 1 and second.face_index == 1
    assert second.squared_distance == exhaustive(second_points, vertices, faces)


def test_ties_and_clamp_floor_are_reported_not_claimed_gradient_parity():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2], [0, 1, 2]])
    result = NativeContactBVH(vertices, faces).minimum(np.array([[.2, .2, 0.], [.3, .3, 0.]]))
    assert result.squared_distance == 0 and result.distance == 1e-6
    assert result.tied_witness and result.floor_ambiguous
    assert not result.native_parity_verified


def test_input_ownership_and_permutations_preserve_values():
    rng = np.random.default_rng(7)
    vertices = rng.normal(size=(30, 3)); faces = np.arange(30).reshape(-1, 3)
    points = rng.normal(size=(10, 3))
    original = vertices.copy(), faces.copy(), points.copy()
    search = NativeContactBVH(vertices, faces)
    a = search.minimum(points)
    b = NativeContactBVH(vertices, faces[::-1]).minimum(points[::-1])
    assert a.squared_distance == b.squared_distance
    for value, before in zip((vertices, faces, points), original): np.testing.assert_array_equal(value, before)
    vertices[:] = 999
    assert search.minimum(points).squared_distance == a.squared_distance
    assert not search.vertices.flags.writeable and not search.faces.flags.writeable


@pytest.mark.parametrize('limits', [ContactSearchLimits(max_pair_evaluations=1),
                                   ContactSearchLimits(max_node_pairs=1, face_leaf_size=1, point_leaf_size=1),
                                   ContactSearchLimits(max_heap_pairs=1, face_leaf_size=1, point_leaf_size=1),
                                   ContactSearchLimits(max_query_seconds=1e-30)])
def test_execution_limits_fail_closed(limits):
    rng = np.random.default_rng(2)
    vertices = rng.normal(size=(60, 3)); faces = np.arange(60).reshape(-1, 3)
    with pytest.raises(ContactSearchBudgetExceeded):
        NativeContactBVH(vertices, faces, limits=limits).minimum(rng.normal(size=(15, 3)))


def test_geometry_allocation_bound_fail_closed():
    with pytest.raises(ContactSearchBudgetExceeded):
        NativeContactBVH(np.eye(3), np.array([[0, 1, 2]]), limits=ContactSearchLimits(max_geometry_bytes=1))


def test_query_allocation_bound_fail_closed():
    search = NativeContactBVH(np.eye(3), np.array([[0, 1, 2]]), limits=ContactSearchLimits(max_query_bytes=1))
    with pytest.raises(ContactSearchBudgetExceeded): search.minimum([[0, 0, 0]])


def test_extreme_finite_coordinates_fail_without_overflow():
    with np.errstate(all='raise'), pytest.raises(ValueError):
        NativeContactBVH(np.eye(3), np.array([[0, 1, 2]])).minimum([[1e300, 0, 0]])


@pytest.mark.parametrize('kwargs', [{'max_pair_evaluations': True}, {'face_leaf_size': 65},
                                   {'point_leaf_size': 33}, {'max_query_seconds': np.inf},
                                   {'max_node_pairs': 0}])
def test_malformed_limits(kwargs):
    with pytest.raises(ValueError): ContactSearchLimits(**kwargs)


@pytest.mark.parametrize('vertices,faces', [(np.eye(3), [[0., 1., 2.]]),
                                          (np.eye(3), [[0, 1, 3]]),
                                          ([[np.nan, 0, 0]], [[0, 0, 0]])])
def test_malformed_geometry(vertices, faces):
    with pytest.raises(ValueError): NativeContactBVH(vertices, np.asarray(faces))


@pytest.mark.parametrize('points,seed', [([[np.nan, 0, 0]], None), ([], None),
                                       ([[0, 0, 0]], (True, 0)), ([[0, 0, 0]], (1, 0)),
                                       ([[0, 0, 0]], (0, 1))])
def test_malformed_queries(points, seed):
    with pytest.raises(ValueError):
        NativeContactBVH(np.eye(3), np.array([[0, 1, 2]])).minimum(points, seed=seed)
