"""Tiny original first-frame queries -> unchanged tracks -> differentiable evidence."""
from dataclasses import replace
import inspect

import numpy as np
import pytest

from world_reward.fixed_shape_point_pose import PointTrackEvidence
from world_reward.joint_point_evidence import bind_joint_point_evidence
from world_reward.point_surface_queries import canonical_surface_queries


def frozen(a):
    result = np.array(a, copy=True); result.flags.writeable = False
    return result


def fixture():
    v = np.array([[-4., -3.13, 0.], [5., -3.13, 0.], [5., 4.07, 0.], [-4., 4.07, 0.]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    k = np.array([[32., 0, 16.], [0, 24., 12.], [0, 0, 1.]])
    queries = canonical_surface_queries(v, f, np.eye(3), np.array([0., 0., 2.]), k,
        np.ones((24, 32), bool), np.ones((24, 32), bool), image_width=32, image_height=24)
    xy = queries.query_points[:, [2, 1]]*[256/32, 256/24]
    native = dict(native_vertices=v, native_faces=f, K=k, image_size=(24, 32),
        frame_index=np.arange(4, dtype=np.int64), source_frame_ids=np.arange(1, 5, dtype=np.int64),
        native_frame_names=tuple(f'original_{i+1:06d}' for i in range(4)),
        source_references=('manufactured only: original automatic ray and native tracker bytes',))
    tracks = PointTrackEvidence(native['frame_index'], native['source_frame_ids'], np.arange(32, dtype=np.int64),
        queries.query_points, np.broadcast_to(xy, (4, 32, 2)), np.ones((4, 32), bool))
    return queries, tracks, native


def bind(data):
    queries, tracks, native = data
    return bind_joint_point_evidence(queries, tracks, **native)


def test_exact_same_attachments_tracks_timeline_and_source_ids_are_frozen_copies():
    data = fixture(); queries, tracks, native = data
    before = {name: a.copy() for name, a in vars(queries).items()}
    evidence = bind(data)
    assert evidence.source_frame_ids.tolist() == [1, 2, 3, 4] and evidence.query_points[0, 0] == 0
    for name in ('barycentric', 'face_indices'):
        np.testing.assert_array_equal(getattr(evidence, name), getattr(queries, name))
    for name in vars(tracks):
        np.testing.assert_array_equal(getattr(evidence, name), getattr(tracks, name))
    points = np.sum(evidence.vertices.astype(np.float64)[evidence.faces[evidence.face_indices]]
        * evidence.barycentric[:, :, None], axis=1)
    np.testing.assert_array_equal(points, queries.canonical_points)
    for name, a in vars(evidence).items():
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable
            with pytest.raises(ValueError): a.flags.writeable = True
            for original in (*vars(queries).values(), *vars(tracks).values()):
                assert not np.shares_memory(a, original)
    for name, a in before.items(): np.testing.assert_array_equal(getattr(queries, name), a)
    native['native_vertices'][:] = 99
    assert evidence.vertices[0, 0] == -4.


def test_query_generation_precedes_future_evidence_and_never_reselects_missing_slots():
    data = fixture(); queries, tracks, native = data
    points_before = queries.canonical_points.copy(); bary_before = queries.barycentric.copy()
    future = tracks.tracks_256.copy(); future[1:] += .125
    missing = replace(tracks, tracks_256=future, native_visible=np.zeros((4, 32), bool))
    first = bind(data); changed = bind((queries, missing, native))
    np.testing.assert_array_equal(first.barycentric, changed.barycentric)
    np.testing.assert_array_equal(queries.barycentric, bary_before)
    np.testing.assert_array_equal(queries.canonical_points, points_before)
    assert changed.query_ids.tolist() == list(range(32)) and not changed.support_counts().any()
    assert first.evidence_sha256 != changed.evidence_sha256


def test_different_f64_selector_mesh_is_not_heuristically_cast_or_refitted():
    data = fixture(); queries, tracks, native = data
    v = native['native_vertices'].astype(np.float64); v[0, 0] += 1e-10
    original = canonical_surface_queries(v, native['native_faces'], np.eye(3), np.array([0., 0., 2.]), native['K'],
        np.ones((24, 32), bool), np.ones((24, 32), bool), image_width=32, image_height=24)
    with pytest.raises(ValueError, match='canonical attachment'):
        bind((original, tracks, native))
    with pytest.raises(ValueError, match='native vertices'):
        bind((queries, tracks, {**native, 'native_vertices': v}))


@pytest.mark.parametrize('fault', ['indices', 'short', 'source', 'names', 'query_reorder',
    'mesh_value', 'faces_value', 'bary_value', 'bary_f32', 'point_value', 'grid', 'grid_order',
    'depth', 'masked_grid', 'query_time', 'readonly_queries', 'readonly_tracks', 'nan', 'outgrid_invisible'])
def test_malformed_or_changed_native_and_original_evidence_fails_whole_binding(fault):
    queries, tracks, native = fixture()
    if fault == 'indices': native['frame_index'] = np.array([0, 1, 3, 4], np.int64)
    elif fault == 'short': native['frame_index'] = np.arange(3, dtype=np.int64)
    elif fault == 'source': native['source_frame_ids'] = np.arange(5, 9, dtype=np.int64)
    elif fault == 'names': native['native_frame_names'] = native['native_frame_names'][:-1]
    elif fault == 'query_reorder': tracks = replace(tracks, query_points=tracks.query_points[::-1])
    elif fault == 'mesh_value': native['native_vertices'] = native['native_vertices']+np.float32(.001)
    elif fault == 'faces_value': native['native_faces'] = native['native_faces'][:, ::-1]
    elif fault == 'bary_value':
        a = queries.barycentric.copy(); a[0, 0] += .0001; a[0, 1] -= .0001
        queries = replace(queries, barycentric=frozen(a))
    elif fault == 'bary_f32': queries = replace(queries, barycentric=frozen(queries.barycentric.astype(np.float32)))
    elif fault == 'point_value': queries = replace(queries, canonical_points=frozen(queries.canonical_points+.001))
    elif fault == 'grid': queries = replace(queries, grid_indices=frozen(queries.grid_indices.astype(np.int32)))
    elif fault == 'grid_order':
        changes = {name: frozen(a[::-1]) for name, a in vars(queries).items()}
        queries = replace(queries, **changes)
        tracks = replace(tracks, query_points=tracks.query_points[::-1], tracks_256=tracks.tracks_256[:, ::-1])
    elif fault == 'depth': queries = replace(queries, camera_depth_m=frozen(np.zeros(32)))
    elif fault == 'masked_grid':
        a = np.ma.array(queries.grid_indices, mask=False); a.flags.writeable = False
        queries = replace(queries, grid_indices=a)
    elif fault == 'query_time':
        a = tracks.query_points.copy(); a[:, 0] = 1; tracks = replace(tracks, query_points=a)
    elif fault == 'readonly_queries': queries.barycentric.flags.writeable = True
    elif fault == 'readonly_tracks': tracks.native_visible.flags.writeable = True
    elif fault == 'nan':
        a = tracks.tracks_256.copy(); a[2, 0, 0] = np.nan
        with pytest.raises(ValueError): replace(tracks, tracks_256=a)
        return
    else:
        a = tracks.tracks_256.copy(); a[2, 0, 0] = 256
        tracks = replace(tracks, tracks_256=a, native_visible=np.zeros((4, 32), bool))
    with pytest.raises(ValueError): bind((queries, tracks, native))


def test_adapter_has_no_refit_tracker_renderer_torch_or_future_selection():
    source = inspect.getsource(bind_joint_point_evidence)
    assert 'np.sum(' in source and 'astype(np.float64)' in source
    for forbidden in ('lstsq', 'pinv', 'canonical_surface_queries(', 'select_query_indices(',
                      'import torch', 'track(', 'clip(', 'argsort('):
        assert forbidden not in source
