from dataclasses import replace
import gc
import inspect
import weakref

import numpy as np
import pytest

from world_reward import fixed_shape_point_pose as op
from world_reward.point_candidate_pool import NativeCandidatePool
from world_reward.point_pose_comparison import compare
from world_reward.point_surface_queries import canonical_surface_queries
from world_reward.rigid_alignment import RigidAlignment


def fixture(frames=3, height=24, width=32):
    v = np.array([[-4., -3.13, 0.], [5., -3.13, 0.], [5., 4.07, 0.], [-4., 4.07, 0.]])
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    k = np.array([[width, 0., width/2], [0., height, height/2], [0., 0., 1.]])
    g = op.FixedObjectGeometry(v, f, np.tile(v, (2048, 1)), np.zeros(3), np.eye(3),
        np.array([0., 0., 2.]), k, (height, width), 'native fixed reconstructed metre gauge; unverified')
    y, x = np.mgrid[:height, :width]
    points = np.stack(((x+.5-width/2)/width*2, (y+.5-height/2)/height*2, np.full((height, width), 2.)), -1)
    ids = np.arange(41, 41+frames, dtype=np.int64)
    rows = [op.FrameObservation(i, int(ids[i]), np.full((height, width, 3), i, np.uint8),
        np.ones((height, width), bool), points, np.ones((height, width), bool)) for i in range(frames)]
    return g, np.arange(frames, dtype=np.int64), ids, rows


class Callbacks:
    def __init__(self, geometry, rows, *, invisible=False):
        self.geometry, self.rows, self.invisible = geometry, rows, invisible
        self.events = []; self.last_queries = None

    def raster(self, vertices, faces, K, width, height):
        self.events.append('raster')
        assert (height, width) == self.geometry.image_size
        np.testing.assert_array_equal(faces, self.geometry.faces)
        np.testing.assert_array_equal(K, self.geometry.K)
        return np.ones((height, width), bool)

    def align(self, sampled, observed, R, t):
        self.events.append('align'); np.testing.assert_array_equal(sampled, self.geometry.surface_points)
        return RigidAlignment(R.copy(), t.copy(), 'unchanged', .1, .1, 32, 0)

    def track(self, video, queries, **kwargs):
        self.events.append('track'); self.last_queries = queries
        assert isinstance(video, tuple) and len(video) == len(self.rows)
        for rgb, row in zip(video, self.rows):
            np.testing.assert_array_equal(rgb, row.rgb); assert not rgb.flags.writeable
        assert not queries.flags.writeable and set(kwargs) == {'frame_index', 'source_frame_ids', 'query_ids'}
        xy = queries[:, [2, 1]] * [256/self.geometry.image_size[1], 256/self.geometry.image_size[0]]
        return op.PointTrackEvidence(kwargs['frame_index'], kwargs['source_frame_ids'], kwargs['query_ids'], queries,
            np.broadcast_to(xy, (len(video), len(xy), 2)), np.full((len(video), len(xy)), not self.invisible, bool))


def run(data, backend):
    return op.compare_fixed_shape_sequence(*data, raster=backend.raster, align=backend.align, track=backend.track)


@pytest.mark.parametrize('grid,frames', [((24, 32), 3), ((30, 48), 4)])
def test_generic_composition_exactly_reuses_three_existing_operators(grid, frames):
    data = fixture(frames, *grid); g, index, source, rows = data; backend = Callbacks(g, rows)
    result = run(data, backend)
    assert backend.events.count('track') == 1 and backend.events[-1] == 'track'
    assert backend.events.count('align') == 25*frames and backend.events.count('raster') == 50*frames
    assert result.tracks.source_frame_ids.tolist() == source.tolist()
    assert len(result.observation_sha256) == frames and result.comparison.pool.rotations.shape == (frames, 25, 3, 3)
    queries = canonical_surface_queries(g.vertices, g.faces, g.R0, g.t0, g.K, rows[0].automatic_mask,
        rows[0].inferred_depth_valid, image_width=grid[1], image_height=grid[0])
    other = Callbacks(g, rows)
    builder = NativeCandidatePool(index, g.vertices, g.faces, g.surface_points, g.mesh_centroid, g.R0, g.t0, g.K, grid[1], grid[0])
    for row in rows: builder.add_frame(row.frame_index, row.inferred_pointmap, row.automatic_mask, other.raster, align=other.align)
    pool = builder.finalize()
    manual = compare(pool, queries.canonical_points, result.tracks.tracks_256, result.tracks.native_visible, g.K, grid[1], grid[0])
    for key in vars(pool): np.testing.assert_array_equal(getattr(result.comparison.pool, key), getattr(pool, key))
    for name in ('baseline', 'candidate'):
        actual, expected = getattr(result.comparison, name), getattr(manual, name)
        for key in vars(actual): np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    for key in vars(queries): np.testing.assert_array_equal(getattr(result.queries, key), getattr(queries, key))
    for record in (result.geometry, result.queries, result.tracks, result.comparison.pool):
        assert all(not a.flags.writeable for a in vars(record).values() if isinstance(a, np.ndarray))
    assert not np.shares_memory(result.geometry.vertices, g.vertices)
    assert not np.shares_memory(result.queries.query_points, backend.last_queries)


def test_no_visible_track_evidence_keeps_native_dynamic_baseline_not_static_rescue():
    data = fixture(); backend = Callbacks(data[0], data[3], invisible=True); result = run(data, backend)
    assert result.comparison.no_visible_evidence.all() and not result.comparison.point_costs.any()
    np.testing.assert_array_equal(result.comparison.baseline.translations, result.comparison.candidate.translations)
    assert result.tracks.native_visible.shape == (3, 32)


@pytest.mark.parametrize('fault', ['short', 'extra', 'order', 'source', 'grid', 'mask_empty', 'under40', 'query_empty'])
def test_missing_or_invalid_original_observations_fail_before_track_without_refill(fault):
    data = list(fixture()); rows = data[3]; backend = Callbacks(data[0], rows)
    if fault == 'short': data[3] = rows[:2]
    elif fault == 'extra': data[3] = rows+[rows[-1]]
    elif fault == 'order': data[3] = [rows[1], rows[0], rows[2]]
    elif fault == 'source': rows[1] = replace(rows[1], source_frame_id=999)
    elif fault == 'grid': rows[1] = replace(rows[1], rgb=np.zeros((30, 32, 3), np.uint8),
        automatic_mask=np.ones((30, 32), bool), inferred_pointmap=np.ones((30, 32, 3)), inferred_depth_valid=np.ones((30, 32), bool))
    elif fault in ('mask_empty', 'under40'):
        mask = np.zeros_like(rows[1].automatic_mask)
        if fault == 'under40': mask.flat[:39] = True
        rows[1] = replace(rows[1], automatic_mask=mask)
    else: rows[0] = replace(rows[0], inferred_depth_valid=np.zeros_like(rows[0].inferred_depth_valid))
    with pytest.raises(ValueError): run(data, backend)
    assert 'track' not in backend.events


@pytest.mark.parametrize('fault', ['index', 'source', 'query_id', 'queries', 'visibility', 'nonfinite', 'runtime', 'mutation'])
def test_native_output_query_binding_is_exact_without_matching_or_silent_reorder(fault):
    data = fixture(); backend = Callbacks(data[0], data[3]); original = backend.track
    def track(*args, **kwargs):
        if fault == 'runtime': raise RuntimeError('native model failure')
        evidence = original(*args, **kwargs)
        if fault == 'index': return replace(evidence, frame_index=np.array([0, 2, 3], np.int64))
        if fault == 'source': return replace(evidence, source_frame_ids=evidence.source_frame_ids+1)
        if fault == 'query_id': return replace(evidence, query_ids=evidence.query_ids[::-1])
        if fault == 'queries': return replace(evidence, query_points=evidence.query_points[::-1])
        if fault == 'visibility': return replace(evidence, native_visible=evidence.native_visible.astype(np.uint8))
        if fault == 'nonfinite': return replace(evidence, tracks_256=np.full_like(evidence.tracks_256, np.nan))
        args[1].flags.writeable = True; args[1][0, 1] += .1; args[1].flags.writeable = False
        return evidence
    backend.track = track
    with pytest.raises((ValueError, RuntimeError)): run(data, backend)
    assert backend.events.count('track') <= 1


@pytest.mark.parametrize('fault', ['frames', 'source_duplicates', 'source_order', 'dtype', 'reflection', 'sample_size', 'camera', 'gauge'])
def test_full_typed_geometry_timeline_contract(fault):
    data = list(fixture()); g = data[0]; backend = Callbacks(g, data[3])
    with pytest.raises(ValueError):
        if fault == 'frames': data[1] = np.array([0, 2, 3], np.int64)
        elif fault == 'source_duplicates': data[2] = np.array([41, 41, 43], np.int64)
        elif fault == 'source_order': data[2] = data[2][::-1]
        elif fault == 'dtype': data[0] = replace(g, faces=g.faces.astype(np.int32))
        elif fault == 'reflection': data[0] = replace(g, R0=np.diag([1., 1., -1.]))
        elif fault == 'sample_size': data[0] = replace(g, surface_points=g.surface_points[:8])
        elif fault == 'camera': data[0] = replace(g, K=np.array([[32., .2, 16.], [0., 24., 12.], [0., 0., 1.]]))
        else: data[0] = replace(g, gauge_convention='')
        run(data, backend)
    assert 'track' not in backend.events


def test_observation_support_keeps_missing_missing_and_inputs_are_copied_readonly():
    rgb = np.zeros((24, 32, 3), np.uint8); depth = np.zeros((24, 32), bool); points = np.full((24, 32, 3), np.nan)
    row = op.FrameObservation(0, 1, rgb, depth, points, depth)
    rgb[:] = 255
    assert not row.rgb.any() and np.isnan(row.inferred_pointmap).all() and not row.inferred_depth_valid.any()
    points[0, 0] = [1., 1., -1.]; depth[0, 0] = True
    with pytest.raises(ValueError): op.FrameObservation(0, 1, rgb, depth, points, depth)


@pytest.mark.parametrize('target', ['sample', 'faces', 'K'])
def test_callback_cannot_mutate_shared_native_geometry_then_return_success(target):
    data = fixture(); backend = Callbacks(data[0], data[3]); raster, align = backend.raster, backend.align
    def corrupt(array):
        array.flags.writeable = True; array.flat[0] += 1; array.flags.writeable = False
    def corrupt_raster(v, f, K, w, h):
        result = raster(v, f, K, w, h)
        if target in ('faces', 'K'): corrupt(f if target == 'faces' else K)
        return result
    def corrupt_align(sample, observations, R, t):
        result = align(sample, observations, R, t)
        if target == 'sample': corrupt(sample)
        return result
    backend.raster, backend.align = corrupt_raster, corrupt_align
    with pytest.raises((ValueError, AssertionError)): run(data, backend)
    assert 'track' not in backend.events


def test_missing_support_is_not_used_to_restrict_native_point_pool_observations():
    data = list(fixture()); rows = data[3]
    rows[1] = replace(rows[1], inferred_depth_valid=np.zeros_like(rows[1].inferred_depth_valid))
    result = run(data, Callbacks(data[0], rows))
    assert result.comparison.pool.valid_candidates[1].all()  # Native mask/finite/Z rule is unchanged.


@pytest.mark.parametrize('field', ['automatic_mask', 'inferred_depth_valid', 'inferred_pointmap'])
@pytest.mark.parametrize('stage', ['track', 'compare'])
def test_earlier_observation_arrays_cannot_mutate_through_callback_closure(monkeypatch, field, stage):
    data = fixture(); backend = Callbacks(data[0], data[3]); old_track, old_compare = backend.track, op.compare
    def mutate():
        array = getattr(data[3][0], field)
        array.flags.writeable = True
        array.flat[0] = not array.flat[0] if array.dtype == np.bool_ else array.flat[0]+.1
        array.flags.writeable = False
    def track(*args, **kwargs):
        result = old_track(*args, **kwargs); mutate(); return result
    def compare(*args, **kwargs):
        result = old_compare(*args, **kwargs); mutate(); return result
    if stage == 'track': backend.track = track
    else: monkeypatch.setattr(op, 'compare', compare)
    with pytest.raises(ValueError, match='observations'): run(data, backend)
    assert backend.events.count('track') == 1
    assert not getattr(data[3][0], field).flags.writeable


def test_generator_releases_pointmaps_and_masks_but_keeps_original_rgb_for_native_tracks():
    geometry, index, source_ids, rows = fixture()
    template = rows[0].inferred_pointmap.copy(); mask = rows[0].automatic_mask.copy(); del rows
    released, rgb_refs = [], []; backend = Callbacks(geometry, [])
    def observations():
        for position in index:
            row = op.FrameObservation(int(position), int(source_ids[position]),
                np.full((*geometry.image_size, 3), position, np.uint8), mask, template, mask)
            released.extend(weakref.ref(a) for a in (row.inferred_pointmap, row.automatic_mask, row.inferred_depth_valid))
            rgb_refs.append(weakref.ref(row.rgb))
            yield row
            del row
    def track(video, queries, **kwargs):
        gc.collect()
        assert len(released) == 3*len(index) and all(ref() is None for ref in released)
        assert isinstance(video, tuple) and len(video) == len(index)
        for frame, ref, rgb in zip(index, rgb_refs, video):
            assert ref() is rgb and not rgb.flags.writeable and np.all(rgb == frame)
        xy = queries[:, [2, 1]] * [256/geometry.image_size[1], 256/geometry.image_size[0]]
        return op.PointTrackEvidence(kwargs['frame_index'], kwargs['source_frame_ids'], kwargs['query_ids'], queries,
            np.broadcast_to(xy, (len(video), len(xy), 2)), np.ones((len(video), len(xy)), bool))
    result = op.compare_fixed_shape_sequence(geometry, index, source_ids, observations(),
        raster=backend.raster, align=backend.align, track=track)
    assert len(result.observation_sha256) == len(index) and not result.comparison.no_visible_evidence.any()
    gc.collect(); assert all(ref() is None for ref in released+rgb_refs)


def test_source_has_no_ycb_grid_or_scorer_solver_backend_override():
    source = inspect.getsource(op.compare_fixed_shape_sequence)
    assert 'canonical_surface_queries(' in source and 'NativeCandidatePool(' in source and 'compare(' in source
    assert '640' not in source and '480' not in source and '96' not in source
    assert source.index('builder.finalize()') < source.index('evidence = track(')
    assert 'select_pose_path' not in source and 'cv2' not in source and 'ground_truth' not in source
