"""Own arrays: independent unary/reference Viterbi, not external quality tests."""
from dataclasses import replace

import numpy as np
import pytest

from world_reward.point_candidate_pool import CandidatePool
from world_reward import point_pose_comparison as operator
from world_reward.pose_selection import select_pose_path


def frozen(array):
    result = np.array(array, copy=True); result.flags.writeable = False; return result


def fixture(frames=3):
    r = np.broadcast_to(np.eye(3), (frames, 25, 3, 3)).copy()
    t = np.zeros((frames, 25, 3)); t[..., 2] = 3.; t[:, 1, 0] = .3
    costs = np.ones((frames, 25)); costs[:, 0] = .02; costs[:, 1] = 0.
    valid = np.zeros((frames, 25), bool); valid[:, :2] = True
    pool = CandidatePool(*map(frozen, (np.arange(frames, dtype=np.int64), r, t, costs, valid, np.ones(frames, dtype=np.int64))))
    points = np.column_stack((np.linspace(-.2, .2, 8), np.linspace(-.1, .1, 8), np.zeros(8)))
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]])
    camera = points + t[0, 0]
    tracks = (camera[:, :2] / camera[:, 2, None] * [800, 800] + [320, 240]) * [256 / 640, 256 / 480]
    return pool, frozen(points), frozen(np.broadcast_to(tracks, (frames, 8, 2))), frozen(np.ones((frames, 8), bool)), frozen(K)


def call(data):
    return operator.compare(*data, 640, 480)


def manual_unary(data):
    pool, points, tracks, visible, K = data
    result = np.zeros_like(pool.image_costs)
    for frame in range(len(pool.frame_index)):
        for candidate in np.flatnonzero(pool.valid_candidates[frame]):
            if not visible[frame].any(): continue
            xyz = points @ pool.rotations[frame, candidate].T + pool.translations[frame, candidate]
            xy = (xyz[:, :2] / xyz[:, 2, None] * [K[0, 0], K[1, 1]] + K[:2, 2]) * [256 / 640, 256 / 480]
            errors = np.linalg.norm(xy[visible[frame]] - tracks[frame, visible[frame]], axis=1)
            result[frame, candidate] = .1 * np.minimum(1., errors / 8.).mean()
    return result


def reference(pool, costs):
    return select_pose_path(pool.rotations, pool.translations, costs, valid_candidates=pool.valid_candidates,
        frame_times=pool.frame_index, translation_weight=1., rotation_weight=.1)


def assert_path(actual, expected):
    for name in ('candidate_indices', 'symmetry_indices', 'rotations', 'translations'):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        assert not getattr(actual, name).flags.writeable
    for name in ('unary_cost', 'transition_cost', 'total_cost'):
        assert getattr(actual, name) == pytest.approx(getattr(expected, name), abs=1e-14)


def test_only_added_unary_changes_selected_original_candidates():
    data = fixture(); pool = data[0]; snapshots = {k: v.copy() for k, v in vars(pool).items()}
    result = call(data); unary = manual_unary(data)
    np.testing.assert_allclose(result.point_costs, unary, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(result.candidate_costs, pool.image_costs + result.point_costs)
    assert_path(result.baseline, reference(pool, pool.image_costs))
    assert_path(result.candidate, reference(pool, pool.image_costs + unary))
    assert result.baseline.candidate_indices.tolist() == [1, 1, 1]
    assert result.candidate.candidate_indices.tolist() == [0, 0, 0]
    assert result.visible_count.tolist() == [8, 8, 8]
    for key, value in vars(result.pool).items():
        np.testing.assert_array_equal(value, snapshots[key]); assert not value.flags.writeable
        assert not np.shares_memory(value, getattr(pool, key))
    for array in (result.point_costs, result.candidate_costs, result.visible_count, result.no_visible_evidence, result.nonfront_witness_count):
        assert not array.flags.writeable


def test_no_visible_tracks_do_not_fabricate_motion_or_change_baseline():
    data = list(fixture()); data[3] = frozen(np.zeros_like(data[3]))
    result = call(data)
    assert np.count_nonzero(result.point_costs) == 0 and result.no_visible_evidence.all()
    assert_path(result.candidate, result.baseline)


def test_distinguishable_180_flip_and_original_query_order_no_symmetry():
    data = list(fixture()); r = data[0].rotations.copy(); r[:, 1] = np.diag([-1., -1., 1.])
    data[0] = replace(data[0], rotations=frozen(r)); result = call(data)
    assert result.candidate.candidate_indices.tolist() == [0, 0, 0]
    np.testing.assert_allclose(result.point_costs, manual_unary(data), atol=1e-14, rtol=0)
    assert not result.candidate.symmetry_indices.any()
    permutation = np.array([7, 2, 6, 0, 1, 5, 3, 4])
    reordered = [data[0], frozen(data[1][permutation]), frozen(data[2][:, permutation]), frozen(data[3][:, permutation]), data[4]]
    consistent = call(reordered)
    np.testing.assert_allclose(consistent.point_costs, result.point_costs, atol=1e-14, rtol=0)
    # A track-only permutation is not secretly realigned/refitted to query order.
    shuffled_tracks = list(data); shuffled_tracks[2] = frozen(data[2][:, permutation])
    assert not np.allclose(call(shuffled_tracks).point_costs, result.point_costs, atol=1e-8)


@pytest.mark.parametrize('occluded', [False, True])
def test_any_original_valid_nonfront_witness_fails_both_branches_before_selection(monkeypatch, occluded):
    data = list(fixture()); t = data[0].translations.copy(); t[1, 1, 2] = -1
    data[0] = replace(data[0], translations=frozen(t))
    if occluded: data[3] = frozen(np.zeros_like(data[3]))
    def forbidden(*a, **k): pytest.fail('A branch ran after native candidate pruning')
    monkeypatch.setattr(operator, 'select_pose_path', forbidden)
    with pytest.raises(ValueError, match='without pruning'): call(data)


def test_original_invalid_slots_stay_original_invalid_not_new_common_mask():
    data = list(fixture()); t = data[0].translations.copy(); t[:, 2, 2] = -10
    data[0] = replace(data[0], translations=frozen(t)); result = call(data)
    np.testing.assert_array_equal(result.pool.valid_candidates, data[0].valid_candidates)
    assert result.nonfront_witness_count[:, 2].tolist() == [8, 8, 8]
    assert not result.pool.valid_candidates[:, 2].any()


@pytest.mark.parametrize('fault', ['type', 'mutable', 'frames', 'indices', 'candidate_count', 'image_range',
    'image_nan', 'valid_dtype', 'valid_empty', 'greedy_invalid', 'greedy_shape', 'reflection', 'rotation_nan', 'translation_nan'])
def test_exact_original_candidate_pool_contract(fault):
    data = list(fixture()); pool = data[0]
    if fault == 'type': data[0] = dict(vars(pool))
    elif fault == 'mutable': data[0] = replace(pool, image_costs=pool.image_costs.copy())
    elif fault == 'frames': data[0] = replace(pool, frame_index=frozen(np.arange(2, dtype=np.int64)))
    elif fault == 'indices': data[0] = replace(pool, frame_index=frozen(np.array([0, 2, 3], dtype=np.int64)))
    elif fault == 'candidate_count': data[0] = replace(pool, rotations=frozen(pool.rotations[:, :24]))
    elif fault == 'image_range': data[0] = replace(pool, image_costs=frozen(pool.image_costs - 2))
    elif fault == 'image_nan': data[0] = replace(pool, image_costs=frozen(np.full_like(pool.image_costs, np.nan)))
    elif fault == 'valid_dtype': data[0] = replace(pool, valid_candidates=frozen(pool.valid_candidates.astype(np.int64)))
    elif fault == 'valid_empty': data[0] = replace(pool, valid_candidates=frozen(np.zeros_like(pool.valid_candidates)))
    elif fault == 'greedy_invalid': data[0] = replace(pool, greedy_indices=frozen(np.array([2, 1, 1], dtype=np.int64)))
    elif fault == 'greedy_shape': data[0] = replace(pool, greedy_indices=frozen(np.array([[1, 1, 1]], dtype=np.int64)))
    elif fault == 'reflection': data[0] = replace(pool, rotations=frozen(np.broadcast_to(np.diag([-1., 1., 1.]), pool.rotations.shape)))
    elif fault == 'rotation_nan': data[0] = replace(pool, rotations=frozen(np.full_like(pool.rotations, np.nan)))
    else: data[0] = replace(pool, translations=frozen(np.full_like(pool.translations, np.inf)))
    with pytest.raises(ValueError): call(data)


@pytest.mark.parametrize('fault', ['mutable_points', 'mutable_tracks', 'mutable_visible', 'mutable_K',
    'duplicate_points', 'query_shape', 'tracks_frames', 'visibility_dtype', 'K_skew', 'K_nonfinite', 'size_bool'])
def test_frozen_query_evidence_and_rgb_camera_contract(fault):
    data = list(fixture())
    if fault.startswith('mutable_'):
        index = {'mutable_points': 1, 'mutable_tracks': 2, 'mutable_visible': 3, 'mutable_K': 4}[fault]; data[index] = data[index].copy()
    elif fault == 'duplicate_points': p = data[1].copy(); p[1] = p[0]; data[1] = frozen(p)
    elif fault == 'query_shape': data[1] = frozen(data[1][:7])
    elif fault == 'tracks_frames': data[2] = frozen(data[2][:2])
    elif fault == 'visibility_dtype': data[3] = frozen(data[3].astype(np.float32))
    elif fault == 'K_skew': K = data[4].copy(); K[0, 1] = 1.; data[4] = frozen(K)
    elif fault == 'K_nonfinite': data[4] = frozen(np.full_like(data[4], np.inf))
    else:
        with pytest.raises(ValueError): operator.compare(*data, True, 480)
        return
    with pytest.raises(ValueError): call(data)


def test_postcheck_detects_mutation_by_called_operator(monkeypatch):
    data = fixture(); original = operator.select_pose_path; count = 0
    def corrupt(*args, **kwargs):
        nonlocal count
        path = original(*args, **kwargs); count += 1
        if count == 1:
            data[2].flags.writeable = True; data[2][0, 0, 0] += 1.; data[2].flags.writeable = False
        return path
    monkeypatch.setattr(operator, 'select_pose_path', corrupt)
    with pytest.raises(ValueError, match='changed during'): call(data)


def test_pool_generation_never_receives_observed_tracks_or_selected_paths(monkeypatch):
    data = fixture(); original = operator.select_pose_path; calls = []
    def record(rotations, translations, costs, **kwargs):
        calls.append((rotations, translations, costs, kwargs)); return original(rotations, translations, costs, **kwargs)
    monkeypatch.setattr(operator, 'select_pose_path', record); result = call(data)
    assert len(calls) == 2
    for rotations, translations, _, kwargs in calls:
        assert rotations is data[0].rotations and translations is data[0].translations
        assert kwargs['valid_candidates'] is data[0].valid_candidates and kwargs['frame_times'] is data[0].frame_index
        assert kwargs['translation_weight'] == 1. and kwargs['rotation_weight'] == .1
        assert set(kwargs) == {'valid_candidates', 'frame_times', 'translation_weight', 'rotation_weight'}
    assert calls[0][2] is data[0].image_costs
    np.testing.assert_array_equal(result.pool.greedy_indices, data[0].greedy_indices)
