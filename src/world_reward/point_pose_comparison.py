"""Same-pool native Viterbi versus one frozen automatic point-association unary.

No meshes, GT, tracking inference or candidate generation enter this operator.
Nonfront canonical witnesses fail the comparison, rather than pruning either
branch. Objective improvement is not evidence of pose/reconstruction accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from world_reward.point_candidate_pool import CandidatePool, HYPOTHESES
from world_reward.point_pose_cost import point_pose_cost
from world_reward.pose_selection import PosePath, select_pose_path

TRANSLATION_WEIGHT, ROTATION_WEIGHT = 1.0, 0.1


def _readonly(value):
    array = np.array(value, copy=True)
    array.flags.writeable = False
    return array


def _array(value, name, *, dtype=None):
    if (type(value) is not np.ndarray or value.flags.writeable
            or (dtype is not None and value.dtype != dtype)
            or (dtype is None and value.dtype not in (np.dtype('float32'), np.dtype('float64')))
            or not np.isfinite(value).all()):
        raise ValueError(f'{name}: original finite readonly ndarray with exact dtype required')
    return value


def _validate_pool(pool):
    if type(pool) is not CandidatePool:
        raise ValueError('Exact original CandidatePool dataclass required')
    index = _array(pool.frame_index, 'frame_index', dtype=np.int64)
    if index.ndim != 1 or len(index) < 3 or not np.array_equal(index, np.arange(len(index), dtype=np.int64)):
        raise ValueError('Full original int64 arange(T), T>=3 required')
    frames = len(index)
    expected = {'rotations': (frames, HYPOTHESES, 3, 3), 'translations': (frames, HYPOTHESES, 3),
        'image_costs': (frames, HYPOTHESES), 'valid_candidates': (frames, HYPOTHESES), 'greedy_indices': (frames,)}
    for name, shape in expected.items():
        dtype = np.bool_ if name == 'valid_candidates' else np.int64 if name == 'greedy_indices' else None
        if _array(getattr(pool, name), name, dtype=dtype).shape != shape:
            raise ValueError('The unchanged full-T native25 candidate pool is required')
    if np.any((pool.image_costs < 0) | (pool.image_costs > 1)):
        raise ValueError('Original native image costs must remain in [0,1]')
    if (not pool.valid_candidates.any(axis=1).all() or np.any(pool.greedy_indices < 0)
            or np.any(pool.greedy_indices >= HYPOTHESES)
            or not pool.valid_candidates[index, pool.greedy_indices].all()):
        raise ValueError('Every original frame needs supported native candidates and greedy seed')
    with np.errstate(over='raise', invalid='raise'):
        try:
            r = pool.rotations.astype(np.float64, copy=False)
            proper = (np.allclose(r @ r.swapaxes(-1, -2), np.eye(3), atol=1e-6, rtol=0)
                and np.allclose(np.linalg.det(r), 1, atol=1e-6, rtol=0))
        except FloatingPointError as exc:
            raise ValueError('Original candidate rotation arithmetic must remain finite') from exc
    if not proper: raise ValueError('All original candidates must remain proper SO(3)')


def _freeze_path(path):
    return PosePath(*(_readonly(getattr(path, key)) for key in
        ('candidate_indices', 'symmetry_indices', 'rotations', 'translations')),
        path.unary_cost, path.transition_cost, path.total_cost)


@dataclass(frozen=True)
class PointPoseComparison:
    """Readonly original pool/selected paths and cost evidence, all full-T.

    Native25 hypotheses/greedy seeds are unchanged by tracking or either path.
    The two paths select supplied candidates only, without symmetry, fitting,
    pose averaging, dropped frames or branches with different validity masks.
    """
    pool: CandidatePool
    baseline: PosePath
    candidate: PosePath
    point_costs: np.ndarray
    candidate_costs: np.ndarray
    visible_count: np.ndarray
    no_visible_evidence: np.ndarray
    nonfront_witness_count: np.ndarray


def compare(pool, canonical_points, tracks_256, native_visible, K, image_width, image_height):
    """Run both native selections; candidate differs only by .1 point unary.

    Canonical points[Q,3] are fixed once, Q in [8,32]; tracks[T,Q,2] are xy
    on the native256 grid; visibility is the original native bool[T,Q]. All
    caller arrays must already be readonly. K and dimensions are RGB-derived
    clip constants supplied by the caller, not sensor/GT calibration. This
    numeric contract does not establish data/model provenance or metre gauge.
    """
    _validate_pool(pool)
    _array(canonical_points, 'canonical_points'); _array(tracks_256, 'tracks_256')
    _array(native_visible, 'native_visible', dtype=np.bool_); _array(K, 'K')
    originals = {name: value for name, value in vars(pool).items()}
    originals.update(canonical_points=canonical_points, tracks_256=tracks_256, native_visible=native_visible, K=K)
    snapshots = {name: _readonly(value) for name, value in originals.items()}
    point = point_pose_cost(canonical_points, tracks_256, native_visible, pool.rotations,
        pool.translations, K, pool.frame_index, image_width=image_width, image_height=image_height,
        valid_candidates=pool.valid_candidates)
    if not np.array_equal(point.valid_candidates, pool.valid_candidates):
        raise ValueError('Canonical witnesses invalidate an original native candidate; comparison stopped without pruning')
    candidate_costs = pool.image_costs + point.costs
    baseline = select_pose_path(pool.rotations, pool.translations, pool.image_costs,
        valid_candidates=pool.valid_candidates, frame_times=pool.frame_index,
        translation_weight=TRANSLATION_WEIGHT, rotation_weight=ROTATION_WEIGHT)
    candidate = select_pose_path(pool.rotations, pool.translations, candidate_costs,
        valid_candidates=pool.valid_candidates, frame_times=pool.frame_index,
        translation_weight=TRANSLATION_WEIGHT, rotation_weight=ROTATION_WEIGHT)
    for name, value in originals.items():
        snapshot = snapshots[name]
        if value.flags.writeable or value.dtype != snapshot.dtype or not np.array_equal(value, snapshot):
            raise ValueError('Original pool/queries/tracks/visibility/camera changed during comparison')
    frozen_pool = CandidatePool(*(snapshots[name] for name in vars(pool)))
    return PointPoseComparison(frozen_pool, _freeze_path(baseline), _freeze_path(candidate),
        _readonly(point.costs), _readonly(candidate_costs), _readonly(point.visible_count),
        _readonly(point.no_visible_evidence), _readonly(point.nonfront_witness_count))
