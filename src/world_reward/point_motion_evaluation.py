"""Private CPU-only relative camera-frame material motion metric.

No I/O, model, mesh, fitting, alignment, scale correction or frame filtering.
Sensor depth MUST already be camera Z in metres. All initial visible truth
pixels contribute, in bounded chunks; truth points/poses never leave this API.
BOP integer pixels: misc.Precomputer.precompute_lazy at af97c1938083dfd512eb6f32a85921ea38198ee4.
Reacquisition is not defined in the preregistration and is explicitly unsupported.
"""
from __future__ import annotations
import numpy as np

MIN_POINTS = 8
PURITY = .95
CHUNK_POINTS = 4096
COHORT_SIZE = 3
MIN_MEDIAN_GAIN = .10
MAX_REGRESSION = .05


def _require(value, message):
    if not value:
        raise ValueError(message)


def _real(value, name):
    _require(not np.ma.isMaskedArray(value), f'{name}: masked arrays forbidden')
    array = np.asarray(value)
    _require(np.issubdtype(array.dtype, np.floating) and np.isfinite(array).all(), f'{name}: finite floating array required')
    return array.astype(np.float64, copy=False)


def _poses(value, frames, name, instances=None):
    poses = _real(value, name)
    shape = (frames, 4, 4) if instances is None else (instances, frames, 4, 4)
    _require(poses.shape == shape, f'{name}: all instances/full original homogeneous trajectory required')
    _require(np.all(poses[..., 3, :] == [0., 0., 0., 1.]), f'{name}: exact homogeneous SE3 row required')
    rotations = poses[..., :3, :3]
    _require(np.allclose(rotations.swapaxes(-2, -1) @ rotations, np.eye(3), atol=1e-6, rtol=0) and
        np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0), f'{name}: proper SO3 only, no scale/reflection')
    return poses


def _mask(value, shape, name):
    array = np.asarray(value)
    _require(not np.ma.isMaskedArray(value) and array.dtype == np.bool_ and array.shape == shape, f'{name}: strict original-grid boolean mask required')
    return array


def _relative(poses):
    rotation = poses[:, :3, :3] @ poses[0, :3, :3].T
    translation = poses[:, :3, 3] - np.einsum('tij,j->ti', rotation, poses[0, :3, 3])
    # Identity at frame0 is an algebraic constraint, not an error alignment fit.
    rotation[0] = np.eye(3); translation[0] = 0.
    _require(np.isfinite(rotation).all() and np.isfinite(translation).all(), 'Relative SE3 overflow')
    return rotation, translation


def evaluate_sequence(frame_indices, baseline_poses, candidate_poses, gt_poses,
        automatic_initial_mask, gt_initial_masks, depth_z_metres, private_K, *, sequence_id):
    """Evaluate frozen branches against the unique initial mask-associated object.

    gt_initial_masks: boolean [instances,H,W], mask_visib at original frame0.
    gt_poses: homogeneous [instances,T,4,4], same initial mask instance ordering
    across frames authenticated by the private caller. No GT object is chosen
    using pose error. All initial truth-mask pixels with finite positive
    sensorZ contribute equally; no random/sparse subset. Return errors only.
    """
    _require(type(sequence_id) is str and 0 < len(sequence_id) <= 80, 'Distinct structural sequence ID required')
    indices = np.asarray(frame_indices)
    _require(not np.ma.isMaskedArray(frame_indices) and indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer) and
        len(indices) >= 3 and np.array_equal(indices, np.arange(len(indices))), 'Every original arange(T), T>=3, required')
    depth = np.asarray(depth_z_metres)
    _require(not np.ma.isMaskedArray(depth_z_metres) and depth.ndim == 2 and min(depth.shape) > 0 and
        np.issubdtype(depth.dtype, np.floating), 'Explicit floating sensor cameraZ metres [H,W] required')
    auto = _mask(automatic_initial_mask, depth.shape, 'Automatic initial mask')
    masks = np.asarray(gt_initial_masks)
    _require(not np.ma.isMaskedArray(gt_initial_masks) and masks.dtype == np.bool_ and masks.ndim == 3 and
        masks.shape[1:] == depth.shape and len(masks) > 0, 'All initial GT visible instance masks required')
    denominator = int(np.count_nonzero(auto)); _require(denominator > 0, 'Empty automatic initial mask: no purity denominator')
    overlaps = np.count_nonzero(masks & auto[None], axis=(1, 2))
    matched = np.flatnonzero(20 * overlaps >= 19 * denominator)
    _require(len(matched) == 1, 'Exactly one GT mask must reach fixed95percent initial automatic-mask purity')
    match = int(matched[0]); support = masks[match] & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(support); point_count = len(xx)
    _require(point_count >= MIN_POINTS, 'At least8 initial visible positive sensorZ material pixels required')
    K = _real(private_K, 'Fixed private sensor K')
    _require(K.shape == (3, 3) and np.array_equal(K[2], [0., 0., 1.]) and K[0, 1] == 0. and K[1, 0] == 0. and
        K[0, 0] > 0. and K[1, 1] > 0., 'One standard fixed BOP pinhole K required')
    gt = _relative(_poses(gt_poses, len(indices), 'GT', len(masks))[match])
    branches = [('baseline', _relative(_poses(baseline_poses, len(indices), 'Baseline'))),
        ('candidate', _relative(_poses(candidate_poses, len(indices), 'Candidate')))]
    result = dict(schema='world_reward.point_motion_evaluation.v1', sequence_id=sequence_id, status='pass',
        frame_indices=indices.tolist(), coverage=1., matched_gt_mask_index=match,
        initial_mask_purity=float(overlaps[match] / denominator), initial_material_points=point_count,
        sensor_depth_units='camera_Z_metres_explicit', truth_pixel_convention='integer_BOP',
        alignment_performed=False, scale_fitted=False, symmetry_minimized=False, reacquisition_supported=False,
        reacquisition_error=None, absolute_shape_or_pose_accuracy_verified=False)
    for name, (rotation, translation) in branches:
        displacement = np.empty(len(indices)); delta = rotation @ gt[0].transpose(0, 2, 1)
        cosine = np.clip((np.trace(delta, axis1=1, axis2=2) - 1.) / 2., -1., 1.)
        angle = np.degrees(np.arccos(cosine)); angle[0] = 0.
        for frame in range(len(indices)):
            total = 0.
            for first in range(0, point_count, CHUNK_POINTS):
                x, y = xx[first:first+CHUNK_POINTS], yy[first:first+CHUNK_POINTS]
                z = depth[y, x].astype(np.float64)
                points = np.column_stack(((x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z))
                residual = points @ (rotation[frame]-gt[0][frame]).T + translation[frame]-gt[1][frame]
                distance = np.hypot(np.hypot(residual[:, 0], residual[:, 1]), residual[:, 2])
                _require(np.isfinite(distance).all(), 'Material displacement overflow: do not delete failing pixels/frames')
                total += float(distance.sum())
            displacement[frame] = total / point_count
        _require(np.isfinite(displacement).all() and np.isfinite(angle).all(), 'All original per-frame errors must remain finite')
        result[name] = dict(displacement_m=displacement.tolist(), rotation_degrees=angle.tolist(),
            mean_displacement_m=float(displacement.mean()), mean_rotation_degrees=float(angle.mean()))
    return result


def cohort_gate(records):
    """Fixed exactly3-scene gate; zero baseline gives unidentified gain FAIL."""
    _require(type(records) in (tuple, list) and len(records) == COHORT_SIZE, 'Exactly3 complete sequence evaluations required')
    _require(all(type(r) is dict and type(r.get('sequence_id')) is str and r['sequence_id'] for r in records) and
        len({r['sequence_id'] for r in records}) == COHORT_SIZE, 'Three distinct sequence IDs required')
    gains, ratios, reasons = [], [], []
    for row in records:
        indices = row.get('frame_indices'); _require(type(indices) is list and len(indices) >= 3 and all(type(i) is int for i in indices) and indices == list(range(len(indices))), 'Complete original cohort timeline required')
        _require(row.get('schema') == 'world_reward.point_motion_evaluation.v1' and row.get('status') == 'pass' and row.get('coverage') == 1., 'All sequence evaluations and100percent coverage required')
        means = []
        for name in ('baseline', 'candidate'):
            branch = row[name]
            for key in ('displacement_m', 'rotation_degrees'):
                values = branch[key]; _require(type(values) is list and len(values) == len(indices) and
                    all(type(v) in (int, float) and np.isfinite(v) and v >= 0. for v in values), 'All original finite nonnegative errors required')
            mean = float(np.mean(branch['displacement_m']))
            _require(type(branch['mean_displacement_m']) in (int, float) and branch['mean_displacement_m'] == mean, 'Recorded mean must match all original errors')
            _require(branch['mean_rotation_degrees'] == float(np.mean(branch['rotation_degrees'])) and
                all(v <= 180. for v in branch['rotation_degrees']), 'Recorded rotation mean and SO3 geodesic range required')
            means.append(mean)
        if means[0] == 0.:
            gains.append(None); ratios.append(None); reasons.append('zero_baseline_gain_unidentified')
        else:
            ratio = means[1] / means[0]; _require(np.isfinite(ratio), 'Finite identified relative gain required')
            ratios.append(ratio); gains.append(1.-ratio)
            # Mathematically identical thresholds without cancellation at .10.
            if ratio > np.nextafter(1.+MAX_REGRESSION, np.inf): reasons.append('scene_regression_exceeds5percent')
    median = None if any(g is None for g in gains) else float(np.median(gains))
    if median is not None and float(np.median(ratios)) > np.nextafter(1.-MIN_MEDIAN_GAIN, np.inf): reasons.append('median_gain_below10percent')
    return dict(schema='world_reward.point_motion_cohort_gate.v1', status='pass' if not reasons else 'fail',
        sequence_ids=[r['sequence_id'] for r in records],
        relative_mean_error_gains=gains, median_relative_mean_error_gain=median, coverage=1.,
        minimum_median_gain=MIN_MEDIAN_GAIN, maximum_scene_regression=MAX_REGRESSION,
        arithmetic_guard='one_float64_ULP_on_error_ratio_only',
        reasons=sorted(set(reasons)), reacquisition_supported=False, reacquisition_error=None,
        challenge_accuracy_verified=False, articulated_HOI_accuracy_verified=False)
