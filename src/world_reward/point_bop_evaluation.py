"""Private BOP/YCBV adapter for the frozen contiguous relative-motion metric.

No I/O/model/mesh/alignment. Instance identity is the unique original obj_id,
never pose-error matching or whichever ordering happens to be present later.
Integer uint16 depth uses the explicitly supplied BOP millimetre depth_scale.
Only the metric's scalar/errors report returns; private arrays stay internal.
"""
from __future__ import annotations
import numpy as np
from world_reward.point_motion_evaluation import evaluate_sequence

FRAMES = 96
HEIGHT, WIDTH = 480, 640


def require(value, message):
    if not value: raise ValueError(message)


def _number(value, name):
    require(type(value) in (int, float) and np.isfinite(value), name + ': finite JSON number required')
    return float(value)


def _vector(value, count, name):
    require(type(value) is list and len(value) == count, name + ': exact original JSON vector required')
    return np.array([_number(x, name) for x in value], dtype=np.float64)


def _frames(value, name):
    require(type(value) is dict and set(value) == {str(i) for i in range(FRAMES)},
            name + ': every original frame0..95 required, no sparse selection')


def _instances(rows):
    require(type(rows) is list and rows, 'All native object instances required')
    ids = []
    for row in rows:
        require(type(row) is dict and type(row.get('obj_id')) is int and row['obj_id'] > 0,
                'Original positive object ID required')
        ids.append(row['obj_id'])
    require(len(ids) == len(set(ids)), 'Duplicate object IDs have no frozen instance identity; no pose matching')
    return ids


def evaluate_bop_scene(baseline_poses, candidate_poses, automatic_initial_mask,
        scene_camera, scene_gt, scene_gt_info, initial_depth_png, initial_visible_masks,
        *, sequence_id):
    """Private evaluator only: exact full96 annotations and initial original PNGs.

    ``initial_visible_masks`` is in original frame-zero scene_gt list order.
    Every later frame is indexed by its unique native obj_id, not list position.
    Camera K/depth_scale must be bit-identical across all96 original records.
    Missing/repeated/replaced IDs or nonconstant calibration stop before errors.
    No GT points, calibration, poses or masks are returned or serialized here.
    """
    for name, value in (('camera', scene_camera), ('ground truth', scene_gt), ('GT info', scene_gt_info)):
        _frames(value, name)
    first_ids = _instances(scene_gt['0'])
    order = {identity: index for index, identity in enumerate(first_ids)}
    poses = np.empty((len(order), FRAMES, 4, 4), dtype=np.float64)
    camera_K = None; depth_scale = None
    for frame in range(FRAMES):
        camera = scene_camera[str(frame)]
        require(type(camera) is dict and {'cam_K', 'depth_scale'} <= set(camera), 'Explicit original BOP camera fields required')
        K = _vector(camera['cam_K'], 9, 'K').reshape(3, 3)
        scale = _number(camera['depth_scale'], 'BOP depth_scale')
        require(scale > 0 and K[0, 0] > 0 and K[1, 1] > 0 and K[0, 1] == K[1, 0] == 0
                and np.array_equal(K[2], [0., 0., 1.]), 'Proper positive-focal fixed BOP camera required')
        if frame == 0: camera_K, depth_scale = K, scale
        require(np.array_equal(K, camera_K) and scale == depth_scale, 'Fixed full-clip sensor camera/units required')
        rows = scene_gt[str(frame)]; ids = _instances(rows)
        info = scene_gt_info[str(frame)]
        require(set(ids) == set(order) and type(info) is list and len(info) == len(rows)
                and all(type(x) is dict for x in info), 'Every original unique instance and GT-info record required')
        for row in rows:
            require({'cam_R_m2c', 'cam_t_m2c'} <= set(row), 'Explicit model-to-camera pose required')
            R = _vector(row['cam_R_m2c'], 9, 'GT rotation').reshape(3, 3)
            t = _vector(row['cam_t_m2c'], 3, 'GT translation') / 1000.
            require(np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0)
                    and np.isclose(np.linalg.det(R), 1., atol=1e-6, rtol=0), 'Proper GT SO3 required; no repair')
            transform = np.eye(4, dtype=np.float64); transform[:3, :3] = R; transform[:3, 3] = t
            poses[order[row['obj_id']], frame] = transform
    z = np.asarray(initial_depth_png); masks = np.asarray(initial_visible_masks)
    require(not np.ma.isMaskedArray(initial_depth_png) and z.dtype == np.uint16
            and z.shape == (HEIGHT, WIDTH), 'Untouched native uint16 sensor-Z PNG required')
    require(not np.ma.isMaskedArray(initial_visible_masks) and masks.dtype == np.uint8
            and masks.shape == (len(order), HEIGHT, WIDTH) and np.all((masks == 0) | (masks == 255)),
            'All original frame-zero binary visible-mask PNGs in native instance order required')
    # depth_scale is millimetres per original sensor value; never inferred/fitted.
    depth_metres = z.astype(np.float64) * depth_scale / 1000.
    require(np.isfinite(depth_metres).all(), 'Native sensor-unit conversion overflow')
    return evaluate_sequence(np.arange(FRAMES, dtype=np.int64), baseline_poses,
        candidate_poses, poses, automatic_initial_mask, masks == 255,
        depth_metres, camera_K, sequence_id=sequence_id)
