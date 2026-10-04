"""Predicted geometry versus an automatic person silhouette: a CPU-only proxy.

No fitting, geometry repair, native-rotation reinterpretation or reference data.
Person-mask support is NOT independent RGB hand/keypoint verification. A human
mesh silhouette includes self-visibility, not occlusion by a separate object.
The caller owns provenance, immutable inputs and any four-record aggregate gate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import types

import numpy as np

IOU_MIN = .7
MASK_RADIUS_PX = 5.
MIN_VECTOR_M, MIN_SIN_ANGLE, SO3_ATOL = 1e-8, 1e-3, 1e-10
BUDGET_SECONDS, MAX_CANDIDATES, CHUNK = 100, 20_000_000, 8192
RENDERER_BYTES = 18543
RENDERER_SHA256 = 'aaea4ce32b68ad6e2e5aa411278cf4dc90264a79ce68e839f349bd8a6fce7444'
SOURCE_FILES = ('infra/bridge_anchor_geometry_qa.py', 'infra/triangle_ray_gate.py')
LANDMARKS = ('wrist', 'index1', 'middle1')


def _load_renderer():
    """Authenticate only the standalone source; never call its fixture/main."""
    path = Path(__file__).with_name('triangle_ray_gate.py')
    data = path.read_bytes()
    if len(data) != RENDERER_BYTES or hashlib.sha256(data).hexdigest() != RENDERER_SHA256:
        raise ValueError('Standalone triangle source bytes/SHA differ from fixed QA pin')
    module = types.ModuleType('_bridge_qa_triangle_renderer')
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    if (module.BUDGET, module.MAX_CANDIDATES, module.CHUNK) != (BUDGET_SECONDS, MAX_CANDIDATES, CHUNK):
        raise ValueError('Standalone triangle work/deadline guards differ')
    return module


def _names(names):
    if (type(names) is not list or len(names) != 127 or
            any(type(n) is not str or not n for n in names) or len(set(names)) != 127):
        raise ValueError('Actual unique 127 joint names required; no ordinal guesses')
    return {name: index for index, name in enumerate(names)}


def anatomical_frame(joints_camera_m, joint_names, side):
    """Column axes: index-radial, middle-distal, their cross product, in camera.

    The origin is the named wrist. Neither normal is claimed to be palm-facing;
    left/right use the same anatomical construction, not a native MHR rotation.
    Degenerate vectors/near-collinearity remain invalid, never repaired.
    """
    if side not in ('l', 'r'):
        raise ValueError('Named anatomical side l or r required')
    names = [side+'_'+name for name in LANDMARKS]
    result = dict(valid=False, reasons=[], names=names, origin_camera_m=None,
                  R_frame_to_camera=None, normalized_cross_length=None)
    try:
        lookup = _names(joint_names)
        if np.ma.isMaskedArray(joints_camera_m):
            raise ValueError('Unmasked predicted camera joints required')
        joints = np.asarray(joints_camera_m)
        if joints.shape != (127, 3) or joints.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Predicted floating 127x3 camera joints required')
        if any(name not in lookup for name in names):
            raise ValueError('Named wrist/index1/middle1 absent; no ordinal fallback')
        wrist, index, middle = joints[[lookup[name] for name in names]].astype(np.float64)
        if not np.isfinite([wrist, index, middle]).all() or np.any(np.array([wrist, index, middle])[:, 2] <= 0):
            raise ValueError('Named hand nonfinite or behind camera')
        with np.errstate(over='ignore', invalid='ignore'):
            radial, distal = index-wrist, middle-wrist
            nr, nd = np.linalg.norm(radial), np.linalg.norm(distal)
        if not np.isfinite([nr, nd]).all() or min(nr, nd) <= MIN_VECTOR_M:
            raise ValueError('Named hand vector is zero or numerically unstable')
        radial, y = radial/nr, distal/nd
        sine = float(np.linalg.norm(np.cross(radial, y)))
        result['normalized_cross_length'] = sine
        if not np.isfinite(sine) or sine < MIN_SIN_ANGLE:
            raise ValueError('Named hand near-collinear; no ideal frame rescue')
        x = radial-y*np.dot(radial, y)
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        rotation = np.column_stack((x, y, z))
        if (not np.isfinite(rotation).all() or
                not np.allclose(rotation.T@rotation, np.eye(3), atol=SO3_ATOL, rtol=0) or
                abs(np.linalg.det(rotation)-1.) > SO3_ATOL):
            raise ValueError('Named hand construction did not produce proper SO3')
        result.update(valid=True, origin_camera_m=wrist.tolist(), R_frame_to_camera=rotation.tolist())
    except ValueError as error:
        result['reasons'].append(str(error))
    return result


def _mask_support(point, K, mask, name):
    """Distance to original foreground pixel centres, not a dilated mask."""
    height, width = mask.shape
    result = dict(name=name, valid=False, reasons=[], uv=None, positive_Z=False,
                  inside_image=False, mask_support_proxy=False,
                  nearest_foreground_within_radius_px=None)
    p = np.asarray(point, dtype=np.float64)
    if not np.isfinite(p).all() or p[2] <= 0:
        result['reasons'].append('Nonfinite or nonpositive camera-Z landmark')
        return result
    result['positive_Z'] = True
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        q = K@p
        uv = q[:2]/q[2]
    if not np.isfinite(uv).all():
        result['reasons'].append('Nonfinite original landmark projection')
        return result
    result['uv'] = uv.tolist()
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        result['reasons'].append('Original landmark projection outside image')
        return result
    result['inside_image'] = True
    low = np.maximum(np.ceil(uv-MASK_RADIUS_PX-.5).astype(np.int64), 0)
    high = np.minimum(np.floor(uv+MASK_RADIUS_PX-.5).astype(np.int64), [width-1, height-1])
    yy, xx = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
    distances = (xx+.5-uv[0])**2+(yy+.5-uv[1])**2
    foreground = mask[low[1]:high[1]+1, low[0]:high[0]+1] & (distances <= MASK_RADIUS_PX**2)
    if not foreground.any():
        result['reasons'].append('No automatic person foreground within fixed radius5px')
        return result
    result.update(valid=True, mask_support_proxy=True,
                  nearest_foreground_within_radius_px=float(np.sqrt(distances[foreground].min())))
    return result


def evaluate_geometry(vertices_camera_m, joints_camera_m, faces, camera_K,
                      person_mask, joint_names, *, deadline=None):
    """Single-record fixed proxy gate; malformed/unsupported geometry fails.

    FP32 predictions are widened to FP64 without modifying their values. All
    supplied faces/vertices remain present. A failed raster/deadline produces an
    explicit invalid result, never clipped geometry or a filled silhouette.
    The caller must require PASS for all four independent input records.
    """
    report = dict(schema='world_reward.bridge_anchor_geometry_qa.v1', status='fail',
        operator_only=True, independent_RGB_hand_support_verified=False,
        hand_anatomical_accuracy_verified=False, accuracy_verified=False,
        contact_verified=False, adoption=False, geometry_mutated=False,
        gate_reasons=[], invalid_reasons=[], left=None, right=None, silhouette=None,
        gates=dict(silhouette_IoU_min=IOU_MIN, mask_support_radius_px=MASK_RADIUS_PX,
                   min_hand_vector_m=MIN_VECTOR_M, min_normalized_cross_length=MIN_SIN_ANGLE,
                   left_required=True, right_diagnostic_only=True),
        renderer=dict(bytes=RENDERER_BYTES, sha256=RENDERER_SHA256,
                      max_candidate_tests=MAX_CANDIDATES, budget_seconds=BUDGET_SECONDS),
        silhouette_visibility='nearest human triangles only; separate object occlusion not modeled')
    try:
        lookup = _names(joint_names)
        if any(np.ma.isMaskedArray(x) for x in (vertices_camera_m, joints_camera_m, faces, camera_K, person_mask)):
            raise ValueError('Unmasked original predictions/camera/person mask required')
        v, j, f, k, mask = map(np.asarray, (vertices_camera_m, joints_camera_m, faces, camera_K, person_mask))
        floats = (np.dtype('float32'), np.dtype('float64'))
        if v.dtype not in floats or v.ndim != 2 or v.shape[1:] != (3,) or not 3 <= len(v) <= 100_000:
            raise ValueError('Original FP32/FP64 Nx3 predicted vertices required')
        if f.dtype != np.int64 or f.ndim != 2 or f.shape[1:] != (3,) or not 1 <= len(f) <= 100_000:
            raise ValueError('Original bounded I64 Fx3 faces required')
        if j.dtype not in floats or j.shape != (127, 3) or not np.isfinite(j).all() or np.any(j[:, 2] <= 0):
            raise ValueError('All127 original predicted camera joints must be finite with positive Z')
        if mask.dtype != np.bool_ or mask.ndim != 2 or not (2 <= mask.shape[0] <= 480 and 2 <= mask.shape[1] <= 640):
            raise ValueError('Original bounded binary boolean automatic person mask required')
        if k.dtype != np.float64 or k.shape != (3, 3):
            raise ValueError('Original FP64 3x3 camera K required')
        now = time.monotonic()
        if deadline is None:
            deadline = float(now+BUDGET_SECONDS)
        if type(deadline) is not float or not np.isfinite(deadline) or not now < deadline <= now+BUDGET_SECONDS:
            raise ValueError('Unexpired absolute monotonic QA deadline within100s required')
        renderer = _load_renderer()
        raster = renderer.render_triangles(v.astype(np.float64), f, k,
                                          int(mask.shape[1]), int(mask.shape[0]), deadline=deadline)
        predicted = raster['face_index'] >= 0
        if not np.array_equal(predicted, np.isfinite(raster['depth'])):
            raise ValueError('Original renderer hit/depth validity disagrees')
        intersection, union = int((predicted & mask).sum()), int((predicted | mask).sum())
        iou = intersection/union if union else None
        report['silhouette'] = dict(valid=bool(union), IoU=iou, intersection_pixels=intersection,
            union_pixels=union, predicted_pixels=int(predicted.sum()), automatic_person_pixels=int(mask.sum()),
            candidate_tests=int(raster['candidate_tests']),
            zero_area_projected_faces=int(raster['zero_area_projected_faces']))
        if iou is None or iou < IOU_MIN:
            report['gate_reasons'].append('Human-only silhouette IoU below fixed .7 or empty union')
        for side, label in (('l', 'left'), ('r', 'right')):
            frame = anatomical_frame(j, joint_names, side)
            frame['landmarks'] = [_mask_support(j[lookup[name]], k, mask, name)
                if name in lookup else dict(name=name, valid=False, reasons=['Named landmark absent'])
                for name in frame['names']]
            frame['mask_support_proxy_passed'] = all(row['valid'] for row in frame['landmarks'])
            report[label] = frame
        if not report['left']['valid']:
            report['gate_reasons'].append('Named left anatomical frame invalid')
        if not report['left']['mask_support_proxy_passed']:
            report['gate_reasons'].append('Named left wrist/index1/middle1 person-mask proxy unsupported')
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU geometry QA deadline exceeded')
        if not report['gate_reasons']:
            report['status'] = 'pass'
    except (ValueError, TimeoutError, OSError) as error:
        report['invalid_reasons'].append(str(error))
        report['status'] = 'fail'
    return report
