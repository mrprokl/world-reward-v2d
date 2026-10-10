"""Clip-global native articulation continuation with independent acceptance QA.

Native ABI: 23 SO3 rotations encoded column0 then column1 (138 controls),
58 SO2 rotations encoded sin/cos (116), then six fixed internal translations.
Source: video_to_data 7c0d3b9 lib_mhr/body_pose.py, 4231 bytes, SHA below.
The 136 direct MHR pose is NOT these controls: only the native decode callback
may produce it. No mesh/axis/Euler interpolation or per-frame alpha is allowed.

One deterministic alpha sequence interpolates full dynamic A toward full B.
Optional translation-only placement is followed by a NEW native decode, never
cached/translated mesh replay. All active frozen witness bounds and predefined
whole-clip automatic RGB/motion gates must pass. Alpha0 means actual moving A
fallback, NOT improvement. Baseline itself must decode/verify against its frozen
bounds; approximate native parity alone cannot certify those tight bounds.
Source/GT provenance and full native direct-control replay belong to the caller.
"""
from dataclasses import dataclass, asdict
from types import MappingProxyType

import numpy as np
from scipy.spatial.transform import Rotation

from . import contact_feasible_placement as contact
from .shared_identity import NATIVE_PARAMETER_DIMS, validate_native_parameters

BODY_ABI = MappingProxyType(dict(rotation_control_order='23_SO3_column0_then_column1_58_SO2_sin_cos_6_internalT',
    so3_blocks=23, so2_blocks=58, rotation_controls=254, internal_translation_start=254,
    source_revision='7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80', source_bytes=4231,
    source_sha256='6ee1e9ec8b692acf8857e827ed21201a4e8068e84a3cb95e986981e0a7949d23'))


@dataclass(frozen=True)
class ContinuationProtocol:
    """Frozen generic search/work policy, not fitted contact/RGB weights."""
    dyadic_backtracks: int = 8
    development_reference: str = 'native_ABI_and_analytic_contract_before_real_continuation_not_quality_calibration'

    def __post_init__(self):
        if (type(self.dyadic_backtracks) is not int or not 0 <= self.dyadic_backtracks <= 12
                or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Bounded externally declared clip-global dyadic policy required')


@dataclass(frozen=True)
class ObservationGate:
    name: str
    maximum_ratio: float
    absolute_numerical_slack: float
    allow_missing: bool = False

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or type(self.allow_missing) is not bool
                or type(self.maximum_ratio) not in (int, float) or not np.isfinite(self.maximum_ratio)
                or self.maximum_ratio < 1 or type(self.absolute_numerical_slack) not in (int, float)
                or not np.isfinite(self.absolute_numerical_slack) or self.absolute_numerical_slack < 0):
            raise ValueError('Explicit nonregression observation metric/ratio/numerical tolerance required')


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _so3(block):
    x, y = block[..., :3], block[..., 3:]
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    if (nx <= 1e-12).any(): raise ValueError('Degenerate native SO3 first column')
    x = x/nx; y = y-(x*y).sum(-1, keepdims=True)*x
    ny = np.linalg.norm(y, axis=-1, keepdims=True)
    if (ny <= 1e-12).any(): raise ValueError('Degenerate native SO3 second column')
    y = y/ny
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def geodesic_body_controls(a, b, alpha):
    """Same native representation; exact endpoints, wrapped SO2, shortest SO3."""
    if (type(alpha) not in (int, float) or not np.isfinite(alpha) or not 0 <= alpha <= 1
            or not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray)
            or a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape
            or a.ndim != 2 or a.shape[1] != 260 or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not _same(a[:, 254:], b[:, 254:])):
        raise ValueError('Complete native FP32 body endpoints/fixed internal translations/one scalar alpha required')
    count = len(a); ra = _so3(a[:, :138].astype(float).reshape(count, 23, 6))
    rb = _so3(b[:, :138].astype(float).reshape(count, 23, 6))
    sa = a[:, 138:254].astype(float).reshape(count, 58, 2)
    sb = b[:, 138:254].astype(float).reshape(count, 58, 2)
    if (np.linalg.norm(sa, axis=-1) <= 1e-12).any() or (np.linalg.norm(sb, axis=-1) <= 1e-12).any():
        raise ValueError('Degenerate native SO2 sin/cos control')
    if alpha == 0: return a.copy()
    if alpha == 1: return b.copy()
    relative = ra.swapaxes(-1, -2)@rb
    increment = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()*alpha
    rotations = ra@Rotation.from_rotvec(increment).as_matrix().reshape(count, 23, 3, 3)
    theta_a = np.arctan2(sa[..., 0], sa[..., 1]); theta_b = np.arctan2(sb[..., 0], sb[..., 1])
    theta = theta_a+alpha*np.arctan2(np.sin(theta_b-theta_a), np.cos(theta_b-theta_a))
    result = a.copy()
    result[:, :138] = np.concatenate((rotations[..., :, 0], rotations[..., :, 1]), axis=-1).reshape(count, 138)
    result[:, 138:254] = np.stack((np.sin(theta), np.cos(theta)), axis=-1).reshape(count, 116)
    return result


def blend_native(a, b, object_a, object_b, alpha):
    count = len(a['mhr_trans'])
    validate_native_parameters(a, count, require_shared_identity=True)
    validate_native_parameters(b, count, require_shared_identity=True)
    for key in a.keys()-{'mhr_trans', 'mhr_body_pose_cont'}:
        if not _same(a[key], b[key]): raise ValueError('Frozen native A/B block changed: '+key)
    oa = contact._float_array(object_a, (count, 3), 'Original object translation')
    ob = contact._float_array(object_b, (count, 3), 'Proposed object translation')
    result = {key: value.copy() for key, value in a.items()}
    result['mhr_body_pose_cont'] = geodesic_body_controls(a['mhr_body_pose_cont'], b['mhr_body_pose_cont'], alpha)
    if alpha == 1: result['mhr_trans'] = b['mhr_trans'].copy()
    elif alpha != 0: result['mhr_trans'] = (a['mhr_trans'].astype(float)+alpha*(b['mhr_trans'].astype(float)-a['mhr_trans'])).astype(np.float32)
    obj = oa if alpha == 0 else ob if alpha == 1 else oa+alpha*(ob-oa)
    validate_native_parameters(result, count, require_shared_identity=True)
    return result, obj


def _decode_geometry(value, count, reference_faces=None, reference_scales=None):
    needed = {'human_vertices', 'human_joints', 'human_keypoints', 'human_faces', 'frame_index', 'pose', 'scales'}
    if not isinstance(value, dict) or set(value) != needed:
        raise ValueError('Native full geometry + directly decoded136 pose/68 scales required; no guessed controls')
    for key, trailing in [('human_vertices', (18439, 3)), ('human_joints', (127, 3)),
                          ('human_keypoints', (70, 3)), ('pose', (136,))]:
        a = value[key]
        if (not isinstance(a, np.ndarray) or a.dtype != np.float32 or a.shape != (count, *trailing)
                or not np.isfinite(a).all() or (key != 'pose' and (a[..., 2] <= 0).any())):
            raise ValueError('Full native FP32 decoded geometry/control ABI failed: '+key)
    f = value['human_faces']; scales = value['scales']; grid = value['frame_index']
    if (not isinstance(f, np.ndarray) or f.dtype != np.int64 or f.ndim != 2 or f.shape[1] != 3 or not len(f)
            or f.min() < 0 or f.max() >= 18439 or grid.dtype != np.int64 or not np.array_equal(grid, np.arange(count))
            or scales.dtype != np.float32 or scales.shape != (68,) or not np.isfinite(scales).all()
            or (reference_faces is not None and not _same(f, reference_faces))
            or (reference_scales is not None and not _same(scales, reference_scales))):
        raise ValueError('Native full chronology/fixed topology/direct clip scales changed')
    return {key: contact._immutable(array) for key, array in value.items()}


def _object_in_front(evidence, translation):
    for rotation, t in zip(evidence.object_rotation, translation):
        z = evidence.object_vertices@rotation[2]+t[2]
        if not np.isfinite(z).all() or (z <= 0).any(): return False
    return True


def _contacts(evidence, geometry, object_translation):
    surface = contact._NearestSurface(evidence.object_vertices, evidence.object_faces)
    active = evidence.activations; gaps = np.full(active.shape, np.nan)
    bounds = evidence.baseline_gap_m+evidence.config.numerical_slack_m
    for frame, side in zip(*np.nonzero(active)):
        identifier = evidence.witness_source_indices[frame, side]
        local = (geometry['human_vertices'][frame, identifier]-object_translation[frame])@evidence.object_rotation[frame]
        gaps[frame, side] = surface.closest(local)[0]
    return bool(np.all(gaps[active] <= bounds[active])), gaps


def _observation_acceptance(baseline, candidate, gates):
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError('Independent whole-clip observation metrics required')
    checks = {}
    for gate in gates:
        a, b = baseline.get(gate.name), candidate.get(gate.name)
        if a is None or b is None:
            checks[gate.name] = bool(gate.allow_missing and a is None and b is None)
        else:
            if any(type(v) not in (float, int) or not np.isfinite(v) or v < 0 for v in (a, b)):
                raise ValueError('Finite nonnegative observation metrics required')
            checks[gate.name] = bool(b <= a*gate.maximum_ratio+gate.absolute_numerical_slack)
    return all(checks.values()), checks


def continue_native_contact(a, b, object_a, object_b, evidence, *, decode_native,
        evaluate_observations, gates, protocol=ContinuationProtocol(), place_translations=None):
    """Whole-clip dyadic line search; independent native decode/observation callbacks.

    decode_native(params)->exact geometry schema above; native direct136 controls
    must correspond to these parameters. evaluate_observations(geometry,objT)
    returns only automatic whole-clip metric dict, not GT/labels. Optional
    place_translations(params,objT,geometry,evidence)->(humanT,objT) may alter ONLY
    full actor translations. It is called after a proposal decode and followed
    by a fresh native decode+contact check. Callback source/provenance is caller
    authenticated; core never calls a model, reads a file or uses ground truth.
    """
    if (not isinstance(evidence, contact.ContactFeasibilityEvidence) or not isinstance(protocol, ContinuationProtocol)
            or not callable(decode_native) or not callable(evaluate_observations)
            or (place_translations is not None and not callable(place_translations))
            or type(gates) not in (tuple, list) or not gates or any(not isinstance(g, ObservationGate) for g in gates)
            or len({g.name for g in gates}) != len(gates)):
        raise ValueError('Explicit frozen evidence/search/gates and independent native/observation callbacks required')
    count = len(evidence.frame_index); baseline_params, baseline_obj = blend_native(a, b, object_a, object_b, 0.)
    if len(baseline_params['mhr_trans']) != count or not np.array_equal(evidence.frame_index, np.arange(count)):
        raise ValueError('Original full native chronology required')
    geometry = _decode_geometry(decode_native({k: v.copy() for k, v in baseline_params.items()}), count)
    ok, gaps = _contacts(evidence, geometry, baseline_obj)
    if not ok or not _object_in_front(evidence, baseline_obj):
        raise ValueError('Actual decoded dynamic A fails frozen contact bounds/positive object depth; no guaranteed fallback')
    baseline_metrics = evaluate_observations(geometry, baseline_obj.copy())
    baseline_valid, _ = _observation_acceptance(baseline_metrics, baseline_metrics, gates)
    if not baseline_valid: raise ValueError('Required baseline observation metrics unavailable; no fabricated acceptance')
    attempts = []; baseline = (baseline_params, baseline_obj, geometry, gaps)
    selected = baseline; selected_alpha = 0.; selected_metrics = baseline_metrics
    for step in range(protocol.dyadic_backtracks+1):
        alpha = 2.**-step
        params, obj = blend_native(a, b, object_a, object_b, alpha)
        row = dict(alpha=alpha, status='fail'); attempts.append(row)
        try:
            current = _decode_geometry(decode_native({k: v.copy() for k, v in params.items()}), count,
                geometry['human_faces'], geometry['scales'])
            if place_translations is not None:
                ht, ot = place_translations({k: v.copy() for k, v in params.items()}, obj.copy(), current, evidence)
                params['mhr_trans'] = contact._float_array(ht, (count, 3), 'Placed native human translation').astype(np.float32)
                obj = contact._float_array(ot, (count, 3), 'Placed object translation')
                validate_native_parameters(params, count, require_shared_identity=True)
                current = _decode_geometry(decode_native({k: v.copy() for k, v in params.items()}), count,
                    geometry['human_faces'], geometry['scales'])
            if not _object_in_front(evidence, obj):
                row['status'] = 'object_depth_rejected'; continue
            feasible, current_gaps = _contacts(evidence, current, obj)
            row['all_frozen_active_contact_bounds_passed'] = feasible
            if not feasible:
                row['status'] = 'contact_rejected'; continue
            metrics = evaluate_observations(current, obj.copy())
            accepted, checks = _observation_acceptance(baseline_metrics, metrics, gates)
            row.update(observation_gates=checks, status='accepted' if accepted else 'observation_rejected')
            if accepted:
                selected = (params, obj, current, current_gaps); selected_alpha = alpha; selected_metrics = metrics; break
        except contact.ContactFeasibilityFailure as exc:
            row.update(status='translation_placement_rejected', failure_frame=exc.frame_index,
                failure_type=type(exc).__name__, physical_infeasibility_claimed=False)
    params, obj, geometry, gaps = selected
    return dict(schema='world_reward.native_contact_continuation.v1',
        status='accepted_native_continuation' if selected_alpha > 0 else 'dynamic_A_fallback_no_improvement',
        alpha=selected_alpha, clip_global_alpha=True, parameters=params, object_translation=obj,
        geometry=geometry, witness_gaps_m=gaps, original_activations=evidence.activations,
        original_witness_ids=evidence.witness_source_indices, observation_metrics=selected_metrics,
        baseline_observation_metrics=baseline_metrics, attempts=attempts, protocol=asdict(protocol),
        body_control_ABI=dict(BODY_ABI), full_original_frames=count, production_adopted=False,
        heldout_accuracy_verified=False, ground_truth_used=False, private_truth_read=False,
        native_decoder_callback_required=True, direct136_controls_from_callback_only=True,
        callback_provenance_independently_verified=False, metric_improvement_claimed=False,
        native_direct_replay_independently_verified=False, whole_hand_minimum_claimed=False,
        physical_contact_verified=False, baseline_fallback_dynamic=True)
