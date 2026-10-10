"""Analytic native ABI/acceptance tests only; not reconstruction validation."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward import native_contact_continuation as core
from world_reward import contact_feasible_placement as contact
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS


def body(count=96, angle=0.):
    out = np.zeros((count, 260), np.float32)
    R = Rotation.from_rotvec([0., 0., angle]).as_matrix()
    out[:, :138] = np.tile(np.r_[R[:, 0], R[:, 1]], 23)
    out[:, 138:254] = np.tile([np.sin(angle), np.cos(angle)], 58)
    return out


def params(count=96):
    p = {k: np.zeros((count, d), np.float32) for k, d in NATIVE_PARAMETER_DIMS.items()}
    p['mhr_global_rot6d'][:] = [1, 0, 0, 1, 0, 0]
    p['mhr_body_pose_cont'] = body(count)
    p['mhr_trans'][:, 2] = 2.; p['mhr_trans'][:, 0] = np.arange(count, dtype=np.float32)*.001
    return p


def make_geometry(p):
    n = len(p['mhr_trans']); scalar = np.arctan2(p['mhr_body_pose_cont'][:, 1], p['mhr_body_pose_cont'][:, 0])
    points = np.broadcast_to(p['mhr_trans'][:, None], (n, 18439, 3)).copy()
    points[..., 2] += np.float32(.01)
    points[:, 0, 2] += (scalar*.1).astype(np.float32)
    return dict(human_vertices=points, human_joints=points[:, :127].copy(), human_keypoints=points[:, :70].copy(),
        human_faces=np.array([[0, 1, 2]], np.int64), frame_index=np.arange(n, dtype=np.int64),
        pose=np.broadcast_to(p['mhr_trans'][:, :1], (n, 136)).copy(), scales=np.zeros(68, np.float32))


def inputs(angle=.2):
    a = params(); b = deepcopy(a); b['mhr_body_pose_cont'] = body(angle=angle)
    oa = np.zeros((96, 3)); oa[:, 2] = 2.; oa[:, 0] = np.arange(96)*.001
    ob = oa.copy(); ob[:, 0] += .04
    v = np.array([[-2., -2., 0], [2., -2., 0], [2., 2., 0], [-2., 2., 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64); ids = np.array([[0], [1]], np.int64)
    active = np.ones((96, 2), bool); active[24:29] = False
    witnesses = make_geometry(a)['human_vertices'][:, [0, 1]].copy().astype(float); witnesses[~active] = np.nan
    sid = np.broadcast_to([0, 1], (96, 2)).copy(); sid[~active] = -1
    e = contact.freeze_contact_witnesses(witnesses, ids, sid, np.tile(np.eye(3), (96, 1, 1)), oa,
        v, f, active, np.arange(96, dtype=np.int64), config=contact.ContactPlacementConfig(.05, 1e-7, 1e-10, 400, 8,
            'analytic_CONT_not_quality'), source_reference='analytic_native_callback', witness_selection_reference='frozen_autoQA')
    return a, b, oa, ob, e


def gates():
    return (core.ObservationGate('RGB', 1.00000001, 1e-12), core.ObservationGate('motion', 1.00000001, 1e-12))


def observations(geometry, obj):
    return dict(RGB=1., motion=.1)


def test_geodesics_use_native_SO3_column_blocks_and_SO2_wrapped_path_exact_endpoints():
    a = body(angle=np.deg2rad(179)); b = body(angle=np.deg2rad(-179))
    a[:, 254:] = .1; b[:, 254:] = .1
    mid = core.geodesic_body_controls(a, b, .5)
    rotations = core._so3(mid[:, :138].astype(float).reshape(96, 23, 6))
    expected = Rotation.from_rotvec([0., 0., np.pi]).as_matrix()
    np.testing.assert_allclose(rotations, np.broadcast_to(expected, rotations.shape), atol=1e-7)
    np.testing.assert_allclose(mid[:, 138:254].reshape(96, 58, 2), np.broadcast_to([0., -1.], (96, 58, 2)), atol=1e-7)
    assert core.geodesic_body_controls(a, b, 0.).tobytes() == a.tobytes()
    assert core.geodesic_body_controls(a, b, 1.).tobytes() == b.tobytes()
    assert mid[:, 254:].tobytes() == a[:, 254:].tobytes()
    assert core.BODY_ABI['so3_blocks'] == 23 and core.BODY_ABI['so2_blocks'] == 58


@pytest.mark.parametrize('bad', ['internal', 'firstcolumn', 'secondcolumn', 'SO2', 'dtype', 'alpha'])
def test_native_control_degeneracy_no_linear_6D_mesh_or_internal_translation_repair(bad):
    a, b = body(), body(angle=.1); alpha = .5
    if bad == 'internal': b[0, 254] = 1
    if bad == 'firstcolumn': a[0, :3] = 0
    if bad == 'secondcolumn': b[0, 3:6] = b[0, :3]
    if bad == 'SO2': a[0, 138:140] = 0
    if bad == 'dtype': a = a.astype(float)
    if bad == 'alpha': alpha = np.ones(96)*.5
    with pytest.raises(ValueError): core.geodesic_body_controls(a, b, alpha)


def test_no_feasible_nonzero_candidate_returns_actual_dynamic_A_no_static_substitution():
    a, b, oa, ob, e = inputs()
    calls = []
    def decoder(p): calls.append(p['mhr_body_pose_cont'].copy()); return make_geometry(p)
    result = core.continue_native_contact(a, b, oa, ob, e, decode_native=decoder,
        evaluate_observations=observations, gates=gates(), protocol=core.ContinuationProtocol(3))
    assert result['status'] == 'dynamic_A_fallback_no_improvement' and result['alpha'] == 0
    assert [r['alpha'] for r in result['attempts']] == [1., .5, .25, .125]
    assert all(r['status'] == 'contact_rejected' for r in result['attempts'])
    assert len(calls) == 5
    assert result['parameters']['mhr_body_pose_cont'].tobytes() == a['mhr_body_pose_cont'].tobytes()
    np.testing.assert_array_equal(result['object_translation'], oa)
    assert np.linalg.norm(result['object_translation'][-1]-result['object_translation'][0]) > .09
    assert np.linalg.norm(result['geometry']['human_vertices'][-1, 0]-result['geometry']['human_vertices'][0, 0]) > .09
    np.testing.assert_array_equal(result['original_activations'], e.activations)
    np.testing.assert_array_equal(result['original_witness_ids'], e.witness_source_indices)


def test_independent_native_decoder_after_translation_roundoff_and_observation_gates():
    a, b, oa, ob, e = inputs(); counts = {'decode': 0, 'eval': 0, 'place': 0}
    def decoder(p): counts['decode'] += 1; return make_geometry(p)
    def placement(p, obj, geometry, evidence):
        counts['place'] += 1
        # Analytic placement accepts articulation while moving the complete
        # dynamic object along Z; no per-finger geometry is edited.
        obj[:, 2] = geometry['human_vertices'][:, 0, 2]-evidence.baseline_gap_m[:, 0]
        obj[~evidence.activations.any(1), 2] = oa[~evidence.activations.any(1), 2]
        return p['mhr_trans'], obj
    def eval_metrics(geometry, obj):
        counts['eval'] += 1
        # Whole-clip evidence rejects larger departures before accepting .25.
        return dict(RGB=1.+(float(obj[:, 0].mean())-float(oa[:, 0].mean()) > .011)*.1, motion=.1)
    # Here one hand's geometry is unaffected by synthetic articulation so the
    # shifted object would hurt it: freeze only genuinely supported first hand.
    active = e.activations.copy(); active[:, 1] = False
    witnesses = make_geometry(a)['human_vertices'][:, [0, 1]].astype(float); witnesses[~active] = np.nan
    sid = e.witness_source_indices.copy(); sid[:, 1] = -1
    e = contact.freeze_contact_witnesses(witnesses, e.source_indices, sid, e.object_rotation, oa,
        e.object_vertices, e.object_faces, active, e.frame_index, config=e.config,
        source_reference='analytic_native_callback', witness_selection_reference='frozen_autoQA')
    result = core.continue_native_contact(a, b, oa, ob, e, decode_native=decoder,
        evaluate_observations=eval_metrics, gates=gates(), place_translations=placement,
        protocol=core.ContinuationProtocol(3))
    assert result['alpha'] == .25 and result['status'] == 'accepted_native_continuation'
    assert counts == dict(decode=7, eval=4, place=3)
    assert result['attempts'][0]['status'] == 'observation_rejected'
    assert result['attempts'][-1]['status'] == 'accepted'
    assert result['geometry']['human_vertices'].dtype == np.float32
    assert result['direct136_controls_from_callback_only'] and not result['native_direct_replay_independently_verified']


def test_stale_translated_mesh_does_not_substitute_for_native_redecode():
    a, b, oa, ob, e = inputs(); calls = []
    def decode(p): calls.append(p['mhr_trans'].copy()); return make_geometry(p)
    def place(p, obj, geometry, evidence):
        moved = p['mhr_trans'].copy(); moved[:, 2] -= .1
        return moved, obj
    result = core.continue_native_contact(a, b, oa, ob, e, decode_native=decode,
        evaluate_observations=observations, gates=gates(), place_translations=place,
        protocol=core.ContinuationProtocol(0))
    assert len(calls) == 3 and np.max(np.abs(calls[2]-calls[1])) > .09
    assert result['status'] == 'dynamic_A_fallback_no_improvement'


def test_native_exact_schema_direct136_and_original_topology_fail_closed():
    a, b, oa, ob, e = inputs()
    for key in ('pose', 'scales', 'human_vertices'):
        def bad(p):
            g = make_geometry(p); del g[key]; return g
        with pytest.raises(ValueError): core.continue_native_contact(a, b, oa, ob, e,
            decode_native=bad, evaluate_observations=observations, gates=gates())
    def badbaseline(p):
        g = make_geometry(p); g['human_vertices'][:, 0, 2] += .01; return g
    with pytest.raises(ValueError, match='dynamic A fails'): core.continue_native_contact(a, b, oa, ob, e,
        decode_native=badbaseline, evaluate_observations=observations, gates=gates())


def test_metrics_must_exist_unless_externally_explicit_textureless_missing_gate():
    a, b, oa, ob, e = inputs(angle=0.)
    with pytest.raises(ValueError): core.continue_native_contact(a, b, oa, ob, e,
        decode_native=make_geometry, evaluate_observations=lambda *_: {}, gates=gates())
    allowed = (core.ObservationGate('RGB', 1.00000001, 1e-12, allow_missing=True),)
    result = core.continue_native_contact(a, b, oa, ob, e, decode_native=make_geometry,
        evaluate_observations=lambda *_: dict(RGB=None), gates=allowed)
    assert result['alpha'] == 1. and not result['metric_improvement_claimed']


def test_only_native_body_articulation_and_two_actor_translations_can_change():
    a, b, oa, ob, e = inputs()
    for key in ('mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_global_rot6d'):
        bad = deepcopy(b); bad[key][0, 0] += 1
        with pytest.raises(ValueError): core.blend_native(a, bad, oa, ob, .5)
    params, obj = core.blend_native(a, b, oa, ob, .5)
    for key in a.keys()-{'mhr_body_pose_cont', 'mhr_trans'}: assert params[key].tobytes() == a[key].tobytes()
    np.testing.assert_allclose(obj, oa+.5*(ob-oa), atol=0)
    assert params['mhr_body_pose_cont'].dtype == np.float32


def test_negative_object_depth_outside_active_contacts_cannot_hide_bad_frames():
    a, b, oa, ob, e = inputs(angle=0.); ob[25, 2] = -1.
    result = core.continue_native_contact(a, b, oa, ob, e, decode_native=make_geometry,
        evaluate_observations=observations, gates=gates(), protocol=core.ContinuationProtocol(0))
    assert result['status'] == 'dynamic_A_fallback_no_improvement'
    assert result['attempts'][0]['status'] == 'object_depth_rejected'


def test_translation_projection_rejection_is_logged_then_next_global_alpha_not_frame_selection():
    a, b, oa, ob, e = inputs(angle=0.); failures = []
    def projection(p, obj, geometry, evidence):
        alpha = round((float(obj[0, 0])-oa[0, 0])/.04, 2)
        if alpha > .3:
            failures.append(alpha); raise contact.ContactFeasibilityFailure(99, 'bounded—notphysical', .003)
        return p['mhr_trans'], obj
    result = core.continue_native_contact(a, b, oa, ob, e, decode_native=make_geometry,
        evaluate_observations=observations, gates=gates(), place_translations=projection,
        protocol=core.ContinuationProtocol(2))
    assert failures == [1., .5] and result['alpha'] == .25
    assert all(r['status'] == 'translation_placement_rejected' for r in result['attempts'][:2])
    assert result['full_original_frames'] == 96 and result['clip_global_alpha']
