"""Manufactured anatomical contact and exact surface contracts; no assets/GT."""
from dataclasses import asdict
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import world_reward.sequence_pose as runtime
from world_reward.sequence_pose import (RGBDepthConfig, SequenceContactConfig,
    SequenceContactEvidence, SequencePoseConfig, refine_sequence)


def config():
    return SequencePoseConfig(1.5, .05, .15, 20., 100., .1, .02, 60,
        'manufactured:anatomical_contact_unit_contract_not_quality_validation')


def contact_config():
    return SequenceContactConfig(.02, 8, 100_000_000,
        'manufactured:fixed_anatomical_contact_unit_scale_not_challenge_selection')


def fixture(n=12, noise=False):
    vertices = np.array([[-.2, -.15, -.1], [.2, -.15, -.1], [.2, .15, -.1], [-.2, .15, -.1],
                         [-.2, -.15, .1], [.2, -.15, .1], [.2, .15, .1], [-.2, .15, .1]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int64)
    time = np.arange(n)/30
    rotations = Rotation.from_rotvec(np.column_stack((time*0, .5*time, time*0))).as_matrix()
    translations = np.column_stack((.2*time, .03*time, 2.+.1*time))
    K = np.array([[250., 0., 128.], [0., 250., 128.], [0., 0., 1.]])
    xyz = vertices[None]@rotations.swapaxes(-1, -2)+translations[:, None]
    xy = xyz[..., :2]/xyz[..., 2, None]*250+128
    if noise: xy += np.random.default_rng(20261101).normal(0, .5, xy.shape)
    visible = np.ones(xy.shape[:2], bool)
    hand_local = np.array([[.2, -.02, 0.], [.2, 0., 0.], [.2, .02, 0.]])
    hands = np.full((n, 2, len(hand_local), 3), np.nan)
    hands[:, 0] = hand_local[None]@rotations.swapaxes(-1, -2)+translations[:, None]
    hand_visible = np.zeros(hands.shape[:-1], bool); hand_visible[:, 0] = True
    active = np.zeros((n, 2), bool); active[:, 0] = True
    evidence = SequenceContactEvidence(active, hands, hand_visible, faces,
        'manufactured:automatic_contact_source_placeholder_not_challenge_provenance')
    return vertices, faces, K, rotations, translations, xy, visible, evidence


def call(f, translations=None, **kwargs):
    v, _, K, r, t, xy, visible, _ = f
    return refine_sequence(v, v, xy, visible, r, t if translations is None else translations,
        np.ones(len(t), bool), K, np.arange(len(t)), 30, config(), **kwargs)


def test_point_to_triangle_face_interior_edge_and_vertex_are_continuous():
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    surface = runtime._ContactTriangleSurface(vertices, np.array([[0, 1, 2]], np.int64))
    p = np.array([[.5, .5, .3], [1., -1., 0.], [-1., -1., 0.], [1., 1., 0.], [2., 2., 0.]])
    np.testing.assert_allclose(surface.distances(p), [.3, 1., np.sqrt(2), 0., np.sqrt(2)], atol=1e-14)
    assert np.linalg.norm(p[0]-vertices, axis=1).min() > .7  # Not nearest vertices.


def test_degenerate_faces_retain_their_segments_and_vertices_without_deletion():
    v = np.array([[0., 0., 0.], [2., 0., 0.], [1., 0., 0.], [10., 0., 0.]])
    faces = np.array([[0, 1, 2], [3, 3, 3]], np.int64)
    surface = runtime._ContactTriangleSurface(v, faces)
    assert len(surface.starts) == 2
    np.testing.assert_allclose(surface.distances(np.array([[1., 1., 0.], [10., 0., 0.]])), [1., 0.])


def test_surface_matches_independent_scalar_reference_and_chunking():
    from world_reward.continuous_surface import numpy_reference_distance_squared
    rng = np.random.default_rng(20261102)
    vertices = rng.normal(size=(21, 3)); faces = np.arange(21).reshape(-1, 3)
    points = rng.normal(size=(35, 3))
    surface = runtime._ContactTriangleSurface(vertices, faces)
    expected = np.sqrt(numpy_reference_distance_squared(points, vertices[faces]))
    np.testing.assert_allclose(surface.distances(points), expected, atol=1e-14)
    repeated = runtime._ContactTriangleSurface(vertices, np.tile(faces, (301, 1)))
    np.testing.assert_allclose(repeated.distances(points), expected, atol=1e-14)


def test_absent_contact_is_exact_RGB_and_RGBD_default_not_pseudo_attraction():
    f = fixture(); original = f[-1]
    absent = SequenceContactEvidence(np.zeros_like(original.activations), original.hand_points_camera,
        original.hand_visible, original.object_faces, original.source_reference)
    baseline = call(f)
    off = call(f, contact_evidence=absent, contact_config=contact_config())
    none = call(f, contact_evidence=None, contact_config=None)
    for result in (off, none):
        np.testing.assert_array_equal(result.rotations, baseline.rotations)
        np.testing.assert_array_equal(result.translations, baseline.translations)
        assert result.diagnostics == baseline.diagnostics
    v, _, _, r, t, _, visible, _ = f
    depth = (v[None]@r.swapaxes(-1, -2)+t[:, None])[..., 2]
    kwargs = dict(tracks_depth_m=depth, depth_visible=visible,
        depth_config=RGBDepthConfig(.02, 'manufactured:unit'))
    d = call(f, **kwargs); dc = call(f, **kwargs, contact_evidence=absent, contact_config=contact_config())
    np.testing.assert_array_equal(d.translations, dc.translations)
    np.testing.assert_array_equal(d.rotations, dc.rotations); assert d.diagnostics == dc.diagnostics


def test_moving_hand_object_contact_preserves_motion_and_reduces_wrong_pose():
    f = fixture(noise=True); truth = f[4]; prior = truth.copy()
    prior[1:, 0] += .015; prior[1:, 2] += .025*(-1.)**np.arange(1, len(prior))
    result = call(f, translations=prior, contact_evidence=f[-1], contact_config=contact_config())
    assert np.linalg.norm(result.translations-truth, axis=1).mean() < .5*np.linalg.norm(prior-truth, axis=1).mean()
    retention = np.linalg.norm(result.translations[-1]-result.translations[0])/np.linalg.norm(truth[-1]-truth[0])
    assert .85 < retention < 1.15
    assert result.diagnostics['static_constraint'] is False and result.diagnostics['contact_constraint'] is True
    assert result.diagnostics['geometry_camera_scale_unchanged'] is True
    assert result.diagnostics['contact_active_entries'] == len(truth)-1
    assert result.diagnostics['contact_activation_verified_by_module'] is False
    assert result.diagnostics['contact_accuracy_verified'] is False and result.diagnostics['quality_verified'] is False
    np.testing.assert_array_equal(result.rotations[0], f[3][0])
    np.testing.assert_array_equal(result.translations[0], prior[0])
    json.dumps(result.diagnostics, allow_nan=False)


def test_contact_resolves_projection_depth_ambiguity_with_no_depth_inputs():
    n = 8; a = Rotation.from_rotvec([0., .1, 0.]).as_matrix()
    t0 = np.array([0., 0., 2.]); t1 = a@t0+np.array([-.2, 0., .035])
    points = []
    for x in np.linspace(-.1, .1, 8):
        ray = np.array([x, 0., 1.])
        depth, _ = np.linalg.lstsq(np.column_stack((a@ray, -ray)), -(t1-a@t0), rcond=None)[0]
        points.append(depth*ray-t0)
    points = np.asarray(points)
    vertices = np.concatenate((points, np.array([[-.3, -.2, 0.], [.3, -.2, 0.], [.3, .2, 0.], [-.3, .2, 0.]])))
    faces = np.array([[8, 9, 10], [8, 10, 11]], np.int64)
    true_r = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    true_t = np.broadcast_to(t0, (n, 3)).copy(); true_r[0] = a; true_t[0] = t1
    prior_r = np.broadcast_to(a, (n, 3, 3)).copy(); prior_t = np.broadcast_to(t1, (n, 3)).copy()
    xyz = points[None]@true_r.swapaxes(-1, -2)+true_t[:, None]
    xy = xyz[..., :2]/xyz[..., 2, None]*250+128
    prior_xyz = points[None]@prior_r.swapaxes(-1, -2)+prior_t[:, None]
    np.testing.assert_allclose(prior_xyz[..., :2]/prior_xyz[..., 2, None]*250+128, xy, atol=1e-12)
    hands = np.full((n, 2, 3, 3), np.nan)
    hands[:, 0] = np.array([[0., -.05, 0.], [0., 0., 0.], [0., .05, 0.]])[None]@true_r.swapaxes(-1, -2)+true_t[:, None]
    supported = np.zeros(hands.shape[:-1], bool); supported[:, 0] = True
    active = np.zeros((n, 2), bool); active[:, 0] = True
    evidence = SequenceContactEvidence(active, hands, supported, faces, 'manufactured:fixed_active_contact')
    K = np.array([[250., 0., 128.], [0., 250., 128.], [0., 0., 1.]])
    args = (vertices, points, xy, np.ones(xy.shape[:2], bool), prior_r, prior_t,
            np.ones(n, bool), K, np.arange(n), 30, config())
    rgb = refine_sequence(*args)
    contact = refine_sequence(*args, contact_evidence=evidence, contact_config=contact_config())
    old_error = np.linalg.norm(rgb.translations-true_t, axis=1).mean()
    new_error = np.linalg.norm(contact.translations-true_t, axis=1).mean()
    assert old_error > .02 and new_error < .003
    assert np.linalg.norm(contact.translations[1]-contact.translations[0]) > .8*np.linalg.norm(true_t[1]-true_t[0])
    assert 'depth_config' not in contact.diagnostics


def test_contact_residual_cannot_self_disable_when_candidate_moves_away(monkeypatch):
    f = fixture(n=6); evidence = f[-1]; actual = runtime.least_squares; seen = []
    def record(fun, x0, **kwargs):
        baseline = len(fun(x0))-(len(f[4])-1)
        start = fun(x0)[baseline:]
        far = x0.reshape(-1, 6).copy(); far[:, 3] += 2.
        residual = fun(far.ravel())
        assert len(residual) == len(fun(x0))
        assert np.all(residual[baseline:] > start+1.)
        for row, frame in enumerate(range(1, len(f[4]))):
            np.testing.assert_array_equal(kwargs['jac_sparsity'][baseline+row].indices,
                                          np.arange((frame-1)*6, frame*6))
        seen.append(True); return actual(fun, x0, **kwargs)
    monkeypatch.setattr(runtime, 'least_squares', record)
    result = call(f, contact_evidence=evidence, contact_config=contact_config())
    assert seen and result.diagnostics['contact_activation_policy'].startswith('caller_qualified_prefrozen')
    np.testing.assert_array_equal(evidence.activations[:, 0], True)


def test_evidence_copies_arrays_and_regular_anatomical_subset_is_fixed():
    f = fixture(); original = f[-1]
    active = original.activations.copy(); points = original.hand_points_camera.copy()
    supported = original.hand_visible.copy(); faces = original.object_faces.copy()
    evidence = SequenceContactEvidence(active, points, supported, faces, 'manufactured:source')
    active[:] = False; points[:] = np.nan; supported[:] = False; faces[:] = 0
    assert evidence.activations[:, 0].all() and evidence.hand_visible[:, 0].all()
    assert np.isfinite(evidence.hand_points_camera[:, 0]).all()
    for value in (evidence.activations, evidence.hand_points_camera, evidence.hand_visible, evidence.object_faces):
        assert not value.flags.writeable
        with pytest.raises(ValueError): value.setflags(write=True)
    config_ = SequenceContactConfig(.02, 2, 100_000_000, 'manufactured:unit')
    rows = runtime._contact_rows(evidence, config_, f[0], len(f[4]))
    np.testing.assert_array_equal(rows[-2], [0, 2])
    assert rows[-1] == 2*(len(f[4])-1)*len(f[1])


@pytest.mark.parametrize('kind', ['active_only', 'config_only', 'wrong_config', 'wrong_evidence'])
def test_contact_route_requires_explicit_evidence_and_config(kind):
    f = fixture(); kwargs = dict(contact_evidence=f[-1], contact_config=contact_config())
    if kind == 'active_only': del kwargs['contact_config']
    if kind == 'config_only': del kwargs['contact_evidence']
    if kind == 'wrong_config': kwargs['contact_config'] = config()
    if kind == 'wrong_evidence': kwargs['contact_evidence'] = {}
    with pytest.raises(ValueError, match='Explicit frozen'): call(f, **kwargs)


@pytest.mark.parametrize('kind', ['activation_dtype', 'activation_shape', 'point_dtype', 'point_shape',
    'support_dtype', 'support_shape', 'fake_unknown', 'nan_visible', 'inf_visible', 'faces_dtype',
    'faces_shape', 'negative_faces', 'empty_faces', 'no_source', 'masked'])
def test_contact_evidence_unknowns_and_anatomical_shapes_are_strict(kind):
    e = fixture()[-1]
    args = [e.activations.copy(), e.hand_points_camera.copy(), e.hand_visible.copy(), e.object_faces.copy(), e.source_reference]
    if kind == 'activation_dtype': args[0] = args[0].astype(np.uint8)
    if kind == 'activation_shape': args[0] = args[0][:, :1]
    if kind == 'point_dtype': args[1] = np.zeros(args[1].shape, np.int32)
    if kind == 'point_shape': args[1] = args[1][..., :2]
    if kind == 'support_dtype': args[2] = args[2].astype(np.uint8)
    if kind == 'support_shape': args[2] = args[2][..., :2]
    if kind == 'fake_unknown': args[1][:, 1] = 0.
    if kind == 'nan_visible': args[1][1, 0, 0] = np.nan
    if kind == 'inf_visible': args[1][1, 0, 0] = np.inf
    if kind == 'faces_dtype': args[3] = args[3].astype(float)
    if kind == 'faces_shape': args[3] = args[3][:, :2]
    if kind == 'negative_faces': args[3][0, 0] = -1
    if kind == 'empty_faces': args[3] = args[3][:0]
    if kind == 'no_source': args[4] = ''
    if kind == 'masked': args[0] = np.ma.array(args[0], mask=False)
    with pytest.raises(ValueError): SequenceContactEvidence(*args)


def test_active_contact_without_selected_supported_witness_fails_not_disables():
    f = fixture(); e = f[-1]; supported = e.hand_visible.copy(); points = e.hand_points_camera.copy()
    supported[3, 0] = False; points[3, 0] = np.nan
    evidence = SequenceContactEvidence(e.activations, points, supported, e.object_faces, e.source_reference)
    with pytest.raises(ValueError, match='do not self-disable'):
        call(f, contact_evidence=evidence, contact_config=contact_config())


def test_work_cap_and_foreign_faces_reject_before_optimizer(monkeypatch):
    f = fixture(); monkeypatch.setattr(runtime, 'least_squares', lambda *a, **k: pytest.fail('Optimizer should not run'))
    with pytest.raises(ValueError, match='work cap'):
        call(f, contact_evidence=f[-1], contact_config=SequenceContactConfig(.02, 8, 1, 'manufactured:unit'))
    e = f[-1]; faces = e.object_faces.copy(); faces[0, 0] = len(f[0])
    evidence = SequenceContactEvidence(e.activations, e.hand_points_camera, e.hand_visible, faces, e.source_reference)
    with pytest.raises(ValueError, match='same complete fixed mesh'):
        call(f, contact_evidence=evidence, contact_config=contact_config())


@pytest.mark.parametrize('change', [dict(contact_sigma_diameter=0.), dict(contact_sigma_diameter=float('nan')),
    dict(max_points_per_hand=9), dict(max_points_per_hand=True), dict(max_point_triangle_pairs=0),
    dict(development_reference='')])
def test_contact_config_positive_external_and_bounded(change):
    kwargs = asdict(contact_config()); kwargs.update(change)
    with pytest.raises(ValueError): SequenceContactConfig(**kwargs)
