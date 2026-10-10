"""Manufactured RGB/depth contracts; no challenge, sensor or learned assets."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import world_reward.sequence_pose as runtime
from world_reward.sequence_pose import RGBDepthConfig, SequencePoseConfig, refine_sequence


def config():
    return SequencePoseConfig(1.5, .05, .15, 20., 100., .1, .02, 60,
                              'manufactured:RGB_depth_unit_contracts_not_quality_validation')


def depth_config():
    # Externally specified manufactured-noise scale, never clip-specific tuning.
    return RGBDepthConfig(.02, 'manufactured:RGB_depth_unit_contracts_not_depth_calibration')


def fixture(n=12):
    rng = np.random.default_rng(20261030)
    vertices = rng.uniform(-.15, .15, (12, 3))
    time = np.arange(n)/30
    rotations = Rotation.from_rotvec(np.column_stack((time*0, .4*time, time*0))).as_matrix()
    translations = np.column_stack((.2*time, .06*time, 2.+.15*time))
    K = np.array([[250., 0., 128.], [0., 250., 128.], [0., 0., 1.]])
    xyz = vertices[None] @ rotations.swapaxes(-1, -2)+translations[:, None]
    xy = xyz[..., :2]/xyz[..., 2, None]*250+128
    visible = np.ones(xy.shape[:2], bool)
    return vertices, K, rotations, translations, xy, xyz[..., 2].copy(), visible


def call(f, **kwargs):
    v, K, r, t, xy, _, visible = f
    return refine_sequence(v, v, xy, visible, r, t, np.ones(len(t), bool),
                           K, np.arange(len(t)), 30, config(), **kwargs)


def test_explicit_none_preserves_exact_RGB_only_results_and_diagnostics():
    f = fixture(); f[3][1:, 0] += .005*(-1.)**np.arange(1, len(f[3]))
    old = call(f)
    explicit = call(f, tracks_depth_m=None, depth_visible=None, depth_config=None)
    np.testing.assert_array_equal(old.rotations, explicit.rotations)
    np.testing.assert_array_equal(old.translations, explicit.translations)
    assert old.diagnostics == explicit.diagnostics
    assert old.diagnostics['method'] == 'fixed_shape_full_T_RGB_SE3_bundle_v1'
    assert not any('depth' in key for key in old.diagnostics)


def test_RGB_route_preserves_original_residual_order_and_sparse_shape(monkeypatch):
    f = fixture(n=6); v, _, _, _, _, _, visible = f
    actual = runtime.least_squares; seen = []
    def record(fun, x0, **kwargs):
        n, q = visible.shape
        assert len(fun(x0)) == 3*(n-1)*q+6*(n-1)+6*(n-2)
        assert kwargs['jac_sparsity'].shape == (len(fun(x0)), 6*(n-1))
        seen.append(fun(x0).copy()); return actual(fun, x0, **kwargs)
    monkeypatch.setattr(runtime, 'least_squares', record)
    call(f); call(f, tracks_depth_m=None, depth_visible=None, depth_config=None)
    np.testing.assert_array_equal(seen[0], seen[1])


def test_exact_projection_equivalent_depth_ambiguity_is_constrained_not_frozen():
    # A rank-two critical configuration admits two different proper rigid poses
    # with exactly identical RGB rays. Fixed mesh, not a depth/scale rescaling.
    n = 8; a = Rotation.from_rotvec([0., .2, 0.]).as_matrix()
    t0 = np.array([0., 0., 2.]); b = np.array([-.4, 0., .2]); t1 = a@t0+b
    vertices = []
    for x in np.linspace(-.2, .2, 8):
        ray = np.array([x, 0., 1.])
        distance, _ = np.linalg.lstsq(np.column_stack((a@ray, -ray)), -b, rcond=None)[0]
        vertices.append(distance*ray-t0)
    vertices = np.asarray(vertices)
    true_r = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    true_t = np.broadcast_to(t0, (n, 3)).copy(); true_r[0] = a; true_t[0] = t1
    prior_r = np.broadcast_to(a, (n, 3, 3)).copy()
    prior_t = np.broadcast_to(t1, (n, 3)).copy()
    truth = vertices[None]@true_r.swapaxes(-1, -2)+true_t[:, None]
    prior = vertices[None]@prior_r.swapaxes(-1, -2)+prior_t[:, None]
    xy = truth[..., :2]/truth[..., 2, None]*250+128
    prior_xy = prior[..., :2]/prior[..., 2, None]*250+128
    np.testing.assert_allclose(prior_xy, xy, atol=1e-12)
    assert np.max(np.abs(prior[..., 2]-truth[..., 2])) > .2
    K = np.array([[250., 0., 128.], [0., 250., 128.], [0., 0., 1.]])
    visible = np.ones(xy.shape[:2], bool)
    inputs = [vertices.copy(), prior_r.copy(), prior_t.copy(), K.copy(), xy.copy(), truth[..., 2].copy()]
    args = (vertices, vertices, xy, visible, prior_r, prior_t, np.ones(n, bool), K, np.arange(n), 30, config())
    rgb = refine_sequence(*args)
    rgbd = refine_sequence(*args, tracks_depth_m=truth[..., 2], depth_visible=visible, depth_config=depth_config())
    old_error = np.linalg.norm(rgb.translations-true_t, axis=1).mean()
    new_error = np.linalg.norm(rgbd.translations-true_t, axis=1).mean()
    assert old_error > .1 and new_error < .002
    assert np.linalg.norm(rgbd.translations[1]-rgbd.translations[0]) > .95*np.linalg.norm(true_t[1]-true_t[0])
    assert rgbd.diagnostics['static_constraint'] is False and rgbd.diagnostics['contact_constraint'] is False
    assert rgbd.diagnostics['depth_observations'] == (n-1)*len(vertices)
    np.testing.assert_array_equal(rgbd.rotations[0], prior_r[0])
    np.testing.assert_array_equal(rgbd.translations[0], prior_t[0])
    for value, initial in zip([vertices, prior_r, prior_t, K, xy, truth[..., 2]], inputs):
        np.testing.assert_array_equal(value, initial)


def test_moving_RGB_depth_reduces_noise_without_erasing_actual_motion():
    f = fixture(); v, K, r, truth, xy, depth, visible = f
    rng = np.random.default_rng(20261031)
    prior = truth+rng.normal(0., .008, truth.shape); prior[0] = truth[0]
    measured = depth+rng.normal(0., .003, depth.shape)
    result = refine_sequence(v, v, xy, visible, r, prior, np.ones(len(truth), bool),
        K, np.arange(len(truth)), 30, config(), tracks_depth_m=measured,
        depth_visible=visible, depth_config=depth_config())
    assert np.linalg.norm(result.translations-truth, axis=1).mean() < .3*np.linalg.norm(prior-truth, axis=1).mean()
    retention = np.linalg.norm(result.translations[-1]-result.translations[0])/np.linalg.norm(truth[-1]-truth[0])
    assert .9 < retention < 1.1
    assert result.diagnostics['geometry_camera_scale_unchanged'] is True
    assert result.diagnostics['depth_source_authenticated_by_module'] is False
    assert result.diagnostics['depth_accuracy_verified'] is False and result.diagnostics['quality_verified'] is False
    json.dumps(result.diagnostics, allow_nan=False)


def test_unknown_depth_has_no_residual_and_sparse_rows_are_local(monkeypatch):
    f = fixture(n=7); _, _, _, _, _, depth, rgb_flags = f
    flags = rgb_flags.copy(); flags[2:5] = False; flags[6, 1::2] = False
    depth[~flags] = np.nan
    actual = runtime.least_squares; seen = []
    def record(fun, x0, **kwargs):
        n, q = rgb_flags.shape
        baseline = 3*(n-1)*q+6*(n-1)+6*(n-2)
        frames, _ = np.nonzero(flags & (np.arange(n)[:, None] > 0))
        residual = fun(x0); pattern = kwargs['jac_sparsity']
        assert len(residual) == baseline+len(frames) and np.isfinite(residual).all()
        for offset, frame in enumerate(frames):
            np.testing.assert_array_equal(pattern[baseline+offset].indices, np.arange((frame-1)*6, frame*6))
        seen.append(True); return actual(fun, x0, **kwargs)
    monkeypatch.setattr(runtime, 'least_squares', record)
    result = call(f, tracks_depth_m=depth, depth_visible=flags, depth_config=depth_config())
    assert seen and result.diagnostics['depth_observations'] == int(flags[1:].sum())
    assert result.diagnostics['depth_supported_frames'] == int(flags.any(1).sum())
    np.testing.assert_allclose(result.translations, f[3], atol=1e-6)
    assert np.isnan(depth[~flags]).all()


def test_shared_RGB_depth_occlusion_remains_unknown_with_full_inferred_pose():
    f = fixture(); v, K, r, t, xy, depth, visible = f
    visible[4:7] = False; xy[~visible] = np.nan; depth[~visible] = np.nan
    result = refine_sequence(v, v, xy, visible, r, t, np.ones(len(t), bool),
        K, np.arange(len(t)), 30, config(), tracks_depth_m=depth,
        depth_visible=visible, depth_config=depth_config())
    assert result.diagnostics['latent_frames'] == 3
    assert len(result.translations) == len(t) and np.isfinite(result.translations).all()
    assert not result.observed_rgb[4:7].any()
    assert np.isnan(depth[4:7]).all()


@pytest.mark.parametrize('kind', ['depth_only', 'flags_only', 'config_only', 'missing_config', 'wrong_config'])
def test_depth_evidence_requires_all_explicit_inputs(kind):
    f = fixture(); kwargs = dict(tracks_depth_m=f[5], depth_visible=f[6], depth_config=depth_config())
    if kind == 'depth_only': kwargs = {'tracks_depth_m': f[5]}
    if kind == 'flags_only': kwargs = {'depth_visible': f[6]}
    if kind == 'config_only': kwargs = {'depth_config': depth_config()}
    if kind == 'missing_config': del kwargs['depth_config']
    if kind == 'wrong_config': kwargs['depth_config'] = config()
    with pytest.raises(ValueError, match='Explicit depth'): call(f, **kwargs)


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'masked', 'zero', 'negative', 'nan', 'inf',
    'flags_shape', 'flags_dtype', 'flags_masked', 'outside_RGB', 'fake_unknown', 'inf_unknown', 'anchor_only'])
def test_depth_unknowns_and_support_are_strict_not_repaired(kind):
    f = fixture(); depth, flags = f[5].copy(), f[6].copy()
    if kind == 'shape': depth = depth[:, :-1]
    if kind == 'dtype': depth = depth.astype(np.int32)
    if kind == 'masked': depth = np.ma.array(depth, mask=False)
    if kind == 'zero': depth[1, 0] = 0.
    if kind == 'negative': depth[1, 0] = -.1
    if kind == 'nan': depth[1, 0] = np.nan
    if kind == 'inf': depth[1, 0] = np.inf
    if kind == 'flags_shape': flags = flags[:, :-1]
    if kind == 'flags_dtype': flags = flags.astype(np.uint8)
    if kind == 'flags_masked': flags = np.ma.array(flags, mask=False)
    if kind == 'outside_RGB': f[6][1, 0] = False; f[4][1, 0] = np.nan
    if kind == 'fake_unknown': flags[1, 0] = False
    if kind == 'inf_unknown': flags[1, 0] = False; depth[1, 0] = np.inf
    if kind == 'anchor_only': flags[1:] = False; depth[~flags] = np.nan
    with pytest.raises(ValueError): call(f, tracks_depth_m=depth, depth_visible=flags, depth_config=depth_config())


@pytest.mark.parametrize('value', [0., -.02, float('nan'), float('inf'), True, '.02', None])
def test_depth_scale_is_explicit_positive_global_number(value):
    with pytest.raises(ValueError): RGBDepthConfig(value, 'manufactured:unit')


@pytest.mark.parametrize('reference', ['', ' ', None, 42])
def test_depth_reference_cannot_claim_unqualified_implicit_calibration(reference):
    with pytest.raises(ValueError): RGBDepthConfig(.02, reference)


def test_depth_config_is_frozen_and_scale_normalized():
    dc = depth_config(); assert asdict(dc)['depth_sigma_diameter'] == .02
    with pytest.raises(AttributeError): dc.depth_sigma_diameter = .03
    f = fixture(n=7); v, K, r, t, xy, depth, flags = f
    prior = t.copy(); prior[1:, 2] += .01*(-1.)**np.arange(1, len(t))
    args = (xy, flags, r, np.ones(len(t), bool), K, np.arange(len(t)), 30, config())
    a = refine_sequence(v, v, args[0], args[1], args[2], prior, *args[3:],
        tracks_depth_m=depth, depth_visible=flags, depth_config=dc)
    factor = 10.
    b = refine_sequence(v*factor, v*factor, args[0], args[1], args[2], prior*factor, *args[3:],
        tracks_depth_m=depth*factor, depth_visible=flags, depth_config=dc)
    np.testing.assert_allclose(a.rotations, b.rotations, atol=1e-6)
    np.testing.assert_allclose(a.translations, b.translations/factor, atol=1e-6)


def test_new_depth_option_does_not_change_sequence_config_fields():
    assert list(asdict(config())) == ['pixel_sigma', 'prior_centroid_sigma_diameter',
        'prior_rotation_sigma_rad', 'acceleration_sigma_diameter_s2',
        'angular_acceleration_sigma_rad_s2', 'prior_weight', 'temporal_weight',
        'max_nfev', 'development_reference']
    assert replace(config(), temporal_weight=.2).temporal_weight == .2
