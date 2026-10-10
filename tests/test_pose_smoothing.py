import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward.pose_smoothing import local_chart_sg_reference


def run(r, t, centroid=None, *, fps=30., indices=None, preserve=True):
    return local_chart_sg_reference(r, t, frame_indices=np.arange(len(t)) if indices is None else indices,
        fps=fps, canonical_centroid=np.zeros(3) if centroid is None else centroid, preserve_time_zero=preserve)


def path(count=61):
    seconds = np.arange(count)/30
    r = Rotation.from_rotvec(np.column_stack([seconds*.08, seconds*-.12, seconds*.2])).as_matrix()
    c = np.column_stack([seconds, seconds**2*.1, 2+seconds**3*.02])
    centroid = np.array([.4, -.2, .3])
    t = c - np.einsum('tij,j->ti', r, centroid)
    return r, t, centroid, c


def test_constant_pose_full_T_and_exact_first_gauge_no_input_mutation():
    r = np.repeat(Rotation.from_rotvec([.4, -.2, .3]).as_matrix()[None], 21, axis=0)
    t = np.tile([.3, -.1, 2.], (21, 1)); before_r, before_t = r.copy(), t.copy()
    output = run(r, t, [.4, .1, -.2])
    np.testing.assert_allclose(output.rotation, r, atol=2e-15)
    np.testing.assert_allclose(output.translation, t, atol=5e-12)
    assert np.array_equal(output.rotation[0], r[0]) and np.array_equal(output.translation[0], t[0])
    assert np.array_equal(r, before_r) and np.array_equal(t, before_t)
    assert output.diagnostics['window_frames'] == 9 and output.diagnostics['dropped_frames'] == 0
    assert not output.diagnostics['accuracy_validated'] and not output.diagnostics['production_adopted']
    json.dumps(output.diagnostics, allow_nan=False)


def test_linear_and_cubic_centroid_and_constant_angular_velocity_exact_at_one_sided_edges():
    r, t, centroid, c = path()
    output = run(r, t, centroid)
    reconstructed = np.einsum('tij,j->ti', output.rotation, centroid) + output.translation
    np.testing.assert_allclose(reconstructed, c, atol=2e-11)
    np.testing.assert_allclose(output.rotation, r, atol=2e-14)
    np.testing.assert_allclose(output.translation, t, atol=2e-11)
    assert np.linalg.norm(output.translation[-1] - output.translation[0]) > 1.
    assert np.linalg.norm(output.rotation[-1] - output.rotation[0]) > .3


def test_rotations_cross_global_axis_angle_pi_without_global_unwrap_or_jump():
    angle = np.linspace(np.pi-.4, np.pi+.4, 41)
    r = Rotation.from_rotvec(np.column_stack([angle, angle*0, angle*0])).as_matrix()
    output = run(r, np.tile([.3, .2, 2.], (41, 1)))
    np.testing.assert_allclose(output.rotation, r, atol=2e-14)
    assert output.diagnostics['maximum_local_angle_radians'] < .17
    assert not output.diagnostics['global_unwrap']


def test_cubic_single_axis_rotation_exact_including_one_sided_endpoints():
    seconds = np.arange(61)/30
    angle = .3+seconds*.1+seconds**2*.02+seconds**3*.03
    axis = np.array([.4, -.2, .3]); axis /= np.linalg.norm(axis)
    r = Rotation.from_rotvec(angle[:, None]*axis).as_matrix()
    output = run(r, np.tile([.3, .2, 2.], (61, 1)))
    np.testing.assert_allclose(output.rotation, r, atol=3e-14)
    np.testing.assert_allclose(output.rotation.transpose(0, 2, 1) @ output.rotation,
        np.broadcast_to(np.eye(3), output.rotation.shape), atol=3e-14)


def test_canonical_origin_change_preserves_identical_reconstructed_geometry():
    r, t, centroid, _ = path()
    rng = np.random.default_rng(90)
    r = Rotation.from_rotvec(rng.normal(0, .006, (len(t), 3))).as_matrix() @ r
    shift = np.array([3., -.8, 1.2])
    shifted_t = t + np.einsum('tij,j->ti', r, shift)
    first = run(r, t, centroid)
    shifted = run(r, shifted_t, centroid-shift)
    np.testing.assert_allclose(shifted.rotation, first.rotation, atol=2e-14)
    np.testing.assert_allclose(shifted.translation,
        first.translation+np.einsum('tij,j->ti', first.rotation, shift), atol=2e-11)


def test_shared_world_gauge_rotation_and_translation_equivariance():
    r, t, centroid, _ = path()
    q = Rotation.from_rotvec([.3, -.7, .2]).as_matrix(); shift = np.array([7., -3., 4.])
    base = run(r, t, centroid)
    changed = run(q@r, t@q.T+shift, centroid)
    np.testing.assert_allclose(changed.rotation, q@base.rotation, atol=2e-14)
    np.testing.assert_allclose(changed.translation, base.translation@q.T+shift, atol=3e-11)


def test_zero_mean_high_frequency_noise_reduces_without_freezing_moving_pose():
    r, t, centroid, c = path(121)
    phase = np.arange(len(t)); noise = .02*np.sin(phase*2*np.pi/3)
    perturbation = np.column_stack([noise, -.4*noise, .7*noise])
    noisy_r = Rotation.from_rotvec(perturbation).as_matrix() @ r
    noisy_c = c + np.column_stack([noise*2, -noise, noise*.3])
    noisy_t = noisy_c - np.einsum('tij,j->ti', noisy_r, centroid)
    result = run(noisy_r, noisy_t, centroid)
    recovered_c = np.einsum('tij,j->ti', result.rotation, centroid) + result.translation
    selected = slice(4, -4)
    before_c = np.mean((noisy_c[selected]-c[selected])**2)
    after_c = np.mean((recovered_c[selected]-c[selected])**2)
    before_angle = Rotation.from_matrix(noisy_r[selected] @ r[selected].transpose(0, 2, 1)).magnitude()
    after_angle = Rotation.from_matrix(result.rotation[selected] @ r[selected].transpose(0, 2, 1)).magnitude()
    assert after_c < .15*before_c and np.mean(after_angle**2) < .15*np.mean(before_angle**2)
    assert np.linalg.norm(recovered_c[-1]-recovered_c[0]) > 3.
    assert Rotation.from_matrix(result.rotation[-1] @ result.rotation[0].T).magnitude() > .8
    assert np.array_equal(result.rotation[0], noisy_r[0]) and np.array_equal(result.translation[0], noisy_t[0])


@pytest.mark.parametrize('fps,window', [(24., 7), (25., 7), (30., 9), (60., 19)])
def test_frozen_physical_duration_odd_nearest_window_not_per_clip_selection(fps, window):
    r, t, c, _ = path()
    output = run(r, t, c, fps=fps)
    assert output.diagnostics['window_frames'] == window
    assert output.diagnostics['polynomial_degree'] == 3
    assert output.diagnostics['proposed_window_seconds'] == .3


def test_preserve_time_zero_is_explicit_optional_not_per_frame_alignment():
    r, t, c, _ = path()
    t[0] += [.1, -.1, .2]
    output = run(r, t, c, preserve=False)
    assert not np.array_equal(output.translation[0], t[0])
    assert not output.diagnostics['time_zero_gauge_preserved']


def test_near_pi_local_chart_rejects_explicitly_instead_of_silent_repair():
    r, t, c, _ = path()
    r[4] = Rotation.from_rotvec([np.pi-1e-4, 0, 0]).as_matrix() @ r[0]
    with pytest.raises(ValueError, match='chart approaches pi'):
        run(r, t, c)


@pytest.mark.parametrize('fault', ['invalidR', 'reflection', 'nanR', 'nanT', 'maskedR', 'maskedT',
    'masked_indices', 'masked_centroid', 'nonconstant_centroid', 'nan_centroid', 'complex',
    'index_hole', 'float_indices', 'offset_indices', 'duplicate_indices', 'bool_indices',
    'badRshape', 'badTshape', 'short', 'bad_fps', 'nan_fps', 'bool_fps', 'bool_gauge'])
def test_invalid_input_fails_without_masking_frames_or_changing_parameters(fault):
    r, t, c, _ = path(); indices = np.arange(len(t)); fps = 30.; preserve = True
    if fault == 'invalidR': r[3, 0, 0] += .1
    elif fault == 'reflection': r[3, :, 0] *= -1
    elif fault == 'nanR': r[3, 0, 0] = np.nan
    elif fault == 'nanT': t[3, 0] = np.nan
    elif fault == 'maskedR': r = np.ma.array(r)
    elif fault == 'maskedT': t = np.ma.array(t)
    elif fault == 'masked_indices': indices = np.ma.array(indices)
    elif fault == 'masked_centroid': c = np.ma.array(c)
    elif fault == 'nonconstant_centroid': c = np.tile(c, (len(t), 1))
    elif fault == 'nan_centroid': c[0] = np.nan
    elif fault == 'complex': t = t.astype(complex)
    elif fault == 'index_hole': indices[3] += 1
    elif fault == 'float_indices': indices = indices.astype(float)
    elif fault == 'offset_indices': indices += 1
    elif fault == 'duplicate_indices': indices[3] = 2
    elif fault == 'bool_indices': indices = indices.astype(bool)
    elif fault == 'badRshape': r = r[:, :2]
    elif fault == 'badTshape': t = t[:, :2]
    elif fault == 'short': r, t, indices = r[:8], t[:8], indices[:8]
    elif fault == 'bad_fps': fps = 0.
    elif fault == 'nan_fps': fps = np.nan
    elif fault == 'bool_fps': fps = True
    elif fault == 'bool_gauge': preserve = 1
    with pytest.raises(ValueError):
        run(r, t, c, fps=fps, indices=indices, preserve=preserve)
