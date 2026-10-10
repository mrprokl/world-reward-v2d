import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from world_reward.sequence_pose import SequencePoseConfig, initialize_missing_poses, refine_sequence


def config():
    return SequencePoseConfig(1.5, .05, .15, 20., 100., .1, .02, 60,
                              'manufactured_nonchallenge_DEV_v1_not_production')


def fixture(n=14):
    rng = np.random.default_rng(20261010)
    v = rng.uniform(-.2, .2, (16, 3))
    k = np.array([[250., 0., 128.], [0., 250., 128.], [0., 0., 1.]])
    a = np.arange(n)/30
    r = Rotation.from_rotvec(np.column_stack((a*0, a*.5, a*0))).as_matrix()
    t = np.column_stack((.2*a, a*0, 2.+a*0))
    xyz = v[None] @ r.swapaxes(-1, -2)+t[:, None]
    xy = xyz[..., :2]/xyz[..., 2, None]*250+128
    return v, k, r, t, xy


def test_missing_bilateral_preserves_observed_and_retains_full_T():
    _, _, r, t, _ = fixture()
    seen = np.ones(len(t), bool); seen[4:9] = False
    ri, ti = r.copy(), t.copy(); ri[~seen] = np.nan; ti[~seen] = np.nan
    out_r, out_t = initialize_missing_poses(ri, ti, seen)
    np.testing.assert_array_equal(out_r[seen], r[seen]); np.testing.assert_array_equal(out_t[seen], t[seen])
    np.testing.assert_allclose(out_r, r, atol=1e-12); np.testing.assert_allclose(out_t, t)


@pytest.mark.parametrize('edge', [0, -1])
def test_missing_edges_need_recovery_not_static_fallback(edge):
    _, _, r, t, _ = fixture(); seen = np.ones(len(t), bool); seen[edge] = False
    r[~seen] = np.nan; t[~seen] = np.nan
    with pytest.raises(ValueError, match='anchor'): initialize_missing_poses(r, t, seen)


def test_rgb_constraints_reduce_pose_noise_without_erasing_motion():
    v, k, r, t, xy = fixture()
    noisy = t.copy(); noisy[1:, 0] += .01*(-1.)**np.arange(1, len(t))
    inputs = [v.copy(), r.copy(), noisy.copy(), xy.copy()]
    out = refine_sequence(v, v, xy, np.ones(xy.shape[:2], bool), r, noisy,
                          np.ones(len(t), bool), k, np.arange(len(t)), 30, config())
    assert np.mean(np.linalg.norm(out.translations-t, axis=1)) < np.mean(np.linalg.norm(noisy-t, axis=1))*.4
    assert np.linalg.norm(out.translations[-1]-out.translations[0]) > .85*np.linalg.norm(t[-1]-t[0])
    assert out.diagnostics['static_constraint'] is False
    for value, before in zip([v, r, noisy, xy], inputs): np.testing.assert_array_equal(value, before)


def test_hidden_coordinates_are_unknown_and_pose_remains_complete():
    v, k, r, t, xy = fixture(); visible = np.ones(xy.shape[:2], bool)
    visible[5:8] = False; xy[~visible] = np.nan
    out = refine_sequence(v, v, xy, visible, r, t, np.ones(len(t), bool), k, np.arange(len(t)), 30, config())
    assert out.diagnostics['latent_frames'] == 3
    assert len(out.translations) == len(t) and np.isfinite(out.translations).all()


def test_canonical_origin_change_does_not_change_solution():
    v, k, r, t, xy = fixture(n=8); visible = np.ones(xy.shape[:2], bool)
    noisy = t.copy(); noisy[1:, 1] += .005*(-1.)**np.arange(1, len(t))
    a = refine_sequence(v, v, xy, visible, r, noisy, np.ones(len(t), bool), k, np.arange(len(t)), 30, config())
    shift = np.array([.4, -.3, .1]); ts = noisy-np.einsum('tij,j->ti', r, shift)
    b = refine_sequence(v+shift, v+shift, xy, visible, r, ts, np.ones(len(t), bool), k, np.arange(len(t)), 30, config())
    ca = np.einsum('tij,j->ti', a.rotations, v.mean(0))+a.translations
    cb = np.einsum('tij,j->ti', b.rotations, (v+shift).mean(0))+b.translations
    np.testing.assert_allclose(ca, cb, atol=1e-4)


def test_constant_angular_velocity_is_not_penalized_as_zero_velocity():
    v, k, r, t, xy = fixture()
    out = refine_sequence(v, v, xy, np.ones(xy.shape[:2], bool), r, t, np.ones(len(t), bool), k,
                          np.arange(len(t)), 30, config())
    np.testing.assert_allclose(out.rotations, r, atol=1e-6)
    np.testing.assert_allclose(out.translations, t, atol=1e-6)


def test_no_support_rejects_silent_zero_loss():
    v, k, r, t, xy = fixture(); flags = np.ones(xy.shape[:2], bool)
    flags[1:, 0] = False; xy[~flags] = np.nan
    with pytest.raises(ValueError, match='witness'): refine_sequence(v, v, xy, flags, r, t,
        np.ones(len(t), bool), k, np.arange(len(t)), 30, config())


def test_absence_is_not_finite_fake_observation():
    v, k, r, t, xy = fixture(); flags = np.ones(xy.shape[:2], bool); flags[3] = False
    with pytest.raises(ValueError, match='NaN'): refine_sequence(v, v, xy, flags, r, t,
        np.ones(len(t), bool), k, np.arange(len(t)), 30, config())
