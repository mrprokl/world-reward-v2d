import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward.rgb_pose_tracking import track_rgb_pose, vetted_tracks


class CVError(Exception):
    def __init__(self, code):
        self.code = code


def scene(planar=False):
    rgb = np.zeros((192, 256, 3), np.uint8)
    mask = np.zeros((192, 256), bool)
    mask[16:176, 24:232] = True
    pixels = np.array([(37 + 26*x, 26 + 20*y) for y in range(8) for x in range(8)], float)
    k = np.array([[300., 0, 128], [0, 300, 96], [0, 0, 1]])
    z = np.full(64, 3.) if planar else np.random.default_rng(9).uniform(2.5, 3.5, 64)
    points = np.column_stack(((pixels[:, 0]-128)*z/300, (pixels[:, 1]-96)*z/300, z))
    angle = .006
    r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    t = np.array([.006, -.004, .01])
    moved = points @ r.T + t
    q = (moved @ k.T)[:, :2] / moved[:, 2:]
    return [rgb, rgb.copy(), mask, mask.copy(), pixels, points, k], r, t, q


def fake_cv(monkeypatch, args, r, t, q):
    state = SimpleNamespace(calls=[], fit=None, refined=None, seed=None, inliers=None, error=None)
    def lk(a, b, p, initial, **kwargs):
        state.calls.append((p.copy(), kwargs))
        if state.error is not None:
            raise CVError(state.error)
        if len(state.calls) == 1:
            result = q
        else:
            ids = np.argmin(np.sum((p.reshape(-1, 1, 2) - q[None])**2, axis=2), axis=1)
            result = args[4][ids]
        return np.asarray(result, np.float32).reshape(-1, 1, 2), np.ones((len(p), 1), np.uint8), None
    def pnp(points, pixels, k, distortion, **kwargs):
        state.fit = (points.copy(), pixels.copy(), kwargs)
        inliers = np.arange(len(points)) if state.inliers is None else state.inliers
        return True, np.zeros((3, 1)), t.reshape(3, 1).copy(), np.asarray(inliers).reshape(-1, 1)
    def refine(points, pixels, k, distortion, rv, tv):
        state.refined = (points.copy(), pixels.copy())
        return rv, tv
    def seed(value):
        state.seed = value
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(error=CVError, COLOR_RGB2GRAY=7,
        SOLVEPNP_EPNP=1, cvtColor=lambda image, code: image[..., 0], calcOpticalFlowPyrLK=lk,
        setRNGSeed=seed, solvePnPRansac=pnp, solvePnPRefineLM=refine,
        Rodrigues=lambda rv: (r.copy(), None)))
    return state


def test_known_rigid_transform_and_no_holdout_fit(monkeypatch):
    args, r, t, q = scene()
    originals = [a.copy() for a in args]
    state = fake_cv(monkeypatch, args, r, t, q)
    result = track_rgb_pose(*args)
    assert result.status == "proposal"
    np.testing.assert_array_equal(result.rotation, r)
    np.testing.assert_array_equal(result.translation, t)
    assert not result.rotation.flags.writeable and not result.translation.flags.writeable
    assert len(result.fit_indices) == len(result.heldout_indices) == 32
    assert set(result.fit_indices).isdisjoint(result.heldout_indices)
    np.testing.assert_array_equal(state.fit[0], args[5][list(result.fit_indices)])
    np.testing.assert_array_equal(state.refined[0], args[5][list(result.inlier_indices)])
    assert state.fit[2] == dict(iterationsCount=200, reprojectionError=3., confidence=.99, flags=1)
    assert state.seed == 0 and result.heldout_median_px < 1e-4
    assert state.calls[0][1] == dict(winSize=(21, 21), maxLevel=3, criteria=(3, 30, .01))
    for a, b in zip(args, originals):
        np.testing.assert_array_equal(a, b)


def test_lm_only_ransac_inliers(monkeypatch):
    args, r, t, q = scene()
    state = fake_cv(monkeypatch, args, r, t, q)
    state.inliers = np.arange(24)
    result = track_rgb_pose(*args)
    assert result.status == "proposal"
    assert len(result.inlier_indices) == 24
    np.testing.assert_array_equal(state.refined[0], state.fit[0][:24])


def test_reserved_disagreement_abstains_without_static_fallback(monkeypatch):
    args, r, t, q = scene()
    cells = np.floor((args[4] - [24, 16]) / [208, 160] * 8).astype(int)
    q[cells.sum(1) % 2 == 1, 0] += 8
    fake_cv(monkeypatch, args, r, t, q)
    result = track_rgb_pose(*args)
    assert result.reason == "reprojection_disagreement"
    assert result.heldout_median_px > 7
    assert result.rotation is result.translation is None


def test_fb_masks_borders_and_failed_nan_tracks():
    a = np.array([[4, 4], [5, 4], [6, 4], [0, 4], [7, 4], [8, 4]], float)
    q, back = a.copy(), a.copy()
    back[0, 0] += 1
    back[1, 0] += 1.001
    q[4] = np.nan
    mask0 = np.ones((12, 12), bool)
    mask1 = mask0.copy()
    mask1[4, 6] = False
    mask0[4, 8] = False
    ok = vetted_tracks(a, q, back, np.ones(6, bool), np.ones(6, bool), mask0, mask1)
    np.testing.assert_array_equal(ok, [True, False, False, False, False, False])
    assert not vetted_tracks(a[:1], q[:1], back[:1], np.zeros(1, bool),
                             np.ones(1, bool), mask0, mask1).any()


def test_sparse_masks_abstain_before_optional_import(monkeypatch):
    args, *_ = scene()
    args[2][:] = False
    monkeypatch.setitem(sys.modules, "cv2", None)
    result = track_rgb_pose(*args)
    assert result.reason == "insufficient_anchor_support"
    assert result.rotation is None


def test_missing_backend_is_error_not_prediction(monkeypatch):
    args, *_ = scene()
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ModuleNotFoundError):
        track_rgb_pose(*args)


@pytest.mark.parametrize("slot,change", [
    (0, lambda a: a.astype(float)), (2, lambda a: a.astype(np.uint8)),
    (4, lambda a: a.astype(np.int64)), (4, lambda a: a + .5),
    (4, lambda a: np.full_like(a, 30)), (5, lambda a: a * [1, 1, -1]),
    (5, lambda a: a + [1, 0, 0]), (5, lambda a: a * np.nan),
    (6, lambda a: a.astype(np.int64)), (6, lambda a: a * 0),
    (0, lambda a: np.ma.array(a, mask=False)),
])
def test_invalid_input_contracts(slot, change):
    args, *_ = scene()
    args[slot] = change(args[slot])
    with pytest.raises(ValueError):
        track_rgb_pose(*args)


def test_inlier_support_can_abstain(monkeypatch):
    args, r, t, q = scene()
    state = fake_cv(monkeypatch, args, r, t, q)
    state.inliers = np.arange(15)
    result = track_rgb_pose(*args)
    assert result.reason == "weak_inlier_support" and state.refined is None


def test_planar_is_only_numerically_supported_not_ambiguity_proof(monkeypatch):
    args, r, t, q = scene(planar=True)
    fake_cv(monkeypatch, args, r, t, q)
    result = track_rgb_pose(*args)
    assert result.status == "proposal" and result.planar_support
    assert "not_accuracy" in result.reason


@pytest.mark.parametrize("code", [-7, -215])
def test_only_numerical_no_convergence_is_abstention(monkeypatch, code):
    args, r, t, q = scene()
    state = fake_cv(monkeypatch, args, r, t, q)
    state.error = code
    if code == -7:
        assert track_rgb_pose(*args).reason == "opencv_no_convergence"
    else:
        with pytest.raises(CVError):
            track_rgb_pose(*args)


def test_improper_rotation_and_backend_bad_indices_raise(monkeypatch):
    args, r, t, q = scene()
    fake_cv(monkeypatch, args, np.diag([-1., 1, 1]), t, q)
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        track_rgb_pose(*args)
    state = fake_cv(monkeypatch, args, r, t, q)
    state.inliers = [0] * 16
    with pytest.raises(ValueError, match="inlier indices"):
        track_rgb_pose(*args)


def test_current_negative_depth_abstains(monkeypatch):
    args, r, t, q = scene()
    fake_cv(monkeypatch, args, r, np.array([0., 0, -10]), q)
    assert track_rgb_pose(*args).reason == "nonpositive_current_depth"
