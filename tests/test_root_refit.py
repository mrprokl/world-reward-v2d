"""Data-free D96 policy math, not native runtime or 3D validation evidence."""
import numpy as np
import pytest

from world_reward import root_refit as fit


def test_exact_native_training_mapping_and_reserved_wrists():
    assert fit.TRAIN_MHR == tuple(range(5, 15))
    assert fit.COCO_TO_MHR[9:11] == (62, 41)
    assert set(fit.TRAIN_COCO).isdisjoint(fit.HELDOUT_COCO)
    assert set(fit.TRAIN_COCO+fit.HELDOUT_COCO) == set(range(17))


def test_bounded_native_euler_and_positive_depth_no_rigid_composition():
    assert np.array_equal(fit.bounded_delta(np.zeros(6)), np.zeros(6))
    for u in (np.full(6, 1e6), np.full(6, -1e6), np.arange(6.)-3):
        d = fit.bounded_delta(u)
        assert np.all(np.abs(d[:2]) <= .30)
        assert np.linalg.norm(d[3:]) <= .30+1e-15
        t = fit.camera_translation(np.array([.1, -.2, 4.]), d)
        assert 3.2-1e-14 <= t[2] <= 5.+1e-14
    epsilon = 1e-6
    for i in range(6):
        u = np.zeros(6); u[i] = epsilon
        derivative = fit.bounded_delta(u)[i]/epsilon
        assert np.isfinite(derivative) and derivative > 0


@pytest.mark.parametrize("fault", ["negative_z", "overbound", "nan", "masked"])
def test_translation_bound_and_finite_guards(fault):
    origin = np.array([0., 0., 4.]); d = np.zeros(6)
    if fault == "negative_z": origin[2] = -1
    elif fault == "overbound": d[3] = .30
    elif fault == "nan": d[0] = np.nan
    else: d = np.ma.array(d, mask=False)
    with pytest.raises(ValueError): fit.camera_translation(origin, d)


def observations():
    xy = np.c_[np.arange(133.)*10, np.arange(133.)*20]
    return xy, np.ones(133)


def test_native_raw_scores_equal_binary_not_probabilities():
    xy, scores = observations(); scores[fit.TRAIN_COCO[0]] = 9.
    scores[fit.TRAIN_COCO[1]] = 0.; scores[fit.TRAIN_COCO[2]] = -3.
    before = scores.copy(); selected, indices, valid = fit.training_observations(xy, scores)
    assert len(selected) == 8 and valid.sum() == 8 and np.array_equal(scores, before)
    assert np.array_equal(selected, xy[list(fit.TRAIN_COCO)][valid])
    assert np.array_equal(indices, np.arange(5, 15)[valid])


@pytest.mark.parametrize("fault", ["five", "extent_x", "extent_y", "nan", "shape", "masked"])
def test_missing_or_degenerate_observations_fail_trial_not_frame_drop(fault):
    xy, scores = observations()
    if fault == "five": scores[list(fit.TRAIN_COCO[:5])] = 0
    elif fault == "extent_x": xy[:, 0] = 0
    elif fault == "extent_y": xy[:, 1] = 0
    elif fault == "nan": scores[0] = np.nan
    elif fault == "shape": xy = xy[:17]
    else: xy = np.ma.array(xy, mask=False)
    with pytest.raises(ValueError): fit.training_observations(xy, scores)


def test_huber_exact_mean_and_prior_not_landmark_count_scaled():
    xy = np.zeros((6, 2)); pred = xy.copy(); pred[:, 0] = 5
    result = fit.objective(pred, xy, np.zeros(6))
    assert result == {"data": .5, "prior": 0., "total": .5}
    pred[:, 0] = 25; delta = np.full(6, .015)
    result = fit.objective(pred, xy, delta)
    assert result["data"] == 4.5 and result["prior"] == pytest.approx(.03)
    assert fit.objective(np.zeros((10, 2)), np.zeros((10, 2)), delta)["prior"] == result["prior"]


def test_observation_only_svd_rank_no_prior_rescue():
    j = np.r_[np.eye(6), np.eye(6)]
    evidence = fit.jacobian_evidence(j, 6)
    assert evidence["minimum_maximum_ratio"] == 1 and evidence["prior_rows_included"] is False
    bad = j.copy(); bad[:, -1] = 0
    with pytest.raises(ValueError, match="Degenerate"): fit.jacobian_evidence(bad, 6)
    with pytest.raises(ValueError): fit.jacobian_evidence(np.r_[bad, np.eye(6)], 6)
    j[:, -1] *= 1e-6
    with pytest.raises(ValueError, match="Degenerate"): fit.jacobian_evidence(j, 6)


def test_select_only_sixty_evaluated_states_first_tie():
    loss = np.ones(60); loss[0] = 2.; loss[12] = loss[30] = .5
    assert fit.best_evaluated(loss) == 12
    assert fit.best_evaluated(np.ones(60)) == 0
    for invalid in (loss[:-1], np.r_[loss, 0.], np.full(60, np.nan), np.full(60, -1.)):
        with pytest.raises(ValueError): fit.best_evaluated(invalid)


def test_private_gates_require_all_three_clips_and_both_hands_no_alignment():
    h = np.array([[10., 9.], [20., 18.], [30., 29.]])
    hands = np.ones((3, 2, 2)); d = fit.quality_decision(h, hands)
    assert d["synthetic_root_refit_hypothesis_supported"] and d["adoption_authorized"] is False
    hands[1, 1, 0] = 1.06
    assert not fit.quality_decision(h, hands)["synthetic_root_refit_hypothesis_supported"]
    h[2, 1] = 32
    assert not fit.quality_decision(h, np.ones((3, 2, 2)))["synthetic_root_refit_hypothesis_supported"]
    with pytest.raises(ValueError): fit.quality_decision(h[:2], hands[:2])


def test_zero_baseline_human_undefined_object_exact_nonregression():
    human = np.array([[0., 0.], [1., .8], [1., .8]])
    hands = np.zeros((3, 2, 2)); d = fit.quality_decision(human, hands)
    assert d["zero_human_baseline_relative_gain_undefined"]
    assert d["per_clip_camera_pve_relative_gain"] is None
    assert not d["synthetic_root_refit_hypothesis_supported"]
    assert d["gates"]["no_per_hand_clip_relative_object_regression_over_5pct"]
    hands[0, 1, 1] = 1e-12
    assert not fit.quality_decision(human, hands)["gates"]["no_per_hand_clip_relative_object_regression_over_5pct"]


def test_finite_extreme_inputs_cannot_create_nonfinite_predictions_or_loss():
    d = fit.bounded_delta(np.full(6, 1e6))
    with pytest.raises(ValueError, match="overflow"):
        fit.camera_translation(np.array([0., 0., 1.7e308]), d)
    with pytest.raises(ValueError, match="Nonfinite objective"):
        fit.objective(np.full((6, 2), 1e308), np.zeros((6, 2)), np.zeros(6))
    with pytest.raises(ValueError, match="bounds"):
        fit.objective(np.zeros((6, 2)), np.zeros((6, 2)), np.full(6, .31))
    with pytest.raises(ValueError, match="relative-gain"):
        fit.quality_decision(np.array([[1e-308, 1e308], [1., 1.], [1., 1.]]), np.ones((3, 2, 2)))
    with pytest.raises(ValueError, match="relative-gain"):
        fit.quality_decision(np.ones((3, 2)), np.full((3, 2, 2), 1.79e308))


def test_masked_objective_targets_cannot_lose_hidden_validity():
    with pytest.raises(ValueError, match="Hidden observation"):
        fit.objective(np.zeros((6, 2)), np.ma.array(np.zeros((6, 2)), mask=True), np.zeros(6))
