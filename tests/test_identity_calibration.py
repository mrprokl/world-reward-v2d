"""Data-free private matching/logistic math, not learned association quality."""
import ast
from pathlib import Path

import numpy as np
import pytest

from world_reward import identity_calibration as identity


def fit_data():
    x = np.array([[-2., 5., .1], [2., 5., .4], [-1., 5., .2], [1., 5., .3],
                  [-3., 5., .2], [3., 5., .3]])
    return dict(features=x, supported=np.ones(6, dtype=bool), labels=np.array([0, 1] * 3),
                clip_ids=("a", "a", "b", "b", "c", "c"),
                identity_ids=("wrong", "target") * 3)


def test_target_id_is_class_at_grasp_index_not_index_or_contact_truth():
    assert identity.target_object_id([12, 4, 19], 1) == 4
    assert identity.target_object_id(np.array([21, 1], dtype=np.uint8), np.int64(0)) == 21
    for ids, index in (([1, 1], 0), ([0, 1], 1), ([22], 0), ([1.], 0), ([True], 0),
                       ([1], True), ([1], -1), ([1], 1), (np.ma.array([1]), 0)):
        with pytest.raises(ValueError):
            identity.target_object_id(ids, index)


def test_majority_identity_is_distinct_from_reciprocal_coverage_and_hand_truth():
    seg = np.array([[12, 12, 12, 12], [12, 12, 4, 4], [255, 255, 0, 0]], dtype=np.uint8)
    fragment = np.zeros(seg.shape, dtype=bool); fragment[0, :2] = True
    result = identity.majority_identity(fragment, seg)
    assert result.object_id == 12 and result.majority_fraction == 1
    assert result.target_recall == 2 / 6 and not result.reciprocal_coverage
    full = seg == 12
    assert identity.majority_identity(full, seg).reciprocal_coverage
    hand = identity.majority_identity(seg == 255, seg)
    assert hand.object_id is None and hand.target_recall == 0
    assert identity.majority_identity(seg == 0, seg).object_id is None
    assert identity.majority_identity(np.zeros(seg.shape, dtype=bool), seg).mask_pixels == 0


def test_exact_half_is_not_a_majority_for_identity_or_coverage():
    seg = np.array([[3, 3, 7, 7]], dtype=np.int32)
    assert identity.majority_identity(np.ones(seg.shape, dtype=bool), seg).object_id is None
    partial = np.array([[True, False, False, False]])
    result = identity.majority_identity(partial, seg)
    assert result.object_id == 3 and result.target_recall == .5
    assert not result.reciprocal_coverage


@pytest.mark.parametrize("seg,mask", [
    (np.zeros((2, 2), dtype=float), np.ones((2, 2), dtype=bool)),
    (np.full((2, 2), 22), np.ones((2, 2), dtype=bool)),
    (np.full((2, 2), -1), np.ones((2, 2), dtype=bool)),
    (np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)),
    (np.ma.array(np.zeros((2, 2), dtype=int)), np.ones((2, 2), dtype=bool)),
    (np.zeros((2, 2), dtype=int), np.ones((2, 2), dtype=np.uint8)),
    (np.zeros((2, 2), dtype=int), np.ma.array(np.ones((2, 2), dtype=bool))),
    (np.zeros((2, 2), dtype=int), np.ones((2, 3), dtype=bool)),
])
def test_matching_rejects_non_native_labels_and_non_boolean_masks(seg, mask):
    with pytest.raises(ValueError):
        identity.majority_identity(mask, seg)


def test_actual_logistic_optimum_matches_independent_standard_objective():
    from scipy.optimize import minimize
    data = fit_data()
    model = identity.fit_identity_logistic(**data)
    z = np.column_stack((np.ones(6), (data["features"] - model.mean) / model.scale))
    y, weight = data["labels"], model.sample_weights
    def objective(theta):
        score = z @ theta
        return weight @ (np.logaddexp(0, score) - y * score) + .5 * np.sum(theta[1:] ** 2)
    reference = minimize(objective, np.zeros(4), method="BFGS", options={"gtol": 1e-8})
    fitted = np.r_[model.intercept, model.coefficients]
    np.testing.assert_allclose(fitted, reference.x, atol=2e-7, rtol=2e-7)
    assert model.l2 == 1 and model.fit_clips == 3 and model.gradient_inf_norm <= 1e-9
    assert model.iterations < 100 and model.objective == pytest.approx(objective(fitted))
    np.testing.assert_allclose(model.sample_weights, np.full(6, 1/6))
    assert model.constant_features.tolist() == [False, True, False]
    assert model.scale[1] == 1 and abs(model.coefficients[1]) < 1e-12


def test_raw_scores_are_not_probabilities_and_fit_only_standardization_is_immutable():
    data = fit_data(); model = identity.fit_identity_logistic(**data)
    mean, scale = model.mean.copy(), model.scale.copy()
    points = np.array([[100., 5., .4], [np.nan, np.inf, np.nan]])
    scores = identity.score_identity_candidates(model, points, np.array([True, False]))
    assert scores[0] > 1 and np.isnan(scores[1])
    np.testing.assert_array_equal(model.mean, mean)
    np.testing.assert_array_equal(model.scale, scale)
    for value in (model.mean, model.scale, model.constant_features, model.coefficients,
                  model.sample_weights, scores):
        assert value.flags.owndata and not value.flags.writeable
    data["features"][:] = 999
    np.testing.assert_array_equal(model.mean, mean)
    with pytest.raises(ValueError):
        identity.score_identity_candidates(model, points, np.array([True, True]))


def test_exact_duplicate_variants_and_input_permutation_do_not_increase_positive_mass():
    data = fit_data(); original = identity.fit_identity_logistic(**data)
    indices = np.array([5, 1, 0, 3, 2, 4, 1, 1])
    copied = dict(features=data["features"][indices], supported=data["supported"][indices],
                  labels=data["labels"][indices], clip_ids=tuple(data["clip_ids"][i] for i in indices),
                  identity_ids=tuple(data["identity_ids"][i] for i in indices))
    model = identity.fit_identity_logistic(**copied)
    np.testing.assert_allclose(model.mean, original.mean, atol=1e-15)
    np.testing.assert_allclose(model.scale, original.scale, atol=1e-15)
    np.testing.assert_allclose(model.coefficients, original.coefficients, atol=1e-15)
    assert model.sample_weights[indices == 1].sum() == pytest.approx(1/6)
    assert model.sample_weights[np.array(copied["clip_ids"]) == "a"].sum() == pytest.approx(1/3)


def test_clips_then_identities_have_equal_mass_not_frame_or_mask_count():
    data = fit_data()
    # One extra distinct wrong-object variant in a; that identity still gets half a's mass.
    data["features"] = np.vstack((data["features"], [-2.5, 5., .15]))
    data["supported"] = np.ones(7, dtype=bool)
    data["labels"] = np.r_[data["labels"], 0]
    data["clip_ids"] += ("a",); data["identity_ids"] += ("wrong",)
    model = identity.fit_identity_logistic(**data)
    assert model.sample_weights[0] == pytest.approx(1/12)
    assert model.sample_weights[-1] == pytest.approx(1/12)
    assert model.sample_weights[1] == pytest.approx(1/6)
    assert model.sample_weights.sum() == pytest.approx(1)


def test_unknown_or_unsupported_are_excluded_only_from_fit_not_made_finite():
    data = fit_data()
    data["features"] = np.vstack((data["features"], [np.nan] * 3, [999.] * 3))
    data["supported"] = np.r_[data["supported"], False, True]
    data["labels"] = np.r_[data["labels"], 1, -1]
    data["clip_ids"] += ("a", "a"); data["identity_ids"] += ("missing", "mixed")
    model = identity.fit_identity_logistic(**data)
    assert model.sample_weights[-2:].tolist() == [0., 0.]
    base = identity.fit_identity_logistic(**fit_data())
    np.testing.assert_allclose(model.mean, base.mean)
    data["clip_ids"] = data["clip_ids"][:-1] + ("empty-clip",)
    with pytest.raises(identity.InsufficientCalibrationError):
        identity.fit_identity_logistic(**data)


@pytest.mark.parametrize("update", [
    {"labels": np.zeros(6, dtype=int)}, {"labels": np.array([0, 0, 0, 1, 0, 1])},
    {"supported": np.zeros(6, dtype=bool)}, {"labels": np.zeros(6)},
    {"labels": np.array([0, 2, 0, 1, 0, 1])}, {"labels": np.ma.array([0, 1] * 3)},
    {"features": np.full((6, 3), np.nan)}, {"features": np.zeros((6, 3), dtype=int)},
    {"features": np.ma.array(np.zeros((6, 3)))}, {"supported": np.ones(6)},
    {"clip_ids": ["a"] * 6}, {"identity_ids": ("same",) * 6},
    {"identity_ids": ("bad\x7f", "good") * 3},
])
def test_insufficient_or_invalid_calibration_fails_without_model(update):
    data = fit_data(); data.update(update)
    with pytest.raises(ValueError):
        identity.fit_identity_logistic(**data)


def test_constant_only_features_keep_binary_base_rate_without_invented_discrimination():
    data = fit_data(); data["features"][:] = 7.
    model = identity.fit_identity_logistic(**data)
    np.testing.assert_array_equal(model.coefficients, np.zeros(3))
    assert model.intercept == 0
    scores = identity.score_identity_candidates(model, data["features"][:2], np.ones(2, dtype=bool))
    decision = identity.decide_identity(scores, np.ones(2, dtype=bool), ("one", "two"), minimum_gap=0)
    assert decision.proposal_id is None


def test_numeric_overflow_and_solver_failure_raise_instead_of_returning_partial_model(monkeypatch):
    data = fit_data()
    with np.errstate(over="ignore"):
        data["features"][:, 0] *= 1e308
    with pytest.raises(ValueError):
        identity.fit_identity_logistic(**data)
    def failed(*args):
        raise np.linalg.LinAlgError("test only")
    monkeypatch.setattr(identity.np.linalg, "solve", failed)
    with pytest.raises(ValueError, match="numerically invalid"):
        identity.fit_identity_logistic(**fit_data())


def test_strict_positive_raw_winner_gap_zero_is_explicit_not_confidence():
    score, support = np.array([2., -1.]), np.ones(2, dtype=bool)
    result = identity.decide_identity(score, support, ("proposal-99", "proposal-3"), minimum_gap=0)
    assert result.proposal_id == "proposal-99" and result.reason == "STRICT_WINNER"
    assert result.raw_best_score == 2 and result.raw_score_gap == 3
    assert not hasattr(result, "probability") and not hasattr(result, "object_id")
    assert identity.decide_identity(score, support, ("a", "b"), minimum_gap=3).proposal_id is None


def test_duplicate_group_scores_do_not_win_by_replication_and_ties_abstain():
    values = np.array([3., 1., 3., 2.])
    result = identity.decide_identity(values, np.ones(4, dtype=bool), ("a", "a", "a", "b"), minimum_gap=0)
    assert result.ranked_scores == (("a", 2.), ("b", 2.))
    assert result.proposal_id is None and result.raw_score_gap == 0
    swapped = identity.decide_identity(values[::-1], np.ones(4, dtype=bool), ("b", "a", "a", "a"), minimum_gap=0)
    assert swapped == result


def test_missing_competitor_and_single_bank_abstain_without_fabricated_score():
    result = identity.decide_identity(np.array([10., np.nan]), np.array([True, False]), ("a", "b"), minimum_gap=0)
    assert result.reason == "UNSUPPORTED_COMPETITOR" and result.proposal_id is None
    assert result.raw_best_score is None
    assert identity.decide_identity(np.array([10.]), np.ones(1, dtype=bool), ("a",), minimum_gap=0).reason == "NO_ALTERNATIVE"
    assert identity.decide_identity(np.array([-1., -2.]), np.ones(2, dtype=bool), ("a", "b"), minimum_gap=0).reason == "NONPOSITIVE_SCORE"
    assert identity.decide_identity(np.array([np.nan]), np.zeros(1, dtype=bool), ("a",), minimum_gap=0).reason == "NO_EVIDENCE"


@pytest.mark.parametrize("gap", [-1, True, np.nan, np.inf, "0", 1j])
def test_decision_requires_frozen_explicit_finite_nonnegative_gap(gap):
    with pytest.raises(ValueError):
        identity.decide_identity(np.array([2., 1.]), np.ones(2, dtype=bool), ("a", "b"), minimum_gap=gap)


def test_full_timeline_preserves_missing_wrong_mixed_fragment_and_occluded_failures():
    seg = np.array([[4, 4, 4, 4], [7, 7, 255, 0]], dtype=np.uint8)
    fragment = np.zeros(seg.shape, dtype=bool); fragment[0, 0] = True
    invisible = seg.copy(); invisible[invisible == 4] = 0
    selections = (seg == 4, fragment, seg == 7, None, seg == 255, np.zeros(seg.shape, dtype=bool))
    result = identity.evaluate_identity_timeline(np.arange(6, dtype=np.int64), selections,
                                                 (seg,) * 5 + (invisible,), 4)
    assert result.correct_identity.tolist() == [True, True, False, False, False, False]
    assert result.reciprocal_coverage.tolist() == [True, False, False, False, False, False]
    assert result.wrong_id.tolist() == [False, False, True, False, False, False]
    assert result.unknown_mask.tolist() == [False, False, False, False, True, False]
    assert result.missing.tolist() == [False, False, False, True, False, True]
    assert result.target_visible.tolist() == [True] * 5 + [False]
    assert result.longest_failure_gap == 5
    categories = np.array([result.correct_identity, result.wrong_id, result.unknown_mask, result.missing])
    np.testing.assert_array_equal(categories.sum(0), np.ones(6))
    for value in vars(result).values():
        if isinstance(value, np.ndarray):
            assert value.flags.owndata and not value.flags.writeable


def test_macro_clip_weights_not_long_clip_or_successful_only_denominator():
    seg = np.array([[1, 2]], dtype=np.uint8)
    a = identity.evaluate_identity_timeline(np.arange(1, dtype=np.int64), (seg == 1,), (seg,), 1)
    b = identity.evaluate_identity_timeline(np.arange(9, dtype=np.int64), (None,) * 9, (seg,) * 9, 1)
    metrics = identity.macro_identity_metrics((a, b))
    assert metrics["correct_identity_rate"] == .5 and metrics["reciprocal_coverage_rate"] == .5
    assert metrics["missing_rate"] == .5 and metrics["frames"] == 10 and metrics["clips"] == 2
    assert metrics["max_failure_gap"] == 9


@pytest.mark.parametrize("indices", [np.arange(2), np.array([0, 2], dtype=np.int64),
                                      np.array([1, 2], dtype=np.int64), np.ma.array(np.arange(2, dtype=np.int64))])
def test_eval_rejects_sparse_or_non_int64_positions(indices):
    if isinstance(indices, np.ndarray) and indices.dtype == np.int64 and np.array_equal(indices, [0, 1]) and not np.ma.isMaskedArray(indices):
        indices = indices.astype(np.int32)
    seg = np.array([[1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        identity.evaluate_identity_timeline(indices, (None,) * 2, (seg,) * 2, 1)


def test_eval_requires_all_frames_same_grid_and_no_selected_private_best_mask():
    seg = np.array([[1]], dtype=np.uint8)
    with pytest.raises(ValueError):
        identity.evaluate_identity_timeline(np.arange(2, dtype=np.int64), (None,), (seg, seg), 1)
    with pytest.raises(ValueError):
        identity.evaluate_identity_timeline(np.arange(2, dtype=np.int64), (None, None), (seg, np.zeros((2, 2), dtype=int)), 1)


def test_source_is_only_pure_numpy_and_does_not_read_labels_or_train_track_models():
    tree = ast.parse(Path(identity.__file__).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imports <= {"__future__", "dataclasses", "numpy"}
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"load", "save", "read_bytes", "read_text", "open", "urlopen"} & calls
