"""Synthetic external errors only; no real annotation, tracker or runtime call."""
from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from world_reward.tracker_noise_null import (
    TrackerErrorClip, evaluate_tracker_noise_null, fit_tracker_noise_null,
)


def clip(key, seed=82, frames=18, indices=(0, 1, 2, 3)):
    rng = np.random.default_rng(seed)
    errors = rng.normal(size=(len(indices), frames, 2))
    shared = rng.normal(size=(frames, 2))
    errors += shared[None]
    return TrackerErrorClip(key, np.arange(frames, dtype=np.int64), np.array(indices, np.int64),
                            np.zeros(len(indices), np.int64), errors, np.ones(errors.shape[:2], bool))


def independent_samples(value, lag=0):
    result = []
    t = len(value.frame_index) - 1
    for i, original_i in enumerate(value.point_indices):
        if original_i % 2:
            continue
        for j, original_j in enumerate(value.point_indices):
            if not original_j % 2:
                continue
            samples = []
            for frame in range(t):
                other = frame + lag
                if (other < t and frame > value.query_frame_index[i]
                        and other > value.query_frame_index[j]
                        and value.observed[i, frame:frame + 2].all()
                        and value.observed[j, other:other + 2].all()):
                    samples.append(np.r_[value.errors[i, frame + 1] - value.errors[i, frame],
                                         value.errors[j, other + 1] - value.errors[j, other]])
            result.append((int(original_i), int(original_j), np.array(samples).reshape(-1, 4)))
    return result


def test_weighted_mle_matches_independent_full_sample_covariance_and_shared_marginals():
    values = (clip("fit-a"), clip("fit-b", 32, frames=14))
    model = fit_tracker_noise_null(values)
    samples, weights = [], []
    for value in values:
        pairs = [rows for _, _, rows in independent_samples(value) if len(rows)]
        for rows in pairs:
            for row in rows:
                samples.append(row); weights.append(1 / (len(values) * len(pairs) * len(rows)))
    mean = sum(w * row for w, row in zip(weights, samples)) / sum(weights)
    covariance = sum(w * np.outer(row - mean, row - mean) for w, row in zip(weights, samples)) / sum(weights)
    np.testing.assert_allclose(model.mean, mean, atol=2e-15)
    np.testing.assert_allclose(model.covariance, covariance, atol=2e-14)
    np.testing.assert_array_equal(model.independent_covariance[:2, :2], model.covariance[:2, :2])
    np.testing.assert_array_equal(model.independent_covariance[2:, 2:], model.covariance[2:, 2:])
    np.testing.assert_array_equal(model.independent_covariance[:2, 2:], 0)


def test_heldout_log_density_gain_matches_inverse_determinant_reference():
    model = fit_tracker_noise_null((clip("fit"),))
    heldout = clip("heldout", 144)
    report = evaluate_tracker_noise_null(model, heldout)
    for name, lag in (("synchronous", 0), ("shifted", len(heldout.frame_index) // 3)):
        means, total, count = [], 0., 0
        for _, _, rows in independent_samples(heldout, lag):
            gains = []
            for row in rows:
                delta = row - model.mean
                def log_density(covariance):
                    return -.5 * (4 * math.log(2 * math.pi) + math.log(np.linalg.det(covariance))
                                  + delta @ np.linalg.inv(covariance) @ delta)
                gains.append(log_density(model.covariance) - log_density(model.independent_covariance))
            means.append(np.mean(gains)); total += sum(gains); count += len(gains)
        assert report[name]["mean_log_likelihood_gain"] == pytest.approx(np.mean(means), abs=2e-14)
        assert report[name]["sample_weighted_mean_gain"] == pytest.approx(total / count, abs=2e-14)
        assert report[name]["complete_samples"] == count


def test_strict_after_query_and_full_missing_denominators_no_initializer_evidence():
    model = fit_tracker_noise_null((clip("fit"),))
    original = clip("heldout", frames=12, indices=(2, 5))
    value = replace(original, query_frame_index=np.array([3, 6], np.int64))
    report = evaluate_tracker_noise_null(model, value)
    assert report["synchronous"]["complete_samples"] == 4  # interval7,8,9,10, not query6.
    assert report["synchronous"]["possible_pair_intervals"] == 11
    assert report["synchronous"]["missing_samples"] == 7
    assert report["shifted"]["lag_original_frames"] == 4
    assert report["shifted"]["out_of_range_shift_samples"] == 4
    assert report["original_point_indices"] == [2, 5]


def test_missing_support_never_zero_imputed_and_empty_pair_stays_in_report():
    model = fit_tracker_noise_null((clip("fit"),))
    value = clip("heldout")
    support = value.observed.copy(); support[1] = False
    errors = value.errors.copy(); errors[1] = np.nan
    value = replace(value, errors=errors, observed=support)
    report = evaluate_tracker_noise_null(model, value)
    assert report["synchronous"]["cross_group_pairs"] == 4
    assert report["synchronous"]["supported_pairs"] == 2
    assert [r["complete_samples"] for r in report["synchronous"]["pairs"]] == [0, 16, 0, 16]
    assert report["synchronous"]["missing_samples"] == 4 * 17 - 32
    none = replace(value, errors=np.full(value.errors.shape, np.nan), observed=np.zeros(value.observed.shape, bool))
    missing = evaluate_tracker_noise_null(model, none)
    assert missing["synchronous"]["mean_log_likelihood_gain"] is None
    assert missing["synchronous"]["missing_samples"] == missing["synchronous"]["possible_pair_intervals"]


def test_source_index_parity_not_row_order_and_coordinate_permutation_preserves_gain():
    fit, heldout = clip("fit"), clip("heldout", 321)
    model = fit_tracker_noise_null((fit,))
    expected = evaluate_tracker_noise_null(model, heldout)
    permuted = replace(fit, errors=fit.errors[..., ::-1])
    test = replace(heldout, errors=heldout.errors[..., ::-1])
    actual = evaluate_tracker_noise_null(fit_tracker_noise_null((permuted,)), test)
    assert actual["synchronous"]["mean_log_likelihood_gain"] == pytest.approx(expected["synchronous"]["mean_log_likelihood_gain"], abs=2e-14)
    # Dropping point0 leaves row0 originallyodd and must NOT redefine group parity.
    only = replace(heldout, point_indices=np.array([1, 2, 3], np.int64),
                   query_frame_index=heldout.query_frame_index[1:], errors=heldout.errors[1:], observed=heldout.observed[1:])
    rows = evaluate_tracker_noise_null(model, only)["synchronous"]["pairs"]
    assert [(r["even_original_index"], r["odd_original_index"]) for r in rows] == [(2, 1), (2, 3)]


@pytest.mark.parametrize("scale", [.125, 8.])
def test_shared_pixel_units_scale_covariance_and_cancel_density_jacobian(scale):
    fit, heldout = clip("fit"), clip("heldout", 121)
    expected = evaluate_tracker_noise_null(fit_tracker_noise_null((fit,)), heldout)
    scaled_fit, scaled_test = replace(fit, errors=fit.errors * scale), replace(heldout, errors=heldout.errors * scale)
    actual = evaluate_tracker_noise_null(fit_tracker_noise_null((scaled_fit,)), scaled_test)
    for mode in ("synchronous", "shifted"):
        assert actual[mode]["mean_log_likelihood_gain"] == pytest.approx(expected[mode]["mean_log_likelihood_gain"], abs=3e-14)


def test_rank_deficient_insufficient_and_no_cross_group_fit_fail_without_jitter():
    zero = clip("fit", frames=8)
    with pytest.raises(ValueError, match="positive definite"):
        fit_tracker_noise_null((replace(zero, errors=np.zeros(zero.errors.shape)),))
    with pytest.raises(ValueError, match="More than four"):
        fit_tracker_noise_null((clip("tiny", frames=3, indices=(0, 1)),))
    with pytest.raises(ValueError, match="complete cross-group"):
        fit_tracker_noise_null((clip("even", indices=(0, 2)),))


def test_fit_clip_not_reused_as_heldout_and_immutable_outputs_noalias():
    value = clip("fit")
    model = fit_tracker_noise_null((value,))
    with pytest.raises(ValueError, match="Heldout"):
        evaluate_tracker_noise_null(model, value)
    for array in (value.errors, value.observed, value.point_indices, model.mean, model.covariance):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        model.fit_clip_keys = ("replacement",)


def test_clip_order_changes_no_weight_and_missing_queries_are_not_fabricated():
    a, b = clip("fit-a"), clip("fit-b", 82, frames=14)
    first, second = fit_tracker_noise_null((a, b)), fit_tracker_noise_null((b, a))
    np.testing.assert_allclose(first.mean, second.mean, atol=2e-15)
    np.testing.assert_allclose(first.covariance, second.covariance, atol=2e-14)
    empty = TrackerErrorClip("no-queries", np.arange(12, dtype=np.int64), np.zeros(0, np.int64),
                             np.zeros(0, np.int64), np.empty((0, 12, 2)), np.empty((0, 12), bool))
    record = evaluate_tracker_noise_null(first, empty)
    assert record["synchronous"]["cross_group_pairs"] == 0
    assert record["synchronous"]["mean_log_likelihood_gain"] is None
    assert record["original_point_indices"] == []


def test_matched_anchor_shift_uses_three_jointly_supported_endpoints_and_same_pairs():
    fit, heldout = clip("fit"), clip("heldout", 330, frames=12)
    support = heldout.observed.copy()
    support[1, 7] = False; support[3, 5] = False
    heldout = replace(heldout, observed=support)
    model = fit_tracker_noise_null((fit,))
    result = evaluate_tracker_noise_null(model, heldout)["matched_anchor_shift"]
    lag, steps = 4, 11
    means, all_deltas = [], []
    expected_counts = []
    for i in (0, 2):
        for j in (1, 3):
            deltas = []
            for t in range(steps):
                u = t + lag
                if (u >= steps or t <= heldout.query_frame_index[i] or t <= heldout.query_frame_index[j]
                        or u <= heldout.query_frame_index[j]
                        or not heldout.observed[i, t:t + 2].all()
                        or not heldout.observed[j, t:t + 2].all()
                        or not heldout.observed[j, u:u + 2].all()):
                    continue
                even = heldout.errors[i, t + 1] - heldout.errors[i, t]
                def gain(frame):
                    value = np.r_[even, heldout.errors[j, frame + 1] - heldout.errors[j, frame]]
                    centered = value - model.mean
                    def density(covariance):
                        return -.5 * (math.log(np.linalg.det(covariance))
                                      + centered @ np.linalg.inv(covariance) @ centered)
                    return density(model.covariance) - density(model.independent_covariance)
                deltas.append(gain(t) - gain(u))
            expected_counts.append(len(deltas))
            if deltas:
                means.append(np.mean(deltas)); all_deltas.extend(deltas)
    assert [row["complete_samples"] for row in result["pairs"]] == expected_counts
    assert result["complete_samples"] == sum(expected_counts)
    assert result["possible_pair_intervals"] == 4 * steps
    assert result["missing_samples"] == 4 * steps - sum(expected_counts)
    assert result["out_of_range_shift_samples"] == 4 * lag
    assert result["mean_log_likelihood_delta"] == pytest.approx(np.mean(means), abs=3e-14)
    assert result["sample_weighted_mean_delta"] == pytest.approx(np.mean(all_deltas), abs=3e-14)
    assert result["descriptive_only"]


def test_matched_anchors_missing_at_synchronous_or_shifted_context_are_not_refilled():
    model = fit_tracker_noise_null((clip("fit"),))
    heldout = clip("heldout", frames=12, indices=(0, 1))
    support = np.zeros(heldout.observed.shape, bool)
    support[0] = True
    support[1, 1:4] = True  # Sync available; no shifted context can coincide.
    heldout = replace(heldout, observed=support)
    result = evaluate_tracker_noise_null(model, heldout)
    assert result["synchronous"]["complete_samples"] > 0
    matched = result["matched_anchor_shift"]
    assert matched["complete_samples"] == 0
    assert matched["mean_log_likelihood_delta"] is None
    assert matched["sample_weighted_mean_delta"] is None
    assert matched["missing_samples"] == matched["possible_pair_intervals"]


@pytest.mark.parametrize("field,value", [
    ("frame_index", np.arange(18, dtype=np.int32)),
    ("point_indices", np.array([0, 0, 2, 3], np.int64)),
    ("point_indices", np.array([0, 1, 2, 32], np.int64)),
    ("query_frame_index", np.array([0, 1, 2, 18], np.int64)),
    ("observed", np.ones((4, 18), np.int64)),
    ("errors", np.full((4, 18, 2), np.nan)),
])
def test_invalid_timeline_original_indices_support_or_observed_numeric_fail(field, value):
    with pytest.raises(ValueError):
        replace(clip("invalid"), **{field: value})
