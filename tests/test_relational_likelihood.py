"""Independent tiny Gaussian/HMM references; no observations, fit or calibration."""
from dataclasses import FrozenInstanceError
from itertools import product
import math

import numpy as np
import pytest

from world_reward.relational_likelihood import conditional_relational_likelihood


def inputs(nh=1, no=1, steps=3):
    rng = np.random.default_rng(804)
    return dict(frame_index=np.arange(steps + 1, dtype=np.int64),
                hand_keys=tuple(f"hand-{i}" for i in range(nh)),
                object_keys=tuple(f"object-{i}" for i in range(no)),
                hand_innovations=rng.normal(size=(nh, steps, 2)),
                object_innovations=rng.normal(size=(no, steps, 2)),
                hand_observed=np.ones((nh, steps, 2), bool),
                object_observed=np.ones((no, steps, 2), bool),
                hand_mean=np.zeros((nh, steps, 2)), object_mean=np.zeros((no, steps, 2)),
                hand_covariance=np.tile(np.diag([1.3, .8]), (nh, 1, 1)),
                object_covariance=np.tile(np.diag([.9, 1.1]), (no, 1, 1)),
                cross_covariance=np.tile(np.array([[.25, -.08], [.04, .2]]), (nh, no, 1, 1)),
                regime_initial=np.array([.3, .7]), regime_transition=np.array([[.8, .2], [.15, .85]]),
                hypothesis_prior=np.full(1 + nh * no, 1 / (1 + nh * no)),
                common_log_factors=np.zeros(steps))


def density(value, mean, covariance, observed):
    selected = np.flatnonzero(observed)
    if not len(selected):
        return 1.
    delta = value[selected] - mean[selected]
    matrix = covariance[np.ix_(selected, selected)]
    return math.exp(-.5 * (delta @ np.linalg.inv(matrix) @ delta)) / math.sqrt(
        (2 * math.pi) ** len(selected) * np.linalg.det(matrix))


def brute_pair(args, i, j):
    steps = len(args["frame_index"]) - 1
    hc, oc = args["hand_covariance"][i], args["object_covariance"][j]
    cross = args["cross_covariance"][i, j]
    covariance = np.block([[hc, cross], [cross.T, oc]])
    emissions = []
    for t in range(steps):
        h, o = args["hand_innovations"][i, t], args["object_innovations"][j, t]
        hm, om = args["hand_mean"][i, t], args["object_mean"][j, t]
        hs, os = args["hand_observed"][i, t], args["object_observed"][j, t]
        joint = density(np.r_[h, o], np.r_[hm, om], covariance, np.r_[hs, os])
        independent = density(h, hm, hc, hs) * density(o, om, oc, os)
        emissions.append((1., joint / independent))
    if not steps:
        return 0.
    total = 0.
    for path in product((0, 1), repeat=steps):
        probability = args["regime_initial"][path[0]]
        for t in range(1, steps):
            probability *= args["regime_transition"][path[t - 1], path[t]]
        for t, state in enumerate(path):
            probability *= emissions[t][state]
        total += probability
    return math.log(total)


@pytest.mark.parametrize("steps", [0, 1, 2, 4])
def test_forward_marginal_likelihood_matches_enumerated_paths_not_map(steps):
    args = inputs(2, 2, steps)
    result = conditional_relational_likelihood(**args)
    for i, j in product(range(2), repeat=2):
        assert result.log_bayes_factors[i, j] == pytest.approx(brute_pair(args, i, j), abs=3e-14)
    assert result.log_bayes_factors.shape == (2, 2)


@pytest.mark.parametrize("which", ["hand", "object", "both"])
def test_missing_entity_marginal_is_exact_no_relation_evidence(which):
    args = inputs()
    for name in (("hand", "object") if which == "both" else (which,)):
        args[name + "_observed"][:] = False
        args[name + "_innovations"][:] = np.nan
    result = conditional_relational_likelihood(**args)
    np.testing.assert_array_equal(result.dependent_log_ratios, 0.)
    np.testing.assert_array_equal(result.log_bayes_factors, 0.)
    assert not result.pair_supported.any()
    np.testing.assert_array_equal(result.pair_log_likelihoods, result.null_log_likelihood)


def test_partial_dimension_marginals_use_cross_submatrix_not_zero_filled_points():
    args = inputs()
    args["hand_observed"][0, :, 1] = False
    args["object_observed"][0, :, 0] = False
    args["hand_innovations"][0, :, 1] = np.nan
    args["object_innovations"][0, :, 0] = np.nan
    result = conditional_relational_likelihood(**args)
    assert result.log_bayes_factors[0, 0] == pytest.approx(brute_pair(args, 0, 0), abs=3e-14)


def test_missing_middle_frame_advances_latent_regime_instead_of_resetting_it():
    args = inputs(steps=3)
    args["hand_observed"][0, 1] = False
    args["hand_innovations"][0, 1] = np.nan
    result = conditional_relational_likelihood(**args)
    assert result.dependent_log_ratios[0, 0, 1] == 0
    assert result.log_bayes_factors[0, 0] == pytest.approx(brute_pair(args, 0, 0), abs=3e-14)


def test_block_diagonal_gaussian_relation_equals_null_for_all_priors_and_support():
    args = inputs(2, 2, 4)
    args["cross_covariance"][:] = 0
    args["regime_initial"] = np.array([0., 1.])
    args["regime_transition"] = np.array([[1., 0.], [0., 1.]])
    args["hand_observed"][0, :, 0] = False
    result = conditional_relational_likelihood(**args)
    np.testing.assert_array_equal(result.log_bayes_factors, 0)


def test_latent_path_mass_is_summed_not_best_path_probability():
    args = inputs(steps=3)
    args["cross_covariance"][:] = 0
    args["regime_initial"] = np.array([.5, .5])
    args["regime_transition"][:] = .5
    result = conditional_relational_likelihood(**args)
    assert result.log_bayes_factors[0, 0] == 0  # Eight equally weighted paths sum to1.
    assert math.log(.5 ** 3) < result.log_bayes_factors[0, 0]  # MAP would report log(1/8).


def test_null_uses_all_entity_marginals_and_one_common_bank_factor():
    args = inputs(2, 2, 2)
    args["common_log_factors"][:] = [-.2, -.7]
    result = conditional_relational_likelihood(**args)
    expected = args["common_log_factors"].sum()
    for prefix in ("hand", "object"):
        for i in range(2):
            for t in range(2):
                expected += math.log(density(args[prefix + "_innovations"][i, t],
                    args[prefix + "_mean"][i, t], args[prefix + "_covariance"][i],
                    args[prefix + "_observed"][i, t]))
    assert result.null_log_likelihood == pytest.approx(expected, abs=2e-14)


def test_common_clutter_and_unselected_entities_cancel_algebraically_not_large_subtraction():
    original = inputs(1, 1, 3)
    a = conditional_relational_likelihood(**original)
    larger = inputs(2, 3, 3)
    for name in ("hand_innovations", "hand_mean", "hand_observed", "hand_covariance"):
        larger[name][0] = original[name][0]
    for name in ("object_innovations", "object_mean", "object_observed", "object_covariance"):
        larger[name][0] = original[name][0]
    larger["cross_covariance"][0, 0] = original["cross_covariance"][0, 0]
    larger["common_log_factors"][:] = -1e15
    b = conditional_relational_likelihood(**larger)
    assert b.log_bayes_factors[0, 0] == a.log_bayes_factors[0, 0]
    assert b.null_log_likelihood != a.null_log_likelihood


def test_bank_order_permutations_relabel_all_components_without_top1_selection():
    args = inputs(2, 3, 3)
    args["hypothesis_prior"] = np.array([.4, .05, .1, .1, .1, .1, .15])
    expected = conditional_relational_likelihood(**args)
    hp, op = np.array([1, 0]), np.array([2, 0, 1])
    for prefix, order in (("hand", hp), ("object", op)):
        args[prefix + "_keys"] = tuple(args[prefix + "_keys"][i] for i in order)
        for suffix in ("innovations", "observed", "mean", "covariance"):
            args[prefix + "_" + suffix] = args[prefix + "_" + suffix][order]
    args["cross_covariance"] = args["cross_covariance"][hp][:, op]
    prior = args["hypothesis_prior"]
    args["hypothesis_prior"] = np.r_[prior[0], prior[1:].reshape(2, 3)[hp][:, op].ravel()]
    actual = conditional_relational_likelihood(**args)
    np.testing.assert_array_equal(actual.log_bayes_factors, expected.log_bayes_factors[hp][:, op])
    np.testing.assert_array_equal(actual.relative_hypothesis_log_weights[1:].reshape(2, 3),
        expected.relative_hypothesis_log_weights[1:].reshape(2, 3)[hp][:, op])
    assert actual.relative_hypothesis_log_weights[0] == expected.relative_hypothesis_log_weights[0]
    assert actual.hand_keys == ("hand-1", "hand-0")


def test_duplicate_representation_cannot_gain_prior_mass_when_external_prior_is_split():
    args = inputs(1, 2, 2)
    for suffix in ("innovations", "observed", "mean", "covariance"):
        args["object_" + suffix][1] = args["object_" + suffix][0]
    args["cross_covariance"][0, 1] = args["cross_covariance"][0, 0]
    args["hypothesis_prior"] = np.array([.4, .3, .3])  # Two views of one hypothesis split .6.
    result = conditional_relational_likelihood(**args)
    assert result.log_bayes_factors[0, 0] == result.log_bayes_factors[0, 1]
    pair_weights = result.relative_hypothesis_log_weights[1:]
    combined = math.log(float(np.exp(pair_weights).sum()))
    assert combined == pytest.approx(math.log(.6) + result.log_bayes_factors[0, 0])
    # No method infers duplicate physical identity from keys or learns that split.


@pytest.mark.parametrize("constant", [-1e15, -1e100])
def test_comparative_weights_remove_huge_common_factors_before_addition(constant):
    args = inputs(2, 2, 3)
    expected = conditional_relational_likelihood(**args)
    args["common_log_factors"][:] = constant
    actual = conditional_relational_likelihood(**args)
    assert actual.null_log_likelihood != expected.null_log_likelihood
    assert actual.relative_hypothesis_log_weights.tobytes() == expected.relative_hypothesis_log_weights.tobytes()
    np.testing.assert_array_equal(actual.relative_hypothesis_log_weights,
        np.r_[0., actual.log_bayes_factors.ravel()] + np.log(args["hypothesis_prior"]))


@pytest.mark.parametrize("scale", [.25, 8.])
def test_shared_units_and_covariance_transform_cancel_density_jacobian(scale):
    args = inputs()
    original = conditional_relational_likelihood(**args)
    for key in ("hand_innovations", "object_innovations", "hand_mean", "object_mean"):
        args[key] *= scale
    for key in ("hand_covariance", "object_covariance", "cross_covariance"):
        args[key] *= scale ** 2
    transformed = conditional_relational_likelihood(**args)
    np.testing.assert_allclose(transformed.log_bayes_factors, original.log_bayes_factors, atol=3e-14)


def test_stationary_correlated_model_can_score_positive_but_cannot_accept_identity():
    args = inputs(steps=4)
    args["hand_innovations"][:] = args["object_innovations"][:] = 0
    args["hand_covariance"][:] = args["object_covariance"][:] = np.eye(2)
    args["cross_covariance"][:] = .8 * np.eye(2)
    result = conditional_relational_likelihood(**args)
    assert result.log_bayes_factors[0, 0] > 0  # Identifiability/calibration is not established.
    for forbidden in ("accepted_identity", "contact", "posterior", "confidence", "calibrated"):
        assert forbidden not in result.__dataclass_fields__


def test_common_camera_estimation_error_can_create_false_relation_evidence():
    args = inputs(steps=20)
    # Independent true motions identically zero; a shared camera error survives
    # subtraction in both residuals. This does not certify a physical relation.
    error = np.tile(np.array([[.5, -.5], [-.5, .5]]), (10, 1))
    args["hand_innovations"][0] = args["object_innovations"][0] = -error
    args["hand_covariance"][:] = args["object_covariance"][:] = np.eye(2)
    args["cross_covariance"][:] = .8 * np.eye(2)
    result = conditional_relational_likelihood(**args)
    assert result.log_bayes_factors[0, 0] > 0
    assert not hasattr(result, "accepted_identity")


@pytest.mark.parametrize("key,value", [
    ("frame_index", np.array([0, 2, 3, 4], dtype=np.int64)),
    ("frame_index", np.arange(4, dtype=np.int32)),
    ("hand_keys", ("same", "same")),
    ("hand_observed", np.ones((1, 3, 2), np.int64)),
    ("regime_initial", np.array([.2, .7])),
    ("regime_initial", np.array([-1., 2.])),
    ("regime_transition", np.array([[.5, .5], [0., 0.]])),
    ("hypothesis_prior", np.array([0., 1.])),
    ("hand_covariance", np.array([[[1., .2], [.1, 1.]]])),
    ("object_covariance", np.array([[[1., 0.], [0., -1.]]])),
    ("cross_covariance", np.array([[[[2., 0.], [0., 2.]]]])),
    ("common_log_factors", np.array([0., np.nan, 0.])),
    ("common_log_factors", np.zeros((1, 3))),
    ("hand_innovations", np.full((1, 3, 2), np.nan)),
])
def test_malformed_bank_or_parameters_fail_whole_without_clipping_or_defaults(key, value):
    args = inputs(); args[key] = value
    with pytest.raises(ValueError):
        conditional_relational_likelihood(**args)


def test_readonly_noalias_outputs_and_zero_pair_prior_are_retained():
    args = inputs(); args["hypothesis_prior"] = np.array([1., 0.])
    result = conditional_relational_likelihood(**args)
    assert result.relative_hypothesis_log_weights[1] == -np.inf
    for value in result.__dict__.values():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.flags.writeable = True
            assert not any(np.shares_memory(value, a) for a in args.values() if isinstance(a, np.ndarray))
    with pytest.raises(FrozenInstanceError):
        result.hand_keys = ("changed",)
