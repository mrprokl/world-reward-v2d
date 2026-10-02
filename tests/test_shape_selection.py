"""Tiny mathematical nested-selection guards, not model/accuracy validation."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from world_reward.shape_selection import select_shape_candidate, shape_schur_complement


def select(**changes):
    values = dict(pose_only_losses=[.04, .04, .04], shape_pose_losses=[.03, .03, .03],
                  pose_only_iou=[.8, .8, .8], shape_pose_iou=[.8, .8, .8], shape_schur_eigenvalues=np.ones(5),
                  saturated_bounds=False, pose_modes_ambiguous=False, heldout_pixels_disjoint=True)
    values.update(changes)
    return select_shape_candidate(**values)


def test_constant_positive_gain_above_one_se_yields_only_unvalidated_proposal():
    result = select()
    assert result.decision == "shape_pose_proposal" and result.reasons == ()
    assert result.report["mean_paired_gain_m2"] == pytest.approx(.01)
    assert result.report["paired_gain_standard_error_m2"] == 0.
    assert result.report["policy_thresholds_validated"] is False
    assert result.report["adoption_authorized"] is False
    assert result.report["challenge_performance_verified"] is False
    assert result.report["statistical_calibration_verified"] is False
    assert result.report["block_independence_verified"] is False
    assert result.report["statistical_identifiability_verified"] is False
    json.dumps(result.report, allow_nan=False)
    with pytest.raises(FrozenInstanceError): result.decision = "pose_only"


@pytest.mark.parametrize("losses", [[.04, .04, .04], [.05, .05, .05], [.04, .04, .02]])
def test_zero_negative_or_gain_equal_to_one_se_defaults_to_pose_only(losses):
    result = select(shape_pose_losses=losses)
    assert result.decision == "pose_only"
    assert "paired_gain_not_above_one_standard_error" in result.reasons


def test_statistics_are_paired_not_independent_model_variances():
    result = select(pose_only_losses=[1., 2., 3., 4.], shape_pose_losses=[.8, 1.9, 2.85, 3.7],
                    pose_only_iou=[.8] * 4, shape_pose_iou=[.8] * 4)
    deltas = np.array([.2, .1, .15, .3])
    assert result.report["mean_paired_gain_m2"] == pytest.approx(deltas.mean())
    assert result.report["paired_gain_standard_error_m2"] == pytest.approx(deltas.std(ddof=1) / 2)
    assert result.report["paired_blocks"] == 4 and result.report["validation_unit"] == "caller_supplied_temporal_block"


@pytest.mark.parametrize("field,reason", [("saturated_bounds", "shape_bounds_saturated"),
                                         ("pose_modes_ambiguous", "pose_modes_ambiguous")])
def test_shape_bounds_or_pose_mode_ambiguity_always_abstains(field, reason):
    result = select(**{field: True})
    assert result.decision == "pose_only" and reason in result.reasons


def test_non_disjoint_validation_can_never_authorize_shape_proposal():
    result = select(heldout_pixels_disjoint=False)
    assert result.decision == "pose_only" and "heldout_pixels_not_disjoint" in result.reasons


def test_silhouette_mean_must_not_regress_and_missing_pair_abstains():
    result = select(shape_pose_iou=[.8, .8, .799999])
    assert result.decision == "pose_only" and "mean_heldout_silhouette_regressed" in result.reasons
    assert select(shape_pose_iou=[.9, .8, .8]).decision == "shape_pose_proposal"
    missing = select(pose_only_iou=None, shape_pose_iou=None)
    assert missing.decision == "pose_only" and "paired_silhouette_validation_missing" in missing.reasons
    assert missing.report["mean_iou_M0"] is None and missing.report["mean_iou_M1"] is None


@pytest.mark.parametrize("eigenvalues", [np.zeros(5), [-1., 1., 1., 1., 1.], [0., 1., 1., 1., 1.], [1e-9, 1., 1., 1., 1.]])
def test_negative_null_or_ill_conditioned_shape_schur_abstains(eigenvalues):
    result = select(shape_schur_eigenvalues=eigenvalues)
    assert result.decision == "pose_only" and "shape_schur_nonpositive_or_ill_conditioned" in result.reasons


def test_schur_relative_threshold_not_an_absolute_information_proof():
    assert select(shape_schur_eigenvalues=[1e-8, 1., 1., 1., 1.]).decision == "shape_pose_proposal"
    tiny = select(shape_schur_eigenvalues=np.full(5, 1e-250))
    assert tiny.decision == "shape_pose_proposal"
    assert tiny.report["absolute_information_verified"] is False
    assert tiny.report["schur_numerical_condition_proxy_pass"] is True


def test_correlated_block_copies_cannot_be_claimed_as_verified_independence():
    result = select(pose_only_losses=np.full(10, .04), shape_pose_losses=np.full(10, .03),
                    pose_only_iou=np.full(10, .8), shape_pose_iou=np.full(10, .8))
    assert result.report["block_independence_verified"] is False
    assert result.report["statistical_calibration_verified"] is False
    assert result.report["adoption_authorized"] is False


@pytest.mark.parametrize("length", [0, 1, 2])
def test_too_few_temporal_blocks_fails_not_a_framewise_victory(length):
    with pytest.raises(ValueError, match="three"):
        select(pose_only_losses=np.ones(length), shape_pose_losses=np.ones(length),
               pose_only_iou=np.ones(length), shape_pose_iou=np.ones(length))


@pytest.mark.parametrize("field", ["pose_only_losses", "shape_pose_losses", "pose_only_iou", "shape_pose_iou", "shape_schur_eigenvalues"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_statistics_or_schur_never_silently_filtered(field, bad):
    size = 5 if field == "shape_schur_eigenvalues" else 3
    values = np.ones(size)
    values[0] = bad
    with pytest.raises(ValueError): select(**{field: values})


@pytest.mark.parametrize("field,value", [("pose_only_losses", [-1., 1., 1.]), ("shape_pose_losses", [-1., 1., 1.]),
                                        ("pose_only_iou", [1.1, .8, .8]), ("shape_pose_iou", [-.1, .8, .8]),
                                        ("shape_pose_losses", [1., 1.]), ("pose_only_losses", np.ones((3, 1))),
                                        ("shape_schur_eigenvalues", [1., 1., 1., 1.]),
                                        ("pose_only_iou", None), ("shape_pose_iou", None)])
def test_mismatched_negative_or_nonvector_inputs_rejected(field, value):
    with pytest.raises(ValueError): select(**{field: value})


@pytest.mark.parametrize("field", ["pose_only_losses", "shape_pose_losses", "pose_only_iou", "shape_pose_iou", "shape_schur_eigenvalues"])
def test_masked_arrays_cannot_hide_validation_records(field):
    value = np.ones(5 if field == "shape_schur_eigenvalues" else 3)
    with pytest.raises(ValueError): select(**{field: np.ma.array(value, mask=False)})


@pytest.mark.parametrize("field", ["saturated_bounds", "pose_modes_ambiguous", "heldout_pixels_disjoint"])
@pytest.mark.parametrize("value", [1, None, np.bool_(True), "false"])
def test_risk_and_heldout_flags_require_explicit_bool(field, value):
    with pytest.raises(ValueError, match="booleans"): select(**{field: value})


def test_huge_finite_losses_use_stable_standard_error():
    result = select(pose_only_losses=[1e200, 1e200, 1e200], shape_pose_losses=[.8e200, .9e200, .7e200])
    assert np.isfinite(result.report["paired_gain_standard_error_m2"])
    assert result.report["mean_paired_gain_m2"] == pytest.approx(.2e200)


def test_selector_does_not_mutate_input_arrays():
    loss0, loss1, iou, eigen = np.full(3, .04), np.full(3, .03), np.full(3, .8), np.ones(5)
    copies = [value.copy() for value in (loss0, loss1, iou, eigen)]
    select(pose_only_losses=loss0, shape_pose_losses=loss1, pose_only_iou=iou, shape_pose_iou=iou,
           shape_schur_eigenvalues=eigen)
    for actual, expected in zip((loss0, loss1, iou, eigen), copies, strict=True): np.testing.assert_array_equal(actual, expected)


def test_schur_matches_full_exact_block_elimination_and_preserves_inputs():
    shape, pose = np.diag([3., 4., 5., 6., 7.]), np.diag([2., 4.])
    cross = np.arange(10, dtype=float).reshape(5, 2) / 10
    copies = [value.copy() for value in (shape, pose, cross)]
    result = shape_schur_complement(shape, pose, cross)
    np.testing.assert_allclose(result, shape - cross @ np.diag([.5, .25]) @ cross.T, atol=1e-15)
    for actual, expected in zip((shape, pose, cross), copies, strict=True): np.testing.assert_array_equal(actual, expected)
    assert result is not shape


def test_schur_exposes_pose_shape_gauge_without_regularization():
    schur = shape_schur_complement(np.eye(5), np.eye(5), np.eye(5))
    np.testing.assert_array_equal(schur, np.zeros((5, 5)))
    assert select(shape_schur_eigenvalues=np.linalg.eigvalsh(schur)).decision == "pose_only"
    schur = shape_schur_complement(np.eye(5), np.eye(5), 2 * np.eye(5))
    np.testing.assert_array_equal(schur, -3 * np.eye(5))


@pytest.mark.parametrize("pose", [np.zeros((2, 2)), np.diag([-1., 1.]), np.diag([1e-9, 1.]),
                                  np.ones((2, 3)), np.zeros((0, 0)), np.ones(2)])
def test_bad_pose_hessian_no_damping_or_pseudoinverse(pose):
    with pytest.raises(ValueError): shape_schur_complement(np.eye(5), pose, np.zeros((5, pose.shape[0])))


@pytest.mark.parametrize("field", ["shape", "pose", "cross"])
def test_nonfinite_masked_wrong_sized_hessians_fail(field):
    values = dict(shape=np.eye(5), pose=np.eye(2), cross=np.zeros((5, 2)))
    values[field] = np.full_like(values[field], np.nan)
    with pytest.raises(ValueError): shape_schur_complement(values["shape"], values["pose"], values["cross"])
    values[field] = np.ma.array(np.eye(5) if field == "shape" else np.eye(2) if field == "pose" else np.zeros((5, 2)), mask=False)
    with pytest.raises(ValueError): shape_schur_complement(values["shape"], values["pose"], values["cross"])


@pytest.mark.parametrize("field", ["shape", "pose"])
def test_asymmetric_hessian_is_not_silently_symmetrized(field):
    shape, pose = np.eye(5), np.eye(2)
    value = shape if field == "shape" else pose
    value[0, 1] = 1e-12
    with pytest.raises(ValueError, match="symmetric"): shape_schur_complement(shape, pose, np.zeros((5, 2)))


def test_schur_rejects_wrong_shape_control_and_cross_dimensions():
    with pytest.raises(ValueError): shape_schur_complement(np.eye(4), np.eye(2), np.zeros((4, 2)))
    with pytest.raises(ValueError): shape_schur_complement(np.eye(5), np.eye(2), np.zeros((2, 5)))
