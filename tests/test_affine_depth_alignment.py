"""Analytical own camera-Z fixtures only; no model, challenge data or GPU."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from world_reward.affine_depth_alignment import (
    UnderconstrainedDepthAlignment, apply_affine_depth_alignment,
    fit_shared_affine_depth_alignment, spatial_split_mask,
)
from world_reward.metric_alignment import fit_shared_depth_scale


def arrays(alpha=1.7, offsets=(.35, -.2, .7)):
    yy, xx = np.indices((32, 32))
    z = np.stack([1.2+.015*xx+.01*yy+i*.4 for i in range(len(offsets))])
    h = alpha*z+np.asarray(offsets)[:, None, None]
    return z, h, np.ones(z.shape, bool)


def fit(z, h, mask, indices=None):
    return fit_shared_affine_depth_alignment(z, h, mask, list(range(len(z))) if indices is None else indices)


def test_known_affine_recovered_once_with_holdout_and_immutable_json():
    data = arrays(); copies = [a.copy() for a in data]
    result = fit(*data, [0, 20, 99])
    assert result.selection == "affine" and result.shared_scale == pytest.approx(1.7, abs=1e-12)
    np.testing.assert_allclose(result.frame_offsets, [.35, -.2, .7], atol=1e-12)
    assert result.heldout_affine_median_relative_error < 1e-12
    assert result.heldout_alpha_only_median_relative_error > .01
    assert result.normalized_design_rank == 4 and result.within_frame_depth_variation_ratio >= .005
    assert result.image_size_hw == (32, 32)
    assert [f.frame_index for f in result.frames] == [0, 20, 99]
    assert all(f.train_pixels+f.heldout_pixels == f.visible_pixels == 1024 for f in result.frames)
    assert all(np.array_equal(a, b) for a, b in zip(data, copies))
    report = result.to_dict(); json.dumps(report, allow_nan=False)
    assert not report["metric_accuracy_verified"] and not report["adoption_authorized"] and not report["block_independence_verified"]
    with pytest.raises(FrozenInstanceError): result.shared_scale = 3


def test_alpha_only_baseline_uses_exact_same_training_pixels_and_zero_offsets():
    z, _, mask = arrays(); h = 2*z
    yy, xx = np.indices(z.shape[1:]); train = mask & ((yy//8+xx//8) % 2 == 0)
    baseline = fit_shared_depth_scale(z, h, train, [0, 1, 2], min_correspondences_per_frame=32, min_supported_frames=3)
    result = fit(z, h, mask)
    assert result.alpha_only_scale == baseline.shared_scale
    assert result.selection == "alpha_only" and result.frame_offsets == (0., 0., 0.)


def test_huber_keeps_contaminated_pixels_supported_without_slope_bias():
    z, h, mask = arrays(); h[:, ::5, ::4] += 8.
    result = fit(z, h, mask)
    assert result.affine_proposal_scale == pytest.approx(1.7, abs=1e-4)
    np.testing.assert_allclose(result.affine_proposal_offsets, [.35, -.2, .7], atol=1e-4)
    assert all(f.visible_pixels == 1024 for f in result.frames)


def test_holdout_changes_selection_not_fitted_parameters():
    z, h, mask = arrays(); first = fit(z, h, mask)
    yy, xx = np.indices(z.shape[1:]); heldout = ((yy//8+xx//8) % 2) != 0
    h[:, heldout] = first.alpha_only_scale*z[:, heldout]
    second = fit(z, h, mask)
    assert second.affine_proposal_scale == first.affine_proposal_scale
    assert second.affine_proposal_offsets == first.affine_proposal_offsets
    assert second.alpha_only_scale == first.alpha_only_scale
    assert second.selection == "alpha_only" and second.frame_offsets == (0., 0., 0.)


def test_one_heldout_frame_regression_cannot_hide_behind_median_gain():
    z, h, mask = arrays(); initial = fit(z, h, mask)
    yy, xx = np.indices(z.shape[1:]); heldout = ((yy//8+xx//8) % 2) != 0
    h[0, heldout] = initial.alpha_only_scale*z[0, heldout]
    result = fit(z, h, mask)
    assert result.heldout_affine_median_relative_error < .95*result.heldout_alpha_only_median_relative_error
    assert result.selection == "alpha_only"


@pytest.mark.parametrize("kind", ["constant", "tiny_variation", "negative_slope"])
def test_interframe_motion_is_not_withinframe_identification(kind):
    z, h, mask = arrays()
    if kind == "constant": z = np.stack([np.full((32, 32), v) for v in (1., 3., 10.)]); h = 2*z+.4
    elif kind == "tiny_variation": z = 2+z*1e-5; h = 2*z+.4
    else: h = 10-z
    with pytest.raises(ValueError if kind == "negative_slope" else UnderconstrainedDepthAlignment) as error:
        fit(z, h, mask)
    if kind == "negative_slope": assert type(error.value) is ValueError


@pytest.mark.parametrize("kind", ["nan", "infinity", "zero", "negative", "numeric_mask", "masked", "grid", "count", "two_frames", "unsupported", "single_split"])
def test_every_selected_pair_frame_and_fixed_grid_required(kind):
    z, h, mask = arrays()
    if kind == "nan": z[1, 0, 0] = np.nan
    elif kind == "infinity": h[1, 0, 0] = np.inf
    elif kind == "zero": z[1, 0, 0] = 0
    elif kind == "negative": h[1, 0, 0] = -1
    elif kind == "numeric_mask": mask = mask.astype(np.uint8)
    elif kind == "masked": z = np.ma.array(z, mask=False)
    elif kind == "grid": h = h[:, :-1]
    elif kind == "count": mask = mask[:2]
    elif kind == "two_frames": z, h, mask = z[:2], h[:2], mask[:2]
    elif kind == "unsupported": mask[1] = False
    else: mask[1] = False; mask[1, :8, :8] = True
    with pytest.raises(ValueError): fit(z, h, mask)


@pytest.mark.parametrize("indices", [[0, 0, 1], [1, 0, 2], [-1, 0, 1], [0., 1, 2], [True, 1, 2], [0, 1]])
def test_original_indices_no_duplicate_drop_or_coercion(indices):
    with pytest.raises(ValueError): fit(*arrays(), indices)


def test_invalid_unselected_pixels_do_not_drift_mask_or_affect_fit():
    z, h, mask = arrays(); mask[:, 0, 0] = False
    original = fit(z, h, mask); z[:, 0, 0] = np.nan; h[:, 0, 0] = -1
    changed = fit(z, h, mask)
    assert original == changed and all(f.visible_pixels == 1023 for f in changed.frames)


def test_shared_spatial_split_is_exact_baseline_fallback_contract():
    z = np.stack([np.full((32, 32), v) for v in (1., 3., 10.)]); h = 2*z+.4; mask = np.ones(z.shape, bool)
    with pytest.raises(UnderconstrainedDepthAlignment): fit(z, h, mask)
    split = spatial_split_mask(32, 32)
    assert split.dtype == np.bool_ and split.sum() == 512
    assert split[0, 0] and not split[0, 8] and split[8, 8]
    fallback = fit_shared_depth_scale(z, h, mask & split, [0, 1, 2], min_correspondences_per_frame=32, min_supported_frames=3)
    assert fallback.supported_frames == 3 and fallback.correspondences == 1536
    for dimensions in ((0, 32), (True, 32), (32., 32)):
        with pytest.raises(ValueError): spatial_split_mask(*dimensions)


def test_affine_applies_full_camera_Z_once_same_rays_not_object_scale():
    z, h, mask = arrays(); result = fit(z, h, mask)
    K = np.array([[600., 0., 16.], [0., 610., 16.], [0., 0., 1.]])
    old = (z[1].copy(), mask[1].copy(), K.copy())
    aligned, points = apply_affine_depth_alignment(z[1], mask[1], K, result, frame_index=1)
    np.testing.assert_allclose(aligned, h[1], atol=1e-12)
    yy, xx = np.indices(aligned.shape)
    np.testing.assert_allclose(points[:, :, 0]/points[:, :, 2]*K[0, 0]+K[0, 2], xx+.5, atol=1e-12)
    np.testing.assert_allclose(points[:, :, 1]/points[:, :, 2]*K[1, 1]+K[1, 2], yy+.5, atol=1e-12)
    assert all(np.array_equal(a, b) for a, b in zip(old, (z[1], mask[1], K)))


def test_apply_never_clips_negative_or_fills_invalid_points():
    z, h, mask = arrays(); result = fit(z, h, mask); K = np.array([[600., 0., 16.], [0., 600., 16.], [0., 0., 1.]])
    negative = replace(result, frame_offsets=(-100.,)*3)
    with pytest.raises(ValueError): apply_affine_depth_alignment(z[0], mask[0], K, negative, frame_index=0)
    with pytest.raises(ValueError): apply_affine_depth_alignment(z[0], mask[0], K, result, frame_index=9)
    invalid = z[0].copy(); invalid[0, 0] = np.nan; valid = mask[0].copy(); valid[0, 0] = False
    aligned, points = apply_affine_depth_alignment(invalid, valid, K, result, frame_index=0)
    assert np.isnan(aligned[0, 0]) and np.isnan(points[0, 0]).all() and not valid[0, 0]


@pytest.mark.parametrize("kind", ["bad_K", "numeric_validity", "changed_grid", "both_grids_changed", "nonfinite_valid", "nonpositive_valid", "bool_index"])
def test_apply_rejects_camera_mask_or_depth_drift(kind):
    z, h, mask = arrays(); result = fit(z, h, mask); depth, valid = z[0].copy(), mask[0].copy()
    K = np.array([[600., 0., 16.], [0., 600., 16.], [0., 0., 1.]])
    if kind == "bad_K": K[0, 1] = 1
    elif kind == "numeric_validity": valid = valid.astype(np.uint8)
    elif kind == "changed_grid": valid = valid[:-1]
    elif kind == "both_grids_changed": depth, valid = depth[:-1], valid[:-1]
    elif kind == "nonfinite_valid": depth[0, 0] = np.inf
    elif kind == "nonpositive_valid": depth[0, 0] = 0
    with pytest.raises(ValueError): apply_affine_depth_alignment(depth, valid, K, result, frame_index=True if kind == "bool_index" else 0)
