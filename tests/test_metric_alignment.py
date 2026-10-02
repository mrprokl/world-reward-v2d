import json

import numpy as np
import pytest

from world_reward.metric_alignment import fit_shared_depth_scale


def arrays(count=3, factor=2.0):
    depth = np.linspace(0.5, 5, 64).reshape(8, 8)
    depth = np.stack([depth + i for i in range(count)])
    return depth, factor * depth, np.ones(depth.shape, dtype=bool)


def fit(depth, human, mask, indices=None, **kwargs):
    return fit_shared_depth_scale(depth, human, mask, list(range(len(depth))) if indices is None else indices, **kwargs)


def test_constant_positive_scale_shared_across_original_frames_and_json():
    depth, human, mask = arrays(factor=2.5)
    before = depth.copy()
    result = fit(depth, human, mask, [1, 20, 99])
    assert result.shared_scale == pytest.approx(2.5)
    assert result.median_absolute_relative_error == pytest.approx(0, abs=1e-14)
    assert result.ratio_p10 == pytest.approx(2.5)
    assert result.ratio_p90 == pytest.approx(2.5)
    assert result.correspondences == 192
    assert result.supported_frames == 3
    assert [frame.frame_index for frame in result.frames] == [1, 20, 99]
    assert all(frame.supported for frame in result.frames)
    assert np.array_equal(depth, before)
    report = result.to_dict()
    assert "not ground-truth" in report["interpretation"]
    assert "offset fixed zero" in report["model"]
    json.dumps(report, allow_nan=False)


def test_pixel_outliers_below_half_do_not_shift_robust_log_median():
    depth, human, mask = arrays()
    human[:, :2] = depth[:, :2] * 20
    result = fit(depth, human, mask)
    assert result.shared_scale == pytest.approx(2)
    assert result.ratio_p10 == pytest.approx(2)
    assert result.ratio_p90 == pytest.approx(20)
    assert result.median_absolute_relative_error == pytest.approx(0, abs=1e-14)


def test_one_corrupted_frame_cannot_dominate_other_supported_frames():
    depth, human, mask = arrays(count=5)
    human[0] = depth[0] * 100
    result = fit(depth, human, mask)
    assert result.shared_scale == pytest.approx(2)
    assert result.frames[0].median_ratio == pytest.approx(100)
    assert result.frames[0].median_absolute_relative_error == pytest.approx(0.98)


def test_equal_per_frame_weight_avoids_large_visible_actor_domination():
    depth, human, mask = arrays()
    human[0] = depth[0] * 10
    mask[1:, 4:] = False  # 64 wrong pixels in one frame vs 32+32 good pixels
    result = fit(depth, human, mask)
    assert result.shared_scale == pytest.approx(2)
    assert result.correspondences == 128
    assert [frame.valid_correspondences for frame in result.frames] == [64, 32, 32]


def test_scale_is_true_frame_weighted_log_l1_median_not_average_frame_medians():
    depth = np.ones((3, 8, 8))
    human = depth.copy()
    human[0].flat[:32] = 1
    human[0].flat[32:] = 100
    human[1] = 2
    human[2] = 3
    result = fit(depth, human, np.ones_like(depth, dtype=bool))
    assert result.shared_scale == pytest.approx(np.sqrt(6))
    assert result.frames[0].median_ratio == pytest.approx(10)
    assert result.frames[1].median_ratio == pytest.approx(2)
    assert result.frames[2].median_ratio == pytest.approx(3)


def test_occluded_frames_missing_support_are_reported_not_filled():
    depth, human, mask = arrays(count=5)
    mask[1] = False
    human[3] = 0
    result = fit(depth, human, mask)
    assert result.shared_scale == pytest.approx(2)
    assert result.supported_frames == 3
    assert result.frames[1].visible_pixels == 0
    assert result.frames[1].median_ratio is None
    assert result.frames[3].nonpositive_finite_pairs == 64
    assert result.frames[3].median_absolute_relative_error is None


def test_selected_nonfinite_and_nonpositive_pairs_explicitly_counted_and_excluded():
    depth, human, mask = arrays()
    depth[:, 0, 0] = np.nan
    human[:, 0, 1] = np.inf
    depth[:, 0, 2] = 0
    human[:, 0, 3] = -1
    depth[:, 0, 4] = np.inf
    human[:, 0, 4] = -1  # count once as nonfinite, not also nonpositive finite
    result = fit(depth, human, mask)
    assert result.shared_scale == pytest.approx(2)
    assert result.correspondences == 3 * 59
    assert all(frame.nonfinite_pairs == 3 for frame in result.frames)
    assert all(frame.nonpositive_finite_pairs == 2 for frame in result.frames)


def test_bad_values_outside_visible_mask_do_not_contribute_to_diagnostics_or_fit():
    depth, human, mask = arrays()
    depth[:, 0, 0] = np.nan
    human[:, 0, 1] = -1
    mask[:, 0, :2] = False
    result = fit(depth, human, mask)
    assert result.correspondences == 186
    assert all(frame.nonfinite_pairs == frame.nonpositive_finite_pairs == 0 for frame in result.frames)


def test_large_depth_motion_and_sparse_temporal_gaps_do_not_reject_or_get_framewise_scale():
    depth = np.stack([np.full((8, 8), z) for z in (0.01, 100, 2)])
    result = fit(depth, 4 * depth, np.ones_like(depth, dtype=bool), [0, 100, 200])
    assert result.shared_scale == pytest.approx(4)
    assert result.supported_frames == 3


def test_additive_depth_bias_is_not_silently_fit_as_offset_or_metric_truth():
    depth, _, mask = arrays()
    human = 2 * depth + 3
    result = fit(depth, human, mask)
    assert result.ratio_p10 < result.ratio_p90
    assert result.median_absolute_relative_error > 0
    assert not hasattr(result, "offset")
    assert not hasattr(result, "frame_scales")


def test_streaming_inputs_produce_identical_shared_fit():
    depth, human, mask = arrays()
    expected = fit(depth, human, mask)
    streamed = fit_shared_depth_scale(iter(depth), iter(human), iter(mask), [0, 1, 2])
    assert streamed == expected


@pytest.mark.parametrize("kind", ["zero", "negative", "nonfinite", "invisible", "too_few_pixels"])
def test_insufficient_supported_frames_fail_explicitly(kind):
    depth, human, mask = arrays()
    if kind == "zero":
        depth[1] = 0
    elif kind == "negative":
        human[1] = -1
    elif kind == "nonfinite":
        depth[1] = np.nan
    elif kind == "invisible":
        mask[1] = False
    else:
        mask[1].flat[31:] = False
    with pytest.raises(ValueError, match="Insufficient supported frames.*per-frame valid correspondences"):
        fit(depth, human, mask)


@pytest.mark.parametrize("kwargs", [
    {"min_correspondences_per_frame": 0}, {"min_correspondences_per_frame": True},
    {"min_correspondences_per_frame": 1.5}, {"min_supported_frames": 2},
    {"min_supported_frames": True},
])
def test_support_configuration_is_explicit_positive_integer(kwargs):
    with pytest.raises(ValueError):
        fit(*arrays(), **kwargs)


@pytest.mark.parametrize("indices", [[], [0, 0, 1], [2, 1, 0], [-1, 0, 1],
    [0.0, 1, 2], [True, 1, 2], [np.nan, 1, 2]])
def test_original_indices_are_strict_nonnegative_integer_provenance(indices):
    with pytest.raises(ValueError):
        fit(*arrays(), indices=indices)


@pytest.mark.parametrize("kind", ["depth_shape", "human_shape", "mask_shape", "count", "varying_grid"])
def test_shapes_counts_and_pixel_grids_must_match(kind):
    depth, human, mask = arrays()
    if kind == "depth_shape":
        depth = depth[0]
    elif kind == "human_shape":
        human = human[:, :7]
    elif kind == "mask_shape":
        mask = mask[:, :7]
    elif kind == "count":
        human = human[:2]
    else:
        depth = [depth[0], depth[1], np.ones((9, 8))]
        human = [human[0], human[1], np.ones((9, 8))]
        mask = [mask[0], mask[1], np.ones((9, 8), dtype=bool)]
    with pytest.raises(ValueError):
        fit(depth, human, mask, [0, 1, 2])


@pytest.mark.parametrize("kind", ["bool_depth", "complex_depth", "string_depth", "numeric_mask", "masked_array"])
def test_no_implicit_depth_or_visibility_coercion(kind):
    depth, human, mask = arrays()
    if kind == "bool_depth":
        depth = depth.astype(bool)
    elif kind == "complex_depth":
        depth = depth.astype(complex)
    elif kind == "string_depth":
        depth = depth.astype(str)
    elif kind == "numeric_mask":
        mask = mask.astype(np.uint8)
    else:
        depth = np.ma.array(depth, mask=False)
    with pytest.raises(ValueError):
        fit(depth, human, mask)


def test_integer_depth_and_numpy_original_indices_are_supported_and_json_native():
    depth = np.full((3, 8, 8), 2, dtype=np.int32)
    result = fit(depth, 3 * depth, np.ones_like(depth, dtype=bool), np.array([0, 2, 4], dtype=np.int64))
    assert result.shared_scale == pytest.approx(3)
    assert all(type(frame.frame_index) is int for frame in result.frames)
    json.dumps(result.to_dict(), allow_nan=False)
