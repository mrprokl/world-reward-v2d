"""Procedural representation contracts only; no Torch/model/native equivalence."""

import json

import numpy as np
import pytest

from world_reward.multiview_representation import (
    to_pytorch3d_once, validate_camera_pointmap, validate_joint_pixel_registration,
    validate_ssi_roundtrip,
)


def scene():
    height, width = 4, 5
    k = np.array([[8., 0., 2.], [0., 9., 1.5], [0., 0., 1.]])
    y, x = np.mgrid[:height, :width]
    z = 2 + (x + y) / 20
    points = np.stack(((x + .5 - k[0, 2]) / k[0, 0] * z,
                       (y + .5 - k[1, 2]) / k[1, 1] * z, z))
    return points, k, np.ones((height, width), dtype=bool)


def test_camera_ray_registration_and_report_are_not_depth_accuracy():
    points, k, valid = scene()
    camera = validate_camera_pointmap(points, k, valid, frame_index=250)
    assert camera.frame_index == 250
    assert camera.report["max_reprojection_error_px"] < 1e-14
    assert not camera.report["metric_depth_accuracy_verified"]
    assert not camera.report["native_model_verified"]
    assert not json.loads(json.dumps(camera.report, allow_nan=False))["ground_truth_used"]
    assert not camera.points_chw.flags.writeable
    assert not camera.valid_pixels.flags.writeable
    assert not camera.intrinsics.flags.writeable


def test_exactly_one_xy_flip_z_unchanged_and_second_flip_rejected():
    points, k, valid = scene()
    camera = validate_camera_pointmap(points, k, valid)
    p3d = to_pytorch3d_once(camera)
    np.testing.assert_array_equal(p3d.points_chw[:2], -points[:2])
    np.testing.assert_array_equal(p3d.points_chw[2], points[2])
    assert p3d.report["coordinate_flip_count"] == 1
    with pytest.raises(ValueError, match="second flip"):
        to_pytorch3d_once(p3d)
    with pytest.raises(ValueError, match="second flip"):
        to_pytorch3d_once(points)


def test_preflipped_pointmap_is_not_accepted_as_external_opencv():
    points, k, valid = scene()
    points[:2] *= -1
    with pytest.raises(ValueError, match="pixel-center"):
        validate_camera_pointmap(points, k, valid)


def test_integer_pixel_rays_fail_half_pixel_contract():
    points, k, valid = scene()
    points[0] -= .5 / k[0, 0] * points[2]
    points[1] -= .5 / k[1, 1] * points[2]
    with pytest.raises(ValueError, match="pixel-center"):
        validate_camera_pointmap(points, k, valid)


@pytest.mark.parametrize("warp", ["rotation", "translation"])
def test_coordinate_only_canonical_warp_without_rgb_reprojection_rejected(warp):
    points, k, valid = scene()
    if warp == "rotation":
        theta = .1
        r = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0., np.cos(theta)]])
        points = (r @ points.reshape(3, -1)).reshape(points.shape)
    else:
        points = points + np.array([.1, 0., .05])[:, None, None]
    with pytest.raises(ValueError, match="pixel-center"):
        validate_camera_pointmap(points, k, valid)


def test_ray_preserving_scaling_passes_but_is_not_metric_validation():
    points, k, valid = scene()
    points *= np.linspace(.5, 2, valid.size).reshape(valid.shape)[None]
    assert not validate_camera_pointmap(points, k, valid).report["metric_depth_accuracy_verified"]


def test_object_motion_virtual_camera_same_object_pixels_are_equivalent_only():
    points, k, valid = scene()
    r = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    t = np.array([.1, .2, 3.])
    object_points = r.T @ (points.reshape(3, -1) - t[:, None])
    virtual_camera = (r @ object_points + t[:, None]).reshape(points.shape)
    camera = validate_camera_pointmap(virtual_camera, k, valid)
    np.testing.assert_allclose(camera.points_chw, points)
    # This algebra does not assert that static background/human pixels obey T.


def test_explicit_invalid_support_keeps_invalid_coordinates_untouched():
    points, k, valid = scene()
    points[:, 0, 0] = np.nan
    valid[0, 0] = False
    camera = validate_camera_pointmap(points, k, valid)
    assert np.isnan(camera.points_chw[:, 0, 0]).all()
    assert camera.report["nonfinite_pixels_outside_support"] == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0., -1.])
def test_bad_supported_depth_not_repaired(bad):
    points, k, valid = scene()
    points[2, 0, 0] = bad
    with pytest.raises(ValueError, match="positive-depth"):
        validate_camera_pointmap(points, k, valid)


@pytest.mark.parametrize("field", ["points", "intrinsics", "valid"])
def test_camera_masked_arrays_reject(field):
    points, k, valid = scene()
    arrays = {"points": points, "intrinsics": k, "valid": valid}
    arrays[field] = np.ma.array(arrays[field], mask=False)
    with pytest.raises(ValueError, match="masked"):
        validate_camera_pointmap(arrays["points"], arrays["intrinsics"], arrays["valid"])


@pytest.mark.parametrize("bad_index", [True, np.bool_(True), -1, 1., "1"])
def test_original_frame_index_strict(bad_index):
    with pytest.raises(ValueError, match="frame_index"):
        validate_camera_pointmap(*scene(), frame_index=bad_index)


@pytest.mark.parametrize("kind", ["hwc", "empty", "bool"])
def test_pointmap_shape_and_real_dtype_strict(kind):
    points, k, valid = scene()
    points = points.transpose(1, 2, 0) if kind == "hwc" else points[:, :0] if kind == "empty" else points.astype(bool)
    with pytest.raises(ValueError):
        validate_camera_pointmap(points, k, valid)


@pytest.mark.parametrize("kind", ["empty", "integer", "wrong_shape"])
def test_explicit_nonempty_boolean_support(kind):
    points, k, valid = scene()
    valid = np.zeros_like(valid) if kind == "empty" else valid.astype(int) if kind == "integer" else valid[:-1]
    with pytest.raises(ValueError, match="boolean"):
        validate_camera_pointmap(points, k, valid)


@pytest.mark.parametrize("entry,value", [((0, 0), 0), ((1, 1), -1), ((2, 2), 2), ((0, 1), .1), ((0, 0), np.nan)])
def test_intrinsics_not_silently_fixed(entry, value):
    points, k, valid = scene()
    k[entry] = value
    with pytest.raises(ValueError, match="intrinsics"):
        validate_camera_pointmap(points, k, valid)


@pytest.mark.parametrize("bad", [True, -1., np.nan, np.inf, "0"])
def test_reprojection_tolerance_strict(bad):
    with pytest.raises(ValueError, match="tolerance"):
        validate_camera_pointmap(*scene(), reprojection_tolerance_px=bad)


@pytest.mark.parametrize("scale", [2., np.array([2., 3., 4.])])
def test_ssi_roundtrip_uses_supplied_scale_shift_not_extra_geometry(scale):
    points, _, valid = scene()
    shift = np.array([.1, -.2, 2.])
    normalized = (points - shift[:, None, None]) / np.broadcast_to(scale, (3,))[:, None, None]
    report = validate_ssi_roundtrip(points, normalized, scale, shift, valid)
    assert report["max_roundtrip_error_m"] < 1e-14
    assert not report["native_normalization_estimator_verified"]


def test_ssi_roundtrip_handles_negative_normalized_z_and_explicit_invalid_pixels():
    points, _, valid = scene()
    points[:, 0, 0] = np.nan
    valid[0, 0] = False
    shift = np.array([0., 0., 3.])
    normalized = (points - shift[:, None, None]) / 2
    assert (normalized[2, valid] < 0).all()
    validate_ssi_roundtrip(points, normalized, 2., shift, valid)


@pytest.mark.parametrize("scale", [0., -1., np.nan, [1., np.inf, 1.], [1., 2.], True])
def test_ssi_invalid_scale_rejected(scale):
    points, _, valid = scene()
    with pytest.raises(ValueError):
        validate_ssi_roundtrip(points, points, scale, [0., 0., 0.], valid)


@pytest.mark.parametrize("bad", ["shift", "normalized", "extra_scale", "rotation"])
def test_ssi_contradictions_reject_without_repair(bad):
    points, _, valid = scene()
    normalized, shift, scale = points.copy(), np.zeros(3), 1.
    if bad == "shift":
        shift[0] = np.nan
    elif bad == "normalized":
        normalized[0, 0, 0] = np.nan
    elif bad == "extra_scale":
        scale = 2.
    else:
        normalized[:2] *= -1
    with pytest.raises(ValueError):
        validate_ssi_roundtrip(points, normalized, scale, shift, valid)


def registered_arrays():
    points, _, _ = scene()
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    mask = np.indices((4, 5))[1] >= 2
    pixels = np.array([[[1, 1], [1, 1], [1, 3]], [[2, 1], [2, 1], [2, 3]]])
    y, x = pixels[..., 0], pixels[..., 1]
    return [rgb, mask, points, rgb[y, x], mask[y, x], points[:, y, x], pixels]


def test_explicit_joint_crop_and_nearest_resize_registration():
    report = validate_joint_pixel_registration(*registered_arrays())
    assert report["target_pixels"] == 6
    assert report["joint_nearest_or_crop_registration_verified"]
    assert not report["native_preprocessor_verified"]


@pytest.mark.parametrize("field", [3, 4, 5])
def test_independently_shifted_rgb_mask_pointmap_rejected(field):
    arrays = registered_arrays()
    arrays[field] = np.flip(arrays[field], axis=-1 if field == 5 else 1)
    with pytest.raises(ValueError, match="registration|differently"):
        validate_joint_pixel_registration(*arrays)


def test_same_shape_canonical_point_warp_is_not_image_registration():
    arrays = registered_arrays()
    arrays[5] += np.array([.1, .2, .3])[:, None, None]
    with pytest.raises(ValueError, match="warped"):
        validate_joint_pixel_registration(*arrays)


@pytest.mark.parametrize("field", list(range(7)))
def test_joint_registration_masked_input_rejected(field):
    arrays = registered_arrays()
    arrays[field] = np.ma.array(arrays[field], mask=False)
    with pytest.raises(ValueError, match="masked"):
        validate_joint_pixel_registration(*arrays)


@pytest.mark.parametrize("bad", ["float", "negative", "out_of_bounds", "shape"])
def test_source_pixel_mapping_strict(bad):
    arrays = registered_arrays()
    if bad == "float":
        arrays[-1] = arrays[-1].astype(float)
    elif bad == "shape":
        arrays[-1] = arrays[-1][:-1]
    else:
        arrays[-1][0, 0, 0] = -1 if bad == "negative" else 4
    with pytest.raises(ValueError, match="pixel"):
        validate_joint_pixel_registration(*arrays)


def test_registration_invalid_points_are_not_filled_or_changed():
    arrays = registered_arrays()
    arrays[2][:, 1, 1] = np.nan
    y, x = arrays[-1][..., 0], arrays[-1][..., 1]
    arrays[5] = arrays[2][:, y, x]
    validate_joint_pixel_registration(*arrays)


def test_no_input_mutation():
    points, k, valid = scene()
    before = points.copy(), k.copy(), valid.copy()
    camera = validate_camera_pointmap(points, k, valid)
    to_pytorch3d_once(camera)
    validate_ssi_roundtrip(points, points, 1., np.zeros(3), valid)
    for original, saved in zip((points, k, valid), before):
        np.testing.assert_array_equal(original, saved)
