"""Tiny generated camera scenes only; no local challenge images or torch."""

import json

import numpy as np
import pytest

from world_reward.pointmap import validate_camera_pointmap


def fixture(height=5, width=7, *, K=None, dtype=np.float64):
    if K is None:
        K = np.array([[9., 0., width / 2], [0., 9., height / 2], [0., 0., 1.]])
    K = np.asarray(K, dtype=dtype)
    yy, xx = np.mgrid[:height, :width]
    depth = (2. + .03 * yy + .07 * xx).astype(dtype)
    points = np.stack(((xx + .5 - K[0, 2]) * depth / K[0, 0],
                       (yy + .5 - K[1, 2]) * depth / K[1, 1], depth), axis=-1).astype(dtype)
    return dict(depth=depth, points=points, validity=np.ones((height, width), dtype=bool),
                intrinsics_normalized=np.diag([1 / width, 1 / height, 1.]) @ K,
                expected_K=K)


def test_centered_camera_passes_and_diagnostics_are_small_json():
    scene = fixture()
    result = validate_camera_pointmap(**scene)
    assert result["status"] == "pass" and result["valid_pixels"] == 35
    assert result["excluded_pixels"] == 0 and result["reprojection_max_error_pixels"] < 1e-12
    assert result["metric_scale_accuracy_verified"] is False
    json.dumps(result, allow_nan=False)


def test_anisotropic_offcenter_camera_is_preserved():
    scene = fixture(K=[[13., 0., 2.3], [0., 6., 1.7], [0., 0., 1.]])
    result = validate_camera_pointmap(**scene)
    np.testing.assert_allclose(result["pixel_K"], scene["expected_K"])
    assert result["reprojection_max_error_pixels"] < 1e-12


def test_float32_geometry_and_fov_roundoff_pass():
    scene = fixture(dtype=np.float32)
    scene["intrinsics_normalized"] = scene["intrinsics_normalized"].astype(np.float32)
    scene["expected_K"] = scene["expected_K"].astype(np.float64)
    scene["expected_K"][0, 0] += .0005
    result = validate_camera_pointmap(**scene)
    assert result["intrinsics_max_absolute_error_pixels"] < .001


def test_validity_explicitly_excludes_invalids_without_changing_inputs():
    scene = fixture()
    scene["validity"][0, :3] = False
    scene["depth"][0, :3] = [np.inf, np.nan, -1]
    scene["points"][0, :3] = [[np.inf, np.inf, np.inf], [np.nan, np.nan, np.nan], [0, 0, -1]]
    copies = {key: np.array(value, copy=True) for key, value in scene.items()}
    result = validate_camera_pointmap(**scene)
    assert result["excluded_pixels"] == 3 and result["valid_pixels"] == 32
    for key, value in copies.items():
        np.testing.assert_array_equal(scene[key], value)


@pytest.mark.parametrize("key,value", [("depth", np.array([])), ("depth", np.ones((1, 5, 7))),
                                      ("points", np.ones((5, 7, 2))), ("points", np.ones((7, 5, 3))),
                                      ("validity", np.ones((7, 5), dtype=bool)),
                                      ("validity", np.ones((5, 7), dtype=np.uint8)),
                                      ("depth", np.ones((5, 7), dtype=bool)),
                                      ("points", np.ones((5, 7, 3), dtype=complex)),
                                      ("intrinsics_normalized", np.eye(4)), ("expected_K", np.eye(4))])
def test_shapes_and_dtypes_are_strict(key, value):
    scene = fixture()
    scene[key] = value
    with pytest.raises(ValueError):
        validate_camera_pointmap(**scene)


@pytest.mark.parametrize("key", ["depth", "points", "validity", "intrinsics_normalized", "expected_K"])
def test_masked_arrays_do_not_hide_validity(key):
    scene = fixture()
    scene[key] = np.ma.array(scene[key], mask=False)
    with pytest.raises(ValueError, match="masked arrays"):
        validate_camera_pointmap(**scene)


@pytest.mark.parametrize("target,entry,value", [("depth", (0, 0), np.nan), ("depth", (0, 0), np.inf),
                                               ("depth", (0, 0), 0), ("depth", (0, 0), -1),
                                               ("points", (0, 0, 0), np.nan),
                                               ("points", (0, 0, 2), 0), ("points", (0, 0, 2), -1)])
def test_invalid_declared_valid_record_is_not_silently_filtered(target, entry, value):
    scene = fixture()
    scene[target][entry] = value
    with pytest.raises(ValueError, match="validity=True"):
        validate_camera_pointmap(**scene)


def test_empty_support_is_not_a_vacuous_pass():
    scene = fixture()
    scene["validity"][:] = False
    with pytest.raises(ValueError, match="no explicitly valid"):
        validate_camera_pointmap(**scene)


def test_euclidean_ray_range_is_not_linear_camera_z():
    scene = fixture()
    scene["depth"] = np.linalg.norm(scene["points"], axis=-1)
    with pytest.raises(ValueError, match="camera Z differs"):
        validate_camera_pointmap(**scene)


def test_point_and_depth_z_are_checked_independently_of_pinhole_projection():
    scene = fixture()
    scene["depth"] += .01
    with pytest.raises(ValueError, match="camera Z differs"):
        validate_camera_pointmap(**scene)


def test_integer_pixel_coordinates_fail_half_pixel_contract():
    scene = fixture()
    scene["points"][..., 0] -= .5 * scene["depth"] / scene["expected_K"][0, 0]
    scene["points"][..., 1] -= .5 * scene["depth"] / scene["expected_K"][1, 1]
    with pytest.raises(ValueError, match="pixel centres"):
        validate_camera_pointmap(**scene)


def test_wrong_xy_axis_directions_fail():
    scene = fixture()
    scene["points"][..., :2] *= -1
    with pytest.raises(ValueError, match="reprojection"):
        validate_camera_pointmap(**scene)


def test_pixel_K_passed_as_normalized_K_fails_instead_of_guessing_units():
    scene = fixture()
    scene["intrinsics_normalized"] = scene["expected_K"].copy()
    with pytest.raises(ValueError, match="expected pixel camera K"):
        validate_camera_pointmap(**scene)


@pytest.mark.parametrize("key", ["intrinsics_normalized", "expected_K"])
@pytest.mark.parametrize("entry,value", [((0, 0), 0), ((1, 1), -1), ((0, 1), .01),
                                        ((1, 0), .01), ((2, 2), 2), ((0, 2), np.inf)])
def test_unsupported_intrinsics_are_not_fit_or_repaired(key, entry, value):
    scene = fixture()
    scene[key][entry] = value
    with pytest.raises(ValueError, match="intrinsics"):
        validate_camera_pointmap(**scene)


def test_expected_camera_focal_mismatch_fails():
    scene = fixture()
    scene["expected_K"][0, 0] += .02
    with pytest.raises(ValueError, match="expected pixel camera K"):
        validate_camera_pointmap(**scene)


def test_reprojection_pixel_tolerance_is_a_radial_bound():
    scene = fixture()
    scene["points"][..., 0] += .0006 * scene["depth"] / scene["expected_K"][0, 0]
    scene["points"][..., 1] += .0006 * scene["depth"] / scene["expected_K"][1, 1]
    assert validate_camera_pointmap(**scene)["reprojection_max_error_pixels"] < .001
    scene["points"][..., 0] += .0002 * scene["depth"] / scene["expected_K"][0, 0]
    scene["points"][..., 1] += .0002 * scene["depth"] / scene["expected_K"][1, 1]
    with pytest.raises(ValueError, match="pixel centres"):
        validate_camera_pointmap(**scene)


def test_pointmap_with_other_positive_unit_scale_remains_consistent_not_metric_proof():
    scene = fixture()
    scene["points"] *= 100
    scene["depth"] *= 100
    assert validate_camera_pointmap(**scene)["metric_scale_accuracy_verified"] is False
