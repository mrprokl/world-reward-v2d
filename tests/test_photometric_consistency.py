"""Data-free RGB/geometry policy tests, not an inference or quality experiment."""
import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from world_reward.photometric_consistency import GAMMAS, gamma_rgb, gamma_variants, geometric_medoid


def vertices(dtype=np.float64):
    value = np.zeros((3, 4, 3), dtype=dtype)
    value[:, :, 2] = 2.
    value[:, :, 0] = np.asarray([0., .1, 1.], dtype=dtype)[:, None]
    return value


def test_gamma_identity_exact_independent_copy_and_fixed_order():
    rgb = np.arange(256, dtype=np.uint8).reshape(8, 32, 1).repeat(3, axis=2)
    before = rgb.copy();results = gamma_variants(rgb)
    assert GAMMAS == (1., .8, 1.2) and len(results) == 3
    np.testing.assert_array_equal(results[0], rgb)
    assert all(v.shape == rgb.shape and v.dtype == np.uint8 for v in results)
    assert all(not np.shares_memory(v, rgb) for v in results)
    assert np.all(results[1] >= rgb) and np.all(results[2] <= rgb)
    results[0][0, 0, 0] = 100
    np.testing.assert_array_equal(rgb, before)
    assert results[1][0, 0, 0] == 0


@pytest.mark.parametrize("gamma", GAMMAS)
def test_full_lut_exact_rounding_no_channel_swap_or_resize(gamma):
    rgb = np.arange(256, dtype=np.uint8).reshape(8, 32, 1).repeat(3, axis=2)
    rgb[:, :, 1] = np.flip(rgb[:, :, 1], axis=1)
    expected = np.floor(255. * (rgb.astype(np.float64) / 255.) ** gamma + .5).astype(np.uint8)
    np.testing.assert_array_equal(gamma_rgb(rgb, gamma), expected)


@pytest.mark.parametrize("bad", [True, np.bool_(False), 0., -1., .9, np.nan, np.inf, "0.8", [1.]])
def test_no_unregistered_gamma_policy(bad):
    with pytest.raises(ValueError): gamma_rgb(np.zeros((2, 3, 3), np.uint8), bad)


@pytest.mark.parametrize("fault", ["float", "masked", "list", "gray", "rgba", "empty", "batch"])
def test_rgb_requires_explicit_original_uint8_grid(fault):
    rgb = np.zeros((2, 3, 3), np.uint8)
    if fault == "float": rgb = rgb.astype(np.float32)
    elif fault == "masked": rgb = np.ma.array(rgb, mask=False)
    elif fault == "list": rgb = rgb.tolist()
    elif fault == "gray": rgb = rgb[:, :, 0]
    elif fault == "rgba": rgb = np.zeros((2, 3, 4), np.uint8)
    elif fault == "empty": rgb = rgb[:0]
    else: rgb = rgb[None]
    with pytest.raises(ValueError): gamma_variants(rgb)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_medoid_existing_full_vertices_not_average_and_no_mutation(dtype):
    data = vertices(dtype);before = data.copy();result = geometric_medoid(data)
    assert result.index == 1
    expected = np.zeros((3, 3), np.float64)
    for i in range(3):
        for j in range(3): expected[i, j] = np.mean(np.sum((data[i].astype(np.float64)-data[j])**2, axis=1))
    np.testing.assert_array_equal(result.pairwise_squared_distances, expected)
    np.testing.assert_array_equal(result.scores, expected.sum(1))
    assert result.chosen_vertices.tobytes() == data[1].tobytes()
    assert result.chosen_vertices.dtype == data.dtype and not np.shares_memory(result.chosen_vertices, data)
    np.testing.assert_array_equal(data, before)
    for array in (result.chosen_vertices, result.scores, result.pairwise_squared_distances):
        with pytest.raises(ValueError): array.flat[0] = 10
    with pytest.raises(FrozenInstanceError): result.index = 0


def test_sham_identical_predictions_exact_zero_choose_original_gamma():
    data = np.repeat(vertices()[0:1], 3, axis=0);result = geometric_medoid(data)
    assert result.index == 0 and result.scores.tobytes() == np.zeros(3, np.float64).tobytes()
    assert not result.pairwise_squared_distances.any()
    assert result.chosen_vertices.tobytes() == data[0].tobytes()


def test_exact_f64_three_way_tie_uses_first_not_nearest_to_average():
    data = np.array([[[0., 0., 2.]], [[1., 1., 2.]], [[1., 0., 3.]]], np.float64)
    result = geometric_medoid(data)
    np.testing.assert_array_equal(result.scores, [4., 4., 4.])
    assert result.index == 0


def test_all_vertices_deformation_contributes_without_surface_subset():
    data = vertices();data[0] = data[1];data[2] = data[1];data[2, :, 0] -= 1.
    data[0, -1, 0] += 5.
    result = geometric_medoid(data)
    assert result.index == 1 and result.pairwise_squared_distances[0, 1] == 25. / 4.


def test_unit_scale_invariance_selection_scores_quadratic_and_f64_precision():
    data = vertices();result = geometric_medoid(data);scaled = geometric_medoid(data * 100.)
    assert result.index == scaled.index
    np.testing.assert_allclose(scaled.scores, result.scores * 10000., rtol=1e-14)
    tiny = vertices();tiny[:, :, 0] = np.array([0., 1e-10, 3e-10])[:, None]
    near = geometric_medoid(tiny)
    assert near.index == 1 and near.scores.min() > 0
    assert near.chosen_vertices.tobytes() == tiny[1].tobytes()


@pytest.mark.parametrize("fault", ["nan", "inf", "masked", "dtype", "list", "count", "shape", "empty", "zeroZ", "negativeZ", "overflow"])
def test_medoid_invalid_geometry_is_fatal_no_confidence_or_private_fallback(fault):
    data = vertices()
    if fault == "nan": data[0, 0, 0] = np.nan
    elif fault == "inf": data[0, 0, 0] = np.inf
    elif fault == "masked": data = np.ma.array(data, mask=False)
    elif fault == "dtype": data = data.astype(np.float16)
    elif fault == "list": data = data.tolist()
    elif fault == "count": data = data[:2]
    elif fault == "shape": data = data[:, :, :2]
    elif fault == "empty": data = data[:, :0]
    elif fault == "zeroZ": data[0, 0, 2] = 0
    elif fault == "negativeZ": data[0, 0, 2] = -1
    else: data[0, 0, 0] = 1e308
    with pytest.raises(ValueError): geometric_medoid(data)


def test_module_is_pure_numpy_policy_no_io_or_model_runtime():
    import world_reward.photometric_consistency as module
    tree = ast.parse(Path(module.__file__).read_text())
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imports == {"numpy"}
    assert not any(isinstance(n, ast.Name) and n.id in {"open", "torch", "confidence", "weights", "truth"} for n in ast.walk(tree))
