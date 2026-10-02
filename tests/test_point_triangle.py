"""NumPy contracts and analytic fixtures only; CUDA proof is the remote gate."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from world_reward.point_triangle import validate_numpy_point_triangles


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "infra/point_triangle_gate.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_point_triangle_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiny_triangle_retained_as_surface_not_default_area_threshold_lines(gate):
    points, triangles, distances, gradient = gate.analytic_fixture()
    actual_points, actual_triangles = validate_numpy_point_triangles(points, triangles)
    np.testing.assert_array_equal(points, actual_points)
    np.testing.assert_array_equal(triangles, actual_triangles)
    assert not np.shares_memory(points, actual_points) and not np.shares_memory(triangles, actual_triangles)
    area = np.linalg.norm(np.cross(triangles[0, 1] - triangles[0, 0], triangles[0, 2] - triangles[0, 0])) / 2
    assert area == pytest.approx(5e-5) and area < 5e-3
    np.testing.assert_allclose(distances, [.0004, .0004, .000441, .0004125], rtol=1e-7)
    np.testing.assert_allclose(gradient, [[0., 0., .04], [0., 0., .04], [-.01, -.008, .04],
                                        [.005, .005, .04]], rtol=1e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_guard_accepts_float32_and_safe_float64_without_mutation(gate, dtype):
    points, triangles, _, _ = gate.analytic_fixture()
    points, triangles = points.astype(dtype), triangles.astype(dtype)
    old_points, old_triangles = points.copy(), triangles.copy()
    result = validate_numpy_point_triangles(points, triangles)
    assert all(value.dtype == np.float32 for value in result)
    np.testing.assert_array_equal(points, old_points)
    np.testing.assert_array_equal(triangles, old_triangles)


@pytest.mark.parametrize("value", [np.empty((0, 3)), np.zeros(3), np.zeros((2, 2)),
                                    np.ones((1, 3), bool), np.ones((1, 3), int),
                                    np.ones((1, 3), complex), np.array([[0., np.nan, 1.]]),
                                    np.array([[0., np.inf, 1.]]), np.array([[1e100, 0., 1.]]),
                                    np.ma.array(np.ones((1, 3)), mask=False)])
def test_invalid_observed_points_fail_before_any_torch_import(gate, value):
    _, triangles, _, _ = gate.analytic_fixture()
    with pytest.raises(ValueError):
        validate_numpy_point_triangles(value, triangles)


@pytest.mark.parametrize("value", [np.empty((0, 3, 3)), np.zeros((3, 3)), np.zeros((1, 3, 3)),
                                    np.ones((1, 3, 3), int), np.ones((1, 3, 3), bool),
                                    np.full((1, 3, 3), np.nan), np.full((1, 3, 3), np.inf),
                                    np.ma.array(np.ones((1, 3, 3)), mask=False)])
def test_invalid_or_degenerate_triangles_fail_without_approximation(gate, value):
    points, _, _, _ = gate.analytic_fixture()
    with pytest.raises(ValueError):
        validate_numpy_point_triangles(points, value)


def test_triangle_collapsing_in_float32_fails_not_silently_line(gate):
    points, _, _, _ = gate.analytic_fixture()
    tiny = np.array([[[1., 0., 0.], [1. + 1e-10, 0., 0.], [1., 1e-10, 0.]]])
    with pytest.raises(ValueError, match="positive area"):
        validate_numpy_point_triangles(points, tiny)


def test_mixed_valid_and_invalid_triangle_not_dropped(gate):
    points, triangles, _, _ = gate.analytic_fixture()
    with pytest.raises(ValueError, match="positive area"):
        validate_numpy_point_triangles(points, np.concatenate([triangles, np.zeros_like(triangles)]))


@pytest.mark.parametrize("xy", [(1e-5, 1e-5), (1e-5, .01)])
def test_positive_area_accepted_not_numerically_rescaled_to_hide_kernel_issue(gate, xy):
    points, _, _, _ = gate.analytic_fixture()
    triangle = np.array([[[0., 0., 0.], [xy[0], 0., 0.], [0., xy[1], 0.]]])
    assert np.linalg.norm(np.cross(triangle[0, 1], triangle[0, 2])) > 0
    _, validated = validate_numpy_point_triangles(points, triangle)
    np.testing.assert_array_equal(validated, triangle.astype(np.float32))


def test_outside_edge_fixture_catches_regularized_barycentric_false_interior(gate):
    points, triangles, distances, _ = gate.analytic_fixture()
    v0, v1, v2 = triangles[0].astype(float)
    p0, p1, p2 = v1 - v0, v2 - v0, points[-1].astype(float) - v0
    d00, d01, d11 = p0 @ p0, p0 @ p1, p1 @ p1
    d20, d21 = p2 @ p0, p2 @ p1
    determinant = d00 * d11 - d01**2
    true_bary = np.array([d11 * d20 - d01 * d21, d00 * d21 - d01 * d20]) / determinant
    regularized = true_bary * determinant / (determinant + 1e-8)
    assert true_bary.sum() > 1 and regularized.sum() < 1
    assert (regularized >= 0).all()
    assert distances[-1] - float(points[-1, 2])**2 == pytest.approx(1.25e-5, rel=1e-6)


def test_duplicates_and_singleton_are_valid_contracts(gate):
    points, triangles, _, _ = gate.analytic_fixture()
    p, t = validate_numpy_point_triangles(points[:1], np.repeat(triangles, 2, axis=0))
    assert p.shape == (1, 3) and t.shape == (2, 3, 3)


def test_analytic_point_and_triangle_gradient_finite_difference_independent_of_cuda(gate):
    points, triangles, _, gradients = gate.analytic_fixture()
    point, triangle = points[0].astype(float), triangles[0].astype(float)
    def plane_distance(p, tri):
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        return np.dot(p - tri[0], normal)**2 / np.dot(normal, normal)
    h = 1e-6
    for axis in range(3):
        offset = np.eye(3)[axis] * h
        fd = (plane_distance(point + offset, triangle) - plane_distance(point - offset, triangle)) / (2 * h)
        assert fd == pytest.approx(gradients[0, axis], abs=1e-10)
    expected = -np.array([.5, .25, .25])[:, None] * gradients[0]
    for vertex in range(3):
        for axis in range(3):
            offset = np.zeros((3, 3)); offset[vertex, axis] = h
            fd = (plane_distance(point, triangle + offset) - plane_distance(point, triangle - offset)) / (2 * h)
            assert fd == pytest.approx(expected[vertex, axis], abs=1e-8)


def test_frozen_report_fail_before_heavy_import_or_cuda(gate, monkeypatch, tmp_path):
    output = tmp_path / "results/point-triangle-gate.json"
    output.parent.mkdir()
    output.write_text("frozen")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([Path("/sys/class/net/lo")]) if str(p) == "/sys/class/net" else original(p))
    monkeypatch.setenv("WR_ROOT", str(tmp_path))
    monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    with pytest.raises(FileExistsError, match="frozen"):
        gate.main()
    assert output.read_text() == "frozen"


def test_wrapper_has_immutable_source_no_data_models_or_network(gate):
    wrapper = Path(gate.__file__).with_name("run_point_triangle_gate.sh").read_text()
    assert "--network none" in wrapper and "--gpus all" in wrapper
    assert "src=$CODE,dst=$CODE,readonly" in wrapper and "WR_CODE_REVISION" in wrapper
    assert "src=$ROOT/data" not in wrapper and "src=$ROOT/weights" not in wrapper
    assert "world-reward/cari4d-source:0.1" in wrapper
