"""CPU-only pinhole/contracts; CUDA raster proof is the separate remote smoke."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_renderer():
    path = Path(__file__).resolve().parents[1] / "infra/camera_render.py"
    spec = importlib.util.spec_from_file_location("world_reward_camera_render_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()
K = np.array([[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]])
TRIANGLE = np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.]])
FACES = np.array([[0, 1, 2]])


def test_original_resolution_pinhole_has_correct_axes_and_no_aspect_stretch():
    points = [[0, 0, 2], [.1, .1, 2], [1, -.25, 4]]
    np.testing.assert_allclose(renderer.project_camera_points(points, K),
                               [[768, 576], [864, 672], [1248, 456]])


def test_off_center_anisotropic_intrinsics_are_not_replaced_by_fov_defaults():
    matrix = np.array([[300., 0, 103.], [0, 170., 57.], [0, 0, 1.]])
    np.testing.assert_allclose(renderer.project_camera_points([[2., -3., 5.]], matrix), [[223., -45.]])


def test_projection_is_scale_invariant_without_changing_camera_z_contract():
    original, scaled = TRIANGLE.copy(), TRIANGLE * 100
    np.testing.assert_array_equal(renderer.project_camera_points(original, K),
                                  renderer.project_camera_points(scaled, K))
    assert not np.array_equal(original[:, 2], scaled[:, 2])


def test_synthetic_tilted_triangle_reference_has_perspective_correct_camera_depth():
    width, height, matrix, uv, triangle, rays, depth, bary = renderer._smoke_reference()
    np.testing.assert_allclose(renderer.project_camera_points(triangle, matrix), uv, atol=1e-12)
    interior = (bary > 1e-4).all(axis=-1)
    assert (width, height) == (128, 96) and interior.sum() > 2000
    np.testing.assert_allclose(bary.sum(axis=-1), 1, atol=1e-12)
    # Correct camera Z is harmonic in screen-space barycentrics, not affine Z.
    harmonic_depth = 1 / np.sum(bary[interior] / triangle[:, 2], axis=-1)
    np.testing.assert_allclose(depth[interior], harmonic_depth, atol=1e-12)
    wrong_affine_depth = bary[interior] @ triangle[:, 2]
    assert np.max(np.abs(depth[interior] - wrong_affine_depth)) > .1
    assert np.max(depth[interior] * (np.linalg.norm(rays[interior], axis=-1) - 1)) > .01


@pytest.mark.parametrize("points", [[], [0, 0, 2], np.zeros((1, 4)),
                                   [[0, 0, 0]], [[0, 0, -1]], [[0, np.nan, 2]],
                                   [[0, 0, np.inf]], [[True, True, True]],
                                   [[0j, 0j, 2j]], [["0", "0", "2"]]])
def test_projection_invalid_points_fail_without_dropping_samples(points):
    with pytest.raises(ValueError):
        renderer.project_camera_points(points, K)


@pytest.mark.parametrize("matrix", [np.eye(4), np.ones((3, 3)),
                                   [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   [[1, .1, 0], [0, 1, 0], [0, 0, 1]],
                                   [[1, 0, 0], [.1, 1, 0], [0, 0, 1]],
                                   [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
                                   [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]])
def test_unsupported_or_nonfinite_intrinsics_fail(matrix):
    with pytest.raises(ValueError):
        renderer.project_camera_points(TRIANGLE, matrix)


def test_projection_overflow_fails_explicitly():
    with pytest.raises(ValueError, match="overflowed"):
        renderer.project_camera_points([[1e300, 0, 1e-300]], K)


def test_mesh_contract_is_independent_of_torch_and_does_not_modify_inputs():
    vertices, matrix, faces = TRIANGLE.copy(), K.copy(), FACES.copy()
    output = renderer._mesh_inputs(vertices, faces, matrix, 1536, 1152, 1e-4)
    assert output[0].dtype == np.float32 and output[1].dtype == np.int64 and output[2].dtype == np.float32
    np.testing.assert_array_equal(vertices, TRIANGLE)
    np.testing.assert_array_equal(matrix, K)
    np.testing.assert_array_equal(faces, FACES)


@pytest.mark.parametrize("faces", [[], [[0, 1]], [[0, 1, 3]], [[0, 1, -1]],
                                  [[0, 0, 1]], [[0., 1., 2.]], [[False, True, True]]])
def test_mesh_rejects_invalid_faces_before_importing_cuda(faces):
    with pytest.raises(ValueError):
        renderer.raster_camera_mesh(TRIANGLE, faces, K)


@pytest.mark.parametrize("width,height", [(0, 2), (2, -1), (2.5, 3), (True, 3), (3, False)])
def test_image_size_never_implicitly_casts_or_resizes(width, height):
    with pytest.raises(ValueError):
        renderer.raster_camera_mesh(TRIANGLE, FACES, K, width, height)


@pytest.mark.parametrize("near", [0., -1., np.nan, np.inf, True, "0.01", [0.01]])
def test_near_plane_contract(near):
    with pytest.raises(ValueError):
        renderer.raster_camera_mesh(TRIANGLE, FACES, K, near_clip_m=near)


def test_near_crossing_is_not_silently_clipped_or_deleted():
    vertices = TRIANGLE.copy()
    vertices[0, 2] = 1e-5
    with pytest.raises(ValueError, match="near plane"):
        renderer.raster_camera_mesh(vertices, FACES, K)


def test_zero_area_triangle_is_not_silently_deleted():
    with pytest.raises(ValueError, match="zero-area"):
        renderer.raster_camera_mesh([[0, 0, 2], [1, 0, 2], [2, 0, 2]], FACES, K)


def test_float32_coordinate_overflow_is_explicit():
    with pytest.raises(ValueError, match="float32"):
        renderer.raster_camera_mesh(TRIANGLE, FACES, K * np.array([[1e100, 1, 1], [1, 1, 1], [1, 1, 1]]))


def test_float32_triangle_collapse_is_explicit():
    points = np.array([[1., 0, 2], [1. + 1e-10, 0, 2], [1., 1e-10, 2]])
    with pytest.raises(ValueError, match="collapse"):
        renderer.raster_camera_mesh(points, FACES, K)


def test_local_render_is_forbidden_before_loading_torch(monkeypatch):
    monkeypatch.setattr(renderer.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Azure Linux CUDA"):
        renderer.raster_camera_mesh(TRIANGLE, FACES, K)


def test_mask_iou_measures_image_evidence_not_missing_object_reward():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [True, False]])
    assert renderer.silhouette_iou(a, b) == pytest.approx(1 / 3)
    assert renderer.silhouette_iou(a, a) == 1
    assert renderer.silhouette_iou(a, ~a) == 0
    assert renderer.silhouette_iou(a, np.zeros_like(a)) == 0


@pytest.mark.parametrize("a,b", [(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)),
                                (np.ones((2, 2)), np.ones((2, 2))),
                                (np.ones((2, 2), dtype=bool), np.ones((2, 3), dtype=bool)),
                                (np.ones((2,), dtype=bool), np.ones((2,), dtype=bool)),
                                (np.zeros((0, 2), dtype=bool), np.zeros((0, 2), dtype=bool))])
def test_mask_iou_invalid_or_empty_unions_fail(a, b):
    with pytest.raises(ValueError):
        renderer.silhouette_iou(a, b)
