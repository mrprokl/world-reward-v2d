"""Actual NumPy contracts/source planning only; shader/autograd run on Azure."""
import ast
import importlib.util
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_soft_probe", infra/"soft_silhouette_probe.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_frozen_declaration_and_native_blur_formula(probe):
    assert (probe.WIDTH, probe.HEIGHT, probe.SOURCE_WIDTH, probe.SOURCE_HEIGHT) == (256, 192, 1024, 768)
    assert probe.SIGMA == probe.GAMMA == 1e-4 and probe.FACES_PER_PIXEL == 8
    assert probe.BLUR_RADIUS == math.log(1./1e-4-1.)*1e-4
    assert probe.STEP_M == .0005 and probe.BUDGET == 180 and probe.PROJECTION_ATOL_PX == 1e-4
    assert probe.TARGET_TRANSLATION_M == (.008, -.006, .014)


def test_cube_closed_outward_positive_geometry_and_frozen_nonmutation(probe):
    vertices, faces, source_K, K = probe.fixture()
    assert vertices.shape == (8, 3) and vertices.dtype == np.float32 and faces.shape == (12, 3)
    assert faces.dtype == np.int64 and np.isfinite(vertices).all() and (vertices[:, 2] > 0).all()
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edgepairs, count = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    assert len(edgepairs) == 18 and (count == 2).all() and len(vertices)-len(edgepairs)+len(faces) == 2
    triangles = vertices[faces].astype(np.float64)
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    assert (np.sum(normal*(triangles.mean(axis=1)-vertices.mean(axis=0)), axis=1) > 0).all()
    assert np.ptp(vertices, axis=0) == pytest.approx([.62, .48, .38], abs=2e-7)
    old = vertices.copy(); probe.camera._mesh_inputs(vertices, faces, K, 256, 192, 1e-4)
    assert np.array_equal(vertices, old)


def test_lowresolution_intrinsics_preserve_edge_pixel_centres_not_index_resize(probe):
    vertices, faces, source_K, K = probe.fixture()
    before = source_K.copy()
    assert np.array_equal(K, np.diag([.25, .25, 1.])@source_K)
    assert np.array_equal(K, [[240., 0., 128.], [0., 260., 96.], [0., 0., 1.]])
    full_pixels = probe.camera.project_camera_points(vertices, source_K)
    low_pixels = probe.camera.project_camera_points(vertices, K)
    np.testing.assert_allclose(low_pixels, full_pixels*.25, atol=1e-12, rtol=0)
    # Lowres cell centre .5 corresponds to source cell edge coordinate2,
    # not source index0 shifted by an invented .5 calibration change.
    low_ray = (np.array([.5, .5])-K[:2, 2])/[K[0, 0], K[1, 1]]
    high_ray = (np.array([2., 2.])-source_K[:2, 2])/[source_K[0, 0], source_K[1, 1]]
    np.testing.assert_allclose(low_ray, high_ray, atol=1e-15, rtol=0)
    assert np.array_equal(before, source_K)


@pytest.mark.parametrize("size", [0, -1, 2., True, np.int64(8)])
def test_scaled_intrinsics_never_implicitly_cast_invalid_grids(probe, size):
    with pytest.raises(ValueError): probe.scaled_intrinsics(np.eye(3), 1024, 768, size, 192)


@pytest.mark.parametrize("matrix", [np.eye(4), np.zeros((3, 3)), np.full((3, 3), np.nan),
    [[1, .1, 0], [0, 1, 0], [0, 0, 1]], np.ma.array(np.eye(3), mask=False)])
def test_scaled_intrinsics_rejects_skew_or_hidden_calibration(probe, matrix):
    with pytest.raises(ValueError): probe.scaled_intrinsics(matrix, 1024, 768, 256, 192)


def test_fixed_negative_gradient_step_no_adaptive_search_and_no_scale(probe):
    gradient = np.array([3., -4., 2.], np.float64); old = gradient.copy()
    step = probe.negative_gradient_step(gradient)
    assert np.linalg.norm(step) == pytest.approx(.0005, abs=1e-15)
    assert np.dot(step, gradient) < 0 and np.array_equal(gradient, old)
    np.testing.assert_allclose(step, -gradient/np.linalg.norm(gradient)*.0005, atol=1e-15, rtol=0)
    probe.validate_measurement(.02, .019, gradient, step)
    vertices, faces, _, _ = probe.fixture()
    moved = vertices.astype(np.float64)+step
    np.testing.assert_allclose(np.ptp(moved, axis=0), np.ptp(vertices.astype(np.float64), axis=0), atol=1e-15, rtol=0)
    # Positive camera Z and triangles are unchanged under rigid translation.
    assert (moved[:, 2] > 0).all() and np.array_equal(faces, probe.fixture()[1])


@pytest.mark.parametrize("gradient", [np.zeros(3), np.array([1., 0., 2.]), np.array([1., 2., np.nan]),
    np.array([1., 2., np.inf]), np.ones(2), np.ones((1, 3)), np.ones(3, np.int64),
    np.ma.array(np.ones(3), mask=False), [1., 2., 3.]])
def test_requires_all_three_finite_nonzero_physical_derivatives(probe, gradient):
    with pytest.raises(ValueError): probe.negative_gradient_step(gradient)


@pytest.mark.parametrize("fault", ["equal", "increase", "negative", "nan", "zero", "reverse", "larger", "shape", "integerloss"])
def test_one_step_measurement_fail_closed(probe, fault):
    gradient = np.array([1., 2., 3.]); step = probe.negative_gradient_step(gradient)
    before, after = .02, .019
    if fault == "equal": after = before
    elif fault == "increase": after = .03
    elif fault == "negative": after = -.1
    elif fault == "nan": after = np.nan
    elif fault == "zero": before = 0.
    elif fault == "reverse": step *= -1
    elif fault == "larger": step *= 2
    elif fault == "shape": step = step[None]
    else: before = 1
    with pytest.raises(ValueError): probe.validate_measurement(before, after, gradient, step)


def test_actual_shader_and_autograd_source_no_numpy_detach_or_mock_in_soft_path(probe):
    tree = ast.parse(Path(probe.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    alpha = next(n for n in ast.walk(run) if isinstance(n, ast.FunctionDef) and n.name == "alpha")
    alpha_source = ast.unparse(alpha); run_source = ast.unparse(run)
    assert "mesh = Meshes(verts=[base + translation[None]], faces=[triangles])" in alpha_source
    assert "fragments = rasterizer(mesh)" in alpha_source and "shader(fragments, mesh)" in alpha_source
    assert not any(s in alpha_source for s in ("detach", "numpy", "inference_mode", "no_grad", "as_tensor"))
    assert "SoftSilhouetteShader" in run_source and "loss.backward()" in run_source and "requires_grad=True" in run_source
    assert "camera._opencv_camera(torch, K, WIDTH, HEIGHT)" in run_source
    assert "camera.raster_camera_mesh(vertices, faces, K, WIDTH, HEIGHT)" in run_source
    assert run_source.index("target_frozen_before_optimization=True") < run_source.index("loss.backward()")
    assert "!= 5" in run_source and "!= 1" in run_source
    # This source inspection proves wiring only, NOT an executed CUDA gradient.


def test_wrapper_offline_exclusive_datafree_cpu_gpu_budget(probe):
    path = Path(probe.__file__).with_name("run_soft_silhouette_probe.sh")
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert probe.IMAGE in text and "--network none" in text and "--memory 16g --cpus 4" in text and "183s" in text
    assert "--env CUBLAS_WORKSPACE_CONFIG=:4096:8" in text and "! -e \"$OUT\"" in text and "(( $# == 0 ))" in text
    mounts = [s for s in text.splitlines() if '--mount "' in s]
    assert len(mounts) == 2 and "src=$CODE,dst=$CODE,readonly" in mounts[0] and "src=$OUT,dst=$OUT" in mounts[1]
    assert all(f"src=$ROOT/{name}" not in text for name in ("data", "weights", "validation", "vendor", "outputs"))


def test_cli_no_tuning_or_challenge_controls_and_cublas_before_torch(probe, monkeypatch):
    for argv in (["--episode", "0"], ["--step", "1"], ["--sigma", ".1"], ["--max-seconds", "999"]):
        with pytest.raises(SystemExit) as exc: probe.main(argv)
        assert exc.value.code == 2
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="before torch"): probe.strict_torch()


def test_regular_source_does_not_accept_symlink(probe, tmp_path):
    path = tmp_path/"source.py"; path.write_text("# source")
    assert probe.regular(path) == path
    link = tmp_path/"link.py"; link.symlink_to(path)
    with pytest.raises(ValueError): probe.regular(link)
    with pytest.raises(ValueError): probe.regular(tmp_path)
