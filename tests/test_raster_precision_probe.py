"""Tiny independent projection references; never GPU, media or holdout inputs."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def probe():
    path = Path(__file__).parents[1]/"infra/raster_precision_probe.py"
    spec = importlib.util.spec_from_file_location("precision_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_exact_new_four_faces_no_mesh_or_failed_cohort_dependency(probe):
    vertices, faces = probe.fixture()
    assert vertices.shape == (10, 3) and vertices.dtype == np.float64
    assert faces.shape == (4, 3) and faces.dtype == np.int64
    assert np.isfinite(vertices).all() and vertices[:, 2].min() == 1.5
    np.testing.assert_array_equal(vertices[6:, 2], 4.5)
    wide = probe.face_geometry(vertices[faces[0]])
    grazing = probe.face_geometry(vertices[faces[1]])
    assert wide["projected_double_area_pixels2"] == pytest.approx(43000)
    assert grazing["projected_double_area_pixels2"] == pytest.approx(.064)
    assert probe.LABELS == ("wide", "grazing", "background_0", "background_1")


def test_plane_reference_pixel_centres_camera_z_and_full_statistics(probe):
    triangle = np.array([[-5., -5., 3.], [-5., 5., 3.], [5., -5., 3.]])
    xy = np.array([[0, 0], [40, 40], [100, 100]])
    result = probe.reference_summary(triangle, xy, np.full(3, 3., np.float32), .5)
    assert result["pixels"] == result["front_pixels"] == 3
    assert result["max_camera_z_error_m"] == result["p99_camera_z_error_m"] == 0
    assert result["count_error_gt_2e_minus5_m"] == result["back_or_tangent_pixels"] == 0
    assert result["minimum_barycentric"] > 0 and 0 < result["minimum_abs_normal_ray_cos"] <= 1
    bad = probe.reference_summary(triangle, xy, np.full(3, 3.001, np.float32), .5)
    assert bad["count_error_gt_2e_minus5_m"] == 3
    with pytest.raises(ValueError): probe.reference_summary(triangle, xy, np.zeros(3), .25)


def test_new_grazing_plane_fp32_projection_can_exceed_gate_without_rescuing_anything(probe):
    vertices, faces = probe.fixture()
    ndc = np.column_stack((-vertices.astype(np.float32)[:, :2]/vertices.astype(np.float32)[:, 2, None]*np.float32(800/240),
                           vertices.astype(np.float32)[:, 2])).astype(np.float32)
    recovered = probe.recovered_projection(ndc)
    p = np.array([[500, 200]]); triangle = vertices[faces[1]]
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    ray = np.r_[(p[0]+.5-[320, 240])/800, 1.]
    original = (normal @ triangle[0])/(normal @ ray)
    result = probe.reference_summary(recovered[faces[1]], p, np.array([original]), .5)
    assert result["max_camera_z_error_m"] > .01
    assert result["count_error_gt_2e_minus5_m"] == 1
    with pytest.raises(ValueError): probe.recovered_projection(ndc.astype(np.float64))


def test_tilted_wide_reference_identifies_half_pixel_offset(probe):
    vertices, faces = probe.fixture(); triangle = vertices[faces[0]]; xy = np.array([[320, 200]])
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    ray = np.r_[(xy[0]+.5-[320, 240])/800, 1.]
    expected = np.array([(normal @ triangle[0])/(normal @ ray)])
    assert probe.reference_summary(triangle, xy, expected, .5)["max_camera_z_error_m"] < 1e-14
    assert probe.reference_summary(triangle, xy, expected, 0.)["max_camera_z_error_m"] > .001
    assert probe.reference_summary(triangle, xy, expected, 1.)["max_camera_z_error_m"] > .001


def test_scoped_bounded_wrapper_and_scalar_only_report(probe):
    source = Path(probe.__file__).read_text(); wrapper_path = Path(probe.__file__).with_name("run_raster_precision_probe.sh")
    wrapper = wrapper_path.read_text()
    assert not any(t in source+wrapper for t in ("authored_rgbd", "tudl", "np.save", "PIL", "sam3", "mhr"))
    assert "signal.alarm(30)" in source and 'receipt.chmod(0o400)' in source
    assert "33s docker run" in wrapper and "--kill-after=10s" in wrapper
    assert '--name "$NAME"' in wrapper and 'docker stop --time 2 "$NAME"' in wrapper and 'docker kill "$NAME"' in wrapper
    assert '--network none' in wrapper and '--memory 4g --cpus 2' in wrapper
    assert wrapper.count('--mount ') == 4
    assert not any(f"src=$ROOT/{folder}" in wrapper for folder in ("weights", "data", "vendor", "outputs", "results", "validation"))
    assert subprocess.run(["bash", "-n", str(wrapper_path)], capture_output=True).returncode == 0
    assert subprocess.run(["bash", str(wrapper_path), "--unknown"], capture_output=True).returncode == 2
