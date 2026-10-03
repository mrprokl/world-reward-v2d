"""Tiny CPU triangles only; no CUDA, renderer media, datasets or model assets."""
import importlib.util
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest


@pytest.fixture
def gate():
    p = Path(__file__).parents[1]/"infra/triangle_ray_gate.py"
    spec = importlib.util.spec_from_file_location("new_triangle_gate_test", p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def triangle(z=2.):
    k = np.array([[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]])
    uv = np.array([[1.25, 1.25], [6.25, 1.25], [1.25, 6.25]])
    v = np.column_stack(((uv-[4, 4])*z/8, np.full(3, z)))
    return v, np.array([[0, 1, 2]], np.int64), k


def test_new64x48_independent_micro_reference_and_decimal(gate):
    result, report = gate.validate_micro_reference(64, 48)
    assert result["depth"].shape == (48, 64) and result["depth"].dtype == np.float64
    assert report["full_grid_pixels"] == 3072 and report["independent_label_disagreements"] == 0
    assert report["Decimal80_samples"] == 20 and report["rear_first_nearest_layer_verified"]
    assert report["full_grid_ray_max_Z_error_m"] < report["threshold_m"] == 1e-8
    assert report["full_grid_plane_max_Z_error_m"] < 1e-8 and report["Decimal80_max_Z_error_m"] < 1e-8
    assert report["stratum_pixels"]["12"] > 0


def test_plane_coverage_missing_nan_and_exact_centres(gate):
    v, f, k = triangle(); result = gate.render_triangles(v, f, k, 8, 8)
    y, x = np.mgrid[:8, :8]
    hit = (x+.5 >= 1.25) & (y+.5 >= 1.25) & ((x+.5)+(y+.5) <= 7.5)
    np.testing.assert_array_equal(result["face_index"] >= 0, hit)
    np.testing.assert_array_equal(result["depth"][hit], 2.)
    assert np.isnan(result["depth"][~hit]).all() and np.isnan(result["barycentric"][~hit]).all()


def test_perspective_barycentrics_color_and_camera_Z_not_range(gate):
    v, f, k = triangle(); z = np.array([2., 3., 4.]); v[:, :2] *= z[:, None]/2; v[:, 2] = z
    result = gate.render_triangles(v, f, k, 8, 8, vertex_colors=np.eye(3))
    hit = result["face_index"] >= 0; bary = result["barycentric"][hit]
    p = bary @ v; y, x = np.nonzero(hit)
    np.testing.assert_allclose(p[:, :2]/p[:, 2, None]*8+[4, 4], np.column_stack((x+.5, y+.5)), atol=2e-15)
    np.testing.assert_allclose(result["depth"][hit], p[:, 2], atol=1e-15)
    np.testing.assert_array_equal(result["rgb"][hit], bary)
    assert np.max(np.linalg.norm(p, axis=1)-result["depth"][hit]) > .1


@pytest.mark.parametrize("reverse", [False, True])
def test_no_backface_cull_rear_first_nearest_and_exact_lexical_ties(gate, reverse):
    v, f, k = triangle(); near = v*.5
    faces = np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5]], np.int64)
    if reverse: faces = faces[:, ::-1].copy()
    result = gate.render_triangles(np.r_[v, near], faces, k, 8, 8, labels=np.array([3, 4, 5], np.int64))
    hit = result["face_index"] >= 0
    assert hit.any() and np.all(result["face_index"][hit] == 1) and np.all(result["labels"][hit] == 4)
    np.testing.assert_array_equal(result["depth"][hit], 1.)


def test_offscreen_and_edge_on_faces_preserved_as_geometry_not_hits(gate):
    v, f, k = triangle(); off = v+[20., 0., 0.]
    edge = np.array([[0., 0., 1.], [0., 0., 2.], [0., 1., 3.]])
    result = gate.render_triangles(np.r_[v, off, edge], np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int64), k, 8, 8)
    assert result["zero_area_projected_faces"] == 1 and not np.any(result["face_index"] > 0)


@pytest.mark.parametrize("fault", ["verticesdtype", "facesdtype", "verticesnan", "znegative", "zrange", "indices",
    "degenerate", "Kskew", "Kprincipal", "Kbottom", "Knan", "Kfocal", "width", "height", "masked"])
def test_invalid_original_geometry_camera_and_bounds_rejected(gate, fault):
    v, f, k = triangle(); width, height = 8, 8
    if fault == "verticesdtype": v = v.astype(np.float32)
    elif fault == "facesdtype": f = f.astype(np.int32)
    elif fault == "verticesnan": v[0, 0] = np.nan
    elif fault == "znegative": v[0, 2] = -1
    elif fault == "zrange": v[0, 2] = 10_001
    elif fault == "indices": f[0, 0] = 3
    elif fault == "degenerate": v[2] = v[1]
    elif fault == "Kskew": k[0, 1] = .1
    elif fault == "Kprincipal": k[0, 2] += .5
    elif fault == "Kbottom": k[2, 0] = .1
    elif fault == "Knan": k[0, 0] = np.nan
    elif fault == "Kfocal": k[0, 0] = 1000
    elif fault == "width": width = True
    elif fault == "height": height = 481
    else: v = np.ma.array(v, mask=False)
    with pytest.raises(ValueError): gate.render_triangles(v, f, k, width, height)


@pytest.mark.parametrize("option,value", [("labels", np.array([-1], np.int64)), ("labels", np.array([1.])),
    ("vertex_colors", np.zeros((3, 3), np.float32)), ("vertex_colors", np.full((3, 3), 1.1))])
def test_optional_original_labels_and_colors_strict(gate, option, value):
    v, f, k = triangle()
    with pytest.raises(ValueError): gate.render_triangles(v, f, k, 8, 8, **{option: value})


def test_candidates_and_deadline_fail_closed_without_input_mutation(gate, monkeypatch):
    v, f, k = triangle(); original = v.copy()
    monkeypatch.setattr(gate, "MAX_CANDIDATES", 1)
    with pytest.raises(ValueError, match="ceiling"): gate.render_triangles(v, f, k, 8, 8)
    np.testing.assert_array_equal(v, original)
    with pytest.raises(ValueError, match="deadline"): gate.render_triangles(v, f, k, 8, 8, deadline=time.monotonic()-1)
    monkeypatch.setattr(gate, "MAX_CANDIDATES", 10000)
    values = iter([10., 12.]); monkeypatch.setattr(gate.time, "monotonic", lambda: next(values))
    with pytest.raises(TimeoutError): gate.render_triangles(v, f, k, 8, 8, deadline=11.)


def test_reference_detects_perturbed_depth_before_cuda(gate, monkeypatch):
    original = gate.render_triangles
    def altered(*args, **kwargs):
        result = original(*args, **kwargs); result["depth"][0, 0] += .001
        return result
    monkeypatch.setattr(gate, "render_triangles", altered)
    with pytest.raises(ValueError, match="reference gate"): gate.validate_micro_reference(64, 48)


def test_readonly_sources_no_model_data_oldfixture_and_bounded_wrapper(gate):
    p = Path(gate.__file__); source = p.read_text(); wrapper = p.with_name("run_triangle_ray_gate.sh")
    text = wrapper.read_text()
    assert not any(word in source+text for word in ("np.save", "PIL", "tudl", "authored_rgbd", "torch.load", "torch.jit.load"))
    assert "CPU_reference_gate_passed=True" in source and "affects_reference_gate=False" in source
    assert "os.O_EXCL, 0o400" in source and "signal.alarm(BUDGET)" in source
    assert "103s docker run" in text and "--memory 4g --cpus 2" in text and "--network none" in text
    assert text.count("--mount ") == 4 and 'exec 9<>"$LOCK"' in text and "flock --nonblock 9" in text
    assert 'docker stop --time 2 "$NAME"' in text and 'docker kill "$NAME"' in text and 'docker rm "$NAME"' in text
    assert 'name=^/${NAME}$' in text and "actual_GPU_compute_empty_after" in text
    assert not any(f"src=$ROOT/{folder}" in text for folder in ("weights", "data", "vendor", "outputs", "results", "validation"))
    assert subprocess.run(["rtk", "proxy", "bash", "-n", str(wrapper)], capture_output=True).returncode == 0
    assert subprocess.run(["rtk", "proxy", "bash", str(wrapper), "--unknown"], capture_output=True).returncode == 2
