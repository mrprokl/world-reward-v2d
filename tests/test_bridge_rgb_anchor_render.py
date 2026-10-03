"""Tiny reference arrays only; no model assets, image generation or GPU work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest


@pytest.fixture
def render(monkeypatch):
    root = Path(__file__).parents[1]; monkeypatch.syspath_prepend(str(root/"infra"))
    spec = importlib.util.spec_from_file_location("new_bridge_anchor_test", root/"infra/bridge_rgb_anchor_render.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def names():
    required = ["l_uparm_ry", "l_elbow_bend", "r_uparm_ry", "r_elbow_bend"]
    required += [f"l_{finger}1_rz" for finger in ("index", "middle", "ring", "pinky")]
    return required+[f"other_{i}" for i in range(249-len(required))]


def skeleton():
    names = ["l_wrist", "l_index1", "l_middle1"]+[f"joint{i}" for i in range(124)]
    joints = np.zeros((127, 3)); joints[1] = [.02, -.08, .01]; joints[2] = [0., -.08, .01]
    return joints, names


def fake_gate(render, tmp_path, fault=None):
    source = {n: dict(bytes=1, sha256=s) for n, s in render.RAY_SOURCES.items()}
    data = dict(schema="world_reward.triangle_ray_gate.v1", stage="new_independent_CPU_triangle_reference",
        status="pass", phase="complete", producer_revision="a"*40, image_id=render.IMAGE, budget_seconds=100,
        CPU_reference_gate_passed=True, sources_rehashed_after=True, models_used=False, challenge_inputs_used=False,
        failed_cohort_read=False, downstream_gates_changed=False, previous_failures_reinterpreted=False,
        inference_quality_verified=False, source_helpers=source, CPU_reference=dict(width=640, height=480,
            full_grid_pixels=307200, independent_label_disagreements=0, Decimal80_samples=20,
            rear_first_nearest_layer_verified=True, threshold_m=1e-8, full_grid_ray_max_Z_error_m=1e-11,
            full_grid_plane_max_Z_error_m=1e-11, Decimal80_max_Z_error_m=1e-11))
    if fault == "status": data["status"] = "fail"
    elif fault == "producer": data["producer_revision"] = "b"*40
    elif fault == "source": data["source_helpers"][next(iter(source))]["sha256"] = "b"*64
    elif fault == "models": data["models_used"] = True
    elif fault == "threshold": data["CPU_reference"]["threshold_m"] = 1e-5
    elif fault == "error": data["CPU_reference"]["Decimal80_max_Z_error_m"] = 1e-6
    elif fault == "nan": data["CPU_reference"]["full_grid_ray_max_Z_error_m"] = float("nan")
    p = tmp_path/"ray-receipt.json"; p.write_text(json.dumps(data)); p.chmod(0o400)
    return p, hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_size, "a"*40


def test_actual_gate_identity_and_proof_are_required_before_models(render, tmp_path):
    p, sha, size, producer = fake_gate(render, tmp_path)
    assert render.require_ray_receipt(p, sha, size, producer) == dict(bytes=size, sha256=sha)
    with pytest.raises(ValueError): render.require_ray_receipt(p, "f"*64, size, producer)
    with pytest.raises(ValueError): render.require_ray_receipt(p, sha, size+1, producer)


@pytest.mark.parametrize("fault", ["status", "producer", "source", "models", "threshold", "error", "nan"])
def test_failed_or_reinterpreted_gate_never_allows_render(render, tmp_path, fault):
    with pytest.raises(ValueError): render.require_ray_receipt(*fake_gate(render, tmp_path, fault))


def test_model_reference_names_bounds_and_future_motion_frozen(render):
    n = names(); bounds = np.tile([-1., 1.], (249, 1)); original = bounds.copy()
    q = [render.controls(n, bounds, clip, 0) for clip in range(4)]
    assert all(v.dtype == np.float32 and v.shape == (204,) for v in q)
    assert all(v.tobytes() == q[0].tobytes() for v in q)
    np.testing.assert_array_equal(q[0][136:], 0.)
    assert render.controls(n, bounds, 0, 47).tobytes() != q[0].tobytes()
    np.testing.assert_array_equal(bounds, original)
    bounds[0, 1] = 0.
    with pytest.raises(ValueError): render.controls(n, bounds, 0, 0)
    with pytest.raises(ValueError): render.controls(n[:-1], original, 0, 0)
    with pytest.raises(ValueError): render.controls(n, original, 0, 48)


def test_named_hand_frame_proper_no_ordinal_guess_or_fallback(render):
    joints, n = skeleton(); r, t = render.hand_frame(joints, n)
    np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-14)
    assert np.linalg.det(r) == pytest.approx(1.)
    np.testing.assert_array_equal(t, joints[0])
    with pytest.raises(ValueError): render.hand_frame(joints, ["missing"]+n[1:])
    joints[1] = joints[2]
    with pytest.raises(ValueError): render.hand_frame(joints, n)


def test_four_new_object_motions_are_full48_and_no_truth_hand_observations(render):
    joints, n = skeleton(); hr, ht = render.hand_frame(joints, n)
    paths = []
    for clip in range(4):
        poses = [render.object_pose(clip, frame, hr, ht) for frame in range(48)]
        assert all(np.linalg.det(r) == pytest.approx(1.) for r, _ in poses)
        paths.append(b"".join(r.tobytes()+t.tobytes() for r, t in poses))
        assert np.linalg.norm(poses[-1][1]-poses[0][1]) > 0 or np.max(np.abs(poses[-1][0]-poses[0][0])) > 0
    assert len(set(paths)) == 4 and render.SEED == 2026100401 and render.FRAMES == 48


def test_new_mesh_scene_and_shared_camera_do_not_modify_inputs(render):
    v, f = render.object_mesh(); original = v.copy(); j, n = skeleton()
    sv, sf, colors, labels, truth = render.scene(v, j, f, n, 0, 0)
    assert sv.dtype == np.float64 and sf.dtype == np.int64 and colors.shape == sv.shape
    assert len(labels) == len(sf) and set(labels) == {0, 1, 2}
    assert np.isfinite(sv).all() and np.min(sv[:, 2]) > 0
    np.testing.assert_array_equal(v, original)
    np.testing.assert_array_equal(truth["camera_K"], render.K)
    assert "human_joints_camera_m" in truth and "object_rotation" in truth


def small_scene():
    k = np.array([[20., 0., 8.], [0., 20., 8.], [0., 0., 1.]])
    vertices = np.array([[-2., -2., 4.], [2., -2., 4.], [2., 2., 4.], [-2., 2., 4.],
                         [-.3, -.3, 2.], [.4, -.3, 2.5], [.1, .4, 2.2]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], np.int64)
    return vertices, faces, k


def test_tiny16x16_full_plane_and64_all_triangle_nearest_ray_proof(render):
    v, f, k = small_scene(); deadline = time.monotonic()+2.
    result = render.ray.render_triangles(v, f, k, 16, 16)
    proof = render.certify_render(v, f, k, result, deadline)
    assert proof["independent_rays"] == 64 and proof["all_original_faces_tested_per_ray"] == 3
    assert proof["full_grid_plane_max_Z_error_m"] < 1e-8 and proof["independent64_ray_max_Z_error_m"] < 1e-8
    result["depth"][0, 0] += .001
    with pytest.raises(ValueError): render.certify_render(v, f, k, result, deadline)


def test_all_triangle_certificate_rejects_nearer_face_hidden_by_wrong_raster(render):
    v, f, k = small_scene()
    wrong = render.ray.render_triangles(v, f[:2], k, 16, 16)
    with pytest.raises(ValueError): render.certify_render(v, f, k, wrong, time.monotonic()+2.)


def test_entity_stratified64_rays_include_human_and_object(render):
    v, f, k = small_scene()
    result = render.ray.render_triangles(v, f, k, 16, 16, labels=np.array([0, 0, 1], np.int64))
    proof = render.certify_render(v, f, k, result, time.monotonic()+2.)
    assert proof["sample_scope"] == "32_global_16_human_16_object"


def test_public_schema_is_only_rgb_original_indices_hash_bytes_dimensions(render):
    manifest = render.public_manifest([dict(sha256="a"*64, bytes=20+i) for i in range(4)])
    assert set(manifest) == {"schema", "images"} and manifest["schema"] == render.SCHEMA
    for i, row in enumerate(manifest["images"]):
        assert set(row) == {"clip_id", "frame_id", "file", "sha256", "bytes", "width", "height"}
        assert row["clip_id"] == i and row["frame_id"] == 0 and row["file"] == f"clip_{i:06d}_frame_000000.png"
    with pytest.raises(ValueError): render.public_manifest([dict(sha256="a"*64, bytes=20)]*3)


def test_cpu_wrapper_fixed_asset_scoped_mounts_and_source_gate_before_model(render):
    path = Path(render.__file__); source = path.read_text(); wrapper = path.with_name("run_bridge_rgb_anchor_render.sh")
    text = wrapper.read_text()
    assert source.index("gate_pin = require_ray_receipt") < source.index("model_pin = identity") < source.index("torch.jit.load")
    assert source.index("proof = certify_render") < source.index("public = out/")
    assert "reference_asset_provenance" in source and "SAM_prediction_provenance_verified=False" in source
    assert "automatic_mask_Body_MoGe_gate_passed=False" in source and "all_four_ray_gates_before_publication=True" in source
    assert "183s docker run" in text and "--memory 12g --cpus 4" in text and "--gpus" not in text
    assert "CUDA_VISIBLE_DEVICES=" in text and "--network none" in text and text.count("--mount ") == 6
    assert render.MODEL_SOURCE in text and 'src=$MODEL,dst=$ROOT/weights/mhr/mhr_model.pt,readonly' in text
    assert all(f'docker {command} ' in text for command in ("stop", "kill", "rm"))
    assert not any(word in source+text for word in ("track_1/", "author_human_field", "joint_rgb_render", "mhr-finger-semantics"))
    assert subprocess.run(["rtk", "proxy", "bash", "-n", str(wrapper)], capture_output=True).returncode == 0
    assert subprocess.run(["rtk", "proxy", "bash", str(wrapper), "--unknown"], capture_output=True).returncode == 2
