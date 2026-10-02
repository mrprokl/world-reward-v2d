"""Procedural render contracts only, no torch/model/assets/rendering locally."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def render(monkeypatch):
    root = Path(__file__).resolve().parents[1]; monkeypatch.syspath_prepend(str(root / "infra"))
    spec = importlib.util.spec_from_file_location("wr_test_hand_render", root / "infra/hand_synthetic_render.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_manifest_has_only_rgb_integrity_grid_no_truth(render):
    manifest = render.public_manifest(["a"*64]*6)
    assert set(manifest) == {"schema", "images"}
    assert manifest["schema"] == "world-reward-hands-rgb-inputs-v1"
    assert [v["file"] for v in manifest["images"]] == [f"case_{i:03d}.png" for i in range(6)]
    assert all(set(v) == {"file", "sha256", "width", "height"} for v in manifest["images"])
    assert all((v["width"], v["height"]) == (1024, 768) for v in manifest["images"])


@pytest.mark.parametrize("digests", [[], ["a"*64]*5, ["a"*64]*7, ["x"*64]*6, [None]*6])
def test_manifest_requires_six_exact_sha_records(render, digests):
    with pytest.raises(ValueError): render.public_manifest(digests)


def semantic_report(render):
    return {"stage": "own_reference_mhr_finger_semantics_support", "status": "pass", "phase_A_verified": True,
            "phase_B_verified": True, "named_finger_joint_partition_verified": True,
            "excluded_joint_invariance_verified": True, "correctives_skeleton_bit_identical": True,
            "model_sha256": render.semantics.MODEL_SHA, "body_checkpoint_sha256": render.semantics.BODY_SHA,
            "body_revision": render.semantics.BODY_REVISION, "network": "none", "challenge_inputs_used": False,
            "script_sha256": render.sha256(Path(render.semantics.__file__))}


def test_semantic_report_current_bytes_not_directory_identity(render):
    report = semantic_report(render); report["producer_path"] = "/old/frozen/source"
    render.require_semantic_report(report)


@pytest.mark.parametrize("key,value", [("status", "fail"), ("phase_A_verified", 1), ("phase_B_verified", False),
    ("model_sha256", "a"*64), ("body_checkpoint_sha256", "a"*64), ("script_sha256", "a"*64),
    ("network", "default"), ("challenge_inputs_used", True)])
def test_semantic_prerequisite_not_implied(render, key, value):
    report = semantic_report(render); report[key] = value
    with pytest.raises(ValueError): render.require_semantic_report(report)


def test_neutral_camera_procedural_extent_once_and_default_focal(render):
    vertices = np.array([[-1., -1., -.2], [1., 1., .2], [0., 0., 0.]])
    K, t = render.camera_from_neutral(vertices)
    pixels = render.project_camera_points(vertices + t, K)
    assert K[0, 0] == K[1, 1] == 1280
    assert np.all(pixels > [0, 0]) and np.all(pixels < [1024, 768])
    offset = [10., 20., 30.]
    _, t_shifted = render.camera_from_neutral(vertices + offset)
    np.testing.assert_allclose(vertices+t, vertices+offset+t_shifted)


@pytest.mark.parametrize("vertices", [np.zeros((3, 2)), np.full((3, 3), np.nan), np.zeros((1, 3))])
def test_camera_invalid_truth_geometry_fails(render, vertices):
    with pytest.raises(ValueError): render.camera_from_neutral(vertices)


def test_named_flexion_uses_names_limits_only_shared_nonfinger_zero(render):
    names = [f"unused_{i}" for i in range(249)]
    slots = np.arange(54); slots = np.roll(slots, 5)
    for i, name in enumerate(f"{side}_{suffix}" for side in "lr" for suffix in render.FLEXION_NAMES):
        names[int(slots[i])] = name
    limits = np.tile([-.7, .8], (249, 1))
    controls, records = render.named_controls(names, limits)
    assert len(records) == 28 and np.count_nonzero(controls[0]) == 0
    assert np.array_equal(controls[4], controls[5]) and np.array_equal(controls[3], controls[4])
    changed = [record["column"] for record in records]
    fixed = np.setdiff1d(np.arange(204), changed)
    assert np.count_nonzero(controls[:, fixed]) == 0
    assert np.count_nonzero(controls[1]) == np.count_nonzero(controls[2]) == 14
    assert all(record["value_rad"] == pytest.approx(.32) for record in records)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "invalid_shape", "nan", "neutral_outside", "range_too_small"])
def test_limits_and_actual_named_parameters_fail_closed(render, fault):
    names = [f"unused_{i}" for i in range(249)]
    for i, name in enumerate(f"{side}_{suffix}" for side in "lr" for suffix in render.FLEXION_NAMES): names[i] = name
    limits = np.tile([-.7, .8], (249, 1))
    if fault == "missing": names[0] = "other"
    elif fault == "duplicate": names[-1] = names[-2]
    elif fault == "invalid_shape": limits = limits.T
    elif fault == "nan": limits[0, 0] = np.nan
    elif fault == "neutral_outside": limits[0] = [.1, .8]
    else: limits[0] = [0., .001]
    with pytest.raises(ValueError): render.named_controls(names, limits)


def test_lbs_actual_weighted_region_not_finite_difference_labels(render):
    names = ["root", "l_wrist", "l_thumb0", "r_wrist", "r_thumb0"]
    indices = np.zeros((18439, 2), dtype=np.int32); weights = np.zeros((18439, 2), dtype=float); weights[:, 0] = 1
    indices[:100, 0] = 1; indices[100:200, 0] = 4
    regions = render.lbs_regions(indices, weights, names)
    assert regions["l"]["vertex_mask"].sum() == regions["r"]["vertex_mask"].sum() == 100
    weights[0] = [-.1, 1.1]
    with pytest.raises(ValueError): render.lbs_regions(indices, weights, names)


def test_shader_public_private_and_wrapper_scope(render):
    source = Path(render.__file__).read_text(); wrapper = Path(render.__file__).with_name("run_hand_synthetic_render.sh").read_text()
    assert "TexturesVertex" in source and "specular_color=((0.,)*3,)" in source
    assert "EGL" not in source and "pyrender" not in source and "import pymomentum" not in source
    assert "--network none" in wrapper and "123s docker run" in wrapper and '"$IMAGE" python' in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor"))


def test_private_camera_transform_once_and_no_side_colors(render):
    v = np.array([[-.2, -.4, 0.], [.2, .4, .05], [0., 0., 0.], [0., .2, 0.]])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    regions = {"l": {"vertex_mask": np.array([True, False, False, False])},
               "r": {"vertex_mask": np.array([False, True, False, False])}}
    camera, scene, _, colors, R, t, occluder = render.camera_scene(v, faces, regions, [0., 0., 3.], 3)
    np.testing.assert_allclose(camera, v @ R.T + t)
    assert np.linalg.det(R) == pytest.approx(1) and len(occluder) == 0
    assert np.array_equal(scene, camera) and np.array_equal(colors[0], colors[1])
    cropped, _, _, _, _, _, _ = render.camera_scene(v, faces, regions, [0., 0., 3.], 5)
    assert cropped[:, 0].mean() < camera[:, 0].mean()


def test_failed_render_retains_private_only_report_never_public_manifest(render, tmp_path, monkeypatch):
    (tmp_path / "validation").mkdir(); (tmp_path / "results").mkdir()
    (tmp_path / "results/mhr-finger-semantics-v2.json").write_text('{"status":"fail"}')
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64); monkeypatch.setattr(render.platform, "system", lambda: "Linux")
    original = render.Path.iterdir
    monkeypatch.setattr(render.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="semantic producer"): render.main([])
    root = tmp_path / "validation/hands_rgb_v1"
    report = json.loads((root / "eval_private/render-report.json").read_text())
    assert report["status"] == "fail" and report["inference_performed"] is False
    assert list((root / "inputs").iterdir()) == [] and not (root / "inputs/manifest.json").exists()
    with pytest.raises(FileExistsError): render.main([])


def test_no_experiment_cli_or_own_shared_identity_broadcast(render):
    with pytest.raises(SystemExit): render.main(["--episode", "15"])
    source = Path(render.__file__).read_text()
    assert "identity_rows(np.zeros(45, np.float32), 6)" in source
    assert "mhr-finger-semantics-v2.json" in source and 'private / "render-report.json"' in source
    assert "public_manifest(images)" in source and "model(identity, params, expression, True)" in source


def test_wrapper_reserves_only_new_dataset_not_shared_parent(render):
    wrapper = Path(render.__file__).with_name("run_hand_synthetic_render.sh").read_text()
    assert 'mkdir "$DEST"' in wrapper and '"$DEST"' in wrapper
    assert '--env WR_RENDER_OUTPUT_RESERVED=1' in wrapper
    assert 'src=$DEST,dst=$DEST' in wrapper
    assert 'src=$ROOT/validation,dst=$ROOT/validation' not in wrapper
    assert 'chown -R' not in wrapper


def test_reserved_empty_dataset_still_fails_prerequisite_without_overwrite(render, tmp_path, monkeypatch):
    dest = tmp_path / "validation/hands_rgb_v1"; dest.mkdir(parents=True)
    (tmp_path / "results").mkdir()
    (tmp_path / "results/mhr-finger-semantics-v2.json").write_text('{"status":"fail"}')
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64); monkeypatch.setenv("WR_RENDER_OUTPUT_RESERVED", "1")
    monkeypatch.setattr(render.platform, "system", lambda: "Linux")
    original = render.Path.iterdir
    monkeypatch.setattr(render.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="semantic producer"): render.main([])
    assert json.loads((dest / "eval_private/render-report.json").read_text())["status"] == "fail"
    with pytest.raises(FileExistsError): render.main([])
