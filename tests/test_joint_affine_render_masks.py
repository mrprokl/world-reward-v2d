"""New own CPU controls/topology/public-firewall tests; no native models/GT."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra)); loaded = []
    for name in ("joint_affine_render", "joint_affine_masks"):
        spec = importlib.util.spec_from_file_location("wr_test_"+name, infra/(name+".py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); loaded.append(module)
    return loaded


def control_fixture():
    names = [f"unused_{i}" for i in range(249)]
    for i, name in enumerate(f"{side}_{suffix}" for side in "lr" for suffix in
        ("uparm_ry", "elbow_bend", "index1_rz", "middle1_rz", "ring1_rz", "pinky1_rz")): names[i] = name
    return names, np.tile([-1., 1.], (249, 1))


def test_fresh_clipconstant_identity_named_limits_scaleszero(modules):
    render, _ = modules; names, bounds = control_fixture(); original_bounds = bounds.copy()
    controls, shape, changes = render.named_controls(names, bounds)
    assert controls.shape == (18, 204) and shape.shape == (18, 45) and len(changes) == 108
    assert np.count_nonzero(controls[:, 136:]) == 0 and np.count_nonzero(shape[:, 1:]) == 0
    np.testing.assert_allclose(shape[:, 0], np.repeat([-.4, 0., .4], 6))
    assert not np.array_equal(controls[0], controls[5]) and np.array_equal(bounds, original_bounds)
    assert all(names[r["column"]] == r["name"] and bounds[r["column"], 0] <= r["value"] <= bounds[r["column"], 1] for r in changes)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "arm_limit", "shape_limit", "neutral", "shape", "nan", "masked"])
def test_native_names_and_limits_fail_without_clipping(modules, fault):
    render, _ = modules; names, bounds = control_fixture()
    if fault == "missing": names[0] = "unknown"
    elif fault == "duplicate": names[-1] = names[0]
    elif fault == "arm_limit": bounds[0, 1] = .1
    elif fault == "shape_limit": bounds[204] = [-.3, .3]
    elif fault == "neutral": bounds[30] = [.1, 1.]
    elif fault == "shape": bounds = bounds[:204]
    elif fault == "nan": bounds[0, 0] = np.nan
    else: bounds = np.ma.array(bounds)
    with pytest.raises(ValueError): render.named_controls(names, bounds)


def test_new_bottle_scale_preserves_closed_oriented_positive_mesh(modules):
    render, _ = modules; vertices, faces = render.bottle_mesh(); before = vertices.copy()
    for scale in render.BOTTLE_SCALES:
        v = vertices*scale
        directed = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
        _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        assert np.all(counts == 2) and np.all(np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1)) == 0)
        assert np.einsum("ij,ij->i", v[faces[:, 0]], np.cross(v[faces[:, 1]], v[faces[:, 2]])).sum() > 0
    assert np.array_equal(before, vertices) and len(vertices) == 194 and len(faces) == 384


def test_new_scene_contexts_scale_once_no_source_mutation(modules):
    render, _ = modules
    human = np.array([[-.15, -.2, 0.], [.15, .2, 0.], [-.1, .15, .1], [.1, -.15, .1]])
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    regions = {"l": {"vertex_mask": np.array([True, False, True, False])}, "r": {"vertex_mask": np.array([False, True, False, True])}}
    old = human.copy(); original_bottle, _ = render.bottle_mesh()
    for clip in range(3):
        result = render.scene(human, faces, regions, np.array([0., 0., 4.]), clip, 2)
        hv, ov, of, scene, topology, colors, K = result
        assert K[0, 0] == render.FOCALS[clip] and np.array_equal(of, render.bottle_mesh()[1])
        assert np.max(topology) < len(scene) and colors.shape == scene.shape
        assert np.linalg.norm(ov[0]-ov[1]) == pytest.approx(np.linalg.norm(original_bottle[0]-original_bottle[1])*render.BOTTLE_SCALES[clip])
        assert np.array_equal(human, old)
    assert render.OFFSETS[1][0] == .25 and render.OFFSETS[1][2] == -.25 and render.OFFSETS[2][2] == .25


def fixture_inputs(render, tmp_path):
    folder = tmp_path/"inputs"; folder.mkdir(); digests = []
    for clip in range(3):
        for frame in range(6):
            p = folder/f"clip_{clip:02d}_frame_{frame:03d}.png"; p.write_bytes(p.name.encode()+b"Own tiny RGB fixture"); digests.append(render.sha256(p))
    manifest = render.public_manifest(digests); (folder/"manifest.json").write_text(json.dumps(manifest)); return folder, manifest


def test_public_only18_rgb_hashes_never_manufacturing_truth(modules, tmp_path):
    render, masks = modules; folder, manifest = fixture_inputs(render, tmp_path); records, receipt = masks.validate_inputs(folder)
    assert set(manifest) == {"schema", "images"} and manifest["schema"] == masks.SCHEMA
    assert len(records) == 18 and receipt["sha256"] == render.sha256(folder/"manifest.json")
    assert all(set(r) == {"file", "sha256", "width", "height"} for r in manifest["images"])
    assert [(r["clip_index"], r["frame_index"]) for r in records] == [(c, f) for c in range(3) for f in range(6)]


@pytest.mark.parametrize("fault", ["schema", "camera", "record_depth", "order", "count", "width", "bool_width", "name", "sha", "bytes", "symlink", "extra"])
def test_public_mask_firewall_fails_no_oracle_fallback(modules, tmp_path, fault):
    render, masks = modules; folder, manifest = fixture_inputs(render, tmp_path); item = manifest["images"][0]
    if fault == "schema": manifest["schema"] = "world-reward-joint-rgb-v1"
    elif fault == "camera": manifest["camera_K"] = []
    elif fault == "record_depth": item["depth"] = "private.npz"
    elif fault == "order": manifest["images"][:2] = reversed(manifest["images"][:2])
    elif fault == "count": manifest["images"].pop()
    elif fault == "width": item["width"] = 768
    elif fault == "bool_width": item["width"] = True
    elif fault == "name": item["file"] = "../outside"
    elif fault == "sha": item["sha256"] = "a"*64
    elif fault == "bytes": (folder/item["file"]).write_bytes(b"changed")
    elif fault == "symlink":
        p = folder/item["file"]; target = tmp_path/"outside"; p.rename(target); p.symlink_to(target)
    else: (folder/"camera.json").write_text("{}")
    (folder/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): masks.validate_inputs(folder)


def test_wrappers_private_truth_firewall_frozen_outputs_budget(modules):
    render, masks = modules
    r = Path(render.__file__).with_name("run_joint_affine_render.sh").read_text()
    m = Path(masks.__file__).with_name("run_joint_affine_masks.sh").read_text()
    assert "183s docker run" in r and "183s docker run" in m
    assert "--network none --memory 16g --cpus 4" in r and "--network none --memory 16g --cpus 4" in m
    rm = [line for line in r.splitlines() if "--mount" in line]; mm = [line for line in m.splitlines() if "--mount" in line]
    assert len(rm) == 4 and sum("readonly" in line for line in rm) == 3
    assert len(mm) == 7 and sum("readonly" in line for line in mm) == 6
    assert all("eval_private" not in line and "src=$BASE,dst=" not in line and "src=$ROOT/validation,dst=" not in line for line in mm)
    assert all("src=$ROOT/outputs" not in line for line in rm+mm)
    assert 'private.mkdir(mode=0o700)' in Path(render.__file__).read_text()
    assert 'chown scenesmith:scenesmith "$DEST"' in r and 'chown scenesmith:scenesmith "$OUT"' in m
    assert "chown -R" not in r+m and "--entrypoint python" in r+m


def test_old_joint_protocol_unchanged_and_new_schema_is_independent(modules):
    import joint_rgb_render, joint_rgb_masks
    render, masks = modules
    assert joint_rgb_render.SCHEMA == joint_rgb_masks.SCHEMA == "world-reward-joint-rgb-v1"
    assert joint_rgb_render.FRAMES == 3 and joint_rgb_render.FOCALS == (1280., 960., 1600.)
    assert render.FRAMES == 6 and render.FOCALS == (1024., 1408., 1792.) and render.TRUTH_KEYS == joint_rgb_render.TRUTH_KEYS
    assert masks.STAGE == "public_joint_affine_rgb_automatic_masks" and masks.FRAMES*masks.CLIPS == 18
    assert render.BUDGET == masks.BUDGET == 180


def test_no_renderer_or_masks_oracle_knob(modules):
    for module in modules:
        with pytest.raises(SystemExit): module.main(["--camera-focal", "1280"])
