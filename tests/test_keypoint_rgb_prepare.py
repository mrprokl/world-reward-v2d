"""Fresh scene/manifests/mask firewalls only; no local native models/rendering."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

INFRA = Path(__file__).parents[1] / "infra"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA)); monkeypatch.syspath_prepend(str(INFRA.parent / "src"))
    loaded = []
    for name in ("keypoint_rgb_render", "keypoint_rgb_masks"):
        spec = importlib.util.spec_from_file_location(name, INFRA / (name + ".py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); loaded.append(module)
    return loaded


def metadata():
    names = [f"unused_{i}" for i in range(249)]
    required = [f"{side}_{suffix}" for side in "lr" for suffix in ("uparm_ry", "elbow_bend", "wrist_ry", "index1_rz", "middle1_rz", "ring1_rz", "pinky1_rz")]
    for i, name in enumerate(required): names[i] = name
    return names, np.tile([-1., 1.], (249, 1))


def test_preregistered_fresh_recipe_and_no_old_globals_changed(modules):
    render, masks = modules
    assert render.FOCALS == (1280.,) * 3 and render.SIDES == ("r", "l", "r")
    assert render.SHAPES == ((-.33, .09), (.21, -.18), (.38, .14)) and render.SCALES == (-.025, .045, .060)
    assert render.YAWS == (-.12, .19, -.21) and render.BOTTLE_DEPTH_OFFSETS == (-.09, -.035, .025, .07, .115)
    assert render.primitives.FOCALS == (1160., 1480., 1720.) and render.primitives.BASE == "validation/identity_rgb_v2"
    assert masks.masks.SCHEMA == "world-reward-hands-rgb-inputs-v1"
    assert masks.QUERIES == (("human", "person."), ("object", "bottle."))


def test_full249_named_motion_clipconstant_identity_and_locked_scales(modules):
    render, _ = modules; names, bounds = metadata(); locked = np.array([0, 1, 2, 11, 12, 15, 16])
    bounds[136:204] = [-.1, .1]; bounds[136 + locked] = 0
    controls, identity, changes = render.named_controls(names, bounds)
    assert controls.shape == (15, 204) and identity.shape == (15, 45) and controls.dtype == identity.dtype == np.float32
    assert len(changes) == 120 and all(r["name"] == names[r["column"]] and r["column"] < 136 for r in changes)
    full = np.c_[controls, identity]; neutral = full[::5].copy(); neutral[:, :136] = 0
    assert (full >= bounds[:, 0]).all() and (full <= bounds[:, 1]).all() and (neutral >= bounds[:, 0]).all() and (neutral <= bounds[:, 1]).all()
    for clip, side in enumerate(render.SIDES):
        block = slice(clip*5, (clip+1)*5)
        assert np.array_equal(controls[block, 136:204], np.broadcast_to(controls[clip*5, 136:204], (5, 68)))
        assert (controls[block, 136+locked] == 0).all() and (identity[block, 2:] == 0).all()
        assert np.array_equal(identity[block, :2], np.broadcast_to(np.asarray(render.SHAPES[clip], np.float32), (5, 2)))
        for f in range(5):
            assert controls[clip*5+f, names.index(side + "_elbow_bend")] == np.float32(.30+.050*f)
            assert controls[clip*5+f, names.index(side + "_wrist_ry")] == np.float32(-.06+.025*f)
            assert controls[clip*5+f, names.index(side + "_index1_rz")] == np.float32(.14+.025*f)
    assert len({row.tobytes() for row in controls[::5, 136:]}) == 3


@pytest.mark.parametrize("fault", ["duplicate", "missing", "masked", "shape", "nan", "scale", "all_locked", "unused_shape", "pose"])
def test_native_bounds_failclosed_no_clipping(modules, fault):
    render, _ = modules; names, bounds = metadata()
    if fault == "duplicate": names[-1] = names[0]
    elif fault == "missing": names[names.index("r_uparm_ry")] = "absent"
    elif fault == "masked": bounds = np.ma.array(bounds, mask=False)
    elif fault == "shape": bounds = bounds[:204]
    elif fault == "nan": bounds[0, 0] = np.nan
    elif fault == "scale": bounds[140] = [-.02, .02]
    elif fault == "all_locked": bounds[136:204] = 0
    elif fault == "unused_shape": bounds[207] = [.1, 1.]
    elif fault == "pose": bounds[names.index("r_elbow_bend")] = [-1., .2]
    with pytest.raises(ValueError): render.named_controls(names, bounds)


def test_rgb_only_manifest_exact15_and_focal_not_leaked(modules):
    render, masks = modules; manifest = render.public_manifest(["a" * 64] * 15)
    assert set(manifest) == {"schema", "images"} and manifest["schema"] == masks.SCHEMA == "world-reward-keypoint-rgb-v1"
    assert [r["file"] for r in manifest["images"]] == [f"clip_{c:02d}_frame_{f:03d}.png" for c in range(3) for f in range(5)]
    assert all(set(r) == {"file", "sha256", "width", "height"} and (r["width"], r["height"]) == (1024, 768) for r in manifest["images"])


@pytest.mark.parametrize("digests", [[], ["a"*64]*14, ["a"*64]*16, ["z"*64]*15, [None]*15])
def test_manifest_partial_or_bad_hash_rejected(modules, digests):
    with pytest.raises(ValueError): modules[0].public_manifest(digests)


def test_fixed_framing_translation_invariant_uses_common1280(modules):
    render, _ = modules; v = np.array([[-.4, -.9, -.1], [.4, .9, .1], [0, 0, 0]])
    values = np.stack([v, v*1.05, v*.9]); c, d, floor = render.fixed_framing(values)
    other = render.fixed_framing(values + [7., 4., -3.])
    assert np.allclose(other[0], c + [7., 4., -3.]) and np.isclose(other[1], d) and np.isclose(other[2], floor)
    for points in values:
        pixels = render.render.project_camera_points(points-c+[0, 0, d], np.array([[1280., 0, 512.], [0, 1280., 384.], [0, 0, 1.]]))
        assert (pixels > 8).all() and (pixels < [1016, 760]).all()


def own_actor():
    x = np.linspace(-.35, .35, 200); y = np.linspace(-.8, .8, 200)
    vertices = np.c_[x, y, .05*np.sin(5*x)]
    faces = np.array([[0, 1, 2], [197, 198, 199]], np.int64)
    left = np.zeros(200, bool); right = left.copy(); left[:60] = True; right[-60:] = True
    return vertices, faces, {"l": {"vertex_mask": left}, "r": {"vertex_mask": right}}


def test_fresh_scene_pose_object_rigidity_trueK_and_material_recipe(modules):
    from scipy.spatial.transform import Rotation
    render, _ = modules; human, faces, regions = own_actor(); before = human.copy()
    center, distance, floor = render.fixed_framing(np.stack([human]*3)); bottle, _ = render.primitives.bottle_mesh()
    spectra = []
    for clip in range(3):
        for frame in range(5):
            hv, ov, of, sv, sf, colors, K = render.scene(human, faces, regions, center, distance, floor, clip, frame)
            R = Rotation.from_rotvec([0., render.YAWS[clip]+.035*(frame-2), 0.]).as_matrix()
            expected = (human-center) @ R.T + [.012*(frame-2), .007*np.sin(.8*frame), .035*(frame-2)+distance]
            assert np.allclose(hv, expected) and np.array_equal(K, [[1280., 0, 512.], [0, 1280., 384.], [0, 0, 1.]])
            assert len(ov) == 194 and of.shape == (384, 3) and np.isclose(np.linalg.norm(ov[0]-ov[100]), 1.08*np.linalg.norm(bottle[0]-bottle[100]))
            assert np.isfinite(sv).all() and sv[:, 2].min() > .01 and (sf >= 0).all() and sf.max() < len(sv)
            assert np.allclose(colors[len(hv)], [.19, .38, .43]) and np.allclose(colors[len(hv)+160], [.25, .27, .29])
            assert np.allclose(colors[:len(hv)][regions["l"]["vertex_mask"]], [.73, .69, .65])
            spectra.append(hv.tobytes())
    assert len(set(spectra)) == 15 and np.array_equal(before, human)


def test_foreground_truth_room_ids_removed_no_object_confound(modules, monkeypatch):
    render, _ = modules; primitive = render.primitives
    monkeypatch.setattr(primitive, "WIDTH", 16); monkeypatch.setattr(primitive, "HEIGHT", 16)
    face = np.zeros((16, 16), np.int64); face[4:8] = 2; face[8:] = 999
    depth = np.full((16, 16), 4., np.float32)
    result, z = primitive.foreground_truth(face, depth, np.zeros((2, 3), int), np.zeros((1, 3), int))
    assert (result[8:] == -1).all() and np.isnan(z[8:]).all() and np.isfinite(z[:8]).all()
    assert (result[4:8] == 2).all()


@pytest.mark.parametrize("entity", ["human", "object"])
def test_data_independent_truth_helper_requires_both64_visible_pixels(modules, monkeypatch, entity):
    primitive = modules[0].primitives
    monkeypatch.setattr(primitive, "WIDTH", 16); monkeypatch.setattr(primitive, "HEIGHT", 16)
    face = np.zeros((16, 16), np.int64); face[4:8] = 2
    if entity == "human": face[:] = 2
    else: face[:] = 0
    with pytest.raises(ValueError, match="visible support"):
        primitive.foreground_truth(face, np.ones((16, 16), np.float32), np.zeros((2, 3), int), np.zeros((1, 3), int))


def input_fixture(masks, tmp_path):
    directory = tmp_path / "inputs"; directory.mkdir(); records = []
    for index in range(15):
        clip, frame = divmod(index, 5); name = f"clip_{clip:02d}_frame_{frame:03d}.png"
        path = directory / name; path.write_bytes(("tiny-RGB-"+str(index)).encode()); path.chmod(0o444)
        records.append(dict(file=name, sha256=masks.masks.identity(path)["sha256"], width=1024, height=768))
    manifest = dict(schema=masks.SCHEMA, images=records)
    def save(): (directory / "manifest.json").write_text(json.dumps(manifest))
    save(); return directory, manifest, save


def test_actual_public_validator_reads_all15_images_readonly(modules, tmp_path):
    masks = modules[1]; directory, _, _ = input_fixture(masks, tmp_path)
    rows, receipt = masks.validate_inputs(directory)
    assert len(rows) == 15 and receipt["sha256"] == masks.masks.identity(directory / "manifest.json")["sha256"]
    assert [(r["clip_index"], r["frame_index"]) for r in rows] == [(c, f) for c in range(3) for f in range(5)]


@pytest.mark.parametrize("fault", ["schema", "topprivate", "rowprivate", "extra", "order", "count", "booldim", "tamper", "symlink"])
def test_public_firewall_before_native_or_private(modules, tmp_path, fault):
    masks = modules[1]; directory, manifest, save = input_fixture(masks, tmp_path)
    if fault == "schema": manifest["schema"] = "world-reward-identity-rgb-v1"
    elif fault == "topprivate": manifest["camera_K"] = [[1]]
    elif fault == "rowprivate": manifest["images"][0]["shape"] = [0]
    elif fault == "extra": (directory / "truth.npz").write_bytes(b"forbidden")
    elif fault == "order": manifest["images"].reverse()
    elif fault == "count": manifest["images"].pop()
    elif fault == "booldim": manifest["images"][0]["width"] = True
    elif fault == "tamper": path = directory / manifest["images"][0]["file"]; path.chmod(0o644); path.write_bytes(b"changed")
    else:
        path = directory / manifest["images"][0]["file"]; path.unlink(); path.symlink_to(directory / manifest["images"][1]["file"])
    save()
    with pytest.raises(ValueError): masks.validate_inputs(directory)


def complete_report():
    rows = []
    for clip in range(3):
        for frame in range(5):
            stem = f"clip_{clip:02d}_frame_{frame:03d}"; row = dict(file=stem+".png", clip_index=clip, frame_index=frame)
            for label, query in (("human", "person."), ("object", "bottle.")):
                row.update({label+"_query": query, label+"_mask_file": stem+"_"+label+".png", label+"_mask_sha256": "a"*64,
                            label+"_mask_pixels": 42, label+"_mask_bytes": 128})
            rows.append(row)
    return dict(records=rows, actual_detector_calls=30, actual_sam2_calls=30, actual_sam2_image_encoder_calls=15)


def test_complete30masks_exact_calls_no_gt(modules): modules[1].completed(complete_report())


@pytest.mark.parametrize("fault", ["detector", "sam", "encoder", "partial", "order", "query", "filename", "sha", "empty", "boolpixels", "bytes"])
def test_incomplete_or_fallback_masks_fail(modules, fault):
    masks = modules[1]; report = complete_report(); row = report["records"][0]
    if fault == "detector": report["actual_detector_calls"] = 29
    elif fault == "sam": report["actual_sam2_calls"] = 29
    elif fault == "encoder": report["actual_sam2_image_encoder_calls"] = 14
    elif fault == "partial": report["records"].pop()
    elif fault == "order": row["frame_index"] = 1
    elif fault == "query": row["human_query"] = "manual actor label"
    elif fault == "filename": row["human_mask_file"] = "../private.png"
    elif fault == "sha": row["human_mask_sha256"] = "not SHA"
    elif fault == "empty": row["human_mask_pixels"] = 0
    elif fault == "boolpixels": row["human_mask_pixels"] = True
    elif fault == "bytes": row["object_mask_bytes"] = 0
    with pytest.raises(ValueError): masks.completed(report)


def test_reused_automatic_selection_and_mask_no_fallback(modules):
    masks = modules[1]
    with pytest.raises(ValueError): masks.masks.select_person([], 1024, 768)
    predicted = np.zeros((1, 768, 1024), np.float32); predicted[:, 20:40, 30:60] = 1.
    value = masks.masks.binary_mask(predicted, np.array([.9]), 1024, 768)
    assert value.dtype == np.uint8 and np.count_nonzero(value) == 600 and set(np.unique(value)) == {0, 255}
    with pytest.raises(RuntimeError): masks.masks.binary_mask(predicted*0, np.array([.9]), 1024, 768)


def test_runtime_imports_lazy_and_wrapper_separate_private_firewall(modules):
    render, masks = modules
    for module in (render, masks):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        imports |= {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
        assert not {"torch", "pytorch3d", "sam2", "transformers"} & imports
        with pytest.raises(SystemExit): module.main(["--oracle"])
    source = Path(masks.__file__).read_text(); assert "eval_private" not in source and "local_files_only=True" in source
    wrapper = INFRA / "run_keypoint_rgb_prepare.sh"; subprocess.run(["bash", "-n", str(wrapper)], check=True)
    assert subprocess.run(["bash", str(wrapper), "--retry"], capture_output=True).returncode == 2
    text = wrapper.read_text(); second = text.split('MASK_IMAGE="', 1)[1]
    assert "src=$BASE,dst=$BASE" not in second and "eval_private" not in second and "weights/mhr" not in second
    assert "src=$BASE/inputs,dst=$BASE/inputs,readonly" in second and "src=$OUT,dst=$OUT" in second
    assert text.count("--network none") == 2 and "123s" in text and "183s" in text and render.IMAGE in text


def test_source_closures_immutable_helpers_and_within_launcher_budget(modules):
    import base64
    import io
    import lzma
    import tarfile
    import azure_job
    root = INFRA.parent
    files = {str(p.relative_to(root)): p.read_bytes() for base in (root/"infra", root/"src", root/"configs") for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (root/"pyproject.toml").read_bytes()
    selected = azure_job.runtime_bundle_paths(files, "infra/run_keypoint_rgb_prepare.sh")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in selected:
            info = tarfile.TarInfo(path); info.size = len(files[path]); archive.addfile(info, io.BytesIO(files[path]))
    assert len(base64.b64encode(lzma.compress(buffer.getvalue(), preset=6))) <= azure_job.MAX_CODE_CONTROL_BYTES
    files["infra/test_mask_closure.sh"] = b'python "$CODE/infra/keypoint_rgb_masks.py"\n'
    mask_only = azure_job.runtime_bundle_paths(files, "infra/test_mask_closure.sh")
    assert "infra/hand_synthetic_masks.py" in mask_only and "infra/keypoint_rgb_render.py" not in mask_only
    assert "infra/identity_rgb_render.py" not in mask_only and "infra/keypoint_rgb_render.py" in selected
