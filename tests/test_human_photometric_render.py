"""Own procedural arrays validate manufacturing contracts, not native execution."""
import ast
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "infra")); monkeypatch.syspath_prepend(str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("human_photometric_render_test", REPO / "infra/human_photometric_render.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def metadata():
    names = ["unused" + str(i) for i in range(249)]
    names[:8] = ["l_uparm_ry", "l_elbow_bend", "l_wrist_ry", "r_uparm_ry", *["l_" + f + "1_rz" for f in ("index", "middle", "ring", "pinky")]]
    limits = np.tile([-1., 1.], (249, 1)); limits[136:204] = [-.1, .1]
    return names, limits


def test_fresh_fixed_recipe_shapes_scales_motion_and_no_historical_mutation(renderer):
    names, limits = metadata(); locked = np.array([0, 1, 2, 11, 12, 15, 16]); limits[136 + locked] = 0
    before = limits.copy(); p, shape, changes = renderer.named_controls(names, limits)
    assert p.dtype == shape.dtype == np.float32 and p.shape == (6, 204) and shape.shape == (6, 45) and len(changes) == 48
    assert np.array_equal(p[:3, :136], p[3:, :136]) and len({r.tobytes() for r in p[:3, :136]}) == 3
    for morphology in range(2):
        assert np.array_equal(shape[morphology * 3:(morphology + 1) * 3, :2], np.tile(np.array(renderer.SHAPES[morphology], np.float32), (3, 1)))
        assert np.all(p[morphology * 3:(morphology + 1) * 3, 136 + locked] == 0)
    assert p[0, names.index("l_uparm_ry")] == np.float32(.16) and p[2, names.index("l_elbow_bend")] == np.float32(.36)
    assert [renderer.factors(g) for g in range(8)] == [(m, a, o) for m in range(2) for a in range(2) for o in range(2)]
    assert np.array_equal(limits, before) and renderer.primitives.BASE == "validation/identity_rgb_v2"


@pytest.mark.parametrize("fault", ["masked", "duplicate", "missing", "inverted", "alllocked", "scale", "shape", "pose", "nan"])
def test_no_parameter_bounds_clipping_or_repair(renderer, fault):
    names, bounds = metadata()
    if fault == "masked": bounds = np.ma.array(bounds, mask=False)
    elif fault == "duplicate": names[-1] = names[0]
    elif fault == "missing": names[0] = "absent"
    elif fault == "inverted": bounds[150] = [.1, -.1]
    elif fault == "alllocked": bounds[136:204] = 0
    elif fault == "scale": bounds[150] = [-.01, .01]
    elif fault == "shape": bounds[204] = [-.1, .1]
    elif fault == "pose": bounds[names.index("l_elbow_bend")] = [-.1, .2]
    else: bounds[0, 0] = np.nan
    with pytest.raises(ValueError): renderer.named_controls(names, bounds)


def geometry():
    x = np.linspace(-.27, .27, 300); h = np.c_[x, np.linspace(-.8, .8, 300), .035 * np.sin(x * 9)]
    j = np.zeros((127, 3)); j[9] = [-.15, -.08, 0.]
    regions = {"l": {"vertex_mask": np.arange(300) < 60}, "r": {"vertex_mask": np.arange(300) >= 240}}
    return h, j, np.array([[0, 1, 2], [297, 298, 299]], np.int32), regions


def test_geometry_byte_identical_for_appearance_and_depth_order(renderer):
    h, j, faces, regions = geometry(); before = h.copy(); center, distance, floor = renderer.fixed_framing(np.stack([h, h * 1.03]))
    bottle, of = renderer.primitives.bottle_mesh()
    for frame in range(3):
        results = [renderer.scene(h, faces, j, 9, regions, center, distance, floor, group, frame) for group in range(4)]
        for row in results[1:]:
            assert np.array_equal(row[0], results[0][0]) and np.array_equal(row[1], results[0][1]) and np.array_equal(row[5], results[0][5])
        for a, b in ((0, 2), (1, 3)): assert np.array_equal(results[a][2], results[b][2]) and np.array_equal(results[a][3], results[b][3])
        assert np.allclose(results[1][2][300:494] - results[0][2][300:494], [0., 0., .5])
        assert np.array_equal(results[0][6], of) and np.allclose(np.ptp(results[0][2][300:494], axis=0), renderer.BOTTLE_SCALE * np.ptp(bottle, axis=0))
        assert not np.array_equal(results[0][4][:300], results[2][4][:300])
    assert np.array_equal(h, before) and renderer.ELBOW_NAME == "l_elbow"


def test_human_only_truth_does_not_relabel_nuisance_or_background(renderer, monkeypatch):
    monkeypatch.setattr(renderer, "WIDTH", 24); monkeypatch.setattr(renderer, "HEIGHT", 12)
    faces = np.array([[0, 1, 2]], np.int32); of = np.array([[0, 1, 2]], np.int64)
    face = np.full((12, 24), -1, np.int64); face[:4] = 0; face[4:8] = 1; z = np.full((12, 24), 3., np.float32)
    ids, depth, visible = renderer.visible_truth(face, z, faces, of)
    assert ids.dtype == np.int64 and depth.dtype == np.float32 and visible.dtype == np.bool_
    assert np.array_equal(ids[:4], face[:4]) and np.all(ids[4:] == -1) and np.isnan(depth[4:]).all() and np.all(depth[:4] == 3)
    back = visible.copy(); back[4:8] = True
    assert renderer.occlusion_evidence(visible, back)["newly_visible_human_pixels"] == 96
    with pytest.raises(ValueError): renderer.occlusion_evidence(visible, visible)
    with pytest.raises(ValueError): renderer.visible_truth(np.full_like(face, 0), z, faces, of)
    bad = z.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError): renderer.visible_truth(face, bad, faces, of)


def test_native_rig_parity_meters_and_no_alignment(renderer):
    v = np.zeros((6, 18439, 3), np.float32); sk = np.zeros((6, 127, 8), np.float32); sk[..., 6:] = 1
    assert renderer.parity(v, sk, v.copy(), sk.copy()) == dict(vertices_m=0., joints_m=0.)
    shifted = v.copy(); shifted[0, 0, 0] = .01
    with pytest.raises(ValueError): renderer.parity(v, sk, shifted, sk)
    shifted = sk.copy(); shifted[0, 0, 0] = .01
    with pytest.raises(ValueError): renderer.parity(v, sk, v, shifted)


def test_exact_bundled_asset_receipt_without_loading_checkpoint(renderer, tmp_path, monkeypatch):
    root = tmp_path; checkpoint = root / renderer.render.semantics.BODY_RELATIVE; checkpoint.mkdir(parents=True)
    (root / "results").mkdir(); data = dict(assets=[dict(repo_id="facebook/sam-3d-body-dinov3", revision=renderer.render.semantics.BODY_REVISION, path=str(checkpoint))])
    (root / "results/weights-acquisition.json").write_text(json.dumps(data)); checks = []
    monkeypatch.setattr(renderer.render.semantics, "regular_hash", lambda *args: checks.append(args))
    model, bundled, identity = renderer.rig_sources(root)
    assert model == root / "weights/mhr/mhr_model.pt" and bundled == checkpoint / "assets/mhr_model.pt"
    assert identity["model_sha256"] == identity["bundled_model_sha256"] == renderer.MODEL_SHA and identity["model_bytes"] == 696110248
    assert checks[0][2:] == checks[1][2:] == (renderer.MODEL_SHA, 696110248)
    data["assets"][0]["revision"] = "a" * 40; (root / "results/weights-acquisition.json").write_text(json.dumps(data))
    with pytest.raises(ValueError): renderer.rig_sources(root)


def test_private_schema_and_asset_only_wrapper_firewall(renderer):
    assert renderer.TRUTH_KEYS == {"human_vertices_camera_m", "human_faces", "human_joints_camera_m", "camera_K", "scene_depth_m",
        "visible_face_indices", "human_visibility", "group_index", "frame_index"}
    wrapper = REPO / "infra/run_human_photometric_prepare.sh"; text = wrapper.read_text(); subprocess.run(["bash", "-n", str(wrapper)], check=True)
    assert subprocess.run(["bash", str(wrapper), "--resume"], capture_output=True).returncode == 2
    assert text.count("docker run") == 1 and "123s" in text and "--gpus all --network none --memory 32g --cpus 4" in text
    assert renderer.IMAGE in text and "assets/mhr_model.pt,dst=" in text and "weights/mhr/mhr_model.pt,dst=" in text
    assert "model.ckpt" not in text and "grounding_dino" not in text and "vendor/video_to_data" not in text
    tree = ast.parse(Path(renderer.__file__).read_text())
    imports = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert not imports & {"torch", "PIL", "pytorch3d", "factorial_rgb_render", "human_photometric_evaluate", "hand_synthetic_infer"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("load", "process_one_image") and isinstance(n.value, ast.Name) and n.value.id in ("torch", "np") for n in ast.walk(tree))


def test_runtime_archive_is_complete_bounded_and_has_no_observer(renderer):
    spec = importlib.util.spec_from_file_location("h101_bundle", REPO / "infra/azure_job.py"); launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    paths = subprocess.check_output(["git", "ls-files", "-c", "-o", "--exclude-standard", "infra", "src", "configs", "pyproject.toml"], cwd=REPO, text=True).splitlines()
    files = {p: (REPO / p).read_bytes() for p in set(paths) if (REPO / p).is_file() and not (REPO / p).is_symlink()}
    selected = launcher.runtime_bundle_paths(files, "infra/run_human_photometric_prepare.sh"); output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as target:
        for path in selected:
            member = tarfile.TarInfo(path); member.size = len(files[path]); target.addfile(member, io.BytesIO(files[path]))
    assert len(launcher.encoded_runtime_archive(output.getvalue())[0]) < 160000
    assert "infra/human_photometric_render.py" in selected and "infra/human_photometric_protocol.py" in selected
    assert not any("factorial_rgb" in path and path.startswith("infra/") for path in selected)
