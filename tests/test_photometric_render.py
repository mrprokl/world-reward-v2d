"""Small public PNG/geometry fixtures; no remote assets or native execution."""
import ast
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "infra")); monkeypatch.syspath_prepend(str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("photometric_render_test", REPO / "infra/photometric_render.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def header(width=1024, height=768, color=2):
    data = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + data + struct.pack(">I", zlib.crc32(b"IHDR" + data) & 0xffffffff)


def public_fixture(module, tmp_path):
    directory = tmp_path / "inputs"; directory.mkdir(); png = directory / "frame_000.png"
    png.write_bytes(header() + b"own_tiny_RGB_header_fixture"); png.chmod(0o444)
    data = module.public_manifest(module.public_identity(png)["sha256"])
    def save():
        path = directory / "manifest.json"
        if path.exists(): path.chmod(0o644)
        path.write_text(json.dumps(data)); path.chmod(0o444)
    save(); return directory, data, save


def test_one_exact_public_rgb_no_manufacturing_fields(module, tmp_path):
    directory, data, _ = public_fixture(module, tmp_path)
    rows, receipt = module.public_inputs(directory)
    assert len(rows) == 1 and set(rows[0]) == {"file", "sha256", "width", "height", "path"}
    assert receipt == module.public_identity(directory / "manifest.json")
    assert data == {"schema": "world-reward-photometric-native-v1", "images": [dict(file="frame_000.png", sha256=rows[0]["sha256"], width=1024, height=768)]}
    assert module.BASE == "validation/photometric_native_v1" and module.BUDGET == 120
    assert not {"camera_K", "parameters", "identity", "labels", "depth", "truth"} & set(data)


@pytest.mark.parametrize("fault", ["schema", "private", "rowprivate", "count", "filename", "booldim", "dim", "sha", "tamper", "writable", "extra", "symlink", "alias", "grayscale", "crc"])
def test_strict_public_firewall_and_rgb_integrity(module, tmp_path, fault):
    directory, data, save = public_fixture(module, tmp_path); png = directory / "frame_000.png"
    if fault == "schema": data["schema"] = "world-reward-factorial-rgb-v1"
    elif fault == "private": data["camera_K"] = module.K
    elif fault == "rowprivate": data["images"][0]["identity"] = [.23, -.16]
    elif fault == "count": data["images"] *= 2
    elif fault == "filename": data["images"][0]["file"] = "../frame_000.png"
    elif fault == "booldim": data["images"][0]["width"] = True
    elif fault == "dim": data["images"][0]["height"] = 1024
    elif fault == "sha": data["images"][0]["sha256"] = "a" * 63
    elif fault == "extra": (directory / "geometry.npz").write_bytes(b"forbidden")
    elif fault == "alias":
        alias = tmp_path / "alias"; alias.symlink_to(directory, target_is_directory=True); directory = alias
    elif fault == "symlink":
        target = tmp_path / "image.png"; target.write_bytes(png.read_bytes()); target.chmod(0o444)
        png.unlink(); png.symlink_to(target)
    else:
        png.chmod(0o644)
        if fault == "tamper": png.write_bytes(png.read_bytes() + b"changed")
        elif fault == "grayscale": png.write_bytes(header(color=0) + b"fixture")
        elif fault == "crc": png.write_bytes(png.read_bytes()[:29] + bytes(4) + b"fixture")
        if fault != "writable": png.chmod(0o444)
        if fault in ("grayscale", "crc"): data["images"][0]["sha256"] = module.public_identity(png)["sha256"]
    save()
    with pytest.raises(ValueError): module.public_inputs(directory)


def metadata():
    names = ["unused" + str(i) for i in range(249)]
    names[:7] = ["l_uparm_ry", "l_elbow_bend", "r_uparm_ry", *["l_" + f + "1_rz" for f in ("index", "middle", "ring", "pinky")]]
    bounds = np.tile([-1., 1.], (249, 1)); bounds[136:204] = [-.1, .1]
    return names, bounds


def test_fixed_recipe_obeys_actual_bounds_and_locked_zero_without_clipping(module):
    names, bounds = metadata(); locked = np.array([0, 1, 2, 11, 12, 15, 16]); bounds[136 + locked] = 0
    before = bounds.copy(); p, shape = module.named_controls(names, bounds)
    assert p.dtype == shape.dtype == np.float32 and p.shape == (1, 204) and shape.shape == (1, 45)
    assert np.array_equal(shape[0, :2], np.array([.23, -.16], np.float32))
    assert np.all(p[0, 136 + locked] == 0) and np.all(p[0, 136:][np.any(bounds[136:204] != 0, axis=1)] == np.float32(.02))
    assert p[0, names.index("l_uparm_ry")] == np.float32(.21) and p[0, names.index("l_elbow_bend")] == np.float32(.36)
    assert p[0, names.index("r_uparm_ry")] == np.float32(-.015) and np.array_equal(bounds, before)


@pytest.mark.parametrize("fault", ["masked", "duplicate", "missing", "inverted", "narrow", "locked", "shape", "nan"])
def test_fixed_recipe_has_no_parameter_repair(module, fault):
    names, bounds = metadata()
    if fault == "masked": bounds = np.ma.array(bounds, mask=False)
    elif fault == "duplicate": names[-1] = names[0]
    elif fault == "missing": names[0] = "absent"
    elif fault == "inverted": bounds[150] = [.1, -.1]
    elif fault == "narrow": bounds[150] = [-.01, .01]
    elif fault == "locked": bounds[136:204] = 0
    elif fault == "shape": bounds[204] = [-.1, .1]
    else: bounds[0, 0] = np.nan
    with pytest.raises(ValueError): module.named_controls(names, bounds)


def meshes():
    index = np.linspace(0., 1., 18439)
    human = np.c_[.22 * np.sin(index * 29), index * 1.7 - .85, .04 * np.cos(index * 13)]
    regions = {"l": {"vertex_mask": np.arange(len(human)) < 200}, "r": {"vertex_mask": np.arange(len(human)) > len(human) - 201}}
    return human, regions


def test_transient_scene_fixed_camera_stripes_bottle_and_inputs_unchanged(module):
    human, regions = meshes(); before = human.copy(); faces = np.array([[0, 1, 2]], np.int32)
    v, f, c, K, nf, no = module.scene(human, human, faces, regions)
    assert v.shape[1:] == (3,) and f.shape[1:] == (3,) and c.shape == v.shape
    assert np.array_equal(K, module.K) and nf == 1 and no == 384 and np.isfinite(v).all() and v[:, 2].min() > .01
    pixels = module.render.project_camera_points(v[:18439], K)
    assert np.all(pixels >= 8) and np.all(pixels < [1016, 760])
    bottle, _ = module.primitives.bottle_mesh()
    assert np.allclose(np.ptp(v[18439:18439 + 194], axis=0), 1.3 * np.ptp(bottle, axis=0))
    assert len(np.unique(c[:18439], axis=0)) == 3 and np.array_equal(human, before)
    shifted = module.scene(human + [2., 3., 4.], human + [2., 3., 4.], faces, regions)
    assert np.allclose(shifted[0], v) and np.array_equal(shifted[3], K)


def test_scene_rejects_masked_missing_regions_and_bad_framing(module):
    human, regions = meshes(); faces = np.array([[0, 1, 2]], np.int32)
    with pytest.raises(ValueError): module.scene(np.ma.array(human, mask=False), human, faces, regions)
    bad = human.copy(); bad[:, 0] *= 20
    with pytest.raises(ValueError): module.scene(bad, human, faces, regions)
    regions["l"]["vertex_mask"][:] = False
    with pytest.raises(ValueError): module.scene(human, human, faces, regions)


def test_wrapper_three_isolated_stages_and_no_truth_exports(module):
    path = REPO / "infra/run_photometric_capability.sh"; text = path.read_text()
    subprocess.run(["rtk", "proxy", "bash", "-n", str(path)], check=True)
    assert subprocess.run(["rtk", "proxy", "bash", str(path), "--retry"], capture_output=True).returncode == 2
    assert text.count("docker run") == 3 and text.count("--stage masks") == text.count("--stage body") == 1
    assert "123s" in text and "183s" in text and "--network none --memory 32g --cpus 4" in text
    assert module.IMAGE in text and "53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1" in text
    mask = text.split("phase=masks")[1].split("phase=body")[0]
    body = text.split("phase=body")[1]
    assert "src=$BASE,dst=$BASE" not in mask + body and "weights/mhr" not in mask + body
    assert "src=$BASE/inputs,dst=$BASE/inputs,readonly" in mask and "src=$BASE/inputs,dst=$BASE/inputs,readonly" in body
    assert "src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly" in body
    assert "eval_private" not in text and "historical" not in text and "HF_HUB_OFFLINE=1" in text
    tree = ast.parse(Path(module.__file__).read_text())
    imported = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert not {"torch", "pytorch3d", "PIL", "body_smoke", "hand_synthetic_infer", "factorial_rgb_render"} & imported
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("savez", "savez_compressed", "process_one_image") for n in ast.walk(tree))
    assert '"inputs", "render-report.json"' in Path(module.__file__).read_text()
