"""Public RGB contract tests, using tiny PNG header fixtures only."""
import ast
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol(monkeypatch):
    spec = importlib.util.spec_from_file_location("human_photometric_protocol_test", REPO / "infra/human_photometric_protocol.py")
    value = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, spec.name, value); spec.loader.exec_module(value); return value


def header(color=2):
    body = struct.pack(">IIBBBBB", 1024, 768, 8, color, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + body + struct.pack(">I", zlib.crc32(b"IHDR" + body) & 0xffffffff)


def fixture(protocol, tmp_path):
    directory = tmp_path / "inputs"; directory.mkdir(); digests = []
    for index in range(24):
        png = directory / protocol.COHORT.frame_name(index); png.write_bytes(header() + f"own_fixture{index}".encode()); png.chmod(0o444)
        digests.append(protocol.identity(png)["sha256"])
    data = protocol.public_manifest(digests)
    def save():
        path = directory / "manifest.json"
        if path.exists(): path.chmod(0o644)
        path.write_text(json.dumps(data)); path.chmod(0o444)
    save(); return directory, data, save


def test_frozen_public_config_and_stdlib_only(protocol):
    c = protocol.COHORT
    assert (c.base, c.schema, c.groups, c.frames, c.count, c.width, c.height) == ("validation/human_photometric_v1", "world-reward-human-photometric-v1", 8, 3, 24, 1024, 768)
    assert c.fixed_K == ((1280., 0., 512.), (0., 1280., 384.), (0., 0., 1.))
    with pytest.raises(FrozenInstanceError): c.frames = 5
    with pytest.raises(ValueError): protocol.PublicCohort(frames=True)
    with pytest.raises(ValueError): protocol.PublicCohort(base="validation/factorial_rgb_v1")
    tree = ast.parse(Path(protocol.__file__).read_text())
    imports = {n.module for n in tree.body if isinstance(n, ast.ImportFrom)} | {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    assert not imports & {"numpy", "torch", "PIL", "human_photometric_render"}
    assert not {"SHAPES", "SCALES", "OCCLUSION_OFFSETS"} & set(vars(protocol))


def test_all24_ordered_public_records_and_receipt(protocol, tmp_path):
    directory, data, _ = fixture(protocol, tmp_path); rows, receipt = protocol.public_inputs(directory)
    assert len(rows) == 24 and receipt == protocol.identity(directory / "manifest.json")
    assert [(r["group_index"], r["frame_index"]) for r in rows] == [(g, f) for g in range(8) for f in range(3)]
    assert all(set(r) == {"file", "sha256", "width", "height", "path", "group_index", "frame_index"} for r in rows)
    assert set(data) == {"schema", "images"} and rows[-1]["file"] == "group_07_frame_002.png"


@pytest.mark.parametrize("index", [-1, 24, True, 1., None])
def test_original_index_only(protocol, index):
    with pytest.raises(ValueError): protocol.COHORT.frame_name(index)


@pytest.mark.parametrize("fault", ["schema", "label", "rowlabel", "count", "order", "filename", "booldim", "sha", "writable", "tamper", "extra", "alias", "symlink", "grayscale", "crc"])
def test_public_whitelist_no_labels_or_truth_fallback(protocol, tmp_path, fault):
    directory, data, save = fixture(protocol, tmp_path); png = directory / data["images"][0]["file"]
    if fault == "schema": data["schema"] = "world-reward-factorial-rgb-v1"
    elif fault == "label": data["morphology"] = [.18, -.11]
    elif fault == "rowlabel": data["images"][0]["camera_K"] = protocol.FIXED_K
    elif fault == "count": data["images"].pop()
    elif fault == "order": data["images"][0], data["images"][1] = data["images"][1], data["images"][0]
    elif fault == "filename": data["images"][0]["file"] = "../private.npz"
    elif fault == "booldim": data["images"][0]["width"] = True
    elif fault == "sha": data["images"][0]["sha256"] = "a" * 63
    elif fault == "extra": (directory / "labels.npz").write_bytes(b"forbidden")
    elif fault == "alias":
        alias = tmp_path / "alias"; alias.symlink_to(directory, target_is_directory=True); directory = alias
    elif fault == "symlink":
        target = tmp_path / "target.png"; target.write_bytes(png.read_bytes()); target.chmod(0o444); png.unlink(); png.symlink_to(target)
    else:
        png.chmod(0o644)
        if fault == "tamper": png.write_bytes(png.read_bytes() + b"changed")
        elif fault == "grayscale": png.write_bytes(header(color=0) + b"fixture")
        elif fault == "crc": png.write_bytes(png.read_bytes()[:29] + bytes(4) + b"fixture")
        if fault != "writable": png.chmod(0o444)
        if fault in ("grayscale", "crc"): data["images"][0]["sha256"] = protocol.identity(png)["sha256"]
    save()
    with pytest.raises(ValueError): protocol.public_inputs(directory)


@pytest.mark.parametrize("values", [[], ["x"] * 24, ["a" * 64] * 23, tuple(["a" * 64] * 24)])
def test_manifest_requires_complete_sha_list(protocol, values):
    with pytest.raises(ValueError): protocol.public_manifest(values)
