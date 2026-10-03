"""Tiny text-file/hash contracts only; no actual images or private data."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "infra/tudl_holdout_inputs.py"
    spec = importlib.util.spec_from_file_location("wr_holdout_inputs_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture(gate, tmp_path):
    directory = tmp_path / "validation/tudl_frame_holdout_v1/inputs"
    directory.mkdir(parents=True)
    rows = []
    for scene, frames in enumerate(gate.FRAME_IDS, 1):
        for frame in frames:
            name = f"scene_{scene:06d}_frame_{frame:06d}.png"
            data = f"tiny undecoded public bytes {scene}:{frame}".encode()
            path = directory / name; path.write_bytes(data); path.chmod(0o444)
            rows.append({"scene_id": scene, "frame_id": frame, "file": name,
                "sha256": hashlib.sha256(data).hexdigest(), "width": 640, "height": 480})
    manifest = {"schema": gate.SCHEMA, "revision": gate.REVISION, "license": gate.LICENSE,
        "selection": gate.SELECTION, "images": rows}
    path = directory / "manifest.json"; path.write_text(json.dumps(manifest)); path.chmod(0o444)
    pins = {"schema": gate.PINS_SCHEMA, "acquisition_report": copy.deepcopy(gate.ACQUISITION),
        "public_files": {p.name: gate.identity(p) for p in directory.iterdir()}}
    return directory, pins, manifest


def write_manifest(gate, directory, pins, manifest=None, raw=None):
    path = directory / "manifest.json"; path.chmod(0o644)
    path.write_bytes(raw if raw is not None else json.dumps(manifest).encode()); path.chmod(0o444)
    pins["public_files"]["manifest.json"] = gate.identity(path)


def test_public_inputs_exact_order_paths_manifest_identity_no_private_access(gate, tmp_path, monkeypatch):
    directory, pins, manifest = fixture(gate, tmp_path)
    original_open = Path.open
    def only_public(self, *args, **kwargs):
        assert self.parent == directory, f"Private/sibling path opened: {self}"
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", only_public)
    records, identity = gate.public_inputs(directory, pins)
    assert [{k: v for k, v in row.items() if k != "path"} for row in records] == manifest["images"]
    assert len(records) == 12 and [row["path"] for row in records] == [directory / name for name in gate.filenames()]
    assert identity == pins["public_files"]["manifest.json"]
    identity["bytes"] = -1
    assert pins["public_files"]["manifest.json"]["bytes"] > 0


@pytest.mark.parametrize("fault", ["none", "notdict", "schema", "missing_receipt", "receipt_sha", "receipt_bytes", "receipt_bool", "producer",
    "script", "receipt_extra", "files_missing", "files_extra", "path_key", "identity_extra", "identity_sha", "identity_upper",
    "identity_short", "identity_zero", "identity_negative", "identity_bool", "identity_float"])
def test_pins_explicit_exact_actual_source_metadata_and_all13_required(gate, tmp_path, fault):
    _, pins, _ = fixture(gate, tmp_path)
    if fault == "none": pins = None
    elif fault == "notdict": pins = []
    elif fault == "schema": pins["schema"] = "legacy"
    elif fault == "missing_receipt": pins.pop("acquisition_report")
    elif fault == "receipt_sha": pins["acquisition_report"]["sha256"] = "0" * 64
    elif fault == "receipt_bytes": pins["acquisition_report"]["bytes"] += 1
    elif fault == "receipt_bool": pins["acquisition_report"]["bytes"] = True
    elif fault == "producer": pins["acquisition_report"]["producer_revision"] = "0" * 40
    elif fault == "script": pins["acquisition_report"]["script_sha256"] = "0" * 64
    elif fault == "receipt_extra": pins["acquisition_report"]["first_seen_trust"] = True
    elif fault == "files_missing": pins["public_files"].pop(gate.filenames()[0])
    elif fault == "files_extra": pins["public_files"]["depth.png"] = pins["public_files"]["manifest.json"]
    elif fault == "path_key": pins["public_files"]["../manifest.json"] = pins["public_files"].pop("manifest.json")
    else:
        identity = pins["public_files"][gate.filenames()[0]]
        if fault == "identity_extra": identity["trusted"] = True
        elif fault == "identity_sha": identity["sha256"] = None
        elif fault == "identity_upper": identity["sha256"] = "A" * 64
        elif fault == "identity_short": identity["sha256"] = "a" * 63
        elif fault == "identity_zero": identity["bytes"] = 0
        elif fault == "identity_negative": identity["bytes"] = -1
        elif fault == "identity_bool": identity["bytes"] = True
        else: identity["bytes"] = float(identity["bytes"])
    with pytest.raises(ValueError): gate.validate_pins(pins)


@pytest.mark.parametrize("fault", ["schema", "revision", "license", "selection", "calibration", "nine", "duplicate", "order", "development",
    "scene_bool", "frame_float", "grid", "sha", "file", "row_extra", "row_missing"])
def test_manifest_exact_new12_full_ids_no_private_fields_or_adaptation(gate, tmp_path, fault):
    directory, pins, manifest = fixture(gate, tmp_path)
    if fault == "schema": manifest["schema"] = "world-reward-tudl-rgb-v1"
    elif fault == "revision": manifest["revision"] = "0" * 40
    elif fault == "license": manifest["license"] = "CC-BY-NC-4.0"
    elif fault == "selection": manifest["selection"] = "new_custom_selection"
    elif fault == "calibration": manifest["cam_K"] = [800, 0, 320]
    elif fault == "nine": manifest["images"] = manifest["images"][:9]
    elif fault == "duplicate": manifest["images"][-1] = manifest["images"][0]
    elif fault == "order": manifest["images"].reverse()
    elif fault == "development": manifest["images"][0]["frame_id"] = 0
    elif fault == "scene_bool": manifest["images"][0]["scene_id"] = True
    elif fault == "frame_float": manifest["images"][0]["frame_id"] = 1788.0
    elif fault == "grid": manifest["images"][0]["width"] = 639
    elif fault == "sha": manifest["images"][0]["sha256"] = "0" * 64
    elif fault == "file": manifest["images"][0]["file"] = "../calibration.json"
    elif fault == "row_extra": manifest["images"][0]["object_id"] = "private"
    else: manifest["images"][0].pop("height")
    write_manifest(gate, directory, pins, manifest)
    with pytest.raises(ValueError): gate.public_inputs(directory, pins)


@pytest.mark.parametrize("raw", [b'{"schema":"a","schema":"b"}', b'{"images":[{"x":1,"x":2}]}',
    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}', b'null', b'[]', b'notJSON'])
def test_duplicate_and_nonfinite_json_never_accepted(gate, tmp_path, raw):
    directory, pins, _ = fixture(gate, tmp_path); write_manifest(gate, directory, pins, raw=raw)
    with pytest.raises(ValueError): gate.public_inputs(directory, pins)


@pytest.mark.parametrize("fault", ["sha", "size", "content", "missing", "extra", "directory", "write", "symlink", "hardlink", "relative", "wrong_namespace", "parent_symlink"])
def test_original_files_canonical_readonly_no_aliases_or_extras(gate, tmp_path, fault):
    directory, pins, _ = fixture(gate, tmp_path); target = directory / gate.filenames()[0]
    if fault == "sha": pins["public_files"][target.name]["sha256"] = "0" * 64
    elif fault == "size": pins["public_files"][target.name]["bytes"] += 1
    elif fault == "content": target.chmod(0o644); target.write_bytes(b"tampered"); target.chmod(0o444)
    elif fault == "missing": target.unlink()
    elif fault == "extra": (directory / "camera.json").write_text("private notallowed")
    elif fault == "directory": target.unlink(); target.mkdir()
    elif fault == "write": target.chmod(0o644)
    elif fault == "symlink": target.unlink(); target.symlink_to(directory / gate.filenames()[1])
    elif fault == "hardlink": os.link(target, tmp_path / "alias")
    elif fault == "relative": directory = Path("validation/tudl_frame_holdout_v1/inputs")
    elif fault == "wrong_namespace": directory.rename(directory.with_name("wrong")); directory = directory.with_name("wrong")
    else:
        alias = tmp_path / "alias"; alias.symlink_to(tmp_path / "validation"); directory = alias / "tudl_frame_holdout_v1/inputs"
    with pytest.raises(ValueError): gate.public_inputs(directory, pins)


def test_every_original_hash_before_first_json_and_every_hash_after(gate, tmp_path, monkeypatch):
    directory, pins, _ = fixture(gate, tmp_path)
    hashed = []; native_identity = gate.identity; native_json = gate.strict_json
    def identity(path): hashed.append(Path(path).name); return native_identity(path)
    def parse(data):
        assert len(hashed) == 13 and set(hashed) == set(pins["public_files"])
        return native_json(data)
    monkeypatch.setattr(gate, "identity", identity); monkeypatch.setattr(gate, "strict_json", parse)
    gate.public_inputs(directory, pins)
    assert len(hashed) == 26 and hashed[:13] == hashed[13:]


def test_wrong_rgb_hash_fails_before_any_json_read(gate, tmp_path, monkeypatch):
    directory, pins, _ = fixture(gate, tmp_path); pins["public_files"][gate.filenames()[-1]]["sha256"] = "0" * 64
    monkeypatch.setattr(gate, "strict_json", lambda *_: pytest.fail("No JSON before all public hashes match"))
    with pytest.raises(ValueError, match="SHA/bytes"): gate.public_inputs(directory, pins)


@pytest.mark.parametrize("fault", ["rgb_mutation", "manifest_mutation", "extra", "pins_mutation"])
def test_post_reread_detects_changes_during_json_read(gate, tmp_path, monkeypatch, fault):
    directory, pins, _ = fixture(gate, tmp_path); native_json = gate.strict_json
    def parse(data):
        result = native_json(data)
        if fault in ("rgb_mutation", "manifest_mutation"):
            target = directory / (gate.filenames()[0] if fault == "rgb_mutation" else "manifest.json")
            target.chmod(0o644); target.write_bytes(b"mutated during read"); target.chmod(0o444)
        elif fault == "extra": (directory / "extra").write_text("unexpected")
        else: pins["public_files"][gate.filenames()[0]]["bytes"] += 1
        return result
    monkeypatch.setattr(gate, "strict_json", parse)
    with pytest.raises(ValueError, match="changed"): gate.public_inputs(directory, pins)


def test_manifest_changed_after_first_hash_rejected_before_json(gate, tmp_path, monkeypatch):
    directory, pins, _ = fixture(gate, tmp_path); native_read = Path.read_bytes
    def read(self):
        data = native_read(self)
        return data + b" " if self.name == "manifest.json" else data
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(gate, "strict_json", lambda *_: pytest.fail("Byte identity must be rechecked before JSON"))
    with pytest.raises(ValueError, match="before JSON"): gate.public_inputs(directory, pins)


def test_import_in_fresh_process_stdlib_only():
    infra = Path(__file__).resolve().parents[1] / "infra"
    code = f"""import sys
sys.path.insert(0,{str(infra)!r})
import tudl_holdout_inputs as h
assert not any(name.split('.')[0] in {{'numpy','torch','joblib','h5py','PIL','cv2','tudl_acquire'}} for name in sys.modules)
assert h.FRAME_IDS == ((1788,3235,5138,6925),(1566,3137,4894,6490),(1512,3227,4819,6647))
"""
    result = subprocess.run(["rtk", "proxy", sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
