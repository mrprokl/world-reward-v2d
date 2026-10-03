"""Tiny opaque RGB bytes; no dataset, depth, image decoding or model access."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def reader(monkeypatch):
    root = Path(__file__).resolve().parents[1]; monkeypatch.syspath_prepend(str(root / "infra"))
    return importlib.import_module("tum_rgbd_depth_inputs")


def cohort(reader, tmp_path):
    directory = tmp_path / reader.BASE / "inputs"; directory.mkdir(parents=True)
    rows = []
    for scene, rank in reader.ORDERED_FRAMES:
        timestamp = f"{1300000000 + scene * 100 + rank}.123456"
        name = f"scene_{scene:06d}_rgb_{timestamp}.png"; raw = f"tiny original RGB {scene}:{rank}".encode()
        path = directory / name; path.write_bytes(raw); path.chmod(0o444)
        rows.append(dict(scene_id=scene, frame_id=rank, file=name, original_rgb_file="rgb/" + timestamp + ".png",
            timestamp=timestamp, sha256=hashlib.sha256(raw).hexdigest(), width=640, height=480))
    manifest = dict(schema=reader.SCHEMA, dataset=reader.DATASET, license=reader.LICENSE, selection=reader.SELECTION, images=rows)
    path = directory / "manifest.json"; path.write_text(json.dumps(manifest)); path.chmod(0o444)
    pins = dict(schema=reader.PINS_SCHEMA, acquisition_report=dict(sha256="a" * 64, bytes=1, producer_revision="b" * 40, script_sha256="c" * 64),
        ordered_frames=[list(pair) for pair in reader.ORDERED_FRAMES], public_files={path.name: reader.identity(path) for path in directory.iterdir()})
    return directory, pins, manifest


def test_all13_pins_hashed_before_manifest_and_after_no_private_reads(reader, tmp_path, monkeypatch):
    directory, pins, manifest = cohort(reader, tmp_path); hashes = []; native_hash, native_parse, native_open = reader.identity, reader.strict_json, Path.open
    def identity(path): hashes.append(Path(path).name); return native_hash(path)
    def parse(raw): assert len(hashes) == 13; return native_parse(raw)
    def open_public(path, *args, **kwargs): assert path.parent == directory; return native_open(path, *args, **kwargs)
    monkeypatch.setattr(reader, "identity", identity); monkeypatch.setattr(reader, "strict_json", parse); monkeypatch.setattr(Path, "open", open_public)
    records, receipt = reader.public_inputs(directory, pins)
    assert hashes[:13] == hashes[13:] and len(hashes) == 26
    assert [{key: value for key, value in row.items() if key != "path"} for row in records] == manifest["images"]
    assert receipt == pins["public_files"]["manifest.json"]


@pytest.mark.parametrize("fault", ["rank", "bool", "timestamp", "private_name", "extra", "byte_bool", "private_receipt"])
def test_independent_pins_fix_all12_sequence_ranks_and_safe_timestamp_names(reader, tmp_path, fault):
    _, pins, _ = cohort(reader, tmp_path)
    if fault == "rank": pins["ordered_frames"][0][1] = 30
    elif fault == "bool": pins["ordered_frames"][0][0] = True
    elif fault in ("timestamp", "private_name"):
        name = reader.filenames(pins)[0]; pins["public_files"]["scene_000001_rgb_1.2.png" if fault == "timestamp" else "eval_private/depth.png"] = pins["public_files"].pop(name)
    elif fault == "extra": pins["public_files"]["depth.npz"] = dict(sha256="a" * 64, bytes=1)
    elif fault == "byte_bool": pins["acquisition_report"]["bytes"] = True
    else: pins["acquisition_report"]["path"] = "private/acquisition-report.json"
    with pytest.raises(ValueError): reader.validate_pins(pins)


@pytest.mark.parametrize("fault", ["private_camera", "depth_row", "original_path", "timestamp", "frame_rank", "license", "schema", "duplicate_json"])
def test_public_manifest_exact_original_timestamp_path_grid_and_no_labels(reader, tmp_path, fault):
    directory, pins, manifest = cohort(reader, tmp_path)
    if fault == "private_camera": manifest["cam_K"] = ["forbidden"]
    elif fault == "depth_row": manifest["images"][0]["depth"] = "depth/private.png"
    elif fault == "original_path": manifest["images"][0]["original_rgb_file"] = "rgb/9999999999.123456.png"
    elif fault == "timestamp": manifest["images"][0]["timestamp"] = 1300000000.123456
    elif fault == "frame_rank": manifest["images"][0]["frame_id"] = 0
    elif fault == "license": manifest["license"] = "CC-BY-NC-4.0"
    elif fault == "schema": manifest["schema"] = "old_TUDL"
    raw = json.dumps(manifest).encode() if fault != "duplicate_json" else b'{"schema":"x","schema":"y","images":[]}'
    path = directory / "manifest.json"; path.chmod(0o644); path.write_bytes(raw); path.chmod(0o444)
    pins["public_files"][path.name] = reader.identity(path)
    with pytest.raises(ValueError): reader.public_inputs(directory, pins)


@pytest.mark.parametrize("fault", ["wrong_pin", "writable", "symlink", "mutated_after_json"])
def test_immutable_public_bytes_fail_closed_before_and_after(reader, tmp_path, monkeypatch, fault):
    directory, pins, _ = cohort(reader, tmp_path); path = directory / reader.filenames(pins)[0]
    if fault == "wrong_pin":
        pins["public_files"][path.name]["sha256"] = "0" * 64
        monkeypatch.setattr(reader, "strict_json", lambda _: pytest.fail("No JSON before independent file hashes match"))
    elif fault == "writable": path.chmod(0o644)
    elif fault == "symlink": path.unlink(); path.symlink_to(directory / reader.filenames(pins)[1])
    else:
        native = reader.strict_json
        def parse(raw): result = native(raw); path.chmod(0o644); path.write_bytes(b"changed"); path.chmod(0o444); return result
        monkeypatch.setattr(reader, "strict_json", parse)
    with pytest.raises(ValueError): reader.public_inputs(directory, pins)
