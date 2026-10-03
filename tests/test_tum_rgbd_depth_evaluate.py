"""Tiny mocked TUM public/private contracts; no real dataset/model/GPU access."""
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import signal
import types

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "infra")); monkeypatch.syspath_prepend(str(root / "src"))
    return importlib.import_module("tum_rgbd_depth_evaluate")


def fake_input_pins(gate):
    identity = dict(bytes=1, sha256="a" * 64)
    names = [f"scene_{scene:06d}_rgb_{1300000000 + scene * 100 + rank}.123456.png" for scene, rank in gate.inputs.ORDERED_FRAMES]
    return dict(schema=gate.inputs.PINS_SCHEMA, ordered_frames=[list(pair) for pair in gate.inputs.ORDERED_FRAMES],
        acquisition_report={**identity, "producer_revision": "b" * 40, "script_sha256": "c" * 64},
        public_files={name: identity.copy() for name in ["manifest.json", *names]})


@pytest.mark.parametrize("fault", ["schema", "bytes", "producer", "outputs", "coefficient", "private_field"])
def test_complete_independent_prediction_pins_fail_closed(gate, fault):
    public = fake_input_pins(gate); identity = dict(bytes=1, sha256="a" * 64)
    pins = dict(schema="world_reward.tum_rgbd_depth_prediction_pins.v1",
        report={**identity, "producer_revision": "b" * 40, "script_sha256": "c" * 64},
        outputs={Path(name).stem + ".npz": identity.copy() for name in gate.inputs.filenames(public)}, coefficients=[1., 1., 1.])
    gate.validate_pins(pins, public)
    if fault == "schema": pins["schema"] = "old_TUDL"
    elif fault == "bytes": pins["report"]["bytes"] = True
    elif fault == "producer": pins["report"]["producer_revision"] = "b" * 39
    elif fault == "outputs": pins["outputs"].pop(next(iter(pins["outputs"])))
    elif fault == "coefficient": pins["coefficients"][0] = np.nan
    else: pins["private_depth"] = "oracle"
    with pytest.raises(ValueError): gate.validate_pins(pins, public)


def private_fixture(gate, tmp_path, monkeypatch):
    root, code = tmp_path / "remote", tmp_path / "code"; base = root / gate.BASE; public_dir, private = base / "inputs", base / "eval_private"
    public_dir.mkdir(parents=True); private.mkdir(mode=0o700); (private / "source").mkdir(mode=0o700)
    repository = Path(__file__).resolve().parents[1]
    for name in [gate.PROTOCOL, *gate.acquisition_source.SOURCE_FILES]:
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((repository / name).read_bytes()); path.chmod(0o444)
    protocol = json.loads((code / gate.PROTOCOL).read_bytes()); selected, records, required = [], [], {}
    # Tiny media identities replace protocol pins only in this call-boundary mock.
    for sequence in protocol["sequences"]:
        for declared in sequence["selected"]:
            scene, rank = sequence["sequence_id"], declared["rgb_zero_based_rank"]
            for kind in ("rgb", "depth"):
                timestamp = declared[kind + "_timestamp"]; name = f"scene_{scene:06d}_{kind}_{timestamp}.png"
                path = (public_dir if kind == "rgb" else private) / name; path.write_bytes(f"tiny opaque {kind} {scene}:{rank}".encode()); path.chmod(0o444 if kind == "rgb" else 0o400)
                identity = gate.inputs.identity(path); declared[kind].update(identity)
                required[("inputs/" if kind == "rgb" else "eval_private/") + name] = identity
                selected.append(dict(file=name, kind=kind, original_file=sequence["name"] + "/" + declared[kind]["path"], scene_id=scene, frame_id=rank, timestamp=timestamp, **identity))
            records.append(dict(scene_id=scene, frame_id=rank, timestamp=declared["rgb_timestamp"], original_rgb_file=declared["rgb"]["path"], sha256=declared["rgb"]["sha256"]))
    manifest = public_dir / "manifest.json"; manifest.write_bytes(b"tiny public manifest"); manifest.chmod(0o444); required["inputs/manifest.json"] = gate.inputs.identity(manifest)
    license_rows = []
    source_names = ["TUM-license-section.html", *[sequence["name"] + "-description.html" for sequence in protocol["sequences"]], "attribution.json"]
    for name in source_names:
        raw = json.dumps(dict(dataset="TUM RGB-D", license="CC-BY-4.0", license_url=protocol["primary"]["license_section"]["license_url"],
            attribution=protocol["primary"]["attribution"], modification="Filename-only selected original RGB/depth subset; PNG bytes unmodified",
            publisher_url=protocol["primary"]["dataset_url"], sequence_specific_terms_checked=True)).encode() if name == "attribution.json" else b"tiny source evidence"
        path = private / "source" / name; path.write_bytes(raw); path.chmod(0o400); identity = gate.inputs.identity(path)
        required["eval_private/source/" + name] = identity; license_rows.append(dict(file="source/" + name, **identity))
        if name == "TUM-license-section.html": protocol["primary"]["license_section"].update(identity)
    descriptions = {sequence["name"]: gate.inputs.identity(private / "source" / (sequence["name"] + "-description.html")) for sequence in protocol["sequences"]}
    monkeypatch.setattr(gate.acquisition_source, "DESCRIPTION_IDS", descriptions)
    archives = [dict(sequence_id=sequence["sequence_id"], url=sequence["archive"]["url"], bytes=sequence["archive"]["bytes"], sha256="d" * 64,
        sha256_independently_preknown=False, sha256_is_first_observed_reproducibility_digest=True,
        nearest_timestamp_pairs_verified=True, selected8_independent_byte_pins_verified=True, archive_license_files_present=False, term_files=[],
        filename_inventories={kind: dict(files=sequence[kind + "_file_count"], sha256="e" * 64) for kind in ("rgb", "depth")}) for sequence in protocol["sequences"]]
    acquisition = dict(stage=gate.acquisition_source.STAGE, status="pass", phase="complete", producer_revision="b" * 40,
        script_sha256=gate.inputs.identity(code / gate.acquisition_source.SOURCE_FILES[0])["sha256"],
        source_helpers={name: gate.inputs.identity(code / name) for name in gate.acquisition_source.SOURCE_FILES}, protocol_identity=gate.PROTOCOL_ID,
        budget_seconds=600, device="cpu", gpu_used=False, inference_performed=False, challenge_inputs_used=False,
        source_camera_or_trajectory_read=False, depth_values_decoded=False, ground_truth_used_for_inference=False,
        independent_full_archive_SHA256_known=False, training_overlap_verified=False, challenge_overlap_verified=False, accuracy_verified=False,
        all24_original_files_hashed_before_depth_values=True, all24_retained_bytes_rehashed_before_public_manifest=True,
        disposable_archives_and_staging_removed=True, outputs_rehashed_after=True, source_helpers_rehashed_after=True,
        images_completed=12, private_depths_completed=12, selected_records=selected, archives=archives, license_evidence=license_rows, output_files=required)
    receipt = private / "acquisition-report.json"; receipt.write_text(json.dumps(acquisition)); receipt.chmod(0o400)
    public_pins = {Path(name).name: identity for name, identity in required.items() if name.startswith("inputs/")}
    public = dict(input_pins=dict(public_files=public_pins, acquisition_report={**gate.inputs.identity(receipt), "producer_revision": "b" * 40,
        "script_sha256": acquisition["script_sha256"]}), input_manifest=required["inputs/manifest.json"])
    original = gate.pinned_json
    monkeypatch.setattr(gate, "pinned_json", lambda path, expected: copy.deepcopy(protocol) if path == code / gate.PROTOCOL else original(path, expected))
    return root, code, public, records, private, acquisition


def test_all_private_byte_pins_and24_original_associations_before_sensor_decode(gate, tmp_path, monkeypatch):
    root, code, public, records, private, _ = private_fixture(gate, tmp_path, monkeypatch)
    original = gate.inputs.strict_json; bytechecked = []; native = gate.inputs.identity
    def identity(path): bytechecked.append(Path(path)); return native(path)
    def parse(raw):
        if b'"dataset": "TUM RGB-D"' in raw:
            assert all(private / name in bytechecked for name in ["source/TUM-license-section.html", "source/attribution.json"])
            assert sum(path.parent == private and "_depth_" in path.name for path in bytechecked) >= 12
        return original(raw)
    monkeypatch.setattr(gate.inputs, "identity", identity); monkeypatch.setattr(gate.inputs, "strict_json", parse)
    depths, frozen, _ = gate.private_inputs(root, code, public, records)
    assert len(depths) == 12 and len(frozen) == 32
    assert [(row["scene_id"], row["frame_id"]) for row in depths] == list(gate.inputs.ORDERED_FRAMES)
    assert len({row["file"] for row in depths}) == 12


@pytest.mark.parametrize("fault", ["private_sha", "extra_private", "decoded_before", "wrong_pair", "missing_source"])
def test_private_provenance_fails_before_any_sensor_values(gate, tmp_path, monkeypatch, fault):
    root, code, public, records, private, acquisition = private_fixture(gate, tmp_path, monkeypatch)
    if fault == "private_sha":
        path = next(path for path in private.iterdir() if "_depth_" in path.name); path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o400)
    elif fault == "extra_private": path = private / "camera.json"; path.write_bytes(b"forbidden camera"); path.chmod(0o400)
    elif fault == "missing_source": (private / "source/TUM-license-section.html").unlink()
    else:
        if fault == "decoded_before": acquisition["depth_values_decoded"] = True
        else: acquisition["selected_records"][0]["frame_id"] = 80
        receipt = private / "acquisition-report.json"; receipt.chmod(0o600); receipt.write_text(json.dumps(acquisition)); receipt.chmod(0o400)
        public["input_pins"]["acquisition_report"].update(gate.inputs.identity(receipt))
    monkeypatch.setattr(gate, "read_sensor_png", lambda _: pytest.fail("No private sensor values before complete byte proof"))
    with pytest.raises((ValueError, FileNotFoundError)): gate.private_inputs(root, code, public, records)


def test_exact_16bit_sensor_decode_no_second_scale_or_resampling(gate, tmp_path, monkeypatch):
    values = np.arange(480 * 640, dtype=np.int32).reshape(480, 640) % 65536
    class Image:
        format = "PNG"; size = (640, 480); mode = "I"
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def __array__(self, dtype=None, copy=None): return values
    path = tmp_path / "sensor.png"; path.write_bytes(b"tiny original IHDR" * 2); path.chmod(0o400)
    import PIL.Image
    monkeypatch.setattr(PIL.Image, "open", lambda _: Image()); calls = []
    monkeypatch.setattr(gate.acquisition_source, "png_header", lambda raw, kind: calls.append((len(raw), kind)))
    raw = gate.read_sensor_png(path)
    assert raw.dtype == np.uint16 and raw.shape == (480, 640) and np.array_equal(raw, values)
    assert calls == [(33, "depth")]


def test_source_numerics_private_order_and_CPU_mounts(gate):
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root / "infra/tum_depth_quality.py").read_bytes()).hexdigest() == gate.QUALITY_SHA
    assert gate.quality.score_frame.__module__ == "tum_depth_quality"
    source = inspect.getsource(gate.main)
    assert source.index("public_predictions(root, code)") < source.index("private_inputs(root, code, public, records)") < source.index("read_sensor_png(")
    assert "original_audit.scoring.score_frame" not in source
    wrapper = (root / "infra/run_tum_rgbd_depth_evaluate.sh").read_text()
    assert "--gpus" not in wrapper and "weights/" not in wrapper and "--network none" in wrapper
    assert "--memory 8g --cpus 4" in wrapper and "183s docker run" in wrapper and "--name" in wrapper
    assert "docker ps -aq" in wrapper and "--kill-after=2s" in wrapper and "trap finish EXIT" in wrapper
