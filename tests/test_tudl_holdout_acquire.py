"""Tiny filename/stub-ZIP tests only; no datasets, network or model imports."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace
import zipfile
import zlib

import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_tudl_holdout_test", infra / "tudl_holdout_acquire.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def filename_inventory(gate):
    names = {}
    for scene, (first, median, last) in gate.DEVELOPMENT_IDS.items():
        ids = [first, *range(first + 1, first + 100), median, *range(median + 1, median + 99), last]
        assert len(ids) == 200
        for frame in reversed(ids): names[f"test/{scene:06d}/rgb/{frame:06d}.png"] = object()
    return names


def rgb_header():
    ihdr = struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def tiny_zip(values):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in values.items(): archive.writestr(name, value)
    return stream.getvalue()


def fake_archives(gate):
    names = filename_inventory(gate); selected = gate.select_rgb_names(names)
    base = {"tudl/camera.json": b'{"private_camera":true}', "tudl/dataset_info.md": b"Hodan BOP CC-BY-SA4",
        "tudl/test_targets_bop19.json": b'["private targets"]'}
    models = {f"{folder}/{name}": b"private tiny mesh fixture" for folder in ("models", "models_eval")
        for name in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")}
    frames = {name: rgb_header() for name in names}
    for scene in (1, 2, 3):
        ids = [frame for s, frame, _ in selected if s == scene]
        for kind in ("camera", "gt", "gt_info"):
            values = {str(frame): ({"private_cam_K": ["not_public"], "depth_scale": 1.}
                if kind == "camera" else [{"obj_id": scene}, {"obj_id": scene}]) for frame in [*ids, 99999]}
            frames[f"test/{scene:06d}/scene_{kind}.json"] = json.dumps(values).encode()
        for frame in ids:
            prefix = f"test/{scene:06d}/"
            frames[prefix + f"depth/{frame:06d}.png"] = b"private sensor placeholder"
            for instance in range(2): frames[prefix + f"mask_visib/{frame:06d}_{instance:06d}.png"] = b"private mask placeholder"
    archives = {}
    for name, values in (("tudl_base.zip", base), ("tudl_models.zip", models), ("tudl_test_bop19.zip", frames)):
        archive = zipfile.ZipFile(io.BytesIO(tiny_zip(values)))
        archives[name] = (archive, gate.source.zip_inventory(archive, [0, 0]))
    return archives, selected


def close_archives(archives):
    for archive, _ in archives.values(): archive.close()


def fake_license(gate, private):
    for name in ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json"):
        gate.source.save_bytes(private / "source/licenses" / name, b"frozen tiny license evidence")
    return [{"url": "https://huggingface.co/pinned", "sha256": "a" * 64}]


def source_namespace(gate, tmp_path, monkeypatch):
    revision = "a" * 40
    code = tmp_path / "jobs" / revision / "run_tudl_holdout_acquire/code"
    for name in gate.HELPERS:
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"# tiny frozen code\n"); path.chmod(0o444)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "__file__", str(code / gate.HELPERS[0]))
    monkeypatch.setattr(gate.source, "__file__", str(code / gate.HELPERS[1]))
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE", str(code))
    monkeypatch.setenv("WR_CODE_REVISION", revision); monkeypatch.setenv("WR_TUDL_HOLDOUT_RESERVED", "1")
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    destination = tmp_path / gate.NAMESPACE; destination.mkdir(parents=True)
    return code, destination


def stub_downloads(gate, monkeypatch):
    archives, _ = fake_archives(gate)
    blobs = {name: tiny_zip({member: archive.read(member) for member in archive.namelist()}) for name, (archive, _) in archives.items()}
    close_archives(archives)
    monkeypatch.setattr(gate.source, "ARCHIVES", {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in blobs.items()})
    def download(url, path, size, sha):
        assert url == gate.source.BASE_URL + path.name
        assert size == len(blobs[path.name]) and sha == hashlib.sha256(blobs[path.name]).hexdigest()
        path.write_bytes(blobs[path.name]); path.chmod(0o400)
    monkeypatch.setattr(gate.source, "download", download)
    monkeypatch.setattr(gate.source, "license_evidence", lambda private: fake_license(gate, private))


def test_filename_only_selection_exact_indices_dynamic_ids_disjoint(gate):
    names = filename_inventory(gate); selected = gate.select_rgb_names(names)
    assert len(selected) == 12
    for scene in (1, 2, 3):
        files = sorted(name for name in names if name.startswith(f"test/{scene:06d}/rgb/"))
        assert [row[2] for row in selected if row[0] == scene] == [files[index] for index in (40, 80, 120, 160)]
        assert not set(gate.DEVELOPMENT_IDS[scene]) & {row[1] for row in selected if row[0] == scene}
    assert not set(selected) & set(gate.source.select_rgb_names(names))
    assert gate.select_rgb_names(dict(reversed(list(names.items())))) == selected


@pytest.mark.parametrize("fault", ["missing", "extra", "old_ids", "wrong_scene", "nonstring", "notdict", "upstream_ids", "indices_overlap"])
def test_selection_rejects_incomplete_changed_development_or_nonpinned_filenames(gate, monkeypatch, fault):
    names = filename_inventory(gate)
    if fault == "missing": names.pop(next(iter(names)))
    elif fault == "extra": names["test/000001/rgb/999999.png"] = None
    elif fault == "old_ids": names["test/000001/rgb/000999.png"] = names.pop("test/000001/rgb/000000.png")
    elif fault == "wrong_scene": names = {k.replace("000001/rgb", "000004/rgb"): v for k, v in names.items()}
    elif fault == "nonstring": names[True] = None
    elif fault == "notdict": names = list(names)
    elif fault == "upstream_ids": monkeypatch.setattr(gate.source, "EXPECTED_IDS", {1: (1, 2, 3)})
    else: monkeypatch.setattr(gate, "INDICES", (0, 80, 120, 160))
    with pytest.raises(ValueError): gate.select_rgb_names(names)


def test_source_pins_and_archive_scope_exact_original_no_new_dataset(gate):
    assert gate.source.REVISION == "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
    assert gate.source.ARCHIVES["tudl_test_bop19.zip"] == (372464733, "cc68c55004dc7822a10910033aedffb3d40bf5bb0a39497b13e566dedddbea14")
    assert sum(row[0] for row in gate.source.ARCHIVES.values()) == 374952356
    assert gate.BUDGET == 600 and gate.INDICES == (40, 80, 120, 160)


def test_full_filename_selection_precedes_every_private_read(gate, tmp_path, monkeypatch):
    archives, selected = fake_archives(gate); checked = []
    original_select = gate.select_rgb_names
    def select(names):
        result = original_select(names); checked.append(tuple(result)); return result
    monkeypatch.setattr(gate, "select_rgb_names", select)
    for archive, _ in archives.values():
        original_read = archive.read
        def read(name, *args, original=original_read, **kwargs):
            assert checked and len(checked[-1]) == 12
            return original(name, *args, **kwargs)
        monkeypatch.setattr(archive, "read", read)
    private, inputs = tmp_path / "eval_private", tmp_path / "inputs"
    private.mkdir(mode=0o700); inputs.mkdir(mode=0o755)
    try: manifest = gate.retain_subset(archives, private, inputs, selected)
    finally: close_archives(archives)
    assert len(checked) == 1 and manifest["schema"] == gate.SCHEMA
    assert len(manifest["images"]) == 12 and len(list(inputs.iterdir())) == 12
    assert all(set(row) == {"scene_id", "frame_id", "file", "sha256", "width", "height"} for row in manifest["images"])
    for scene in (1, 2, 3):
        annotations = json.loads((private / f"source/test/{scene:06d}/scene_gt.json").read_text())
        assert len(annotations) == 4 and "99999" not in annotations
        assert all(len(instances) == 2 for instances in annotations.values())
        assert len(list((private / f"source/test/{scene:06d}/mask_visib").iterdir())) == 8
    attribution = json.loads((private / "source/licenses/attribution-holdout.json").read_text())
    assert attribution["same_development_scenes_and_objects"] is True and attribution["independent_scenes_or_objects"] is False


@pytest.mark.parametrize("fault", ["development", "reverse", "truncated", "path", "bool", "listrow"])
def test_bad_selected_contract_rejects_before_private_reads(gate, tmp_path, fault):
    archives, selected = fake_archives(gate)
    if fault == "development": selected[0] = gate.source.select_rgb_names(archives["tudl_test_bop19.zip"][1])[0]
    elif fault == "reverse": selected.reverse()
    elif fault == "truncated": selected.pop()
    elif fault == "path": selected[0] = (1, selected[0][1], "../private")
    elif fault == "bool": selected[0] = (True, selected[0][1], selected[0][2])
    else: selected[0] = list(selected[0])
    for archive, _ in archives.values(): archive.read = lambda *_: pytest.fail("No private read before selection validation")
    try:
        with pytest.raises(ValueError): gate.retain_subset(archives, tmp_path / "private", tmp_path / "inputs", selected)
    finally: close_archives(archives)


@pytest.mark.parametrize("fault", ["mask", "depth", "camera_key", "gt_instances", "private_license"])
def test_missing_private_instance_or_source_fails_no_silent_drop(gate, tmp_path, fault):
    archives, selected = fake_archives(gate); _, frame, _ = selected[0]
    if fault in ("mask", "depth"):
        suffix = f"mask_visib/{frame:06d}_000001.png" if fault == "mask" else f"depth/{frame:06d}.png"
        del archives["tudl_test_bop19.zip"][1]["test/000001/" + suffix]
    elif fault in ("camera_key", "gt_instances"):
        archive, _ = archives["tudl_test_bop19.zip"]; original_read = archive.read
        def read(name):
            data = original_read(name)
            if name == f"test/000001/scene_{'camera' if fault == 'camera_key' else 'gt'}.json":
                values = json.loads(data)
                if fault == "camera_key": del values[str(frame)]
                else: values[str(frame)] = []
                return json.dumps(values).encode()
            return data
        archive.read = read
    else:
        archive, _ = archives["tudl_base.zip"]; original_read = archive.read
        archive.read = lambda name: b"non-commercial" if name.endswith(".md") else original_read(name)
    try:
        with pytest.raises(ValueError): gate.retain_subset(archives, tmp_path / "private", tmp_path / "inputs", selected)
    finally: close_archives(archives)


def test_main_pass_exact_inventory_source_prepost_and_permissions(gate, tmp_path, monkeypatch):
    _, destination = source_namespace(gate, tmp_path, monkeypatch); stub_downloads(gate, monkeypatch)
    gate.main([])
    path = destination / "eval_private/acquisition-report.json"; receipt = json.loads(path.read_text())
    assert receipt["status"] == "pass" and receipt["phase"] == "complete" and receipt["images_completed"] == 12
    assert receipt["rgb_filename_inventory_count"] == 600 and len(receipt["selected_archive_records"]) == 12
    assert receipt["development_frame_ids_disjoint"] is True
    assert receipt["source_helpers"] == receipt["source_helpers_after"] and receipt["source_helpers_unchanged"] is True
    assert receipt["retained_outputs_unchanged"] is True and receipt["archive_sources_unchanged"] is True
    for field in ("inference_performed", "challenge_inputs_used", "private_annotations_exported_as_inference_inputs",
            "ground_truth_used_for_inference", "training_overlap_verified", "challenge_overlap_verified", "accuracy_verified",
            "independent_scenes_or_objects", "temporal_adjacency_or_acceleration_truth_verified"):
        assert receipt[field] is False
    assert receipt["same_development_scenes_and_objects"] is True
    assert path.stat().st_mode & 0o777 == 0o400
    assert len(receipt["public_files"]) == 13 and len(receipt["retained_files"]) == 60
    assert len(list((destination / "inputs").iterdir())) == 13 and not (destination / "eval_private/.downloads").exists()
    assert all(row["sha256"] == gate.identity(destination / "eval_private" / row["file"])["sha256"] for row in receipt["retained_files"])
    frozen = path.read_bytes()
    with pytest.raises(ValueError, match="empty canonical"): gate.main([])
    assert path.read_bytes() == frozen


@pytest.mark.parametrize("fault", ["private_field", "legacy_schema", "selection", "order", "bool", "sha", "missing", "camera", "selected_bool"])
def test_public_manifest_exact_new_schema_without_private_values(gate, tmp_path, fault):
    archives, selected = fake_archives(gate)
    private, inputs = tmp_path / "private", tmp_path / "inputs"
    private.mkdir(mode=0o700); inputs.mkdir(mode=0o755)
    try: manifest = gate.retain_subset(archives, private, inputs, selected)
    finally: close_archives(archives)
    if fault == "private_field": manifest["cam_K"] = ["private"]
    elif fault == "legacy_schema": manifest["schema"] = "world-reward-tudl-rgb-v1"
    elif fault == "selection": manifest["selection"] = gate.source.SELECTION
    elif fault == "order": manifest["images"].reverse()
    elif fault == "bool": manifest["images"][0]["scene_id"] = True
    elif fault == "sha": manifest["images"][0]["sha256"] = "notsha"
    elif fault == "missing": manifest["images"].pop()
    elif fault == "camera": manifest["images"][0]["camera"] = ["private"]
    else: selected[0] = (True, selected[0][1], selected[0][2])
    with pytest.raises(ValueError): gate.validate_manifest(manifest, selected)


def test_public_retention_restores_umask_and_all_private_parents_are700(gate, tmp_path):
    archives, selected = fake_archives(gate)
    private, inputs = tmp_path / "private", tmp_path / "inputs"
    private.mkdir(mode=0o700); inputs.mkdir(mode=0o755)
    previous = os.umask(0o022)
    try:
        gate.retain_subset(archives, private, inputs, selected)
        actual = os.umask(0o022)
        assert actual == 0o022
        assert (private / "source/base").stat().st_mode & 0o777 == 0o700
        assert (private / "source/base/tudl").stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o700 for path in private.rglob("*") if path.is_dir())
        assert inputs.stat().st_mode & 0o777 == 0o755
    finally:
        os.umask(previous); close_archives(archives)


def test_timeout_main_restores_handlers_umask_and_removes_partial_archives(gate, tmp_path, monkeypatch):
    _, destination = source_namespace(gate, tmp_path, monkeypatch)
    handlers = {gate.signal.SIGALRM: object(), gate.signal.SIGTERM: object()}; original = handlers.copy(); alarms = []
    def signal(number, handler): previous = handlers[number]; handlers[number] = handler; return previous
    monkeypatch.setattr(gate.signal, "signal", signal); monkeypatch.setattr(gate.signal, "alarm", alarms.append)
    monkeypatch.setattr(gate.source, "license_evidence", lambda private: fake_license(gate, private))
    def expired(url, path, size, sha):
        path.write_bytes(b"partial fixture"); handlers[gate.signal.SIGALRM]()
    monkeypatch.setattr(gate.source, "download", expired)
    previous = os.umask(0o022)
    try:
        with pytest.raises(TimeoutError): gate.main([])
        assert os.umask(0o022) == 0o022
    finally: os.umask(previous)
    assert handlers == original and alarms == [600, 0]
    receipt = json.loads((destination / "eval_private/acquisition-report.json").read_text())
    assert receipt["status"] == "fail" and receipt["error_type"] == "TimeoutError"
    assert receipt["disposable_archives_removed"] is True and not (destination / "eval_private/.downloads").exists()


@pytest.mark.parametrize("fault", ["public_extra", "public_write", "private_extra", "empty_directory", "private_write", "private_dir_mode", "symlink", "changed_manifest"])
def test_post_inventory_rejects_extra_symlink_permission_and_manifest_changes(gate, tmp_path, monkeypatch, fault):
    _, destination = source_namespace(gate, tmp_path, monkeypatch); stub_downloads(gate, monkeypatch); gate.main([])
    manifest = json.loads((destination / "inputs/manifest.json").read_text())
    names = filename_inventory(gate); selected = gate.select_rgb_names(names)
    public = destination / "inputs"; private = destination / "eval_private"
    if fault == "public_extra": (public / "camera.json").write_bytes(b"forbidden calibration")
    elif fault == "public_write": (public / manifest["images"][0]["file"]).chmod(0o644)
    elif fault == "private_extra": gate.source.save_bytes(private / "source/extra.txt", b"notselected")
    elif fault == "empty_directory": (private / "source/empty").mkdir(mode=0o700)
    elif fault == "private_write": (private / "source/base/tudl/camera.json").chmod(0o644)
    elif fault == "private_dir_mode": (private / "source/test/000001").chmod(0o755)
    elif fault == "symlink": (private / "source/extra").symlink_to(public)
    else:
        path = public / "manifest.json"; path.chmod(0o644); path.write_text("{}"); path.chmod(0o444)
    with pytest.raises(ValueError): gate.output_inventory(destination, manifest, selected)


@pytest.mark.parametrize("fault", ["download", "archive_hash", "source_mutation"])
def test_failure_keeps_immutable_receipt_cleans_archives_and_never_passes(gate, tmp_path, monkeypatch, fault):
    code, destination = source_namespace(gate, tmp_path, monkeypatch); stub_downloads(gate, monkeypatch)
    original = gate.source.download
    def download(url, path, size, sha):
        if fault == "download": path.write_bytes(b"partial"); raise ValueError("tiny download failure")
        original(url, path, size, sha)
        if fault == "archive_hash": path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o400)
        else:
            helper = code / gate.HELPERS[1]; helper.chmod(0o644); helper.write_bytes(b"# changed\n"); helper.chmod(0o444)
    monkeypatch.setattr(gate.source, "download", download)
    with pytest.raises(ValueError): gate.main([])
    path = destination / "eval_private/acquisition-report.json"; receipt = json.loads(path.read_text())
    assert receipt["status"] == "fail" and receipt["disposable_archives_removed"] is True
    assert not (destination / "eval_private/.downloads").exists() and path.stat().st_mode & 0o777 == 0o400
    if fault == "source_mutation": assert receipt["source_helpers_unchanged"] is False


@pytest.mark.parametrize("fault", ["revision", "wrong_root", "namespace", "executing", "import_path", "helper_write", "helper_symlink"])
def test_source_namespace_actual_launcher_not_implicit_current_checkout(gate, tmp_path, monkeypatch, fault):
    code, _ = source_namespace(gate, tmp_path, monkeypatch)
    root, revision, executing = tmp_path, "a" * 40, code / gate.HELPERS[0]
    if fault == "revision": revision = "a" * 39
    elif fault == "wrong_root": root = tmp_path / "other"
    elif fault == "namespace": code = code.parent / "other"
    elif fault == "executing": executing = code / "other.py"
    elif fault == "import_path": monkeypatch.setattr(gate.source, "__file__", str(tmp_path / "unbound.py"))
    elif fault == "helper_write": (code / gate.HELPERS[1]).chmod(0o644)
    else:
        path = code / gate.HELPERS[1]; path.unlink(); path.symlink_to(code / gate.HELPERS[0])
    with pytest.raises(ValueError): gate.bound_source(root, code, revision, executing)


@pytest.mark.parametrize("fault", ["reservation", "nonempty", "nonlinux", "args"])
def test_main_rejects_before_any_acquisition(gate, tmp_path, monkeypatch, fault):
    _, destination = source_namespace(gate, tmp_path, monkeypatch)
    if fault == "reservation": monkeypatch.delenv("WR_TUDL_HOLDOUT_RESERVED")
    elif fault == "nonempty": (destination / "previous.json").write_text("immutable")
    elif fault == "nonlinux": monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(gate.source, "download", lambda *_: pytest.fail("No network in reject tests"))
    with pytest.raises((ValueError, RuntimeError, SystemExit)): gate.main(["--episode", "0"] if fault == "args" else [])
    assert not (destination / "eval_private").exists()


def test_fresh_process_import_is_stdlib_only_no_network_or_media():
    infra = Path(__file__).resolve().parents[1] / "infra"
    script = f"""import sys
sys.path.insert(0,{str(infra)!r})
import tudl_holdout_acquire
assert not any(name.split('.')[0] in {{'numpy','torch','joblib','h5py','PIL','cv2'}} for name in sys.modules)
assert tudl_holdout_acquire.NAMESPACE == 'validation/tudl_frame_holdout_v1'
"""
    result = subprocess.run(["rtk", "proxy", sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
