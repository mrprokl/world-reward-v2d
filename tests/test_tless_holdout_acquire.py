"""Tiny own ZIP/PNG headers and inventories only; no network/assets/models."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import struct
import subprocess
import zipfile
import zlib

import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("tless_test_gate", infra / "tless_holdout_acquire.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def names(gate, count=200):
    return {f"test_primesense/{scene:06d}/rgb/{frame * 3:06d}.png": object()
        for scene in gate.SCENES for frame in reversed(range(count))}


def png_header(width=720, height=540, color=2):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))


def zip_bytes(values):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in values.items(): archive.writestr(name, value)
    return data.getvalue()


def fake_archives(gate):
    inventory = names(gate, 5); selected = gate.select_rgb_names(inventory)
    base = {"tless/dataset_info.md": b"T-LESS\nLicense: CC-BY-4.0\nAuthors Hodan et al.",
        "tless/camera_primesense.json": b'{"private_camera":true}', "tless/test_targets_bop19.json": b'["private targets"]'}
    test = {name: png_header() for name in inventory}
    for scene in gate.SCENES:
        ids = [frame for s, frame, _ in selected if s == scene]; prefix = f"test_primesense/{scene:06d}/"
        for kind in ("camera", "gt", "gt_info"):
            value = {str(frame): (dict(cam_K=[800., 0., 360., 0., 800., 270., 0., 0., 1.], depth_scale=.1)
                if kind == "camera" else ([dict(obj_id=1), dict(obj_id=2)] if kind == "gt" else [{}, {}])) for frame in (*ids, 999999)}
            test[prefix + f"scene_{kind}.json"] = json.dumps(value).encode()
        for frame in ids:
            test[prefix + f"depth/{frame:06d}.png"] = b"own tiny depth identity"
            for instance in range(2): test[prefix + f"mask_visib/{frame:06d}_{instance:06d}.png"] = b"own tiny mask identity"
    archives = {}
    for name, values in (("tless_base.zip", base), ("tless_test_primesense_bop19.zip", test)):
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes(values)))
        archives[name] = (archive, gate.zip_inventory(archive, [0, 0]))
    return archives, selected


def close(archives):
    for archive, _ in archives.values(): archive.close()


def split(tmp_path):
    destination = tmp_path / "validation/tless_frame_holdout_v1"; destination.mkdir(parents=True)
    private, public = destination / "eval_private", destination / "inputs"
    private.mkdir(mode=0o700); public.mkdir(mode=0o755)
    (private / "source").mkdir(mode=0o700)
    (private / "acquisition-report.json").write_text("{}"); (private / "acquisition-report.json").chmod(0o400)
    return destination, private, public


def fake_license(gate, private):
    for name in ("huggingface-README.md", "BOP-TLESS-section.html", "attribution.json"):
        gate.save_bytes(private / "source/licenses" / name, b"own source licence evidence")
    return {"own_tiny_test": True}


def test_uniform_filename_only_ranks_exact200_and_other_count(gate):
    selected = gate.select_rgb_names(names(gate)); assert len(selected) == 12
    assert gate.selection_ranks(200) == (40, 80, 120, 160)
    assert gate.selection_ranks(11) == (2, 4, 6, 8)
    assert gate.selection_ranks(5) == (1, 2, 3, 4)
    assert [row[1] for row in selected[:4]] == [120, 240, 360, 480]
    assert [row[0] for row in selected] == [1] * 4 + [10] * 4 + [20] * 4
    inventory = names(gate); assert gate.select_rgb_names(dict(reversed(list(inventory.items())))) == selected
    for count in (4, True, 0):
        with pytest.raises(ValueError): gate.selection_ranks(count)


def test_subset_rechecks_selection_before_any_archive_value(gate, tmp_path):
    archives, selected = fake_archives(gate); _, private, public = split(tmp_path)
    class NoRead:
        def read(self, *_): raise AssertionError("Private values read before selection")
    guarded = {name: (NoRead(), members) for name, (_, members) in archives.items()}
    with pytest.raises(ValueError): gate.retain_subset(guarded, private, public, selected[:-1])
    close(archives)


def disk_archives(gate, archives, tmp_path):
    result, pins = {}, {}
    for name, (archive, members) in archives.items():
        raw = zip_bytes({key: archive.read(key) for key in members}); path = tmp_path / name
        path.write_bytes(raw); path.chmod(0o400); item = zipfile.ZipFile(path)
        result[name] = (item, gate.zip_inventory(item, [0, 0]))
        pins[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), git_blob_sha1="a" * 40)
    close(archives)
    return result, pins


def test_all12_original_RGBs_and_all24_masks_private_attribution_public(gate, tmp_path, monkeypatch):
    archives, selected = fake_archives(gate); destination, private, public = split(tmp_path); fake_license(gate, private)
    archives, pins = disk_archives(gate, archives, tmp_path); monkeypatch.setattr(gate, "ARCHIVES", pins)
    try: manifest = gate.retain_subset(archives, private, public, selected)
    finally: close(archives)
    gate.source.save_bytes(public / "manifest.json", (json.dumps(manifest) + "\n").encode(), 0o444)
    hashes, retained = gate.output_inventory(destination, manifest, selected)
    assert len(hashes) == 13 and manifest["attribution"] == gate.ATTRIBUTION
    assert len([row for row in retained if "/mask_visib/" in row["file"]]) == 24
    assert len([row for row in retained if "/depth/" in row["file"]]) == 12
    assert all("models" not in row["file"] and not row["file"].endswith(".ply") for row in retained)
    for row in manifest["images"]:
        assert set(row) == {"scene_id", "frame_id", "file", "sha256", "width", "height"}
        assert (public / row["file"]).read_bytes() == png_header()
    for scene in gate.SCENES:
        values = json.loads((private / f"source/test_primesense/{scene:06d}/scene_gt.json").read_bytes())
        assert set(values) == {str(frame) for s, frame, _ in selected if s == scene}
        assert all(len(value) == 2 for value in values.values())


@pytest.mark.parametrize("text", [b"T-LESS licence: CC-BY-NC-4.0", b"T-LESS research use only", b"T-LESS all rights reserved", b"T-LESS attribution without licence", b"T-LESS License: unknown"])
def test_embedded_missing_or_conflicting_licence_STOP(gate, text):
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes({"tless/dataset_info.md": text})))
    try:
        members = gate.zip_inventory(archive, [0, 0])
        with pytest.raises(ValueError): gate.check_embedded_licenses({"tless_base.zip": (archive, members)})
    finally: archive.close()


@pytest.mark.parametrize("fault", ["depth", "mask", "extra_mask", "info", "camnan", "privateRGBresize"])
def test_missing_or_partial_instances_camera_and_original_grid_fail(gate, tmp_path, monkeypatch, fault):
    archives, selected = fake_archives(gate)
    values = {name: archive.read(name) for archive, _ in archives.values() for name in archive.namelist()}
    close(archives); scene, frame, rgb = selected[-1]; prefix = f"test_primesense/{scene:06d}/"
    if fault == "depth": values.pop(prefix + f"depth/{frame:06d}.png")
    elif fault == "mask": values.pop(prefix + f"mask_visib/{frame:06d}_000001.png")
    elif fault == "extra_mask": values[prefix + f"mask_visib/{frame:06d}_000002.png"] = b"extra"
    elif fault == "privateRGBresize": values[rgb] = png_header(640, 480)
    else:
        kind = "gt_info" if fault == "info" else "camera"; name = prefix + f"scene_{kind}.json"
        data = json.loads(values[name]); data[str(frame)] = [{}] if fault == "info" else {"cam_K": [float("nan")] * 9, "depth_scale": 1.}
        values[name] = json.dumps(data).encode()
    archives = {}
    for name, prefix in (("tless_base.zip", "tless/"), ("tless_test_primesense_bop19.zip", "test_primesense/")):
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes({k: v for k, v in values.items() if k.startswith(prefix)})))
        archives[name] = (archive, gate.zip_inventory(archive, [0, 0]))
    _, private, public = split(tmp_path)
    archives, pins = disk_archives(gate, archives, tmp_path); monkeypatch.setattr(gate, "ARCHIVES", pins)
    try:
        with pytest.raises(ValueError): gate.retain_subset(archives, private, public, selected)
    finally: close(archives)


@pytest.mark.parametrize("name", ["../private", "/absolute", "test_primesense//bad", "test_primesense/./bad", "test_primesense/../bad", "back\\slash"])
def test_zip_unsafe_names_fail_without_read(gate, name):
    archive = zipfile.ZipFile(io.BytesIO(zip_bytes({name: b"own"})))
    try:
        with pytest.raises(ValueError): gate.zip_inventory(archive, [0, 0])
    finally: archive.close()


def test_zip_symlink_budget_and_layout_unknown_models_rejected(gate, monkeypatch):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        info = zipfile.ZipInfo("link"); info.external_attr = (stat.S_IFLNK | 0o777) << 16; archive.writestr(info, "target")
    archive = zipfile.ZipFile(io.BytesIO(data.getvalue()))
    with pytest.raises(ValueError): gate.zip_inventory(archive, [0, 0])
    archive.close(); archive = zipfile.ZipFile(io.BytesIO(zip_bytes({"own": b"abc"})))
    monkeypatch.setattr(gate, "MAX_MEMBER", 2)
    with pytest.raises(ValueError): gate.zip_inventory(archive, [0, 0])
    archive.close()
    with pytest.raises(ValueError): gate.inspect_layout("tless_base.zip", {"tless/models/obj.ply": None})
    with pytest.raises(ValueError): gate.inspect_layout("tless_models.zip", {})


def test_exact_protocol_readonly_beforeJSON_and_pins(gate, tmp_path):
    repo = Path(__file__).resolve().parents[1]; path = tmp_path / gate.PROTOCOL; path.parent.mkdir(parents=True)
    raw = (repo / gate.PROTOCOL).read_bytes(); path.write_bytes(raw); path.chmod(0o444)
    protocol, identity = gate.load_protocol(tmp_path)
    assert protocol["revision"] == gate.REVISION and protocol["archives"] == gate.ARCHIVES
    assert identity["sha256"] == hashlib.sha256(raw).hexdigest()
    for field in ("archives", "selection", "acquisition"):
        wrong = copy.deepcopy(protocol); wrong[field] = {}
        path.chmod(0o600); path.write_text(json.dumps(wrong)); path.chmod(0o444)
        with pytest.raises(ValueError): gate.load_protocol(tmp_path)


def test_archive_hash_pass_before_ZIP_parsing_and_cleanup_on_failure(gate, tmp_path, monkeypatch):
    destination, private, _ = split(tmp_path); (private / "source").rmdir()
    monkeypatch.setattr(gate, "license_evidence", lambda private: fake_license(gate, private))
    def wrong_download(url, path, size, sha): path.write_bytes(b"not ZIP or valid pinned bytes"); path.chmod(0o400)
    monkeypatch.setattr(gate.source, "download", wrong_download)
    def forbid(*_): raise AssertionError("ZIP parsed before complete archive SHA/bytes")
    monkeypatch.setattr(gate.zipfile, "ZipFile", forbid)
    report = {"archives": []}
    with pytest.raises(ValueError): gate.acquire(destination, report, lambda: None)
    assert report["disposable_archives_removed"] and not (private / ".downloads").exists()


def test_complete_stub_acquisition_cleanup_and_full_inventory(gate, tmp_path, monkeypatch):
    archives, selected = fake_archives(gate)
    blobs = {name: zip_bytes({member: archive.read(member) for member in members}) for name, (archive, members) in archives.items()}
    close(archives)
    monkeypatch.setattr(gate, "ARCHIVES", {name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), git_blob_sha1="a" * 40) for name, raw in blobs.items()})
    monkeypatch.setattr(gate, "license_evidence", lambda private: fake_license(gate, private))
    def download(url, path, size, sha):
        raw = blobs[path.name]; assert size == len(raw) and sha == hashlib.sha256(raw).hexdigest()
        path.write_bytes(raw); path.chmod(0o400)
    monkeypatch.setattr(gate.source, "download", download)
    destination, private, _ = split(tmp_path); (private / "source").rmdir(); report = {"archives": []}
    manifest, actual = gate.acquire(destination, report, lambda: None)
    assert actual == selected and len(manifest["images"]) == report["images_completed"] == 12
    assert report["archive_sources_unchanged"] and report["all_selected_object_instances_retained"] and report["embedded_licence_checked"]
    assert report["disposable_archives_removed"] and len(report["public_files"]) == 13


def test_wrapper_cpu_no_install_GPU_full_models_and_no_args(gate):
    path = Path(__file__).resolve().parents[1] / "infra/run_tless_holdout_acquire.sh"; wrapper = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert subprocess.run(["bash", str(path), "--unknown"], capture_output=True).returncode == 2
    assert "--gpus" not in wrapper and "docker " not in wrapper and "pip install" not in wrapper
    assert "603s" in wrapper and "--kill-after=10s" in wrapper and "runuser -u scenesmith" in wrapper
    assert "out.exists()" in wrapper and "WR_TLESS_HOLDOUT_RESERVED=1" in wrapper
    assert gate.BUDGET == 600 and set(gate.ARCHIVES) == {"tless_base.zip", "tless_test_primesense_bop19.zip"}
