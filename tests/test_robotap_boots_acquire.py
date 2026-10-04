"""Own tiny byte streams/ZIPs only: no network, assets, RGB/GT or model loads."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import time
import zipfile

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "infra/robotap_boots_acquire.py"
    spec = importlib.util.spec_from_file_location("robotap_acquire_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def protocol(gate, tmp_path):
    original = Path(__file__).resolve().parents[1] / "configs/robotap_boots_protocol.json"
    path = tmp_path / "protocol.json"; path.write_bytes(original.read_bytes()); path.chmod(0o400)
    return gate.read_protocol(path)


def tiny_zip(values):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in values: archive.writestr(name, value)
    data.seek(0)
    return zipfile.ZipFile(data)


def test_whole_protocol_byte_pin_before_json_parser(gate, tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"; path.write_bytes(b"not JSON"); path.chmod(0o400)
    def forbidden(*_): raise AssertionError("JSON opened before full byte pin")
    monkeypatch.setattr(gate.json, "loads", forbidden)
    with pytest.raises(ValueError, match="Whole frozen"): gate.read_protocol(path)


def test_protocol_hard_pins_and_honest_pending_scope(gate, protocol):
    assert protocol["dataset"]["archive"]["bytes"] == 13558087507
    assert protocol["dataset"]["archive"]["generation"] == "1693927735577112"
    assert protocol["dataset"]["archive"]["md5_hex"] == "08dc00f12b10a7d70e0afd53ab822762"
    assert protocol["checkpoint"]["bytes"] == 218886140
    assert protocol["checkpoint"]["sha256"] == "8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b"
    assert len(protocol["source"]["files"]) == 7
    assert not protocol["planned_diagnostic"]["frame_and_query_matrix_frozen"]
    assert not protocol["planned_diagnostic"]["oracle_initial_query_is_track1_automatic_readiness"]
    assert protocol["budget_seconds"] == gate.BUDGET == 1200


@pytest.mark.parametrize("names,selected", [(["robotap.pkl"], []), (["z.pkl", "a.pkl", "c.pkl", "b.pkl"], ["a.pkl", "b.pkl", "c.pkl"])])
def test_filename_only_selection_never_claims_video_or_frame_verification(gate, names, selected):
    class NoValues:
        def read(self, *_): raise AssertionError("No private value reads")
    selection = gate.freeze_selection(dict.fromkeys(names, NoValues()))
    assert selection["first_three_pickle_files_frozen"] == selected
    assert selection["selection_pending"] and selection["video_selection_pending"]
    assert not selection["benchmark_ready"] and not selection["frame_cardinalities_verified"]
    assert not selection["video_identity_from_pickle_filename_verified"]


@pytest.mark.parametrize("name", ["../escape.pkl", "/absolute.pkl", "a/../b.pkl", "a\\b.pkl", "a//b.pkl", "a/./b.pkl"])
def test_zip_traversal_and_noncanonical_names_stop_before_member_values(gate, protocol, name):
    with tiny_zip([(name, b"own tiny bytes")]) as archive:
        with pytest.raises(ValueError, match="Unsafe"): gate.zip_inventory(archive, protocol)


def test_zip_duplicates_symlink_and_file_parent_collision_fail(gate, protocol):
    with pytest.warns(UserWarning): archive = tiny_zip([("same.pkl", b"a"), ("same.pkl", b"b")])
    with archive, pytest.raises(ValueError, match="Unsafe"): gate.zip_inventory(archive, protocol)
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        info = zipfile.ZipInfo("link.pkl"); info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16; archive.writestr(info, "target")
    data.seek(0)
    with zipfile.ZipFile(data) as archive, pytest.raises(ValueError, match="Unsafe"): gate.zip_inventory(archive, protocol)
    with tiny_zip([("a", b"file"), ("a/b.pkl", b"own bytes")]) as archive:
        with pytest.raises(ValueError, match="parent collision"): gate.zip_inventory(archive, protocol)


def test_zip_frozen_expansion_and_embedded_text_bounds_before_values(gate, protocol):
    with tiny_zip([("own.pkl", b"abcd")]) as archive:
        limited = copy.deepcopy(protocol); limited["maximum_expanded_bytes"] = 3
        with pytest.raises(ValueError, match="expansion"): gate.zip_inventory(archive, limited)
    with tiny_zip([("README.md", b"abcd"), ("own.pkl", b"x")]) as archive:
        limited = copy.deepcopy(protocol); limited["maximum_embedded_text_bytes"] = 3
        with pytest.raises(ValueError, match="text"): gate.zip_inventory(archive, limited)
    with pytest.raises(ValueError, match="Only original opaque"):
        gate.freeze_selection({"own.pkl": object(), "rgb.mp4": object()})


@pytest.mark.parametrize("text", [b"License: CC-BY-NC-4.0", b"License: CC-BY-SA-4.0", b"Research use only", b"all rights reserved", b"License: MIT", b"License: unknown", b"Copyright 2023 Google LLC", b"License: Apache 2.0", b"\xff"])
def test_embedded_contradictory_or_incomplete_explicit_terms_stop(gate, text):
    with pytest.raises(ValueError): gate.check_embedded_text("LICENSE.txt", text)


def test_embedded_attribution_or_complete_ccby4_and_original_opaque_retention(gate, protocol, tmp_path):
    gate.check_embedded_text("README.md", b"Vecerik et al., RoboTAP")
    gate.check_embedded_text("LICENSE.txt", b"License: CC-BY-4.0")
    private = tmp_path / "eval_private"; private.mkdir(mode=0o700)
    values = [("robotap/robotap.pkl", b"not-a-real-pickle: unchanged own bytes"), ("robotap/README.md", b"Authors: Vecerik et al.")]
    with tiny_zip(values) as archive:
        members, expanded = gate.zip_inventory(archive, protocol)
        gate.freeze_selection(members)
        retained = gate.extract_members(archive, members, private, protocol, time.monotonic() + 5)
    assert expanded == sum(len(value) for _, value in values)
    for name, raw in values:
        path = private / "pickles" / name
        assert path.read_bytes() == raw and path.stat().st_mode & 0o777 == 0o400
    assert all(r["file"].startswith("eval_private/pickles/") for r in retained)


class Response(io.BytesIO):
    def __init__(self, data, headers, url):
        super().__init__(data); self.headers = headers; self.url = url; self.status = 200
    def geturl(self): return self.url


def tiny_pin(data, *, gcs=False):
    pin = {"url": "https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip?generation=123",
        "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    headers = {"Content-Length": str(len(data))}
    if gcs:
        import base64
        md5 = hashlib.md5(data)
        pin.update(generation="123", metageneration="1", content_type="application/zip",
            md5_hex=md5.hexdigest(), md5_base64=base64.b64encode(md5.digest()).decode())
        headers.update({"x-goog-generation": "123", "x-goog-metageneration": "1",
            "x-goog-stored-content-length": str(len(data)), "Content-Type": "application/zip",
            "ETag": '"' + md5.hexdigest() + '"'})
    return pin, headers


def test_one_streaming_download_original_sha_md5_readonly_no_overwrite(gate, tmp_path, monkeypatch):
    raw = b"own tiny stream"; pin, headers = tiny_pin(raw, gcs=True); calls = []
    def open_once(request, timeout):
        calls.append(request.full_url); return Response(raw, headers, request.full_url)
    monkeypatch.setattr(gate.urllib.request, "urlopen", open_once)
    path = tmp_path / "data.zip"
    result = gate.download(pin, path, time.monotonic() + 5)
    assert len(calls) == 1 and result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["publisher_md5_verified"] and result["generation"] == "123"
    assert path.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError): gate.download(pin, path, time.monotonic() + 5)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("fault", ["generation", "response_size", "md5", "sha", "truncated", "oversize", "redirect"])
def test_streaming_integrity_failure_no_retry(gate, tmp_path, monkeypatch, fault):
    raw = b"own tiny stream"; pin, headers = tiny_pin(raw, gcs=True); returned = raw; url = pin["url"]
    if fault == "generation": headers["x-goog-generation"] = "999"
    elif fault == "response_size": headers["Content-Length"] = "1"
    elif fault == "md5": pin["md5_base64"] = "wrong"
    elif fault == "sha": pin["sha256"] = "0" * 64
    elif fault == "truncated": returned = raw[:-1]
    elif fault == "oversize": returned = raw + b"x"
    else: url = "http://evil.invalid/blob"
    calls = []
    def open_once(request, timeout): calls.append(1); return Response(returned, headers, url)
    monkeypatch.setattr(gate.urllib.request, "urlopen", open_once)
    with pytest.raises(ValueError): gate.download(pin, tmp_path / "data.zip", time.monotonic() + 5)
    assert len(calls) == 1


def test_expired_deadline_prevents_network_or_extraction(gate, protocol, tmp_path, monkeypatch):
    def forbidden(*_, **__): raise AssertionError("Network reached after fixed deadline")
    monkeypatch.setattr(gate.urllib.request, "urlopen", forbidden)
    pin, _ = tiny_pin(b"own")
    with pytest.raises(TimeoutError): gate.download(pin, tmp_path / "late.zip", time.monotonic() - 1)
    private = tmp_path / "eval_private"; private.mkdir()
    with tiny_zip([("own.pkl", b"own")]) as archive:
        members, _ = gate.zip_inventory(archive, protocol)
        with pytest.raises(TimeoutError): gate.extract_members(archive, members, private, protocol, time.monotonic() - 1)
    assert not list(private.iterdir())


def test_complete_output_firewall_rejects_tamper_extra_and_private_export(gate, tmp_path):
    destination = tmp_path / "out"; destination.mkdir(mode=0o700)
    (destination / "assets").mkdir(mode=0o700); (destination / "eval_private").mkdir(mode=0o700)
    (destination / "report.json").write_text("{}")
    gate.save_bytes(destination / "assets/source.py", b"own source", 0o444)
    gate.save_bytes(destination / "eval_private/pickles/own.pkl", b"own opaque")
    expected = {name: gate.identity(destination / name) for name in ("assets/source.py", "eval_private/pickles/own.pkl")}
    assert gate.verify_retained(destination, expected) == expected
    (destination / "assets/private.pkl").write_bytes(b"private export"); (destination / "assets/private.pkl").chmod(0o444)
    with pytest.raises(ValueError, match="inventory"): gate.verify_retained(destination, expected)
    (destination / "assets/private.pkl").unlink()
    (destination / "eval_private/pickles/own.pkl").chmod(0o600)
    with pytest.raises(ValueError, match="permissions"): gate.verify_retained(destination, expected)


def test_wrapper_static_cpu_timeout_and_no_gpu_or_unpickle(gate):
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "infra/run_robotap_boots_acquire.sh").read_text()
    module = (root / "infra/robotap_boots_acquire.py").read_text()
    assert "1203s" in wrapper and "--kill-after=10s" in wrapper
    assert "mkdir -m 700" in wrapper and "runuser -u scenesmith" in wrapper
    assert "docker run" not in wrapper and "nvidia-smi" not in wrapper
    assert "import pickle" not in module and "import torch" not in module and "numpy" not in module


def test_private_files_mode400_before_first_byte_with_umask_zero(gate, protocol, tmp_path, monkeypatch):
    import os
    raw = b"own tiny stream"; pin, headers = tiny_pin(raw); target = tmp_path / "download.zip"
    class ObservedResponse(Response):
        def read(self, *args):
            if target.exists(): assert target.stat().st_mode & 0o777 == 0o400
            return super().read(*args)
    monkeypatch.setattr(gate.urllib.request, "urlopen", lambda request, timeout: ObservedResponse(raw, headers, request.full_url))
    original = os.umask(0)
    try:
        gate.download(pin, target, time.monotonic() + 5)
        written = tmp_path / "new/parents/private.txt"
        gate.save_bytes(written, b"own private")
        assert written.stat().st_mode & 0o777 == 0o400
        assert written.parent.stat().st_mode & 0o777 == 0o700
        assert written.parent.parent.stat().st_mode & 0o777 == 0o700
        private = tmp_path / "eval_private"; private.mkdir(mode=0o700)
        with tiny_zip([("nested/own.pkl", b"own opaque")]) as archive:
            members, _ = gate.zip_inventory(archive, protocol)
            gate.extract_members(archive, members, private, protocol, time.monotonic() + 5)
        assert (private / "pickles/nested/own.pkl").stat().st_mode & 0o777 == 0o400
        assert (private / "pickles").stat().st_mode & 0o777 == 0o700
    finally: os.umask(original)


@pytest.mark.parametrize("fault", [None, "archive_inode", "extraction_failure"])
def test_lifecycle_receipt_before_archive_delete_and_single_pickle_pending(gate, protocol, tmp_path, monkeypatch, fault):
    destination = tmp_path / "out"; destination.mkdir(mode=0o700)
    for name in ("assets", "eval_private"): (destination / name).mkdir(mode=0o700)
    (destination / "report.json").write_bytes(b"{}")
    raw_zip = io.BytesIO()
    with zipfile.ZipFile(raw_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("robotap.pkl", b"own bytes, not an executable pickle")
    raw = raw_zip.getvalue(); protocol = copy.deepcopy(protocol)
    protocol["source"]["files"] = {"README.md": {"bytes": 10, "sha256": hashlib.sha256(b"own source").hexdigest()}}
    protocol["checkpoint"].update(bytes=11, sha256=hashlib.sha256(b"own weights").hexdigest())
    def sources(protocol, assets, deadline):
        gate.save_bytes(assets / "tapnet_source/README.md", b"own source", 0o444)
        gate.save_bytes(assets / "dataset-attribution.json", b"own attribution", 0o444)
        return {"README.md": {"bytes": 10, "sha256": hashlib.sha256(b"own source").hexdigest()}}
    def downloads(pin, path, deadline):
        value = raw if path.name == "robotap.zip" else b"own weights"
        gate.save_bytes(path, value)
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    monkeypatch.setattr(gate, "primary_sources", sources); monkeypatch.setattr(gate, "download", downloads)
    original_unlink = Path.unlink; observed = []
    def unlink(path, *args, **kwargs):
        if path.name == "robotap.zip":
            receipt = json.loads((destination / "eval_private/retention-receipt.json").read_bytes())
            assert receipt["no_unpickle_or_decode"] and not receipt["filename_selection"]["benchmark_ready"]
            assert (destination / "eval_private/pickles/robotap.pkl").read_bytes() == b"own bytes, not an executable pickle"
            observed.append(True)
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    if fault == "archive_inode":
        original_verify = gate.verify_retained; swapped = []
        def verify(destination, expected):
            result = original_verify(destination, expected)
            if not swapped:
                path = destination / "eval_private/.downloads/robotap.zip"
                replacement = path.with_name("replacement.zip")
                gate.save_bytes(replacement, path.read_bytes()); replacement.replace(path); swapped.append(True)
            return result
        monkeypatch.setattr(gate, "verify_retained", verify)
    elif fault == "extraction_failure":
        def fail(*_, **__): raise ValueError("Own injected pre-receipt failure")
        monkeypatch.setattr(gate, "extract_members", fail)
    report = {}
    if fault:
        with pytest.raises(ValueError): gate.acquire(destination, protocol, report, lambda: None, time.monotonic() + 5)
        assert observed == [] and (destination / "eval_private/.downloads/robotap.zip").read_bytes() == raw
        assert not report.get("disposable_archive_removed_after_successful_retention_receipt", False)
        return
    gate.acquire(destination, protocol, report, lambda: None, time.monotonic() + 5)
    assert observed == [True] and report["disposable_archive_removed_after_successful_retention_receipt"]
    assert report["filename_selection"]["selection_pending"] and not report["benchmark_ready"]
    assert not (destination / "eval_private/.downloads").exists()
