"""Tiny synthetic HTTP/ZIP contracts only: no network, assets, ORT or inference."""
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("dwpose_acquire", Path(__file__).parents[1] / "infra/dwpose_acquire.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wheel_bytes(package="flatbuffers", *, extra=(), version=None, tags=None, requires=None, license=True):
    spec = gate.WHEELS[package]; version = version or spec["version"]
    prefix = f"{package}-{spec['version']}.dist-info/"
    metadata = f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n"
    for value in spec["requires"] if requires is None else requires: metadata += f"Requires-Dist: {value}\n"
    wheel = "Wheel-Version: 1.0\n"
    for value in spec["tags"] if tags is None else tags: wheel += f"Tag: {value}\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(prefix + "METADATA", metadata)
        archive.writestr(prefix + "WHEEL", wheel)
        if license: archive.writestr(prefix + "licenses/LICENSE", "Own tiny license fixture\n")
        archive.writestr(package + "/__init__.py", "raise RuntimeError('Never execute')\n")
        for name, raw, mode in extra:
            info = zipfile.ZipInfo(name); info.external_attr = mode << 16
            archive.writestr(info, raw)
    return buffer.getvalue()


class Response(io.BytesIO):
    def __init__(self, raw, url="https://cas-bridge.xethub.hf.co/public?X-Amz-Signature=not-logged", headers=None):
        super().__init__(raw); self.url = url; self.headers = headers or {}

    def geturl(self):
        return self.url


@pytest.mark.parametrize("url", ["https://huggingface.co/repo", "https://raw.githubusercontent.com/repo",
                                 "https://files.pythonhosted.org/file", "https://cas-bridge.xethub.hf.co/x?token=a"])
def test_https_allowlist(url):
    gate.validate_https(url)


@pytest.mark.parametrize("url", ["http://huggingface.co/x", "https://hf.co.evil.com/x", "https://evil.com/x",
                                 "https://huggingface.co@evil.com/x", "https://a:b@huggingface.co/x",
                                 "https://huggingface.co:8443/x", "file:///tmp/x"])
def test_https_rejects_redirect_leak_hosts(url):
    with pytest.raises(ValueError): gate.validate_https(url)


@pytest.mark.parametrize("name", ["../x", "x/../y", "/x", "x//y", "x/./y", "x\\y", "C:x", "x\0y", "", "//"])
def test_asset_path_rejects(name):
    with pytest.raises(ValueError): gate.safe_relative(name)


def test_stream_exact_hash_bytes_exclusive_and_readonly(tmp_path):
    raw = b"own procedural bytes"
    path = tmp_path / "asset"
    gate.download("https://huggingface.co/x", path, len(raw), sha(raw), opener=lambda _: Response(raw))
    assert path.read_bytes() == raw and stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        gate.download("https://huggingface.co/x", path, len(raw), sha(raw), opener=lambda _: Response(raw))


@pytest.mark.parametrize("raw,size,expected,headers", [
    (b"1234", 3, sha(b"123"), {}), (b"12", 3, sha(b"123"), {}),
    (b"123", 3, sha(b"456"), {}), (b"123", 3, sha(b"123"), {"Content-Length": "4"}),
    (b"123", 3, sha(b"123"), {"Content-Encoding": "gzip"}),
    (b"<!DOCTYPE html>bad", 17, sha(b"<!DOCTYPE html>bad"), {}),
    (b"version https://git-lfs.github.com/spec/v1", 43, "0" * 64, {}),
])
def test_stream_failure_retains_partial(tmp_path, raw, size, expected, headers):
    path = tmp_path / "partial"
    with pytest.raises(ValueError):
        gate.download("https://huggingface.co/x", path, size, expected, opener=lambda _: Response(raw, headers=headers))
    assert path.exists()  # no hidden retry/deletion


def test_stream_rejects_unsafe_final_redirect(tmp_path):
    with pytest.raises(ValueError):
        gate.download("https://huggingface.co/x", tmp_path / "asset", 1, sha(b"x"), opener=lambda _: Response(b"x", url="https://evil.com/token"))


@pytest.mark.parametrize("size", [True, 0, -1, 200000001, 1.0])
def test_stream_invalid_size_before_io(tmp_path, size):
    def forbidden(_): raise AssertionError("No request")
    with pytest.raises(ValueError): gate.download("https://huggingface.co/x", tmp_path / "x", size, "0" * 64, forbidden)


@pytest.mark.parametrize("package", ["onnxruntime", "flatbuffers"])
def test_wheel_metadata_paths_license_retained_no_python_expansion(tmp_path, package):
    path = tmp_path / "fixture.whl"; path.write_bytes(wheel_bytes(package))
    target = tmp_path / "audit"
    row = gate.audit_wheel(path, package, target)
    assert row["version"] == gate.WHEELS[package]["version"]
    assert row["requires_dist"] == gate.WHEELS[package]["requires"]
    assert row["archive_expanded"] is False and row["symlinks_present"] is False
    assert len(row["retained_texts"]) == 3
    assert not (target / package).exists()
    for record in row["retained_texts"]:
        file = target / record["file"]
        assert gate.digest(file) == record["sha256"]
        assert stat.S_IMODE(file.stat().st_mode) == 0o444


@pytest.mark.parametrize("extra", [
    [("../evil", b"bad", stat.S_IFREG | 0o644)],
    [("/evil", b"bad", stat.S_IFREG | 0o644)],
    [("a\\evil", b"bad", stat.S_IFREG | 0o644)],
    [("link", b"target", stat.S_IFLNK | 0o777)],
    [("fifo", b"", stat.S_IFIFO | 0o644)],
    [("other-1.dist-info/METADATA", b"", stat.S_IFREG | 0o644)],
    [("flatbuffers-25.12.19.dist-info/METADATA", b"duplicate", stat.S_IFREG | 0o644)],
])
def test_wheel_rejects_unsafe_members_before_any_expansion(tmp_path, extra):
    path = tmp_path / "bad.whl"
    with pytest.warns(UserWarning) if "METADATA" in extra[0][0] and "other" not in extra[0][0] else nullcontext():
        path.write_bytes(wheel_bytes(extra=extra))
    target = tmp_path / "audit"
    with pytest.raises(ValueError): gate.audit_wheel(path, "flatbuffers", target)
    assert not target.exists()


@pytest.mark.parametrize("kwargs", [{"version": "0"}, {"tags": ["cp312-none-any"]}, {"requires": ["surprise"]}, {"license": False}])
def test_wheel_rejects_metadata_mismatch(tmp_path, kwargs):
    path = tmp_path / "bad.whl"; path.write_bytes(wheel_bytes(**kwargs))
    with pytest.raises(ValueError): gate.audit_wheel(path, "flatbuffers", tmp_path / "audit")


def test_wheel_rejects_regular_file_as_ancestor(tmp_path):
    extra = [("tree", b"not a directory", stat.S_IFREG | 0o644), ("tree/LICENSE", b"fixture", stat.S_IFREG | 0o644)]
    path = tmp_path / "bad.whl"; path.write_bytes(wheel_bytes(extra=extra))
    with pytest.raises(ValueError, match="collision"):
        gate.audit_wheel(path, "flatbuffers", tmp_path / "audit")


def metadata_fixture():
    data = {f"https://huggingface.co/api/models/yzd-v/DWPose/revision/{gate.MODEL_REV}?blobs=true": {
        "sha": gate.MODEL_REV, "cardData": {"license": "apache-2.0"}, "siblings": [
            {"rfilename": gate.ASSETS[0][0], "size": gate.ASSETS[0][1], "lfs": {"sha256": gate.ASSETS[0][2], "size": gate.ASSETS[0][1]}}]}}
    for package, spec in gate.WHEELS.items():
        name, size, digest, url = next(r for r in gate.ASSETS if r[0].startswith(f"wheels/{package}-"))
        data[f"https://pypi.org/pypi/{package}/{spec['version']}/json"] = {
            "info": {"version": spec["version"], "requires_dist": spec["requires"]},
            "urls": [{"filename": Path(name).name, "size": size, "digests": {"sha256": digest}, "url": url,
                      "yanked": False, "upload_time_iso_8601": "2025-12-19T23:16:13.622515Z"}]}
    return data


def test_primary_independent_metadata(tmp_path):
    records = gate.primary_metadata(metadata_fixture().__getitem__)
    assert len(records) == 3 and records[0]["revision"] == gate.MODEL_REV


@pytest.mark.parametrize("which", ["revision", "license", "size", "digest", "dependencies", "late", "yanked"])
def test_primary_metadata_failclosed(which):
    data = metadata_fixture(); model = next(iter(data.values())); wheel = list(data.values())[1]
    if which == "revision": model["sha"] = "0" * 40
    elif which == "license": model["cardData"]["license"] = "noncommercial"
    elif which == "size": model["siblings"][0]["size"] = 1
    elif which == "digest": model["siblings"][0]["lfs"]["sha256"] = "0" * 64
    elif which == "dependencies": wheel["info"]["requires_dist"] = ["extra"]
    elif which == "late": wheel["urls"][0]["upload_time_iso_8601"] = "2026-10-01T00:00:00Z"
    elif which == "yanked": wheel["urls"][0]["yanked"] = True
    with pytest.raises(ValueError): gate.primary_metadata(data.__getitem__)


def tiny_assets():
    values = {"source/onnxpose.py": b"from typing import List, Tuple\nimport cv2\nimport numpy as np\nimport onnxruntime as ort\n",
              "dw-ll_ucoco_384.onnx": b"own non-ONNX acquisition fixture"}
    for package, spec in gate.WHEELS.items():
        values[f"wheels/{package}-{spec['version']}.whl"] = wheel_bytes(package)
    return values


def test_acquisition_exclusive_publish_rehashes_no_execution(tmp_path, monkeypatch):
    values = tiny_assets()
    rows = tuple((name, len(raw), sha(raw), "https://huggingface.co/" + name) for name, raw in values.items())
    monkeypatch.setattr(gate, "ASSETS", rows)
    (tmp_path / gate.BASE).mkdir(parents=True)
    report = {"files": []}; checkpoints = []
    def downloader(url, destination, size, digest):
        raw = values[url.split("huggingface.co/", 1)[1]]
        gate.download(url, destination, size, digest, opener=lambda _: Response(raw))
    gate.acquire(tmp_path, report, lambda: checkpoints.append(json.dumps(report)), metadata=lambda: [], downloader=downloader)
    assert report["status"] == "pass" and report["final_assets_rehashed"] is True
    assert len(report["files"]) == 4 and len(report["wheel_audits"]) == 2
    assert not (tmp_path / gate.BASE / ".downloads").exists()
    assert "onnxruntime" in report["source_import_names"]
    assert checkpoints and all(stat.S_IMODE((tmp_path / gate.BASE / r["file"]).stat().st_mode) == 0o444 for r in report["files"])


def test_failure_preserves_published_and_partial_no_retry(tmp_path, monkeypatch):
    values = tiny_assets(); rows = tuple((n, len(v), sha(v), "https://huggingface.co/" + n) for n, v in values.items())
    monkeypatch.setattr(gate, "ASSETS", rows); (tmp_path / gate.BASE).mkdir(parents=True)
    calls = []
    def downloader(url, destination, size, digest):
        calls.append(url)
        if len(calls) == 2:
            destination.write_bytes(b"partial")
            raise ValueError("failed")
        raw = values[url.split("huggingface.co/", 1)[1]]
        gate.download(url, destination, size, digest, opener=lambda _: Response(raw))
    report = {"files": []}
    with pytest.raises(ValueError): gate.acquire(tmp_path, report, lambda: None, metadata=lambda: [], downloader=downloader)
    assert len(calls) == 2 and len(report["files"]) == 1
    assert (tmp_path / gate.BASE / ".downloads/01.part").read_bytes() == b"partial"


def test_no_overwrite_json(tmp_path):
    path = tmp_path / "receipt.json"; gate.save_json(path, {"status": "fail"})
    with pytest.raises(FileExistsError): gate.save_json(path, {"status": "pass"})
    assert json.loads(path.read_text())["status"] == "fail"


def test_main_rejects_nonremote_before_network(monkeypatch):
    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("WR_ROOT", "/tmp/no-local-acquisition")
    with pytest.raises(ValueError): gate.main([])


def test_frozen_known_pins_and_wrapper_contract():
    assert len(gate.ASSETS) == 9 and sum(r[1] for r in gate.ASSETS) == 158360974
    assert gate.SOURCE_REV == "3dca5db79d9f9ffdd378753ddf6ec66535aace88"
    assert gate.ASSETS[0][2] == "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"
    wrapper = Path(__file__).parents[1] / "infra/run_dwpose_acquire.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    assert subprocess.run(["bash", str(wrapper), "--unknown"], capture_output=True).returncode == 2
    text = wrapper.read_text()
    assert "env -i" in text and '"$CODE/infra/dwpose_acquire.py"' in text
    assert "--gpus" not in text and "pip install" not in text
