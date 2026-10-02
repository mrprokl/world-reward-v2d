"""Small public-download stubs; no weights/source downloads, Torch or GPU."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.request

import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1]/"infra/da3_metric_acquire.py"
    spec = importlib.util.spec_from_file_location("wr_da3_acquire_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def record(name, data, git=True):
    return (name, len(data), hashlib.sha256(data).hexdigest(), hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() if git else None)


class Response(io.BytesIO):
    def __init__(self, data, url="https://raw.githubusercontent.com/a/b/pinned/file", **headers):
        super().__init__(data); self.url = url; self.headers = headers

    def geturl(self): return self.url


@pytest.mark.parametrize("fault", [None, "short", "long", "sha", "git", "length", "encoding", "redirect"])
def test_bounded_exclusive_download_hashes(gate, tmp_path, monkeypatch, fault):
    data = b"tiny public model"; _, size, sha, blob = record("file", data)
    body = data[:-1] if fault == "short" else data+b"!" if fault == "long" else data
    headers = {"Content-Length": str(size+1)} if fault == "length" else {"Content-Encoding": "gzip"} if fault == "encoding" else {}
    monkeypatch.setattr(gate, "open_public", lambda u: Response(body, "http://huggingface.co/model" if fault == "redirect" else u, **headers))
    path = tmp_path/"asset"
    if fault:
        with pytest.raises(ValueError): gate.download("https://huggingface.co/model", path, size, "a"*64 if fault == "sha" else sha, "a"*40 if fault == "git" else blob)
    else:
        gate.download("https://huggingface.co/model", path, size, sha, blob)
        assert path.read_bytes() == data and path.stat().st_mode & 0o777 == 0o444
        with pytest.raises(FileExistsError): gate.download("https://huggingface.co/model", path, size, sha, blob)


@pytest.mark.parametrize("url", ["http://huggingface.co/file", "https://evil.com/file", "https://huggingface.co.evil/file", "https://user:secret@huggingface.co/file", "https://huggingface.co:444/file"])
def test_public_https_and_redirects_fail_before_follow(gate, url):
    with pytest.raises(ValueError): gate.validate_https(url)
    request = urllib.request.Request("https://huggingface.co/file")
    with pytest.raises(ValueError): gate.HTTPSRedirects().redirect_request(request, None, 302, "Found", {}, url)


def primary_fixture(gate):
    tree = {"sha": gate.SOURCE_REV, "truncated": False, "tree": [{"path": n, "size": s, "sha": b, "type": "blob", "mode": "100644"} for n, s, _, b in gate.SOURCE_RECORDS]}
    model = {"sha": gate.MODEL_REV, "cardData": {"license": "apache-2.0"}, "siblings": [{"rfilename": n, "size": s, "blobId": b or gate.MODEL_POINTER_SHA,
        **({"lfs": {"sha256": sha, "size": s}} if b is None else {})} for n, s, sha, b in gate.MODEL_RECORDS]}
    return tree, model


@pytest.mark.parametrize("fault", [None, "commit", "tree", "source_blob", "model_pin", "model_sha", "license"])
def test_independent_primary_tree_and_publisher_metadata(gate, monkeypatch, fault):
    tree, model = primary_fixture(gate)
    if fault == "commit": tree["sha"] = "a"*40
    elif fault == "tree": tree["truncated"] = True
    elif fault == "source_blob": tree["tree"][0]["sha"] = "a"*40
    elif fault == "model_pin": model["sha"] = "a"*40
    elif fault == "model_sha": model["siblings"][-1]["lfs"]["sha256"] = "a"*64
    elif fault == "license": model["cardData"]["license"] = "cc-by-nc-4.0"
    monkeypatch.setattr(gate, "fetch_json", lambda u: tree if "api.github.com" in u else model)
    if fault:
        with pytest.raises(ValueError): gate.primary_metadata()
    else: assert gate.primary_metadata()["source_git_tree_verified"] is True


def tiny_sources(gate, root, monkeypatch):
    source_files = {"LICENSE": b"Apache License", "README.md": b"Upstream", "requirements.txt": b"torch\nevo\n", "src/depth_anything_3/cfg.py": b"import torch\nimport json\n"}
    model_files = {"config.json": b"{}", "README.md": b"license: apache-2.0", "model.safetensors": b"tiny weights"}
    monkeypatch.setattr(gate, "SOURCE_RECORDS", tuple(record(n, d) for n, d in source_files.items()))
    monkeypatch.setattr(gate, "MODEL_RECORDS", tuple(record(n, d, n != "model.safetensors") for n, d in model_files.items()))
    for relative in (gate.WEIGHTS, gate.SOURCE, "results"): (root/relative).mkdir(parents=True)
    monkeypatch.setattr(gate, "primary_metadata", lambda: {"source_git_tree_verified": True, "publisher_model_manifest_verified": True})
    def download(u, p, s, sha, blob):
        data = (source_files if "raw.githubusercontent.com" in u else model_files)[u.rsplit("/", 1)[-1] if "cfg.py" not in u else "src/depth_anything_3/cfg.py"]
        p.write_bytes(data); p.chmod(0o444)
    monkeypatch.setattr(gate, "download", download)
    return source_files, model_files


@pytest.mark.parametrize("fail", [False, True])
def test_publish_exact_files_cleanup_and_partial_evidence(gate, tmp_path, monkeypatch, fail):
    source_files, model_files = tiny_sources(gate, tmp_path, monkeypatch); report = {"files": [], "status": "fail"}; snapshots = []
    original = gate.download
    def download(u, *args):
        if fail and u.endswith("model.safetensors"):
            args[0].write_bytes(b"partial"); raise ValueError("Rejected model bytes")
        original(u, *args)
    monkeypatch.setattr(gate, "download", download)
    if fail:
        with pytest.raises(ValueError): gate.acquire(tmp_path, report, lambda: snapshots.append(json.loads(json.dumps(report))))
    else: gate.acquire(tmp_path, report, lambda: snapshots.append(json.loads(json.dumps(report))))
    assert report["disposable_downloads_removed"] is True and not (tmp_path/gate.WEIGHTS/".downloads").exists()
    assert report["status"] == ("fail" if fail else "pass") and snapshots
    assert (tmp_path/gate.SOURCE/"source_manifest.json").is_file()
    assert not (tmp_path/gate.SOURCE/"src/depth_anything_3/__init__.py").exists()
    assert not (tmp_path/gate.WEIGHTS/"model.safetensors").exists() if fail else (tmp_path/gate.WEIGHTS/"model.safetensors").read_bytes() == model_files["model.safetensors"]
    if not fail:
        assert report["runtime_external_import_names"] == ["torch"] and report["requirements_installed"] is False
        assert report["upstream_requirements"] == ["torch", "evo"]  # audit metadata, never pip install


def test_constants_exact_closure_bounds_no_stubs_or_highlevel_api(gate):
    paths = [r[0] for r in gate.SOURCE_RECORDS]
    assert len(paths) == len(set(paths)) == 32
    assert sum(n.endswith(".py") for n in paths) == 27
    assert sum(s for n, s, *_ in gate.SOURCE_RECORDS if n.endswith(".py")) == 192452
    assert sum(s for _, s, *_ in gate.SOURCE_RECORDS) == 225327
    assert "src/depth_anything_3/configs/da3metric-large.yaml" in paths
    assert "src/depth_anything_3/__init__.py" not in paths
    assert not any("/api.py" in p or "/export/" in p or "/train/" in p or "/gs.py" in p for p in paths)
    assert gate.MODEL_RECORDS[-1][1:3] == (1336734448, "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776")


def test_main_remote_exclusive_failure_receipt(gate, tmp_path, monkeypatch):
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_DA3_OUTPUTS_RESERVED", "1")
    monkeypatch.setattr(gate.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError): gate.main([])
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); monkeypatch.setattr(gate.os, "geteuid", lambda: 1000)
    for relative in (gate.WEIGHTS, gate.SOURCE, "results"): (tmp_path/relative).mkdir(parents=True)
    def failed(*args): raise ValueError("Never print https://host?sig=credential")
    monkeypatch.setattr(gate, "acquire", failed)
    with pytest.raises(RuntimeError, match="inspect immutable receipt"): gate.main([])
    report = json.loads((tmp_path/gate.REPORT).read_text())
    assert report["status"] == "fail" and report["error_type"] == "ValueError" and "?sig=" not in report["error"]
    assert (tmp_path/gate.REPORT).stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError): gate.main([])
    with pytest.raises(SystemExit): gate.main(["--repo", "arbitrary"])


def test_wrapper_scoped_no_gpu_install_or_recursive_chown():
    path = Path(__file__).resolve().parents[1]/"infra/run_da3_metric_acquire.sh"
    text = path.read_text()
    assert "723s runuser -u scenesmith" in text and "WR_DA3_OUTPUTS_RESERVED=1" in text
    assert "chown -R" not in text and "docker run" not in text and "pip install" not in text
    assert 'mkdir -p "$OUT"' in text and 'python3 "$CODE/infra/da3_metric_acquire.py"' in text
