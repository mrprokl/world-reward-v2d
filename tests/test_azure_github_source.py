"""Offline HTTPS spies and real temporary publication; never Azure/network."""
import base64
import hashlib
import importlib.util
import io
import json
import lzma
from pathlib import Path
import random
import shlex
import subprocess
import sys
import threading
import time
import urllib.request

import pytest
import tarfile

spec = importlib.util.spec_from_file_location("azure_github_test", Path(__file__).resolve().parents[1] / "infra/azure_job.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
REV = "b" * 40


def fixture_archive(*, large=False):
    files = {"infra/run_smoke.sh": b"#!/bin/bash\ntrue\n", "infra/smoke.py": b"FROZEN=1\n", "src/pkg/__init__.py": b"", "pyproject.toml": b"[project]\nname='tiny'\n"}
    if large:
        files["configs/payload.json"] = base64.b64encode(random.Random(94).randbytes(155000))
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.mtime = 1700000000.125; info.uid = 0; info.gid = 0
            info.pax_headers = {"comment": "git metadata unchanged", "mtime": "1700000000.125"}
            archive.addfile(info, io.BytesIO(data))
    raw = stream.getvalue(); encoded, sha = launcher.encoded_runtime_archive(raw)
    return raw, encoded, sha, files


class Response:
    def __init__(self, url, data, fault=None):
        self.url = url; self.data = io.BytesIO(data); self.closed = False
        self.status = 302 if fault == "status" else 200
        self.headers = {"Content-Length": str(len(data)), "Content-Encoding": "gzip" if fault == "encoding" else "identity"}
        if fault == "length": self.headers["Content-Length"] = "999999"
        if fault == "chunked": self.headers.pop("Content-Length")
    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def geturl(self): return self.url
    def read(self, count): return self.data.read(count)


class Opener:
    def __init__(self, files, fault=None):
        self.files = files; self.fault = fault; self.calls = []; self.responses = []
    def open(self, request, *, timeout):
        assert 0 < timeout <= 90 and request.full_url.startswith(launcher.GITHUB_RAW_ROOT + REV + "/")
        assert dict(request.header_items()) == {"Accept-encoding": "identity"}
        path = request.full_url.split(REV + "/", 1)[1]; self.calls.append(path)
        if self.fault == "timeout": raise TimeoutError("offline timeout")
        data = self.files[path]
        if self.fault == "hash": data = b"x" * len(data) if data else b"x"
        if self.fault == "overflow": data += b"x"
        if self.fault == "short": data = data[:-1]
        url = "https://other.example.invalid/file" if self.fault == "redirect" else request.full_url
        response = Response(url, data, self.fault)
        if self.fault in ("overflow", "short"): response.headers.pop("Content-Length")
        self.responses.append(response); return response


def test_exact_tar_pax_executable_empty_and_xz_reconstruction_without_network():
    raw, encoded, sha, files = fixture_archive()
    descriptor = launcher.github_archive_descriptor(raw, REV, sha); opener = Opener(files)
    rebuilt, compressed = launcher.reconstruct_github_archive(descriptor, opener=opener)
    assert rebuilt == raw and compressed == base64.b64decode(encoded)
    assert sha == hashlib.sha256(compressed).hexdigest()
    assert sorted(opener.calls) == sorted(files) and all(r.closed for r in opener.responses)
    with tarfile.open(fileobj=io.BytesIO(rebuilt)) as archive:
        assert archive.getmember("infra/run_smoke.sh").mode == 0o755
        assert archive.getmember("src/pkg/__init__.py").size == 0
        assert archive.getmember("infra/run_smoke.sh").pax_headers["comment"] == "git metadata unchanged"


@pytest.mark.parametrize("fault", ["hash", "overflow", "short", "redirect", "status", "encoding", "length", "timeout"])
def test_https_faults_fail_closed_without_retry(fault):
    raw, _, sha, files = fixture_archive(); opener = Opener(files, fault)
    with pytest.raises((RuntimeError, TimeoutError)):
        launcher.reconstruct_github_archive(launcher.github_archive_descriptor(raw, REV, sha), opener=opener)
    assert len(opener.calls) <= len(files) and len(opener.calls) == len(set(opener.calls))


def test_chunked_body_still_exactly_bounded():
    raw, _, sha, files = fixture_archive()
    assert launcher.reconstruct_github_archive(launcher.github_archive_descriptor(raw, REV, sha), opener=Opener(files, "chunked"))[0] == raw


def test_real_opener_has_no_proxy_redirect_auth_or_repository_code_execution(monkeypatch):
    raw, _, sha, files = fixture_archive(); spy = Opener(files); handlers = []
    def build(*args): handlers.extend(args); return spy
    monkeypatch.setattr(urllib.request, "build_opener", build)
    launcher.reconstruct_github_archive(launcher.github_archive_descriptor(raw, REV, sha))
    assert handlers[0].proxies == {} and len(handlers) == 2
    assert handlers[1].redirect_request(None, None, 302, "", {}, "https://other.invalid") is None


def test_four_concurrent_workers_and_one_global_deadline():
    raw, _, sha, files = fixture_archive(); descriptor = launcher.github_archive_descriptor(raw, REV, sha)
    lock = threading.Lock(); release = threading.Event(); active = 0; maximum = 0
    class Concurrent(Opener):
        def open(self, request, *, timeout):
            nonlocal active, maximum
            with lock:
                active += 1; maximum = max(active, maximum)
                if active == 4: release.set()
            assert release.wait(2)
            result = super().open(request, timeout=timeout)
            with lock: active -= 1
            return result
    launcher.reconstruct_github_archive(descriptor, opener=Concurrent(files))
    assert maximum == 4


def test_deadline_after_last_body_or_compression_cannot_return_success(monkeypatch):
    raw, _, sha, files = fixture_archive(); descriptor = launcher.github_archive_descriptor(raw, REV, sha)
    original = launcher.lzma.compress
    def late(*args, **kwargs):
        output = original(*args, **kwargs); monkeypatch.setattr(time, "monotonic", lambda: 10**15); return output
    monkeypatch.setattr(launcher.lzma, "compress", late)
    with pytest.raises(RuntimeError): launcher.reconstruct_github_archive(descriptor, opener=Opener(files))


@pytest.mark.parametrize("fault", ["revision", "offset", "tar_hash", "xz_hash", "duplicate", "nonzero_skeleton"])
def test_descriptor_integrity_faults(fault):
    raw, _, sha, files = fixture_archive(); descriptor = launcher.github_archive_descriptor(raw, REV, sha)
    if fault == "revision": descriptor["revision"] = "main"
    elif fault == "offset": descriptor["files"][0]["offset"] += 1
    elif fault == "tar_hash": descriptor["tar_sha256"] = "0" * 64
    elif fault == "xz_hash": descriptor["archive_sha256"] = "0" * 64
    elif fault == "duplicate": descriptor["files"].append(descriptor["files"][0])
    else:
        skeleton = bytearray(lzma.decompress(base64.b64decode(descriptor["skeleton"])))
        skeleton[descriptor["files"][0]["offset"]] = 1
        descriptor["skeleton"] = base64.b64encode(lzma.compress(skeleton)).decode()
    with pytest.raises(RuntimeError): launcher.reconstruct_github_archive(descriptor, opener=Opener(files))


def local_phase(tmp_path, *, fault=None, collision=False, foreign=False, publication_fault=None):
    raw, encoded, sha, files = fixture_archive(large=True)
    command = launcher.transport_commands(encoded, sha, raw, REV, "infra/run_smoke.sh", "github-test", [], github_source=True)[0]
    root = tmp_path / "remote"; (root / "jobs").mkdir(parents=True); (root / "results").mkdir()
    (root / "foreign").write_text("untouched")
    job = root / "jobs" / REV / "run_smoke"
    if collision: job.mkdir(parents=True); (job / "original").write_text("do not replace")
    bins = tmp_path / "bin"; bins.mkdir()
    for name in ("systemctl", "systemd-run"):
        p = bins / name; p.write_text("#!/bin/bash\n" + ("printf '%s\\n' \"$*\" >> \"$HOME/launched\"\n" if name == "systemd-run" else "exit 0\n")); p.chmod(0o755)
    script = command[2].replace("ROOT=/srv/scenesmith/world-reward", "ROOT=" + shlex.quote(str(root)))
    spy = """import urllib.request
+class FakeResponse:
+ status=200
+ def __init__(self,url,data):self.url=url;self.data=io.BytesIO(data);self.headers={'Content-Length':str(len(data))}
+ def __enter__(self):return self
+ def __exit__(self,*args):pass
+ def geturl(self):return self.url
+ def read(self,count):return self.data.read(count)
+class FakeOpener:
+ def open(self,request,timeout):
+  data=base64.b64decode(FILES[request.full_url.split(REV+'/',1)[1]])
+  if FAULT=='hash':data=b'x'*len(data)
+  if FAULT=='timeout':raise TimeoutError('offline')
+  if FOREIGN:(stage/'foreign').write_text('unknown')
+  return FakeResponse(request.full_url,data)
+urllib.request.build_opener=lambda *args:FakeOpener()
+""".replace("\n+", "\n")
    spy = "import io,base64\nFILES=" + repr({p: base64.b64encode(d).decode() for p, d in files.items()}) + "\nREV=" + repr(REV) + "\nFAULT=" + repr(fault) + "\nFOREIGN=" + repr(foreign) + "\n" + spy
    script = script.replace("import os,signal,stat\n", spy + "\nimport os,signal,stat\n")
    if sys.platform != "linux":
        old = "libc=ctypes.CDLL(None,use_errno=True);rename=getattr(libc,'renameat2',None);require(rename is not None)"
        new = "def rename(a,old,b,new,flags):return int(os.rename(old,new) or 0) if flags==1 and not Path(os.fsdecode(new)).exists() else -1"
        script = script.replace(old, new)
    if publication_fault == "collision":
        script = script.replace("require(rename(-100", "job.mkdir();(job/'original').write_text('racing owner')\n  require(rename(-100", 1)
    if publication_fault == "extract":
        script = script.replace("os.utime(target,(m.mtime,m.mtime))", "os.utime(target,(m.mtime,m.mtime));raise RuntimeError('offline extraction failure')", 1)
    result = subprocess.run(["bash", "-c", script], env={"PATH": str(bins) + ":/usr/bin:/bin", "HOME": str(tmp_path)}, capture_output=True, timeout=15)
    return result, root, job, command, raw, sha


def test_single_phase_large_payload_exact_publication_source_modes_ack_and_no_git_clone(tmp_path):
    result, root, job, command, raw, sha = local_phase(tmp_path)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout.decode().splitlines() == [command[1]]
    assert len(command[2].encode()) <= launcher.STAGED_SCRIPT_BYTES
    assert (job / "source-sha256").read_text() == sha + "\n"
    assert (job / "revision").read_text() == REV + "\n"
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            target = job / "code" / member.name
            assert target.read_bytes() == archive.extractfile(member).read()
            assert target.stat().st_mode & 0o777 == member.mode & 0o555
    assert all(p.stat().st_mode & 0o777 == 0o555 for p in (job / "code", *(p for p in (job / "code").rglob("*") if p.is_dir())))
    assert not list((root / "jobs").glob(".runtime-stage-*"))
    assert (tmp_path / "launched").exists() and (root / "foreign").read_text() == "untouched"
    assert "git clone" not in command[2] and "ITIMER_REAL,90" in command[2]


@pytest.mark.parametrize("fault", ["hash", "timeout"])
def test_fetch_failure_cleans_owned_stage_without_publication_or_launch(tmp_path, fault):
    result, root, job, *_ = local_phase(tmp_path, fault=fault)
    assert result.returncode != 0 and not job.exists() and not (tmp_path / "launched").exists()
    assert not list((root / "jobs").glob(".runtime-stage-*"))
    assert (root / "foreign").read_text() == "untouched"


def test_unknown_stage_file_is_not_deleted_on_failure(tmp_path):
    result, root, job, *_ = local_phase(tmp_path, fault="hash", foreign=True)
    assert result.returncode != 0 and not job.exists()
    stages = list((root / "jobs").glob(".runtime-stage-*")); assert len(stages) == 1
    assert (stages[0] / "foreign").read_text() == "unknown"


def test_collision_preserves_existing_snapshot_and_never_fetches_or_launches(tmp_path):
    result, root, job, *_ = local_phase(tmp_path, collision=True)
    assert result.returncode != 0 and (job / "original").read_text() == "do not replace"
    assert not list((root / "jobs").glob(".runtime-stage-*")) and not (tmp_path / "launched").exists()


@pytest.mark.parametrize("fault", ["collision", "extract"])
def test_atomic_racing_target_or_partial_extract_never_launches_and_cleans_only_stage(tmp_path, fault):
    result, root, job, *_ = local_phase(tmp_path, publication_fault=fault)
    assert result.returncode != 0 and not (tmp_path / "launched").exists()
    assert not list((root / "jobs").glob(".runtime-stage-*"))
    if fault == "collision": assert (job / "original").read_text() == "racing owner"
    else: assert not job.exists()
    assert (root / "foreign").read_text() == "untouched"


def test_flags_exclusive_and_control_caps_fail_before_azure(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args: pytest.fail("no Git read on incompatible flags"))
    with pytest.raises(SystemExit): launcher.main(["--name", "test", "--script", "infra/run_smoke.sh", "--github-source", "--reuse-published"])
    raw, encoded, sha, _ = fixture_archive()
    with pytest.raises(ValueError): launcher.transport_commands(encoded, sha, raw, REV, "infra/run_smoke.sh", "test", [], github_source=True, reuse_published=True)
    monkeypatch.setattr(launcher, "STAGED_SCRIPT_BYTES", 100)
    with pytest.raises(RuntimeError, match="120KB"): launcher.transport_commands(encoded, sha, raw, REV, "infra/run_smoke.sh", "test", [], github_source=True)
    monkeypatch.setattr(launcher, "MAX_CODE_CONTROL_BYTES", 100)
    with pytest.raises(RuntimeError, match="256KB"): launcher.github_archive_descriptor(raw, REV, sha)


def test_explicit_github_cli_one_invocation_without_local_http(monkeypatch):
    raw, _, _, files = fixture_archive(); calls = []
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("no local network"))
    def git(argv):
        assert argv[:3] == ["rtk", "proxy", "git"]
        if argv[3] == "cat-file": return b"commit" if argv[4] == "-t" else b""
        if argv[3] == "rev-parse": return (REV + "\n").encode()
        if argv[3] == "archive": return raw
        pytest.fail(str(argv))
    monkeypatch.setattr(launcher.subprocess, "check_output", git)
    monkeypatch.setattr(launcher, "invoke_transport", lambda *args: calls.append(args))
    launcher.main(["--name", "test", "--script", "infra/run_smoke.sh", "--revision", REV, "--github-source"])
    assert len(calls) == 1 and "github-published-dispatched" in calls[0][1]


@pytest.mark.parametrize("mode", ["inline", "github", "reuse"])
def test_cli_caps_actual_transport_not_unused_inline_archive(monkeypatch, mode):
    raw, _, _, _ = fixture_archive(large=True); calls = []
    source_archive, paths = launcher.runtime_archive(raw, "infra/run_smoke.sh")
    encoded, sha = launcher.encoded_runtime_archive(source_archive)
    assert "configs/payload.json" in paths
    # Descriptor is tiny; the actual full source remains intact remotely.
    monkeypatch.setattr(launcher, "MAX_CODE_CONTROL_BYTES", 10000)
    assert len(encoded) > launcher.MAX_CODE_CONTROL_BYTES
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: pytest.fail("no local HTTP"))
    def git(argv):
        assert argv[:3] == ["rtk", "proxy", "git"]
        if argv[3] == "cat-file": return b"commit" if argv[4] == "-t" else b""
        if argv[3] == "rev-parse": return (REV + "\n").encode()
        if argv[3] == "archive": return raw
        pytest.fail(str(argv))
    monkeypatch.setattr(launcher.subprocess, "check_output", git)
    monkeypatch.setattr(launcher, "invoke_transport", lambda *args: calls.append(args))
    argv = ["--name", "test", "--script", "infra/run_smoke.sh", "--revision", REV]
    if mode != "inline":
        monkeypatch.setattr(launcher, "encoded_runtime_archive", lambda *_: pytest.fail("unused inline encoding"))
        launcher.main([*argv, "--github-source" if mode == "github" else "--reuse-published"])
        assert len(calls) == 1
        assert ("github-published-dispatched" if mode == "github" else "published-reused-dispatched") in calls[0][1]
        assert sha in calls[0][0]
        if mode == "github":
            assert '"configs/payload.json"' in calls[0][0]
        assert len(calls[0][0].encode()) <= launcher.STAGED_SCRIPT_BYTES
        if mode == "reuse":
            assert all(x not in calls[0][0] for x in ("base64 -d", "PY_PUBLISH", "mkdir ", "chmod "))
    else:
        with pytest.raises(RuntimeError, match="256KB"):
            launcher.main(argv)
        assert calls == []
