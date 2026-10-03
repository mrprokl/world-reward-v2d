"""Tiny byte streams only; no SSH, Azure, model, media or archive extraction."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def receiver():
    path = Path(__file__).resolve().parents[1] / "infra/frontend_peer_receive.py"
    spec = importlib.util.spec_from_file_location("wr_peer_receiver", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def destination(tmp_path):
    path = tmp_path / "incoming"; path.mkdir(mode=0o700)
    return path


def run(receiver, destination, raw=b"tiny known asset archive", stream=None):
    return receiver.receive(stream or io.BytesIO(raw), destination, len(raw), hashlib.sha256(raw).hexdigest())


def test_exact_private_stream_and_receipt(receiver, destination):
    raw = b"tiny known asset archive"; report = run(receiver, destination, raw)
    assert report["status"] == "pass" and report["receipt_written"] is True
    assert report["replica_ready"] is report["license_eligibility_verified"] is report["extraction_performed"] is False
    assert (destination / "archive.tar").read_bytes() == raw
    assert json.loads((destination / "report.json").read_bytes()) == report
    assert {p.name for p in destination.iterdir()} == {"archive.tar", "report.json"}
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in destination.iterdir())
    with pytest.raises(FileExistsError): run(receiver, destination)


def test_private_creation_even_umask_zero(receiver, destination):
    class Stream(io.BytesIO):
        def read(self, size=-1):
            assert (destination / "archive.tar").stat().st_mode & 0o777 == 0o400
            return super().read(size)
    before = os.umask(0)
    try: assert run(receiver, destination, stream=Stream(b"tiny known asset archive"))["status"] == "pass"
    finally: os.umask(before)


@pytest.mark.parametrize("kind", ["truncated", "oversize", "tampered", "nonbinary", "overread", "deadline", "secret"])
def test_failed_stream_private_partial_receipt_sanitized(receiver, destination, kind):
    raw = b"tiny known asset archive"
    class Broken:
        def read(self, n):
            if kind == "nonbinary": return "not bytes"
            if kind == "overread": return raw + b"excess"
            if kind == "deadline": raise TimeoutError("unsafe supplied message")
            raise ValueError("hf_" + "x" * 30 + " private token")
    stream = io.BytesIO(raw[:-1]) if kind == "truncated" else io.BytesIO(raw + b"!") if kind == "oversize" else io.BytesIO(b"X" + raw[1:]) if kind == "tampered" else Broken()
    report = run(receiver, destination, raw, stream)
    assert report["status"] == "fail" and report["receipt_written"] is True
    assert (destination / "archive.tar").is_file() and (destination / "archive.tar").stat().st_mode & 0o777 == 0o400
    assert json.loads((destination / "report.json").read_bytes()) == report
    assert "hf_" not in json.dumps(report) and "unsafe supplied" not in json.dumps(report)


@pytest.mark.parametrize("kind", ["mode", "relative", "alias", "parent_alias", "file", "existing", "symlink"])
def test_destination_preconditions_preserve_user_work(receiver, destination, tmp_path, kind):
    target = destination
    if kind == "mode": destination.chmod(0o755)
    elif kind == "relative": target = Path("relative")
    elif kind == "alias": target = tmp_path / "alias"; target.symlink_to(destination)
    elif kind == "parent_alias":
        parent = tmp_path / "parent_alias"; parent.symlink_to(tmp_path); target = parent / "incoming"
    elif kind == "file": destination.rmdir(); destination.write_bytes(b"user work")
    elif kind == "existing": (destination / "archive.tar").write_bytes(b"user work")
    else: (destination / "archive.tar").symlink_to("missing")
    with pytest.raises((ValueError, FileExistsError)): run(receiver, target)
    assert not (destination / "report.json").exists()
    if kind == "existing": assert (destination / "archive.tar").read_bytes() == b"user work"
    if kind == "file": assert destination.read_bytes() == b"user work"


@pytest.mark.parametrize("kind", ["directory", "archive", "hardlink", "mode", "unknown_receipt"])
def test_stream_time_replacements_never_modified_or_overwritten(receiver, destination, tmp_path, kind):
    raw = b"tiny known asset archive"; foreign = b"foreign user work"
    class Stream(io.BytesIO):
        once = False
        def read(self, n):
            if not self.once:
                self.once = True
                path = destination / "archive.tar"
                if kind == "directory":
                    destination.rename(tmp_path / "original"); destination.mkdir(mode=0o700)
                    (destination / "user").write_bytes(foreign)
                elif kind == "archive":
                    path.unlink(); path.write_bytes(foreign); path.chmod(0o644)
                elif kind == "hardlink": os.link(path, tmp_path / "alias")
                elif kind == "mode": path.chmod(0o644)
                else: (destination / "report.json").write_bytes(foreign)
            return super().read(n)
    report = run(receiver, destination, raw, Stream(raw))
    assert report["status"] == "fail"
    if kind == "directory":
        assert (destination / "user").read_bytes() == foreign and not (destination / "report.json").exists()
    elif kind == "archive":
        assert (destination / "archive.tar").read_bytes() == foreign and (destination / "archive.tar").stat().st_mode & 0o777 == 0o644
    elif kind == "unknown_receipt": assert (destination / "report.json").read_bytes() == foreign


@pytest.mark.parametrize("connection,command", [
    (None, "world-reward-frontend-assets-v1"), ("10.0.0.5 1234 10.0.0.9 2222", "world-reward-frontend-assets-v1"),
    ("10.0.0.4 1234 10.0.0.8 2222", "world-reward-frontend-assets-v1"),
    ("10.0.0.4 1234 10.0.0.9 22", "world-reward-frontend-assets-v1"),
    ("10.0.0.4 0 10.0.0.9 2222", "world-reward-frontend-assets-v1"),
    ("10.0.0.4 65536 10.0.0.9 2222", "world-reward-frontend-assets-v1"),
    ("10.0.0.4 1234 10.0.0.9 2222 extra", "world-reward-frontend-assets-v1"),
    ("10.0.0.4 1234 10.0.0.9 2222", "world-reward-frontend-assets-v1; sh"),
])
def test_exact_forced_peer(receiver, connection, command):
    with pytest.raises(ValueError): receiver.validate_peer(connection, command)


def test_fixed_main_contract_no_extra_flags_or_privileged_imports(receiver):
    receiver.validate_peer("10.0.0.4 1234 10.0.0.9 2222", receiver.COMMAND)
    for args in ([], ["--receive", "--root", "/tmp"], ["--receive", "--size", "1"]):
        with pytest.raises(SystemExit): receiver.main(args)
    assert receiver.DESTINATION == Path("/srv/world-reward-data/frontend-assets-v1")
    assert receiver.ARCHIVE_BYTES == 19911464960 and receiver.DEADLINE == 1800
    source = Path(receiver.__file__).read_text()
    assert not any(x in source for x in ("import subprocess", "import socket", "import torch", "import tarfile", "extractall", "os.system", "SSH_AUTH_SOCK"))


@pytest.mark.parametrize("fault", ["host", "uid", "peer", "command", "source"])
def test_main_checks_host_peer_and_source_before_stream(receiver, monkeypatch, fault):
    monkeypatch.setattr(receiver.platform, "system", lambda: "Darwin" if fault == "host" else "Linux")
    monkeypatch.setattr(receiver.os, "geteuid", lambda: 1000 if fault == "uid" else 0)
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.5 1234 10.0.0.9 2222" if fault == "peer" else "10.0.0.4 1234 10.0.0.9 2222")
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "unexpected" if fault == "command" else receiver.COMMAND)
    def source(readonly=False):
        assert readonly is True
        raise ValueError("Source must be checked before stream")
    monkeypatch.setattr(receiver, "_source_identity", source)
    monkeypatch.setattr(receiver, "receive", lambda *args: pytest.fail("Untrusted main opened incoming stream"))
    with pytest.raises(ValueError): receiver.main(["--receive"])


def test_source_tampering_invalidates_completed_stream(receiver, destination, monkeypatch):
    original = receiver._source_identity; calls = []
    def changed(readonly=False):
        value = original(readonly); calls.append(1)
        return value if len(calls) == 1 else {**value, "sha256": "0" * 64}
    monkeypatch.setattr(receiver, "_source_identity", changed)
    report = run(receiver, destination)
    assert report["status"] == "fail" and report["receipt_written"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("size,sha", [(True, "0" * 64), (0, "0" * 64), (-1, "0" * 64), (19911464961, "0" * 64), (1, "not SHA")])
def test_bad_pins_before_any_write(receiver, destination, size, sha):
    with pytest.raises(ValueError): receiver.receive(io.BytesIO(b"x"), destination, size, sha)
    assert not any(destination.iterdir())
