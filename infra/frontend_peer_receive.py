"""Private Azure SSH byte receiver; no extraction, network client or model code.

The launching server authenticates this source before installing its forced
command. Only the fixed published archive hash authorizes incoming bytes.
Failed owned streams remain private evidence; a fresh directory is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys

DESTINATION = Path("/srv/world-reward-data/frontend-assets-v1")
ARCHIVE_BYTES = 19_911_464_960
ARCHIVE_SHA256 = "5b817ea15e98f1f18165529fcac3ca0b7fc9f88b96d22f2195396db6ffbf8342"
COMMAND = "world-reward-frontend-assets-v1"
DEADLINE = 1800
CHUNK = 4 * 1024 * 1024


def canonical(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Canonical nonsymlink receiver path required")
    return path


def _state(path):
    s = path.lstat()
    return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_nlink, s.st_uid


def _directory(path, owned=None):
    canonical(path); current = _state(path)
    if (not stat.S_ISDIR(current[2]) or current[2] & 0o777 != 0o700
            or current[5] != os.geteuid() or owned is not None and current[:2] != owned):
        raise ValueError("Original owned private receiver directory required")
    return current[:2]


def _source_identity(readonly=False):
    source = canonical(Path(__file__)); before = _state(source)
    if (not stat.S_ISREG(before[2]) or before[4] != 1 or not 1 <= before[3] <= 100_000
            or readonly and (before[2] & 0o222 or before[5] != os.geteuid())):
        raise ValueError("Actual immutable receiver source required")
    raw = source.read_bytes()
    if _state(source) != before: raise ValueError("Receiver source changed")
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _archive_state(path, owned):
    canonical(path); current = _state(path)
    if (current[:2] != owned or not stat.S_ISREG(current[2])
            or current[2] & 0o777 != 0o400 or current[4] != 1 or current[5] != os.geteuid()):
        raise ValueError("Owned private archive replaced or aliased")
    return current


def _receipt(destination, owned, report):
    _directory(destination, owned)
    raw = json.dumps(report, sort_keys=True, allow_nan=False).encode() + b"\n"
    if len(raw) > 4000: raise ValueError("Receiver receipt exceeds bound")
    path = destination / "report.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, "wb") as stream:
        info = os.fstat(stream.fileno()); receipt_owned = info.st_dev, info.st_ino
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    _directory(destination, owned); _archive_state(path, receipt_owned)


def receive(stream, destination, size, sha):
    """Receive exact bytes into one fresh private directory; return tiny verdict.

    Invalid preconditions raise before writes. Once reception starts, a failed
    stream retains its owned partial archive and a sanitized failure receipt.
    No unknown file is replaced, deleted or chmodded, including on failure.
    """
    if (type(size) is not int or not 1 <= size <= ARCHIVE_BYTES or type(sha) is not str
            or re.fullmatch(r"[0-9a-f]{64}", sha) is None):
        raise ValueError("Bounded independent transport byte and SHA pins required")
    destination = canonical(destination); owned = _directory(destination)
    if any(destination.iterdir()): raise FileExistsError("Fresh empty receiver directory required")
    source = _source_identity()
    report = dict(schema="world_reward.frontend_peer_receive.v1", stage="frontend_peer_receive",
        status="fail", phase="stream", expected_archive=dict(bytes=size, sha256=sha),
        bytes_received=0, receiver_source_identity=source, extraction_performed=False,
        replica_ready=False, license_eligibility_verified=False, receipt_written=False)
    path = destination / "archive.tar"; digest = hashlib.sha256(); archive_owned = None
    try:
        _directory(destination, owned)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "wb") as output:
            info = os.fstat(output.fileno()); archive_owned = info.st_dev, info.st_ino
            while report["bytes_received"] < size:
                requested = min(CHUNK, size - report["bytes_received"])
                block = stream.read(requested)
                if type(block) is not bytes or not block or len(block) > requested:
                    raise ValueError("Truncated or malformed bounded binary stream")
                output.write(block); digest.update(block); report["bytes_received"] += len(block)
            extra = stream.read(1)
            if type(extra) is not bytes or extra:
                raise ValueError("Exactly pinned bytes followed by EOF required")
            output.flush(); os.fsync(output.fileno())
        report["phase"] = "verify"
        before = _archive_state(path, archive_owned)
        if before[3] != size or digest.hexdigest() != sha:
            raise ValueError("Independent incoming archive identity differs")
        final = hashlib.sha256()
        with path.open("rb") as sealed:
            for block in iter(lambda: sealed.read(CHUNK), b""): final.update(block)
        if (_archive_state(path, archive_owned) != before or final.hexdigest() != sha
                or _source_identity() != source):
            raise ValueError("Received archive or source changed after stream")
        _directory(destination, owned)
        if {p.name for p in destination.iterdir()} != {"archive.tar"}:
            raise ValueError("Unknown result appeared during receive")
        report.update(status="pass", phase="complete", archive_identity=dict(bytes=size, sha256=sha))
    except BaseException as error:
        report.update(status="fail", error_type=type(error).__name__,
            error="Incoming asset transport failed closed; owned partial bytes retained")
    try:
        report["receipt_written"] = True
        _receipt(destination, owned, report)
    except BaseException as error:
        report.update(status="fail", phase="receipt", receipt_written=False,
            error_type=type(error).__name__, error="Owned receiver receipt could not be safely committed")
    return report


def validate_peer(connection, original_command):
    fields = connection.split() if type(connection) is str else []
    if (len(fields) != 4 or fields[0] != "10.0.0.4" or fields[2:] != ["10.0.0.9", "2222"]
            or not re.fullmatch(r"[0-9]{1,5}", fields[1]) or not 1 <= int(fields[1]) <= 65535
            or original_command != COMMAND):
        raise ValueError("Only exact authorized private SSH peer and forced command allowed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, add_help=False)
    parser.add_argument("--receive", action="store_true", required=True); parser.parse_args(argv)
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise ValueError("Linux root-owned forced receiver required")
    validate_peer(os.environ.get("SSH_CONNECTION"), os.environ.get("SSH_ORIGINAL_COMMAND"))
    _source_identity(readonly=True)
    def expired(*_): raise TimeoutError("Private transport deadline expired")
    previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(DEADLINE)
    try:
        report = receive(sys.stdin.buffer, DESTINATION, ARCHIVE_BYTES, ARCHIVE_SHA256)
    finally:
        signal.alarm(0)
        for sig, handler in previous.items(): signal.signal(sig, handler)
    print(json.dumps({k: report[k] for k in ("stage", "status", "bytes_received", "receipt_written", "replica_ready")}, sort_keys=True), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    try: sys.exit(main())
    except BaseException as error:
        if isinstance(error, SystemExit): raise
        print(json.dumps(dict(stage="frontend_peer_receive", status="fail", error_type=type(error).__name__,
            error="Private forced receiver precondition failed")), flush=True)
        sys.exit(1)
