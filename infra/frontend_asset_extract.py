"""Private, non-promoting extraction of a caller-pinned frontend asset archive.

Complete genuine archive verification precedes every output write. There is no
merge, cleanup of partial trees, model execution, image import or eligibility
claim. The caller records the returned small receipt on the remote control host.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import tarfile

SOURCE_FILES = ("infra/frontend_asset_extract.py", "infra/frontend_asset_archive.py")
SCHEMA = "world_reward.frontend_asset_extract.v1"


def _state(path):
    s = Path(path).lstat()
    return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink


def _canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Canonical private asset path required")
    return path


def _read_source(path, pin):
    if (type(pin) is not dict or set(pin) != {"bytes", "sha256"} or type(pin["bytes"]) is not int
            or not 1 <= pin["bytes"] <= 1_000_000 or type(pin["sha256"]) is not str
            or len(pin["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in pin["sha256"])):
        raise ValueError("Independent extractor and archive source pins required")
    path = _canonical(path); before = _state(path)
    if not stat.S_ISREG(before[2]) or before[6] != 1 or before[3] != pin["bytes"]:
        raise ValueError("Bounded single-link source required")
    raw = path.read_bytes()
    if _state(path) != before or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Independent helper source pin differs before import")
    return raw


def _contract(source_pins):
    if type(source_pins) is not dict or set(source_pins) != set(SOURCE_FILES):
        raise ValueError("Complete independent extractor/archive helper pins required")
    code = _canonical(Path(__file__).parent.parent)
    for name in SOURCE_FILES: _read_source(code / name, source_pins[name])
    path = code / SOURCE_FILES[1]
    spec = importlib.util.spec_from_file_location("world_reward_verified_frontend_archive", path)
    module = importlib.util.module_from_spec(spec)
    # Compile the bytes already authenticated above, never an unpinned path
    # import or transitive GPU/runtime module. The archive helper is stdlib only.
    exec(compile(_read_source(path, source_pins[SOURCE_FILES[1]]), str(path), "exec"), module.__dict__)
    return module


def _write(path, stream, row, gate, directories):
    _canonical(path.parent)
    if path.parent not in directories or _state(path.parent)[:2] != directories[path.parent]:
        raise ValueError("Owned private output parent was replaced")
    reader = gate._Reader(stream, row["bytes"])
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    with os.fdopen(fd, "wb") as output:
        inode = os.fstat(output.fileno())
        for block in iter(lambda: reader.read(4 * 1024 * 1024), b""): output.write(block)
        reader.finish(row); output.flush(); os.fsync(output.fileno())
        if _state(path)[:2] != (inode.st_dev, inode.st_ino): raise ValueError("Owned extracted file was replaced")
    if _state(path)[2] & 0o777 != 0o400: raise ValueError("Extracted asset must remain private readonly")


def _inventory(destination, entries, manifest_raw, gate, directories):
    expected = {destination / name: row for name, row in entries.items()}
    expected[destination / gate.MANIFEST] = dict(type="file", bytes=len(manifest_raw), sha256=hashlib.sha256(manifest_raw).hexdigest(),
        git_blob_sha1=hashlib.sha1(b"blob " + str(len(manifest_raw)).encode() + b"\0" + manifest_raw).hexdigest())
    pending = [destination]; observed = set()
    while pending:
        directory = pending.pop(); _canonical(directory)
        if directory not in directories or _state(directory)[:2] != directories[directory] or _state(directory)[2] & 0o777 != 0o700:
            raise ValueError("Owned complete private directory inventory differs")
        for path in directory.iterdir():
            observed.add(path); before = _state(path)
            if stat.S_ISDIR(before[2]): pending.append(path); continue
            if path not in expected: raise ValueError("Unlisted extracted path forbidden")
            row = expected[path]
            if row["type"] == "symlink":
                _canonical(path.parent)
                if not stat.S_ISLNK(before[2]) or before[6] != 1 or os.readlink(path) != row["link"]:
                    raise ValueError("Literal audited relative link changed")
                if path.resolve() != (destination / row["target"]).resolve() or not path.resolve().is_relative_to(destination):
                    raise ValueError("Extracted HF link graph escaped private payload")
            else:
                if before[2] & 0o777 != 0o400 or gate._identity(path, row["bytes"], empty=row["bytes"] == 0) != {k: row[k] for k in ("bytes", "sha256", "git_blob_sha1")}:
                    raise ValueError("Complete extracted file bytes/blob/mode differs")
            if _state(path) != before: raise ValueError("Extracted payload changed during inventory")
    if observed != set(expected) | (set(directories) - {destination}): raise ValueError("Missing or foreign private payload member")


def extract_archive(archive_path, archive_pin, manifest_pin, inventory_code, inventory_source_pins, destination, source_pins):
    """Return PASS/FAIL receipt, retaining owned private partial output on failure.

    Source pins are independently supplied byte/SHA identities for SOURCE_FILES.
    Archive and manifest pins are from the frozen remote export, not recomputed
    trust assertions. A failure before reservation creates no destination.
    """
    report = dict(schema=SCHEMA, stage="frontend_asset_extract", status="fail", promotion_performed=False, models_loaded=False,
        images_imported=False, license_eligibility_verified=False, training_overlap_verified=False,
        replica_ready=False, CUDA_verified=False, partial_output_retained=False)
    reserved = False; directories = {}
    try:
        gate = _contract(source_pins)
        gate._pin(archive_pin, gate.MAX_TOTAL + 10_000_000); gate._pin(manifest_pin, gate.MAX_JSON)
        report.update(source_helpers=source_pins, archive_identity=archive_pin, manifest_identity=manifest_pin)
        archive_path = _canonical(archive_path); before = _state(archive_path)
        report["prewrite_archive_verification"] = gate.verify_archive(archive_path, archive_pin, manifest_pin, inventory_code, inventory_source_pins)
        if _state(archive_path) != before: raise ValueError("Pinned archive changed before any output write")
        with tarfile.open(archive_path, "r|") as archive:
            first = archive.next()
            if first is None or first.name != gate.MANIFEST or not first.isreg() or first.size != manifest_pin["bytes"]:
                raise ValueError("Verified first manifest changed before extraction")
            manifest_raw = archive.extractfile(first).read()
            if hashlib.sha256(manifest_raw).hexdigest() != manifest_pin["sha256"]: raise ValueError("Pinned manifest changed before output reservation")
            manifest = gate._parse(manifest_raw); entries = manifest["entries"]
            if _state(archive_path) != before: raise ValueError("Archive changed before destination reservation")
            destination = _canonical(destination)
            if not destination.parent.is_dir() or destination.exists() or destination.is_symlink():
                raise ValueError("Absent private destination under existing data disk required; no merge/overwrite")
            destination.mkdir(mode=0o700); reserved = True; directories[destination] = _state(destination)[:2]
            report.update(imported_relative_root=destination.name, partial_output_retained=True)
            parents = {destination}
            for name in entries:
                gate._name(name); current = (destination / name).parent
                while current != destination: parents.add(current); current = current.parent
            for path in sorted(parents - {destination}, key=lambda p: (len(p.parts), str(p))):
                _canonical(path.parent)
                if _state(path.parent)[:2] != directories[path.parent]: raise ValueError("Owned directory parent changed")
                path.mkdir(mode=0o700); directories[path] = _state(path)[:2]
            row = dict(bytes=len(manifest_raw), sha256=hashlib.sha256(manifest_raw).hexdigest(), git_blob_sha1=hashlib.sha1(b"blob " + str(len(manifest_raw)).encode() + b"\0" + manifest_raw).hexdigest())
            _write(destination / gate.MANIFEST, io.BytesIO(manifest_raw), row, gate, directories)
            links = []
            for name in sorted(entries):
                row = entries[name]; member = archive.next(); expected = gate._info(name, row)
                if member is None or (member.name, member.type, member.size, member.linkname) != (name, expected.type, expected.size, expected.linkname):
                    raise ValueError("Verified member changed during extraction")
                if row["type"] == "symlink": links.append((name, row))
                else: _write(destination / name, archive.extractfile(member), row, gate, directories)
            if archive.next() is not None: raise ValueError("Foreign archive member during extraction")
        # Every regular file exists and is hash-checked before a single HF link
        # is created. Only the three already-verified literal relative links.
        for name, row in links:
            path = destination / name; _canonical(path.parent)
            if _state(path.parent)[:2] != directories[path.parent]: raise ValueError("Owned link parent changed")
            os.symlink(row["link"], path)
        _inventory(destination, entries, manifest_raw, gate, directories)
        gate.verify_archive(archive_path, archive_pin, manifest_pin, inventory_code, inventory_source_pins)
        if _state(archive_path) != before: raise ValueError("Original archive changed after extraction")
        _contract(source_pins)
        _inventory(destination, entries, manifest_raw, gate, directories)
        report.update(status="pass", partial_output_retained=False, imported_files=sum(r["type"] == "file" for r in entries.values()),
            imported_links=len(links), imported_bytes=sum(r.get("bytes", 0) for r in entries.values()),
            payload_entries_sha256=manifest["inventory"]["entries_sha256"], complete_payload_rehashed=True,
            manifest_preserved_byte_identically=True, source_helpers_rehashed=True, archive_rehashed_after_extraction=True)
    except Exception as caught:
        report.update(error_type=type(caught).__name__, error="Asset extraction failed closed; no merge/promotion or partial tree cleanup", partial_output_retained=reserved)
    return report
