"""Strict CPU-only archive contract for an independently pinned frontend audit.

No extraction, acquisition, image operations, model imports or eligibility claims.
The caller supplies old inventory/source pins; this module never invents those
pins from the files it is about to trust. All production archive bytes stay Azure.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import tarfile
import types

SCHEMA = "world_reward.frontend_asset_archive.v1"
MANIFEST = "world-reward-frontend-assets-manifest.json"
INVENTORY_SOURCES = ("infra/frontend_replica_inventory.py", "infra/run_frontend_replica_inventory.sh")
MAX_TOTAL = 21_000_000_000
MAX_JSON = 2_000_000
REG4 = ("dinov2_vitl14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth")
RELEASES = {"dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _parse(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError("Duplicate archive metadata key")
            result[key] = value
        return result
    def bad(_): raise ValueError("Nonfinite archive metadata")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(item):
        if isinstance(item, dict):
            for v in item.values(): finite(v)
        elif isinstance(item, list):
            for v in item: finite(v)
        elif isinstance(item, float) and not math.isfinite(item): bad(None)
    finite(value)
    return value


def _name(name):
    if (type(name) is not str or not name or "\\" in name or "\0" in name or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name or any(p in ("", ".", "..", ".git", ".secrets", "__pycache__") for p in name.split("/"))):
        raise ValueError("Noncanonical archive whitelist member")
    return name


def _canonical(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)) or path.resolve() != path:
        raise ValueError("Canonical nonaliased archive path required")
    return path


def _state(path):
    s = Path(path).lstat()
    return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink


def _pin(pin, maximum=MAX_TOTAL, *, blob=False, empty=False):
    expected = {"bytes", "sha256", "git_blob_sha1"} if blob else {"bytes", "sha256"}
    if (type(pin) is not dict or set(pin) != expected or type(pin["bytes"]) is not int
            or not (0 if empty else 1) <= pin["bytes"] <= maximum
            or type(pin["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) is None
            or blob and (type(pin["git_blob_sha1"]) is not str or re.fullmatch(r"[0-9a-f]{40}", pin["git_blob_sha1"]) is None)):
        raise ValueError("Independent bounded byte/SHA pin required")


def _identity(path, maximum=MAX_TOTAL, *, empty=False):
    path = _canonical(path); before = _state(path)
    if not stat.S_ISREG(before[2]) or before[6] != 1 or not (0 if empty else 1) <= before[3] <= maximum:
        raise ValueError("Bounded single-link regular archive input required")
    sha = hashlib.sha256(); blob = hashlib.sha1(b"blob " + str(before[3]).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): sha.update(block); blob.update(block)
    if _state(path) != before: raise ValueError("Archive input changed while hashing")
    return {"bytes": before[3], "sha256": sha.hexdigest(), "git_blob_sha1": blob.hexdigest()}


def _read_pinned(path, pin):
    _pin(pin, MAX_JSON)
    actual = _identity(path, MAX_JSON)
    if {k: actual[k] for k in pin} != pin: raise ValueError("Pinned metadata identity differs before parsing")
    before = _state(path); raw = Path(path).read_bytes()
    if _state(path) != before or len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("Pinned metadata changed before parsing")
    return raw


def _source(code, pins):
    code = _canonical(code)
    if type(pins) is not dict or set(pins) != set(INVENTORY_SOURCES): raise ValueError("Both independent old inventory source pins required")
    for name, pin in pins.items():
        _pin(pin, 1_000_000, blob=True)
        if _identity(code / name, 1_000_000) != pin or (code / name).stat().st_mode & 0o222:
            raise ValueError("Frozen inventory helper identity/mode differs")
    raw = (code / INVENTORY_SOURCES[0]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != pins[INVENTORY_SOURCES[0]]["sha256"]: raise ValueError("Inventory source changed before import")
    module = types.ModuleType("world_reward_authenticated_frontend_inventory")
    module.__file__ = str(code / INVENTORY_SOURCES[0])
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module


def _inventory_pin(pin):
    if type(pin) is not dict or set(pin) != {"bytes", "sha256", "producer_revision", "entries_sha256", "files", "links", "total_bytes"}:
        raise ValueError("Independent complete inventory receipt pin required")
    _pin({k: pin[k] for k in ("bytes", "sha256")}, MAX_JSON)
    if (type(pin["producer_revision"]) is not str or re.fullmatch(r"[0-9a-f]{40}", pin["producer_revision"]) is None
            or type(pin["entries_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", pin["entries_sha256"]) is None
            or any(type(pin[k]) is not int for k in ("files", "links", "total_bytes"))
            or not 1 <= pin["files"] <= 2000 or pin["links"] != 3 or not 1 <= pin["total_bytes"] <= MAX_TOTAL):
        raise ValueError("Independent bounded inventory cohort metadata required")


def _repositories(gate):
    return (("vendor/video_to_data", gate.UPSTREAM, None),
        ("weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main", gate.DINO_REVS["dinov3"], "dinov3"),
        ("weights/sam3d/torch_home/hub/facebookresearch_dinov2_main", gate.DINO_REVS["dinov2"], "dinov2"))


def _entries(gate, entries, sources, moge):
    if type(entries) is not dict or not entries or len(entries) > 2000 or type(sources) is not list or len(sources) != 3:
        raise ValueError("Complete bounded audited asset inventory required")
    exact = {**{n: "asset" for n in gate.FIXED}, **{n: "license_card_or_config" for n in gate.CARDS},
        **{n: "source_receipt" for n in gate.RECEIPTS[:2]},
        **{"weights/sam3d/torch_home/hub/checkpoints/" + n: "reg4_first_observed_receipt_bound" for n in REG4}}
    used = set(exact)
    for summary, (repository, revision, dino) in zip(sources, _repositories(gate)):
        rows = {n: r for n, r in entries.items() if n.startswith(repository + "/") and r.get("role") == "public_source"}
        relative = {n[len(repository) + 1:] for n in rows}
        required = {"hubconf.py", "LICENSE.md" if dino == "dinov3" else "LICENSE", "MODEL_CARD.md"} if dino else {
            "LICENSE", gate.BODY_PACKAGE + "/LICENSE", gate.BODY_PACKAGE + "/__init__.py",
            *[gate.CARI + "/lib_mhr/" + n for n in gate.LIB_MHR], *[gate.CARI + "/prep/" + n for n in gate.PREP],
            *[gate.CARI + "/" + n for n in gate.EXTRA_SOURCE]}
        expected = dict(path=repository, revision=revision, selected_worktree_clean_verified=True,
            complete_checkout_clean_verified=False, source_files=len(rows), source_sha256=gate.digest_json(rows))
        if summary != expected or not required <= relative or not all(gate.selected_source(n, dino=dino) for n in relative):
            raise ValueError("Audited selected source closure differs")
        if not dino and {n[len(gate.BODY_PACKAGE) + 1:] for n in relative if n.startswith(gate.BODY_PACKAGE + "/data/")} != gate.BODY_DATA:
            raise ValueError("Original six Body data-Python helpers required")
        used.update(rows)
    if type(moge) is not dict or set(moge) != {"model.pt", "README.md"}: raise ValueError("Exactly two audited MoGe bindings required")
    graph = set(); links = 0
    for label, binding in moge.items():
        current = gate.MOGE_SNAPSHOT + "/" + label; seen = set()
        for _ in range(5):
            if current in seen or current not in entries: raise ValueError("Missing or cyclic audited MoGe graph")
            seen.add(current); graph.add(current); row = entries[current]
            if row.get("role") != "internal_moge1": raise ValueError("MoGe graph role differs")
            if row.get("type") == "file": break
            link = row.get("link")
            if type(link) is not str or not link or PurePosixPath(link).is_absolute() or "\\" in link or "\0" in link:
                raise ValueError("Exact relative audited MoGe link required")
            destination = _name(posixpath.normpath(posixpath.join(posixpath.dirname(current), link)))
            if (row != dict(type="symlink", link=link, target=destination, role="internal_moge1") or not
                    (re.fullmatch(re.escape(gate.MOGE_REPO) + r"/blobs/[0-9a-f]{40,64}", destination)
                    or re.fullmatch(re.escape(gate.HF) + r"/blobs/[0-9a-f]{2}/[0-9a-f]{40,64}", destination))):
                raise ValueError("MoGe link escapes audited exact blob roots")
            current = destination
        else: raise ValueError("Audited MoGe chain exceeds four links")
        identity = {k: row.get(k) for k in ("bytes", "sha256", "git_blob_sha1")}
        expected = dict(snapshot=gate.MOGE_SNAPSHOT + "/" + label, resolved_file=current, links=len(seen) - 1,
            **identity, independent_primary_model_identity_verified=label == "model.pt", XET_path_discovered_not_fabricated=True)
        if binding != expected or label == "model.pt" and (identity["sha256"], identity["bytes"]) != gate.MOGE_EXPECTED:
            raise ValueError("Audited independent MoGe identity differs")
    used.update(graph)
    if used != set(entries): raise ValueError("Foreign/evidence/data/cache member not export eligible")
    for name, row in entries.items():
        _name(name)
        if name in exact and row.get("role") != exact[name]: raise ValueError("Original audited asset role differs")
        if row.get("type") == "symlink": links += 1; continue
        if set(row) != {"type", "bytes", "sha256", "git_blob_sha1", "role"} or row["type"] != "file":
            raise ValueError("Only audited regular files and three MoGe links allowed")
        empty = row["role"] == "public_source" and name.endswith("/__init__.py")
        maximum = 1_000_000 if row["role"] in ("public_source", "license_card_or_config") else MAX_JSON if row["role"] == "source_receipt" else 7_000_000_000
        _pin({k: row[k] for k in ("bytes", "sha256", "git_blob_sha1")}, maximum, blob=True, empty=empty)
        if name in gate.FIXED and (row["sha256"], row["bytes"]) != gate.FIXED[name]: raise ValueError("Independent fixed asset identity differs")
    if links != 3 or sum(r.get("bytes", 0) for r in entries.values()) > MAX_TOTAL: raise ValueError("Original three-link bounded payload required")


def _receipt(gate, name, raw, original_root, entries):
    value = _parse(raw); gate.no_secrets(value)
    if name == gate.RECEIPTS[0]:
        bindings = (("nvidia/cari4d_commercial", "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9", "path", "weights/cari4d/cari4d"),
            ("facebook/sam-3d-body-dinov3", gate.BODY_REV, "path", gate.BODY),
            ("facebook/sam-3d-objects", gate.OBJECT_REV, "path", gate.OBJECT),
            ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", "path", "weights/sam2"),
            ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", "path", "weights/grounding_dino"),
            ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340", "cache_dir", "weights/cari4d/hf_home/hub"),
            ("Ruicheng/moge-vitl", gate.MOGE_REV, "cache_dir", gate.HF))
        expected = dict(assets=[dict(repo_id=repo, revision=revision, **{field: str(Path(original_root) / folder)}) for repo, revision, field, folder in bindings],
            cari4d_sha256="78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3", mhr_license="Apache-2.0",
            scope="principal_hf_and_mhr_assets_only", auxiliary_assets_required=["FoundationPose", "DINOv2", "DINOv3_torch_hub"])
        if value != expected: raise ValueError("Minimal original principal acquisition receipt schema/provenance differs")
    else:
        if (type(value) is not dict or set(value) != {"scope", "source_revisions", "checkpoints", "foundationpose_acquired"}
                or value["scope"] != "DINO_source_and_checkpoints_only" or value["source_revisions"] != gate.DINO_REVS
                or value["foundationpose_acquired"] is not False or type(value["checkpoints"]) is not list or len(value["checkpoints"]) != 4):
            raise ValueError("Minimal original auxiliary receipt schema differs")
        seen = set()
        for row in value["checkpoints"]:
            if type(row) is not dict or set(row) != {"filename", "url", "sha256", "bytes", "hash_source"}: raise ValueError("Auxiliary receipt unknown fields forbidden")
            filename = row["filename"]
            if filename in seen or filename not in (*RELEASES, *REG4): raise ValueError("Original four auxiliary checkpoint records required")
            seen.add(filename); _pin({k: row[k] for k in ("bytes", "sha256")}, 7_000_000_000)
            model = filename.removeprefix("dinov2_").split("_")[0]
            if row["url"] != "https://dl.fbaipublicfiles.com/dinov2/dinov2_" + model + "/" + filename: raise ValueError("Original auxiliary URL differs")
            if filename in RELEASES:
                if row["sha256"] != RELEASES[filename] or row["hash_source"] != "official_pinned_downloader": raise ValueError("Original released auxiliary hash differs")
            else:
                asset = entries["weights/sam3d/torch_home/hub/checkpoints/" + filename]
                if row["hash_source"] != "first_observed_https_download" or any(row[k] != asset[k] for k in ("bytes", "sha256")):
                    raise ValueError("Original reg4 first-observed receipt binding differs")


def _check_inputs(root, entries, gate):
    states = {}
    for name, row in entries.items():
        path = root / name; _canonical(path.parent); before = _state(path)
        if row["type"] == "symlink":
            if not stat.S_ISLNK(before[2]) or before[6] != 1 or os.readlink(path) != row["link"]: raise ValueError("Original audited link changed")
        else:
            actual = _identity(path, row["bytes"], empty=row["bytes"] == 0)
            if actual != {k: row[k] for k in actual}: raise ValueError("Original audited file bytes/blob changed")
            if name in gate.RECEIPTS[:2]: _receipt(gate, name, _read_pinned(path, {k: row[k] for k in ("bytes", "sha256")}), str(root), entries)
        if _state(path) != before: raise ValueError("Audited prerequisite changed during preflight")
        states[name] = before
    snapshots = _canonical(root / (gate.MOGE_REPO + "/snapshots"))
    if {p.name for p in snapshots.iterdir()} != {gate.MOGE_REV}: raise ValueError("MoGe snapshot selection changed")
    return states


def _info(name, row):
    result = tarfile.TarInfo(name); result.uid = result.gid = result.mtime = 0; result.uname = result.gname = ""
    result.mode = 0o777 if row["type"] == "symlink" else 0o444
    result.type = tarfile.SYMTYPE if row["type"] == "symlink" else tarfile.REGTYPE
    result.size = row.get("bytes", 0); result.linkname = row.get("link", "")
    return result


class _Reader:
    def __init__(self, stream, size):
        self.stream = stream; self.size = size; self.count = 0; self.sha = hashlib.sha256(); self.blob = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    def read(self, size=-1):
        block = self.stream.read(size); self.count += len(block); self.sha.update(block); self.blob.update(block)
        return block
    def finish(self, row):
        if self.read(1) or dict(bytes=self.count, sha256=self.sha.hexdigest(), git_blob_sha1=self.blob.hexdigest()) != {k: row[k] for k in ("bytes", "sha256", "git_blob_sha1")}:
            raise ValueError("Archive streaming file identity differs")


def build_archive(root, inventory_report, inventory_code, inventory_pin, inventory_source_pins, destination, exporter_pin):
    """Return a small receipt; exclusively create destination, unlink owned failure.

    inventory_pin = {bytes,sha256,producer_revision,entries_sha256,files,links,
    total_bytes}. exporter_pin independently pins this module's bytes/SHA.
    The first archive member is the byte-pinned, derived payload manifest.
    """
    _pin(exporter_pin, 1_000_000)
    _read_pinned(Path(__file__), exporter_pin)
    _inventory_pin(inventory_pin)
    root = _canonical(root); gate = _source(inventory_code, inventory_source_pins)
    if Path(inventory_report).stat().st_mode & 0o777 != 0o400: raise ValueError("Frozen inventory receipt must remain0400")
    raw = _read_pinned(inventory_report, {k: inventory_pin[k] for k in ("bytes", "sha256")}); report = _parse(raw)
    if (report.get("stage") != "frontend_replica_prerequisite_inventory" or report.get("status") != "pass"
            or report.get("producer_revision") != inventory_pin["producer_revision"] or report.get("source_helpers") != inventory_source_pins
            or report.get("entries_sha256") != inventory_pin["entries_sha256"] or gate.digest_json(report.get("entries")) != inventory_pin["entries_sha256"]
            or report.get("actual_prerequisites_unchanged") is not True or any(report.get(k) is not False for k in (
                "challenge_data_read", "private_validation_read", "predictions_read", "GPU_used", "models_loaded", "raw_build_receipts_transfer_eligible", "license_eligibility_verified", "training_overlap_verified"))):
        raise ValueError("Independently frozen genuine inventory PASS required")
    evidence = report.get("evidence_only_files")
    if type(evidence) is not dict or set(evidence) != set(gate.RECEIPTS[2:]) or any(r.get("transfer_eligible") is not False or r.get("raw_build_receipt_must_not_be_exported") is not True for r in evidence.values()):
        raise ValueError("Original three raw build receipts must remain evidence only")
    entries = report["entries"]; _entries(gate, entries, report["sources"], report["internal_moge1"])
    counts = dict(files=sum(r["type"] == "file" for r in entries.values()), links=sum(r["type"] == "symlink" for r in entries.values()), total_bytes=sum(r.get("bytes", 0) for r in entries.values()))
    if any(counts[k] != inventory_pin[k] for k in counts): raise ValueError("Independent full audited inventory quantities differ")
    states = _check_inputs(root, entries, gate)
    manifest = dict(schema=SCHEMA, original_root=str(root), inventory=inventory_pin, inventory_source_helpers=inventory_source_pins,
        exporter_source_identity=exporter_pin, entries=entries, sources=report["sources"], internal_moge1=report["internal_moge1"],
        license_eligibility_verified=False, training_overlap_verified=False, replica_ready=False, CUDA_verified=False,
        raw_build_receipts_exported=False, images_exported=False)
    manifest_raw = _json(manifest)
    if len(manifest_raw) > MAX_JSON: raise ValueError("Derived archive manifest exceeds bound")
    destination = _canonical(destination)
    if not destination.parent.is_dir(): raise ValueError("Existing canonical owned archive parent required")
    owned = None
    try:
        # Restricted checkpoints must never inherit a permissive umask while
        # their multi-GB archive is still being streamed or checked.
        with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400), "wb") as output:
            owned = _state(destination)[:2]
            with tarfile.open(fileobj=output, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(_info(MANIFEST, dict(type="file", bytes=len(manifest_raw))), io.BytesIO(manifest_raw))
                for name in sorted(entries):
                    row = entries[name]
                    if _state(root / name) != states[name]: raise ValueError("Archive input changed before stream")
                    if row["type"] == "symlink": archive.addfile(_info(name, row))
                    else:
                        with (root / name).open("rb") as stream:
                            reader = _Reader(stream, row["bytes"]); archive.addfile(_info(name, row), reader); reader.finish(row)
                    if _state(root / name) != states[name]: raise ValueError("Archive input changed during stream")
            output.flush(); os.fsync(output.fileno())
        if _check_inputs(root, entries, gate) != states: raise ValueError("Original prerequisite state changed after archive")
        _source(inventory_code, inventory_source_pins); _read_pinned(inventory_report, {k: inventory_pin[k] for k in ("bytes", "sha256")}); _read_pinned(Path(__file__), exporter_pin)
        destination.chmod(0o400); identity = _identity(destination, MAX_TOTAL + 10_000_000)
        return dict(schema=SCHEMA, stage="frontend_asset_archive", status="pass", inventory=inventory_pin,
            inventory_source_helpers=inventory_source_pins, exporter_source_identity=exporter_pin,
            manifest_identity=dict(bytes=len(manifest_raw), sha256=hashlib.sha256(manifest_raw).hexdigest()),
            archive_identity={k: identity[k] for k in ("bytes", "sha256")}, payload_entries_sha256=gate.digest_json(entries),
            **counts, license_eligibility_verified=False, training_overlap_verified=False, replica_ready=False, CUDA_verified=False,
            raw_build_receipts_exported=False, images_exported=False, transferred=False)
    except BaseException:
        if owned is not None and destination.exists() and not destination.is_symlink() and _state(destination)[:2] == owned: destination.unlink()
        raise


def verify_archive(path, archive_pin, manifest_pin, inventory_code, inventory_source_pins):
    """Verify a caller-pinned archive/first manifest by streaming, never extract."""
    _pin(archive_pin, MAX_TOTAL + 10_000_000); _pin(manifest_pin, MAX_JSON)
    gate = _source(inventory_code, inventory_source_pins)
    before = _state(_canonical(path)); identity = _identity(path, MAX_TOTAL + 10_000_000)
    if {k: identity[k] for k in archive_pin} != archive_pin: raise ValueError("Independent archive identity differs before TAR parsing")
    with tarfile.open(path, mode="r|") as archive, Path(path).open("rb") as sealed:
        first = archive.next()
        if first is None or first.name != MANIFEST or not first.isreg() or first.size != manifest_pin["bytes"]: raise ValueError("Pinned derived manifest must be first regular member")
        manifest_raw = archive.extractfile(first).read()
        if hashlib.sha256(manifest_raw).hexdigest() != manifest_pin["sha256"]: raise ValueError("Independent first manifest identity differs")
        manifest = _parse(manifest_raw)
        keys = {"schema", "original_root", "inventory", "inventory_source_helpers", "exporter_source_identity", "entries", "sources", "internal_moge1", "license_eligibility_verified", "training_overlap_verified", "replica_ready", "CUDA_verified", "raw_build_receipts_exported", "images_exported"}
        if (type(manifest) is not dict or set(manifest) != keys or manifest["schema"] != SCHEMA or manifest["inventory_source_helpers"] != inventory_source_pins
                or any(manifest[k] is not False for k in ("license_eligibility_verified", "training_overlap_verified", "replica_ready", "CUDA_verified", "raw_build_receipts_exported", "images_exported"))):
            raise ValueError("Exact non-eligibility archive manifest required")
        _canonical(manifest["original_root"])
        _pin(manifest["exporter_source_identity"], 1_000_000)
        _inventory_pin(manifest["inventory"])
        entries = manifest["entries"]; _entries(gate, entries, manifest["sources"], manifest["internal_moge1"])
        counts = dict(files=sum(r["type"] == "file" for r in entries.values()), links=sum(r["type"] == "symlink" for r in entries.values()), total_bytes=sum(r.get("bytes", 0) for r in entries.values()))
        if gate.digest_json(entries) != manifest["inventory"]["entries_sha256"] or any(counts[k] != manifest["inventory"][k] for k in counts):
            raise ValueError("Derived inventory entry digest/quantities differ")
        planned = [(MANIFEST, dict(type="file", bytes=len(manifest_raw))), *[(n, entries[n]) for n in sorted(entries)]]
        size = 0
        for index, (name, row) in enumerate(planned):
            member = first if index == 0 else archive.next(); expected = _info(name, row)
            if (member is None or member.name != name or member.type != expected.type or member.size != expected.size or member.linkname != expected.linkname
                    or (member.uid, member.gid, member.mtime, member.uname, member.gname, member.mode) != (0, 0, 0, "", "", expected.mode)
                    or not set(member.pax_headers) <= {"path", "linkpath", "size"}): raise ValueError("Extra/duplicate/unsafe/nondeterministic TAR member")
            header = expected.tobuf(format=tarfile.PAX_FORMAT)
            sealed.seek(size)
            if member.offset != size or member.offset_data != size + len(header) or sealed.read(len(header)) != header:
                raise ValueError("Exact deterministic TAR headers required")
            padding = (-expected.size) % 512
            sealed.seek(size + len(header) + expected.size)
            if sealed.read(padding) != b"\0" * padding: raise ValueError("Nonzero TAR member padding forbidden")
            size += len(header) + (expected.size + 511) // 512 * 512
            if index and row["type"] == "file":
                stream = archive.extractfile(member); reader = _Reader(stream, row["bytes"])
                if name in gate.RECEIPTS[:2]:
                    raw = reader.read(); _receipt(gate, name, raw, manifest["original_root"], entries)
                else:
                    while reader.read(4 * 1024 * 1024): pass
                reader.finish(row)
        if archive.next() is not None: raise ValueError("Unlisted archive member forbidden")
        expected_size = ((size + 1024 + tarfile.RECORDSIZE - 1) // tarfile.RECORDSIZE) * tarfile.RECORDSIZE
        sealed.seek(size)
        if expected_size != archive_pin["bytes"] or sealed.read(tarfile.RECORDSIZE + 1024) != b"\0" * (expected_size - size):
            raise ValueError("Unexpected/nonzero archive trailer or padding")
    after_identity = _identity(path, MAX_TOTAL + 10_000_000)
    if _state(path) != before or {k: after_identity[k] for k in archive_pin} != archive_pin:
        raise ValueError("Pinned archive changed during verification")
    _source(inventory_code, inventory_source_pins)
    return dict(schema=SCHEMA, stage="frontend_asset_archive_verify", status="pass", archive_identity=archive_pin,
        manifest_identity=manifest_pin, payload_entries_sha256=gate.digest_json(entries), extraction_performed=False,
        license_eligibility_verified=False, training_overlap_verified=False, replica_ready=False, CUDA_verified=False)
