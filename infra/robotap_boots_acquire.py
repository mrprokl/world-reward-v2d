"""Azure VM02-only RoboTAP/Boots acquisition. Never unpickle, decode, or infer.

Acquisition PASS attests pinned bytes and safe private retention, not benchmark
readiness, unseen training data, automatic Track1 queries, or reconstruction.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path("/srv/scenesmith/world-reward")
JOB = "run_robotap_boots_acquire"
NAMESPACE = "validation/robotap_boots_v1"
STAGE = "external_robotap_bootstapir_opaque_acquisition"
HELPERS = ("infra/robotap_boots_acquire.py", "infra/run_robotap_boots_acquire.sh", "configs/robotap_boots_protocol.json")
PROTOCOL_BYTES = 6339
PROTOCOL_SHA256 = "830cbf41b88ab7a2c856884172f226eb4cd7773b169d647af33b729d0d7ac1cf"
BUDGET = 1200
CHUNK = 1024 * 1024
FIELDS = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Canonical absolute nonsymlink path required")
    return path


def identity(path, *, readonly=True):
    path = canonical(path); before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or (readonly and before.st_mode & 0o222):
        raise ValueError("Nonempty readonly regular file required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""): digest.update(block)
    after = path.lstat()
    if FIELDS(before) != FIELDS(after): raise ValueError("Original file changed while hashing")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def bound_source(root, code, revision, executing):
    root, code, executing = map(canonical, (root, code, executing))
    if (root != ROOT or not root.is_dir() or type(revision) is not str
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or code != root / "jobs" / revision / JOB / "code" or not code.is_dir()
            or executing != code / HELPERS[0]):
        raise ValueError("Actual immutable VM02 acquisition source namespace required")
    helpers = {name: identity(code / name) for name in HELPERS}
    markers = {name: identity(code.parent / name, readonly=False) for name in ("revision", "source-sha256")}
    if ((code.parent / "revision").read_bytes() != (revision + "\n").encode()
            or re.fullmatch(b"[0-9a-f]{64}\n", (code.parent / "source-sha256").read_bytes()) is None):
        raise ValueError("Original revision/archive dispatch markers required")
    for name in markers:
        if identity(code.parent / name, readonly=False) != markers[name]: raise ValueError("Dispatch marker changed")
    return {"helpers": helpers, "markers": markers}


def read_protocol(path):
    # Hash the ENTIRE preregistration before JSON parsing, network, or extraction.
    if identity(path) != {"sha256": PROTOCOL_SHA256, "bytes": PROTOCOL_BYTES}:
        raise ValueError("Whole frozen RoboTAP/Boots protocol identity differs")
    data = canonical(path).read_bytes()
    if len(data) != PROTOCOL_BYTES or hashlib.sha256(data).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Protocol changed before JSON parsing")
    return json.loads(data)


def validate_https(url):
    parsed = urllib.parse.urlsplit(url); host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.fragment
            or parsed.port not in (None, 443) or not (host in ("storage.googleapis.com", "raw.githubusercontent.com", "huggingface.co")
                or host.endswith(".hf.co"))):
        raise ValueError("Public allowlisted HTTPS source required")


def check_deadline(deadline):
    if time.monotonic() >= deadline: raise TimeoutError("Fixed RoboTAP acquisition deadline exceeded")


def fetch_text(pin, deadline):
    validate_https(pin["url"]); check_deadline(deadline)
    with urllib.request.urlopen(urllib.request.Request(pin["url"], headers={"Accept-Encoding": "identity"}), timeout=30) as response:
        validate_https(response.geturl()); data = response.read(pin["bytes"] + 1)
    check_deadline(deadline)
    if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("Pinned primary source/license bytes differ")
    return data


def private_writer(path):
    """Exclusive regular FD0400 before its first byte, regardless of caller umask."""
    path = canonical(path); missing = []; parent = path.parent
    while not parent.exists(): missing.append(parent); parent = parent.parent
    canonical(parent)
    for directory in reversed(missing): directory.mkdir(mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    os.fchmod(fd, 0o400)
    return os.fdopen(fd, "wb")


def download(pin, destination, deadline):
    """One streaming request, exact pin checks, exclusive path; no retry/resume."""
    validate_https(pin["url"]); check_deadline(deadline)
    count = 0; sha = hashlib.sha256(); md5 = hashlib.md5()
    with urllib.request.urlopen(urllib.request.Request(pin["url"], headers={"Accept-Encoding": "identity"}), timeout=30) as response:
        validate_https(response.geturl()); headers = response.headers
        if response.status != 200: raise ValueError("Complete original HTTP200 response required")
        if headers.get("Content-Encoding", "identity") != "identity": raise ValueError("Original unencoded download required")
        size_header = headers.get("Content-Length")
        if size_header is not None and size_header != str(pin["bytes"]): raise ValueError("Pinned response byte count differs")
        if "generation" in pin:
            expected = {"x-goog-generation": pin["generation"], "x-goog-metageneration": pin["metageneration"],
                "x-goog-stored-content-length": str(pin["bytes"]), "Content-Type": pin["content_type"]}
            if any(headers.get(key) != value for key, value in expected.items()):
                raise ValueError("Original pinned GCS generation/size/type differs")
            if headers.get("ETag") != '"' + pin["md5_hex"] + '"': raise ValueError("Published GCS MD5 differs")
        with private_writer(destination) as stream:
            while True:
                check_deadline(deadline); block = response.read(min(CHUNK, pin["bytes"] - count + 1))
                if not block: break
                count += len(block)
                if count > pin["bytes"]: raise ValueError("Download exceeds pinned bytes")
                sha.update(block); md5.update(block); stream.write(block)
            stream.flush(); os.fsync(stream.fileno())
    check_deadline(deadline)
    result = {"bytes": count, "sha256": sha.hexdigest()}
    if count != pin["bytes"] or (pin.get("sha256") is not None and sha.hexdigest() != pin["sha256"]):
        raise ValueError("Pinned download bytes/SHA256 differ")
    if "md5_hex" in pin:
        if (md5.hexdigest() != pin["md5_hex"]
                or base64.b64encode(md5.digest()).decode() != pin["md5_base64"]):
            raise ValueError("Published archive MD5 mismatch")
        result.update(md5_hex=md5.hexdigest(), generation=pin["generation"], publisher_md5_verified=True,
            sha256_kind="observed_stream_digest_not_independently_published_pin")
    Path(destination).chmod(0o400)
    return result


def save_bytes(path, data, mode=0o400):
    path = Path(path); canonical(path)
    with private_writer(path) as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    path.chmod(mode)


def zip_inventory(archive, protocol):
    """Only central-directory names/types/counts; no member values opened."""
    members = {}; normalized = set(); total = 0; total_text = 0
    infos = archive.infolist()
    if not infos or len(infos) > protocol["maximum_archive_members"]: raise ValueError("Archive member count outside fixed bound")
    for info in infos:
        name = info.filename; path = PurePosixPath(name); key = name.rstrip("/")
        kind = stat.S_IFMT(info.external_attr >> 16)
        if (not name or not key or "\\" in name or "\x00" in name or path.is_absolute()
                or any(p in ("", ".", "..") for p in key.split("/")) or str(path) != key
                or key in normalized or info.flag_bits & 1
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                or kind not in ((0, stat.S_IFDIR) if info.is_dir() else (0, stat.S_IFREG))
                or info.file_size < 0 or info.compress_size < 0
                or (info.is_dir() and info.file_size != 0)):
            raise ValueError("Unsafe, duplicate or unsupported original ZIP member")
        normalized.add(key); total += info.file_size
        if total > protocol["maximum_expanded_bytes"]: raise ValueError("Fixed ZIP expansion budget exceeded")
        if not info.is_dir():
            if text_member(name):
                total_text += info.file_size
                if (info.file_size > protocol["maximum_embedded_text_bytes"]
                        or total_text > protocol["maximum_total_embedded_text_bytes"]):
                    raise ValueError("Embedded text outside frozen byte bound")
            members[name] = info
    for name in members:
        if any(str(parent) in members for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError("ZIP regular-file parent collision")
    if not members: raise ValueError("Empty original ZIP")
    return members, total


def text_member(name):
    leaf = PurePosixPath(name).name.lower()
    return bool(re.fullmatch(r"(?:readme|license|licence|notice|attribution|authors|copyright)(?:[-_.a-z0-9]*)", leaf))


def freeze_selection(members):
    names = sorted(members)
    pickles = [name for name in names if PurePosixPath(name).suffix.lower() in (".pkl", ".pickle")]
    if not pickles or any(name not in pickles and not text_member(name) for name in names):
        raise ValueError("Only original opaque pickles and embedded attribution/license text supported")
    selected = pickles[:3] if len(pickles) >= 3 else []
    return {"pickle_member_names": pickles, "first_three_pickle_files_frozen": selected,
        "file_selection_frozen_before_private_member_values": bool(selected),
        "selection_pending": True, "video_selection_pending": True,
        "video_identity_from_pickle_filename_verified": False, "frame_cardinalities_verified": False,
        "benchmark_ready": False, "single_or_grouped_pickle_requires_later_separate_frozen_reader": True}


def check_embedded_text(name, data):
    try: text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error: raise ValueError("Embedded license/attribution text must be auditable UTF8") from error
    low = text.lower()
    if re.search(r"by-nc|by-sa|non[ -]?commercial|research[ -]+(?:use|only)|all rights reserved", low):
        raise ValueError("Embedded RoboTAP terms conflict with pinned CC-BY4")
    # Only an explicit dataset declaration needs full concordant terms. Generic
    # citations and absent embedded licenses are not invented license evidence.
    if re.search(r"licen[cs]e|creative commons|copyright", low):
        ccby = bool(re.search(r"cc[ -]?by[ -]?4(?:\.0)?|creativecommons\.org/licenses/by/4\.0|creative commons.*(?:attribution|by).*4\.0", low))
        apache = "apache" in low and ("2.0" in low or "version 2" in low)
        if not ccby or apache: raise ValueError("Embedded explicit dataset license is incomplete or contradictory")
    return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def extract_members(archive, members, private, protocol, deadline):
    """Retain original bytes streaming; CRC is checked by ZipExtFile at EOF."""
    retained = []; text_bytes = 0
    for name in sorted(members):
        check_deadline(deadline); info = members[name]; path = private / "pickles" / name
        canonical(path)
        digest = hashlib.sha256(); count = 0
        with archive.open(info) as source, private_writer(path) as target:
            while block := source.read(CHUNK):
                check_deadline(deadline); count += len(block)
                if count > info.file_size: raise ValueError("Expanded member exceeds original declared size")
                digest.update(block); target.write(block)
            target.flush(); os.fsync(target.fileno())
        if count != info.file_size or count == 0: raise ValueError("Nonempty exact original member size required")
        path.chmod(0o400)
        record = {"file": str(path.relative_to(private.parent)), "bytes": count, "sha256": digest.hexdigest(),
            "archive_member": name, "zip_crc32": f"{info.CRC:08x}"}
        if text_member(name):
            text_bytes += count
            if count > protocol["maximum_embedded_text_bytes"] or text_bytes > protocol["maximum_total_embedded_text_bytes"]:
                raise ValueError("Embedded text exceeds fixed audit budget")
            check_embedded_text(name, path.read_bytes())
        retained.append(record)
    return retained


def primary_sources(protocol, assets, deadline):
    records = {}; readme = None
    for name, pin in protocol["source"]["files"].items():
        data = fetch_text(pin, deadline)
        if name == "README.md": readme = data.decode("utf-8")
        save_bytes(assets / "tapnet_source" / name, data, 0o444)
        records[name] = {"bytes": pin["bytes"], "sha256": pin["sha256"]}
    if (readme is None or "as well as the RGB-Stacking videos and RoboTAP videos" not in readme
            or "https://creativecommons.org/licenses/by/4.0/legalcode" not in readme
            or "All pre-trained model checkpoints released in this repository" not in readme
            or "are also licensed under Apache 2.0" not in readme):
        raise ValueError("Pinned primary dataset/checkpoint license scope not explicit")
    save_bytes(assets / "dataset-attribution.json", (json.dumps(protocol["attribution"], indent=2) + "\n").encode(), 0o444)
    return records


def verify_retained(destination, expected):
    """Exact original retained inventory, no extra files/symlinks or private export."""
    actual = {}; directories = set()
    for path in destination.rglob("*"):
        canonical(path); mode = path.lstat().st_mode
        relative = str(path.relative_to(destination))
        if stat.S_ISDIR(mode):
            if mode & 0o777 != 0o700: raise ValueError("Every acquisition subdirectory must be0700")
            directories.add(relative)
        elif stat.S_ISREG(mode):
            if relative == "report.json" or relative.startswith("eval_private/.downloads/"): continue
            required = 0o400 if relative.startswith("eval_private/") else 0o444
            if mode & 0o777 != required: raise ValueError("Exact public asset/private pickle file permissions required")
            actual[relative] = identity(path)
        else: raise ValueError("Only original regular files/directories allowed")
    if actual != expected: raise ValueError("Complete retained file inventory differs")
    roots = {p.name for p in destination.iterdir()}
    if roots != {"assets", "eval_private", "report.json"}: raise ValueError("Exact acquisition output roots required")
    return actual


def azure_vm02_identity(protocol):
    """Small instance metadata only, never credentials/tokens or compute models."""
    url = "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(url, headers={"Metadata": "true"}), timeout=5) as response:
        data = response.read(65537)
    if len(data) > 65536: raise ValueError("Azure metadata byte bound exceeded")
    value = json.loads(data)
    if (value.get("name") != protocol["azure_vm_name"]
            or str(value.get("resourceGroupName", "")).lower() != protocol["azure_resource_group"].lower()):
        raise ValueError("Actual owned Azure VM02 identity required")
    return {"name": value["name"], "resource_group": value["resourceGroupName"], "azure_instance_metadata_verified": True}


def acquire(destination, protocol, report, persist, deadline):
    assets, private = destination / "assets", destination / "eval_private"
    downloads = private / ".downloads"; downloads.mkdir(mode=0o700)
    report.update(phase="primary_sources"); persist()
    report["primary_sources"] = primary_sources(protocol, assets, deadline); persist()
    report.update(phase="checkpoint"); persist()
    checkpoint = protocol["checkpoint"]
    report["checkpoint"] = download(checkpoint, assets / checkpoint["file"], deadline)
    (assets / checkpoint["file"]).chmod(0o444); persist()
    report.update(phase="archive_download"); persist()
    pin = protocol["dataset"]["archive"]; archive_path = downloads / pin["name"]
    report["archive"] = download(pin, archive_path, deadline)
    archive_state = FIELDS(archive_path.lstat()); persist()
    with zipfile.ZipFile(archive_path) as archive:
        members, expanded = zip_inventory(archive, protocol)
        selection = freeze_selection(members)
        inventory = [{"file": name, "bytes": info.file_size, "compressed_bytes": info.compress_size,
            "zip_crc32": f"{info.CRC:08x}"} for name, info in sorted(members.items())]
        save_bytes(private / "zip-member-inventory.json", (json.dumps(inventory, indent=2) + "\n").encode())
        report.update(phase="retain_original_opaque_members", expanded_bytes=expanded,
            original_member_count=len(members), filename_selection=selection,
            selection_frozen_before_first_private_member_value=True); persist()
        report["retained_members"] = extract_members(archive, members, private, protocol, deadline); persist()
    expected = {"assets/tapnet_source/" + name: record for name, record in report["primary_sources"].items()}
    expected["assets/" + checkpoint["file"]] = {k: report["checkpoint"][k] for k in ("sha256", "bytes")}
    expected["assets/dataset-attribution.json"] = identity(assets / "dataset-attribution.json")
    expected["eval_private/zip-member-inventory.json"] = identity(private / "zip-member-inventory.json")
    expected.update({row["file"]: {k: row[k] for k in ("sha256", "bytes")} for row in report["retained_members"]})
    # Successful immutable retained receipt must exist BEFORE archive deletion.
    report["retained_files"] = verify_retained(destination, expected)
    receipt = private / "retention-receipt.json"
    save_bytes(receipt, (json.dumps({"schema": "world-reward-robotap-opaque-retention-v1",
        "archive": report["archive"], "retained_files": expected, "filename_selection": selection,
        "no_unpickle_or_decode": True}, indent=2) + "\n").encode())
    expected["eval_private/retention-receipt.json"] = identity(receipt)
    report["retained_files"] = verify_retained(destination, expected)
    check_deadline(deadline)
    if (FIELDS(archive_path.lstat()) != archive_state
            or identity(archive_path) != {k: report["archive"][k] for k in ("sha256", "bytes")}
            or FIELDS(archive_path.lstat()) != archive_state):
        raise ValueError("Original archive inode/state/bytes changed after extraction")
    archive_path.unlink(); downloads.rmdir()
    report.update(disposable_archive_removed_after_successful_retention_receipt=True,
        original_pickles_unmodified=True, pickle_or_rgb_or_gt_decoded=False, benchmark_ready=False)
    report["retained_files"] = verify_retained(destination, expected)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("All external data/checkpoints stay on Azure Linux")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    source_before = bound_source(root, code, revision, Path(__file__))
    protocol = read_protocol(code / HELPERS[2])
    destination = canonical(root / NAMESPACE)
    if (os.environ.get("WR_ROBOTAP_OUTPUT_RESERVED") != "1" or not destination.is_dir()
            or any(destination.iterdir()) or destination.stat().st_mode & 0o777 != 0o700):
        raise ValueError("Exclusively reserved fresh0700 acquisition output required")
    oldmask = os.umask(0o077)
    for name in ("assets", "eval_private"): (destination / name).mkdir(mode=0o700)
    started = time.monotonic(); deadline = started + BUDGET
    report = {"stage": STAGE, "status": "fail", "phase": "runtime_preflight", "producer_revision": revision,
        "source_before": source_before, "script_sha256": source_before["helpers"][HELPERS[0]]["sha256"],
        "protocol_identity": source_before["helpers"][HELPERS[2]], "budget_seconds": BUDGET,
        "dataset_license": "CC-BY-4.0", "model_and_code_license": "Apache-2.0", "device": "cpu", "gpu_used": False,
        "inference_performed": False, "evaluation_performed": False, "challenge_inputs_used": False,
        "training_overlap_verified": False, "challenge_overlap_verified": False, "benchmark_ready": False,
        "full_hoi_accuracy_verified": False, "cari4d_victory_verified": False,
        "oracle_initial_query_is_track1_automatic_readiness": False, "pickle_or_rgb_or_gt_decoded": False}
    path = destination / "report.json"; error = None
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.monotonic() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate()
            stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Fixed RoboTAP acquisition exceeded1200s")
        previous_alarm = signal.signal(signal.SIGALRM, expired); previous_term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(BUDGET)
        try:
            persist(); report["azure_identity"] = azure_vm02_identity(protocol)
            free = shutil.disk_usage(destination).free; report["initial_free_disk_bytes"] = free
            if free < protocol["minimum_free_disk_bytes"]: raise ValueError("Fixed minimum64GiB free disk required")
            persist(); acquire(destination, protocol, report, persist, deadline)
        except Exception as caught:
            error = caught
            # Do not include URL-bearing network exceptions or signed redirect URLs.
            report.update(error_type=type(caught).__name__, error=(str(caught) if type(caught) in (ValueError, TimeoutError)
                else "Acquisition failed; immutable phase and byte receipts retained"))
        finally:
            try:
                source_after = bound_source(root, code, revision, Path(__file__))
                report["source_after"] = source_after
                if source_after != source_before: raise ValueError("Frozen source/dispatch markers changed")
                read_protocol(code / HELPERS[2]); report["source_and_protocol_after_reverified"] = True
            except Exception as caught:
                report["source_and_protocol_after_reverified"] = False
                if error is None: error = caught; report["error_type"] = type(caught).__name__
            if error is None: report.update(status="pass", phase="complete")
            persist(); path.chmod(0o400)
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term)
            os.umask(oldmask)
    if error is not None: raise RuntimeError("RoboTAP acquisition failed; inspect sealed tiny receipt") from None


if __name__ == "__main__": main()
