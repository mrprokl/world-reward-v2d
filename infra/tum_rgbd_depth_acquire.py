"""Azure stdlib-only TUM subset acquisition; no image/depth values decoded.

Original archives have no independently known SHA256. Their observed hashes are
reproducibility evidence; selected bytes must match independent frozen LFS pins.
Public RGB exposure is last, after every filename, license and byte gate passes.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import struct
import tarfile
import time
from urllib.request import Request, urlopen

ROOT = Path("/srv/scenesmith/world-reward")
BASE = "validation/tum_rgbd_depth_holdout_v1"
PROTOCOL = "configs/tum_rgbd_depth_protocol.json"
PROTOCOL_ID = dict(bytes=21939, sha256="ed1f038546ac073d2b52f01874934a6683e05357f9730cbac1a5112930ded035")
SOURCE_FILES = ("infra/tum_rgbd_depth_acquire.py", "infra/run_tum_rgbd_depth_acquire.sh")
SCHEMA = "world_reward.tum_rgbd_depth_public.v1"
SELECTION = "sorted_RGB_indices_40_80_120_160_per_sequence_before_depth_values"
STAGE = "external_tum_rgbd_depth_holdout_acquisition"
BUDGET, MAX_MEMBERS, MAX_MEMBER_BYTES = 600, 30000, 4 * 1024 * 1024
DESCRIPTION_IDS = {
    "rgbd_dataset_freiburg1_desk": dict(bytes=239, sha256="7ef42a6bf23b54ed20538a9047eedb4d91201cff3dd36612ae272ad1ae2c341d"),
    "rgbd_dataset_freiburg2_xyz": dict(bytes=341, sha256="ba032acbaa6681507ec926be893ae20a1801c4b0e7e9be646f2a80ff43a96eb5"),
    "rgbd_dataset_freiburg3_long_office_household": dict(bytes=320, sha256="41aa30f7b484482ab2d1f55ed2a54663633adf8f866ebe8436311a822bcb1ad8")}


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Canonical absolute nonsymlink path required")
    return path


def identity(path):
    path = canonical(path); before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size <= 0:
        raise ValueError("Nonempty regular file without aliases required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Original bytes changed while hashing")
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def bytes_identity(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save(path, raw, mode=0o400):
    with Path(path).open("xb") as stream: stream.write(raw)
    Path(path).chmod(mode)


def bound_source(root, code, revision, executing):
    root, code = canonical(root), canonical(code)
    if (root != ROOT or re.fullmatch(r"[0-9a-f]{40}", revision) is None or not root.is_dir()
            or code != root / "jobs" / revision / "run_tum_rgbd_depth_acquire/code"
            or not code.is_dir() or Path(executing) != code / SOURCE_FILES[0]):
        raise ValueError("Exact immutable remote acquisition producer required")
    if {str(p.relative_to(code)) for p in (code / "infra").rglob("*") if p.is_file()} != set(SOURCE_FILES):
        raise ValueError("Only two stdlib acquisition source helpers permitted")
    helpers = {name: identity(code / name) for name in SOURCE_FILES}
    if any((code / name).stat().st_mode & 0o222 for name in (*SOURCE_FILES, PROTOCOL)):
        raise ValueError("Original source and protocol must be readonly")
    before = identity(code / PROTOCOL)
    if before != PROTOCOL_ID: raise ValueError("Entire frozen TUM protocol changed")
    frozen = json.loads((code / PROTOCOL).read_bytes())
    if identity(code / PROTOCOL) != before: raise ValueError("Protocol changed during interpretation")
    return helpers, before, frozen


def text_request(url, limit):
    if not url.startswith("https://cvg.cit.tum.de/"): raise ValueError("Only original TUM publisher text permitted")
    with urlopen(Request(url, headers={"User-Agent": "WorldReward-TUM-acquisition/1.0"}), timeout=30) as response:
        if response.status != 200 or response.geturl() != url: raise ValueError("Original publisher response/redirect changed")
        raw = response.read(limit + 1)
    if len(raw) > limit: raise ValueError("Publisher text exceeds fixed byte bound")
    return raw


def license_section(raw, expected):
    text = raw.decode("utf-8"); start = re.search(r'<h[1-6][^>]*id="license"[^>]*>.*?</h[1-6]>', text, re.S)
    end = re.search(r'<h[1-6][^>]*>', text[start.end():]) if start else None
    if not start or not end: raise ValueError("Unambiguous publisher license section required")
    section = text[start.start():start.end() + end.start()].encode()
    if bytes_identity(section) != {k: expected[k] for k in ("bytes", "sha256")}:
        raise ValueError("Primary CC-BY-4.0 license section drift; no waiver")
    return section


def sequence_description(raw, sequence):
    name = sequence["name"].removeprefix("rgbd_dataset_"); text = raw.decode("utf-8")
    match = re.search(r"<a name=['\"]" + re.escape(name) + r"['\"]>Sequence.*?</a></b><br>\s*<i>(.*?)</i>", text, re.S)
    if not match or bytes_identity(match[0].encode()) != DESCRIPTION_IDS[sequence["name"]]:
        raise ValueError("Frozen publisher sequence description changed; review terms before acquisition")
    if sequence["archive"]["url"].removeprefix("https://cvg.cit.tum.de") not in text[match.end():match.end() + 1000]:
        raise ValueError("Selected original archive is not linked by publisher sequence")
    return match[0].encode()


def license_evidence(private, frozen):
    source = private / "source"; source.mkdir(mode=0o700)
    section = license_section(text_request(frozen["primary"]["dataset_url"], 350000), frozen["primary"]["license_section"])
    save(source / "TUM-license-section.html", section)
    descriptions = text_request(frozen["primary"]["download_url"], 750000)
    records = [{"file": "source/TUM-license-section.html", **bytes_identity(section)}]
    for sequence in frozen["sequences"]:
        raw = sequence_description(descriptions, sequence); name = sequence["name"] + "-description.html"
        save(source / name, raw); records.append({"file": "source/" + name, **bytes_identity(raw)})
    raw = (json.dumps(dict(dataset="TUM RGB-D", license="CC-BY-4.0", license_url=frozen["primary"]["license_section"]["license_url"],
        attribution=frozen["primary"]["attribution"], modification="Filename-only selected original RGB/depth subset; PNG bytes unmodified",
        publisher_url=frozen["primary"]["dataset_url"], sequence_specific_terms_checked=True), indent=2) + "\n").encode()
    save(source / "attribution.json", raw); records.append({"file": "source/attribution.json", **bytes_identity(raw)})
    return records


def download_archive(record, destination):
    url = record["url"]
    if not re.fullmatch(r"https://cvg\.cit\.tum\.de/rgbd/dataset/freiburg[123]/rgbd_dataset_freiburg[123]_[a-z_]+\.tgz", url):
        raise ValueError("Original selected publisher archive URL required")
    digest = hashlib.sha256(); count = 0
    with urlopen(Request(url, headers={"User-Agent": "WorldReward-TUM-acquisition/1.0"}), timeout=30) as response:
        if (response.status != 200 or response.geturl() != url or response.headers.get("Content-Length") != str(record["bytes"])
                or response.headers.get("ETag") != record["etag"] or response.headers.get("Last-Modified") != record["last_modified"]):
            raise ValueError("Original GET headers differ from predeclared primary provenance")
        with destination.open("xb") as stream:
            while True:
                chunk = response.read(min(4 * 1024 * 1024, record["bytes"] - count + 1))
                if not chunk: break
                count += len(chunk)
                if count > record["bytes"]: raise ValueError("Archive exceeds declared compressed byte bound")
                stream.write(chunk); digest.update(chunk)
    if count != record["bytes"]: raise ValueError("Original archive truncated")
    destination.chmod(0o400)
    observed = dict(bytes=count, sha256=digest.hexdigest())
    if identity(destination) != observed: raise ValueError("Archive bytes changed after original download")
    return observed


def member_name(member, sequence_name):
    name = member.name
    if name.startswith("./"): name = name[2:]
    if member.isdir() and name.endswith("/"): name = name[:-1]
    path = PurePosixPath(name)
    if (not name or name.startswith("/") or "\\" in name or str(path) != name
            or any(p in (".", "..") for p in path.parts) or path.parts[0] != sequence_name
            or not (member.isfile() or member.isdir()) or member.size < 0 or member.size > MAX_MEMBER_BYTES):
        raise ValueError("Unsafe or aliased/nonregular archive member; no extraction")
    return name


def select_pairs(names, sequence):
    inventories = {kind: sorted(name for name in names if re.fullmatch(
        re.escape(sequence["name"] + "/" + kind + "/") + r"[0-9]+\.[0-9]{6}\.png", name)) for kind in ("rgb", "depth")}
    if any(len(inventories[k]) != sequence[k + "_file_count"] for k in inventories):
        raise ValueError("Original RGB/depth filename counts differ from pinned complete mirror inventory")
    selected = []; used = set()
    for declared in sequence["selected"]:
        rgb = inventories["rgb"][declared["rgb_zero_based_rank"]]; timestamp = Decimal(PurePosixPath(rgb).stem)
        ordered = sorted(inventories["depth"], key=lambda name: (abs(Decimal(PurePosixPath(name).stem) - timestamp), name))
        depth = ordered[0]; offset = abs(Decimal(PurePosixPath(depth).stem) - timestamp)
        if (offset > Decimal(".020000") or depth in used or len(ordered) > 1
                and abs(Decimal(PurePosixPath(ordered[1]).stem) - timestamp) == offset
                or rgb != sequence["name"] + "/" + declared["rgb"]["path"]
                or depth != sequence["name"] + "/" + declared["depth"]["path"]
                or offset != Decimal(declared["timestamp_offset_seconds"])):
            raise ValueError("Frozen unique nearest timestamp association/ranks changed; no replacement")
        selected.append((rgb, depth)); used.add(depth)
    return selected, {kind: dict(files=len(values), sha256=hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest())
        for kind, values in inventories.items()}


def png_header(raw, kind):
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[8:16] != b"\0\0\0\rIHDR":
        raise ValueError("Original PNG IHDR metadata required; no image decoding")
    values = struct.unpack(">IIBBBBB", raw[16:29])
    if values != (640, 480, 8 if kind == "rgb" else 16, 2 if kind == "rgb" else 0, 0, 0, 0):
        raise ValueError("Original640x480 RGB8/depth16gray metadata required; no conversion")


def archive_terms(raw):
    text = raw.decode("utf-8").lower()
    if re.search(r"non.?commercial|research.only|cc.by.nc|cc.by.sa|all rights reserved", text):
        raise ValueError("Contrary archive terms; no inferred exception")
    if not re.search(r"cc.by.4\.0|creativecommons\.org/licenses/by/4\.0|bsd.2.clause", text):
        raise ValueError("Unknown archive license terms require independent review")


def output_name(scene, kind, timestamp): return f"scene_{scene:06d}_{kind}_{timestamp}.png"


def scan_archive(path, sequence, staging):
    expected = {sequence["name"] + "/" + row[kind]["path"]: (kind, row) for row in sequence["selected"] for kind in ("rgb", "depth")}
    seen, retained, terms, expanded = set(), [], [], 0
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = member_name(member, sequence["name"])
            if name in seen or len(seen) >= MAX_MEMBERS: raise ValueError("Duplicate/excessive archive member inventory")
            seen.add(name); expanded += member.size
            if expanded > sequence["archive"]["bytes"] * 6 + 100 * 1024 * 1024:
                raise ValueError("Archive expanded byte bound exceeded")
            if name in expected:
                kind, row = expected[name]
                if member.size != row[kind]["bytes"]: raise ValueError("Selected original member byte count differs from independent pin")
                raw = archive.extractfile(member).read(member.size + 1)
                if bytes_identity(raw) != {k: row[kind][k] for k in ("bytes", "sha256")}:
                    raise ValueError("Selected original member SHA256 differs from independent LFS pin")
                png_header(raw, kind); filename = output_name(sequence["sequence_id"], kind, row[kind + "_timestamp"])
                save(staging / filename, raw)
                retained.append(dict(file=filename, kind=kind, original_file=name, scene_id=sequence["sequence_id"],
                    frame_id=row["rgb_zero_based_rank"], timestamp=row[kind + "_timestamp"], **bytes_identity(raw)))
            elif member.isfile() and re.fullmatch(r"(?i)(license|copying|terms)(\.(txt|md))?", PurePosixPath(name).name):
                if member.size > 65536 or len(terms) >= 16: raise ValueError("Archive term text bound exceeded")
                raw = archive.extractfile(member).read(member.size + 1); archive_terms(raw)
                terms.append(dict(original_file=name, raw=raw, **bytes_identity(raw)))
    pairs, inventory = select_pairs(seen, sequence)
    if len(retained) != 8 or {row["original_file"] for row in retained} != set(expected):
        raise ValueError("All eight selected original files required before publication")
    return retained, dict(members=len(seen), expanded_bytes=expanded, filename_inventories=inventory,
        member_inventory_sha256=hashlib.sha256(("\n".join(sorted(seen)) + "\n").encode()).hexdigest(),
        nearest_timestamp_pairs_verified=True, archive_license_files_present=bool(terms)), terms


def output_inventory(destination, frozen):
    public, private = destination / "inputs", destination / "eval_private"
    names = {output_name(s["sequence_id"], kind, r[kind + "_timestamp"]) for s in frozen["sequences"] for r in s["selected"] for kind in ("rgb", "depth")}
    if {p.name for p in public.iterdir()} != {"manifest.json", *[name for name in names if "_rgb_" in name]}:
        raise ValueError("Exactly twelve public RGB files and one manifest required")
    if {p.name for p in private.iterdir()} != {"source", "acquisition-report.json", *[name for name in names if "_depth_" in name]}:
        raise ValueError("Exactly twelve private depths, source terms and receipt required")
    for directory, mode in ((public, 0o755), (private, 0o700), (private / "source", 0o700)):
        if canonical(directory).stat().st_mode & 0o777 != mode: raise ValueError("Original public/private directory permissions changed")
    if {p.name for p in destination.iterdir()} != {"inputs", "eval_private"}:
        raise ValueError("Only the original public/private output split required")
    files = {}; expected = {}
    for sequence in frozen["sequences"]:
        for row in sequence["selected"]:
            for kind in ("rgb", "depth"):
                folder = "inputs" if kind == "rgb" else "eval_private"
                expected[folder + "/" + output_name(sequence["sequence_id"], kind, row[kind + "_timestamp"])] = {
                    key: row[kind][key] for key in ("bytes", "sha256")}
    for directory, mode in ((public, 0o444), (private, 0o400)):
        for path in directory.rglob("*"):
            canonical(path)
            if path.is_dir():
                if path != private / "source": raise ValueError("Unexpected nested output directory")
                continue
            if path == private / "acquisition-report.json": continue
            if canonical(path).stat().st_mode & 0o777 != mode: raise ValueError("Original public/private file permissions changed")
            files[str(path.relative_to(destination))] = identity(path)
    if any(files.get(name) != value for name, value in expected.items()):
        raise ValueError("All original selected24 output byte pins must remain exact")
    return files


def acquire(destination, frozen, report, persist):
    public, private = destination / "inputs", destination / "eval_private"
    report["license_evidence"] = license_evidence(private, frozen); persist()
    staging = private / ".staging"; staging.mkdir(mode=0o700); retained = []
    try:
        for sequence in frozen["sequences"]:
            report.update(phase="original_download", active_sequence=sequence["name"]); persist()
            path = staging / (sequence["name"] + ".tgz")
            observed = download_archive(sequence["archive"], path)
            row = dict(sequence_id=sequence["sequence_id"], url=sequence["archive"]["url"], **observed,
                sha256_independently_preknown=False, sha256_is_first_observed_reproducibility_digest=True)
            report["archives"].append(row); report.update(phase="filename_and_selected_byte_gates"); persist()
            records, checks, terms = scan_archive(path, sequence, staging); row.update(checks)
            if identity(path) != observed: raise ValueError("Original archive changed during streaming scan")
            path.unlink(); retained.extend(records)
            row["term_files"] = []
            for index, term in enumerate(terms):
                raw = term.pop("raw"); name = f"sequence_{sequence['sequence_id']:06d}_archive_term_{index:02d}.txt"
                save(private / "source" / name, raw); row["term_files"].append({**term, "file": "source/" + name})
            row["selected8_independent_byte_pins_verified"] = True; persist()
        if len(retained) != 24 or {p.name for p in staging.iterdir()} != {r["file"] for r in retained}:
            raise ValueError("Complete24-byte-verified cohort required before public exposure")
        report.update(all24_original_files_hashed_before_depth_values=True, depth_values_decoded=False,
            phase="publish_frozen_subset", selected_records=retained); persist()
        for row in retained:
            target = public if row["kind"] == "rgb" else private
            (staging / row["file"]).rename(target / row["file"]); (target / row["file"]).chmod(0o444 if row["kind"] == "rgb" else 0o400)
        for row in retained:
            target = public if row["kind"] == "rgb" else private
            if identity(target / row["file"]) != {key: row[key] for key in ("bytes", "sha256")}:
                raise ValueError("All24 selected bytes must replay before public manifest JSON")
        report["all24_retained_bytes_rehashed_before_public_manifest"] = True; persist()
        images = []
        for sequence in frozen["sequences"]:
            for declared in sequence["selected"]:
                name = output_name(sequence["sequence_id"], "rgb", declared["rgb_timestamp"])
                images.append(dict(scene_id=sequence["sequence_id"], frame_id=declared["rgb_zero_based_rank"], file=name,
                    original_rgb_file=declared["rgb"]["path"], timestamp=declared["rgb_timestamp"],
                    sha256=declared["rgb"]["sha256"], width=640, height=480))
        manifest = dict(schema=SCHEMA, dataset="TUM RGB-D", license="CC-BY-4.0", selection=SELECTION, images=images)
        save(public / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o444)
    finally:
        shutil.rmtree(staging); report["disposable_archives_and_staging_removed"] = not staging.exists()
    files = output_inventory(destination, frozen)
    source_expected = {"eval_private/" + row["file"]: {key: row[key] for key in ("bytes", "sha256")}
        for row in [*report["license_evidence"], *[term for archive in report["archives"] for term in archive["term_files"]]]}
    if {key: value for key, value in files.items() if key.startswith("eval_private/source/")} != source_expected:
        raise ValueError("Exact retained license/description/optional-term source inventory changed")
    return files


def cleanup_failure(destination, frozen):
    """Remove only reserved media names; unknown files/aliases cause STOP."""
    allowed = {output_name(s["sequence_id"], k, r[k + "_timestamp"]) for s in frozen["sequences"] for r in s["selected"] for k in ("rgb", "depth")}
    removed = 0
    for directory in (destination / "inputs", destination / "eval_private"):
        for path in directory.iterdir():
            if path.name in ("source", "acquisition-report.json"): continue
            if path.name not in allowed | {"manifest.json"} or path.is_symlink() or not path.is_file():
                raise ValueError("Unknown failed output preserved; no broad cleanup")
            path.unlink(); removed += 1
    return removed


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("Heavy external data stays on Azure Linux CPU")
    root, code, revision = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    helpers, protocol_id, frozen = bound_source(root, code, revision, Path(__file__)); destination = canonical(root / BASE)
    if os.environ.get("WR_TUM_OUTPUT_RESERVED") != "1" or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("Exclusive empty reserved new output required; no reuse/overwrite")
    private = destination / "eval_private"; private.mkdir(mode=0o700); (destination / "inputs").mkdir(mode=0o755)
    report = dict(stage=STAGE, status="fail", phase="license", producer_revision=revision,
        script_sha256=helpers[SOURCE_FILES[0]]["sha256"], source_helpers=helpers, protocol_identity=protocol_id,
        budget_seconds=BUDGET, archives=[], device="cpu", gpu_used=False, inference_performed=False,
        challenge_inputs_used=False, source_camera_or_trajectory_read=False, depth_values_decoded=False,
        ground_truth_used_for_inference=False, independent_full_archive_SHA256_known=False,
        training_overlap_verified=False, challenge_overlap_verified=False, accuracy_verified=False)
    started = time.perf_counter(); path = private / "acquisition-report.json"; error = None
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole original TUM acquisition exceeded600s; no retry")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        oldmask = os.umask(0o077)
        try:
            persist(); report["output_files"] = acquire(destination, frozen, report, persist)
            if output_inventory(destination, frozen) != report["output_files"]: raise ValueError("Retained bytes changed")
            report["outputs_rehashed_after"] = True
        except Exception as caught: error = caught; report.update(error_type=type(caught).__name__, error=str(caught)[:250])
        finally:
            try:
                if bound_source(root, code, revision, Path(__file__)) != (helpers, protocol_id, frozen):
                    raise ValueError("Original producer/protocol changed after acquisition")
                report["source_helpers_rehashed_after"] = True
            except Exception as caught:
                report.update(source_recheck_error_type=type(caught).__name__, source_recheck_error=str(caught)[:250])
                if error is None: error = caught
            if error is not None:
                try: report["failed_partial_media_removed"] = cleanup_failure(destination, frozen)
                except Exception as caught:
                    report.update(cleanup_error_type=type(caught).__name__, cleanup_error=str(caught)[:250]); error = caught
            else: report.update(status="pass", phase="complete", images_completed=12, private_depths_completed=12)
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); os.umask(oldmask)
            persist(); path.chmod(0o400)
    if error is not None: raise error


if __name__ == "__main__": main()
