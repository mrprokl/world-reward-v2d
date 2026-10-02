"""Acquire pinned external TUD-L validation on Azure; publish nine RGBs only.

Selection uses ZIP RGB filenames before opening any private annotation values.
Models, sensor depth, cameras and all selected instances remain evaluation-only.
This is not challenge data, inference, an overlap audit or an accuracy claim.
"""
import argparse
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
import time
import urllib.parse
import urllib.request
import zipfile
import zlib

REVISION = "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
BASE_URL = f"https://huggingface.co/datasets/bop-benchmark/tudl/resolve/{REVISION}/"
ARCHIVES = {
    "tudl_base.zip": (3394, "963dc6044eaf575a0f458b2804152da552b7fbca385d0ab01c115c6285e500ad"),
    "tudl_models.zip": (2484229, "e649682d78d8e1ea463633cd09421b49b43600b32d84587eec3bcbe4a63e6a0f"),
    "tudl_test_bop19.zip": (372464733, "cc68c55004dc7822a10910033aedffb3d40bf5bb0a39497b13e566dedddbea14"),
}
README_URL = f"https://huggingface.co/datasets/bop-benchmark/tudl/raw/{REVISION}/README.md"
BOP_URL = "https://bop.felk.cvut.cz/datasets/"
README_SHA = "f35ac7b30195da3b9b01d156fe7f0626e0ae4ac7f6d5757eb0b6ee3b8f31dcab"
BOP_SECTION_SHA = "aeccefac103003c079c96b8963809a9267a5703f7448b4fa51619853e620b31a"
SELECTION = "sorted_RGB_filenames_first_median_index_n_div_2_last_per_scene_before_private_values"
EXPECTED_IDS = {1: (0, 4074, 8227), 2: (3, 4013, 7710), 3: (4, 4028, 7969)}
MAX_EXPANDED, MAX_FILES = 2_000_000_000, 10000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def fetch(url, limit):
    """Bounded public HTTPS source text; no credentials or redirects to HTTP."""
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=30) as response:
        validate_https(response.geturl())
        data = response.read(limit+1)
    if len(data) > limit: raise ValueError("Public metadata exceeds fixed byte bound")
    return data


def validate_https(url):
    parsed = urllib.parse.urlsplit(url); host = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username or parsed.password or not (host == "huggingface.co" or host.endswith(".hf.co") or host == "bop.felk.cvut.cz"):
        raise ValueError("Require public allowlisted HTTPS source")


def download(url, destination, size, expected_sha):
    validate_https(url); count = 0; h = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=30) as response, destination.open("xb") as output:
        validate_https(response.geturl())
        while block := response.read(min(1024*1024, size-count+1)):
            count += len(block)
            if count > size: raise ValueError("Pinned archive exceeded exact byte count")
            h.update(block); output.write(block)
    if count != size or h.hexdigest() != expected_sha: raise ValueError("Pinned archive byte count/SHA mismatch")
    destination.chmod(0o400)


def zip_inventory(archive, budget):
    """Inspect every member before extraction, including members later discarded."""
    members = {}; normalized = set()
    for info in archive.infolist():
        name = info.filename; path = PurePosixPath(name)
        mode = info.external_attr >> 16; kind = stat.S_IFMT(mode)
        if (not name or "\\" in name or "\x00" in name or path.is_absolute() or any(p in ("", ".", "..") for p in name.rstrip("/").split("/"))
                or str(path) != name.rstrip("/") or name.rstrip("/") in normalized
                or kind not in ((0, stat.S_IFDIR) if info.is_dir() else (0, stat.S_IFREG)) or info.flag_bits & 1
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError("Unsafe, duplicate, symlink or unsupported ZIP member")
        normalized.add(name.rstrip("/")); budget[0] += info.file_size; budget[1] += 1
        if info.file_size > 128_000_000 or budget[0] > MAX_EXPANDED or budget[1] > MAX_FILES:
            raise ValueError("ZIP expansion/file bound exceeded")
        if not info.is_dir(): members[name] = info
    return members


def inspect_layout(name, members):
    if name == "tudl_base.zip":
        if set(members) != {"tudl/camera.json", "tudl/dataset_info.md", "tudl/test_targets_bop19.json"}: raise ValueError("Unexpected base ZIP layout")
    elif name == "tudl_models.zip":
        expected = {f"{folder}/{file}" for folder in ("models", "models_eval") for file in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")}
        if set(members) != expected: raise ValueError("Unexpected models ZIP layout")
    else:
        pattern = r"test/00000[123]/(?:scene_(?:camera|gt|gt_info)\.json|(?:rgb|depth)/[0-9]{6}\.png|(?:mask|mask_visib)/[0-9]{6}_[0-9]{6}\.png)"
        if any(not re.fullmatch(pattern, p) for p in members): raise ValueError("Unexpected BOP19 ZIP layout")
        for scene in EXPECTED_IDS:
            prefix = f"test/{scene:06d}/"
            if any(prefix+name not in members for name in ("scene_camera.json", "scene_gt.json", "scene_gt_info.json")):
                raise ValueError("Missing original scene annotations")


def select_rgb_names(members):
    selected = []
    for scene in EXPECTED_IDS:
        files = sorted(p for p in members if re.fullmatch(fr"test/{scene:06d}/rgb/[0-9]{{6}}\.png", p))
        if len(files) != 200: raise ValueError("Pinned BOP19 scene must contain exactly200 RGB filenames")
        chosen = [files[i] for i in (0, len(files)//2, len(files)-1)]
        if tuple(int(PurePosixPath(p).stem) for p in chosen) != EXPECTED_IDS[scene]: raise ValueError("Pinned filename-only cohort changed")
        selected.extend((scene, int(PurePosixPath(p).stem), p) for p in chosen)
    return selected


def png_dimensions(data):
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\x00\x00\x00\rIHDR": raise ValueError("Require native RGB PNG IHDR")
    width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data[16:29])
    if ((width, height, bits, color) != (640, 480, 8, 2) or compression or filtering or interlace not in (0, 1)
            or zlib.crc32(data[12:29]) != struct.unpack(">I", data[29:33])[0]): raise ValueError("Original RGB PNG640x480 contract failed")
    return width, height


def save_bytes(path, data, mode=0o400):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream: stream.write(data)
    path.chmod(mode)


def license_evidence(private):
    readme = fetch(README_URL, 4096); page = fetch(BOP_URL, 500_000)
    start = page.index(b'<h3 id="TUD-L">'); end = page.index(b'<div class="download_links">', start)
    section = page[start:end]
    if hashlib.sha256(readme).hexdigest() != README_SHA or hashlib.sha256(section).hexdigest() != BOP_SECTION_SHA:
        raise ValueError("Pinned concordant TUD-L licence source changed")
    folder = private/"source/licenses"
    save_bytes(folder/"huggingface-README.md", readme); save_bytes(folder/"BOP-TUD-L-section.html", section)
    attribution = {"dataset": "TU Dresden Light (TUD-L)", "attribution": "Hodan, Michel et al., BOP: Benchmark for 6D Object Pose Estimation, ECCV2018",
                   "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                   "upstream_modified": "Filename-only nine-RGB subset; selected annotation dictionaries filtered, all instances retained"}
    save_bytes(folder/"attribution.json", (json.dumps(attribution, indent=2)+"\n").encode())
    return [{"url": README_URL, "sha256": README_SHA}, {"url": BOP_URL+"#TUD-L", "sha256": BOP_SECTION_SHA, "scope": "447byte TUD-L licence section only"}]


def retain_subset(archives, private, inputs, selected):
    for name in ("tudl_base.zip", "tudl_models.zip"):
        archive, members = archives[name]
        for member in members:
            data = archive.read(member)
            if member.endswith((".md", ".txt")) and re.search(rb"(?i)non.?commercial|by-nc|all rights reserved", data): raise ValueError("Archive licence conflicts with concordant CC-BY-SA4")
            path = private/"source"/("base" if name == "tudl_base.zip" else "")/member
            save_bytes(path, data)
    archive, members = archives["tudl_test_bop19.zip"]; images = []
    # Filename selection has already completed; only now may annotation values be read.
    for scene in EXPECTED_IDS:
        frames = [frame for s, frame, _ in selected if s == scene]; prefix = f"test/{scene:06d}/"
        annotations = {}
        for kind in ("camera", "gt", "gt_info"):
            value = json.loads(archive.read(prefix+f"scene_{kind}.json"))
            if not isinstance(value, dict) or any(str(f) not in value for f in frames): raise ValueError("Missing selected private annotation keys")
            annotations[kind] = {str(f): value[str(f)] for f in frames}
        for frame in frames:
            gt = annotations["gt"][str(frame)]; info = annotations["gt_info"][str(frame)]
            if not isinstance(gt, list) or not gt or not isinstance(info, list) or len(info) != len(gt): raise ValueError("All selected instances require private annotations")
            masks = {p for p in members if p.startswith(prefix+f"mask_visib/{frame:06d}_")}
            if masks != {prefix+f"mask_visib/{frame:06d}_{i:06d}.png" for i in range(len(gt))}: raise ValueError("All selected instance visibility masks required")
            depth_name = prefix+f"depth/{frame:06d}.png"
            if depth_name not in members: raise ValueError("Selected sensor depth missing")
            for member in [depth_name, *sorted(masks)]: save_bytes(private/"source"/member, archive.read(member))
        for kind, value in annotations.items(): save_bytes(private/"source"/prefix/f"scene_{kind}.json", (json.dumps(value, sort_keys=True)+"\n").encode())
    for scene, frame, member in selected:
        data = archive.read(member); width, height = png_dimensions(data)
        name = f"scene_{scene:06d}_frame_{frame:06d}.png"; save_bytes(inputs/name, data, 0o444)
        images.append({"scene_id": scene, "frame_id": frame, "file": name, "sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height})
    return {"schema": "world-reward-tudl-rgb-v1", "revision": REVISION, "license": "CC-BY-SA-4.0", "selection": SELECTION, "images": images}


def acquire(destination, report, persist):
    private, inputs = destination/"eval_private", destination/"inputs"; downloads = private/".downloads"
    (private/"source").mkdir(mode=0o700)
    downloads.mkdir(mode=0o700); archives = {}; budget = [0, 0]
    try:
        report["license_evidence"] = license_evidence(private); persist()
        for name, (size, sha) in ARCHIVES.items():
            report.update(phase="download", active_archive=name); persist(); path = downloads/name
            download(BASE_URL+name, path, size, sha)
            archive = zipfile.ZipFile(path); archives[name] = (archive, {})
            members = zip_inventory(archive, budget); inspect_layout(name, members); archives[name] = (archive, members)
            report["archives"].append({"file": name, "url": BASE_URL+name, "bytes": size, "sha256": sha, "members": len(members)}); persist()
        selected = select_rgb_names(archives["tudl_test_bop19.zip"][1])
        report.update(phase="retain_subset", selected_records=[{"scene_id": s, "frame_id": f, "archive_rgb_file": p} for s, f, p in selected],
                      selection_before_private_annotation_values=True); persist()
        manifest = retain_subset(archives, private, inputs, selected)
        report["selected_records"] = [{k: r[k] for k in ("scene_id", "frame_id", "file")} | {"rgb_sha256": r["sha256"]} for r in manifest["images"]]
        save_bytes(inputs/"manifest.json", (json.dumps(manifest, indent=2)+"\n").encode(), 0o444)
        report.update(public_manifest_sha256=digest(inputs/"manifest.json"), retained_files=[{"file": str(p.relative_to(private)), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted((private/"source").rglob("*")) if p.is_file()], expanded_bytes_inspected=budget[0], status="pass", phase="complete")
        report.pop("active_archive", None)
    finally:
        for archive, _ in archives.values(): archive.close()
        shutil.rmtree(downloads); report["disposable_archives_removed"] = True


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("External data acquisition stays on Azure Linux")
    root = Path(os.environ["WR_ROOT"]); destination = root/"validation/tudl_rgb_v1"; revision = os.environ["WR_CODE_REVISION"]
    if (not re.fullmatch(r"[0-9a-f]{40}", revision) or not destination.is_dir() or any(destination.iterdir())
            or destination.resolve() != destination.absolute() or os.environ.get("WR_TUDL_OUTPUT_RESERVED") != "1"):
        raise ValueError("Require immutable source and exclusively reserved regular empty cohort")
    private = destination/"eval_private"; private.mkdir(mode=0o700); (destination/"inputs").mkdir(mode=0o755)
    report = {"stage": "external_tudl_rgb_only_validation_acquisition", "status": "fail", "phase": "licence", "archives": [],
              "code_revision": revision, "script_sha256": digest(Path(__file__)), "dataset_revision": REVISION, "license": "CC-BY-SA-4.0",
              "selection": SELECTION, "expected_scene_frame_ids": EXPECTED_IDS, "budget_seconds": 600, "device": "cpu", "gpu_used": False,
              "challenge_inputs_used": False, "inference_performed": False, "challenge_overlap_verified": False, "accuracy_verified": False,
              "private_annotations_exported_as_inference_inputs": False, "train_or_full_test_downloaded": False}
    started = time.perf_counter(); path = private/"acquisition-report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Pinned TUD-L acquisition exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(600)
        try: persist(); acquire(destination, report, persist)
        except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o400)


if __name__ == "__main__": main()
