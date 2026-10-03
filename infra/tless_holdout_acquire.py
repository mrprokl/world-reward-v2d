"""Azure-only pinned T-LESS acquisition: twelve RGBs, sealed sensor truth.

No inference, model/mesh acquisition, camera fitting or accuracy claims. Whole
archive bytes and layouts precede extraction; filename selection precedes every
private annotation value. All selected object instances remain evaluation-only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import struct
import time
import zipfile
import zlib

import tudl_holdout_acquire as shared
import tudl_holdout_inputs as strict

source = shared.source
ROOT = Path("/srv/scenesmith/world-reward")
NAMESPACE = "validation/tless_frame_holdout_v1"
PROTOCOL = "configs/tless_frame_holdout_protocol.json"
PROTOCOL_IDENTITY = {"bytes": 3563, "sha256": "cff213716625f4070d4547f8a4cb0f361692d93f19aaebad8eac1cc0d0e9bd1c"}
STAGE = "external_tless_rgb_only_frame_holdout_acquisition"
SCHEMA = "world-reward-tless-frame-holdout-rgb-v1"
REVISION = "5fd309a04476a842d93abfb584fba9ee7caecdf1"
BASE_URL = f"https://huggingface.co/datasets/bop-benchmark/tless/resolve/{REVISION}/"
ARCHIVES = {
    "tless_base.zip": {"bytes": 49597, "sha256": "dd70ca884b7c471a530a952f70c5ab2c212f3d2c2f371be86397442b97d70a7e", "git_blob_sha1": "aa13c0f04b155533cdcb6a8bbd037dd05fdb7cdb"},
    "tless_test_primesense_bop19.zip": {"bytes": 825276992, "sha256": "1a18f6bbfb5ac4ced8529f7a35225adfed88c0f62ef38067933e2b541ef1d00b", "git_blob_sha1": "72f63df493066d4e431e949f4cbd8db73ce8f194"}}
LICENSE_SOURCES = {
    "huggingface_README": {"url": f"https://huggingface.co/datasets/bop-benchmark/tless/raw/{REVISION}/README.md", "bytes": 30,
        "sha256": "8f0c7e5b67cd5f155f6570a713fd5f75ed8202614734e37df3566efa045460d2", "git_blob_sha1": "7da79263d0e701a53a0d662d245c4fc3db39b287"},
    "BOP_TLESS_section": {"url": "https://bop.felk.cvut.cz/datasets/#T-LESS", "bytes": 958,
        "sha256": "aa30ba3f957cca7a0982243086467a360139905da7fc0f4598b8a68db6797131"}}
FORMAT_SOURCE = {"url": "https://raw.githubusercontent.com/thodan/bop_toolkit/b72b3015c87a96fa6398c2ef4c196e85f798d3e6/bop_toolkit_lib/dataset_params.py",
    "bytes": 32303, "sha256": "a935fc4f6fd42f367a0bbf816036f783fb4c5d87200a8615ce1e08e6d51af05e", "revision": "b72b3015c87a96fa6398c2ef4c196e85f798d3e6"}
SCENES = (1, 10, 20)
WIDTH, HEIGHT, BUDGET = 720, 540, 600
MAX_EXPANDED, MAX_FILES, MAX_MEMBER, MAX_RETAINED = 8_000_000_000, 500_000, 128_000_000, 256_000_000
HELPERS = ("infra/tless_holdout_acquire.py", "infra/run_tless_holdout_acquire.sh", "infra/tudl_holdout_acquire.py",
    "infra/tudl_acquire.py", "infra/tudl_holdout_inputs.py", PROTOCOL)
ATTRIBUTION = {"dataset": "T-LESS", "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "creators": "Tomas Hodan, Pavel Haluza, Stepan Obdrzalek, Jiri Matas, Manolis Lourakis, Xenophon Zabulis",
    "citation": "Hodan et al., T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects, WACV 2017",
    "source_url": "https://bop.felk.cvut.cz/datasets/#T-LESS", "revision": REVISION,
    "modifications": "Filename-only twelve-RGB subset; RGB bytes unchanged; private annotation dictionaries filtered to selected frames, all object instances retained"}


def exact(value, expected):
    if type(value) is not type(expected): return False
    if type(expected) is dict: return set(value) == set(expected) and all(exact(value[k], v) for k, v in expected.items())
    if type(expected) is list: return len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
    return value == expected


def bound_source(root, code, revision, executing):
    root, code = Path(root), Path(code)
    if (root != ROOT or not root.is_dir() or root.resolve() != root or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or code != root / "jobs" / revision / "run_tless_holdout_acquire/code" or code.resolve() != code
            or Path(executing) != code / HELPERS[0] or Path(shared.__file__).resolve() != code / HELPERS[2]
            or Path(source.__file__).resolve() != code / HELPERS[3] or Path(strict.__file__).resolve() != code / HELPERS[4]):
        raise ValueError("Actual immutable Azure T-LESS source namespace and original helpers required")
    return {name: shared.identity(code / name) for name in HELPERS}


def load_protocol(code):
    path = code / PROTOCOL; identity = shared.identity(path); raw = path.read_bytes()
    if identity != PROTOCOL_IDENTITY: raise ValueError("Exact original frozen protocol byte identity required")
    if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != identity: raise ValueError("Protocol changed before JSON")
    protocol = strict.strict_json(raw)
    required = {"schema": "world-reward-tless-frame-holdout-protocol-v1", "dataset": "bop-benchmark/tless",
        "revision": REVISION, "license": "CC-BY-4.0", "archives": ARCHIVES, "license_sources": LICENSE_SOURCES,
        "source_format": FORMAT_SOURCE, "acquisition": {"budget_seconds": BUDGET, "maximum_expanded_zip_bytes": MAX_EXPANDED,
        "maximum_zip_members": MAX_FILES, "maximum_member_bytes": MAX_MEMBER, "maximum_retained_bytes": MAX_RETAINED, "models_acquired": False},
        "selection": {"scene_ids": list(SCENES), "ranks": "floor(k*N/5), k=1,2,3,4 on sorted native RGB filenames in each fixed scene",
        "N_200_ranks": [40, 80, 120, 160], "minimum_RGB_files_per_scene": 5, "images": 12, "before_private_annotation_values": True},
        "public_inputs": {"width": WIDTH, "height": HEIGHT, "RGB_files_only_plus_attributed_manifest": True, "private_K_depth_masks_meshes_or_labels": False}}
    if type(protocol) is not dict or any(k not in protocol or not exact(protocol[k], v) for k, v in required.items()):
        raise ValueError("Exact preregistered archive/licence/selection/budget protocol required")
    if shared.identity(path) != identity: raise ValueError("Frozen protocol changed")
    return protocol, identity


def checked_text(data, identity):
    if {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} != {k: identity[k] for k in ("bytes", "sha256")}:
        raise ValueError("Pinned source/licence text changed")
    return data


def save_bytes(path, data, mode=0o400):
    """Original writer with private intermediate-directory permissions."""
    oldmask = os.umask(0o077)
    try: source.save_bytes(path, data, mode)
    finally: os.umask(oldmask)


def license_evidence(private):
    readme = checked_text(source.fetch(LICENSE_SOURCES["huggingface_README"]["url"], 4096), LICENSE_SOURCES["huggingface_README"])
    page = source.fetch("https://bop.felk.cvut.cz/datasets/", 100_000)
    start = page.index(b'<h3 id="T-LESS">'); end = page.index(b'<div class="download_links">', start)
    section = checked_text(page[start:end], LICENSE_SOURCES["BOP_TLESS_section"])
    if b"license: cc-by-4.0" not in readme or b"https://creativecommons.org/licenses/by/4.0/" not in section:
        raise ValueError("Concordant explicit commercial-compatible CC-BY4 evidence required")
    folder = private / "source/licenses"
    save_bytes(folder / "huggingface-README.md", readme)
    save_bytes(folder / "BOP-TLESS-section.html", section)
    save_bytes(folder / "attribution.json", (json.dumps(ATTRIBUTION, indent=2) + "\n").encode())
    return LICENSE_SOURCES


def zip_inventory(archive, budget):
    """T-LESS-sized fixed bounds, no mutation of original TUD-L globals."""
    members, normalized = {}, set()
    for info in archive.infolist():
        name = info.filename; path = PurePosixPath(name); kind = stat.S_IFMT(info.external_attr >> 16)
        if (not name or "\\" in name or "\x00" in name or path.is_absolute()
                or any(p in ("", ".", "..") for p in name.rstrip("/").split("/"))
                or str(path) != name.rstrip("/") or name.rstrip("/") in normalized
                or kind not in ((0, stat.S_IFDIR) if info.is_dir() else (0, stat.S_IFREG))
                or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError("Unsafe, duplicate, symlink or unsupported ZIP member")
        normalized.add(name.rstrip("/")); budget[0] += info.file_size; budget[1] += 1
        if info.file_size > MAX_MEMBER or budget[0] > MAX_EXPANDED or budget[1] > MAX_FILES:
            raise ValueError("Preregistered ZIP expansion/member budget exceeded; no sweep")
        if not info.is_dir(): members[name] = info
    return members


def inspect_layout(name, members):
    if name == "tless_base.zip":
        required = {"tless/dataset_info.md", "tless/test_targets_bop19.json", "tless/camera_primesense.json"}
        allowed = re.compile(r"tless/(?:camera(?:_(?:primesense|kinect|canon))?\.json|dataset_info\.md|test_targets_bop19\.json|(?:README|LICENSE|COPYING)(?:\.(?:md|txt))?)", re.I)
        if not required <= set(members) or any(not allowed.fullmatch(p) for p in members):
            raise ValueError("Unrecognized minimal T-LESS base layout; no guess, models or hidden assets")
    elif name == "tless_test_primesense_bop19.zip":
        pattern = r"test_primesense/(?:00000[1-9]|00001[0-9]|000020)/(?:scene_(?:camera|gt|gt_info)\.json|(?:rgb|depth)/[0-9]{6}\.png|(?:mask|mask_visib)/[0-9]{6}_[0-9]{6}\.png)"
        if any(not re.fullmatch(pattern, p) for p in members): raise ValueError("Unrecognized native single-view BOP19 test layout")
        for scene in SCENES:
            if any(f"test_primesense/{scene:06d}/scene_{kind}.json" not in members for kind in ("camera", "gt", "gt_info")):
                raise ValueError("Missing original selected scene annotation files")
    else: raise ValueError("Only two preregistered archives allowed")


def selection_ranks(count):
    if type(count) is not int or count < 5: raise ValueError("At least five original RGB filenames per fixed scene required")
    ranks = tuple(k * count // 5 for k in (1, 2, 3, 4))
    if len(set(ranks)) != 4 or max(ranks) >= count: raise ValueError("Four distinct preregistered uniform ranks required")
    return ranks


def select_rgb_names(members):
    if type(members) is not dict or any(type(name) is not str for name in members): raise ValueError("Filename-only member mapping required")
    result = []
    for scene in SCENES:
        files = sorted(p for p in members if re.fullmatch(fr"test_primesense/{scene:06d}/rgb/[0-9]{{6}}\.png", p))
        for rank in selection_ranks(len(files)):
            name = files[rank]; result.append((scene, int(PurePosixPath(name).stem), name))
    if len(result) != 12 or len(set(result)) != 12: raise ValueError("All twelve original filename-selected RGBs required")
    return result


def png_dimensions(data):
    if len(data) < 33 or data[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR": raise ValueError("Native RGB PNG header required")
    width, height, bits, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data[16:29])
    if ((width, height, bits, color) != (WIDTH, HEIGHT, 8, 2) or compression or filtering or interlace not in (0, 1)
            or zlib.crc32(data[12:29]) != struct.unpack(">I", data[29:33])[0]):
        raise ValueError("Unmodified original RGB PNG720x540 required, no resize")
    return width, height


def check_embedded_licenses(archives):
    archive, members = archives["tless_base.zip"]
    texts = {}
    for name in members:
        if name.lower().endswith((".md", ".txt")) or re.search(r"(?i)/(?:LICENSE|COPYING)$", name):
            data = archive.read(name)
            if len(data) > 1_000_000 or re.search(rb"(?i)non.?commercial|by[- ]?nc|all rights reserved|research (?:use|purposes) only", data):
                raise ValueError("Embedded dataset terms conflict with CC-BY4; STOP")
            texts[name] = data
    info = texts.get("tless/dataset_info.md")
    if not info or not re.search(rb"(?i)t[- ]?less", info): raise ValueError("Original embedded T-LESS dataset attribution missing")
    declarations = [data for data in texts.values() if re.search(rb"(?i)licen[cs]e|creativecommons|CC[- ]?BY", data)]
    if not declarations or any(not re.search(rb"(?i)creativecommons\.org/licenses/by/4\.0|CC[- ]BY[- ]4\.0(?:[^A-Za-z0-9-]|$)", data) for data in declarations):
        raise ValueError("Embedded licence missing, ambiguous or inconsistent; no inferred waiver")
    return sorted(texts)


def verified_archives(archives):
    """Recheck immutable source files even when retention is called directly."""
    if type(archives) is not dict or set(archives) != set(ARCHIVES): raise ValueError("Exactly two pinned source archives required")
    for name, (archive, members) in archives.items():
        path = getattr(archive, "filename", None); expected = {k: ARCHIVES[name][k] for k in ("bytes", "sha256")}
        if type(path) is not str or shared.identity(Path(path)) != expected:
            raise ValueError("Complete original archive SHA/bytes required before any member interpretation")
        inspect_layout(name, members)


def retain_subset(archives, private, public, selected):
    verified_archives(archives)
    if selected != select_rgb_names(archives["tless_test_primesense_bop19.zip"][1]):
        raise ValueError("Complete filename-only selection must precede every private annotation read")
    check_embedded_licenses(archives)
    retained_bytes = 0
    def save(path, data, mode=0o400):
        nonlocal retained_bytes
        retained_bytes += len(data)
        if retained_bytes > MAX_RETAINED: raise ValueError("Fixed selected-byte budget exceeded")
        save_bytes(path, data, mode)
    archive, members = archives["tless_base.zip"]
    for name in members:
        data = archive.read(name)
        if name.endswith(".json"): strict.strict_json(data)
        save(private / "source/base" / name, data)
    archive, members = archives["tless_test_primesense_bop19.zip"]
    # Fully verified selection above; only now do we interpret private labels.
    for scene in SCENES:
        prefix = f"test_primesense/{scene:06d}/"; frames = [frame for s, frame, _ in selected if s == scene]; annotations = {}
        for kind in ("camera", "gt", "gt_info"):
            value = strict.strict_json(archive.read(prefix + f"scene_{kind}.json"))
            if type(value) is not dict or any(str(frame) not in value for frame in frames): raise ValueError("Original selected annotation keys missing")
            annotations[kind] = {str(frame): value[str(frame)] for frame in frames}
        for frame in frames:
            key = str(frame); gt, info, camera = (annotations[kind][key] for kind in ("gt", "gt_info", "camera"))
            if (type(gt) is not list or not gt or type(info) is not list or len(info) != len(gt)
                    or any(type(row) is not dict or type(row.get("obj_id")) is not int or not 1 <= row["obj_id"] <= 30 for row in gt)):
                raise ValueError("All original instance identities and matching GTinfo required")
            K = camera.get("cam_K") if type(camera) is dict else None; scale = camera.get("depth_scale") if type(camera) is dict else None
            if (type(K) is not list or len(K) != 9 or any(type(v) not in (int, float) or not math.isfinite(v) for v in K)
                    or type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
                raise ValueError("Original finite private camera/depthscale required; never public")
            masks = {p for p in members if p.startswith(prefix + f"mask_visib/{frame:06d}_")}
            expected = {prefix + f"mask_visib/{frame:06d}_{i:06d}.png" for i in range(len(gt))}
            depth = prefix + f"depth/{frame:06d}.png"
            if masks != expected or depth not in members: raise ValueError("Every selected object mask and native depth required")
            for name in (depth, *sorted(masks)): save(private / "source" / name, archive.read(name))
        for kind, value in annotations.items(): save(private / "source" / prefix / f"scene_{kind}.json", (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
    images = []
    for scene, frame, name in selected:
        raw = archive.read(name); width, height = png_dimensions(raw); file = f"scene_{scene:06d}_frame_{frame:06d}.png"
        save(public / file, raw, 0o444); images.append(dict(scene_id=scene, frame_id=frame, file=file,
            sha256=hashlib.sha256(raw).hexdigest(), width=width, height=height))
    return dict(schema=SCHEMA, revision=REVISION, license="CC-BY-4.0", selection="floor_kN_div5_k1to4_per_fixed_scene_before_private_values",
        attribution=ATTRIBUTION, images=images)


def output_inventory(destination, manifest, selected):
    public, private = destination / "inputs", destination / "eval_private"
    if {p.name for p in destination.iterdir()} != {"inputs", "eval_private"}: raise ValueError("No sibling output assets allowed")
    expected = {"schema": SCHEMA, "revision": REVISION, "license": "CC-BY-4.0", "selection": "floor_kN_div5_k1to4_per_fixed_scene_before_private_values", "attribution": ATTRIBUTION}
    if type(manifest) is not dict or set(manifest) != {*expected, "images"} or any(not exact(manifest[k], v) for k, v in expected.items()):
        raise ValueError("Original attributed RGB-only manifest required")
    if type(manifest["images"]) is not list or len(manifest["images"]) != 12 or len(selected) != 12: raise ValueError("Exactly twelve public RGB records required")
    for row, (scene, frame, _) in zip(manifest["images"], selected):
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or not exact({k: row[k] for k in ("scene_id", "frame_id", "file", "width", "height")},
                    dict(scene_id=scene, frame_id=frame, file=f"scene_{scene:06d}_frame_{frame:06d}.png", width=WIDTH, height=HEIGHT))
                or type(row["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("Ordered unmodified original RGB identity required; no private fields")
    public_names = {"manifest.json", *[row["file"] for row in manifest["images"]]}
    if {p.name for p in public.iterdir()} != public_names: raise ValueError("Public scope is twelve RGBs and attributed manifest only")
    public_hashes = {name: shared.identity(public / name) for name in sorted(public_names)}
    if strict.strict_json((public / "manifest.json").read_bytes()) != manifest: raise ValueError("Manifest changed after saving")
    if any(public_hashes[row["file"]]["sha256"] != row["sha256"] for row in manifest["images"]): raise ValueError("Original RGB bytes changed")
    required = {"source/licenses/" + name for name in ("huggingface-README.md", "BOP-TLESS-section.html", "attribution.json")}
    base = private / "source/base"; base_names = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
    inspect_layout("tless_base.zip", dict.fromkeys(base_names)); required.update("source/base/" + name for name in base_names)
    for scene in SCENES:
        prefix = f"source/test_primesense/{scene:06d}/"; frames = [frame for s, frame, _ in selected if s == scene]; annotations = {}
        for kind in ("camera", "gt", "gt_info"):
            required.add(prefix + f"scene_{kind}.json"); value = strict.strict_json((private / (prefix + f"scene_{kind}.json")).read_bytes())
            if type(value) is not dict or set(value) != {str(frame) for frame in frames}: raise ValueError("Private dictionaries must contain only selected keys")
            annotations[kind] = value
        for frame in frames:
            gt, info = annotations["gt"][str(frame)], annotations["gt_info"][str(frame)]
            if type(gt) is not list or not gt or type(info) is not list or len(info) != len(gt): raise ValueError("All selected instances retained")
            required.add(prefix + f"depth/{frame:06d}.png")
            required.update(prefix + f"mask_visib/{frame:06d}_{i:06d}.png" for i in range(len(gt)))
    if {p.name for p in private.iterdir()} != {"source", "acquisition-report.json"}: raise ValueError("No disposable ZIPs or extras may survive")
    actual, dirs = set(), set()
    for path in (private / "source", *(private / "source").rglob("*")):
        if path.is_symlink(): raise ValueError("Private output symlink forbidden")
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode) != 0o700: raise ValueError("Private source directories must remain0700")
            dirs.add(str(path.relative_to(private)))
        elif path.is_file():
            if stat.S_IMODE(path.stat().st_mode) != 0o400: raise ValueError("Private source files must remain0400")
            actual.add(str(path.relative_to(private)))
        else: raise ValueError("Only regular private files/directories permitted")
    expected_dirs = {str(p) for name in required for p in PurePosixPath(name).parents if str(p) != "."}
    if actual != required or dirs != expected_dirs: raise ValueError("Exact full-instance private inventory required")
    if stat.S_IMODE(public.stat().st_mode) != 0o755 or stat.S_IMODE(private.stat().st_mode) != 0o700:
        raise ValueError("Original strict public/private directory split required")
    if any(stat.S_IMODE((public / name).stat().st_mode) != 0o444 for name in public_names): raise ValueError("Public input files must remain0444")
    return public_hashes, [{"file": name, **shared.identity(private / name)} for name in sorted(required)]


def acquire(destination, report, persist):
    private, public = destination / "eval_private", destination / "inputs"; downloads = private / ".downloads"
    (private / "source").mkdir(mode=0o700); downloads.mkdir(mode=0o700)
    archives, observed, budget = {}, {}, [0, 0]
    try:
        report["license_evidence"] = license_evidence(private); persist()
        for name, expected in ARCHIVES.items():
            report.update(phase="download", active_archive=name); persist(); path = downloads / name
            source.download(BASE_URL + name, path, expected["bytes"], expected["sha256"])
            observed[name] = shared.identity(path)
            if observed[name] != {k: expected[k] for k in ("bytes", "sha256")}:
                raise ValueError("Original complete archive SHA/bytes must pass before ZIP parsing")
            archive = zipfile.ZipFile(path); archives[name] = (archive, {})
            members = zip_inventory(archive, budget); inspect_layout(name, members); archives[name] = (archive, members)
            report["archives"].append(dict(file=name, url=BASE_URL + name, **expected, members=len(members))); persist()
        selected = select_rgb_names(archives["tless_test_primesense_bop19.zip"][1]); members = archives["tless_test_primesense_bop19.zip"][1]
        selections = []
        for scene in SCENES:
            names = sorted(p for p in members if re.fullmatch(fr"test_primesense/{scene:06d}/rgb/[0-9]{{6}}\.png", p))
            selections.append(dict(scene_id=scene, RGB_filename_count=len(names), ranks=list(selection_ranks(len(names))),
                filename_inventory_sha256=hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()))
        report.update(phase="retain_subset", selection_before_private_annotation_values=True, filename_selections=selections,
            selected_archive_records=[dict(scene_id=s, frame_id=f, archive_rgb_file=name) for s, f, name in selected]); persist()
        manifest = retain_subset(archives, private, public, selected)
        save_bytes(public / "manifest.json", (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(), 0o444)
        for name, expected in observed.items():
            if shared.identity(downloads / name) != expected: raise ValueError("Original archive changed during extraction")
        report["archive_sources_unchanged"] = True
    finally:
        try:
            for archive, _ in archives.values(): archive.close()
        finally:
            shutil.rmtree(downloads); report["disposable_archives_removed"] = not downloads.exists()
    public_hashes, retained = output_inventory(destination, manifest, selected)
    report.update(public_files=public_hashes, retained_files=retained, public_manifest_sha256=public_hashes["manifest.json"]["sha256"],
        images_completed=12, original_frame_coverage_verified=True, all_selected_object_instances_retained=True,
        embedded_licence_checked=True, expanded_bytes_inspected=budget[0], zip_members_inspected=budget[1],
        selected_records=[{k: row[k] for k in ("scene_id", "frame_id", "file")} | {"rgb_sha256": row["sha256"]} for row in manifest["images"]])
    report.pop("active_archive", None)
    return manifest, selected


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("All T-LESS acquisition stays on Azure Linux")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    helpers = bound_source(root, code, revision, Path(__file__)); protocol, protocol_identity = load_protocol(code)
    destination = root / NAMESPACE
    if (os.environ.get("WR_TLESS_HOLDOUT_RESERVED") != "1" or not destination.is_dir() or destination.resolve() != destination
            or any(p.is_symlink() for p in (destination, *destination.parents)) or any(destination.iterdir())):
        raise ValueError("Exclusively reserved empty canonical T-LESS output required; no resume/overwrite")
    private = destination / "eval_private"; private.mkdir(mode=0o700); (destination / "inputs").mkdir(mode=0o755)
    report = dict(stage=STAGE, status="fail", phase="licence", producer_revision=revision,
        script_sha256=helpers[HELPERS[0]]["sha256"], source_helpers=helpers, protocol_identity=protocol_identity,
        dataset_revision=REVISION, license="CC-BY-4.0", scene_ids=list(SCENES), budget_seconds=BUDGET, device="cpu", gpu_used=False,
        archives=[], challenge_inputs_used=False, inference_performed=False, ground_truth_used_for_inference=False,
        private_annotations_exported_as_inference_inputs=False, models_or_meshes_acquired=False,
        training_overlap_verified=False, challenge_overlap_verified=False, accuracy_verified=False,
        full_human_object_quality_verified=False, temporal_quality_verified=False, train_or_full_test_downloaded=False,
        new_capture_dataset_vs_TUDL=True, exact_cross_dataset_object_geometry_disjointness_verified=False,
        preregistered_future_method=protocol["future_frozen_method"], preregistered_future_evaluation=protocol["future_private_evaluation"])
    started = time.perf_counter(); error = None; path = private / "acquisition-report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Pinned T-LESS acquisition exceeded600s; no budget increase")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        oldmask = os.umask(0o077)
        try:
            persist(); manifest, selected = acquire(destination, report, persist)
            public_hashes, retained = output_inventory(destination, manifest, selected)
            if public_hashes != report["public_files"] or retained != report["retained_files"]: raise ValueError("Original retained outputs changed")
            report["retained_outputs_unchanged"] = True
        except Exception as caught: error = caught; report.update(error_type=type(caught).__name__, error=str(caught))
        finally:
            try:
                after = bound_source(root, code, revision, Path(__file__)); report["source_helpers_after"] = after
                if after != helpers or load_protocol(code) != (protocol, protocol_identity): raise ValueError("Frozen source/protocol changed")
                report["source_helpers_unchanged"] = True
            except Exception as caught:
                report.update(source_helpers_unchanged=False, source_recheck_error_type=type(caught).__name__, source_recheck_error=str(caught))
                if error is None: error = caught
            if error is None: report.update(status="pass", phase="complete")
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); os.umask(oldmask)
            persist(); path.chmod(0o400)
    if error is not None: raise error


if __name__ == "__main__": main()
