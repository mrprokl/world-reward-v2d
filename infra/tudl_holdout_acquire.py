"""Azure-only, filename-selected TUD-L frame holdout; no model inference.

Twelve RGBs are new frames of the SAME three development scenes/objects, not
independent scenes, objects or a verified training-unseen benchmark. All RGB
filenames are inspected before private annotation values. Private sensor depth,
calibration, masks and models are retained only for later sealed evaluation.
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
import time
import zipfile

import tudl_acquire as source

ROOT = Path("/srv/scenesmith/world-reward")
NAMESPACE = "validation/tudl_frame_holdout_v1"
STAGE = "external_tudl_rgb_only_frame_holdout_acquisition"
SCHEMA = "world-reward-tudl-frame-holdout-rgb-v1"
SELECTION = "sorted_RGB_indices_40_80_120_160_per_scene_before_private_values"
INDICES = (40, 80, 120, 160)
DEVELOPMENT_INDICES = (0, 100, 199)
DEVELOPMENT_IDS = {1: (0, 4074, 8227), 2: (3, 4013, 7710), 3: (4, 4028, 7969)}
HELPERS = ("infra/tudl_holdout_acquire.py", "infra/tudl_acquire.py")
BUDGET = 600


def identity(path, readonly=True):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Canonical nonsymlink absolute file required")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or (readonly and before.st_mode & 0o222):
        raise ValueError("Nonempty readonly regular original file required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    after = path.lstat()
    fields = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if fields(before) != fields(after): raise ValueError("File changed while hashing")
    return {"sha256": digest.hexdigest(), "bytes": after.st_size}


def bound_source(root, code, revision, executing):
    root, code = Path(root), Path(code)
    if (type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or root != ROOT or not root.is_dir() or root.resolve() != root
            or code != root / "jobs" / revision / "run_tudl_holdout_acquire" / "code"
            or not code.is_dir() or code.resolve() != code
            or any(p.is_symlink() for p in (code, *code.parents))
            or Path(executing) != code / HELPERS[0]
            or Path(source.__file__).resolve() != code / HELPERS[1]):
        raise ValueError("Actual immutable holdout dispatch/source namespace required")
    return {name: identity(code / name) for name in HELPERS}


def select_rgb_names(members):
    """Use filename keys only; never inspect ZIP/annotation member values."""
    if (type(members) is not dict or any(type(name) is not str for name in members)
            or source.EXPECTED_IDS != DEVELOPMENT_IDS
            or set(INDICES) & set(DEVELOPMENT_INDICES)):
        raise ValueError("Pinned filename inventory and disjoint fixed selection required")
    # This verifies exactly200 names/scene and all original development IDs.
    original = source.select_rgb_names(members)
    selected = []
    for scene in (1, 2, 3):
        files = sorted(name for name in members if re.fullmatch(fr"test/{scene:06d}/rgb/[0-9]{{6}}\.png", name))
        for index in INDICES:
            name = files[index]
            frame = int(PurePosixPath(name).stem)
            if frame in DEVELOPMENT_IDS[scene]: raise ValueError("Development frame reused")
            selected.append((scene, frame, name))
    if len(selected) != 12 or len(set(selected)) != 12 or set(selected) & set(original):
        raise ValueError("Exactly twelve disjoint original frame identities required")
    return selected


def validate_manifest(manifest, selected):
    expected = {"schema": SCHEMA, "revision": source.REVISION, "license": "CC-BY-SA-4.0", "selection": SELECTION}
    if (type(manifest) is not dict or set(manifest) != {*expected, "images"}
            or any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in expected.items())
            or type(manifest["images"]) is not list or len(manifest["images"]) != 12
            or type(selected) is not list or len(selected) != 12
            or any(type(row) is not tuple or len(row) != 3 or type(row[0]) is not int
                or type(row[1]) is not int or type(row[2]) is not str for row in selected)):
        raise ValueError("Exact twelve-RGB holdout manifest required")
    for record, (scene, frame, _) in zip(manifest["images"], selected):
        if (type(record) is not dict or set(record) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(record.get(k)) is not int for k in ("scene_id", "frame_id", "width", "height"))
                or (record["scene_id"], record["frame_id"], record["width"], record["height"]) != (scene, frame, 640, 480)
                or record["file"] != f"scene_{scene:06d}_frame_{frame:06d}.png"
                or type(record["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None):
            raise ValueError("Original RGB-only frame/grid/SHA contract differs")


def retain_subset(archives, private, inputs, selected):
    # Recheck the COMPLETE filename selection before the first private read,
    # including when this function is used directly instead of through main.
    if (type(selected) is not list or any(type(row) is not tuple or len(row) != 3
            or type(row[0]) is not int or type(row[1]) is not int or type(row[2]) is not str for row in selected)
            or selected != select_rgb_names(archives["tudl_test_bop19.zip"][1])):
        raise ValueError("Complete filename-only holdout selection must precede private values")
    oldmask = os.umask(0o077)
    try:
        original = source.retain_subset(archives, private, inputs, selected)
    finally:
        os.umask(oldmask)
    manifest = {"schema": SCHEMA, "revision": source.REVISION, "license": "CC-BY-SA-4.0",
                "selection": SELECTION, "images": original["images"]}
    validate_manifest(manifest, selected)
    attribution = {"dataset": "TU Dresden Light (TUD-L)", "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Hodan, Michel et al., BOP: Benchmark for 6D Object Pose Estimation, ECCV2018",
        "upstream_modified": "Filename-only twelve-RGB frame holdout; selected annotation dictionaries filtered, all instances retained",
        "same_development_scenes_and_objects": True,
        "independent_scenes_or_objects": False,
        "original_nine_RGB_attribution_retained_as_source_evidence": True}
    source.save_bytes(private / "source/licenses/attribution-holdout.json", (json.dumps(attribution, indent=2) + "\n").encode())
    return manifest


def output_inventory(destination, manifest, selected):
    """Complete retained public/private file inventory; no media/array decoding."""
    inputs, private = destination / "inputs", destination / "eval_private"
    validate_manifest(manifest, selected)
    if {p.name for p in destination.iterdir()} != {"inputs", "eval_private"}:
        raise ValueError("No artifacts outside the split public/private namespace")
    for directory, mode in ((inputs, 0o755), (private, 0o700)):
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o777 != mode:
            raise ValueError("Strict original public/private directory permissions required")
    public_names = {"manifest.json", *[r["file"] for r in manifest["images"]]}
    if {p.name for p in inputs.iterdir()} != public_names:
        raise ValueError("Public inputs contain only twelve RGBs and their manifest")
    public = {name: identity(inputs / name) for name in sorted(public_names)}
    if any((inputs / name).stat().st_mode & 0o777 != 0o444 for name in public_names):
        raise ValueError("Original public RGB/manifest files must be0444")
    for record in manifest["images"]:
        if public[record["file"]]["sha256"] != record["sha256"]: raise ValueError("Public RGB changed")
    if json.loads((inputs / "manifest.json").read_text()) != manifest:
        raise ValueError("Saved public manifest does not reread identically")
    expected = {"source/licenses/" + name for name in
        ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json", "attribution-holdout.json")}
    expected.update("source/base/tudl/" + name for name in ("camera.json", "dataset_info.md", "test_targets_bop19.json"))
    expected.update("source/" + folder + "/" + name for folder in ("models", "models_eval")
        for name in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply"))
    for scene in (1, 2, 3):
        prefix = f"source/test/{scene:06d}/"
        frames = [frame for s, frame, _ in selected if s == scene]
        for kind in ("camera", "gt", "gt_info"):
            name = prefix + f"scene_{kind}.json"; expected.add(name)
            values = json.loads((private / name).read_text())
            if type(values) is not dict or set(values) != {str(frame) for frame in frames}:
                raise ValueError("Only selected private annotation keys may be retained")
        gt = json.loads((private / (prefix + "scene_gt.json")).read_text())
        for frame in frames:
            if type(gt[str(frame)]) is not list or not gt[str(frame)]: raise ValueError("All original private object instances required")
            expected.add(prefix + f"depth/{frame:06d}.png")
            expected.update(prefix + f"mask_visib/{frame:06d}_{instance:06d}.png" for instance in range(len(gt[str(frame)])))
    if {p.name for p in private.iterdir()} != {"source", "acquisition-report.json"}:
        raise ValueError("Disposable archives/extras must not survive in private output")
    names, directories = set(), {"source"}
    expected_directories = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    if (private / "source").stat().st_mode & 0o777 != 0o700:
        raise ValueError("Every private source directory must be0700")
    for path in (private / "source").rglob("*"):
        if path.is_symlink(): raise ValueError("No symlink in retained private inventory")
        if path.is_dir():
            if path.stat().st_mode & 0o777 != 0o700: raise ValueError("Every private source directory must be0700")
            directories.add(str(path.relative_to(private)))
        elif path.is_file(): names.add(str(path.relative_to(private)))
        else: raise ValueError("Only regular private files and directories required")
    if names != expected or directories != expected_directories:
        raise ValueError("Exact complete selected private source inventory required")
    retained = []
    for name in sorted(expected):
        path = private / name
        if path.stat().st_mode & 0o777 != 0o400: raise ValueError("Private source files must be0400")
        retained.append({"file": name, **identity(path)})
    return public, retained


def acquire(destination, report, persist):
    private, inputs = destination / "eval_private", destination / "inputs"
    downloads = private / ".downloads"
    (private / "source").mkdir(mode=0o700)
    downloads.mkdir(mode=0o700)
    archives = {}; observed = {}; budget = [0, 0]
    try:
        report["license_evidence"] = source.license_evidence(private); persist()
        for name, (size, sha) in source.ARCHIVES.items():
            report.update(phase="download", active_archive=name); persist()
            path = downloads / name
            source.download(source.BASE_URL + name, path, size, sha)
            observed[name] = identity(path)
            if observed[name] != {"bytes": size, "sha256": sha}: raise ValueError("Pinned archive bytes/SHA differ")
            archive = zipfile.ZipFile(path); archives[name] = (archive, {})
            members = source.zip_inventory(archive, budget); source.inspect_layout(name, members)
            archives[name] = (archive, members)
            report["archives"].append({"file": name, "url": source.BASE_URL + name,
                **observed[name], "members": len(members)}); persist()
            if name == "tudl_test_bop19.zip": report["bop19_subset_zip_downloaded"] = True
        members = archives["tudl_test_bop19.zip"][1]
        selected = select_rgb_names(members)
        rgb_names = sorted(name for name in members if re.fullmatch(r"test/00000[123]/rgb/[0-9]{6}\.png", name))
        report.update(phase="retain_subset", selection_before_private_annotation_values=True,
            rgb_filename_inventory_count=len(rgb_names),
            rgb_filename_inventory_sha256=hashlib.sha256(("\n".join(rgb_names) + "\n").encode()).hexdigest(),
            selected_archive_records=[{"scene_id": scene, "frame_id": frame, "archive_rgb_file": name,
                "sorted_rgb_index": INDICES[index % 4]} for index, (scene, frame, name) in enumerate(selected)])
        persist()
        manifest = retain_subset(archives, private, inputs, selected)
        source.save_bytes(inputs / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o444)
        for name in observed:
            if identity(downloads / name) != observed[name]: raise ValueError("Original archive changed during extraction")
        report["archive_sources_unchanged"] = True
    finally:
        try:
            for archive, _ in archives.values(): archive.close()
        finally:
            shutil.rmtree(downloads)
            report["disposable_archives_removed"] = not downloads.exists()
    public, retained = output_inventory(destination, manifest, selected)
    report.update(public_files=public, retained_files=retained,
        public_manifest_sha256=public["manifest.json"]["sha256"],
        selected_records=[{k: row[k] for k in ("scene_id", "frame_id", "file")} | {"rgb_sha256": row["sha256"]}
            for row in manifest["images"]],
        images_completed=12, expanded_bytes_inspected=budget[0],
        original_frame_coverage_verified=True, development_frame_ids_disjoint=True)
    report.pop("active_archive", None)
    return manifest, selected


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("Holdout acquisition stays on Azure Linux")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision = os.environ["WR_CODE_REVISION"]
    helpers = bound_source(root, code, revision, Path(__file__))
    destination = root / NAMESPACE
    if (os.environ.get("WR_TUDL_HOLDOUT_RESERVED") != "1" or not destination.is_dir()
            or destination.resolve() != destination or any(p.is_symlink() for p in (destination, *destination.parents))
            or any(destination.iterdir())):
        raise ValueError("Exclusively reserved empty canonical holdout required")
    private = destination / "eval_private"; private.mkdir(mode=0o700)
    (destination / "inputs").mkdir(mode=0o755)
    report = {"stage": STAGE, "status": "fail", "phase": "licence", "producer_revision": revision,
        "script_sha256": helpers[HELPERS[0]]["sha256"], "source_helpers": helpers,
        "dataset_revision": source.REVISION, "license": "CC-BY-SA-4.0", "selection": SELECTION,
        "selection_indices": list(INDICES), "development_selection_indices": list(DEVELOPMENT_INDICES),
        "development_scene_frame_ids": {str(scene): list(ids) for scene, ids in DEVELOPMENT_IDS.items()},
        "same_development_scenes_and_objects": True, "independent_scenes_or_objects": False,
        "temporal_adjacency_or_acceleration_truth_verified": False,
        "budget_seconds": BUDGET, "device": "cpu", "gpu_used": False, "archives": [],
        "challenge_inputs_used": False, "inference_performed": False, "ground_truth_used_for_inference": False,
        "private_annotations_exported_as_inference_inputs": False,
        "training_overlap_verified": False, "challenge_overlap_verified": False,
        "accuracy_verified": False, "train_or_full_test_downloaded": False,
        "bop19_subset_zip_downloaded": False}
    started = time.perf_counter(); path = private / "acquisition-report.json"; error = None
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
            stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Pinned TUD-L holdout acquisition exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(BUDGET)
        oldmask = os.umask(0o077)
        try:
            persist(); manifest, selected = acquire(destination, report, persist)
            public, retained = output_inventory(destination, manifest, selected)
            if public != report["public_files"] or retained != report["retained_files"]:
                raise ValueError("Immutable retained outputs changed during final reread")
            report["retained_outputs_unchanged"] = True
        except Exception as caught:
            error = caught; report.update(error_type=type(caught).__name__, error=str(caught))
        finally:
            try:
                after = bound_source(root, code, revision, Path(__file__))
                report["source_helpers_after"] = after
                if after != helpers: raise ValueError("Frozen producer/acquisition helpers changed")
                report["source_helpers_unchanged"] = True
            except Exception as caught:
                report.update(source_recheck_error_type=type(caught).__name__, source_recheck_error=str(caught),
                    source_helpers_unchanged=False)
                if error is None:
                    error = caught; report.update(error_type=type(caught).__name__, error=str(caught))
            if error is None: report.update(status="pass", phase="complete")
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
            os.umask(oldmask)
            persist(); path.chmod(0o400)
    if error is not None: raise error


if __name__ == "__main__": main()
