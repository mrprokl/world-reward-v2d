"""Azure-only new TUD-L frame cohort for a frozen whole-valid-support anchor.

Only filename-selected RGBs enter future inference. These are new records of
previously observed scenes/objects, not unseen geometry or training clearance.
Native stdlib download/ZIP/retention/hash primitives stay genuinely unchanged.
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
import tudl_holdout_acquire as prior

ROOT = Path("/srv/scenesmith/world-reward")
NAMESPACE = "validation/tudl_whole_support_holdout_v1"
STAGE = "external_tudl_rgb_only_whole_support_holdout_acquisition"
SCHEMA = "world-reward-tudl-whole-support-holdout-rgb-v1"
SELECTION = "sorted_RGB_indices_30_70_110_150_per_scene_before_private_values"
INDICES, BUDGET = (30, 70, 110, 150), 600
PROTOCOL = "configs/tudl_whole_support_protocol.json"
PROTOCOL_SHA = "cb2d194e042ddd3893a9c5531c436635f5cbb57fa54497f69b1136a2886c1b1c"
DEVELOPMENT_IDS = {1: (0, 4074, 8227), 2: (3, 4013, 7710), 3: (4, 4028, 7969)}
PRIOR_IDS = {1: (1788, 3235, 5138, 6925), 2: (1566, 3137, 4894, 6490), 3: (1512, 3227, 4819, 6647)}
HELPERS = ("infra/tudl_whole_support_acquire.py", "infra/run_tudl_whole_support_acquire.sh",
    "infra/tudl_acquire.py", "infra/tudl_holdout_acquire.py")
FROZEN = {HELPERS[2]: "a923ce3602653827d547b76c3445b72ea14f34bcc3e4db4a8d02d77eab864b61",
    HELPERS[3]: "1cc7e7e80e37a5134e2d282d99106b5b28bf5e268476e85805598c8d3b5af6a8"}
identity = prior.identity


def bound_source(root, code, revision, executing):
    root, code = Path(root), Path(code)
    if (root != ROOT or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or not root.is_dir() or not code.is_dir() or root.resolve() != root or code.resolve() != code
            or code != root / "jobs" / revision / "run_tudl_whole_support_acquire/code"
            or any(path.is_symlink() for path in (code, *code.parents)) or Path(executing) != code / HELPERS[0]
            or Path(source.__file__).resolve() != code / HELPERS[2] or Path(prior.__file__).resolve() != code / HELPERS[3]
            or prior.source is not source or identity is not prior.identity):
        raise ValueError("Actual immutable whole-support producer and genuine source primitives required")
    helpers = {name: identity(code / name) for name in HELPERS}
    if any(helpers[name]["sha256"] != digest for name, digest in FROZEN.items()):
        raise ValueError("Original acquired source helpers must remain byte-identical")
    receipt = identity(code / PROTOCOL)
    if receipt != {"bytes": 5217, "sha256": PROTOCOL_SHA}:
        raise ValueError("Entire preregistered acquisition/method/evaluation protocol changed")
    protocol = json.loads((code / PROTOCOL).read_bytes())
    if (source.REVISION != protocol["revision"] or source.ARCHIVES != {
            name: (row["bytes"], row["sha256"]) for name, row in protocol["archives"].items()}):
        raise ValueError("Genuine original dataset revision/archive inventory differs from preregistration")
    if identity(code / PROTOCOL) != receipt: raise ValueError("Frozen protocol changed while reading")
    return helpers, receipt, protocol


def select_rgb_names(members):
    """Inspect filename keys alone; verify every previously observed rank/ID."""
    if (type(members) is not dict or any(type(name) is not str for name in members)
            or source.EXPECTED_IDS != DEVELOPMENT_IDS or prior.DEVELOPMENT_IDS != DEVELOPMENT_IDS
            or prior.INDICES != (40, 80, 120, 160) or INDICES != (30, 70, 110, 150)
            or set(INDICES) & {0, 100, 199, 40, 80, 120, 160}):
        raise ValueError("Pinned new and excluded filename ranks required; no selection tuning")
    original = source.select_rgb_names(members); old = prior.select_rgb_names(members); selected = []
    for scene in (1, 2, 3):
        files = sorted(name for name in members if re.fullmatch(fr"test/{scene:06d}/rgb/[0-9]{{6}}\.png", name))
        if tuple(row[1] for row in old if row[0] == scene) != PRIOR_IDS[scene]:
            raise ValueError("Previous observed frame identity inventory changed")
        for index in INDICES:
            name = files[index]; frame = int(PurePosixPath(name).stem)
            if frame in (*DEVELOPMENT_IDS[scene], *PRIOR_IDS[scene]): raise ValueError("Previously observed record reused")
            selected.append((scene, frame, name))
    if len(set(selected)) != 12 or set(selected) & {*original, *old}:
        raise ValueError("Exactly twelve new disjoint filename-selected records required")
    return selected


def validate_manifest(manifest, selected):
    expected = dict(schema=SCHEMA, revision=source.REVISION, license="CC-BY-SA-4.0", selection=SELECTION)
    if (type(manifest) is not dict or set(manifest) != {*expected, "images"}
            or any(type(manifest.get(key)) is not type(value) or manifest[key] != value for key, value in expected.items())
            or type(manifest["images"]) is not list or len(manifest["images"]) != 12
            or type(selected) is not list or len(selected) != 12):
        raise ValueError("Exact twelve public RGB manifest required; no private fields")
    for row, record in zip(manifest["images"], selected):
        if type(record) is not tuple or len(record) != 3 or any(type(record[k]) is not int for k in (0, 1)):
            raise ValueError("Exact original selected filename identity required")
        scene, frame, _ = record
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(row.get(key)) is not int for key in ("scene_id", "frame_id", "width", "height"))
                or (row["scene_id"], row["frame_id"], row["width"], row["height"]) != (scene, frame, 640, 480)
                or row["file"] != f"scene_{scene:06d}_frame_{frame:06d}.png"
                or type(row["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("Original RGB-only frame/order/grid/identity changed")


def retain_subset(archives, private, inputs, selected):
    if (type(selected) is not list or any(type(row) is not tuple or len(row) != 3
            or type(row[0]) is not int or type(row[1]) is not int or type(row[2]) is not str for row in selected)
            or selected != select_rgb_names(archives["tudl_test_bop19.zip"][1])):
        raise ValueError("Complete new filename-only cohort must precede every private read")
    oldmask = os.umask(0o077)
    try: original = source.retain_subset(archives, private, inputs, selected)
    finally: os.umask(oldmask)
    manifest = dict(schema=SCHEMA, revision=source.REVISION, license="CC-BY-SA-4.0", selection=SELECTION, images=original["images"])
    validate_manifest(manifest, selected)
    attribution = dict(dataset="TU Dresden Light (TUD-L)", license="CC-BY-SA-4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="Hodan, Michel et al., BOP: Benchmark for 6D Object Pose Estimation, ECCV2018",
        upstream_modified="New filename-only twelve-RGB whole-support cohort; selected annotation dictionaries filtered, all instances retained",
        same_development_scenes_and_objects=True, independent_scenes_or_objects=False,
        original_nine_RGB_attribution_retained_as_source_evidence=True)
    source.save_bytes(private / "source/licenses/attribution-whole-support.json", (json.dumps(attribution, indent=2) + "\n").encode())
    return manifest


def output_inventory(destination, manifest, selected):
    inputs, private = destination / "inputs", destination / "eval_private"; validate_manifest(manifest, selected)
    if {path.name for path in destination.iterdir()} != {"inputs", "eval_private"}:
        raise ValueError("Only public/private split may survive acquisition")
    for directory, mode in ((inputs, 0o755), (private, 0o700)):
        if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o777 != mode:
            raise ValueError("Strict public0755/private0700 permissions required")
    names = {"manifest.json", *[row["file"] for row in manifest["images"]]}
    if {path.name for path in inputs.iterdir()} != names: raise ValueError("Only thirteen public RGB/manifest files allowed")
    public = {name: identity(inputs / name) for name in sorted(names)}
    if any((inputs / name).stat().st_mode & 0o777 != 0o444 for name in names): raise ValueError("Public files must be0444")
    if any(public[row["file"]]["sha256"] != row["sha256"] for row in manifest["images"]): raise ValueError("Original RGB hash changed")
    if json.loads((inputs / "manifest.json").read_bytes()) != manifest: raise ValueError("Saved manifest changed")
    expected = {"source/licenses/" + name for name in ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json", "attribution-whole-support.json")}
    expected.update("source/base/tudl/" + name for name in ("camera.json", "dataset_info.md", "test_targets_bop19.json"))
    expected.update("source/" + folder + "/" + name for folder in ("models", "models_eval") for name in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply"))
    for scene in (1, 2, 3):
        prefix = f"source/test/{scene:06d}/"; frames = [frame for s, frame, _ in selected if s == scene]; annotations = {}
        for kind in ("camera", "gt", "gt_info"):
            name = prefix + f"scene_{kind}.json"; expected.add(name); annotations[kind] = json.loads((private / name).read_bytes())
            if type(annotations[kind]) is not dict or set(annotations[kind]) != {str(frame) for frame in frames}:
                raise ValueError("Only selected complete private annotation keys allowed")
        for frame in frames:
            gt, info = (annotations[kind][str(frame)] for kind in ("gt", "gt_info"))
            if type(gt) is not list or not gt or type(info) is not list or len(info) != len(gt):
                raise ValueError("All original object instances and corresponding metadata required")
            expected.add(prefix + f"depth/{frame:06d}.png")
            expected.update(prefix + f"mask_visib/{frame:06d}_{instance:06d}.png" for instance in range(len(gt)))
    if {path.name for path in private.iterdir()} != {"source", "acquisition-report.json"}: raise ValueError("Private temporary/extras survived")
    tree = private / "source"; actual, directories = set(), {"source"}
    if tree.stat().st_mode & 0o777 != 0o700: raise ValueError("Private directories must be0700")
    for path in tree.rglob("*"):
        if path.is_symlink(): raise ValueError("Private source symlink forbidden")
        if path.is_dir():
            if path.stat().st_mode & 0o777 != 0o700: raise ValueError("Private directories must be0700")
            directories.add(str(path.relative_to(private)))
        elif path.is_file():
            if path.stat().st_mode & 0o777 != 0o400: raise ValueError("Private files must be0400")
            actual.add(str(path.relative_to(private)))
        else: raise ValueError("Only regular retained files/directories allowed")
    expected_dirs = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    if actual != expected or directories != expected_dirs: raise ValueError("Exact complete private source inventory required")
    return public, [{"file": name, **identity(private / name)} for name in sorted(expected)]


def acquire(destination, report, persist):
    private, inputs = destination / "eval_private", destination / "inputs"; downloads = private / ".downloads"
    (private / "source").mkdir(mode=0o700); downloads.mkdir(mode=0o700)
    archives, observed, budget = {}, {}, [0, 0]
    try:
        report["license_evidence"] = source.license_evidence(private); persist()
        for name, (size, digest) in source.ARCHIVES.items():
            report.update(phase="download", active_archive=name); persist(); path = downloads / name
            source.download(source.BASE_URL + name, path, size, digest); observed[name] = identity(path)
            if observed[name] != {"bytes": size, "sha256": digest}: raise ValueError("Original archive hash/bytes differ")
            archive = zipfile.ZipFile(path); archives[name] = (archive, {})
            members = source.zip_inventory(archive, budget); source.inspect_layout(name, members); archives[name] = (archive, members)
            report["archives"].append(dict(file=name, url=source.BASE_URL + name, **observed[name], members=len(members))); persist()
        members = archives["tudl_test_bop19.zip"][1]; selected = select_rgb_names(members)
        names = sorted(name for name in members if re.fullmatch(r"test/00000[123]/rgb/[0-9]{6}\.png", name))
        report.update(phase="retain_subset", selection_before_private_annotation_values=True,
            rgb_filename_inventory_count=len(names), rgb_filename_inventory_sha256=hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest(),
            selected_archive_records=[dict(scene_id=scene, frame_id=frame, archive_rgb_file=name, sorted_rgb_index=INDICES[index % 4])
                for index, (scene, frame, name) in enumerate(selected)]); persist()
        manifest = retain_subset(archives, private, inputs, selected)
        source.save_bytes(inputs / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o444)
        if any(identity(downloads / name) != observed[name] for name in observed): raise ValueError("Archive source changed during retention")
        report["archive_sources_unchanged"] = True
    finally:
        try:
            for archive, _ in archives.values(): archive.close()
        finally:
            shutil.rmtree(downloads); report["disposable_archives_removed"] = not downloads.exists()
    public, retained = output_inventory(destination, manifest, selected)
    report.update(public_files=public, retained_files=retained, images_completed=12,
        public_manifest_sha256=public["manifest.json"]["sha256"], expanded_bytes_inspected=budget[0], original_frame_coverage_verified=True,
        all_previously_observed_frame_ids_disjoint=True, selected_records=[{key: row[key] for key in ("scene_id", "frame_id", "file")} | {"rgb_sha256": row["sha256"]} for row in manifest["images"]])
    report.pop("active_archive", None); return manifest, selected


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("External data acquisition stays on Azure Linux")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    helpers, protocol_id, protocol = bound_source(root, code, revision, Path(__file__)); destination = root / NAMESPACE
    if (os.environ.get("WR_TUDL_WHOLE_SUPPORT_RESERVED") != "1" or not destination.is_dir() or destination.resolve() != destination
            or any(path.is_symlink() for path in (destination, *destination.parents)) or any(destination.iterdir())):
        raise ValueError("Exclusively reserved empty canonical whole-support cohort required")
    private = destination / "eval_private"; private.mkdir(mode=0o700); (destination / "inputs").mkdir(mode=0o755)
    report = dict(stage=STAGE, status="fail", phase="license", producer_revision=revision,
        script_sha256=helpers[HELPERS[0]]["sha256"], source_helpers=helpers, protocol_identity=protocol_id, protocol=protocol,
        dataset_revision=source.REVISION, license="CC-BY-SA-4.0", selection=SELECTION, selection_indices=list(INDICES),
        development_scene_frame_ids=DEVELOPMENT_IDS, prior_holdout_scene_frame_ids=PRIOR_IDS, archives=[],
        same_development_scenes_and_objects=True, independent_scenes_or_objects=False, budget_seconds=BUDGET, device="cpu", gpu_used=False,
        challenge_inputs_used=False, inference_performed=False, ground_truth_used_for_inference=False,
        private_annotations_exported_as_inference_inputs=False, training_overlap_verified=False, challenge_overlap_verified=False,
        accuracy_verified=False, train_or_full_test_downloaded=False, temporal_adjacency_or_acceleration_truth_verified=False)
    started = time.perf_counter(); path = private / "acquisition-report.json"; error = None
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Pinned new TUD-L whole-support acquisition exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET); oldmask = os.umask(0o077)
        try:
            persist(); manifest, selected = acquire(destination, report, persist)
            if output_inventory(destination, manifest, selected) != (report["public_files"], report["retained_files"]): raise ValueError("Frozen retained outputs changed")
            report["retained_outputs_unchanged"] = True
        except Exception as caught: error = caught; report.update(error_type=type(caught).__name__, error=str(caught))
        finally:
            try:
                after = bound_source(root, code, revision, Path(__file__))
                if after != (helpers, protocol_id, protocol): raise ValueError("Frozen acquisition source/protocol changed")
                report.update(source_helpers_after=after[0], source_helpers_unchanged=True)
            except Exception as caught:
                report.update(source_helpers_unchanged=False, source_recheck_error_type=type(caught).__name__, source_recheck_error=str(caught))
                if error is None: error = caught; report.update(error_type=type(caught).__name__, error=str(caught))
            if error is None: report.update(status="pass", phase="complete")
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); os.umask(oldmask)
            persist(); path.chmod(0o400)
    if error is not None: raise error


if __name__ == "__main__": main()
