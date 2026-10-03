"""Frozen twelve-frame TUD-L object-depth comparison, never prediction refitting.

Independent public byte pins and the entire prediction contract are checked
before any private file is opened. Original scoring helpers retain camera XYZ,
sensor integer pixels and all visible object instances. A PASS receipt means
evaluation executed; the scientific decision remains separate and limited.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import stat
import time
import zipfile

import numpy as np
import tudl_anchor_infer as inference
import tudl_holdout_inputs as inputs
import tudl_evaluate as scoring

ROOT = Path("/srv/scenesmith/world-reward")
BASE = "validation/tudl_frame_holdout_v1"
PREDICTION_PINS = "configs/tudl_anchor_prediction_pins.json"
PINS_SCHEMA = "world-reward-tudl-anchor-prediction-pins-v1"
STAGE = "private_tudl_frame_holdout_border_anchor_camera_quality"
IMAGE_ID = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
BUDGET = 180
SOURCE_FILES = tuple(dict.fromkeys(("infra/tudl_anchor_evaluate.py", "infra/tudl_evaluate.py",
    "infra/run_tudl_anchor_evaluate.sh", *inference.SOURCE_FILES)))


def exact(value, expected):
    """JSON equality without bool/int or int/float substitutions."""
    if type(value) is not type(expected): return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(exact(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
    return value == expected


def require_fields(value, expected, reason):
    if type(value) is not dict or any(k not in value or not exact(value[k], v) for k, v in expected.items()):
        raise ValueError(reason)


def identity_record(value):
    inputs._identity_record(value)
    return value


def pinned_json(path, expected):
    identity_record(expected)
    if inputs.identity(path) != expected: raise ValueError("Independent SHA/bytes differ before JSON interpretation")
    data = path.read_bytes()
    if {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} != expected:
        raise ValueError("Pinned JSON changed before interpretation")
    result = inputs.strict_json(data)
    if inputs.identity(path) != expected: raise ValueError("Pinned JSON changed during interpretation")
    return result


def validate_prediction_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "report", "outputs", "coefficients"}
            or not exact(pins["schema"], PINS_SCHEMA)):
        raise ValueError("Independently committed exact prediction pins required")
    receipt = pins["report"]
    if type(receipt) is not dict or set(receipt) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Actual report/source/producer must be independently pinned")
    identity_record({k: receipt[k] for k in ("sha256", "bytes")})
    for key, length in (("producer_revision", 40), ("script_sha256", 64)):
        if type(receipt[key]) is not str or re.fullmatch(r"[0-9a-f]{%d}" % length, receipt[key]) is None:
            raise ValueError("Exact actual producer/source identity required")
    names = {Path(name).stem + ".npz" for name in inputs.filenames()}
    if type(pins["outputs"]) is not dict or set(pins["outputs"]) != names:
        raise ValueError("All twelve and only twelve NPZ byte identities required")
    for value in pins["outputs"].values(): identity_record(value)
    coefs = pins["coefficients"]
    if type(coefs) is not list or len(coefs) != 3 or any(type(v) is not float or not math.isfinite(v) or v <= 0 for v in coefs):
        raise ValueError("Exactly three finite positive scene coefficients required")


def source_identities(code):
    return {name: inputs.identity(code / name) for name in SOURCE_FILES}


def marker_identity(path):
    """Original host0644 markers are bind-mounted RO, not chmod-rewritten."""
    path = inputs._canonical(path); before = inputs._state(path)
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_nlink != 1:
        raise ValueError("Original regular nonsymlink dispatch marker required")
    data = path.read_bytes()
    if not data or inputs._state(path) != before: raise ValueError("Original dispatch marker changed during hashing")
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def bound_source(root, code, revision, executing):
    root, code = Path(root), Path(code)
    if (root != ROOT or root.resolve() != root or not root.is_dir()
            or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or code != root / "jobs" / revision / "run_tudl_anchor_evaluate/code"
            or code.resolve() != code or not code.is_dir() or Path(executing) != code / SOURCE_FILES[0]):
        raise ValueError("Actual immutable CPU evaluation dispatch/source namespace required")
    modules = {"infra/tudl_anchor_infer.py": inference, "infra/tudl_holdout_inputs.py": inputs,
        "infra/tudl_evaluate.py": scoring, "infra/tudl_infer.py": inference.moge_native,
        "infra/object_synthetic_observations.py": inference.moge_assets,
        "infra/da3_metric_infer.py": inference.da3, "infra/da3_metric_acquire.py": inference.da3.acquisition}
    if any(Path(module.__file__).resolve() != code / name for name, module in modules.items()):
        raise ValueError("All original imported input/numerical helpers must resolve inside frozen code")
    for function, name in ((scoring.sha256, "src/world_reward/data.py"),
            (inference.validate_camera_pointmap, "src/world_reward/pointmap.py"),
            (scoring.validate_camera_pointmap, "src/world_reward/pointmap.py")):
        if Path(function.__code__.co_filename).resolve() != code / name:
            raise ValueError("Original utility/pointmap implementation must resolve inside frozen code")
    markers = {}
    for name in ("revision", "source-sha256"):
        path = code.parent / name; markers[name] = marker_identity(path); data = path.read_bytes()
        if (name == "revision" and data != (revision + "\n").encode()
                or name == "source-sha256" and re.fullmatch(b"[0-9a-f]{64}\n", data) is None):
            raise ValueError("Exact original immutable dispatch markers required")
    return {"helpers": source_identities(code), "markers": markers}


def prediction_contract():
    values = dict(stage=inference.STAGE, status="pass", phase="complete", image_id=IMAGE_ID,
        budget_seconds=inference.BUDGET, network="none", private_truth_read=False, challenge_inputs_used=False,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], adoption_performed=False,
        training_overlap_verified=False, full_license_closure_verified=False, metric_scale_accuracy_verified=False,
        calibration_accuracy_verified=False, independent_scenes_or_objects=False, same_development_scenes_and_objects=True,
        human_or_object_geometry_changed=False, per_frame_anchor_fit=False, offset_fit=False,
        semantic_background_exclusion_performed=False, border_is_background_proxy_only=True,
        anchor_rule="median_of_four_per_image_median_MoGe_Z_div_DA3_Z_on_fixed_10percent_border_per_scene",
        border_definition="xx<64_or_xx>=576_or_yy<48_or_yy>=432",
        minimum_border_pairs=inference.MIN_BORDER_PAIRS, minimum_border_coverage=inference.MIN_BORDER_COVERAGE,
        fixed_camera_K=inference.K.tolist(), prediction_pixel_convention="pixel_centres_plus_0.5",
        candidate_validity="exact_unchanged_native_MoGe_validity_no_drop_or_expansion", apply_mask=False,
        force_projection=True, native_camera_support_verified=True, native_DA3_arguments=inference.da3.NATIVE_ARGUMENTS,
        checkpoint_states_loaded=inference.da3.STATE_COUNT, checkpoint_load_strict=True,
        native_sky_correction_unchanged=True, native_sky_threshold=.3, native_sky_quantile=.99,
        metric_depth_scaling_applied_once=True, metric_depth_factor_rule="actual_processed_focal_mean_divided_by_300",
        postprocess="metric_camera_Z_then_bilinear_align_corners_false_then_one_scene_anchor_then_fixed_K_plus_0.5",
        confidence_validity_exclusion=False, pointmap_geometry_filled=False, seed=0, RNG_seed_reset_before_each_forward=True,
        MoGe_model_released_before_DA3_load=True, outputs_completed=12, original_frame_coverage_verified=True,
        sources_assets_rechecked=True, source_assets_after_reverified=True, saved_arrays_reread_verified=True,
        baseline_MoGe_bytes_preserved=True, actual_native_inference=True)
    for backend in ("MoGe", "DA3"):
        for counter in ("attempts", "returns", "calls_completed"): values[backend + "_" + counter] = 12
    return values


def validate_asset_metadata(receipt, root):
    """Check frozen producer metadata only; do not mount/read model assets."""
    moge = receipt.get("MoGe_bindings"); assets = inference.moge_assets
    if type(moge) is not dict or set(moge) != {"acquisition_report", "model_asset", "model_source", "python_source_files",
            "source_directory", "focal_geometry_source_sha256"}: raise ValueError("Complete original MoGe source/model metadata required")
    identity_record(moge["acquisition_report"])
    require_fields(moge, {"model_asset": {"sha256": assets.MODEL_SHA, "bytes": assets.MODEL_BYTES},
        "focal_geometry_source_sha256": inference.moge_native.GEOMETRY_SHA}, "Pinned native MoGe model/focal source differs")
    files = moge["python_source_files"]
    if type(files) is not dict or not files: raise ValueError("Complete original MoGe Python source inventory required")
    for name, value in files.items():
        if (type(name) is not str or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
                or ".." in PurePosixPath(name).parts or not name.endswith(".py")):
            raise ValueError("Canonical relative Python source records required")
        identity_record(value)
    require_fields(files, {"model/v2.py": {**files.get("model/v2.py", {}), "sha256": assets.SOURCE_V2_SHA},
        "utils/geometry_torch.py": {**files.get("utils/geometry_torch.py", {}), "sha256": inference.moge_native.GEOMETRY_SHA}},
        "Original MoGe infer/focal source inventory differs")
    model_source = moge["model_source"]
    require_fields(model_source, {"revision": assets.SOURCE_REVISION, "python_files": len(files),
        "python_source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "v2_source_sha256": assets.SOURCE_V2_SHA, "full_dependency_license_closure_verified": False},
        "Original full MoGe source metadata differs")
    direct = model_source.get("direct_url")
    require_fields(direct, {"url": "https://github.com/microsoft/MoGe.git"}, "Original MoGe VCS source required")
    require_fields(direct.get("vcs_info"), {"vcs": "git", "commit_id": assets.SOURCE_REVISION}, "Original MoGe commit required")
    directory = moge["source_directory"]
    if type(directory) is not str or not Path(directory).is_absolute() or Path(directory).name != "moge":
        raise ValueError("Actual installed original MoGe source directory metadata required")
    da3 = receipt.get("DA3_assets"); acq = inference.da3.acquisition
    if type(da3) is not dict or set(da3) != {"acquisition_report", "source_manifest_sha256", "source_identity", "model_sha256"}:
        raise ValueError("Complete native DA3 asset metadata required")
    identity_record(da3["acquisition_report"])
    records = [dict(file=acq.SOURCE + "/" + name,
        url=f"https://raw.githubusercontent.com/{acq.SOURCE_REPO}/{acq.SOURCE_REV}/{name}",
        bytes=size, sha256=digest, git_blob_sha1=blob) for name, size, digest, blob in acq.SOURCE_RECORDS]
    entries = [{**row, "file": str(Path(row["file"]).relative_to(acq.SOURCE))} for row in records]
    manifest = dict(schema="world-reward-da3-metric-source-v1", source_revision=acq.SOURCE_REV, python_files=27,
        python_bytes=192452, source_bytes=225327, files=entries, source_modified=False,
        root_namespace_init_fabricated=False, high_level_api_acquired=False)
    require_fields(da3, {"source_identity": {"files": records}, "model_sha256": acq.MODEL_RECORDS[-1][2],
        "source_manifest_sha256": hashlib.sha256((json.dumps(manifest, indent=2) + "\n").encode()).hexdigest()},
        "Pinned unchanged native DA3 source/model metadata differs")
    wheel = receipt.get("DA3_dependency")
    expected_names = {"addict/__init__.py", "addict/addict.py", *["addict-2.4.0.dist-info/" + name
        for name in ("LICENSE", "METADATA", "WHEEL", "top_level.txt", "RECORD")]}
    if type(wheel) is not dict or set(wheel) != {"sha256", "bytes", "path", "import_origin", "source_sha256"}:
        raise ValueError("Complete unchanged native wheel metadata required")
    require_fields(wheel, {"sha256": inference.ADDICT_SHA, "bytes": 3832, "path": str(root / inference.ADDICT_PATH),
        "import_origin": str(root / inference.ADDICT_PATH) + "/addict/__init__.py"}, "Original pinned native wheel differs")
    if type(wheel["source_sha256"]) is not dict or set(wheel["source_sha256"]) != expected_names:
        raise ValueError("All original wheel source/license members required")
    if any(type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in wheel["source_sha256"].values()):
        raise ValueError("Exact wheel member SHA identities required")


def public_predictions(root, code):
    """Return all12 readonly arrays/records/frozen identities before private IO."""
    root, code = Path(root), Path(code)
    records, public = inference.load_public(root, code)
    pin_path = code / PREDICTION_PINS; pin_identity = inputs.identity(pin_path)
    pins = pinned_json(pin_path, pin_identity); validate_prediction_pins(pins)
    pred = root / BASE / "anchor_predictions_v1"
    if (pred.resolve() != pred or not pred.is_dir()
            or {p.name for p in pred.iterdir()} != {"report.json", *pins["outputs"]}):
        raise ValueError("Exactly twelve frozen predictions and one original report required")
    frozen = [(pin_path, pin_identity), (code / inference.PIN_FILE, public["input_pin_identity"]),
        (root / BASE / "inputs/manifest.json", public["input_manifest"])]
    frozen += [(row["path"], public["input_pins"]["public_files"][row["file"]]) for row in records]
    receipt_identity = {k: pins["report"][k] for k in ("sha256", "bytes")}
    # Check every output byte identity before decoding even the prediction report.
    for name, expected in pins["outputs"].items():
        if inputs.identity(pred / name) != expected: raise ValueError("Independent full prediction SHA/bytes differ")
        frozen.append((pred / name, expected))
    receipt = pinned_json(pred / "report.json", receipt_identity); frozen.append((pred / "report.json", receipt_identity))
    require_fields(receipt, prediction_contract(), "Complete GT-free actual public prediction contract required")
    helpers = {name: inputs.identity(code / name) for name in inference.SOURCE_FILES}
    require_fields(receipt, {"producer_revision": pins["report"]["producer_revision"],
        "script_sha256": pins["report"]["script_sha256"], "source_helpers": helpers,
        **public, "scene_anchor_coefficients": pins["coefficients"]}, "Actual public/source/producer/scene anchor binding differs")
    if pins["report"]["script_sha256"] != helpers[inference.SOURCE_FILES[0]]["sha256"]:
        raise ValueError("Actual independently pinned inference source differs from imported helper")
    frozen += [(code / name, identity) for name, identity in helpers.items()]
    validate_asset_metadata(receipt, root)
    outputs, calls = receipt.get("outputs"), receipt.get("native_focal_solver_calls")
    if type(outputs) is not list or len(outputs) != 12 or type(calls) is not list or len(calls) != 12:
        raise ValueError("All twelve ordered outputs and original native focal calls required")
    arrays, ratios = [], []
    output_fields = {"file", "scene_id", "frame_id", "sha256", "bytes", "rgb_sha256", "decoded_RGB_sha256", "arrays",
        "original_MoGe_arrays", "pointmap_checks", "scene_anchor", "border_proxy", "processed_camera_K", "metric_depth_factor"}
    for index, (record, row, call) in enumerate(zip(records, outputs, calls)):
        scene, frame = record["scene_id"], record["frame_id"]; name = Path(record["file"]).stem + ".npz"
        coefficient = pins["coefficients"][scene - 1]
        if type(row) is not dict or set(row) != output_fields: raise ValueError("Exact original prediction output metadata required")
        require_fields(row, {"file": name, "scene_id": scene, "frame_id": frame, **pins["outputs"][name],
            "rgb_sha256": record["sha256"], "scene_anchor": coefficient}, "Original ordered output identity differs")
        if type(row["decoded_RGB_sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["decoded_RGB_sha256"]) is None:
            raise ValueError("Original decoded RGB identity required")
        expected_call = dict(file=record["file"], scene_id=scene, frame_id=frame, **{"pass": "fixed_prior"},
            focal_prior_supplied=True, original_solver_returned=True)
        require_fields(call, expected_call, "Exact original native call order/support required")
        if (set(call) != {*expected_call, "native_nearest64_valid_pixels"}
                or type(call["native_nearest64_valid_pixels"]) is not int or not 2 <= call["native_nearest64_valid_pixels"] <= 4096):
            raise ValueError("Original native solver support required, no fallback")
        path = pred / name
        with zipfile.ZipFile(path) as archive:
            if (len(archive.namelist()) != len(inference.ARRAY_KEYS)
                    or set(archive.namelist()) != {key + ".npy" for key in inference.ARRAY_KEYS}
                    or sum(info.file_size for info in archive.infolist()) > 16_000_000):
                raise ValueError("Exact original bounded NPZ arrays required, no auxiliary truth")
        with np.load(path, allow_pickle=False) as archive: data = {key: archive[key].copy() for key in archive.files}
        checks = inference.validate_prediction_arrays(data, scene, frame, coefficient)
        original = {key: data["moge_" + key] for key in ("depth", "points", "validity")}; original["K"] = data["K"]
        ratio = inference.border_ratio(data["moge_depth"], data["moge_validity"], data["da3_depth"])
        ratio.update(scene_id=scene, frame_id=frame); ratios.append(ratio)
        require_fields(row, {"arrays": inference.array_identities(data), "original_MoGe_arrays": inference.array_identities(original),
            "pointmap_checks": checks, "border_proxy": ratio}, "Original array/geometry/border replay metadata differs")
        processed = np.asarray(row["processed_camera_K"], np.float64)
        factor = inference.da3.metric_factor(processed)
        if not exact(row["metric_depth_factor"], factor): raise ValueError("Original processed focal scaling differs")
        for value in data.values(): value.flags.writeable = False
        arrays.append(data)
    require_fields(receipt, {"ratio_diagnostics": ratios, "scene_ratio_medians":
        [[row["ratio_median"] for row in ratios[start:start + 4]] for start in (0, 4, 8)],
        "scene_anchor_coefficients": inference.scene_anchors(ratios)}, "Frozen median-of-four scene coefficient replay differs")
    recheck(frozen)
    hashes = {"prediction_report_identity": receipt_identity, "prediction_producer_revision": pins["report"]["producer_revision"],
        "prediction_script_sha256": pins["report"]["script_sha256"], "prediction_pin_identity": pin_identity,
        "input_pin_identity": public["input_pin_identity"], "public_manifest_sha256": public["input_manifest"]["sha256"],
        "scene_anchor_coefficients": pins["coefficients"], "prediction_assets_verified_from_frozen_receipt_only": True}
    return arrays, records, frozen, hashes


def recheck(frozen):
    for path, expected in frozen:
        if inputs.identity(path) != expected: raise ValueError("Original frozen artifact changed")


def private_inputs(private, records, manifest_sha):
    """Only called after all public predictions passed; hash ALL private first."""
    ap = private / "acquisition-report.json"; acquisition_identity = {k: inputs.ACQUISITION[k] for k in ("sha256", "bytes")}
    acquisition = pinned_json(ap, acquisition_identity)
    expected = dict(stage="external_tudl_rgb_only_frame_holdout_acquisition", status="pass", phase="complete",
        producer_revision=inputs.ACQUISITION["producer_revision"], script_sha256=inputs.ACQUISITION["script_sha256"],
        dataset_revision=inputs.REVISION, license=inputs.LICENSE, selection=inputs.SELECTION, selection_indices=[40, 80, 120, 160],
        development_selection_indices=[0, 100, 199], same_development_scenes_and_objects=True, independent_scenes_or_objects=False,
        temporal_adjacency_or_acceleration_truth_verified=False, device="cpu", gpu_used=False, challenge_inputs_used=False,
        inference_performed=False, ground_truth_used_for_inference=False, private_annotations_exported_as_inference_inputs=False,
        training_overlap_verified=False, challenge_overlap_verified=False, accuracy_verified=False,
        train_or_full_test_downloaded=False, bop19_subset_zip_downloaded=True, selection_before_private_annotation_values=True,
        archive_sources_unchanged=True, disposable_archives_removed=True, retained_outputs_unchanged=True,
        source_helpers_unchanged=True, original_frame_coverage_verified=True, development_frame_ids_disjoint=True,
        images_completed=12, public_manifest_sha256=manifest_sha,
        selected_records=[{k: row[k] for k in ("scene_id", "frame_id", "file")} | {"rgb_sha256": row["sha256"]} for row in records])
    require_fields(acquisition, expected, "Exact complete actual holdout private acquisition required")
    retained = acquisition.get("retained_files")
    if type(retained) is not list or not retained: raise ValueError("Complete private retained inventory required")
    names, frozen = [], [(ap, acquisition_identity)]
    for row in retained:
        if type(row) is not dict or set(row) != {"file", "sha256", "bytes"}: raise ValueError("Exact private file identities required")
        name = row["file"]
        if (type(name) is not str or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
                or ".." in PurePosixPath(name).parts or not name.startswith("source/")):
            raise ValueError("Canonical private original source paths required")
        identity = identity_record({k: row[k] for k in ("sha256", "bytes")}); path = private / name
        if inputs.identity(path) != identity or stat.S_IMODE(path.stat().st_mode) != 0o400:
            raise ValueError("Original readonly private file SHA/bytes/permissions differ")
        names.append(name); frozen.append((path, identity))
    if names != sorted(set(names)): raise ValueError("Complete sorted unique private source inventory required")
    if {p.name for p in private.iterdir()} != {"source", "acquisition-report.json"}:
        raise ValueError("Unexpected private artifacts")
    actual_names, directories = set(), {"source"}
    for path in (private / "source", *(private / "source").rglob("*")):
        if path.is_symlink(): raise ValueError("No private symlinks allowed")
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode) != 0o700: raise ValueError("Original private directories must be0700")
            directories.add(str(path.relative_to(private)))
        elif path.is_file(): actual_names.add(str(path.relative_to(private)))
        else: raise ValueError("Only original regular private files/directories allowed")
    if actual_names != set(names): raise ValueError("Private retained byte inventory differs")
    fixed_names = {"source/licenses/" + name for name in
        ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json", "attribution-holdout.json")}
    fixed_names.update("source/base/tudl/" + name for name in ("camera.json", "dataset_info.md", "test_targets_bop19.json"))
    fixed_names.update("source/" + folder + "/" + name for folder in ("models", "models_eval")
        for name in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply"))
    scenes = {}
    for scene, frame_ids in enumerate(inputs.FRAME_IDS, 1):
        prefix = f"source/test/{scene:06d}/"; annotations = {}
        for kind in ("camera", "gt", "gt_info"):
            name = prefix + f"scene_{kind}.json"; fixed_names.add(name)
            if name not in actual_names: raise ValueError("Original private annotation receipt missing")
            values = inputs.strict_json((private / name).read_bytes())
            if type(values) is not dict or set(values) != {str(frame) for frame in frame_ids}:
                raise ValueError("Exact four private selected frame keys required")
            annotations[kind] = values
        for frame in frame_ids:
            gt, info = annotations["gt"][str(frame)], annotations["gt_info"][str(frame)]
            if (type(gt) is not list or not gt or type(info) is not list or len(info) != len(gt)
                    or any(type(row) is not dict or type(row.get("obj_id")) is not int or row["obj_id"] not in (1, 2, 3) for row in gt)):
                raise ValueError("All original object instance/GTinfo records required")
            fixed_names.add(prefix + f"depth/{frame:06d}.png")
            fixed_names.update(prefix + f"mask_visib/{frame:06d}_{instance:06d}.png" for instance in range(len(gt)))
        scenes[scene] = annotations
    expected_dirs = {str(p) for name in fixed_names for p in PurePosixPath(name).parents if str(p) != "."}
    if actual_names != fixed_names or directories != expected_dirs:
        raise ValueError("Exact original full-instance private filenames/directories required")
    recheck(frozen)
    return scenes, frozen, acquisition_identity


def score_arrays(data):
    """Views of unchanged predictions, not new K, points, scales or alignment."""
    result = {"scene_id": data["scene_id"], "frame_id": data["frame_id"]}
    for old, new in (("moge", "fixed"), ("anchored", "learned")):
        for key in ("depth", "points", "validity"): result[new + "_" + key] = data[old + "_" + key]
        result[new + "_K"] = data["K"]
    return result


def decision(frames):
    ids = [(scene, frame) for scene, values in enumerate(inputs.FRAME_IDS, 1) for frame in values]
    if type(frames) is not list or len(frames) != 12 or [(r["scene_id"], r["frame_id"]) for r in frames] != ids:
        raise ValueError("All twelve preselected ordered frames required")
    scenes = []; coverage = True
    for scene in (1, 2, 3):
        rows = frames[(scene - 1) * 4:scene * 4]; means = {}
        for mode in ("fixed", "learned"):
            values = [row[mode]["sensor_visible_camera_chamfer_half_cm"] for row in rows]
            if any(type(v) is not float or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Complete finite nonnegative paired camera scores required")
            means[mode] = float(np.mean(values))
            for row in rows:
                value = row[mode]["sensor_valid_object_coverage"]
                if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Finite original per-frame object coverage required")
                coverage &= value >= .95
        if any(row["fixed"]["sensor_valid_object_coverage"] != row["learned"]["sensor_valid_object_coverage"]
                or row.get("exact_candidate_validity_matches_baseline") is not True for row in rows):
            raise ValueError("Candidate validity must be exactly unchanged baseline, not tolerance-based")
        scenes.append({"scene_id": scene, **means})
    before = np.array([row["fixed"] for row in scenes]); after = np.array([row["learned"] for row in scenes])
    gain = (before - after) / before if np.all(before > 0) else None
    supported = bool(gain is not None and coverage and np.median(gain) >= .05 and np.min(gain) >= -.05)
    return {"scenes": scenes, "per_scene_relative_gain": None if gain is None else gain.tolist(),
        "median_paired_scene_relative_gain": None if gain is None else float(np.median(gain)),
        "coverage_gate_pass": bool(coverage), "exact_candidate_validity_matches_baseline": True,
        "scientific_decision": "SUPPORT_NARROW_OBJECT_DEPTH_HYPOTHESIS" if supported else "REJECT",
        "real_object_camera_hypothesis_supported": supported, "adoption_performed": False,
        "training_overlap_verified": False, "independent_scenes_or_objects": False,
        "human_quality_verified": False, "temporal_quality_verified": False, "full_v2d_score_verified": False,
        "verified_victory_over_CARI4D": False}


def run(root, report, *, code=None):
    root = Path(root); code = Path(code) if code is not None else Path(os.environ["WR_CODE"])
    source_before = source_identities(code)
    arrays, records, frozen, hashes = public_predictions(root, code)
    report.update(**hashes, predictions_frozen_before_private_truth_read=True, frames=[])
    private = root / BASE / "eval_private"
    scenes, private_frozen, acquisition_identity = private_inputs(private, records, hashes["public_manifest_sha256"])
    report["phase"] = "private_object_sensor_scoring"
    for data, record in zip(arrays, records):
        scene, frame = record["scene_id"], record["frame_id"]; annotations = scenes[scene]; key = str(frame)
        value = annotations["camera"][key]
        if type(value) is not dict: raise ValueError("Original per-frame private camera metadata required")
        K = scoring.camera(np.asarray(value.get("cam_K"), np.float64).reshape(3, 3)); scale = value.get("depth_scale")
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("Original finite positive per-image sensor depth_scale required")
        directory = private / f"source/test/{scene:06d}"
        depth = scoring.read_png(directory / f"depth/{frame:06d}.png", depth=True) * float(scale) / 1000
        visible = np.zeros((inputs.HEIGHT, inputs.WIDTH), np.bool_)
        count = len(annotations["gt"][key])
        for instance in range(count): visible |= scoring.read_png(directory / f"mask_visib/{frame:06d}_{instance:06d}.png")
        unchanged = inference.array_identities(data)
        row = scoring.score_frame(score_arrays(data), depth, K, visible)
        if inference.array_identities(data) != unchanged: raise ValueError("Original prediction arrays changed during private scoring")
        row.update(scene_id=scene, frame_id=frame, object_instances_scored=count,
            exact_candidate_validity_matches_baseline=bool(np.array_equal(data["moge_validity"], data["anchored_validity"])),
            private_focal_error_diagnostic={mode: float(abs(data["K"][0, 0] - K[0, 0]) / K[0, 0]) for mode in ("fixed", "learned")})
        report["frames"].append(row)
    result = decision(report["frames"])
    recheck(frozen + private_frozen)
    if source_identities(code) != source_before: raise ValueError("Original full numerical/source closure changed during evaluation")
    report.update(decision=result, acquisition_report_identity=acquisition_identity, original_frame_coverage_verified=True,
        private_sensor_K_used_for_truth_only=True, private_GT_masks_used_for_evaluation_only=True,
        all_original_object_instances_scored=True, predicted_XYZ_camera_or_scale_changed=False,
        alignment_performed=False, oracle_ray_diagnostic_performed=False, GT_pixel_convention="integer_BOP",
        prediction_pixel_convention="unchanged_plus_0.5", training_overlap_excluded=False,
        sensor_noise_free_verified=False, human_quality_verified=False, temporal_quality_verified=False,
        full_v2d_score_verified=False, adoption_performed=False, frozen_artifacts_after_reverified=True,
        status="pass", phase="complete")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote Linux network-none CPU evaluation required")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    bound = bound_source(root, code, revision, Path(__file__))
    out = root / BASE / "quality_anchor_v1"
    if (image != IMAGE_ID or out.resolve() != out or not out.is_dir() or any(out.iterdir())
            or stat.S_IMODE(out.stat().st_mode) != 0o700):
        raise ValueError("Exact original VM02 image and exclusively reserved0700 quality output required")
    report = dict(stage=STAGE, status="fail", phase="integrity", producer_revision=revision, image_id=image,
        script_sha256=bound["helpers"][SOURCE_FILES[0]]["sha256"], source_helpers=bound["helpers"],
        dispatch_markers=bound["markers"], budget_seconds=BUDGET, network="none", device="cpu", gpu_used=False,
        challenge_inputs_used=False, external_private_sensor_truth_used=True, hand_labeled_test=False,
        oracle_modes=[], adoption_performed=False, full_v2d_score_verified=False)
    started = time.perf_counter(); error = None
    def expired(*_): raise TimeoutError("Whole private frozen holdout evaluation exceeded180s")
    alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try: run(root, report, code=code)
    except Exception as caught:
        error = caught; report.update(status="fail", error_type=type(caught).__name__, error=str(caught))
    finally:
        try:
            if bound_source(root, code, revision, Path(__file__)) != bound: raise ValueError("Original source/dispatch markers changed")
            report["source_markers_after_reverified"] = True
        except Exception as caught:
            report.update(status="fail", source_markers_after_reverified=False,
                source_recheck_error_type=type(caught).__name__, source_recheck_error=str(caught))
            if error is None: error = caught
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter() - started
        path = out / "report.json"
        with path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o400)
    if {p.name for p in out.iterdir()} != {"report.json"}: raise ValueError("Only one immutable private quality receipt permitted")
    if error is not None: raise error


if __name__ == "__main__": main()
