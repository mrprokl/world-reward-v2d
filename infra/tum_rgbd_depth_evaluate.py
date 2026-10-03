"""Frozen TUM scalar sensor-Z evaluation; no private camera or prediction fit.

All public bytes, twelve predictions, original source/model metadata and all
private identities pass before any uint16 depth decode. Scoring uses genuine
frozen scalar quality primitives; no TUD-L private scorer is called.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import time
import zipfile

import numpy as np
import tum_rgbd_depth_infer as inference
import tum_depth_quality as quality
import tum_rgbd_depth_acquire as acquisition_source
import tudl_anchor_evaluate as original_audit

inputs, contract = inference.inputs, inference.contract
ROOT, BASE, IMAGE_ID = inference.ROOT, inference.BASE, inference.IMAGE_ID
PINS = "configs/tum_rgbd_depth_prediction_pins.json"
PROTOCOL = "configs/tum_rgbd_depth_protocol.json"
PROTOCOL_ID = dict(bytes=21939, sha256="ed1f038546ac073d2b52f01874934a6683e05357f9730cbac1a5112930ded035")
STAGE = "private_tum_rgbd_scalar_depth_quality"
BUDGET = 180
QUALITY_SHA = "09d097a5377027f7eebfe167c0ea2d1a45e75af2a02a8dd8a240076dec1f1745"
SOURCE_FILES = tuple(dict.fromkeys(("infra/tum_rgbd_depth_evaluate.py", "infra/run_tum_rgbd_depth_evaluate.sh",
    "infra/tum_depth_quality.py", "infra/tum_rgbd_depth_acquire.py", "infra/run_tum_rgbd_depth_acquire.sh",
    "infra/tudl_anchor_evaluate.py", "infra/tudl_evaluate.py", "infra/run_tudl_anchor_evaluate.sh",
    "infra/run_tudl_anchor_evaluate_v2.sh", *inference.SOURCE_FILES)))
exact, require = original_audit.exact, original_audit.require_fields


def identity_record(value):
    inputs._record(value)
    return value


def pinned_json(path, expected):
    identity_record(expected)
    if inputs.identity(path) != expected: raise ValueError("Independent bytes differ before JSON")
    raw = path.read_bytes()
    if {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} != expected:
        raise ValueError("Pinned JSON changed before interpretation")
    result = inputs.strict_json(raw)
    if inputs.identity(path) != expected: raise ValueError("Pinned JSON changed during interpretation")
    return result


def validate_pins(pins, inputs_pins):
    if type(pins) is not dict or set(pins) != {"schema", "report", "outputs", "coefficients"} or pins["schema"] != "world_reward.tum_rgbd_depth_prediction_pins.v1":
        raise ValueError("Independent complete whole-support prediction pins required")
    row = pins["report"]
    if type(row) is not dict or set(row) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Independent actual producer/source/report identity required")
    identity_record({k: row[k] for k in ("sha256", "bytes")})
    for k, n in (("producer_revision", 40), ("script_sha256", 64)):
        if type(row[k]) is not str or re.fullmatch(r"[0-9a-f]{%d}" % n, row[k]) is None:
            raise ValueError("Actual producer/source digest required")
    names = {Path(name).stem + ".npz" for name in inputs.filenames(inputs_pins)}
    if type(pins["outputs"]) is not dict or set(pins["outputs"]) != names: raise ValueError("Exactly twelve frozen predictions required")
    for value in pins["outputs"].values(): identity_record(value)
    coeff = pins["coefficients"]
    if type(coeff) is not list or len(coeff) != 3 or any(type(v) is not float or not math.isfinite(v) or v <= 0 for v in coeff):
        raise ValueError("Exactly three positive finite frozen scene coefficients required")


def prediction_fields(pins, public):
    fields = dict(stage=inference.STAGE, status="pass", phase="complete", image_id=IMAGE_ID,
        producer_revision=pins["report"]["producer_revision"], script_sha256=pins["report"]["script_sha256"],
        budget_seconds=inference.BUDGET, network="none", private_truth_read=False, challenge_inputs_used=False,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], adoption_performed=False,
        training_overlap_verified=False, full_license_closure_verified=False, calibration_accuracy_verified=False,
        metric_scale_accuracy_verified=False, new_TUM_recordings_relative_to_TUDL=True, different_object_identity_fully_proven=False,
        sparse_early_frames_not_complete_temporal_validation=True,
        human_contact_or_full_HOI_evaluated=False, human_or_object_geometry_changed=False, per_frame_anchor_fit=False,
        offset_fit=False, semantic_mask_or_confidence_weight_used=False, dispersion_recorded_not_filtered=True,
        support_diagnostics_recorded_before_gate=True,
        support_rule="median_of_four_per_image_median_MoGe_Z_div_native_DA3_Z_on_all_original_MoGe_valid_pixels_per_scene",
        minimum_pairs=contract.MIN_PAIRS, minimum_grid_coverage=contract.MIN_COVERAGE, fixed_camera_K=inference.GRID.K.tolist(),
        prediction_pixel_convention="pixel_centres_plus_0.5", candidate_validity="exact_unchanged_native_MoGe_validity_no_drop_or_expansion",
        apply_mask=False, force_projection=True, native_camera_support_verified=True,
        native_DA3_arguments=inference.native.da3.NATIVE_ARGUMENTS, checkpoint_states_loaded=inference.native.da3.STATE_COUNT,
        checkpoint_load_strict=True, native_sky_correction_unchanged=True, native_sky_threshold=.3, native_sky_quantile=.99,
        metric_depth_scaling_applied_once=True, metric_depth_factor_rule="actual_processed_focal_mean_divided_by_300",
        postprocess="metric_camera_Z_then_bilinear_align_corners_false_then_one_scene_anchor_then_fixed_K_plus_0.5",
        confidence_validity_exclusion=False, pointmap_geometry_filled=False, seed=0, RNG_seed_reset_before_each_forward=True,
        MoGe_model_released_before_DA3_load=True, outputs_completed=12, original_frame_coverage_verified=True,
        sources_assets_rechecked=True, source_assets_after_reverified=True, saved_arrays_reread_verified=True,
        baseline_MoGe_bytes_preserved=True, actual_native_inference=True, scene_anchor_coefficients=pins["coefficients"], **public)
    for backend in ("MoGe", "DA3"):
        for counter in ("attempts", "returns", "calls_completed"): fields[backend + "_" + counter] = 12
    return fields


def read_npz(path, keys, maximum):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if (len(entries) != len(keys) or {e.filename for e in entries} != {k + ".npy" for k in keys}
                or sum(e.file_size for e in entries) > maximum or any(e.flag_bits & 1 for e in entries)):
            raise ValueError("Exact bounded original NPZ arrays required")
    with np.load(path, allow_pickle=False) as archive: return {k: archive[k].copy() for k in archive.files}


def recheck(frozen):
    if any(inputs.identity(path) != expected for path, expected in frozen): raise ValueError("Frozen input changed")


def public_predictions(root, code):
    records, public = inference.load_public(root, code)
    pp = code / PINS; pins = pinned_json(pp, inputs.identity(pp)); validate_pins(pins, public["input_pins"])
    pred = root / BASE / inference.OUTPUT
    if inputs._canonical(pred) != pred or {p.name for p in pred.iterdir()} != {"report.json", *pins["outputs"]}:
        raise ValueError("Exact frozen complete prediction namespace required")
    frozen = [(pp, inputs.identity(pp)), (code / inference.PIN_FILE, public["input_pin_identity"])]
    frozen += [(root / BASE / "inputs" / name, value) for name, value in public["input_pins"]["public_files"].items()]
    frozen += [(pred / name, expected) for name, expected in pins["outputs"].items()]
    recheck(frozen)  # ALL outputs before even the public prediction JSON.
    rid = {k: pins["report"][k] for k in ("sha256", "bytes")}
    receipt = pinned_json(pred / "report.json", rid); frozen.append((pred / "report.json", rid))
    require(receipt, prediction_fields(pins, public), "Complete frozen blind native prediction contract required")
    helpers = {name: inputs.identity(code / name) for name in inference.SOURCE_FILES}
    require(receipt, {"source_helpers": helpers}, "Same frozen numerical producer helpers required")
    if any(helpers[name]["sha256"] != digest for name, digest in inference.FROZEN_HELPERS.items()):
        raise ValueError("Original native/grid numerical bytes differ")
    if helpers[inference.SOURCE_FILES[0]]["sha256"] != pins["report"]["script_sha256"]:
        raise ValueError("Independent producer script differs")
    original_audit.validate_asset_metadata(receipt, root)  # Metadata only, NO model mounts/loads.
    rows, calls = receipt.get("outputs"), receipt.get("native_focal_solver_calls")
    if type(rows) is not list or len(rows) != 12 or type(calls) is not list or len(calls) != 12:
        raise ValueError("Every original output and native support call required")
    arrays, ratios = [], []
    output_fields = {"file", "scene_id", "frame_id", "sha256", "bytes", "rgb_sha256", "decoded_RGB_sha256", "arrays",
        "original_MoGe_arrays", "pointmap_checks", "scene_anchor", "whole_support", "processed_camera_K", "metric_depth_factor"}
    for record, row, call in zip(records, rows, calls):
        if type(row) is not dict or set(row) != output_fields: raise ValueError("Exact original output metadata required")
        _, decoded = inference.read_rgb(record)
        require(row, {"decoded_RGB_sha256": decoded}, "Frozen decoded public RGB identity differs")
        scene, frame = record["scene_id"], record["frame_id"]; name = Path(record["file"]).stem + ".npz"
        coefficient = pins["coefficients"][scene - 1]
        require(row, dict(file=name, scene_id=scene, frame_id=frame, **pins["outputs"][name], rgb_sha256=record["sha256"], scene_anchor=coefficient), "Original ordered prediction differs")
        require(call, dict(file=record["file"], scene_id=scene, frame_id=frame, **{"pass": "fixed_prior"}, focal_prior_supplied=True, original_solver_returned=True), "Original focal call order differs")
        if set(call) != {"file", "scene_id", "frame_id", "pass", "focal_prior_supplied", "original_solver_returned", "native_nearest64_valid_pixels"} or type(call.get("native_nearest64_valid_pixels")) is not int or not 2 <= call["native_nearest64_valid_pixels"] <= 4096:
            raise ValueError("Original native64 support required")
        data = read_npz(pred / name, contract.ARRAY_KEYS, 16_000_000)
        checks = contract.validate_prediction_arrays(inference.GRID, data, scene, frame, coefficient)
        ratio = contract.whole_support_ratio(inference.GRID, data["moge_depth"], data["moge_validity"], data["da3_depth"])
        ratio.update(scene_id=scene, frame_id=frame); contract.require_support(ratio); ratios.append(ratio)
        original = {k: data["moge_" + k] for k in ("depth", "points", "validity")}; original["K"] = data["K"]
        require(row, dict(arrays=contract.array_identities(data), original_MoGe_arrays=contract.array_identities(original), pointmap_checks=checks, whole_support=ratio), "Original arrays/geometry/whole-support metadata differs")
        processed = np.asarray(row.get("processed_camera_K"), np.float64)
        if not exact(row.get("metric_depth_factor"), contract.metric_factor(inference.GRID, processed)):
            raise ValueError("Original native metric focal scaling differs")
        for value in data.values(): value.flags.writeable = False
        arrays.append(data)
    coefficients = contract.scene_anchors(inference.GRID, ratios, list(inputs.ordered_frames(public["input_pins"])))
    require(receipt, dict(whole_support_diagnostics=ratios, scene_anchor_coefficients=list(coefficients.values()),
        scene_ratios=[[r["ratio_median"] for r in ratios[start:start + 4]] for start in (0, 4, 8)]), "Frozen equal-weight median4 rule differs")
    recheck(frozen)
    return arrays, records, frozen, public, pins


def private_inputs(root, code, public, records):
    """Audit whole acquisition and private bytes before any sensor PNG decode."""
    protocol_path = code / PROTOCOL; protocol = pinned_json(protocol_path, PROTOCOL_ID)
    private = root / BASE / "eval_private"; expected = public["input_pins"]["acquisition_report"]
    ap = private / "acquisition-report.json"; rid = {key: expected[key] for key in ("sha256", "bytes")}
    acquisition = pinned_json(ap, rid)
    require(acquisition, dict(stage=acquisition_source.STAGE, status="pass", phase="complete",
        producer_revision=expected["producer_revision"], script_sha256=expected["script_sha256"],
        source_helpers={name: inputs.identity(code / name) for name in acquisition_source.SOURCE_FILES},
        protocol_identity=PROTOCOL_ID, budget_seconds=600, device="cpu", gpu_used=False, inference_performed=False,
        challenge_inputs_used=False, source_camera_or_trajectory_read=False, depth_values_decoded=False,
        ground_truth_used_for_inference=False, independent_full_archive_SHA256_known=False,
        training_overlap_verified=False, challenge_overlap_verified=False, accuracy_verified=False,
        all24_original_files_hashed_before_depth_values=True, all24_retained_bytes_rehashed_before_public_manifest=True,
        disposable_archives_and_staging_removed=True, outputs_rehashed_after=True, source_helpers_rehashed_after=True,
        images_completed=12, private_depths_completed=12), "Complete original licensed/byte-pinned TUM acquisition required")
    if inputs.identity(code / acquisition_source.SOURCE_FILES[0])["sha256"] != expected["script_sha256"]:
        raise ValueError("Actual independent acquisition script differs")
    required, depth_rows, selected = {}, [], []
    for sequence in protocol["sequences"]:
        for declared in sequence["selected"]:
            scene, rank = sequence["sequence_id"], declared["rgb_zero_based_rank"]
            record = records[(scene - 1) * 4 + (40, 80, 120, 160).index(rank)]
            require(record, dict(scene_id=scene, frame_id=rank, timestamp=declared["rgb_timestamp"],
                original_rgb_file=declared["rgb"]["path"], sha256=declared["rgb"]["sha256"]), "Original primary RGB selection differs")
            for kind in ("rgb", "depth"):
                timestamp = declared[kind + "_timestamp"]; name = f"scene_{scene:06d}_{kind}_{timestamp}.png"
                identity = {key: declared[kind][key] for key in ("bytes", "sha256")}; identity_record(identity)
                row = dict(file=name, kind=kind, original_file=sequence["name"] + "/" + declared[kind]["path"],
                    scene_id=scene, frame_id=rank, timestamp=timestamp, **identity)
                selected.append(row); required[("inputs/" if kind == "rgb" else "eval_private/") + name] = identity
                if kind == "rgb" and public["input_pins"]["public_files"].get(name) != identity:
                    raise ValueError("Independent public RGB bytes differ from original primary LFS pin")
                if kind == "depth": depth_rows.append(row)
    actual_selected = acquisition.get("selected_records")
    if (type(actual_selected) is not list or len(actual_selected) != 24
            or any(type(row) is not dict or set(row) != {"file", "kind", "original_file", "scene_id", "frame_id", "timestamp", "bytes", "sha256"} for row in actual_selected)
            or len({row.get("file") for row in actual_selected if type(row) is dict}) != 24
            or not exact(sorted(actual_selected, key=lambda row: row["file"]), sorted(selected, key=lambda row: row["file"]))):
        raise ValueError("Every original selected RGB/depth rank, timestamp, path and independent pin required")
    for name, identity in public["input_pins"]["public_files"].items():
        required["inputs/" + name] = identity
    archives = acquisition.get("archives")
    if type(archives) is not list or len(archives) != 3: raise ValueError("Three original acquisition archives required")
    expected_source = {"source/TUM-license-section.html", "source/attribution.json",
        *["source/" + sequence["name"] + "-description.html" for sequence in protocol["sequences"]]}
    license_rows = acquisition.get("license_evidence")
    if type(license_rows) is not list or len(license_rows) != 5: raise ValueError("Complete original primary license/description inventory required")
    for row in license_rows:
        if type(row) is not dict or set(row) != {"file", "bytes", "sha256"} or row["file"] not in expected_source:
            raise ValueError("Exact primary license record required")
        required["eval_private/" + row["file"]] = identity_record({key: row[key] for key in ("bytes", "sha256")})
    if {row["file"] for row in license_rows} != expected_source: raise ValueError("Primary license source missing or duplicate")
    for archive, sequence in zip(archives, protocol["sequences"]):
        require(archive, dict(sequence_id=sequence["sequence_id"], url=sequence["archive"]["url"], bytes=sequence["archive"]["bytes"],
            sha256_independently_preknown=False, sha256_is_first_observed_reproducibility_digest=True,
            nearest_timestamp_pairs_verified=True, selected8_independent_byte_pins_verified=True), "Original archive provenance/selection gates required")
        identity_record({key: archive[key] for key in ("bytes", "sha256")})
        inventories = archive.get("filename_inventories")
        if type(inventories) is not dict or set(inventories) != {"rgb", "depth"}: raise ValueError("Complete filename inventories required")
        for kind in ("rgb", "depth"):
            row = inventories[kind]
            if type(row) is not dict or set(row) != {"files", "sha256"} or type(row["files"]) is not int or row["files"] != sequence[kind + "_file_count"]:
                raise ValueError("Original complete RGB/depth filename counts differ")
            if type(row["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None: raise ValueError("Original filename inventory SHA required")
        terms = archive.get("term_files")
        if type(terms) is not list or len(terms) > 16 or archive.get("archive_license_files_present") is not bool(terms):
            raise ValueError("Original bounded archive terms metadata required")
        for index, term in enumerate(terms):
            name = f"source/sequence_{sequence['sequence_id']:06d}_archive_term_{index:02d}.txt"
            if type(term) is not dict or set(term) != {"file", "original_file", "bytes", "sha256"} or term["file"] != name:
                raise ValueError("Exact original archive term identity required")
            expected_source.add(name); required["eval_private/" + name] = identity_record({key: term[key] for key in ("bytes", "sha256")})
    output_files = acquisition.get("output_files")
    if not exact(output_files, required): raise ValueError("Complete output inventory must equal original24 pins plus license/manifest")
    frozen = [(ap, rid), (protocol_path, PROTOCOL_ID)]
    for name, identity in required.items():
        path = root / BASE / name
        if name.startswith("eval_private/") and stat.S_IMODE(path.stat().st_mode) != 0o400: raise ValueError("Every original private file must be0400")
        frozen.append((path, identity))
    recheck(frozen)  # ALL private identities before even license interpretation, let alone sensor decode.
    if {path.name for path in private.iterdir()} != {"source", "acquisition-report.json", *[row["file"] for row in depth_rows]}:
        raise ValueError("Only original12 private sensor files plus source/receipt allowed")
    if stat.S_IMODE(private.stat().st_mode) != 0o700 or stat.S_IMODE((private / "source").stat().st_mode) != 0o700:
        raise ValueError("Private directories must remain0700")
    if {"source/" + path.name for path in (private / "source").iterdir()} != expected_source:
        raise ValueError("Exact primary license and archive-term source tree required")
    for row in license_rows:
        raw = (private / row["file"]).read_bytes()
        if row["file"] == "source/TUM-license-section.html" and {key: row[key] for key in ("bytes", "sha256")} != {key: protocol["primary"]["license_section"][key] for key in ("bytes", "sha256")}:
            raise ValueError("Pinned original CC-BY4 license evidence differs")
        elif row["file"].endswith("-description.html"):
            name = row["file"].removeprefix("source/").removesuffix("-description.html")
            if {key: row[key] for key in ("bytes", "sha256")} != acquisition_source.DESCRIPTION_IDS.get(name):
                raise ValueError("Original selected sequence descriptions differ")
        elif row["file"] == "source/attribution.json":
            require(inputs.strict_json(raw), dict(dataset="TUM RGB-D", license="CC-BY-4.0",
                license_url=protocol["primary"]["license_section"]["license_url"], attribution=protocol["primary"]["attribution"],
                modification="Filename-only selected original RGB/depth subset; PNG bytes unmodified",
                publisher_url=protocol["primary"]["dataset_url"], sequence_specific_terms_checked=True), "Original attribution required")
    for archive in archives:
        for term in archive["term_files"]: acquisition_source.archive_terms((private / term["file"]).read_bytes())
    recheck(frozen)
    return depth_rows, frozen, rid


def read_sensor_png(path):
    """Original uint16 PNG only; no conversion, correction, resizing or fill."""
    from PIL import Image
    before = inputs.identity(path)
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != quality.SHAPE[::-1] or image.mode not in ("I;16", "I;16L", "I;16B", "I"):
            raise ValueError("Original TUM16bit sensor PNG required")
        values = np.asarray(image).copy()
    # Pillow commonly decodes 16-bit PNG as I/int32. Exact bounded conversion,
    # verified by the original IHDR, preserves every unsigned16 sensor value.
    with path.open("rb") as stream: header = stream.read(33)
    acquisition_source.png_header(header, "depth")
    if values.dtype not in (np.uint16, np.int32) or values.shape != quality.SHAPE or np.any(values < 0) or np.any(values > 65535):
        raise ValueError("Every original unsigned16 sensor value must be retained")
    raw = values.astype(np.uint16, copy=False)
    if not np.array_equal(raw, values) or inputs.identity(path) != before: raise ValueError("Original sensor values/bytes changed during decode")
    return raw


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    if (root != ROOT or code != root / "jobs" / revision / "run_tum_rgbd_depth_evaluate/code" or inputs._canonical(code) != code
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None or Path(__file__) != code / SOURCE_FILES[0]
            or os.environ["WR_IMAGE_ID"] != IMAGE_ID or platform.system() != "Linux"
            or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}):
        raise ValueError("Actual immutable CPU-only isolated private evaluation required")
    modules = {"infra/tum_rgbd_depth_infer.py": inference, "infra/tum_rgbd_depth_inputs.py": inputs,
        "infra/whole_support_anchor_contract.py": contract, "infra/tum_depth_quality.py": quality,
        "infra/tum_rgbd_depth_acquire.py": acquisition_source, "infra/tudl_anchor_evaluate.py": original_audit}
    if any(Path(module.__file__).resolve() != code / name for name, module in modules.items()): raise ValueError("Genuine frozen helper paths required")
    helpers = {name: inputs.identity(code / name) for name in SOURCE_FILES}; out = root / BASE / "quality_anchor_v1"
    if helpers["infra/tum_depth_quality.py"]["sha256"] != QUALITY_SHA:
        raise ValueError("Original frozen scalar quality helper bytes changed")
    if inputs._canonical(out) != out or not out.is_dir() or any(out.iterdir()): raise ValueError("Fresh exclusive evaluation output required")
    report = dict(stage=STAGE, status="fail", phase="public_validation", producer_revision=revision, image_id=IMAGE_ID,
        script_sha256=helpers[SOURCE_FILES[0]]["sha256"], source_helpers=helpers, budget_seconds=BUDGET,
        network="none", device="cpu", gpu_used=False, prediction_refit=False, alignment=False, challenge_inputs_used=False,
        private_truth_used_for_evaluation_only=True, private_camera_read=False, source_trajectory_read=False)
    started = time.perf_counter(); frozen = []; error = None
    def expired(*_): raise TimeoutError("Whole private scalar TUM evaluation exceeded180s")
    alarm, term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        arrays, records, frozen, public, pins = public_predictions(root, code)
        report.update(phase="private_evaluation", predictions_frozen_before_private_truth_read=True,
            prediction_report={key: pins["report"][key] for key in ("bytes", "sha256")}, scene_anchor_coefficients=pins["coefficients"])
        depths, private_frozen, acquisition_id = private_inputs(root, code, public, records); frozen += private_frozen
        rows = []
        for data, record, depth in zip(arrays, records, depths):
            if (depth["scene_id"], depth["frame_id"]) != (record["scene_id"], record["frame_id"]): raise ValueError("Original ordered sensor pair differs")
            unchanged = contract.array_identities(data); raw = read_sensor_png(root / BASE / "eval_private" / depth["file"])
            row = quality.score_frame(data["moge_depth"], data["anchored_depth"], data["moge_validity"], data["anchored_validity"], raw)
            if contract.array_identities(data) != unchanged: raise ValueError("Frozen prediction mutated during scoring")
            row.update(scene_id=record["scene_id"], frame_id=record["frame_id"], RGB_timestamp=record["timestamp"], depth_timestamp=depth["timestamp"]); rows.append(row)
        report.update(frames=rows, decision=quality.decision(rows), acquisition_report_identity=acquisition_id,
            status="pass", phase="complete", scalar_sensor_Z_only=True, sensor_factor=5000,
            private_camera_or_alignment_used=False, all_original12_records_scored=True)
    except Exception as exc: error = exc; report.update(error_type=type(exc).__name__, error=str(exc)[:300])
    finally:
        try:
            recheck(frozen)
            if {name: inputs.identity(code / name) for name in SOURCE_FILES} != helpers: raise ValueError("Evaluation source changed")
            report["all_frozen_artifacts_rechecked"] = True
        except Exception as exc: report.update(status="fail", postcheck_error_type=type(exc).__name__, postcheck_error=str(exc)[:300]); error = exc
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter() - started; path = out / "report.json"
        with path.open("x") as stream: json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o400)
    if error is not None: raise error


if __name__ == "__main__": main()
