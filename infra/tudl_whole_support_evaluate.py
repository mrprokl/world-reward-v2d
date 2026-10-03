"""Private evaluation of independently frozen new whole-support TUD-L predictions.

All public bytes,12predictions, native source/model metadata pass before private
sensor calibration/depth/visible-instance masks. Integer BOP truth rays remain
unchanged; predicted +.5 camera pointmaps are never realigned or refitted.
"""
import argparse
import hashlib
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
import tudl_whole_support_infer as inference
import tudl_anchor_evaluate as original_audit

inputs, contract = inference.inputs, inference.contract
ROOT, BASE, IMAGE_ID = inference.ROOT, inference.BASE, inference.IMAGE_ID
PINS = "configs/tudl_whole_support_prediction_pins.json"
STAGE = "private_tudl_whole_support_camera_quality"
BUDGET, SAMPLES = 180, 8192
SOURCE_FILES = tuple(dict.fromkeys(("infra/tudl_whole_support_evaluate.py", "infra/run_tudl_whole_support_evaluate.sh",
    "infra/tudl_anchor_evaluate.py", "infra/tudl_evaluate.py", "infra/run_tudl_anchor_evaluate.sh",
    "infra/run_tudl_anchor_evaluate_v2.sh", "infra/tudl_whole_support_acquire.py",
    "infra/run_tudl_whole_support_acquire.sh", "infra/tudl_holdout_acquire.py", "infra/tudl_acquire.py", *inference.SOURCE_FILES)))
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
    if type(pins) is not dict or set(pins) != {"schema", "report", "outputs", "coefficients"} or pins["schema"] != "world_reward.tudl_whole_support_prediction_pins.v1":
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
        metric_scale_accuracy_verified=False, new_frame_holdout_same_development_scenes_and_objects=True,
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
    # All frozen predictions/asset metadata have already passed before private IO.
    private = root / BASE / "eval_private"; expected = public["input_pins"]["acquisition_report"]
    rid = {k: expected[k] for k in ("bytes", "sha256")}; ap = private / "acquisition-report.json"
    acquisition = pinned_json(ap, rid)
    require(acquisition, dict(stage="external_tudl_rgb_only_whole_support_holdout_acquisition", status="pass", phase="complete",
        producer_revision=expected["producer_revision"], script_sha256=expected["script_sha256"],
        dataset_revision=inputs.REVISION, license=inputs.LICENSE, selection=inputs.SELECTION, selection_indices=[30,70,110,150],
        same_development_scenes_and_objects=True, independent_scenes_or_objects=False, device="cpu", gpu_used=False,
        challenge_inputs_used=False, inference_performed=False, ground_truth_used_for_inference=False,
        private_annotations_exported_as_inference_inputs=False, training_overlap_verified=False, challenge_overlap_verified=False,
        selection_before_private_annotation_values=True, archive_sources_unchanged=True, disposable_archives_removed=True,
        source_helpers_unchanged=True, images_completed=12, public_manifest_sha256=public["input_manifest"]["sha256"],
        public_files=public["input_pins"]["public_files"]), "Successful actual new filename-selected acquisition required")
    render_names = ("infra/tudl_whole_support_acquire.py", "infra/run_tudl_whole_support_acquire.sh", "infra/tudl_acquire.py", "infra/tudl_holdout_acquire.py")
    require(acquisition, {"source_helpers": {n:inputs.identity(code/n) for n in render_names}}, "Same frozen acquisition source required")
    if inputs.identity(code/render_names[0])["sha256"] != expected["script_sha256"]:
        raise ValueError("Independent actual acquisition script differs")
    protocol_path = code / "configs/tudl_whole_support_protocol.json"; protocol_id=inputs.identity(protocol_path)
    protocol=pinned_json(protocol_path,protocol_id)
    require(acquisition, {"protocol_identity": protocol_id}, "Same complete preregistration required")
    require(protocol["future_private_evaluation"], dict(samples_per_frame=8192,
        sampling="all_if_count<=8192_else_sorted_PCG64_seed0_choice_without_replacement",minimum_visible_object_pixels=32,
        minimum_median_relative_gain=.05,maximum_any_scene_relative_regression=.05,
        minimum_each_frame_baseline_and_candidate_object_coverage=.95,exact_candidate_validity_matches_baseline=True,no_alignment=True),
        "Original preregistered private quality gates required")
    require(protocol["frozen_method"], dict(minimum_native_valid_pairs=1024,minimum_native_valid_full_grid_coverage=.25,
        candidate_validity="exact unchanged native MoGe validity; no drop or expansion",per_frame_anchor_fit=False,offset_fit=False,
        alignment_performed=False,private_camera_used_for_prediction=False), "Frozen method support/gauge required")
    retained=acquisition.get("retained_files")
    if type(retained) is not list or not retained:raise ValueError("Complete original private byte inventory required")
    frozen=[(ap,rid),(protocol_path,protocol_id)];names=[]
    for row in retained:
        if type(row) is not dict or set(row)!={"file","sha256","bytes"}:raise ValueError("Exact original private identity records")
        name=row["file"]
        if (type(name) is not str or PurePosixPath(name).is_absolute() or str(PurePosixPath(name))!=name
                or ".." in PurePosixPath(name).parts or not name.startswith("source/")):raise ValueError("Canonical original private path")
        identity=identity_record({k:row[k] for k in ("sha256","bytes")});path=private/name
        if stat.S_IMODE(path.stat().st_mode)!=0o400:raise ValueError("Original readonly private files0400")
        names.append(name);frozen.append((path,identity))
    recheck(frozen)  # ALL private byte identities before ANY annotation values.
    if names!=sorted(set(names)) or {p.name for p in private.iterdir()}!={"source","acquisition-report.json"}:
        raise ValueError("Exact sorted complete private namespace")
    actual=set();directories=set()
    for path in (private/"source",*(private/"source").rglob("*")):
        if path.is_symlink():raise ValueError("Private symlinks forbidden")
        if path.is_dir():
            if stat.S_IMODE(path.stat().st_mode)!=0o700:raise ValueError("Original private dirs0700")
            directories.add(str(path.relative_to(private)))
        elif path.is_file():actual.add(str(path.relative_to(private)))
        else:raise ValueError("Only regular private source")
    fixed={"source/licenses/"+n for n in ("huggingface-README.md","BOP-TUD-L-section.html","attribution.json","attribution-whole-support.json")}
    fixed.update("source/base/tudl/"+n for n in ("camera.json","dataset_info.md","test_targets_bop19.json"))
    fixed.update("source/"+folder+"/"+n for folder in ("models","models_eval") for n in ("models_info.json","obj_000001.ply","obj_000002.ply","obj_000003.ply"))
    frames=inputs.ordered_frames(public["input_pins"]);scenes={}
    for scene in (1,2,3):
        frame_ids=[f for s,f in frames if s==scene];prefix=f"source/test/{scene:06d}/";annotations={}
        for kind in ("camera","gt","gt_info"):
            name=prefix+f"scene_{kind}.json";fixed.add(name)
            if name not in actual:raise ValueError("Original private annotation missing")
            values=inputs.strict_json((private/name).read_bytes())
            if type(values) is not dict or set(values)!={str(f) for f in frame_ids}:raise ValueError("Exact selected private4frames")
            annotations[kind]=values
        for frame in frame_ids:
            gt,info=annotations["gt"][str(frame)],annotations["gt_info"][str(frame)]
            if (type(gt) is not list or not gt or type(info) is not list or len(info)!=len(gt)
                    or any(type(r) is not dict or type(r.get("obj_id")) is not int or r["obj_id"] not in (1,2,3) for r in gt)):
                raise ValueError("All original object instances required")
            fixed.add(prefix+f"depth/{frame:06d}.png")
            fixed.update(prefix+f"mask_visib/{frame:06d}_{i:06d}.png" for i in range(len(gt)))
        scenes[scene]=annotations
    expected_dirs={str(p) for n in fixed for p in PurePosixPath(n).parents if str(p)!="."}
    if actual!=fixed or actual!=set(names) or directories!=expected_dirs:raise ValueError("Complete original all-instance inventory")
    recheck(frozen)
    return scenes,frozen,rid


def score_arrays(data):
    return original_audit.score_arrays(data)


def decision(rows, ordered_frames):
    if type(rows) is not list or [(r["scene_id"],r["frame_id"]) for r in rows]!=list(ordered_frames) or len(rows)!=12:
        raise ValueError("All twelve filename-selected records must contribute")
    scenes=[];coverage=True
    for scene in (1,2,3):
        frames=rows[(scene-1)*4:scene*4];means={}
        for mode in ("fixed","learned"):
            values=[r[mode]["sensor_visible_camera_chamfer_half_cm"] for r in frames]
            covers=[r[mode]["sensor_valid_object_coverage"] for r in frames]
            if any(type(v) is not float or not math.isfinite(v) or v<0 for v in values):raise ValueError("Finite complete paired scores")
            if any(type(v) is not float or not math.isfinite(v) or not 0<=v<=1 for v in covers):raise ValueError("Finite coverage")
            means[mode]=float(np.mean(values));coverage &= all(v>=.95 for v in covers)
        if any(r["exact_candidate_validity_matches_baseline"] is not True or r["fixed"]["sensor_valid_object_coverage"]!=r["learned"]["sensor_valid_object_coverage"] for r in frames):
            raise ValueError("No candidate validity change")
        scenes.append(dict(scene_id=scene,**means))
    before=np.array([r["fixed"] for r in scenes]);after=np.array([r["learned"] for r in scenes])
    gain=(before-after)/before if np.all(before>0) else None
    supported=bool(gain is not None and coverage and np.median(gain)>=.05 and np.min(gain)>=-.05)
    return dict(scenes=scenes,per_scene_relative_gain=None if gain is None else gain.tolist(),
        median_paired_scene_relative_gain=None if gain is None else float(np.median(gain)),coverage_gate_pass=bool(coverage),
        real_object_camera_hypothesis_supported=supported,scientific_decision="SUPPORT_NARROW_WHOLE_SUPPORT_DEPTH_HYPOTHESIS" if supported else "REJECT",
        adoption_performed=False,independent_scenes_or_objects=False,training_overlap_verified=False,
        human_quality_verified=False,temporal_quality_verified=False,full_v2d_score_verified=False,verified_victory_over_CARI4D=False)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    if (root != ROOT or code != root / "jobs" / revision / "run_tudl_whole_support_evaluate/code" or inputs._canonical(code) != code
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None or Path(__file__) != code / SOURCE_FILES[0]
            or os.environ["WR_IMAGE_ID"] != IMAGE_ID or platform.system() != "Linux"
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}):
        raise ValueError("Actual immutable CPU-only isolated private evaluation required")
    modules = {"infra/tudl_whole_support_infer.py": inference, "infra/tudl_anchor_evaluate.py": original_audit,
        "infra/tudl_evaluate.py": original_audit.scoring, "infra/tudl_whole_support_inputs.py": inputs,
        "infra/whole_support_anchor_contract.py": contract}
    if any(Path(module.__file__).resolve() != code / name for name, module in modules.items()):
        raise ValueError("Original exact frozen numerical helper paths required")
    helpers = {name: inputs.identity(code / name) for name in SOURCE_FILES}
    out = root / BASE / "quality_anchor_v1"
    if inputs._canonical(out) != out or not out.is_dir() or any(out.iterdir()): raise ValueError("Fresh exclusive evaluation output required")
    report = dict(stage=STAGE, status="fail", phase="public_validation", producer_revision=revision, image_id=IMAGE_ID,
        script_sha256=helpers[SOURCE_FILES[0]]["sha256"], source_helpers=helpers, budget_seconds=BUDGET,
        network="none", device="cpu", gpu_used=False, prediction_refit=False, alignment=False,
        challenge_inputs_used=False, private_truth_used_for_evaluation_only=True, independent_scenes_or_objects=False)
    started = time.perf_counter(); frozen = []; error = None
    def expired(*_): raise TimeoutError("Whole private TUDL whole-support evaluation exceeded180s")
    alarm, term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        arrays, records, frozen, public, pins = public_predictions(root, code)
        report.update(phase="private_evaluation", predictions_frozen_before_private_truth_read=True,
            prediction_report={k: pins["report"][k] for k in ("bytes", "sha256")}, scene_anchor_coefficients=pins["coefficients"])
        scenes, private_frozen, acquisition_id = private_inputs(root, code, public, records); frozen += private_frozen
        rows = []
        for data, record in zip(arrays, records):
            scene,frame=record["scene_id"],record["frame_id"];annotations=scenes[scene];key=str(frame)
            camera=annotations["camera"][key]
            k=original_audit.scoring.camera(np.array(camera["cam_K"],np.float64).reshape(3,3));scale=camera.get("depth_scale")
            if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:raise ValueError("Original positive sensor scale")
            directory=root/BASE/"eval_private"/f"source/test/{scene:06d}"
            depth=original_audit.scoring.read_png(directory/f"depth/{frame:06d}.png",depth=True)*float(scale)/1000
            visible=np.zeros(inference.GRID.shape,np.bool_);count=len(annotations["gt"][key])
            for i in range(count):visible |= original_audit.scoring.read_png(directory/f"mask_visib/{frame:06d}_{i:06d}.png")
            unchanged=contract.array_identities(data)
            row=original_audit.scoring.score_frame(score_arrays(data),depth,k,visible)
            if contract.array_identities(data)!=unchanged:raise ValueError("Frozen prediction mutated")
            row.update(scene_id=scene,frame_id=frame,object_instances_scored=count,
                exact_candidate_validity_matches_baseline=bool(np.array_equal(data["moge_validity"],data["anchored_validity"])))
            rows.append(row)
        report.update(frames=rows,decision=decision(rows,inputs.ordered_frames(public["input_pins"])),acquisition_report_identity=acquisition_id,
            status="pass",phase="complete",GT_pixel_convention="integer_BOP",prediction_pixel_convention="unchanged_plus_0.5",
            private_sensor_K_used_for_truth_only=True,all_original_object_instances_scored=True,alignment_performed=False)

    except Exception as exc:
        error = exc; report.update(error_type=type(exc).__name__, error=str(exc)[:300])
    finally:
        try:
            recheck(frozen)
            if {name: inputs.identity(code / name) for name in SOURCE_FILES} != helpers: raise ValueError("Evaluation source changed")
            report["all_frozen_artifacts_rechecked"] = True
        except Exception as exc:
            report.update(status="fail", postcheck_error_type=type(exc).__name__, postcheck_error=str(exc)[:300]); error = exc
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter() - started
        path = out / "report.json"
        with path.open("x") as stream:
            import json
            json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o400)
    if error is not None: raise error


if __name__ == "__main__": main()
