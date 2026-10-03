"""CPU-only H98 public artifact contracts, never native inference or private IO.

Pins are mandatory external immutable evidence, not discovered from outputs.
Receipts and byte replay establish lineage/ABI, not geometric accuracy or rights.
"""
from dataclasses import asdict
from importlib import metadata, util
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import root5_rgb_observe as observe
from world_reward import metric_alignment, root_refit as policy
from world_reward.data import sha256

protocol, baseline, native = observe.protocol, observe.body, observe.native
body = baseline
COHORT = protocol.COHORT
BASE, OUT = COHORT.base, COHORT.base+"/root_fit_v1"
STAGE, BUDGET = "public_root5_rgb_native_fixed_depth_refit", 300
PRODUCER_REVISION = "8084688a4d84bbad9ba8c0580e4d1b2803745511"
NATIVE_PRODUCER_REVISION = "feae71ea16a1d942f08e95ccafc131b6467dffb9"
NATIVE_SOURCE_SHA = "0e034078bf9026b338783867293aa5f95f8c1955b5b21760b7916408ec496018"
MASK_SOURCE_SHA = "cb48d7d0a53cfd7b9f5c7ebe9e0904af91e844d187813fd1f973538b158cd4b5"
MANIFEST_SHA = "16d909d206ea9a71f402c01a699e69d63603e5b1366e1e24d55ac208846f466b"
PIN_SCHEMA = "world-reward-root5-public-pins-v1"
PIN_KEYS = {"schema", "producer_revision", "manifest_sha256", "automatic_masks_sha256", "baseline_sha256", "dwpose_sha256"}
BOUNDS = np.array([.30, .30, .30/np.sqrt(3.), .30/np.sqrt(3.), .30/np.sqrt(3.)], np.float32)
CANDIDATE_KEYS = frozenset(native.BLOCKS) | {"vertices_camera_m", "keypoints_camera_m", "joints_camera_m", "joint_global_rotations",
    "human_faces", "hand_mask_left", "hand_mask_right", "camera_K", "clip_index", "frame_index", "latent", "physical_delta"}
PROXY_KEYS = {"object_points_camera_m", "pixel_indices", "shared_depth_scale", "clip_index", "frame_index"}
require_fields = observe.require_fields


def validate_pins(pins):
    """Exact six-field external receipt contract; missing/extra/dynamic pins fail."""
    if not isinstance(pins, dict) or set(pins) != PIN_KEYS:
        raise ValueError("Explicit immutable H98 public receipt pins required")
    require_fields(pins, dict(schema=PIN_SCHEMA, manifest_sha256=MANIFEST_SHA))
    revisions = pins["producer_revision"]
    if (not isinstance(revisions, dict) or set(revisions) != {"masks", "baseline", "dwpose"}
            or revisions != dict(masks=PRODUCER_REVISION, baseline=NATIVE_PRODUCER_REVISION, dwpose=NATIVE_PRODUCER_REVISION)
            or any(type(v) is not str or not re.fullmatch("[0-9a-f]{40}", v) for v in revisions.values())):
        raise ValueError("Explicit immutable stage-specific observer revisions required")
    if any(type(pins[k]) is not str or not re.fullmatch("[0-9a-f]{64}", pins[k]) for k in PIN_KEYS if k.endswith("sha256")):
        raise ValueError("Exact hexadecimal public receipt pins required")
    return {**pins, "producer_revision": dict(revisions)}


def load_pins(path):
    protocol.identity(path)
    return validate_pins(json.loads(Path(path).read_text()))


def helper_identities(fit_source=None):
    result = observe.source_identity() | dict(public_consumer=sha256(Path(__file__)),
        root_refit=sha256(Path(policy.__file__)), metric_alignment=sha256(Path(metric_alignment.__file__)))
    if fit_source is not None: result["fit"] = native.regular(fit_source)["sha256"]
    return result


def observer_sources(root, revisions):
    """Source-only historical files at their canonical producer path, not CODE aliases."""
    modules = (observe, protocol, observe.masks, baseline, observe.dw_helper, observe.depth_camera, native,
        native.masks, native.masks.masks, native.human, native.human.body, native.depth_model, native.joint,
        native.regions_helper, baseline.masks, observe.smoke, observe.smoke.acquisition, observe.smoke.audit)
    paths = [Path(m.__file__) for m in modules]+[Path(native.__file__).with_name("camera_render.py"),
        Path(observe.smoke.__file__).with_name("run_dwpose_smoke.sh")]
    if sha256(Path(observe.__file__)) != NATIVE_SOURCE_SHA: raise ValueError("Frozen corrected observer source changed")
    result = [observe.historical_mask_source(root)[0]]
    for revision in dict.fromkeys(revisions[k] for k in ("baseline", "dwpose")):
        directory = Path(root)/"jobs"/revision/"run_root5_rgb_observe/code/infra"
        for current in {p.name: p for p in paths}.values():
            path = directory/current.name
            native.regular(path, sha256(current), immutable=True); result.append(path)
    return result


def require_stage(report, stage, sources, revision):
    require_fields(report, dict(stage=observe.STAGES[stage], status="pass", phase="complete", frames=COHORT.count,
        cohort=asdict(COHORT), producer_revision=revision, script_sha256=sources["observer"], source_helpers=sources,
        network="none", private_truth_read=False, ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False,
        oracle_modes=[], fitting_performed=False, quality_verified=False, accuracy_verified=False, adoption_authorized=False,
        full_HOI_verified=False, all_cases_retained=True, budget_seconds=observe.BUDGETS[stage], gpu_used=stage!="dwpose"))
    if any(k in report for k in ("error", "error_type")): raise ValueError("Completed producer cannot retain failure evidence")
    if stage != "masks": require_fields(report, dict(image_id=native.IMAGE_ID))


def validate_body_bindings(root, report):
    """Reuse pinned native asset checks without an old cohort/stage main or decoder."""
    model = report.get("body_model", {}); load = model.get("checkpoint_loading", {})
    require_fields(model, dict(body_revision=native.human.body.BODY_REVISION, upstream_revision=native.human.body.UPSTREAM_REVISION,
        dinov3_revision=native.human.body.DINOV3_REVISION))
    require_fields(load, dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention", unexpected_keys=[], parameter_tensors_loaded=1101))
    names = load.get("retained_mhr_asset_buffer_names")
    if not isinstance(names, list) or len(names) != 113 or any(type(n) is not str for n in names) or len(set(names)) != 113:
        raise ValueError("All immutable rig buffers and trainable checkpoint tensors required")
    assets = model.get("body_assets", {})
    if assets.get("model.ckpt") != dict(sha256=native.human.BODY_SHA, bytes=native.human.BODY_BYTES):
        raise ValueError("Exact Body network checkpoint required")
    if native.human.body._source_identity(root) != model.get("inference_source_identity") or native.human.body._body_assets(root)[1] != assets:
        raise ValueError("Native Body source/model assets differ")
    asset, receipt, asset_id = native.depth_model.model_asset(root)
    if (receipt, asset_id) != (report.get("acquisition_report"), report.get("MoGe_model_asset")):
        raise ValueError("MoGe acquisition/model bytes differ")
    dist = metadata.distribution("moge"); direct = dist.read_text("direct_url.json"); spec = util.find_spec("moge")
    if spec is None or not spec.submodule_search_locations: raise ValueError("Actual installed MoGe source missing")
    directory = Path(next(iter(spec.submodule_search_locations)))
    if (native.depth_model.installed_source(directory, json.loads(direct) if direct else {}) != report.get("MoGe_source")
        or sha256(directory/"utils/geometry_torch.py") != observe.depth_camera.GEOMETRY_SHA):
        raise ValueError("Native MoGe/focal solver source differs")
    paths = [Path(root)/"weights/mhr/mhr_model.pt", Path(root)/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py",
             Path(root)/"results/mhr-finger-semantics-v4.json", Path(root)/"results/weights-acquisition.json", Path(asset)]
    for path, pin in zip(paths[:3], (native.REFERENCE_MODEL_SHA256, native.CONVERTER_SHA256, report.get("semantic_report_sha256"))):
        native.regular(path, pin)
    semantic = json.loads(paths[2].read_text()); native.regions_helper.require_semantic_report(semantic)
    if report.get("joint_names") != semantic["joint_names"] or semantic.get("source_image_id") != native.IMAGE_ID:
        raise ValueError("Actual native semantic joint names/image differ")
    body_dir = Path(root)/"weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3"
    paths += [body_dir/name for name in assets]
    return paths


def validate_baseline(report, records, inputs):
    require_fields(report, dict(public_inputs=inputs, raw_frozen_before_shared=True, paired_frozen_before_reference=True,
        sources_assets_rechecked=True, conversion_fidelity_verified=True, actual_body_inference=True, actual_MoGe_inference=True,
        actual_shared_native_forward=True, body_calls_completed=15, MoGe_calls_completed=15, raw_parity_head_calls_completed=15,
        raw_keypoint_head_calls_completed=15, shared_head_calls_completed=15, official_reference_calls=1, body_inference_type="body",
        camera_K=[list(r) for r in COHORT.fixed_K], focal_fitted=False, scale_fit=False, inverse_fit_performed=False,
        identity_source="first_original_RGB_shape45_and_scale28_per_five_frame_clip", expression_zero=True, keypoint_count=308,
        joint_count=127, hand_regions_native_verified=True, model_sha256=native.REFERENCE_MODEL_SHA256, converter_sha256=native.CONVERTER_SHA256,
        seed=0, TF32=False, CUBLAS_WORKSPACE_CONFIG=":4096:8"))
    fidelity = report.get("reference_fidelity", {}); errors = np.asarray(report.get("official_reference_per_frame_mean_mm"))
    if (errors.shape != (15,) or errors.dtype.kind != "f" or not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > 2)
            or fidelity.get("per_frame_mean_mm") != errors.tolist()): raise ValueError("All15 independent reference mean fidelities required")
    for key in ("max_point_mm_diagnostic", "max_joint_distance_mm_diagnostic"):
        value = fidelity.get(key)
        if type(value) not in (float, int) or not np.isfinite(value) or value < 0: raise ValueError("Finite reference diagnostics required")
    calls, focal = report.get("calls"), report.get("native_focal_solver_calls")
    if not isinstance(calls, list) or not isinstance(focal, list) or len(calls) != 15 or len(focal) != 15:
        raise ValueError("All original Body/MoGe call evidence required")
    for row, solver, record in zip(calls, focal, records):
        require_fields(row, dict(file=record["file"]))
        require_fields(solver, {k: record[k] for k in ("file", "clip_index", "frame_index")}|dict(focal_prior_supplied=True, original_solver_returned=True))
        if type(solver.get("native_nearest64_valid_pixels")) is not int or not 2 <= solver["native_nearest64_valid_pixels"] <= 4096:
            raise ValueError("Native focal support cannot use its fallback")
        errors = row.get("native_forward_errors", {})
        if set(errors) != {"vertices_m", "joints_m", "keypoints_m", "controls"} or any(type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v <= 1e-5 for v in errors.values()):
            raise ValueError("Actual fresh raw head parity evidence differs")
        if not re.fullmatch("[0-9a-f]{64}", str(row.get("decoded_RGB_sha256", ""))): raise ValueError("Decoded original RGB evidence missing")


def validate_dw_report(report, records, inputs):
    require_fields(report, dict(public_inputs=inputs, actual_sessions=1, private_prefix_packages_installed=True, private_prefix_removed=True,
        final_inputs_source_assets_rehashed=True, native_cpu_abi_verified=True, native_source_modified=False, own_feed_cast=False,
        channel_swap=False, full_image_fallback=False, raw_scores_clamped=False, confidence_threshold_applied=False,
        wrapper_neck134_used=False, global_image_modified=False))
    graph = dict(inputs=[dict(name="input", type="tensor(float)", shape=["batch", 3, 384, 288])],
        outputs=[dict(name="simcc_"+a, type="tensor(float)", shape=["batch", "MatMulsimcc_"+a+"_dim_1", "MatMulsimcc_"+a+"_dim_2"]) for a in ("x", "y")],
        providers=["CPUExecutionProvider"], custom_metadata={})
    rows, calls = report.get("records"), report.get("calls")
    if report.get("graph") != graph or not isinstance(rows, list) or not isinstance(calls, list) or len(rows) != 15 or len(calls) != 15:
        raise ValueError("One native symbolic graph and all15 CPU observations required")
    for row, call, record in zip(rows, calls, records):
        require_fields(row, {k: record[k] for k in ("file", "clip_index", "frame_index")}|dict(rgb_sha256=record["sha256"], prediction_file=Path(record["file"]).stem+".npz"))
        if type(row.get("positive_score_count")) is not int or not 0 <= row["positive_score_count"] <= 133:
            raise ValueError("Original native positive-score count required")
        require_fields(call, dict(supplied_container="list", effective_feed_shape=[1, 3, 384, 288], effective_runtime_conversion_observed=False,
            delegated_unmodified=True, run_completed=True))
        simcc = call.get("raw_simcc")
        if not isinstance(simcc, list) or len(simcc) != 2 or row.get("raw_simcc") != simcc: raise ValueError("Two actual native SimCC arrays required")
        for value, dtype, shape in [(call.get("supplied_array"), "float64", [3,384,288])]+list(zip(simcc, ("float32",)*2, ([1,133,576],[1,133,768]))):
            if (not isinstance(value, dict) or set(value) != {"dtype", "shape", "sha256"} or value["dtype"] != dtype or value["shape"] != shape
                or type(value["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", value["sha256"])):
                raise ValueError("Unmodified float64 native feed/SimCC identities required")


def dw_bound_files(root, report):
    if report.get("capability_evidence") != observe.dw_helper.validate_smoke(root) or report.get("assets") != observe.smoke.validate_assets(root):
        raise ValueError("Exact D95 replay and D94 asset/notice evidence required")
    if report.get("dependencies") != observe.smoke.dependency_identity(): raise ValueError("Pinned CPU image dependencies differ")
    paths = [Path(root)/observe.smoke.acquisition.BASE/row["file"] for row in report["assets"]["files"]]
    for row in report["assets"]["retained_notices"]:
        paths.append(Path(root)/observe.smoke.audit.OUT/row["file"])
    paths += [Path(root)/p for p in ("results/dwpose-acquisition-v1.json", "results/dwpose-wheel-audit-v2/report.json",
        "results/dwpose-wheel-audit-v3/report.json", "validation/dwpose_smoke_v1/report.json", "validation/dwpose_smoke_v2/report.json")]
    return paths


def public_predictions(root, pins):
    """Audit every frozen public artifact/model/source before any fit/private IO."""
    pins = validate_pins(pins); root = Path(root); base = root/BASE
    paths = observer_sources(root, pins["producer_revision"]); sources = observe.source_identity()
    reports = {}; receipts = {}
    for stage, key in (("masks", "automatic_masks_sha256"), ("baseline", "baseline_sha256"), ("dwpose", "dwpose_sha256")):
        path = base/observe.FOLDERS[stage]/"report.json"; receipts[stage] = protocol.identity(path)
        if receipts[stage]["sha256"] != pins[key]: raise ValueError("Pinned H98 producer report changed")
        reports[stage] = json.loads(path.read_text())
        stage_sources = observe.mask_source_helpers(root) if stage == "masks" else sources
        require_stage(reports[stage], stage, stage_sources, pins["producer_revision"][stage]); paths.append(path)
    records, inputs = observe.public_inputs(root, COHORT)
    if inputs["manifest"] != dict(sha256=pins["manifest_sha256"], bytes=2196) or inputs["automatic_masks"] != receipts["masks"]:
        raise ValueError("Exact new public manifest/mask receipts required")
    assets = reports["masks"].get("model_assets")
    if (assets != observe.masks.validate_assets(root, reports["masks"]["image_id"])
            or any(set(row) != {"path", "sha256", "bytes"} for row in assets.values())):
        raise ValueError("Exact pinned detector/SAM assets required")
    sam = reports["masks"].get("sam2_source", {})
    vcs = sam.get("vcs", {}); info = vcs.get("vcs_info", {})
    if (vcs.get("url") != "https://github.com/facebookresearch/sam2.git" or info.get("vcs") != "git"
            or type(info.get("commit_id")) is not str or not re.fullmatch("[0-9a-f]{40}", info["commit_id"])
            or type(sam.get("python_files")) is not int or sam["python_files"] < 4
            or not re.fullmatch("[0-9a-f]{64}", str(sam.get("python_source_sha256", "")))
            or sam.get("prior_upstream_source_pin_verified") is not False or sam.get("full_import_license_closure_verified") is not False):
        raise ValueError("Frozen grounding-image SAM2 source evidence required; no foreign package substitution")
    critical = sam.get("critical_sources", {})
    if set(critical) != {"build_sam.py", "sam2_image_predictor.py", "modeling/sam2_base.py", "utils/transforms.py"}:
        raise ValueError("Complete historical SAM2 critical-source inventory required")
    for row in [*critical.values(), sam.get("config", {})]:
        if (not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"} or type(row["path"]) is not str
                or not Path(row["path"]).is_absolute() or type(row["bytes"]) is not int or row["bytes"] <= 0
                or type(row["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Exact historical SAM2 source/config byte identities required")
    paths += [Path(row["path"]) for row in assets.values()]+[root/"results/image-grounding.json", root/"results/weights-acquisition.json"]
    b, d = reports["baseline"], reports["dwpose"]
    validate_baseline(b, records, inputs)
    if b.get("previous_failed_baseline") != observe.previous_baseline_failure(root, COHORT) or b.get("previous_failure_rewritten") is not False:
        raise ValueError("Original zero-call baseline bootstrap failure must remain unchanged")
    paths.append(base/"baseline_v1/report.json"); paths += validate_body_bindings(root, b)
    raw = native.frozen_rows(base/observe.FOLDERS["baseline"], b["raw_outputs"], records, "raw", baseline.validate_raw)
    pairs = native.frozen_rows(base/observe.FOLDERS["baseline"], b["paired_outputs"], records, "paired", baseline.validate_pair)
    if {p.name for p in (base/observe.FOLDERS["baseline"]).iterdir()} != {"report.json", "raw", "paired"}: raise ValueError("Exact baseline output inventory required")
    from PIL import Image
    for record, original, pair, call in zip(records, raw, pairs, b["calls"]):
        baseline.validate_pair(pair, record, original)
        hm, om = observe.read_public_masks(record, Image)
        rgb = observe.read_rgb(record, COHORT, Image)
        box, _ = native.human.derived_bbox(rgb, hm)
        if (not np.array_equal(original["human_mask"], hm>0) or not np.array_equal(original["object_mask"], om>0)
                or call.get("bbox_xyxy") != box.tolist() or call["decoded_RGB_sha256"] != hashlib.sha256(rgb.tobytes()).hexdigest()):
            raise ValueError("Raw body/depth must bind exactly the same original RGB/automatic masks")
    baseline.validate_clip_constants(pairs, raw); native.validate_region_masks(pairs, b)
    validate_dw_report(d, records, inputs); paths += dw_bound_files(root, d)
    observe.dw_helper.validate_artifacts(base/"dwpose_v1", d["records"]); dw = []
    for row, record, original in zip(d["records"], records, raw):
        path = base/"dwpose_v1"/row["prediction_file"]
        with np.load(path, allow_pickle=False) as saved: data = {k: saved[k] for k in saved.files}
        observe.dw_helper.validate_prediction(data, record)
        native.array(data["keypoints"], (1,133,2), "float64")
        if row["positive_score_count"] != int(data["validity"].sum()): raise ValueError("Native validity count differs")
        yy, xx = np.nonzero(original["human_mask"])
        if not len(xx) or not np.array_equal(data["bbox"], np.array([[xx.min(),yy.min(),xx.max()+1,yy.max()+1]],np.float32)):
            raise ValueError("DWPose and Body must use the same automatic human bbox")
        dw.append(data); paths.append(path)
    for record in records: paths += [record["path"], record["human_mask_path"], record["object_mask_path"]]
    paths.append(base/"inputs/manifest.json")
    paths += [base/observe.FOLDERS["baseline"]/r["artifact"] for r in b["raw_outputs"]+b["paired_outputs"]]
    frozen = [native.regular(path) for path in dict.fromkeys(paths)]
    return records, raw, pairs, dw, dict(input_manifest_sha256=pins["manifest_sha256"], mask_report_sha256=pins["automatic_masks_sha256"],
        baseline_report_sha256=pins["baseline_sha256"], dwpose_report_sha256=pins["dwpose_sha256"], observer_revisions=pins["producer_revision"],
        public_pins=pins, frozen_files=frozen)


def object_sample(raw, scale):
    support = raw["object_mask"] & raw["validity"]; indices = np.flatnonzero(support)
    if len(indices) < 32 or type(scale) not in (float, int) or not np.isfinite(scale) or scale <= 0: raise ValueError("Every object needs32 valid pixels/one positive shared scale")
    if len(indices) > 8192: indices = np.sort(np.random.default_rng(0).choice(indices, 8192, replace=False))
    points = (raw["raw_points"].reshape(-1,3)[indices].astype(np.float64)*scale).astype(np.float32)
    if not np.isfinite(points).all() or np.any(points[:,2] <= 0): raise ValueError("No object proxy filling/clipping or independent rescaling")
    return points, indices.astype(np.int64)


def validate_proxy(data, record, raw):
    if set(data) != PROXY_KEYS: raise ValueError("Exact frozen common object proxy required")
    scale = float(native.array(data["shared_depth_scale"], (), "float64")); expected, indices = object_sample(raw, scale)
    if (native.array(data["object_points_camera_m"], expected.shape, "float32").tobytes() != expected.tobytes()
        or native.array(data["pixel_indices"], indices.shape, "int64").tobytes() != indices.tobytes()):
        raise ValueError("Original object support/shared scale/sampling changed")
    for k in ("clip_index", "frame_index"):
        if int(native.array(data[k], (), "int64")) != record[k]: raise ValueError("Original proxy frame differs")
    return data


def frozen_proxies(root, records, rows, raw_frames, pairs):
    values = native.frozen_rows(Path(root)/OUT, rows, records, "proxies", lambda data, record:
        validate_proxy(data, record, raw_frames[record["clip_index"]*COHORT.frames+record["frame_index"]]))
    for clip in range(COHORT.clips):
        if any(v["shared_depth_scale"].tobytes() != values[clip*COHORT.frames]["shared_depth_scale"].tobytes()
               for v in values[clip*COHORT.frames:(clip+1)*COHORT.frames]): raise ValueError("Clip object scale must be byte-constant")
    return values


def validate_candidate(data, record, raw, pair):
    if set(data) != CANDIDATE_KEYS: raise ValueError("Exact complete fixed-Z native root5 candidate required")
    native.common(data, record)
    for k, shape in native.BLOCKS.items(): native.array(data[k], shape, "float32")
    latent = native.array(data["latent"], (5,), "float32"); delta = native.array(data["physical_delta"], (5,), "float32")
    if np.any(np.abs(delta) > BOUNDS) or not np.allclose(delta, np.tanh(latent)*BOUNDS, atol=3e-8, rtol=3e-7):
        raise ValueError("Native float32 bounded root5 parameterization differs")
    for k, original in (("body_pose_params",raw["body_pose_params"]),("hand_pose_params",raw["hand_pose_params"]),
        ("expr_params",raw["expr_params"]),("shape_params",pair["shared_shape_params"]),("scale_params",pair["shared_scale_params"]),
        ("human_faces",pair["human_faces"]),("camera_K",pair["camera_K"]),("hand_mask_left",pair["hand_mask_left"]),("hand_mask_right",pair["hand_mask_right"])):
        if data[k].tobytes() != original.tobytes(): raise ValueError("Fixed native identity/body/hands/topology/camera/regions changed")
    for side in ("left", "right"): native.array(data["hand_mask_"+side], (native.VERTICES,), "bool")
    controls = data["mhr_model_params"]
    if (np.any(controls[:3]) or controls[3:6].tobytes() != data["global_rot"].tobytes() or controls[6:].tobytes() != pair["shared_model_controls"][6:].tobytes()
        or data["pred_cam_t"][2:].tobytes() != raw["pred_cam_t"][2:].tobytes()
        or not np.allclose(data["global_rot"],raw["global_rot"]+delta[2:],atol=1e-7,rtol=1e-7)
        or not np.allclose(data["pred_cam_t"][:2],raw["pred_cam_t"][:2]+delta[:2],atol=5e-7,rtol=2e-7)):
        raise ValueError("Only nativeEuler and externalXY may change; Z/remainder byte-fixed")
    for k, shape in (("vertices_camera_m",(native.VERTICES,3)),("keypoints_camera_m",(308,3)),("joints_camera_m",(127,3))):
        if np.any(native.array(data[k],shape,"float32")[:,2] <= 0): raise ValueError("Every camera point needs positiveZ; no clipping")
    baseline.rotations(data["joint_global_rotations"])
    return data


def frozen_candidates(root, records, rows, raw_frames, pairs):
    return native.frozen_rows(Path(root)/OUT,rows,records,"candidates",lambda data,record:
        validate_candidate(data,record,raw_frames[record["clip_index"]*COHORT.frames+record["frame_index"]],pairs[record["clip_index"]*COHORT.frames+record["frame_index"]]))


def jacobian_evidence(jacobian,points):
    j=np.asarray(jacobian)
    if (np.ma.isMaskedArray(jacobian)or type(points)is not int or not 6<=points<=10 or j.shape!=(points*2,5)
            or j.dtype.kind!="f"or not np.isfinite(j).all()):raise ValueError("Observation-only finite2Nby5 Jacobian required")
    singular=np.linalg.svd(j.astype(np.float64),compute_uv=False);ratio=float(singular[-1]/singular[0])if singular[0]>0 else 0.
    evidence=dict(singular_values=singular.tolist(),minimum_maximum_ratio=ratio,observation_rows=points*2,columns=5,prior_rows_included=False)
    return evidence


def validate_schedule(row,observation,candidate):
    target,indices,validity=policy.training_observations(observation["keypoints"][0],observation["scores"][0]);n=len(target)
    target=target.astype(np.float32).astype(np.float64)
    if (row.get("training_validity")!=validity.tolist()or row.get("training_MHR_indices")!=indices.tolist()or row.get("training_points")!=n
            or type(row.get("training_points")) is not int or type(row.get("adam_updates")) is not int
            or row.get("adam_updates")!=59 or row.get("heldout_used_for_fit")is not False
            or row.get("observation_compute_dtype") != "float32"
            or row.get("heldout_COCO_indices")!=list(policy.HELDOUT_COCO)):
        raise ValueError("Exact training support and unobserved heldout policy required")
    def array(key,shape):
        v=np.asarray(row.get(key))
        if np.ma.isMaskedArray(row.get(key))or v.shape!=shape or v.dtype.kind!="f"or not np.isfinite(v).all():raise ValueError("Complete finite public training trace required: "+key)
        return v.astype(np.float64)
    u=array("evaluated_latents",(60,5));d=array("evaluated_physical_deltas",(60,5));xy=array("evaluated_projected_points",(60,n,2))
    losses=array("evaluated_losses",(60,));gradients=array("update_gradients",(59,5));J=array("initial_observation_jacobian_values",(2*n,5))
    if np.any(u[0])or not np.allclose(d,np.tanh(u)*BOUNDS.astype(np.float64),atol=1e-6,rtol=1e-5):raise ValueError("Trace starts zero and preserves physical root5 policy")
    m=np.zeros(5);v=np.zeros(5);reconstructed=u[0].copy()
    for i,g in enumerate(gradients):
        m=.9*m+.1*g;v=.999*v+.001*g*g
        reconstructed-=.01*(m/(1-.9**(i+1)))/(np.sqrt(v/(1-.999**(i+1)))+1e-8)
        if not np.allclose(u[i+1],reconstructed,atol=1e-6,rtol=1e-5):raise ValueError("Recorded evaluated state differs from59 actual-gradient Adam updates")
    calculated=[]
    for projected,delta in zip(xy,d):
        six=np.r_[delta[:2],0.,delta[2:]];calculated.append(policy.objective(projected,target,six)["total"])
    if not np.allclose(losses,calculated,atol=1e-6,rtol=1e-5):raise ValueError("Recorded objective differs from frozen public observations/physical prior")
    best=policy.best_evaluated(losses)
    if (type(row.get("best_evaluated_index"))is not int or row["best_evaluated_index"]!=best
            or not np.isclose(row.get("selected_total_objective",np.nan),losses[best],atol=1e-6,rtol=1e-5)
            or u[best].astype(np.float32).tobytes()!=candidate["latent"].tobytes()
            or d[best].astype(np.float32).tobytes()!=candidate["physical_delta"].tobytes()
            or np.asarray(row.get("selected_physical_delta"),np.float32).tobytes()!=candidate["physical_delta"].tobytes()):
        raise ValueError("Candidate must be the first best evaluated state, byte-identical to trace")
    evidence=jacobian_evidence(J,n)
    if evidence["minimum_maximum_ratio"]<1e-5:raise ValueError("Initial unregularized5DOF observation rank insufficient")
    saved=row.get("initial_observation_jacobian",{})
    if (saved.get("columns")!=5 or saved.get("observation_rows")!=2*n or saved.get("prior_rows_included")is not False
            or not np.allclose(saved.get("singular_values",[]),evidence["singular_values"],atol=1e-6,rtol=1e-5)
            or not np.isclose(saved.get("minimum_maximum_ratio",np.nan),evidence["minimum_maximum_ratio"],atol=1e-6,rtol=1e-5)):
        raise ValueError("Saved initial rank evidence differs from actual public Jacobian")
    return dict(states=60,updates=59,observation_points=n,best_evaluated_index=best,atol=1e-6,rtol=1e-5,
                manual_Adam_and_public_objectives_verified=True,heldout_used=False)

validate_trace = validate_schedule
