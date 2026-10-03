"""CPU-only D96 public artifact/source checks; no model forward or private IO."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import keypoint_rgb_baseline as baseline
from world_reward import metric_alignment, root_refit as policy
from world_reward.data import sha256

native = baseline.native
BASE = baseline.BASE
OUT = BASE+"/root_fit_v1"
STAGE = "public_keypoint_rgb_native_root_refit"
BUDGET = 180
BASELINE_SHA = "8ebfce153ea5ff708578c155d60adb97e737b672e2c2df56df38c635fbbc80c3"
BASELINE_REVISION = "f5c78f7abfec3c6234bbbee7f7dad70736836ae7"
BASELINE_SCRIPT = "4950979a0dfbeb6759f0a7149a9d79ca61fdb13c96cf09c5349fca1f6f73459b"
DW_SHA = "949bdf218514335cedc2745b081f2a79f93d4372bc66a0894a1bda7749e29c49"
DW_REVISION = "776e479c0a3492078313b93b8bf92e7443b274cf"
DW_SCRIPT = "f6fbf3e578e2a3c603a9644f341401576cc6360694ec52cf909907ba00771121"
DW_KEYS = {"keypoints", "scores", "validity", "bbox", "clip_index", "frame_index"}
CANDIDATE_KEYS = frozenset(native.BLOCKS) | {"vertices_camera_m", "keypoints_camera_m", "joints_camera_m",
    "joint_global_rotations", "human_faces", "hand_mask_left", "hand_mask_right", "camera_K", "clip_index", "frame_index",
    "latent", "physical_delta"}
PROXY_KEYS = {"object_points_camera_m", "pixel_indices", "shared_depth_scale", "clip_index", "frame_index"}
OPTIMIZER = dict(evaluated_states_per_frame=60, updates_per_frame=59, learning_rate=.01, betas=[.9, .999], eps=1e-8)


def require_fields(data, expected):
    if not isinstance(data, dict) or any(type(data.get(k)) is not type(v) or data[k] != v for k, v in expected.items()):
        raise ValueError("Exact frozen public producer contract differs")


def helper_identities(fit_source=None):
    source = Path(__file__).with_name("keypoint_rgb_fit.py") if fit_source is None else Path(fit_source)
    return baseline.helper_identities() | {"fit": native.regular(source)["sha256"], "public_consumer": sha256(Path(__file__)), "root_refit": sha256(Path(policy.__file__)),
                                          "metric_alignment": sha256(Path(metric_alignment.__file__))}


def array_identity(value):
    if type(value) is not np.ndarray or value.dtype.hasobject or not np.isfinite(value).all():
        raise ValueError("Finite unmasked native array required")
    return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())


def validate_dw(data, record):
    if set(data) != DW_KEYS: raise ValueError("Exact independent native133 schema required")
    points = native.array(data["keypoints"], (1, 133, 2), "float64")
    scores = native.array(data["scores"], (1, 133), "float32")
    validity = native.array(data["validity"], (1, 133), "bool")
    box = native.array(data["bbox"], (1, 4), "float32")
    if (not np.array_equal(validity, scores > 0) or np.any(box[:, :2] < 0)
            or box[0, 2] > native.WIDTH or box[0, 3] > native.HEIGHT or np.any(box[:, 2:] <= box[:, :2])):
        raise ValueError("Unclamped positive-score validity and bounded automatic bbox required")
    for k in ("clip_index", "frame_index"):
        if int(native.array(data[k], (), "int64")) != record[k]: raise ValueError("Original independent frame order differs")
    policy.training_observations(points[0], scores[0])
    return data


def validate_dw_report(report, records):
    require_fields(report, dict(stage="public_keypoint_rgb_native_dwpose133_observations", status="pass", phase="complete",
        producer_revision=DW_REVISION, script_sha256=DW_SCRIPT, image_id=native.IMAGE_ID, device="cpu", network="none",
        actual_sessions=1, requested_calls=15, all_cases_retained=True, native_cpu_abi_verified=True,
        final_inputs_source_assets_rehashed=True, private_prefix_removed=True, private_truth_read=False, ground_truth_used=False,
        challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[], fitting_performed=False, quality_verified=False,
        accuracy_verified=False, adoption_authorized=False, native_source_modified=False, own_feed_cast=False, channel_swap=False,
        full_image_fallback=False, raw_scores_clamped=False, confidence_threshold_applied=False, wrapper_neck134_used=False,
        global_image_modified=False, gpu_used=False))
    graph = dict(inputs=[dict(name="input", type="tensor(float)", shape=["batch", 3, 384, 288])],
        outputs=[dict(name="simcc_"+a, type="tensor(float)", shape=["batch", "MatMulsimcc_"+a+"_dim_1", "MatMulsimcc_"+a+"_dim_2"]) for a in ("x", "y")],
        providers=["CPUExecutionProvider"], custom_metadata={})
    if report.get("graph") != graph or not isinstance(report.get("calls"), list) or len(report["calls"]) != 15:
        raise ValueError("Actual pinned symbolic graph/all15 CPU calls required")
    rows = report.get("records")
    if not isinstance(rows, list) or len(rows) != 15: raise ValueError("All independent observations required")
    for row, call, record in zip(rows, report["calls"], records):
        require_fields(row, {k: record[k] for k in ("file", "clip_index", "frame_index")} | {"rgb_sha256": record["sha256"],
            "prediction_file": Path(record["file"]).stem+".npz"})
        require_fields(call, dict(supplied_container="list", effective_feed_shape=[1, 3, 384, 288],
            effective_runtime_conversion_observed=False, delegated_unmodified=True, run_completed=True))
        for meta, dtype, shape in [(call.get("supplied_array", {}), "float64", [3, 384, 288])]+[
                (m, "float32", s) for m, s in zip(call.get("raw_simcc", []), ([1, 133, 576], [1, 133, 768]))]:
            if meta.get("dtype") != dtype or meta.get("shape") != shape or not re.fullmatch("[0-9a-f]{64}", str(meta.get("sha256", ""))):
                raise ValueError("Original unchanged native feed/SimCC identity required")
        if len(call.get("raw_simcc", [])) != 2 or row.get("raw_simcc") != call["raw_simcc"]:
            raise ValueError("Actual two native133 SimCC arrays required")


def dw_bound_files(root, report):
    """Rehash receipt-bound assets/notices and original capability receipts."""
    assets = report.get("assets", {}); rows = assets.get("files")
    if not isinstance(rows, list) or len(rows) != 9: raise ValueError("Exact nine pinned DWPose assets required")
    paths = []
    for row in rows:
        name = row.get("file", "")
        if (not isinstance(name, str) or not name or any(p in ("", ".", "..") for p in name.split("/"))
                or name.startswith("/") or "\\" in name): raise ValueError("Safe pinned asset path required")
        path = Path(root)/"weights/dwpose_native_v1"/name
        if native.regular(path, row["sha256"])["bytes"] != row["bytes"]: raise ValueError("Pinned DWPose asset size differs")
        paths.append(path)
    for row in assets.get("retained_notices", []):
        name = row["file"]
        if name.startswith("/") or ".." in name.split("/") or "\\" in name: raise ValueError("Safe retained notice path required")
        path = Path(root)/"results/dwpose-wheel-audit-v3"/name
        if native.regular(path, row["sha256"])["bytes"] != row["bytes"]: raise ValueError("Retained license notice differs")
        paths.append(path)
    receipts = [("results/dwpose-acquisition-v1.json", assets["original_receipt_sha256"]),
        ("results/dwpose-wheel-audit-v2/report.json", assets["previous_failed_receipt_sha256"]),
        ("results/dwpose-wheel-audit-v3/report.json", assets["audit_receipt"]["sha256"]),
        ("validation/dwpose_smoke_v2/report.json", report["capability_evidence"]["actual_pass_receipt"]["sha256"]),
        ("validation/dwpose_smoke_v1/report.json", report["capability_evidence"]["previous_failed_receipt"]["sha256"])]
    for relative, digest in receipts:
        path = Path(root)/relative; native.regular(path, digest, immutable=True); paths.append(path)
    source_directory = Path(root)/"jobs"/DW_REVISION/"run_keypoint_rgb_dwpose/code/infra"
    sources = report.get("sources")
    if not isinstance(sources, dict) or not sources: raise ValueError("Frozen native DWPose source inventory required")
    for name, row in sources.items():
        if (not isinstance(name, str) or Path(name).name != name or not name.endswith((".py", ".sh"))
                or not isinstance(row, dict) or set(row) != {"sha256", "bytes"}):
            raise ValueError("Exact safe native DWPose source text identities required")
        path = source_directory/name
        if native.regular(path, row["sha256"])["bytes"] != row["bytes"]: raise ValueError("Actual frozen DWPose source bytes differ")
        paths.append(path)
    return paths


def public_predictions(root):
    """CPU-only full public/source/asset freeze; usable by the private evaluator."""
    root = Path(root); base = root/BASE; records, _ = baseline.public_inputs(root)
    bp, dp = base/"baseline_v1/report.json", base/"dwpose_v1/report.json"
    native.regular(bp, BASELINE_SHA, immutable=True); native.regular(dp, DW_SHA, immutable=True)
    b, d = json.loads(bp.read_text()), json.loads(dp.read_text())
    require_fields(b, dict(stage=baseline.STAGE, status="pass", phase="complete", producer_revision=BASELINE_REVISION,
        script_sha256=BASELINE_SCRIPT, frames=15, all_cases_retained=True, network="none", private_truth_read=False,
        ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        raw_frozen_before_shared=True, paired_frozen_before_reference=True, sources_assets_rechecked=True,
        conversion_fidelity_verified=True, actual_body_inference=True, actual_MoGe_inference=True, actual_shared_native_forward=True,
        body_calls_completed=15, MoGe_calls_completed=15, raw_parity_head_calls_completed=15, raw_keypoint_head_calls_completed=15,
        shared_head_calls_completed=15, official_reference_calls=1, accuracy_verified=False, adoption_performed=False))
    errors = np.asarray(b.get("official_reference_per_frame_mean_mm"))
    if errors.shape != (15,) or not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > 2):
        raise ValueError("Actual full15 reference fidelity required")
    baseline.validate_producer_bindings(root, b)
    raw = native.frozen_rows(bp.parent, b["raw_outputs"], records, "raw", baseline.validate_raw)
    pairs = native.frozen_rows(bp.parent, b["paired_outputs"], records, "paired", baseline.validate_pair)
    for data, original, record in zip(pairs, raw, records): baseline.validate_pair(data, record, original)
    baseline.validate_clip_constants(pairs, raw); native.validate_region_masks(pairs, b)
    if {p.name for p in bp.parent.iterdir()} != {"raw", "paired", "report.json"}: raise ValueError("Exact baseline inventory required")
    validate_dw_report(d, records); dw = []
    if {p.name for p in dp.parent.iterdir()} != {"report.json", *[Path(r["file"]).stem+".npz" for r in records]}:
        raise ValueError("Exact frozen independent observation inventory required")
    for row, record, original in zip(d["records"], records, raw):
        path = dp.parent/row["prediction_file"]; receipt = native.regular(path, immutable=True)
        if {k: receipt[k] for k in ("sha256", "bytes")} != row["prediction"]: raise ValueError("Frozen independent prediction bytes differ")
        with np.load(path, allow_pickle=False) as saved: data = {k: saved[k] for k in saved.files}
        validate_dw(data, record)
        if any(array_identity(data[k]) != row[k] for k in ("keypoints", "scores", "validity", "bbox")):
            raise ValueError("Independent canonical arrays changed")
        yy, xx = np.nonzero(original["human_mask"])
        if len(xx) == 0 or not np.array_equal(data["bbox"], np.array([[xx.min(), yy.min(), xx.max()+1, yy.max()+1]], np.float32)):
            raise ValueError("Independent observations must use the same automatic person bbox")
        dw.append(data)
    paths = [base/"inputs/manifest.json", base/"automatic_masks/report.json", bp, dp]+dw_bound_files(root, d)
    for r in records: paths.extend([r["path"], r["human_mask_path"], r["object_mask_path"]])
    paths += [bp.parent/r["artifact"] for r in b["raw_outputs"]+b["paired_outputs"]]
    paths += [dp.parent/r["prediction_file"] for r in d["records"]]
    frozen = [{"path": str(p), **native.regular(p)} for p in paths]
    lineage = dict(baseline_report_sha256=BASELINE_SHA, dwpose_report_sha256=DW_SHA,
        input_manifest_sha256=baseline.MANIFEST_SHA, mask_report_sha256=baseline.MASK_SHA, frozen_files=frozen)
    return records, raw, pairs, dw, lineage


def object_sample(raw, scale):
    """Sorted original-pixel sampling, independent of fitted body parameters."""
    support = raw["object_mask"] & raw["validity"]
    indices = np.flatnonzero(support)
    if len(indices) < 32: raise ValueError("Every original object requires32 valid automatic pixels")
    if len(indices) > 8192: indices = np.sort(np.random.default_rng(0).choice(indices, 8192, replace=False))
    points = (raw["raw_points"].reshape(-1, 3)[indices].astype(np.float64)*scale).astype(np.float32)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0): raise ValueError("No proxy fill, clipping or invalid point removal")
    return points, indices.astype(np.int64)


def validate_proxy(data, record, raw):
    if set(data) != PROXY_KEYS: raise ValueError("Exact frozen common object schema required")
    scale = float(native.array(data["shared_depth_scale"], (), "float64"))
    if scale <= 0: raise ValueError("Positive shared metric consistency scale required")
    expected, indices = object_sample(raw, scale)
    if (native.array(data["object_points_camera_m"], expected.shape, "float32").tobytes() != expected.tobytes()
            or native.array(data["pixel_indices"], indices.shape, "int64").tobytes() != indices.tobytes()):
        raise ValueError("Frozen object support, sampling or shared scale changed")
    for k in ("clip_index", "frame_index"):
        if int(native.array(data[k], (), "int64")) != record[k]: raise ValueError("Original proxy frame differs")
    return data


def frozen_proxies(root, records, rows, raw_frames, pairs):
    values = native.frozen_rows(Path(root)/OUT, rows, records, "proxies",
        lambda data, record: validate_proxy(data, record, raw_frames[record["clip_index"]*5+record["frame_index"]]))
    for c in range(3):
        if any(v["shared_depth_scale"].tobytes() != values[c*5]["shared_depth_scale"].tobytes() for v in values[c*5:c*5+5]):
            raise ValueError("Common object scale must be byte-constant across all5frames")
    return values


def validate_candidate(data, record, raw, pair):
    if set(data) != CANDIDATE_KEYS: raise ValueError("Exact full native root candidate required")
    native.common(data, record)
    for k, shape in native.BLOCKS.items(): native.array(data[k], shape, "float32")
    latent = native.array(data["latent"], (6,), "float32"); delta = native.array(data["physical_delta"], (6,), "float32")
    policy.physical_delta(delta.astype(np.float64))
    if not np.allclose(delta, policy.bounded_delta(latent), atol=3e-8, rtol=3e-7): raise ValueError("Physical update is not the frozen tanh parameterization")
    for k, original in (("body_pose_params", raw["body_pose_params"]), ("hand_pose_params", raw["hand_pose_params"]),
            ("expr_params", raw["expr_params"]), ("shape_params", pair["shared_shape_params"]), ("scale_params", pair["shared_scale_params"]),
            ("human_faces", pair["human_faces"]), ("camera_K", pair["camera_K"]), ("hand_mask_left", pair["hand_mask_left"]), ("hand_mask_right", pair["hand_mask_right"])):
        if data[k].tobytes() != original.tobytes(): raise ValueError("Fixed body/hands/identity/geometry/camera evidence changed")
    controls = data["mhr_model_params"]
    if (np.any(controls[:3]) or controls[3:6].tobytes() != data["global_rot"].tobytes()
            or controls[6:].tobytes() != pair["shared_model_controls"][6:].tobytes()
            or not np.allclose(data["global_rot"], raw["global_rot"]+delta[3:], atol=1e-7, rtol=1e-7)
            or not np.allclose(data["pred_cam_t"], policy.camera_translation(raw["pred_cam_t"], delta.astype(np.float64)), atol=5e-7, rtol=2e-7)):
        raise ValueError("Only native rootEuler and external cameraXYZ may change")
    for k, shape in (("vertices_camera_m", (native.VERTICES, 3)), ("keypoints_camera_m", (308, 3)), ("joints_camera_m", (127, 3))):
        if np.any(native.array(data[k], shape, "float32")[:, 2] <= 0): raise ValueError("Every exported point requires positive cameraZ; no clipping")
    baseline.rotations(data["joint_global_rotations"])
    return data


def frozen_candidates(root, records, rows, raw_frames, pairs):
    return native.frozen_rows(Path(root)/OUT, rows, records, "candidates", lambda data, record:
        validate_candidate(data, record, raw_frames[record["clip_index"]*5+record["frame_index"]], pairs[record["clip_index"]*5+record["frame_index"]]))
