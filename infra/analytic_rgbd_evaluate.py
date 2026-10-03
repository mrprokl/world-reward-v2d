"""Private evaluation of frozen NEW analytic RGBD predictions, never refitting.

All public and prediction identities/contracts are checked before any private
file opens. Truth uses the renderer's +.5 pixel rays, not BOP integer pixels.
No alignment, camera/scale correction, masking changes or adoption occurs.
"""
import argparse
import hashlib
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
from scipy.spatial import cKDTree
import analytic_rgbd_infer as inference
import tudl_anchor_evaluate as original_audit

inputs, contract = inference.inputs, inference.contract
ROOT, BASE, IMAGE_ID = inference.ROOT, inference.BASE, inference.IMAGE_ID
PINS = "configs/analytic_rgbd_prediction_pins.json"
STAGE = "private_analytic_rgbd_border_anchor_camera_quality"
BUDGET, SAMPLES = 180, 8192
SOURCE_FILES = ("infra/analytic_rgbd_evaluate.py", "infra/run_analytic_rgbd_evaluate.sh",
    "infra/tudl_anchor_evaluate.py", "infra/tudl_evaluate.py", "infra/run_tudl_anchor_evaluate.sh",
    "infra/run_tudl_anchor_evaluate_v2.sh", "infra/analytic_rgbd_render.py",
    "infra/run_analytic_rgbd_render.sh", *inference.SOURCE_FILES)
TRUTH_KEYS = frozenset({"depth", "visible", "K", "scene_id", "frame_id", "camera_R", "camera_t"})
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


def validate_pins(pins):
    if type(pins) is not dict or set(pins) != {"schema", "report", "outputs", "coefficients"} or pins["schema"] != "world_reward.analytic_rgbd_prediction_pins.v1":
        raise ValueError("Independent complete analytic prediction pins required")
    row = pins["report"]
    if type(row) is not dict or set(row) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Independent actual producer/source/report identity required")
    identity_record({k: row[k] for k in ("sha256", "bytes")})
    for k, n in (("producer_revision", 40), ("script_sha256", 64)):
        if type(row[k]) is not str or re.fullmatch(r"[0-9a-f]{%d}" % n, row[k]) is None:
            raise ValueError("Actual producer/source digest required")
    names = {Path(name).stem + ".npz" for name in inputs.filenames()}
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
        metric_scale_accuracy_verified=False, new_analytic_object_only_cohort=True, real_world_generalization_verified=False,
        human_contact_or_full_HOI_evaluated=False, human_or_object_geometry_changed=False, per_frame_anchor_fit=False,
        offset_fit=False, semantic_background_exclusion_performed=False, border_is_background_proxy_only=True,
        anchor_rule="median_of_four_per_image_median_MoGe_Z_div_DA3_Z_on_fixed_10percent_border_per_scene",
        border_definition="xx<64_or_xx>=576_or_yy<48_or_yy>=432", minimum_border_pairs=contract.MIN_PAIRS,
        minimum_border_coverage=contract.MIN_COVERAGE, fixed_camera_K=inference.GRID.K.tolist(),
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
    pp = code / PINS; pins = pinned_json(pp, inputs.identity(pp)); validate_pins(pins)
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
        "original_MoGe_arrays", "pointmap_checks", "scene_anchor", "border_proxy", "processed_camera_K", "metric_depth_factor"}
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
        ratio = contract.border_ratio(inference.GRID, data["moge_depth"], data["moge_validity"], data["da3_depth"])
        ratio.update(scene_id=scene, frame_id=frame); ratios.append(ratio)
        original = {k: data["moge_" + k] for k in ("depth", "points", "validity")}; original["K"] = data["K"]
        require(row, dict(arrays=contract.array_identities(data), original_MoGe_arrays=contract.array_identities(original), pointmap_checks=checks, border_proxy=ratio), "Original arrays/geometry/border metadata differs")
        processed = np.asarray(row.get("processed_camera_K"), np.float64)
        if not exact(row.get("metric_depth_factor"), contract.metric_factor(inference.GRID, processed)):
            raise ValueError("Original native metric focal scaling differs")
        for value in data.values(): value.flags.writeable = False
        arrays.append(data)
    coefficients = contract.scene_anchors(inference.GRID, ratios, list(inputs.ORDERED_FRAMES))
    require(receipt, dict(ratio_diagnostics=ratios, scene_anchor_coefficients=list(coefficients.values()),
        scene_ratio_medians=[[r["ratio_median"] for r in ratios[start:start + 4]] for start in (0, 4, 8)]), "Frozen equal-weight median4 rule differs")
    recheck(frozen)
    return arrays, records, frozen, public, pins


GEOMETRY_KEYS = frozenset({"semiaxes_m", "material_seed", "plane_z_m", "plane_extent_xy_m"})


def expected_pose(protocol, scene, frame):
    item = protocol["objects"][scene - 1]
    t = protocol["motion"]["instants"][frame]
    axis = np.array(protocol["motion"]["axis"], np.float64); axis /= np.linalg.norm(axis)
    x, y, z = axis; skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = item["start_angle_rad"] + item["angle_extent_rad"] * t
    rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    translation = np.array([-.04 + .08 * t, .015 * math.sin(2 * math.pi * t), item["depth_m"] + .06 * t])
    return rotation, translation


def validate_geometry(geometry, protocol, scene):
    if type(geometry) is not dict or set(geometry) != GEOMETRY_KEYS: raise ValueError("Exact analytic scene geometry required")
    item = protocol["objects"][scene - 1]
    expected = {"semiaxes_m": np.array(item["semiaxes_m"], np.float64),
        "material_seed": np.array(item["material_seed"], np.int64),
        "plane_z_m": np.array(item["background_z_m"], np.float64),
        "plane_extent_xy_m": np.array(protocol["appearance"]["background_extent_xy_m"], np.float64)}
    if any(not isinstance(geometry[k], np.ndarray) or np.ma.isMaskedArray(geometry[k])
            or geometry[k].dtype != value.dtype or not np.array_equal(geometry[k], value) for k, value in expected.items()):
        raise ValueError("Clip-constant original geometry/scale/plane/seed differs")
    return expected


def validate_analytic_truth(grid, truth, geometry, protocol, scene, frame):
    # Independent completed-square reference. No renderer imports or pixel drops.
    if type(truth) is not dict or set(truth) != TRUTH_KEYS: raise ValueError("Exact analytic truth arrays required")
    for k, shape, dtype in (("depth", grid.shape, np.float32), ("visible", grid.shape, np.bool_),
            ("K", (3, 3), np.float64), ("scene_id", (), np.int64), ("frame_id", (), np.int64),
            ("camera_R", (3, 3), np.float64), ("camera_t", (3,), np.float64)):
        a = truth[k]
        if not isinstance(a, np.ndarray) or np.ma.isMaskedArray(a) or a.shape != shape or a.dtype != dtype:
            raise ValueError("Exact original analytic truth shape/dtype required")
    g = validate_geometry(geometry, protocol, scene); rotation, translation = expected_pose(protocol, scene, frame)
    if (int(truth["scene_id"]) != scene or int(truth["frame_id"]) != frame or not np.array_equal(truth["K"], grid.K)
            or not np.allclose(truth["camera_R"], rotation, rtol=0, atol=1e-12)
            or not np.allclose(truth["camera_t"], translation, rtol=0, atol=1e-12)):
        raise ValueError("Frozen recipe rigid camera pose or IDs differ")
    yy, xx = np.indices(grid.shape)
    rays = np.stack(((xx + .5 - grid.K[0, 2]) / grid.K[0, 0], (yy + .5 - grid.K[1, 2]) / grid.K[1, 1], np.ones(grid.shape)), -1)
    # Camera ray transformed into object's unit-sphere metric.
    v = (rays @ rotation) / g["semiaxes_m"]
    origin = (-translation @ rotation) / g["semiaxes_m"]
    norm = np.einsum("...i,...i->...", v, v); projection = np.einsum("...i,i->...", v, origin) / norm
    closest = origin - projection[..., None] * v
    radius2 = 1. - np.einsum("...i,...i->...", closest, closest)
    tangent = np.abs(radius2) / np.maximum(1., np.einsum("...i,...i->...", closest, closest)) <= 1e-12
    if np.any(tangent): raise ValueError("Ambiguous analytic tangency: entire cohort STOP")
    nearest = -projection - np.sqrt(np.maximum(radius2, 0.) / norm)
    plane = float(g["plane_z_m"])
    visible = (radius2 > 0) & (nearest > .01) & (nearest < plane)
    z = np.where(visible, nearest, plane)
    hit = rays * plane
    if np.any(np.abs(hit[..., :2]) > g["plane_extent_xy_m"]): raise ValueError("Physical finite plane fails full-grid support")
    if (not np.isfinite(truth["depth"]).all() or np.any(truth["depth"] <= 0)
            or not np.array_equal(truth["visible"], visible)
            or np.max(np.abs(truth["depth"].astype(np.float64) - z)) > 1e-6):
        raise ValueError("Saved full-grid analytic nearest depth/visibility differs")
    return dict(checked_pixels=int(z.size), visible_pixels=int(visible.sum()),
        saved_depth_max_error_m=float(np.max(np.abs(truth["depth"].astype(np.float64) - z))))


def private_inputs(root, code, public, records):
    # ONLY after complete public_predictions, all twelve arrays and native metadata.
    private = root / BASE / "eval_private"; expected = public["input_pins"]["manufacture_report"]
    rid = {k: expected[k] for k in ("bytes", "sha256")}
    receipt = pinned_json(private / "render-report.json", rid)
    require(receipt, dict(schema="world_reward.analytic_rgbd_manufacture.v1", status="pass", phase="complete",
        stage="analytic_object_rgbd_manufacture", image_id=IMAGE_ID,
        producer_revision=expected["producer_revision"], script_sha256=expected["script_sha256"], budget_seconds=360,
        challenge_inputs_used=False, inference_performed=False, synthetic_transfer_only=True,
        public_files=public["input_pins"]["public_files"], public_manifest_sha256=public["input_manifest"]["sha256"]), "Successful new analytic manufacture required")
    render_helpers = {name: inputs.identity(code / name) for name in
        ("infra/analytic_rgbd_render.py", "infra/run_analytic_rgbd_render.sh")}
    require(receipt, {"source_helpers": render_helpers}, "Original manufacture source must remain byte-identical")
    if render_helpers["infra/analytic_rgbd_render.py"]["sha256"] != expected["script_sha256"]:
        raise ValueError("Independent manufacture source differs")
    protocol_path = code / "configs/analytic_rgbd_protocol.json"
    protocol_id = inputs.identity(protocol_path); protocol = pinned_json(protocol_path, protocol_id)
    require(receipt, {"protocol_identity": {"file": "configs/analytic_rgbd_protocol.json", **protocol_id}}, "Frozen complete recipe differs")
    require(protocol["private_evaluation"], dict(samples_per_frame=SAMPLES,
        sampling="all_if_count<=8192_else_sorted_PCG64_seed0_choice_without_replacement", minimum_visible_object_pixels=32,
        camera_pixel_rays="integer_indices_plus_0.5", minimum_median_relative_gain=.05,
        maximum_any_scene_relative_regression=.05, minimum_each_frame_object_coverage=.95, alignment=False), "Frozen quality gate differs")
    names = {Path(r["file"]).stem + ".npz" for r in records} | {f"scene_{scene:06d}_geometry.npz" for scene in (1, 2, 3)}
    files = receipt.get("private_files")
    if type(files) is not dict or set(files) != names or {p.name for p in private.iterdir()} != {*names, "render-report.json"}:
        raise ValueError("Exact complete private byte inventory required")
    frozen = [(private / "render-report.json", rid), (protocol_path, protocol_id)]
    for name, value in files.items():
        identity_record(value); path = private / name
        if stat.S_IMODE(path.stat().st_mode) != 0o400: raise ValueError("Original private arrays must remain400")
        frozen.append((path, value))
    recheck(frozen)  # ALL private bytes before ANY private arrays.
    cases = receipt.get("cases")
    if type(cases) is not list or len(cases) != 12: raise ValueError("All twelve original analytic gates required")
    geometries = {scene: read_npz(private / f"scene_{scene:06d}_geometry.npz", GEOMETRY_KEYS, 4096) for scene in (1, 2, 3)}
    for scene, geometry in geometries.items(): validate_geometry(geometry, protocol, scene)
    for record, case in zip(records, cases):
        require(case, dict(scene_id=record["scene_id"], frame_id=record["frame_id"], status="pass", checked_pixels=307200,
            closed_convex_positive_implicit_ellipsoid=True, independent_discriminant_silhouette_verified=True,
            first_hit_entering_derivative=True, decimal_precision=70, decimal_probes_checked=9), "Original analytic gate/order differs")
        bounds = {"independent_root_max_error_m": 1e-10, "independent_implicit_residual_max": 5e-12,
            "fp32_depth_cast_max_error_m": 1e-6, "decimal_root_max_error_m": 1e-10}
        if any(type(case.get(k)) is not float or not math.isfinite(case[k]) or not 0 <= case[k] <= limit for k, limit in bounds.items()):
            raise ValueError("Original independently preregistered precision gates required")
        if (type(case.get("discriminant_min_relative_margin")) is not float or not math.isfinite(case["discriminant_min_relative_margin"])
                or case["discriminant_min_relative_margin"] <= 1e-12
                or type(case.get("visible_pixels")) is not int or not 1024 <= case["visible_pixels"] < 307200):
            raise ValueError("Original entire-grid ambiguity/visible-support gates required")
    return private, frozen, cases, rid, protocol, geometries


def sample_indices(count):
    if count < 1: raise ValueError("Empty paired object support cannot be omitted")
    return np.arange(count) if count <= SAMPLES else np.sort(np.random.default_rng(0).choice(count, SAMPLES, replace=False))


def score_frame(grid, data, truth, scene, frame):
    if type(truth) is not dict or set(truth) != TRUTH_KEYS: raise ValueError("Exact original truth arrays required")
    for key, shape, dtype in (("depth", grid.shape, np.float32), ("visible", grid.shape, np.bool_),
            ("K", (3, 3), np.float64), ("scene_id", (), np.int64), ("frame_id", (), np.int64),
            ("camera_R", (3, 3), np.float64), ("camera_t", (3,), np.float64)):
        value = truth[key]
        if not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value) or value.shape != shape or value.dtype != dtype:
            raise ValueError("Original truth grid/dtype required")
    z, visible, k = truth["depth"], truth["visible"], truth["K"]
    if (int(truth["scene_id"]) != scene or int(truth["frame_id"]) != frame or not np.array_equal(k, grid.K)
            or not np.isfinite(z).all() or np.any(z <= 0) or not np.isfinite(truth["camera_R"]).all()
            or not np.isfinite(truth["camera_t"]).all() or not np.allclose(truth["camera_R"] @ truth["camera_R"].T, np.eye(3), atol=1e-12)
            or not np.isclose(np.linalg.det(truth["camera_R"]), 1., atol=1e-12)):
        raise ValueError("Original positive depth/shared camera/rigid pose/IDs required")
    count = int(visible.sum())
    if count < 32: raise ValueError("Each original frame needs32 visible object pixels")
    valid = data["moge_validity"]
    if not np.array_equal(valid, data["anchored_validity"]): raise ValueError("Candidate validity changed")
    common = visible & valid; indices = sample_indices(int(common.sum())); y, x = np.nonzero(common)
    depth = z[common].astype(np.float64)
    xyz = np.column_stack(((x + .5 - k[0, 2]) / k[0, 0] * depth, (y + .5 - k[1, 2]) / k[1, 1] * depth, depth))[indices]
    coverage = int(common.sum()) / count; scores = {}
    for mode in ("moge", "anchored"):
        points = data[mode + "_points"][common][indices].astype(np.float64)
        if not np.isfinite(points).all(): raise ValueError("Finite actual camera predictions required")
        directed = cKDTree(xyz).query(points, workers=1)[0].mean(), cKDTree(points).query(xyz, workers=1)[0].mean()
        errors = np.abs(data[mode + "_depth"][common].astype(np.float64) - z[common])
        scores[mode] = dict(visible_camera_chamfer_half_cm=float(.5 * sum(directed) * 100),
            camera_Z_absrel=float(np.mean(errors / z[common])), camera_Z_mae_cm=float(errors.mean() * 100),
            object_coverage=coverage)
    return dict(scene_id=scene, frame_id=frame, **scores, visible_object_pixels=count,
        paired_common_pixels=int(common.sum()), samples=len(indices), coverage_gate_pass=bool(coverage >= .95))


def decision(rows):
    if type(rows) is not list or [(r["scene_id"], r["frame_id"]) for r in rows] != list(inputs.ORDERED_FRAMES):
        raise ValueError("All twelve original frames in order must contribute")
    scenes = [dict(scene_id=scene, **{mode: float(np.mean([r[mode]["visible_camera_chamfer_half_cm"] for r in rows[(scene - 1) * 4:scene * 4]])) for mode in ("moge", "anchored")}) for scene in (1, 2, 3)]
    before, after = (np.array([r[mode] for r in scenes]) for mode in ("moge", "anchored"))
    if not np.isfinite(np.r_[before, after]).all() or np.any(before < 0) or np.any(after < 0): raise ValueError("Finite complete paired scores required")
    gain = (before - after) / before if np.all(before > 0) else None
    coverage = all(r["coverage_gate_pass"] is True for r in rows)
    return dict(scenes=scenes, median_paired_scene_relative_gain=float(np.median(gain)) if gain is not None else None,
        per_scene_relative_gain=gain.tolist() if gain is not None else None, coverage_gate_pass=coverage,
        analytic_object_depth_hypothesis_supported=bool(gain is not None and coverage and np.median(gain) >= .05 and np.min(gain) >= -.05),
        adoption_performed=False, synthetic_transfer_only=True, real_world_generalization_verified=False,
        full_HOI_verified=False, verified_victory_over_CARI4D=False)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    if (root != ROOT or code != root / "jobs" / revision / "run_analytic_rgbd_evaluate/code" or inputs._canonical(code) != code
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None or Path(__file__) != code / SOURCE_FILES[0]
            or os.environ["WR_IMAGE_ID"] != IMAGE_ID or platform.system() != "Linux"
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}):
        raise ValueError("Actual immutable CPU-only isolated private evaluation required")
    modules = {"infra/analytic_rgbd_infer.py": inference, "infra/tudl_anchor_evaluate.py": original_audit,
        "infra/tudl_evaluate.py": original_audit.scoring, "infra/analytic_rgbd_inputs.py": inputs,
        "infra/rgbd_anchor_contract.py": contract}
    if any(Path(module.__file__).resolve() != code / name for name, module in modules.items()):
        raise ValueError("Original exact frozen numerical helper paths required")
    helpers = {name: inputs.identity(code / name) for name in SOURCE_FILES}
    out = root / BASE / "quality_anchor_v1"
    if inputs._canonical(out) != out or not out.is_dir() or any(out.iterdir()): raise ValueError("Fresh exclusive evaluation output required")
    report = dict(stage=STAGE, status="fail", phase="public_validation", producer_revision=revision, image_id=IMAGE_ID,
        script_sha256=helpers[SOURCE_FILES[0]]["sha256"], source_helpers=helpers, budget_seconds=BUDGET,
        network="none", device="cpu", gpu_used=False, prediction_refit=False, alignment=False,
        challenge_inputs_used=False, private_truth_used_for_evaluation_only=True, synthetic_transfer_only=True)
    started = time.perf_counter(); frozen = []; error = None
    def expired(*_): raise TimeoutError("Whole private analytic evaluation exceeded180s")
    alarm, term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        arrays, records, frozen, public, pins = public_predictions(root, code)
        report.update(phase="private_evaluation", predictions_frozen_before_private_truth_read=True,
            prediction_report={k: pins["report"][k] for k in ("bytes", "sha256")}, scene_anchor_coefficients=pins["coefficients"])
        private, private_frozen, cases, manufacture_id, protocol, geometries = private_inputs(root, code, public, records); frozen += private_frozen
        rows = []
        for data, record, case in zip(arrays, records, cases):
            scene, frame = record["scene_id"], record["frame_id"]
            truth = read_npz(private / (Path(record["file"]).stem + ".npz"), TRUTH_KEYS, 6_000_000)
            reference = validate_analytic_truth(inference.GRID, truth, geometries[scene], protocol, scene, frame)
            row = score_frame(inference.GRID, data, truth, scene, frame); row["independent_analytic_reference"] = reference
            if row["visible_object_pixels"] != case["visible_pixels"]: raise ValueError("Original private visibility differs")
            rows.append(row)
        report.update(frames=rows, decision=decision(rows), manufacture_report_identity=manufacture_id, status="pass", phase="complete")
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
