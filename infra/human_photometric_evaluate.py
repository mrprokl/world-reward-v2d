"""One private H101 diagnostic, after every public native artifact is frozen."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import human_photometric_observe as public
import human_photometric_render as manufacture
import joint_rgb_evaluate as alignment
from camera_render import raster_camera_mesh
from world_reward.data import sha256
from world_reward import human_photometric_metrics as metrics

COHORT = public.COHORT
BASE, OUT, STAGE, BUDGET = COHORT.base, COHORT.base + "/quality_v1", "private_human_photometric_paired_quality", 120
IMAGE = public.core.IMAGE
PIN_SCHEMA = "world-reward-human-photometric-quality-pins-v1"
ROLES = ("render", "masks", "body")
PRODUCER_KEYS = {"producer_revision", "report_sha256", "report_bytes", "script_sha256"}
MODES = metrics.MODES


def regular(path, expected=None, immutable=True):
    path = Path(path)
    if (not path.is_file() or path.resolve() != path.absolute() or any(p.is_symlink() for p in (path, *path.parents))
            or (immutable and path.stat().st_mode & 0o222)):
        raise ValueError("Canonical frozen regular file required")
    value = dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)
    if expected is not None and value["sha256"] != expected: raise ValueError("Frozen file SHA differs")
    return value


def load_pins(path):
    receipt = regular(path); data = json.loads(Path(path).read_text())
    if type(data) is not dict or set(data) != {"schema", *ROLES} or data["schema"] != PIN_SCHEMA:
        raise ValueError("Explicit complete quality producer pins required")
    for role in ROLES:
        row = data[role]
        if type(row) is not dict or set(row) != PRODUCER_KEYS: raise ValueError("Exact producer pin fields required")
        for field, count in (("producer_revision", 40), ("report_sha256", 64), ("script_sha256", 64)):
            if type(row[field]) is not str or not re.fullmatch("[0-9a-f]{" + str(count) + "}", row[field]):
                raise ValueError("Actual completed source/receipt hashes required")
        if type(row["report_bytes"]) is not int or row["report_bytes"] <= 0: raise ValueError("Positive receipt bytes required")
    if data["masks"]["producer_revision"] != data["body"]["producer_revision"] or data["masks"]["script_sha256"] != data["body"]["script_sha256"]:
        raise ValueError("Same frozen automatic-mask/native observer source required")
    return data, receipt


def producer_source(root, pin, role):
    stem, file = ("run_human_photometric_prepare", "human_photometric_render.py") if role == "render" else ("run_human_photometric_observe", "human_photometric_observe.py")
    return Path(root) / "jobs" / pin["producer_revision"] / stem / "code" / "infra" / file


def pinned_report(path, pin):
    receipt = regular(path, pin["report_sha256"])
    if receipt["bytes"] != pin["report_bytes"]: raise ValueError("Pinned receipt size differs")
    data = json.loads(Path(path).read_text())
    if type(data) is not dict or any(k in data for k in ("error", "error_type")): raise ValueError("Completed passing producer only")
    return data, receipt


def helper_hashes():
    return dict(evaluator=sha256(Path(__file__)), public=public.helper_hashes(), manufacture=manufacture.helper_hashes(),
        alignment=sha256(Path(alignment.__file__)), camera=sha256(Path(__file__).with_name("camera_render.py")), metrics=sha256(Path(metrics.__file__)))


def public_predictions(root, pins):
    """Complete24/360/source/model/1392-scope audit BEFORE private label access."""
    root = Path(root); frozen = []
    for role in ROLES:
        expected = Path(manufacture.__file__) if role == "render" else Path(public.__file__)
        if pins[role]["script_sha256"] != sha256(expected): raise ValueError("Actual producer source must match the checked module")
        frozen.append(regular(producer_source(root, pins[role], role), pins[role]["script_sha256"]))
    records, inputs = public.public_masks(root, pins["masks"]["producer_revision"])
    mask_path = root / BASE / "automatic_masks/report.json"; mask_report, mask_receipt = pinned_report(mask_path, pins["masks"]); frozen.append(mask_receipt)
    public.completed_masks(mask_report)
    for key in public.masks.ASSETS:
        asset = regular(root / "weights" / key, immutable=False); expected = mask_report["model_assets"][key]
        if asset != expected: raise ValueError("Exact original detector/SAM2 assets differ")
        frozen.append(asset)
    frozen.extend((regular(root / "results/image-grounding.json", immutable=False), regular(root / "results/weights-acquisition.json", immutable=False)))
    if json.loads((root / "results/image-grounding.json").read_text()).get("Id") != public.core.MASK_IMAGE:
        raise ValueError("Exact frozen grounding image receipt required")
    body_path = root / BASE / "predictions_v1/report.json"; body, body_receipt = pinned_report(body_path, pins["body"]); frozen.append(body_receipt)
    public.core.require_fields(body, dict(stage=public.STAGES["body"], status="pass", phase="complete", cohort=asdict(COHORT), frames=24,
        producer_revision=pins["body"]["producer_revision"], image_id=IMAGE, script_sha256=pins["body"]["script_sha256"], source_helpers=public.helper_hashes(),
        network="none", private_truth_read=False, ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        fitting_performed=False, camera_fit_performed=False, geometry_averaged=False, all_cases_retained=True, source_inputs_assets_rehashed=True,
        quality_verified=False, accuracy_verified=False, adoption_authorized=False, full_HOI_verified=False, public_inputs=inputs,
        seed=0, torch_version="2.5.1+cu124", CUDA_version="12.4", CUBLAS_WORKSPACE_CONFIG=":4096:8"))
    public.completed_body(body)
    model = body.get("body_model")
    if type(model) is not dict: raise ValueError("Actual strict native model loading evidence required")
    public.core.require_fields(model, dict(body_revision=public.native.human.body.BODY_REVISION, upstream_revision=public.native.human.body.UPSTREAM_REVISION,
        dinov3_revision=public.native.human.body.DINOV3_REVISION, inference_source_identity=public.native.human.body._source_identity(root),
        body_assets=public.native.human.body._body_assets(root)[1]))
    public.core.require_fields(model.get("checkpoint_loading"), dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention", parameter_tensors_loaded=1101, unexpected_keys=[]))
    retained = model["checkpoint_loading"].get("retained_mhr_asset_buffer_names")
    if type(retained) is not list or len(retained) != 113 or len(set(retained)) != 113: raise ValueError("Exact113 retained immutable asset buffers required")
    if model["body_assets"]["model.ckpt"] != dict(sha256=public.native.human.BODY_SHA, bytes=public.native.human.BODY_BYTES): raise ValueError("Original Body checkpoint differs")
    if model["body_assets"]["assets/mhr_model.pt"] != dict(sha256=manufacture.MODEL_SHA, bytes=manufacture.MODEL_BYTES): raise ValueError("Same pinned native MHR asset required")
    semantic_path = root / "results/mhr-finger-semantics-v4.json"; semantic_id = regular(semantic_path, body["semantic_report_sha256"], immutable=False)
    semantic = json.loads(semantic_path.read_text()); public.native.regions_helper.require_semantic_report(semantic)
    if semantic.get("source_image_id") != IMAGE or body.get("joint_names") != semantic["joint_names"]: raise ValueError("Actual semantic names/source image differ")
    frozen.append(semantic_id); directory = public.native.human.body._body_assets(root)[0]
    frozen.extend(regular(directory / name, value["sha256"], immutable=False) for name, value in model["body_assets"].items())
    out = root / BASE / "predictions_v1"; rig_path = out / "rig_metadata.npz"
    if public.core.identity(rig_path) != body.get("rig_metadata"): raise ValueError("Frozen native rig metadata changed")
    with np.load(rig_path, allow_pickle=False) as saved: rig = public.validate_rig({k: saved[k] for k in saved.files})
    if body.get("hand_region_sha256") != {side:manufacture.array_sha(rig["hand_mask_" + side]) for side in ("left", "right")}:
        raise ValueError("Actual named LBS region hashes changed")
    frozen.append(regular(rig_path)); frames = public.frozen_frames(out, body["records"], records, body["rig_metadata"])
    for record, row in zip(records, body["records"]):
        frozen.append(regular(record["path"], record["sha256"])); frozen.append(regular(record["human_mask_path"], record["human_mask_sha256"]))
        folder = out / Path(record["file"]).stem
        frozen.extend(regular(folder / r["file"], r["identity"]["sha256"]) for r in row["artifacts"])
    frozen.append(regular(root / BASE / "inputs/manifest.json", inputs["manifest"]["sha256"]))
    return records, frames, rig, body, inputs, frozen


def rendering(root, pins, records, rig, body, inputs):
    private = Path(root) / BASE / "eval_private"; report, receipt = pinned_report(private / "render-report.json", pins["render"])
    public.core.require_fields(report, dict(stage=manufacture.STAGE, status="pass", phase="complete", frames=24, groups=8,
        code_revision=pins["render"]["producer_revision"], image_id=IMAGE, script_sha256=pins["render"]["script_sha256"], helper_source_sha256=manufacture.helper_hashes(),
        network="none", budget_seconds=120, challenge_inputs_used=False, inference_performed=False, optimizer_performed=False,
        synthetic_truth_used_for_rendering_only=True, all_truth_private=True, accuracy_verified=False, photorealism_verified=False,
        truth_keys=sorted(manufacture.TRUTH_KEYS), human_faces_sha256=manufacture.FACE_SHA, human_faces_dtype="int32",
        reference_forward_attempts=2, reference_forward_returns=2, bundled_forward_attempts=1, bundled_forward_returns=1,
        actual_reference_forward_calls=2, actual_bundled_forward_calls=1, actual_total_native_forward_calls=3, raster_attempts=24, raster_returns=24,
        bundled_reference_same_joint_names=True, bundled_reference_same_topology=True, all_factors_geometry_independent=True,
        public_manifest_sha256=inputs["manifest"]["sha256"], public_manifest_bytes=inputs["manifest"]["bytes"], source_assets_rechecked=True,
        semantic_report_sha256=body["semantic_report_sha256"], joint_names=body["joint_names"], shape_first_two=[list(v) for v in manufacture.SHAPES],
        scale68_scalars=list(manufacture.SCALES), bottle_scale=manufacture.BOTTLE_SCALE, nuisance_Z_offsets=list(manufacture.OCCLUSION_OFFSETS), native_elbow_name=manufacture.ELBOW_NAME))
    errors = report.get("bundled_reference_parity")
    if type(errors) is not dict or set(errors) != {"vertices_m", "joints_m"} or any(type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v <= 1e-5 for v in errors.values()):
        raise ValueError("Actual bundled/reference parity proof required")
    actual_rig = report.get("rig_identity")
    if type(actual_rig) is not dict or actual_rig.get("model_sha256") != manufacture.MODEL_SHA or actual_rig.get("bundled_model_sha256") != manufacture.MODEL_SHA or actual_rig.get("model_bytes") != manufacture.MODEL_BYTES or actual_rig.get("bundled_model_bytes") != manufacture.MODEL_BYTES:
        raise ValueError("Exact same reference/bundled model bytes required")
    if actual_rig.get("body_revision") != public.native.human.body.BODY_REVISION or actual_rig.get("acquisition_report_sha256") != sha256(Path(root) / "results/weights-acquisition.json"):
        raise ValueError("Pinned native manufacturing acquisition differs")
    if report.get("native_elbow_index") != report["joint_names"].index(manufacture.ELBOW_NAME): raise ValueError("Exact named manufacturing elbow required")
    cases = report.get("cases")
    if type(cases) is not list or len(cases) != 24: raise ValueError("All24 source-bound private manufacturing rows required")
    truths = []
    for record, row in zip(records, cases):
        public.core.require_fields(row, dict(file=record["file"], rgb_sha256=record["sha256"], group_index=record["group_index"], frame_index=record["frame_index"],
            morphology_index=record["group_index"] // 4, appearance_index=(record["group_index"] // 2) % 2, occlusion_index=record["group_index"] % 2))
        if type(row.get("truth_sha256")) is not str or not re.fullmatch("[0-9a-f]{64}", row["truth_sha256"]): raise ValueError("Frozen private truth SHA required")
        if type(row.get("visible_nuisance_pixels")) is not int or row["visible_nuisance_pixels"] < 64: raise ValueError("Actual nuisance support required")
        path = private / (Path(record["file"]).stem + ".npz"); truths.append(regular(path, row["truth_sha256"]))
    rig_path = private / "rig.npz"; rig_receipt = regular(rig_path, report["rig_sha256"])
    with np.load(rig_path, allow_pickle=False) as saved: manufacturing = {k:saved[k] for k in saved.files}
    if set(manufacturing) != {"controls", "shape45", "parameter_limits", "hand_mask_left", "hand_mask_right"}: raise ValueError("Exact private manufacturing rig required")
    expected_p, expected_s, changes = manufacture.named_controls(report["parameter_names"], manufacturing["parameter_limits"])
    if (manufacturing["controls"].dtype != np.float32 or manufacturing["shape45"].dtype != np.float32
            or not np.array_equal(manufacturing["controls"], expected_p) or not np.array_equal(manufacturing["shape45"], expected_s)
            or any(manufacturing["hand_mask_" + s].dtype != np.bool_ or not np.array_equal(manufacturing["hand_mask_" + s], rig["hand_mask_" + s]) for s in ("left", "right"))
            or report.get("named_controls") != changes): raise ValueError("Unchanged prescribed bounded rig/actual hand regions required")
    if {p.name for p in private.iterdir()} != {"render-report.json", "rig.npz", *[Path(r["file"]).stem + ".npz" for r in records]}:
        raise ValueError("Exact private manufacturing inventory required")
    return report, truths, [receipt, rig_receipt, *truths]


def validate_truth(data, record, rig):
    if set(data) != manufacture.TRUTH_KEYS or any(np.ma.isMaskedArray(v) for v in data.values()): raise ValueError("Exact unmasked private truth schema required")
    for key, shape in (("human_vertices_camera_m", (18439, 3)), ("human_joints_camera_m", (127, 3)), ("camera_K", (3, 3))):
        value = data[key]
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all(): raise ValueError("Finite native private geometry required")
    if np.any(data["human_vertices_camera_m"][:, 2] <= .01) or np.any(data["human_joints_camera_m"][:, 2] <= .01): raise ValueError("Positive private camera geometry required")
    faces = data["human_faces"]
    if (faces.dtype != np.int32 or faces.shape != (36874, 3) or faces.min() < 0 or faces.max() >= 18439
            or manufacture.array_sha(faces) != manufacture.FACE_SHA or not np.array_equal(faces, rig["human_faces"])):
        raise ValueError("Exact I32 private/source native topology required")
    if not np.array_equal(data["camera_K"], np.asarray(COHORT.fixed_K)): raise ValueError("Fixed original private camera required")
    for key in ("group_index", "frame_index"):
        if data[key].shape != () or data[key].dtype != np.int64 or data[key].item() != record[key]: raise ValueError("Original private frame index required")
    z, ids, visible = data["scene_depth_m"], data["visible_face_indices"], data["human_visibility"]
    shape = (COHORT.height, COHORT.width)
    if (z.dtype != np.float32 or z.shape != shape or ids.dtype != np.int64 or ids.shape != shape or visible.dtype != np.bool_ or visible.shape != shape
            or visible.sum() < 64 or ids.min() < -1 or ids.max() >= len(faces) or not np.array_equal(visible, ids >= 0)
            or not np.isfinite(z[visible]).all() or np.any(z[visible] <= 0) or not np.isnan(z[~visible]).all()):
        raise ValueError("Original human-only visible depth/face grid required")
    return data


def shared_group_alignment(first_baseline, first_truth):
    sim = alignment.fit_human_sim3(first_baseline, first_truth)
    R = np.asarray(sim["rotation"]); scale = sim["scale"]
    if R.shape != (3, 3) or not np.isclose(np.linalg.det(R), 1., atol=1e-10) or scale <= 0: raise ValueError("Proper positive shared Sim3 required")
    return sim


def frame_metrics(prediction, truth, rig, sim):
    p = prediction["vertices_camera_m"]; q = truth["human_vertices_camera_m"]
    aligned = alignment.transform(p, sim); joints = alignment.transform(prediction["joints_camera_m"], sim)
    hands = [float(np.linalg.norm(aligned[rig["hand_mask_" + s]] - q[rig["hand_mask_" + s]], axis=1).mean() * 100) for s in ("left", "right")]
    return dict(raw_camera=metrics.geometry_diagnostic(p, q), aligned_human_pve_cm=float(np.linalg.norm(aligned - q, axis=1).mean() * 100),
        aligned_joint_pve_cm=float(np.linalg.norm(joints - truth["human_joints_camera_m"], axis=1).mean() * 100), aligned_hand_pve_cm=hands), joints


def raw_iou(prediction, rig, target):
    mask, _ = raster_camera_mesh(prediction["vertices_camera_m"], rig["human_faces"], rig["camera_K"], COHORT.width, COHORT.height)
    mask = mask.detach().cpu().numpy()
    if mask.dtype != np.bool_ or mask.shape != target.shape or target.dtype != np.bool_: raise ValueError("Actual original-grid raw silhouette required")
    union = np.count_nonzero(mask | target)
    if not union: raise ValueError("Complete automatic/predicted human support required")
    return float(np.count_nonzero(mask & target) / union)


def recheck(rows):
    for row in rows:
        if regular(row["path"], row["sha256"], immutable=False) != row: raise ValueError("Audited frozen source/input/output changed")


def run(root, pins, report, persist):
    helpers = helper_hashes(); records, frames, rig, producer, inputs, frozen = public_predictions(root, pins)
    report.update(phase="public_predictions_audited", all_public_artifacts_frozen_before_private=True,
        public_manifest_sha256=inputs["manifest"]["sha256"], body_report_sha256=pins["body"]["report_sha256"], native_scoped_heads_verified=public.SCOPED_HEADS); persist()
    report["private_truth_read"] = True; persist()
    rendering_report, truths, private_frozen = rendering(root, pins, records, rig, producer, inputs)
    frozen += private_frozen
    if "torch" in sys.modules: raise ValueError("Evaluation GPU runtime imported prematurely")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from PIL import Image
    if not torch.cuda.is_available(): raise ValueError("Native raw-camera CUDA rasters required")
    torch.use_deterministic_algorithms(True, warn_only=False); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    pve = np.empty((24, 3), np.float64); hands = np.empty((24, 3, 2), np.float64); iou = np.empty((24, 3), np.float64)
    all_joints = np.empty((8, 3, 3, 127, 3), np.float64); true_joints = np.empty((8, 3, 127, 3), np.float64)
    group_sim, geometry, visibility = {}, {}, {}; contrasts = []; output = []
    for i, (record, frame, truth_row) in enumerate(zip(records, frames, truths)):
        with np.load(truth_row["path"], allow_pickle=False) as saved: truth = validate_truth({k:saved[k] for k in saved.files}, record, rig)
        group, pose = record["group_index"], record["frame_index"]
        if pose == 0: group_sim[group] = shared_group_alignment(frame["baseline"]["vertices_camera_m"], truth["human_vertices_camera_m"])
        geom = dict(vertices_sha256=manufacture.array_sha(truth["human_vertices_camera_m"]), joints_sha256=manufacture.array_sha(truth["human_joints_camera_m"]))
        key = (group // 4, pose)
        if key in geometry and geometry[key] != geom: raise ValueError("Manufactured human geometry differs across appearance/occlusion")
        geometry[key] = geom
        if rendering_report["cases"][i].get("geometry") != geom: raise ValueError("Private geometry hash differs from rendering receipt")
        if rendering_report["cases"][i].get("visible_human_pixels") != int(truth["human_visibility"].sum()):
            raise ValueError("Actual private visible support differs from render receipt")
        contrast_key = (group // 4, (group // 2) % 2, pose)
        if group % 2 == 0: visibility[contrast_key] = truth["human_visibility"]
        else: contrasts.append(dict(morphology_index=group // 4, appearance_index=(group // 2) % 2, frame_index=pose,
            **manufacture.occlusion_evidence(visibility.pop(contrast_key), truth["human_visibility"])))
        automatic = public.native.joint.read_mask(record["human_mask_path"], Image) > 0
        if int(np.count_nonzero(automatic)) != record["human_mask_pixels"]: raise ValueError("Automatic mask support changed")
        row = dict(file=record["file"], group_index=group, frame_index=pose, modes={})
        for mode_index, mode in enumerate(MODES):
            values, joint_values = frame_metrics(frame[mode], truth, rig, group_sim[group])
            report["raster_attempts"] += 1; persist(); overlap = raw_iou(frame[mode], rig, automatic); report["raster_returns"] += 1
            values["raw_camera_automatic_human_iou"] = overlap; row["modes"][mode] = values
            pve[i, mode_index] = values["aligned_human_pve_cm"]; hands[i, mode_index] = values["aligned_hand_pve_cm"]; iou[i, mode_index] = overlap
            all_joints[group, pose, mode_index] = joint_values
        true_joints[group, pose] = truth["human_joints_camera_m"]; output.append(row); report.update(frames_scored=len(output)); persist()
    if visibility or contrasts != rendering_report.get("occlusion_evidence"): raise ValueError("All12 unchanged geometry/visibility contrasts required")
    group_pve = pve.reshape(8, 3, 3).mean(1); group_hands = hands.reshape(8, 3, 3, 2).mean(1)
    decision = metrics.quality_decision(group_pve, iou, group_hands)
    accelerations = {mode:metrics.second_difference_diagnostic(all_joints[:, :, j]).tolist() for j, mode in enumerate(MODES)}
    accelerations["truth"] = metrics.second_difference_diagnostic(true_joints).tolist()
    recheck(frozen); records_after, _, _, _, inputs_after, _ = public_predictions(root, pins)
    if records_after != records or inputs_after != inputs or helper_hashes() != helpers: raise ValueError("Full public/helper lineage changed after private evaluation")
    report.update(phase="complete", status="pass", frame_metrics=output, group_aligned_human_pve_cm=group_pve.tolist(), group_aligned_hand_pve_cm=group_hands.tolist(),
        frame_raw_camera_automatic_human_iou=iou.tolist(), shared_group_Sim3=[group_sim[g] for g in range(8)], second_joint_difference_cm=accelerations,
        decision=decision, actual_raster_calls=72, all_public_private_sources_rehashed=True, adoption_authorized=False, full_HOI_verified=False,
        learned_inference_calls=0, optimizer_calls=0, alignment_shared_all_methods=True, primary_private_diagnostic_completed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False); parser.add_argument("--quality-pins", required=True); args = parser.parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); code = Path(os.environ["WR_CODE"]); out = root / OUT; revision = os.environ["WR_CODE_REVISION"]
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or os.geteuid() != 1000
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.environ["WR_IMAGE_ID"] != IMAGE
            or not re.fullmatch("[0-9a-f]{40}", revision) or not out.is_dir() or any(out.iterdir()) or out.resolve() != out.absolute()
            or any(p.is_symlink() for p in (out, *out.parents)) or Path(args.quality_pins) != code / "configs/human_photometric_quality_pins.json"):
        raise ValueError("Fresh canonical offline source-bound quality stage required")
    pins, pin_receipt = load_pins(args.quality_pins)
    report = dict(stage=STAGE, status="fail", phase="public_integrity", producer_revision=revision, image_id=IMAGE, script_sha256=sha256(Path(__file__)),
        helper_source_sha256=helper_hashes(), quality_pins_sha256=pin_receipt["sha256"], network="none", budget_seconds=BUDGET, frames_scored=0,
        private_truth_read=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[], fitting_performed=False, learned_inference_calls=0,
        optimizer_calls=0, raster_attempts=0, raster_returns=0, adoption_authorized=False, full_HOI_verified=False)
    receipt = out / "report.json"; started = time.perf_counter()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Private human photometric diagnostic exceeded120s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, pins, report, persist)
            if regular(args.quality_pins) != pin_receipt: raise ValueError("Explicit pinned configuration changed")
        except BaseException as error: report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o444)


if __name__ == "__main__": main()
