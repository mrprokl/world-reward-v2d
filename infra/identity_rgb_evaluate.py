"""Paired identity quality, with private truth read after both modes freeze.

Raw camera-space human PVE is primary. Relative hand/object errors use one
common RGB-derived visible-object point proxy, not a predicted rigid mesh or
verified contact/penetration. No GT alignment or private inference input.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

from world_reward.data import sha256
from world_reward.metric_alignment import fit_shared_depth_scale

BASE = "validation/identity_rgb_v2"
STAGE = "private_paired_clip_identity_rgb_quality"
CLIPS, FRAMES, VERTICES, BUDGET = 3, 5, 18439, 120
GATES = {"median_human_relative_gain_min": .05, "worst_clip_human_regression_max": .05,
         "worst_clip_per_hand_interaction_increase_cm_max": 1e-4}


def sample_points(value):
    """Fixed seed, maximum8192 visible points; no residual-driven selection."""
    value = np.asarray(value)
    if len(value) <= 8192: return value
    return value[np.sort(np.random.default_rng(0).choice(len(value), 8192, replace=False))]


def visible_object_points(depth, ids, K, human_faces, object_faces):
    """Scene z-buffer unprojection at pixel centers, including human occlusion."""
    selected = (ids >= human_faces) & (ids < human_faces+object_faces)
    if selected.sum() < 32:
        raise ValueError("Private scene must expose finite visible object support")
    y, x = np.nonzero(selected); z = depth[selected].astype(np.float64)
    if not np.isfinite(z).all() or np.any(z <= 0): raise ValueError("Invalid visible object depth")
    p = np.c_[((x+.5-K[0, 2])/K[0, 0])*z, ((y+.5-K[1, 2])/K[1, 1])*z, z]
    return sample_points(p), int(selected.sum())


def regular_hash(path, expected=None, *, immutable=True):
    path = Path(path)
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Canonical regular frozen artifact required")
    if immutable and path.stat().st_mode & 0o222:
        raise ValueError("Predictions/validation artifacts must be read-only")
    digest = sha256(path)
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or digest != expected):
        raise ValueError("Frozen artifact SHA differs")
    return digest


def require_fields(report, expected):
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise ValueError("Complete source-bound producer contract differs")


def geometry(value, count, name):
    if np.ma.isMaskedArray(value): raise ValueError("Hidden geometry validity is forbidden")
    value = np.asarray(value)
    if value.shape != (count, 3) or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError("Finite complete camera-space geometry required: " + name)
    return value.astype(np.float64)


def paired_metrics(raw, shared, truth, object_proxy, true_object_visible, left, right):
    raw, shared, truth = (geometry(v, VERTICES, n) for v, n in
                          ((raw, "raw"), (shared, "shared"), (truth, "truth")))
    if any(np.ma.isMaskedArray(v) for v in (object_proxy, true_object_visible, left, right)):
        raise ValueError("Hidden point/hand validity is forbidden")
    p, q = np.asarray(object_proxy), np.asarray(true_object_visible)
    for value in (p, q):
        if value.ndim != 2 or value.shape[1] != 3 or not len(value) or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("Common finite visible-object point proxy required")
    hand_masks = (np.asarray(left), np.asarray(right))
    if any(m.dtype != np.bool_ or m.shape != (VERTICES,) or m.sum() < 50 for m in hand_masks) or np.any(hand_masks[0] & hand_masks[1]):
        raise ValueError("Actual disjoint semantic hand masks required")
    pc, qc = np.median(p, axis=0), np.median(q, axis=0)
    output = {}
    for name, human in (("raw", raw), ("shared", shared)):
        relative = [np.linalg.norm((pc-human[m].mean(0))-(qc-truth[m].mean(0))) * 100 for m in hand_masks]
        output[name] = {"human_pve_cm": float(np.linalg.norm(human-truth, axis=1).mean()*100),
                        "relative_hand_visible_object_vector_cm": float(np.mean(relative)),
                        "per_hand_relative_vector_cm": [float(v) for v in relative]}
    return output


def decision(clip_metrics):
    if len(clip_metrics) != CLIPS:
        raise ValueError("Every complete clip is required")
    h = np.array([[r[m]["human_pve_cm"] for m in ("raw", "shared")] for r in clip_metrics])
    g = np.array([[r[m]["relative_hand_visible_object_vector_cm"] for m in ("raw", "shared")] for r in clip_metrics])
    hands = np.array([[r[m]["per_hand_relative_vector_cm"] for m in ("raw", "shared")] for r in clip_metrics])
    if (hands.shape != (CLIPS, 2, 2) or not np.isfinite(np.r_[h.ravel(), g.ravel(), hands.ravel()]).all()
            or np.any(h < 0) or np.any(g < 0) or np.any(hands < 0)):
        raise ValueError("Finite nonnegative paired errors required")
    defined = bool(np.all(h[:, 0] > 0))
    gain = (h[:, 0]-h[:, 1])/h[:, 0] if np.all(h[:, 0] > 0) else None
    interaction_regression = (g[:, 1]-g[:, 0])/g[:, 0] if np.all(g[:, 0] > 0) else None
    gates = {"human_gain": gain is not None and float(np.median(gain)) >= GATES["median_human_relative_gain_min"],
             "human_nonregression": gain is not None and float((-gain).max()) <= GATES["worst_clip_human_regression_max"],
             "interaction_nonregression": float((hands[:, 1]-hands[:, 0]).max()) <= GATES["worst_clip_per_hand_interaction_increase_cm_max"]}
    return {"gates": gates, "median_human_relative_gain": float(np.median(gain)) if gain is not None else None,
            "per_clip_human_relative_gain": gain.tolist() if gain is not None else None,
            "per_clip_interaction_relative_regression": interaction_regression.tolist() if interaction_regression is not None else None,
            "per_clip_per_hand_interaction_increase_cm": (hands[:, 1]-hands[:, 0]).tolist(),
            "zero_baseline_relative_gain_undefined": not defined,
            "synthetic_identity_hypothesis_supported": bool(defined and all(gates.values())),
            "adoption_authorized": False, "full_HOI_verified": False, "real_domain_verified": False,
            "rigid_object_mesh_or_contact_verified": False}


def common_object_proxies(raw_frames):
    """A single baseline-human scale per clip, fixed for both identity modes."""
    if len(raw_frames) != CLIPS*FRAMES:
        raise ValueError("All public original frames required")
    proxies, diagnostics = [], []
    for clip in range(CLIPS):
        rows = raw_frames[clip*FRAMES:(clip+1)*FRAMES]
        if any(np.ma.isMaskedArray(v) for r in rows for v in r.values()):
            raise ValueError("Hidden observation validity is forbidden")
        visible = [r["silhouette"] & r["human_mask"] & ~r["object_mask"] & r["validity"] for r in rows]
        fit = fit_shared_depth_scale([r["raw_depth"] for r in rows], [r["rendered_depth"] for r in rows], visible, list(range(FRAMES)))
        if fit.supported_frames != FRAMES or any(not f.supported for f in fit.frames):
            raise ValueError("Every original frame needs baseline-human scale support")
        diagnostics.append({"clip_index": clip, **fit.to_dict()})
        for row in rows:
            selected = row["object_mask"] & row["validity"]
            if selected.sum() < 32:
                raise ValueError("Every original frame requires common public object support")
            p = row["raw_points"][selected].astype(np.float64)*fit.shared_scale
            if not np.isfinite(p).all() or np.any(p[:, 2] <= 0):
                raise ValueError("Common scaled object proxy invalid; no repair")
            proxies.append(sample_points(p))
    return proxies, diagnostics


def public_predictions(root):
    """Complete public byte/ABI audit before any evaluation-private read."""
    import identity_rgb_infer as inference
    records, inputs = inference.public_inputs(root)
    folder = Path(root)/BASE/"predictions_v2"
    rp = folder/"report.json"; digest = regular_hash(rp)
    report = json.loads(rp.read_text())
    require_fields(report, {"stage": inference.STAGE, "status": "pass", "phase": "complete",
        "network": "none", "ground_truth_used": False, "private_truth_read": False,
        "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "raw_frozen_before_shared": True, "body_calls_completed": 15, "MoGe_calls_completed": 15,
        "shared_head_calls_completed": 15, "official_reference_calls": 1,
        "paired_frozen_before_reference": True, "frames": 15, "all_cases_retained": True,
        "sources_assets_rechecked": True, "conversion_fidelity_verified": True,
        "actual_body_inference": True, "actual_MoGe_inference": True, "actual_shared_native_forward": True,
        "image_id": "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7",
        "camera_K": [[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]],
        "script_sha256": sha256(Path(inference.__file__)), "helper_source_sha256": inference.helper_identities()})
    revision = report.get("producer_revision")
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Immutable producing revision required")
    means = np.asarray(report.get("official_reference_per_frame_mean_mm"))
    if means.shape != (15,) or means.dtype.kind != "f" or not np.isfinite(means).all() or np.any(means < 0) or np.any(means > 2.):
        raise ValueError("All fifteen independent official reference gates must pass")
    raw_rows, pair_rows = report.get("raw_outputs"), report.get("paired_outputs")
    if not isinstance(raw_rows, list) or not isinstance(pair_rows, list) or len(raw_rows) != 15 or len(pair_rows) != 15:
        raise ValueError("All fifteen frozen raw and shared predictions required")
    raws, pairs, frozen = [], [], [(rp, digest)]
    for record, rr, pr in zip(records, raw_rows, pair_rows):
        for rows, name, output in ((rr, "raw", raws), (pr, "paired", pairs)):
            require_fields(rows, {"file": record["file"], "clip_index": record["clip_index"],
                "frame_index": record["frame_index"], "rgb_sha256": record["sha256"],
                "artifact": name+"/"+Path(record["file"]).stem+".npz"})
            path = folder/rows["artifact"]; h = regular_hash(path, rows.get("sha256"))
            if type(rows.get("bytes")) is not int or rows["bytes"] != path.stat().st_size:
                raise ValueError("Frozen prediction byte count differs")
            with np.load(path, allow_pickle=False) as archive: output.append({k: archive[k].copy() for k in archive.files})
            frozen.append((path, h))
        inference.validate_raw(raws[-1], record)
        inference.validate_pair(pairs[-1], record, raws[-1])
    for clip in range(CLIPS):
        anchor = raws[clip*FRAMES]
        for pair in pairs[clip*FRAMES:(clip+1)*FRAMES]:
            for shared, original in (("shared_shape_params", "shape_params"), ("shared_scale_params", "scale_params")):
                if pair[shared].tobytes() != anchor[original].tobytes():
                    raise ValueError("Shared identity must be exact first-RGB identity for every clip frame")
            if pair["shared_model_controls"][136:].tobytes() != pairs[clip*FRAMES]["shared_model_controls"][136:].tobytes():
                raise ValueError("Expanded shared68 scales must remain byte-constant")
            for key in ("human_faces", "hand_mask_left", "hand_mask_right", "camera_K"):
                if not np.array_equal(pair[key], pairs[0][key]):
                    raise ValueError("Paired topology/hand support/camera cannot change across frames")
    for name, rows in (("raw", raw_rows), ("paired", pair_rows)):
        if {p.name for p in (folder/name).iterdir()} != {Path(r["artifact"]).name for r in rows}:
            raise ValueError("Unexpected raw/paired prediction artifacts")
    if {p.name for p in folder.iterdir()} != {"raw", "paired", "report.json"}:
        raise ValueError("Unexpected public prediction directories")
    if inference.public_inputs(root) != (records, inputs):
        raise ValueError("Public inputs changed during prediction audit")
    inference.validate_producer_bindings(root, report)
    inference.validate_region_masks(pairs, report)
    for path, h in frozen: regular_hash(path, h)
    return records, inputs, raws, pairs, report, frozen, digest


def validate_truth(truth, record, faces):
    from joint_rgb_render import TRUTH_KEYS
    if set(truth) != TRUTH_KEYS:
        raise ValueError("Exact foreground-only evaluation truth required")
    geometry(truth["human_vertices_camera_m"], VERTICES, "truth")
    geometry(truth["object_vertices_camera_m"], 194, "object truth")
    hf, of = truth["human_faces"], truth["object_faces"]
    if (hf.dtype.kind not in "iu" or hf.shape != (36874, 3) or not np.array_equal(hf, faces)
            or np.any(hf < 0) or np.any(hf >= VERTICES) or of.dtype != np.int64 or of.shape != (384, 3)
            or np.any(of < 0) or np.any(of >= 194) or np.any(of[:, 0] == of[:, 1])
            or np.any(of[:, 1] == of[:, 2]) or np.any(of[:, 0] == of[:, 2])):
        raise ValueError("Human topology/correspondence must be identical, not fitted")
    for key in ("clip_index", "frame_index"):
        value = truth[key]
        if value.shape != () or value.dtype != np.int64 or int(value) != record[key]:
            raise ValueError("Private original clip/frame order differs")
    focal = (1160., 1480., 1720.)[record["clip_index"]]
    K = truth["camera_K"]
    if K.dtype != np.float64 or not np.array_equal(K, [[focal, 0., 512.], [0., focal, 384.], [0., 0., 1.]]):
        raise ValueError("Exact independent manufactured private camera required")
    z, ids = truth["scene_depth_m"], truth["visible_face_indices"]
    total = len(faces)+len(truth["object_faces"])
    if (z.shape != (768, 1024) or z.dtype != np.float32 or ids.shape != z.shape or ids.dtype != np.int64
            or np.any(ids < -1) or np.any(ids >= total) or not np.isnan(z[ids < 0]).all()
            or not np.isfinite(z[ids >= 0]).all() or np.any(z[ids >= 0] <= 0)):
        raise ValueError("Background cannot enter foreground object truth")


def run(root, report):
    records, inputs, raws, pairs, producer, frozen, digest = public_predictions(root)
    # This public-only baseline scale is common to both modes and never refit.
    proxies, alignment = common_object_proxies(raws)
    report.update(phase="public_predictions_audited", predictions_frozen_before_private=True,
        prediction_report_sha256=digest, common_object_scale_diagnostics=alignment)
    # FIRST private read, only after all thirty raw/paired artifacts validate.
    report["private_truth_used_for_evaluation_only"] = True
    private = Path(root)/BASE/"eval_private"; rp = private/"render-report.json"
    rh = regular_hash(rp); render = json.loads(rp.read_text())
    import identity_rgb_render as renderer
    import hand_synthetic_render as hand
    require_fields(render, {"stage": "own_moving_identity_rgb_render", "status": "pass", "phase": "complete",
        "frames": 15, "actual_MHR_reference_used": True, "actual_reference_forward_calls": 2,
        "challenge_inputs_used": False, "inference_performed": False, "accuracy_verified": False,
        "all_truth_private": True, "identity_clip_constant": True, "background_face_ids_removed_from_truth": True,
        "synthetic_truth_used_for_rendering_only": True, "photorealism_verified": False,
        "image_id": "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7",
        "public_manifest_sha256": inputs["public_manifest_sha256"],
        "script_sha256": sha256(Path(renderer.__file__)), "render_helper_sha256": sha256(Path(hand.__file__)),
        "joint_helper_sha256": sha256(Path(__file__).with_name("joint_rgb_render.py")),
        "camera_helper_sha256": sha256(Path(__file__).with_name("camera_render.py")), "model_sha256": hand.semantics.MODEL_SHA})
    if not isinstance(render.get("code_revision"), str) or not re.fullmatch("[0-9a-f]{40}", render["code_revision"]):
        raise ValueError("Immutable actual render revision required")
    semantic_path = private/"semantic-report.json"; sh = regular_hash(semantic_path, render.get("semantic_report_sha256"))
    semantic = json.loads(semantic_path.read_text()); hand.require_semantic_report(semantic)
    if semantic.get("source_image_id") != render.get("image_id"):
        raise ValueError("Exact renderer/semantic producer image binding required")
    frozen += [(rp, rh), (semantic_path, sh)]
    rig_path = private/"rig.npz"; rig_sha = regular_hash(rig_path, render.get("rig_sha256"))
    with np.load(rig_path, allow_pickle=False) as file: rig = {k: file[k].copy() for k in file.files}
    if set(rig) != {"controls", "shape45", "parameter_limits"}:
        raise ValueError("Exact own manufactured identity rig required")
    bounds = rig["parameter_limits"]
    if bounds.shape != (249, 2) or bounds.dtype.kind != "f" or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Actual native control limits required")
    controls, shapes, _ = renderer.named_controls(render.get("parameter_names"), bounds)
    if (rig["controls"].dtype != np.float32 or rig["shape45"].dtype != np.float32
            or not np.array_equal(rig["controls"], controls) or not np.array_equal(rig["shape45"], shapes)):
        raise ValueError("Fresh fixed manufactured identities/poses must reproduce the frozen recipe")
    frozen.append((rig_path, rig_sha))
    cases = render.get("cases")
    if not isinstance(cases, list) or len(cases) != 15:
        raise ValueError("All private original cases required")
    if {p.name for p in private.iterdir()} != {"rig.npz", "render-report.json", "semantic-report.json",
            *[Path(r["file"]).stem+".npz" for r in records]}:
        raise ValueError("Only declared private camera/geometry artifacts allowed")
    scores = []
    for record, pair, proxy, case in zip(records, pairs, proxies, cases):
        require_fields(case, {"clip_index": record["clip_index"], "frame_index": record["frame_index"],
                              "file": record["file"], "rgb_sha256": record["sha256"]})
        path = private/(Path(record["file"]).stem+".npz"); th = regular_hash(path, case.get("truth_sha256"))
        with np.load(path, allow_pickle=False) as file: truth = {k: file[k].copy() for k in file.files}
        validate_truth(truth, record, pair["human_faces"])
        visible, count = visible_object_points(truth["scene_depth_m"], truth["visible_face_indices"], truth["camera_K"],
                                               len(truth["human_faces"]), len(truth["object_faces"]))
        values = paired_metrics(pair["raw_vertices_camera_m"], pair["shared_vertices_camera_m"], truth["human_vertices_camera_m"],
                                proxy, visible, pair["hand_mask_left"], pair["hand_mask_right"])
        scores.append({"clip_index": record["clip_index"], "frame_index": record["frame_index"],
                       "true_visible_object_pixels": count, **values}); frozen.append((path, th))
    clips = []
    for clip in range(CLIPS):
        rows = scores[clip*FRAMES:(clip+1)*FRAMES]
        clips.append({"clip_index": clip, **{m: {
            **{key: float(np.mean([r[m][key] for r in rows])) for key in ("human_pve_cm", "relative_hand_visible_object_vector_cm")},
            "per_hand_relative_vector_cm": np.mean([r[m]["per_hand_relative_vector_cm"] for r in rows], axis=0).tolist()}
            for m in ("raw", "shared")}})
    for path, h in frozen: regular_hash(path, h)
    import identity_rgb_infer as inference
    if inference.public_inputs(root) != (records, inputs):
        raise ValueError("Public RGB/masks changed during private evaluation")
    report.update(status="pass", phase="complete", frame_metrics=scores, clip_metrics=clips, decision=decision(clips),
        private_truth_used_for_evaluation_only=True, private_render_report_sha256=rh,
        no_gt_alignment=True, all_frames_scored=True, all_cases_retained=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); out = root/BASE/"quality_v2"
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not out.is_dir() or out.resolve() != out.absolute() or any(out.iterdir())):
        raise ValueError("Fresh exclusive canonical offline CPU evaluation required")
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image):
        raise ValueError("Immutable producing source/image required")
    report = {"stage": STAGE, "status": "fail", "phase": "public_integrity", "producer_revision": revision,
        "image_id": image, "script_sha256": sha256(Path(__file__)), "budget_seconds": BUDGET,
        "network": "none", "challenge_inputs_used": False, "adoption_authorized": False, "accuracy_verified": False,
        "full_HOI_verified": False, "photorealism_verified": False, "predictions_frozen_before_private": False,
        "private_truth_used_for_evaluation_only": False, "hyperparameters": GATES,
        "object_estimate": "same public visible-depth point proxy for both modes; no rigid mesh/contact", "frame_metrics": []}
    start = time.perf_counter(); path = out/"report.json"
    with path.open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start; handle.seek(0); json.dump(report, handle, allow_nan=False)
            handle.write("\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Identity private quality exceeded120s")
        old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, report)
        except BaseException as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
