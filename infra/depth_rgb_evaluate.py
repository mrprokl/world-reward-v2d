"""Private all-visible-pixel Z comparison after both public producers freeze.

No ground-truth alignment, intrinsics input, trimming or invalid-pixel penalty.
An incomplete native object support makes the primary metric undefined and
rejects the whole comparison, rather than rewarding removal of difficult pixels.
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

BASE = "validation/depth_rgb_v1"
STAGE = "private_all_visible_object_depth_rgb_quality"
CLIPS, FRAMES, BUDGET = 3, 5, 120
GATES = {"median_object_relative_gain_min": .05, "worst_clip_object_regression_max": .05,
         "object_visible_pixel_coverage_required": 1.0}


def regular_hash(path, expected=None):
    path = Path(path)
    if path.resolve() != path.absolute() or not path.is_file() or path.stat().st_mode & 0o222:
        raise ValueError("Canonical immutable regular artifact required")
    digest = sha256(path)
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or digest != expected):
        raise ValueError("Frozen artifact SHA differs")
    return digest


def require_fields(report, expected):
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Complete source-bound stage contract differs")


def visible_z_metric(predicted, validity, truth, support):
    """Uncapped relative Z on every fixed visible pixel, or explicit incompleteness."""
    if any(np.ma.isMaskedArray(v) for v in (predicted, validity, truth, support)):
        raise ValueError("Hidden observation validity is forbidden")
    p, v, z, s = (np.asarray(x) for x in (predicted, validity, truth, support))
    if (p.ndim != 2 or p.shape != z.shape or p.shape != v.shape or p.shape != s.shape
            or p.dtype != np.float32 or z.dtype != np.float32 or v.dtype != np.bool_ or s.dtype != np.bool_
            or s.sum() <= 64 or not np.isfinite(z[s]).all() or np.any(z[s] <= 0)):
        raise ValueError("Complete positive private Z/support and native grids required")
    accepted = v & np.isfinite(p) & (p > 0)
    count, invalid = int(s.sum()), int(np.count_nonzero(s & ~accepted))
    result = {"visible_pixels": count, "invalid_pixels": invalid, "coverage": (count-invalid)/count,
              "mean_absolute_relative_z": None, "mean_absolute_z_cm": None, "complete": invalid == 0}
    if invalid == 0:
        error = np.abs(p[s].astype(np.float64)-z[s].astype(np.float64))
        result.update(mean_absolute_relative_z=float(np.mean(error/z[s].astype(np.float64))),
                      mean_absolute_z_cm=float(np.mean(error)*100))
    return result


def decision(frame_metrics):
    if len(frame_metrics) != CLIPS*FRAMES:
        raise ValueError("All fifteen original metric records required")
    means, clips = [], []
    for clip in range(CLIPS):
        rows = frame_metrics[clip*FRAMES:(clip+1)*FRAMES]
        if any(type(r.get("clip_index")) is not int or r["clip_index"] != clip
               or type(r.get("frame_index")) is not int or r["frame_index"] != frame for frame, r in enumerate(rows)):
            raise ValueError("Original unaltered frame order required")
        pair = []
        for backend in ("moge", "da3"):
            values = [r[backend]["object"] for r in rows]
            for value in values:
                count, invalid = value.get("visible_pixels"), value.get("invalid_pixels")
                if (type(count) is not int or count <= 64 or type(invalid) is not int or not 0 <= invalid <= count
                        or type(value.get("coverage")) is not float or value["coverage"] != (count-invalid)/count
                        or type(value.get("complete")) is not bool or value["complete"] != (invalid == 0)
                        or (invalid > 0 and value.get("mean_absolute_relative_z") is not None)):
                    raise ValueError("Exact native coverage/count/completeness metric contract required")
            complete = all(v.get("complete") is True and v.get("invalid_pixels") == 0 and v.get("coverage") == 1.0 for v in values)
            numeric = [v.get("mean_absolute_relative_z") for v in values]
            if complete and any(type(v) is not float or not np.isfinite(v) or v < 0 for v in numeric):
                raise ValueError("Complete support requires finite nonnegative primary metrics")
            pair.append(float(np.mean(numeric)) if complete else None)
        means.append(pair); clips.append({"clip_index": clip, "moge_mean_absolute_relative_z": pair[0], "da3_mean_absolute_relative_z": pair[1]})
    complete = all(v is not None for pair in means for v in pair)
    gains = None
    if complete:
        values = np.asarray(means)
        if np.all(values[:, 0] > 0): gains = (values[:, 0]-values[:, 1])/values[:, 0]
    gates = {"all_native_object_support_complete": complete,
             "object_gain": gains is not None and float(np.median(gains)) >= GATES["median_object_relative_gain_min"],
             "object_nonregression": gains is not None and float((-gains).max()) <= GATES["worst_clip_object_regression_max"]}
    return {"clip_metrics": clips, "gates": gates, "per_clip_object_relative_gain": gains.tolist() if gains is not None else None,
            "median_object_relative_gain": float(np.median(gains)) if gains is not None else None,
            "relative_gain_undefined": gains is None, "synthetic_object_z_hypothesis_supported": bool(all(gates.values())),
            "adoption_authorized": False, "human_accuracy_verified": False, "calibration_accuracy_verified": False,
            "full_HOI_verified": False, "rigid_object_mesh_or_contact_verified": False, "real_domain_verified": False}


def public_predictions(root):
    import depth_rgb_infer as inference
    records, inputs = inference.public_inputs(root)
    outputs, frozen, reports = {}, [], {}
    for backend in ("moge", "da3"):
        folder = Path(root)/BASE/(backend+"_predictions_v1")
        rp = folder/"report.json"; digest = regular_hash(rp); report = json.loads(rp.read_text())
        require_fields(report, {"stage": inference.STAGE, "backend": backend, "status": "pass", "phase": "complete",
            "network": "none", "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
            "hand_labeled_test": False, "oracle_modes": [], "camera_K": inference.CAMERA_K.tolist(),
            "camera_fitted": False, "scale_fit": False, "identity_fit": False, "alignment_performed": False,
            "human_inference_performed": False, "masks_used": False, "pointmap_geometry_filled": False,
            "actual_network_calls": 15, "frames": 15, "all_cases_retained": True, "actual_native_inference": True,
            "sources_assets_rechecked": True, "image_id": inference.IMAGES[backend], **inputs})
        if not re.fullmatch("[0-9a-f]{40}", str(report.get("producer_revision", ""))):
            raise ValueError("Immutable actual producing revision required")
        if backend == "da3":
            require_fields(report, {"checkpoint_states_loaded": 406, "checkpoint_load_strict": True,
                "native_sky_correction_unchanged": True, "native_sky_threshold": .3, "native_sky_quantile": .99,
                "confidence_validity_exclusion": False, "metric_depth_scaling_applied_once": True,
                "native_processed_grid": list(inference.PROCESSED_HW)})
        else:
            calls = report.get("native_focal_solver_calls")
            if (not isinstance(calls, list) or len(calls) != 15
                    or report.get("focal_geometry_source_sha256") != inference.moge_camera.GEOMETRY_SHA
                    or any(c.get("original_solver_returned") is not True or c.get("focal_prior_supplied") is not True
                           or type(c.get("native_nearest64_valid_pixels")) is not int or c["native_nearest64_valid_pixels"] < 2 for c in calls)):
                raise ValueError("Every original native fixed-prior MoGe focal call required")
        inference.validate_bindings(root, report)
        rows = report.get("outputs")
        if not isinstance(rows, list) or len(rows) != 15: raise ValueError("All fifteen frozen backend outputs required")
        names, arrays = {"report.json"}, []
        for record, row in zip(records, rows):
            filename = Path(record["file"]).stem+".npz"; names.add(filename)
            require_fields(row, {"file": record["file"], "artifact": filename, "clip_index": record["clip_index"],
                                 "frame_index": record["frame_index"], "rgb_sha256": record["sha256"]})
            p = folder/filename; h = regular_hash(p, row.get("sha256"))
            if backend == "da3":
                K = np.asarray(row.get("processed_camera_K"), dtype=np.float64)
                factor = inference.metric_factor(K)
                if type(row.get("metric_depth_factor")) is not float or row["metric_depth_factor"] != factor:
                    raise ValueError("Per-frame native processed focal scaling differs")
            else:
                call = calls[record["clip_index"]*FRAMES+record["frame_index"]]
                require_fields(call, {k: record[k] for k in ("file", "clip_index", "frame_index")})
            if type(row.get("bytes")) is not int or p.stat().st_size != row["bytes"]: raise ValueError("Frozen prediction byte count differs")
            with np.load(p, allow_pickle=False) as file: value = {k: file[k].copy() for k in file.files}
            inference.validate_arrays(value, record); arrays.append(value); frozen.append((p, h))
        if {p.name for p in folder.iterdir()} != names: raise ValueError("No extra backend artifacts allowed")
        outputs[backend] = arrays; reports[backend] = digest; frozen.append((rp, digest))
    if inference.public_inputs(root) != (records, inputs): raise ValueError("Public RGB changed during audit")
    for p, h in frozen: regular_hash(p, h)
    return records, inputs, outputs, frozen, reports


def validate_truth(data, record):
    import depth_rgb_render as renderer
    import rgb_cohort_protocol as protocol
    if set(data) != renderer.TRUTH_KEYS or any(np.ma.isMaskedArray(v) for v in data.values()):
        raise ValueError("Exact explicit foreground-only private truth required")
    for key, shape, kind in (("human_vertices_camera_m", (18439, 3), "f"), ("object_vertices_camera_m", (194, 3), "f"),
                             ("human_faces", (36874, 3), "iu"), ("object_faces", (384, 3), "iu")):
        a = data[key]
        if np.ma.isMaskedArray(a) or a.shape != shape or a.dtype.kind not in kind or not np.isfinite(a).all():
            raise ValueError("Exact finite native/private topology required")
    for key, count in (("human_faces", 18439), ("object_faces", 194)):
        f = data[key]
        if np.any(f < 0) or np.any(f >= count) or np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 0] == f[:, 2]) or np.any(f[:, 1] == f[:, 2]):
            raise ValueError("Nondegenerate original triangle indices required")
    for key in ("clip_index", "frame_index"):
        if data[key].shape != () or data[key].dtype != np.int64 or int(data[key]) != record[key]: raise ValueError("Private original frame order differs")
    K = data["camera_K"]; focal = protocol.D88.focals[record["clip_index"]]
    if K.dtype != np.float64 or not np.array_equal(K, [[focal, 0., 512.], [0., focal, 384.], [0., 0., 1.]]):
        raise ValueError("Actual independently manufactured private camera differs")
    z, ids = data["scene_depth_m"], data["visible_face_indices"]
    if (z.shape != (768, 1024) or z.dtype != np.float32 or ids.shape != z.shape or ids.dtype != np.int64
            or np.any(ids < -1) or np.any(ids >= 36874+384) or not np.isnan(z[ids < 0]).all()
            or not np.isfinite(z[ids >= 0]).all() or np.any(z[ids >= 0] <= 0)):
        raise ValueError("Valid foreground Z only; room pixels must be unknown")
    human = (ids >= 0) & (ids < 36874); obj = (ids >= 36874) & (ids < 36874+384)
    if human.sum() <= 64 or obj.sum() <= 64: raise ValueError("Every foreground entity needs fixed visible support")
    return human, obj


def run(root, report):
    records, inputs, outputs, frozen, producers = public_predictions(root)
    report.update(predictions_frozen_before_private=True, prediction_report_sha256=producers)
    report["private_truth_used_for_evaluation_only"] = True
    import depth_rgb_render as renderer
    import rgb_cohort_protocol as protocol
    private = Path(root)/BASE/"eval_private"; rp = private/"render-report.json"; rh = regular_hash(rp)
    render = json.loads(rp.read_text())
    require_fields(render, {"stage": renderer.STAGE, "status": "pass", "phase": "complete", "frames": 15,
        "actual_MHR_reference_used": True, "actual_reference_forward_calls": 2, "challenge_inputs_used": False,
        "inference_performed": False, "all_truth_private": True, "identity_clip_constant": True,
        "background_face_ids_removed_from_truth": True, "synthetic_truth_used_for_rendering_only": True,
        "image_id": renderer.IMAGE_ID, "script_sha256": sha256(Path(renderer.__file__)),
        "helper_source_sha256": renderer.helper_hashes(), "protocol_sha256": protocol.D88.digest(),
        "model_sha256": renderer.render.semantics.MODEL_SHA, "public_manifest_sha256": inputs["public_manifest_sha256"]})
    if not re.fullmatch("[0-9a-f]{40}", str(render.get("code_revision", ""))): raise ValueError("Immutable render producing revision required")
    sp = private/"semantic-report.json"; sh = regular_hash(sp, render.get("semantic_report_sha256")); semantic = json.loads(sp.read_text())
    renderer.render.require_semantic_report(semantic)
    if semantic.get("source_image_id") != renderer.IMAGE_ID: raise ValueError("Native semantic/image identity differs")
    rigp = private/"rig.npz"; rigsha = regular_hash(rigp, render.get("rig_sha256"))
    with np.load(rigp, allow_pickle=False) as file: rig = {k: file[k].copy() for k in file.files}
    if set(rig) != {"controls", "shape45", "parameter_limits"}: raise ValueError("Exact private manufacturing rig required")
    controls, shapes, _ = protocol.named_controls(render.get("parameter_names"), rig["parameter_limits"])
    if (rig["controls"].dtype != np.float32 or rig["shape45"].dtype != np.float32
            or not np.array_equal(rig["controls"], controls) or not np.array_equal(rig["shape45"], shapes)):
        raise ValueError("Frozen fresh manufacturing recipe does not replay")
    frozen += [(rp, rh), (sp, sh), (rigp, rigsha)]
    cases = render.get("cases")
    if not isinstance(cases, list) or len(cases) != 15: raise ValueError("All original private cases required")
    if {p.name for p in private.iterdir()} != {"rig.npz", "render-report.json", "semantic-report.json", *[Path(r["file"]).stem+".npz" for r in records]}:
        raise ValueError("No unexpected private artifacts allowed")
    metrics, topology = [], None
    for record, case in zip(records, cases):
        require_fields(case, {"file": record["file"], "clip_index": record["clip_index"], "frame_index": record["frame_index"], "rgb_sha256": record["sha256"]})
        p = private/(Path(record["file"]).stem+".npz"); h = regular_hash(p, case.get("truth_sha256"))
        with np.load(p, allow_pickle=False) as file: truth = {k: file[k].copy() for k in file.files}
        human, obj = validate_truth(truth, record)
        if topology is not None and any(not np.array_equal(truth[k], topology[k]) for k in ("human_faces", "object_faces")):
            raise ValueError("Private mesh topology cannot change across original frames")
        topology = {k: truth[k] for k in ("human_faces", "object_faces")}
        index = record["clip_index"]*FRAMES+record["frame_index"]
        row = {"clip_index": record["clip_index"], "frame_index": record["frame_index"]}
        for backend in ("moge", "da3"):
            prediction = outputs[backend][index]
            row[backend] = {"object": visible_z_metric(prediction["depth"], prediction["validity"], truth["scene_depth_m"], obj),
                            "human_depth_diagnostic_only": visible_z_metric(prediction["depth"], prediction["validity"], truth["scene_depth_m"], human)}
        metrics.append(row); frozen.append((p, h))
    for p, h in frozen: regular_hash(p, h)
    import depth_rgb_infer as inference
    if inference.public_inputs(root) != (records, inputs): raise ValueError("Public RGB changed during private scoring")
    report.update(status="pass", phase="complete", frame_metrics=metrics, decision=decision(metrics),
                  all_frames_scored=True, no_gt_alignment=True, private_render_report_sha256=rh)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); out = root/BASE/"quality_v1"
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or root != Path("/srv/scenesmith/world-reward") or out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir())
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"):
        raise ValueError("Fresh exclusive pinned-image offline CPU evaluation required")
    report = dict(stage=STAGE, status="fail", phase="public_integrity", producer_revision=revision, image_id=image,
        script_sha256=sha256(Path(__file__)), network="none", budget_seconds=BUDGET, gates=GATES,
        predictions_frozen_before_private=False, private_truth_used_for_evaluation_only=False,
        challenge_inputs_used=False, adoption_authorized=False, full_HOI_verified=False, human_accuracy_verified=False,
        calibration_accuracy_verified=False, real_domain_verified=False, object_mesh_or_contact_verified=False,
        missing_support_rule="undefined primary metric and whole comparison reject; no intersection, trimming, clipping or penalty",
        frame_metrics=[])
    started = time.perf_counter(); path = out/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole private depth evaluation exceeds120s")
        old_alarm = signal.signal(signal.SIGALRM, expired); old_term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, report)
        except BaseException as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old_alarm); signal.signal(signal.SIGTERM, old_term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
