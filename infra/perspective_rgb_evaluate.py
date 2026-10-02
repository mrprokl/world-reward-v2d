"""Private camera-only synthetic evaluation after all public predictions freeze.

Abstentions retain the image-diagonal focal. Native fields are replayed by the
unchanged own CPU solver before the first private-file read. Passing this
non-photorealistic experiment does not adopt calibration or verify full HOI.
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

# Runtime source audit closure: /infra/perspective_rgb_render.py and /infra/camera_render.py.
import perspective_rgb_infer as inference
from world_reward.data import sha256

BUDGET = 60
STAGE = "private_procedural_perspective_camera_quality"
RENDER_IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
GATES = {"strong_accepted_min": 5, "strong_focal_median_max": .05,
         "strong_focal_worst_max": .15, "strong_gravity_median_deg_max": 5.,
         "strong_gravity_worst_deg_max": 10., "weak_accepted_max": 0,
         "all_cases_median_relative_gain_min": .05}


def regular_hash(path, expected=None, *, immutable=False):
    path = Path(path)
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Canonical regular frozen file required")
    digest = sha256(path)
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or digest != expected):
        raise ValueError("Frozen file SHA differs: "+str(path))
    if immutable and path.stat().st_mode & 0o222:
        raise ValueError("Prediction artifacts must remain read-only")
    return digest


def require_fields(report, expected):
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise ValueError("Complete explicit producer contract differs")


def same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def public_predictions(root):
    """Audit all nine public files and exact CPU solver replay, without truth."""
    records, inputs = inference.public_inputs(root)
    base = Path(root)/inference.BASE
    out = base/"predictions_v1"
    rp = out/"report.json"
    digest = regular_hash(rp, immutable=True)
    report = json.loads(rp.read_text())
    require_fields(report, {"stage": inference.STAGE, "status": "pass", "phase": "complete",
        "network": "none", "ground_truth_used": False, "private_truth_read": False,
        "challenge_inputs_used": False, "oracle_modes": [], "accuracy_verified": False,
        "adoption_authorized": False, "actual_frontend_calls": 9, "seeds_reset_calls": 9,
        "frames": 9, "all_cases_retained": True, "inputs_assets_rechecked": True,
        "full_geocalib_package_imported": False, "perspective_fields_imported": False,
        "lm_optimizer_imported": False, "original_camera_convention": "edge_center_W/2_H/2",
        "image_id": inference.native.IMAGE_ID, "inputs": inputs,
        "script_sha256": sha256(Path(inference.__file__)), "native_helper_sha256": inference.NATIVE_SHA,
        "solver_sha256": sha256(Path(inference.calibration.__file__)),
        "preprocess": {**inference.PREPROCESS, "output_width": 416, "output_height": 320,
                       "resize": "bilinear_antialias_align_corners_false", "crop": "center_left5"}})
    revision = report.get("producer_revision")
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision) or report.get("code_revision") != revision:
        raise ValueError("Immutable actual inference revision required")
    binding = report.get("assets")
    if not isinstance(binding, dict): raise ValueError("Original asset binding required")
    d82 = inference.validate_d82(root, binding)
    if not same_json(report.get("d82"), d82): raise ValueError("Native D82 reference binding differs")
    ap = Path(root)/inference.native.assets.REPORT
    ah = regular_hash(ap, binding.get("receipt_sha256"))
    acquisition = json.loads(ap.read_text())
    require_fields(acquisition, {"stage": "pinned_geocalib_frontend_assets_acquisition", "status": "pass",
        "phase": "complete", "source_revision": inference.native.assets.SOURCE_REV,
        "segnext_revision": inference.native.assets.SEGNEXT_REV, "source_subset_verified": True,
        "camera_solver_acquired": False, "challenge_inputs_used": False, "ground_truth_used": False})
    if acquisition.get("files") != binding.get("files"):
        raise ValueError("Actual local acquisition and inference asset inventories differ")
    require_fields(report.get("checkpoint_load"), {"state_load_strict": True, "state_all_finite": True,
        "parameter_keys": 748, "buffer_keys": 141, "state_key_mapping": "none"})
    rows = report.get("outputs")
    if not isinstance(rows, list) or len(rows) != 9: raise ValueError("Every ordered prediction required")
    frozen = [(rp, digest), (ap, ah), (Path(root)/inference.native.REPORT, inference.D82_SHA)]
    predicted = []
    from PIL import Image
    for record, row in zip(records, rows):
        index = record["case_index"]
        require_fields(row, {"case_index": index, "file": record["file"], "rgb_sha256": record["rgb_sha256"],
                             "fields_file": f"case_{index:02d}.npz"})
        with Image.open(record["image_path"]) as image:
            if image.mode != "RGB" or image.size != (1024, 768) or image.format != "PNG":
                raise ValueError("Original public RGB PNG grid differs")
        path = out/row["fields_file"]
        h = regular_hash(path, row.get("fields_sha256"), immutable=True)
        if type(row.get("fields_bytes")) is not int or row["fields_bytes"] != path.stat().st_size:
            raise ValueError("Field archive byte size differs")
        with np.load(path, allow_pickle=False) as file: fields = {k: file[k].copy() for k in file.files}
        summary = inference.validate_fields(fields)
        if not same_json(row.get("fields_summary"), summary): raise ValueError("Saved field summaries differ")
        fit = inference.fit_perspective_calibration(fields["up_field"], fields["latitude_field"],
            fields["up_confidence"], fields["latitude_confidence"], **inference.PREPROCESS)
        K = inference.original_camera(fit)
        if not same_json(row.get("calibration"), fit.to_dict()) or not same_json(row.get("camera_K"), K.tolist()):
            raise ValueError("Exact public CPU solver replay differs; do not repair producer camera")
        predicted.append({"K": K, "accepted": fit.accepted, "reason": fit.reason, "gravity": fit.gravity})
        frozen.append((path, h))
    if report.get("accepted_cases") != sum(p["accepted"] for p in predicted):
        raise ValueError("Producer acceptance count differs")
    if {p.name for p in out.iterdir()} != {"report.json", *[r["fields_file"] for r in rows]}:
        raise ValueError("Unexpected public prediction artifacts")
    if not same_json(report, json.loads(rp.read_text())): raise ValueError("Prediction report changed during CPU replay")
    for path, h in frozen: regular_hash(path, h)
    if inference.public_inputs(root) != (records, inputs): raise ValueError("Public inputs changed before private evaluation")
    return predicted, records, inputs, frozen, digest


def camera_metrics(predicted, truth):
    """Score all nine cameras; no rejected case disappears from focal error."""
    if set(truth) != {"camera_K", "gravity", "strong"} or len(predicted) != 9:
        raise ValueError("Exact aggregate camera-only truth and all nine predictions required")
    K, gravity, strong = (np.asarray(truth[k]) for k in ("camera_K", "gravity", "strong"))
    if (K.dtype != np.float64 or K.shape != (9,3,3) or gravity.dtype != np.float64 or gravity.shape != (9,3)
            or strong.dtype != np.bool_ or strong.shape != (9,) or not np.array_equal(strong, [True]*6+[False]*3)
            or not np.isfinite(K).all() or not np.isfinite(gravity).all()
            or np.any(K[:,0,0] <= 0) or np.any(K[:,1,1] <= 0)
            or not np.array_equal(K[:,2], np.tile([0.,0.,1.], (9,1)))
            or np.any(K[:,0,1] != 0) or np.any(K[:,1,0] != 0)
            or not np.array_equal(K[:,:2,2], np.tile([512.,384.], (9,1)))
            or not np.allclose(np.linalg.norm(gravity, axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError("Finite positive pixel cameras, unit gravity and declared six/three split required")
    rows, errors, baseline, strong_gravity = [], [], [], []
    for i, pred in enumerate(predicted):
        P = np.asarray(pred.get("K")); accepted = pred.get("accepted")
        if type(accepted) is not bool or P.shape != (3,3) or P.dtype != np.float64 or not np.isfinite(P).all():
            raise ValueError("Exact finite public camera and boolean acceptance required")
        f = float(P[0,0])
        # Literal sx/sy inverse transport can differ by a final float64 ulp.
        # Preserve and score both saved diagonals; never rewrite producer K.
        structure = P.copy(); structure[0,0] = structure[1,1] = 1.
        if (f <= 0 or P[1,1] <= 0 or abs(f-P[1,1]) > 8*np.finfo(float).eps*max(f,P[1,1])
                or not np.array_equal(structure, [[1.,0.,512.],[0.,1.,384.],[0.,0.,1.]])):
            raise ValueError("Original shared-focal camera required")
        g = pred.get("gravity")
        if accepted:
            g = np.asarray(g, dtype=np.float64)
            if g.shape != (3,) or not np.isfinite(g).all() or not np.isclose(np.linalg.norm(g), 1., atol=1e-12, rtol=0):
                raise ValueError("Accepted gravity cannot be missing or invalid")
            dot = float(g@gravity[i])
            if abs(dot) > 1+8*np.finfo(float).eps: raise ValueError("Gravity dot product overflow")
            angle = float(np.rad2deg(np.arccos(np.clip(dot, -1., 1.))))
            if strong[i]: strong_gravity.append(angle)
        else:
            if g is not None or f != 1280.: raise ValueError("Abstentions must retain explicit diagonal focal and missing gravity")
            angle = None
        pair = np.abs(np.array([P[0,0], P[1,1]])/np.array([K[i,0,0],K[i,1,1]])-1.)
        base = np.abs(1280./np.array([K[i,0,0],K[i,1,1]])-1.)
        errors.append(float(pair.mean())); baseline.append(float(base.mean()))
        rows.append({"case_index": i, "strong": bool(strong[i]), "accepted": accepted, "reason": pred["reason"],
            "relative_fx_error": float(pair[0]), "relative_fy_error": float(pair[1]),
            "mean_relative_focal_error": errors[-1], "fixed_diagonal_relative_focal_error": baseline[-1],
            "accepted_gravity_angle_deg": angle})
    e, b = np.asarray(errors), np.asarray(baseline)
    before, after = float(np.median(b)), float(np.median(e))
    gain = (before-after)/before if before > 0 else None
    strong_accepted = sum(r["accepted"] for r in rows[:6]); weak_accepted = sum(r["accepted"] for r in rows[6:])
    gm = float(np.median(strong_gravity)) if strong_gravity else None
    gw = float(max(strong_gravity)) if strong_gravity else None
    gates = {"strong_acceptance": strong_accepted >= GATES["strong_accepted_min"],
        "strong_focal_median": float(np.median(e[:6])) <= GATES["strong_focal_median_max"],
        "strong_focal_worst": float(e[:6].max()) <= GATES["strong_focal_worst_max"],
        "strong_gravity_median": gm is not None and gm <= GATES["strong_gravity_median_deg_max"],
        "strong_gravity_worst": gw is not None and gw <= GATES["strong_gravity_worst_deg_max"],
        "weak_abstention": weak_accepted <= GATES["weak_accepted_max"],
        "all_nine_focal_gain": gain is not None and gain >= GATES["all_cases_median_relative_gain_min"]}
    return {"cases": rows, "strong_accepted": strong_accepted, "weak_accepted": weak_accepted,
        "gravity_missing_count": 9-sum(r["accepted"] for r in rows), "strong_gravity_count": len(strong_gravity),
        "strong_focal_median": float(np.median(e[:6])), "strong_focal_worst": float(e[:6].max()),
        "strong_gravity_median_deg": gm, "strong_gravity_worst_deg": gw,
        "all_nine_fixed_median_relative_focal_error": before, "all_nine_candidate_median_relative_focal_error": after,
        "all_nine_median_relative_gain": gain, "absolute_percentage_point_drop": 100*(before-after),
        "decision": {"gates": gates, "camera_only_synthetic_hypothesis_supported": all(gates.values()),
                     "adoption_authorized": False, "full_HOI_verified": False}}


def run(root, report):
    predicted, records, inputs, frozen, receipt_sha = public_predictions(root)
    report.update(phase="public_solver_replay_complete", predictions_frozen_before_private=True,
                  exact_cpu_solver_replay_verified=True, prediction_report_sha256=receipt_sha)
    # This is the first evaluation-private read. No private path enters public_predictions.
    base = Path(root)/inference.BASE
    rp = base/"eval_private/render-report.json"
    rh = regular_hash(rp); render = json.loads(rp.read_text())
    report["private_synthetic_truth_used_for_evaluation"] = True
    require_fields(render, {"stage": "own_procedural_perspective_rgb_render", "status": "pass",
        "synthetic_truth_used_for_rendering_only": True, "inference_performed": False,
        "challenge_inputs_used": False, "photorealism_verified": False, "accuracy_verified": False,
        "actual_MHR_reference_used": True, "actual_reference_forward_calls": 2, "image_id": RENDER_IMAGE,
        "strong_cases": 6, "weak_cases": 3, "public_manifest_sha256": inputs["public_inputs_sha256"]})
    if not re.fullmatch("[0-9a-f]{40}", render.get("code_revision", "")) or not re.fullmatch("sha256:[0-9a-f]{64}", render.get("image_id", "")):
        raise ValueError("Immutable actual renderer source/image required")
    import hand_synthetic_render as hand
    import joint_rgb_render as joint
    require_fields(render, {"model_sha256": hand.semantics.MODEL_SHA,
        "script_sha256": sha256(Path(__file__).with_name("perspective_rgb_render.py")),
        "render_helper_sha256": sha256(Path(hand.__file__)), "joint_helper_sha256": sha256(Path(joint.__file__)),
        "camera_helper_sha256": sha256(Path(__file__).with_name("camera_render.py"))})
    semantic = Path(root)/"results/mhr-finger-semantics-v4.json"
    sh = regular_hash(semantic, render.get("semantic_report_sha256"))
    semantic_report = json.loads(semantic.read_text())
    hand.require_semantic_report(semantic_report)
    if semantic_report.get("source_image_id") != render["image_id"]:
        raise ValueError("Private renderer must use its actual semantic-provenance image")
    if not isinstance(render.get("cases"), list) or len(render["cases"]) != 9:
        raise ValueError("All independently rendered cases required")
    for record, case in zip(records, render["cases"]):
        require_fields(case, {"file": record["file"], "rgb_sha256": record["rgb_sha256"]})
    tp = base/"eval_private/calibration_truth.npz"
    th = regular_hash(tp, render.get("truth_sha256"))
    with np.load(tp, allow_pickle=False) as file: truth = {k: file[k].copy() for k in file.files}
    scores = camera_metrics(predicted, truth)
    for path, h in frozen+[(rp,rh),(semantic,sh),(tp,th)]: regular_hash(path,h)
    if inference.public_inputs(root) != (records, inputs): raise ValueError("Public RGB changed during evaluation")
    report.update(scores, status="pass", phase="complete", private_render_report_sha256=rh,
                  private_truth_sha256=th, semantic_report_sha256=sh,
                  renderer_image_id=render["image_id"], frontend_image_id=inference.native.IMAGE_ID,
                  renderer_and_frontend_images_asserted_identical=False)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote network-none CPU evaluator required")
    root = Path(os.environ["WR_ROOT"]); out = root/inference.BASE/"quality_v1"
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir())
            or not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image)):
        raise ValueError("Exclusive fresh canonical output and immutable evaluator required")
    report = {"stage": STAGE, "status": "fail", "phase": "public_integrity", "budget_seconds": BUDGET,
        "producer_revision": revision, "image_id": image, "script_sha256": sha256(Path(__file__)), "network": "none",
        "GPU_used": False, "challenge_inputs_used": False, "private_synthetic_truth_used_for_evaluation": False,
        "calibration_fitting_inputs_include_truth": False, "evaluation_alignment_performed": False,
        "photorealism_verified": False, "adoption_authorized": False, "full_HOI_verified": False, "gates": GATES}
    start = time.perf_counter()
    def expired(*_): raise TimeoutError("Whole public replay and private camera evaluation exceeded60s")
    alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        run(root, report)
        report["private_synthetic_truth_used_for_evaluation"] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter()-start
        with (out/"report.json").open("x") as handle: json.dump(report, handle, allow_nan=False, indent=2); handle.write("\n")
    print(json.dumps({k: report[k] for k in ("stage", "status", "decision", "elapsed_seconds")}))


if __name__ == "__main__": main()
