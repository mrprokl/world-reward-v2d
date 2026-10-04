"""Frozen native TAP first-query 2D diagnostic; no inference, GT export or HOI claim."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import numpy as np
import robotap_boots_public as public

source = public.source
ROOT = source.ROOT
BASE = public.INPUT
OUT = BASE + "/eval_v1"
JOB = "run_robotap_boots_evaluate"
PINS = "configs/robotap_boots_evaluation_pins.json"
FILES = ("infra/robotap_boots_evaluate.py", "infra/run_robotap_boots_evaluate.sh", "infra/robotap_boots_public.py", "infra/robotap_boots_acquire.py", "configs/robotap_boots_protocol.json", public.PINS, PINS)
IMAGE = public.IMAGE
BUDGET = 240
NAMES = tuple(f"video_{i:03d}.npz" for i in range(3))
PRED_KEYS = {"tracks", "tracks_256", "occlusion", "expected_dist", "visible", "query_points", "point_indices", "frame_index", "static_tracks", "static_visible"}
VISIBILITY_NUMERIC_GUARD = 16 * np.finfo(np.float32).eps


def require(value, message):
    if not value: raise ValueError(message)


def producer(value):
    require(type(value) is dict and set(value) == {"sha256", "bytes", "producer_revision", "script_sha256"}, "Independent producer pins required")
    public.record({k: value[k] for k in ("sha256", "bytes")})
    for key, length in (("producer_revision", 40), ("script_sha256", 64)):
        require(type(value[key]) is str and re.fullmatch(fr"[0-9a-f]{{{length}}}", value[key]), "Actual producer identity required")


def validate_pins(pins):
    require(type(pins) is dict and set(pins) == {"schema", "public_report", "public_manifest", "public_selection", "public_files", "inference_report", "prediction_files"}
            and pins["schema"] == "world-reward-robotap-boots-evaluation-pins-v1", "Independent frozen evaluation pins required")
    for key in ("public_report", "inference_report"): producer(pins[key])
    for key in ("public_manifest", "public_selection"): public.record(pins[key])
    for key in ("public_files", "prediction_files"):
        require(type(pins[key]) is dict and set(pins[key]) == set(NAMES), "Exact original three artifact identities required")
        for row in pins[key].values(): public.record(row)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as file:
        require(len(file.files) == len(set(file.files)), "Duplicate NPZ keys forbidden")
        return {key: file[key] for key in file.files}


def validate_prediction(data, queries, indices, row):
    n, t = len(queries), row["frames"]
    require(type(data) is dict and set(data) == PRED_KEYS and all(type(v) is np.ndarray for v in data.values()), "Exact original native prediction arrays required")
    require(1 <= n <= 32 and queries.dtype == np.float64 and queries.shape == (n, 3)
            and indices.dtype == np.int64 and indices.shape == (n,), "Full selected query shapes required")
    for key in ("tracks", "tracks_256"):
        require(data[key].dtype == np.float32 and data[key].shape == (n, t, 2) and np.isfinite(data[key]).all(), "Full finite FP32 native tracks required")
    for key in ("occlusion", "expected_dist"):
        require(data[key].dtype == np.float32 and data[key].shape == (n, t) and np.isfinite(data[key]).all(), "Full native finite logits required")
    for key in ("visible", "static_visible"):
        require(data[key].dtype == np.bool_ and data[key].shape == (n, t), "Full original visibility arrays required")
    require(data["query_points"].dtype == np.float64 and np.array_equal(data["query_points"], queries)
            and data["point_indices"].dtype == np.int64 and np.array_equal(data["point_indices"], indices)
            and data["frame_index"].dtype == np.int64 and np.array_equal(data["frame_index"], np.arange(t, dtype=np.int64)), "No dropped/reordered frames or initial queries")
    wh = np.asarray([row["width"], row["height"]], dtype=np.float32)
    require(np.array_equal(data["tracks"], data["tracks_256"] * wh / np.float32(256)), "Exact native original/256 coordinate conversion required")
    static = np.broadcast_to(queries[:, [2, 1]][:, None], (n, t, 2))
    require(data["static_tracks"].dtype == np.float64 and np.array_equal(data["static_tracks"], static)
            and data["static_visible"].all(), "Unmodified static initial-point control required")
    # Float64 check of FP32 native CUDA sigmoid decision: allow its rounding
    # interval near0.5, never replace the saved native visibility used to score.
    def inverse_sigmoid(x):
        x = x.astype(np.float64); e = np.exp(-np.abs(x))
        return np.where(x >= 0, e / (1 + e), 1 / (1 + e))
    probability = inverse_sigmoid(data["occlusion"]) * inverse_sigmoid(data["expected_dist"])
    require(np.all(data["visible"][probability > .5 + VISIBILITY_NUMERIC_GUARD])
            and not np.any(data["visible"][probability < .5 - VISIBILITY_NUMERIC_GUARD]), "Native visibility disagrees outside FP32 numerical guard")
    return int(np.count_nonzero(np.abs(probability - .5) <= VISIBILITY_NUMERIC_GUARD))


def public_predictions(base, pins):
    """Complete frozen public/prediction firewall before any private GT access."""
    validate_pins(pins); pub, infer = base / "public_v1", base / "infer_v1"
    for path, pin in ((pub / "report.json", pins["public_report"]), (pub / "inputs/manifest.json", pins["public_manifest"]),
            (pub / "selection.json", pins["public_selection"]), (infer / "report.json", pins["inference_report"])):
        public.check_pinned(path, pin)
    report = public.strict_json((pub / "report.json").read_bytes()); prediction = public.strict_json((infer / "report.json").read_bytes())
    manifest = public.strict_json((pub / "inputs/manifest.json").read_bytes()); selection = public.strict_json((pub / "selection.json").read_bytes())
    expected_files = {"manifest.json": pins["public_manifest"], **pins["public_files"]}
    require(report.get("status") == "pass" and report.get("stage") == "external_robotap_oracle_initial_query_public_adapter"
            and report.get("producer_revision") == pins["public_report"]["producer_revision"]
            and report.get("source_before", {}).get("files", {}).get("infra/robotap_boots_public.py", {}).get("sha256") == pins["public_report"]["script_sha256"]
            and report.get("source_after") == report.get("source_before") and report.get("public_files") == expected_files
            and all(report.get(k) is True for k in ("source_after_reverified", "originals_after_reverified", "initial_queries_are_external_oracles"))
            and all(report.get(k) is False for k in ("future_labels_available_to_inference", "inference_performed", "evaluation_performed", "gpu_used", "challenge_inputs_used")), "Actual public oracle adapter PASS required")
    require(prediction.get("status") == "pass" and prediction.get("stage") == "public_robotap_native_bootstapir_predictions"
            and prediction.get("producer_revision") == pins["inference_report"]["producer_revision"] and prediction.get("script_sha256") == pins["inference_report"]["script_sha256"]
            and prediction.get("native_calls_attempted") == prediction.get("native_calls_completed") == 3
            and all(prediction.get(k) is True for k in ("actual_native_inference", "all_original_frames_retained", "all_original_selected_queries_retained", "all_input_assets_sources_rechecked", "oracle_initial_queries"))
            and all(prediction.get(k) is False for k in ("private_pickles_read", "future_tracks_or_visibility_read", "evaluation_performed", "challenge_inputs_used", "adoption_performed")), "Actual original native inference PASS required")
    require(prediction.get("inference_config") == {"pyramid_level": 1, "resolution": [256, 256], "query_chunk_size": 32,
        "is_training": False, "compute_dtype": "float32", "AMP": False, "resize": "native utils.bilinear align_corners=False",
        "normalize": "RGB float32 /255*2-1", "frame_offload": False,
        "visibility": "(1-sigmoid(occlusion))*(1-sigmoid(expected_dist))>0.5"}, "Original frozen native inference configuration required")
    require(manifest.get("schema") == "world-reward-robotap-boots-public-v1" and manifest.get("initial_query_is_external_oracle") is True
            and manifest.get("future_tracks_or_visibility_public") is False and manifest.get("frame_crop_or_resize") is False
            and all(manifest.get(k) is False for k in ("training_overlap_verified", "challenge_overlap_verified", "full_hoi_accuracy_verified"))
            and manifest.get("selection") == selection and report.get("frozen_selection_before_future_label_access") == selection, "Frozen original selection/query disclosure required")
    require(type(prediction.get("videos")) is list and len(prediction["videos"]) == 3, "All three actual inference output receipts required")
    rows = manifest.get("videos"); require(type(rows) is list and len(rows) == 3 and type(selection) is list and len(selection) == 3, "All three frozen original videos required")
    require({p.name for p in (pub / "inputs").iterdir()} == set(expected_files)
            and {p.name for p in (infer / "predictions").iterdir()} == set(NAMES), "No extra or missing public/prediction artifacts")
    records = []
    for i, row in enumerate(rows):
        name = NAMES[i]; require(row.get("file") == name and row.get("source_pickle") == public.PICKLES[i]
            and selection[i] == {"pickle_file": public.PICKLES[i], "video_key": row.get("video_key")} and row.get("all_original_frames_retained") is True, "Original split/key/frame order required")
        for key in ("frames", "height", "width", "query_count"): require(type(row.get(key)) is int and row[key] > 0, "Original positive dimensions required")
        public.check_pinned(pub / "inputs" / name, pins["public_files"][name]); public.check_pinned(infer / "predictions" / name, pins["prediction_files"][name])
        actual = prediction["videos"][i]
        require(actual.get("file") == name and actual.get("output") == pins["prediction_files"][name]
                and actual.get("input") == pins["public_files"][name] and actual.get("frames") == row["frames"]
                and actual.get("query_count") == row["query_count"] and actual.get("point_indices") == row["point_indices"], "Actual source-linked inference output receipt differs")
        data = load_arrays(pub / "inputs" / name); require(set(data) == {"video", "query_points", "point_indices"}, "Public arrays only")
        video, queries, indices = (data[k] for k in ("video", "query_points", "point_indices"))
        require(set(data) == {"video", "query_points", "point_indices"} and video.dtype == np.uint8 and video.shape == (row["frames"], row["height"], row["width"], 3)
                and queries.dtype == np.float64 and queries.shape == (row["query_count"], 3) and np.isfinite(queries).all()
                and indices.dtype == np.int64 and indices.tolist() == row["point_indices"] and np.all(np.diff(indices) > 0)
                and np.all((indices >= 0) & (indices < 32)) and np.all(queries[:, 0] == np.floor(queries[:, 0]))
                and np.all((queries[:, 0] >= 0) & (queries[:, 0] < row["frames"])), "Full public RGB/query arrays required")
        pred = load_arrays(infer / "predictions" / name); ambiguous = validate_prediction(pred, queries, indices, row)
        records.append(dict(row=row, queries=queries, indices=indices, prediction=pred, video_sha256=hashlib.sha256(video.tobytes()).hexdigest(), visibility_numeric_guard_count=ambiguous))
        del data, video; gc.collect()
    return records


def tap_metrics(queries, gt_occluded, gt_tracks, pred_occluded, pred_tracks):
    """Native TAP first-query formulas, raster256, strictly AFTER query frame."""
    n, t = gt_occluded.shape
    require(queries.shape == (n, 3) and gt_tracks.shape == pred_tracks.shape == (n, t, 2)
            and gt_occluded.dtype == pred_occluded.dtype == np.bool_ and pred_occluded.shape == (n, t)
            and np.isfinite(queries).all() and np.isfinite(gt_tracks).all() and np.isfinite(pred_tracks).all(), "Finite full TAP metric arrays required")
    frames = np.round(queries[:, 0]).astype(np.int64); require(np.all((frames >= 0) & (frames < t)), "Original query frame bounds required")
    evaluation = np.arange(t)[None, :] > frames[:, None]; visible = ~gt_occluded; pred_visible = ~pred_occluded
    count, positive = int(evaluation.sum()), int((visible & evaluation).sum())
    require(count > 0 and positive > 0, "Native metric denominator zero; no fill or fallback")
    metrics = {"occlusion_accuracy": float(((pred_occluded == gt_occluded) & evaluation).sum() / count)}
    distance = np.sum(np.square(pred_tracks - gt_tracks), axis=-1)
    for threshold in (1, 2, 4, 8, 16):
        within = distance < threshold**2; correct = within & visible
        tp = int((correct & pred_visible & evaluation).sum())
        fp = int(((~visible | ~within) & pred_visible & evaluation).sum())
        metrics[f"pts_within_{threshold}"] = float((correct & evaluation).sum() / positive)
        metrics[f"jaccard_{threshold}"] = float(tp / (positive + fp))
    metrics["average_jaccard"] = float(np.mean([metrics[f"jaccard_{k}"] for k in (1, 2, 4, 8, 16)]))
    metrics["average_pts_within_thresh"] = float(np.mean([metrics[f"pts_within_{k}"] for k in (1, 2, 4, 8, 16)]))
    return metrics


def decision(rows):
    require(type(rows) is list and len(rows) == 3, "Three complete diagnostic metrics required")
    require(all(type(row.get(k)) is dict and all(type(v) is float and np.isfinite(v) and 0 <= v <= 1 for v in row[k].values()) for row in rows for k in ("boots", "static")), "Finite original metric results required")
    names = rows[0]["boots"].keys(); aggregate = {key: {name: float(np.mean([row[key][name] for row in rows])) for name in names} for key in ("boots", "static")}
    delta = aggregate["boots"]["average_jaccard"] - aggregate["static"]["average_jaccard"]
    improved = sum(row["boots"]["average_jaccard"] > row["static"]["average_jaccard"] for row in rows)
    accepted = delta >= .05 and improved >= 2 and aggregate["boots"]["occlusion_accuracy"] >= aggregate["static"]["occlusion_accuracy"]
    return {"scientific_decision": "ACCEPT_2D_DIAGNOSTIC" if accepted else "REJECT", "mean_per_video_metrics": aggregate,
        "absolute_average_jaccard_gain": delta, "videos_with_positive_AJ_gain": improved,
        "static_control_is_valid_motion_prediction": False, "full_hoi_or_cari4d_victory_verified": False}


def evaluate_private(base, records, acquisition_pins, deadline):
    public.acquire_binding(base, acquisition_pins); results = []
    for i, item in enumerate(records):
        source.check_deadline(deadline); name = public.PICKLES[i]; data = public.read_private(base / name, acquisition_pins["pickles"][name]); key = public.first_key(data)
        require(key == item["row"]["video_key"], "Frozen first lexicographic video key differs")
        example = data[key]; video, queries, indices, unavailable = public.initial_queries(example)
        require(np.array_equal(queries, item["queries"]) and np.array_equal(indices, item["indices"])
                and unavailable == item["row"]["unavailable_original_indices"]
                and hashlib.sha256(video.tobytes()).hexdigest() == item["video_sha256"], "Original public RGB/oracle query replay differs")
        gt = example["points"][indices] * np.array([256, 256]); occ = example["occluded"][indices]
        pred = item["prediction"]; static = pred["static_tracks"] * np.array([256, 256]) / np.array([video.shape[2], video.shape[1]])
        results.append({"video_index": i, "boots": tap_metrics(queries, occ, gt, ~pred["visible"], pred["tracks_256"]),
            "static": tap_metrics(queries, occ, gt, ~pred["static_visible"], static)})
        del data, example, video, gt, occ; gc.collect()
    return results


def source_binding(root, code, revision):
    require(root == ROOT and code == root / "jobs" / revision / JOB / "code" and re.fullmatch(r"[0-9a-f]{40}", revision)
            and Path(__file__) == code / FILES[0] and Path(public.__file__).resolve() == code / FILES[2], "Actual immutable evaluator source required")
    files = {name: source.identity(code / name) for name in FILES}
    markers = {name: source.identity(code.parent / name, readonly=False) for name in ("revision", "source-sha256")}
    require((code.parent / "revision").read_bytes() == (revision + "\n").encode() and re.fullmatch(b"[0-9a-f]{64}\n", (code.parent / "source-sha256").read_bytes()), "Original dispatch markers required")
    return {"files": files, "markers": markers}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code, revision = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    require(os.uname().sysname == "Linux" and os.geteuid() == 1000 and os.environ.get("WR_IMAGE_ID") == IMAGE
            and os.environ.get("WR_AZURE_VM02_VERIFIED") == "1" and {p.name for p in Path('/sys/class/net').iterdir()} == {"lo"}, "Offline original VM02 CPU evaluator required")
    before = source_binding(root, code, revision); source.read_protocol(code / FILES[4])
    pins = public.strict_json((code / PINS).read_bytes()); validate_pins(pins)
    acquisition_pins = public.strict_json((code / public.PINS).read_bytes()); public.validate_pins(acquisition_pins)
    output = source.canonical(root / OUT); require(output.is_dir() and {p.name for p in output.iterdir()} == {".container.cid"}, "Fresh owned evaluation output required")
    start = time.monotonic(); deadline = start + BUDGET; error = None; private_started = False
    report = {"stage": "private_robotap_native_first_query_point_quality", "status": "fail", "phase": "public_predictions", "producer_revision": revision,
        "source_before": before, "metric_source": public.TAP_SOURCE, "metric_license": "Apache-2.0", "budget_seconds": BUDGET,
        "oracle_initial_queries": True, "static_control_is_valid_motion_prediction": False, "training_overlap_verified": False,
        "challenge_overlap_verified": False, "full_hoi_accuracy_verified": False, "cari4d_victory_verified": False,
        "minimum_absolute_AJ_gain": .05, "minimum_positive_AJ_videos": 2, "occlusion_accuracy_must_not_worsen": True,
        "inference_performed": False, "gpu_used": False, "challenge_inputs_used": False, "future_GT_arrays_serialized": False}
    def expired(*_): raise TimeoutError("Fixed240s evaluation budget exceeded")
    previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM)}; signal.alarm(BUDGET)
    try:
        records = public_predictions(root / BASE, pins); report.update(phase="private_evaluation", predictions_frozen_before_private_truth=True); private_started = True
        rows = evaluate_private(root / BASE, records, acquisition_pins, deadline)
        report.update(videos=rows, **decision(rows), evaluated_videos=3, all_original_frames_and_selected_queries_retained=True,
            visibility_rounding_guard=VISIBILITY_NUMERIC_GUARD, visibility_rounding_interval_counts=[r["visibility_numeric_guard_count"] for r in records])
    except Exception as caught: error = caught; report.update(error_type=type(caught).__name__, error="Frozen point diagnostic failed closed")
    finally:
        try:
            require(source_binding(root, code, revision) == before, "Original source/markers changed")
            public_predictions(root / BASE, pins)
            if private_started: public.acquire_binding(root / BASE, acquisition_pins)
            report["sources_predictions_and_private_after_reverified"] = True
        except Exception as caught:
            if error is None: error = caught
            report["final_integrity_failed"] = True
        report.update(status="pass" if error is None else "fail", elapsed_seconds=time.monotonic() - start)
        source.save_bytes(output / "report.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(), 0o400)
        signal.alarm(0)
        for s, handler in previous.items(): signal.signal(s, handler)
    if error is not None: raise RuntimeError("Frozen RoboTAP evaluation failed; inspect sealed tiny receipt") from None


if __name__ == "__main__": main()
