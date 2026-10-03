"""Fresh RGB-only DWPose-133 observations; no fitting, truth or quality claim.

Use the unchanged pinned native preprocessing/decode and a single CPU session.
Float64 list feeds are delegated intact, not silently cast to the ONNX dtype.
"""
import argparse
import gc
from importlib import util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

import numpy as np
import dwpose_smoke as smoke

BASE = "validation/keypoint_rgb_v1"
OUT = BASE+"/dwpose_v1"
STAGE = "public_keypoint_rgb_native_dwpose133_observations"
WIDTH, HEIGHT, CLIPS, FRAMES, BUDGET = 1024, 768, 3, 5, 180
MANIFEST_BYTES = 2199
MANIFEST_SHA = "082549b5a1f8a4a7d687bc17ec6847f3628d6d4230186053951caa2454d2979d"
MASK_SHA = "d9cae7c4c7131b599ebcafc58a415b9f763286806f80d4a6a46ee114249d8568"
MASK_REVISION = "42f457fc46fbeb814befb48b8104403b06922f63"
MASK_SOURCE_SHA = "d7269a1bd7d1041b902b9278770b0fcd25d1a4455001fa964deba985898760f7"
SMOKE_SHA = "d96eb5c8c4030bf2e16924093aef03a9636f12cc2f3d3a8b7475731fe49d9a79"
SMOKE_REVISION = "9b0a4e19ac8a13b50417089e351913b2aacf0892"
SMOKE_SOURCE_SHA = "ea0beb43dd698261a5b2accdbd0432dcc8de82b56958b0e93cec8062089ce72c"
PREDICTION_KEYS = {"keypoints", "scores", "validity", "bbox", "clip_index", "frame_index"}


def require_fields(data, expected):
    if not isinstance(data, dict) or any(type(data.get(k)) is not type(v) or data[k] != v for k, v in expected.items()):
        raise ValueError("Exact completed producer fields required")


def source_identity():
    rows = smoke.source_identity()
    if rows["dwpose_smoke.py"]["sha256"] != SMOKE_SOURCE_SHA:
        raise ValueError("Actual passing D95 helper source changed")
    for name in (Path(__file__).name, "run_keypoint_rgb_dwpose.sh"):
        rows[name] = smoke.identity(Path(__file__).with_name(name))
    return rows


def validate_smoke(root):
    previous = smoke.validate_previous_smoke(root)
    path = root/smoke.OUT/"report.json"; receipt = smoke.identity(path, sha=SMOKE_SHA)
    data = json.loads(path.read_text())
    require_fields(data, {"stage": smoke.STAGE, "status": "pass", "phase": "complete",
        "producer_revision": SMOKE_REVISION, "script_sha256": SMOKE_SOURCE_SHA, "image_id": smoke.audit.IMAGE,
        "previous_failed_smoke_sha256": smoke.PREVIOUS_SMOKE_SHA, "previous_failure_rewritten": False,
        "device": "cpu", "network": "none", "native_cpu_abi_verified": True, "two_session_byte_replay_verified": True,
        "private_prefix_packages_installed": True, "private_prefix_removed": True, "final_inputs_source_assets_rehashed": True,
        "native_source_modified": False, "own_feed_cast": False, "channel_swap": False, "full_image_fallback": False,
        "raw_scores_clamped": False, "confidence_threshold_applied": False, "wrapper_neck134_used": False,
        "global_image_modified": False, "gpu_used": False, "private_truth_read": False, "ground_truth_used": False,
        "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [], "independent_quality_cohort": False,
        "accuracy_verified": False, "license_clearance_verified": False, "adoption_authorized": False})
    smoke.validate_replay(data.get("sessions", []))
    return {"actual_pass_receipt": receipt, "previous_failed_receipt": previous}


def public_inputs(root):
    """Audit all15 public RGBs and human masks; never open private or object PNGs."""
    base = root/BASE; public = base/"inputs"; maskdir = base/"automatic_masks"
    manifest_path, mask_path = public/"manifest.json", maskdir/"report.json"
    receipt = {"manifest": smoke.identity(manifest_path, sha=MANIFEST_SHA, size=MANIFEST_BYTES),
               "automatic_masks": smoke.identity(mask_path, sha=MASK_SHA)}
    manifest, masks = json.loads(manifest_path.read_text()), json.loads(mask_path.read_text())
    if set(manifest) != {"schema", "images"} or manifest["schema"] != "world-reward-keypoint-rgb-v1":
        raise ValueError("Only fresh RGB-only schema accepted")
    require_fields(masks, {"stage": "public_keypoint_rgb_automatic_masks", "status": "pass", "phase": "complete",
        "frames": 15, "producer_revision": MASK_REVISION, "script_sha256": MASK_SOURCE_SHA,
        "network": "none", "private_truth_read": False, "ground_truth_used": False, "challenge_inputs_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "human_query": "person.", "object_query": "bottle.",
        "actual_detector_calls": 30, "actual_sam2_calls": 30, "actual_sam2_image_encoder_calls": 15,
        "actual_automatic_inference_verified": True, "all_cases_retained": True,
        "input_manifest_sha256": MANIFEST_SHA, "input_manifest_bytes": MANIFEST_BYTES})
    images, rows = manifest.get("images"), masks.get("records")
    if not isinstance(images, list) or not isinstance(rows, list) or len(images) != 15 or len(rows) != 15:
        raise ValueError("No case selection, omission or partial masks accepted")
    records = []
    for index, (item, row) in enumerate(zip(images, rows)):
        clip, frame = divmod(index, FRAMES); filename = f"clip_{clip:02d}_frame_{frame:03d}.png"
        if (not isinstance(item, dict) or set(item) != {"file", "sha256", "width", "height"} or item["file"] != filename
                or type(item["width"]) is not int or type(item["height"]) is not int or (item["width"], item["height"]) != (WIDTH, HEIGHT)
                or not re.fullmatch("[0-9a-f]{64}", str(item["sha256"]))):
            raise ValueError("Original ordered15 RGB metadata required")
        require_fields(row, {"file": filename, "rgb_sha256": item["sha256"], "clip_index": clip, "frame_index": frame})
        for label, query in (("human", "person."), ("object", "bottle.")):
            require_fields(row, {label+"_query": query, label+"_mask_file": Path(filename).stem+"_"+label+".png"})
            if (not re.fullmatch("[0-9a-f]{64}", str(row.get(label+"_mask_sha256", "")))
                    or type(row.get(label+"_mask_bytes")) is not int or row[label+"_mask_bytes"] <= 0
                    or type(row.get(label+"_mask_pixels")) is not int or not 0 < row[label+"_mask_pixels"] <= WIDTH*HEIGHT):
                raise ValueError("Complete automatic person/object metadata required")
        rgb = smoke.identity(public/filename, sha=item["sha256"])
        human = smoke.identity(maskdir/row["human_mask_file"], sha=row["human_mask_sha256"], size=row["human_mask_bytes"])
        records.append({**item, "path": public/filename, "clip_index": clip, "frame_index": frame,
                        "human_mask_file": row["human_mask_file"], "human_mask_path": maskdir/row["human_mask_file"],
                        "human_mask_pixels": row["human_mask_pixels"], "rgb_identity": rgb, "human_mask_identity": human})
    if {p.name for p in public.iterdir()} != {"manifest.json", *[r["file"] for r in records]}:
        raise ValueError("Only15 public PNGs and manifest may be exposed")
    return records, receipt


def validate_prediction(data, record):
    """Pure NumPy exact native133 artifact contract, including raw-score validity."""
    if set(data) != PREDICTION_KEYS:
        raise ValueError("Exact native prediction keys required")
    valid = smoke.validate_prediction(data["keypoints"], data["scores"])
    bbox = data["bbox"]
    if (type(data["validity"]) is not np.ndarray or data["validity"].dtype != np.bool_
            or not np.array_equal(data["validity"], valid) or type(bbox) is not np.ndarray or bbox.dtype != np.float32
            or bbox.shape != (1, 4) or not np.isfinite(bbox).all() or np.any(bbox[:, :2] < 0)
            or bbox[0, 2] > WIDTH or bbox[0, 3] > HEIGHT or np.any(bbox[:, 2:] <= bbox[:, :2])):
        raise ValueError("Automatic bounded bbox and native positive-score validity required")
    for name in ("clip_index", "frame_index"):
        if type(data[name]) is not np.ndarray or data[name].shape != () or data[name].dtype != np.int64 or data[name].item() != record[name]:
            raise ValueError("Original scalar clip/frame identity required")


def validate_artifacts(out, records):
    if len(records) != 15: raise ValueError("All15 native predictions required")
    names = {"report.json"}
    for index, row in enumerate(records):
        clip, frame = divmod(index, FRAMES); expected = f"clip_{clip:02d}_frame_{frame:03d}"
        require_fields(row, {"file": expected+".png", "prediction_file": expected+".npz", "clip_index": clip, "frame_index": frame})
        names.add(row["prediction_file"]); path = out/row["prediction_file"]
        if smoke.identity(path) != row["prediction"] or path.stat().st_mode & 0o222:
            raise ValueError("Frozen native output bytes changed")
        with np.load(path, allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}; validate_prediction(data, row)
            if any(smoke.array_identity(data[k]) != row[k] for k in ("keypoints", "scores", "validity", "bbox")):
                raise ValueError("Canonical native output arrays changed")
    if {p.name for p in out.iterdir()} != names: raise ValueError("Exact15 predictions+receipt inventory required")


def perform(root, out, report, persist, started):
    sources = source_identity(); evidence = validate_smoke(root); assets = smoke.validate_assets(root)
    records, inputs = public_inputs(root)
    report.update(sources=sources, capability_evidence=evidence, assets=assets, public_inputs=inputs,
                  dependencies=smoke.dependency_identity(), phase="private_prefix_install"); persist()
    if any(name in sys.modules or util.find_spec(name) is not None for name in ("onnxruntime", "flatbuffers")):
        raise ValueError("Only disposable freshly installed runtime packages allowed")
    from PIL import Image
    with smoke.private_prefix(report, persist) as prefix:
        command = smoke.pip_argv(prefix, root); report["pip_argv"] = command; persist()
        subprocess.run(command, check=True, timeout=max(.001, BUDGET-(time.perf_counter()-started)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report["private_prefix_packages_installed"] = True; ort, origins = smoke.import_runtime(prefix)
        report.update(runtime_imports=origins, phase="native_source_load"); persist()
        source = root/smoke.acquisition.BASE/"source/onnxpose.py"
        spec = util.spec_from_file_location("world_reward_keypoint_native_onnxpose", source)
        native = util.module_from_spec(spec); spec.loader.exec_module(native)
        options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(root/smoke.acquisition.BASE/smoke.acquisition.ASSETS[0][0]), sess_options=options, providers=["CPUExecutionProvider"])
        session.disable_fallback(); graph = smoke.session_metadata(session)
        report.update(actual_sessions=1, graph=graph, phase="graph_metadata_validation"); persist()
        smoke.validate_session(session, graph=graph); smoke.validate_options(session, ort)
        proxy = smoke.SessionProxy(session, report["calls"], persist)
        for row in records:
            report.update(phase="native_rgb_inference", active_file=row["file"]); persist()
            with Image.open(row["path"]) as rgb, Image.open(row["human_mask_path"]) as human:
                if rgb.format != "PNG" or rgb.mode != "RGB" or rgb.size != (WIDTH, HEIGHT) or human.format != "PNG" or human.mode != "L" or human.size != (WIDTH, HEIGHT):
                    raise ValueError("Original RGB/binary human-mask PNG required")
                image, mask = np.ascontiguousarray(np.asarray(rgb)), np.asarray(human)
            if image.dtype != np.uint8 or image.shape != (HEIGHT, WIDTH, 3) or np.count_nonzero(mask) != row["human_mask_pixels"]:
                raise ValueError("Original RGB/grid/human area differs")
            bbox = smoke.actor_bbox(mask); points, scores = native.inference_pose(proxy, bbox.copy(), image)
            data = {"keypoints": points, "scores": scores, "validity": smoke.validate_prediction(points, scores), "bbox": bbox,
                    "clip_index": np.asarray(row["clip_index"], np.int64), "frame_index": np.asarray(row["frame_index"], np.int64)}
            validate_prediction(data, row); name = Path(row["file"]).stem+".npz"; path = out/name
            with path.open("xb") as stream: np.savez(stream, **data)
            path.chmod(0o444)
            result = {k: row[k] for k in ("file", "clip_index", "frame_index")}
            result.update(rgb_sha256=row["sha256"], prediction_file=name, prediction=smoke.identity(path),
                          raw_simcc=report["calls"][-1]["raw_simcc"], positive_score_count=int(data["validity"].sum()),
                          **{k: smoke.array_identity(data[k]) for k in ("keypoints", "scores", "validity", "bbox")})
            report["records"].append(result); persist()
        del proxy, session; gc.collect()
        if report["actual_sessions"] != 1 or len(report["calls"]) != 15 or any(not r["run_completed"] for r in report["calls"]):
            raise ValueError("Exactly one actual CPU session/all15 unchanged native calls required")
        validate_artifacts(out, report["records"])
        if sources != source_identity() or evidence != validate_smoke(root) or assets != smoke.validate_assets(root) or (records, inputs) != public_inputs(root):
            raise ValueError("Frozen source/evidence/assets/public inputs changed")
        report.update(final_inputs_source_assets_rehashed=True, native_cpu_abi_verified=True)
    if not report["private_prefix_removed"]: raise ValueError("Disposable runtime prefix not removed")
    report.update(status="pass", phase="complete", all_cases_retained=True); report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ.get("WR_ROOT", "")); out = root/OUT; revision = os.environ.get("WR_CODE_REVISION", "")
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root.absolute()
            or os.geteuid() != 1000 or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or os.environ.get("WR_IMAGE_ID") != smoke.audit.IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or not re.fullmatch("[0-9a-f]{40}", revision) or not out.is_dir() or any(out.iterdir())
            or any(p.is_symlink() for p in (out, *out.parents))):
        raise ValueError("Fresh canonical offline CPU-only reserved output required")
    report = {"stage": STAGE, "status": "fail", "phase": "public_assets_integrity", "producer_revision": revision,
        "image_id": smoke.audit.IMAGE, "script_sha256": smoke.identity(Path(__file__))["sha256"], "budget_seconds": BUDGET,
        "network": "none", "device": "cpu", "actual_sessions": 0, "calls": [], "records": [], "requested_calls": 15,
        "private_prefix_packages_installed": False, "private_prefix_removed": False, "native_cpu_abi_verified": False,
        "all_cases_retained": False, "native_source_modified": False, "own_feed_cast": False, "channel_swap": False,
        "full_image_fallback": False, "confidence_threshold_applied": False, "raw_scores_clamped": False, "wrapper_neck134_used": False,
        "global_image_modified": False, "gpu_used": False, "private_truth_read": False, "ground_truth_used": False,
        "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [], "fitting_performed": False,
        "quality_verified": False, "accuracy_verified": False, "adoption_authorized": False, "license_clearance_verified": False,
        "training_overlap_excluded": False}
    started = time.perf_counter(); path = out/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole keypoint DWPose observations exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); perform(root, out, report, persist, started)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)[:3000]); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
