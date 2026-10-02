"""CPU writer parity on nine frozen predicted-depth frames, including batch8+1.

The original501-frame H5 stays immutable. New H5 payloads must equal its nine
selected PNGs and alignment metadata, not merely agree with each other. This
is write-only throughput, not full preparation performance or accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import tempfile
import time

import numpy as np
import depth_batch_gate as procedural
from world_reward.data import sha256

STAGE = "own_native_depth_batch_actual_data_write_parity_and_throughput"
EPISODE, TOTAL, FRAMES, BATCH, WORKERS, BUDGET = 15, 501, 9, 8, 8, 240
HEIGHT, WIDTH = 1152, 1536
METHOD = "world_reward_shared_predicted_human_scale"
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
LEGACY_PREP_SHA = "579498b140805824398adc83c62d732246f95e3a200138a7a9af5acff52c4f0f"
LEGACY_PREP_REVISION = "5d4f5db0d115bea084ce696531f8e4f430c7d5e6"
LEGACY_DEPTH_SHA = "660e197006f00b14480a2ab23d2512acf634c169d27349e0e2ae430a9619a498"
DEPTH_REPORT_SHA = "611d52a98501e7720eb33c71ff90c6c0009a9f223e91b76435c3144d99a6fd2a"
ALIGNMENT_REPORT_SHA = "788ac611068848b55ea24df6c098a1b973cb0ba2e847b2f3ca432cb36e277332"
ALIGNMENT_SCRIPT_SHA = "c4d1841fd00a3071732e796cb6cfad9c48c1d12c497a225e2ed9dadd36b8c383"
REFERENCE_H5_SHA = "a9458af15b5897704d59c1fd00ae7d8e9bde86c4c2e96bcb1b26fbd3e6128c8f"


def regular(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute(): raise ValueError("Require canonical regular readonly input")
    value = sha256(path)
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or value != expected):
        raise ValueError("Frozen predicted input SHA mismatch")
    return value


def checked_report(path, stage):
    identity = regular(path); report = json.loads(path.read_text())
    expected = {"stage": stage, "status": "pass", "input_track": "track_1", "ground_truth_used": False,
                "hand_labeled_test": False, "oracle_modes": []}
    if not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Require exact passing no-oracle predicted input reports")
    episode = report.get("episode_index")
    legacy = (stage == "world_reward_native_cari_inputs" and "episode_index" not in report
              and report.get("script_sha256") == LEGACY_PREP_SHA and report.get("producer_revision") == LEGACY_PREP_REVISION)
    legacy |= (stage == "monocular_moge2_full_video" and "episode_index" not in report and "producer_revision" not in report
               and report.get("script_sha256") == LEGACY_DEPTH_SHA and identity == DEPTH_REPORT_SHA)
    if not legacy and (type(episode) is not int or episode != EPISODE): raise ValueError("Input report episode identity mismatch")
    if not isinstance(report.get("input_sha256"), str) or not re.fullmatch("[0-9a-f]{64}", report["input_sha256"]): raise ValueError("Original video identity missing")
    return report, identity, legacy


def load_inputs(root, native):
    base = root/f"outputs/episode_{EPISODE:06d}"
    paths = {"prep": base/"cari_inputs/report.json", "depth": base/"depth_full/report.json", "alignment": base/"scale_smoke/report.json"}
    reports, hashes, legacy = {}, {}, False
    for key, stage in (("prep", "world_reward_native_cari_inputs"), ("depth", "monocular_moge2_full_video"), ("alignment", "predicted_human_anchored_moge2_pointmaps")):
        reports[key], hashes[key], was_legacy = checked_report(paths[key], stage); legacy |= was_legacy
    prep, depth, alignment = reports["prep"], reports["depth"], reports["alignment"]
    if (hashes["depth"] != DEPTH_REPORT_SHA or hashes["alignment"] != ALIGNMENT_REPORT_SHA
            or alignment.get("script_sha256") != ALIGNMENT_SCRIPT_SHA): raise ValueError("Require exact frozen actualepisode15 source reports")
    if (len({r["input_sha256"] for r in reports.values()}) != 1 or type(prep.get("frames")) is not int or prep["frames"] != TOTAL
            or prep.get("original_frame_coverage_verified") is not True or type(depth.get("total_video_frames")) is not int
            or depth["total_video_frames"] != TOTAL or depth.get("input_dataset_revision") != DATASET_REVISION
            or prep.get("input_report_sha256", {}).get("depth") != hashes["depth"]
            or prep.get("input_report_sha256", {}).get("alignment") != hashes["alignment"]
            or alignment.get("frame_indices") != [0, TOTAL//2, TOTAL-1]): raise ValueError("Full501-frame depth/scale/preparation chain mismatch")
    frames = depth.get("frames")
    if (not isinstance(frames, list) or len(frames) != TOTAL or any(not isinstance(r, dict) or type(r.get("frame_index")) is not int for r in frames)
            or [r["frame_index"] for r in frames] != list(range(TOTAL))): raise ValueError("Require complete original501-frame predicted depth coverage")
    scale = alignment.get("depth_alignment", {}).get("shared_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0: raise ValueError("Positive clip-shared scale required")
    reference = base/"cari_inputs/aligned_depth.h5"
    if prep.get("depth_h5") != str(reference): raise ValueError("Original reference H5 path mismatch")
    hashes["reference_H5"] = regular(reference, prep.get("file_sha256", {}).get("depth_h5"))
    if hashes["reference_H5"] != REFERENCE_H5_SHA: raise ValueError("Frozen actual501-frame reference H5 identity differs")
    records = []; sources = [(paths[k], hashes[k]) for k in paths]+[(reference, hashes["reference_H5"])]
    for index in range(FRAMES):
        source = base/f"depth_full/{index:06d}.npz"; digest = regular(source, frames[index].get("output_sha256")); sources.append((source, digest))
        with np.load(source, allow_pickle=False) as arrays:
            if set(arrays.files) != {"depth", "mask", "intrinsics", "frame_index"}: raise ValueError("Exact full predicted-depth NPZ schema required")
            d, valid, frame = arrays["depth"].copy(), arrays["mask"].copy(), arrays["frame_index"]
            if (d.shape != (HEIGHT, WIDTH) or d.dtype != np.float32 or valid.shape != d.shape or valid.dtype != np.bool_
                    or frame.ndim != 0 or frame.dtype.kind not in "iu" or int(frame) != index): raise ValueError("Original predicted-depth grid/frame/dtype mismatch")
        # Preserve exact production operations and dtype; no positive-count substitution.
        raw = np.where(valid, d, 0.); aligned = raw*scale
        if not np.isfinite(raw).all() or not np.isfinite(aligned).all() or (raw < 0).any() or (aligned < 0).any() or (raw > 65.535).any() or (aligned > 65.535).any():
            raise ValueError("Native encoding would saturate or hide invalid predicted geometry")
        records.append(native.DepthFrameRecord(index, raw, aligned, scale, 0., int(valid.sum())))
    return reports, hashes, records, reference, sources, legacy


def text(value):
    if isinstance(value, bytes): return value.decode("utf-8")
    if isinstance(value, str): return value
    raise ValueError("Native H5 identity attributes must be UTF8 strings")


def read_receipt(native, path, records, *, total, camera=None, identity=None):
    import h5py
    with h5py.File(path, "r") as handle:
        cameras = list(handle["frame_names"])
        if len(cameras) != 1 or (camera is not None and cameras != [camera]): raise ValueError("Require one exact native camera")
        camera = cameras[0]
        if not re.fullmatch("[A-Za-z0-9_.-]+", camera) or not bool(handle.attrs.get("complete", False)): raise ValueError("Native camera/completion mismatch")
        method = text(handle.attrs["depth_alignment_method"])
        actual_identity = json.loads(text(handle.attrs[native.DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE]))
        if method != METHOD or not isinstance(actual_identity, dict) or (identity is not None and actual_identity != identity): raise ValueError("Native alignment identity mismatch")
        names = [text(v) for v in handle[f"frame_names/{camera}"][:]]
        if names != [f"{i:06d}" for i in range(total)]: raise ValueError("Exact original frame names/order required")
        meta = {}
        for key, dtype, expected in (("scale", np.float32, [r.scale for r in records]), ("shift", np.float32, [r.shift for r in records]),
                ("valid_count", np.int32, [r.valid_count for r in records])):
            dataset = handle[f"alignment/{camera}/{key}"]; selected = dataset[:len(records)]
            if dataset.shape != (total,) or dataset.dtype != np.dtype(dtype) or not np.array_equal(selected, np.asarray(expected, dtype=dtype)):
                raise ValueError("Native alignment metadata bytes/dtype mismatch")
            meta[key] = {"dtype": str(selected.dtype), "values": selected.tolist(), "sha256": hashlib.sha256(selected.tobytes()).hexdigest()}
        payloads = {}
        for kind in ("raw", "aligned"):
            dataset = handle[f"{kind}/{camera}"]
            if dataset.shape != (total,): raise ValueError("Native encoded payload coverage mismatch")
            payloads[kind] = []
            for record in records:
                payload = dataset[record.index]; source = record.raw_depth_m if kind == "raw" else record.aligned_depth_m
                if payload.dtype != np.uint8 or payload.ndim != 1 or not np.array_equal(native.decode_depth_png_uint16(payload), (source.astype(np.float32)*1000.).astype(np.uint16)):
                    raise ValueError("Reference/native PNG quantization mismatch")
                payloads[kind].append({"sha256": hashlib.sha256(payload.tobytes()).hexdigest(), "bytes": int(payload.size)})
    return camera, actual_identity, {"frame_names": names[:len(records)], "alignment_metadata": meta, "encoded_payloads": payloads}


def run_write(native, path, records, mode, camera, identity):
    if mode not in ("single", "batch8"): raise ValueError("Fixed writer mode required")
    started = time.perf_counter()
    with native.MHRDepthH5Writer(path, {camera: [f"{r.index:06d}" for r in records]}, alignment_method=METHOD,
                               alignment_input_identity=identity, encoding_workers=WORKERS) as writer:
        if mode == "single":
            for r in records: writer.write_frame(camera, r.index, r.raw_depth_m, r.aligned_depth_m, scale=r.scale, shift=r.shift, valid_count=r.valid_count)
        else:
            for start in range(0, len(records), BATCH): writer.write_frames(camera, records[start:start+BATCH])
        writer.mark_complete()
    return time.perf_counter()-started


def validate_receipt(native, path, records, camera, identity):
    started = time.perf_counter()
    validation = native.validate_depth_h5(path, expected_cameras=[camera], expected_alignment_method=METHOD,
        expected_alignment_input_identity=identity, validation_workers=WORKERS)
    if (validation["validation_mode"] != "exhaustive" or validation["frame_counts"] != {camera: len(records)}
            or tuple(validation["frame_shapes"][camera]) != (HEIGHT, WIDTH)): raise ValueError("Official exhaustive validator omitted native frames/grid")
    _, _, receipt = read_receipt(native, path, records, total=len(records), camera=camera, identity=identity)
    return time.perf_counter()-started, receipt


def run_gate(native, output, report, path):
    reports, hashes, records, reference, sources, legacy = load_inputs(Path(os.environ["WR_ROOT"]), native)
    camera, identity, expected = read_receipt(native, reference, records, total=TOTAL)
    prep_validation = reports["prep"].get("depth_validation", {})
    expected_identity = {"depth_backend": "moge2", "depth_model_id": "Ruicheng/moge-2-vitl-normal",
        "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340", "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
        "alignment_report_sha256": hashes["alignment"], "monocular_depth": {"backend": "moge2", "model_id": "Ruicheng/moge-2-vitl-normal",
        "model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340", "source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030"},
        "ground_truth_used": False, "alignment": "one_predicted_human_anchored_clip_scalar_no_offset"}
    if (prep_validation.get("validation_mode") != "exhaustive" or prep_validation.get("frame_counts") != {camera: TOTAL}
            or tuple(prep_validation.get("frame_shapes", {}).get(camera, ())) != (HEIGHT, WIDTH)
            or identity != expected_identity):
        raise ValueError("Frozen reference lacks complete official validation/shared no-GT scale identity")
    report.update(input_report_sha256=hashes, reference_H5_sha256=hashes["reference_H5"], legacy_prep_source_identity_verified=legacy,
                  camera=camera, alignment_input_identity=identity, reference_receipt=expected,
                  selected_sources=[{"file": str(p), "sha256": h} for p, h in sources], original501_frame_coverage_verified=True)
    original_arrays = [hashlib.sha256(a.tobytes()).hexdigest() for r in records for a in (r.raw_depth_m, r.aligned_depth_m)]
    report["native_input_array_sha256"] = original_arrays
    procedural.write_report(path, report)
    try:
        with tempfile.TemporaryDirectory(prefix="depth-batch-actual-", dir=output) as temporary:
            temporary = Path(temporary)
            warmup_reference = {"frame_names": expected["frame_names"][:2],
                "encoded_payloads": {k: v[:2] for k, v in expected["encoded_payloads"].items()},
                "alignment_metadata": {key: {"dtype": v["dtype"], "values": v["values"][:2],
                    "sha256": hashlib.sha256(np.asarray(v["values"][:2], dtype=v["dtype"]).tobytes()).hexdigest()}
                    for key, v in expected["alignment_metadata"].items()}}
            report["warmup_receipts"] = {}
            for mode in ("single", "batch8"):
                target = temporary/f"warmup-{mode}.h5"; run_write(native, target, records[:2], mode, camera, identity)
                _, receipt = validate_receipt(native, target, records[:2], camera, identity)
                report["warmup_receipts"][mode] = receipt
                if receipt != warmup_reference: raise ValueError("Warmup native payload differs from frozen two-frame reference")
            report["warmup"] = {"frames_per_mode": 2, "modes": ["single", "batch8"], "excluded_from_timings": True}
            for repeat, order in enumerate((("single", "batch8"), ("batch8", "single"))):
                trial = {"repeat": repeat, "order": list(order), "modes": {}, "reference_byte_parity": False}
                report["trials"].append(trial); procedural.write_report(path, report)
                for mode in order:
                    target = temporary/f"repeat{repeat}-{mode}.h5"; report.update(phase="write", active_mode=mode, active_repeat=repeat); procedural.write_report(path, report)
                    trial["modes"][mode] = {"write_seconds": run_write(native, target, records, mode, camera, identity)}
                    report["phase"] = "official_exhaustive_validation"; procedural.write_report(path, report)
                    elapsed, receipt = validate_receipt(native, target, records, camera, identity)
                    trial["modes"][mode].update(validation_seconds=elapsed, native_payload_receipt=receipt); procedural.write_report(path, report)
                    if receipt != expected: raise ValueError("New native payload/order/metadata differs from frozen original501-frame reference")
                trial["reference_byte_parity"] = True; procedural.write_report(path, report)
    finally: report["temporary_H5_removed"] = not any(output.glob("depth-batch-actual-*"))
    for source, digest in sources: regular(source, digest)
    if original_arrays != [hashlib.sha256(a.tobytes()).hexdigest() for r in records for a in (r.raw_depth_m, r.aligned_depth_m)]:
        raise ValueError("Native writer altered original raw/aligned predicted arrays")
    single = float(np.median([t["modes"]["single"]["write_seconds"] for t in report["trials"]]))
    batch = float(np.median([t["modes"]["batch8"]["write_seconds"] for t in report["trials"]]))
    report.update(median_single_write_seconds=single, median_batch8_write_seconds=batch, median_write_speedup=single/batch,
                  reference_byte_parity=True, source_inputs_unchanged=True, source_arrays_unchanged=True, phase="complete")
    if single/batch < 1.25: raise ValueError("Predeclared writer median speedup below1.25; no adoption")
    report["status"] = "pass"


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Require remote isolated CPU network-none")
    root = Path(os.environ["WR_ROOT"]); output = root/"results/depth-batch-actual-v1"; path = output/"report.json"
    if output.resolve() != output.absolute() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require fresh exclusively reserved benchmark")
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image): raise ValueError("Immutable source/image required")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "shared_benchmark_helper_sha256": sha256(Path(procedural.__file__)),
        "native_source_sha256": procedural.SOURCE_SHA, "native_source_revision": procedural.REVISION, "trials": [],
        "episode_index": EPISODE, "selected_frame_indices": list(range(FRAMES)), "source_total_frames": TOTAL,
        "budget_seconds": BUDGET, "batch_size": BATCH, "tail_frames": 1, "encoding_workers": WORKERS, "validation_workers": WORKERS,
        "device": "cpu", "gpu_used": False, "challenge_inputs_used": True, "input_scope": "previously verified predicted monocular depth only",
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False,
        "min_median_write_speedup": 1.25, "warmup_frames_per_mode": 2, "compression_or_validation_changed": False,
        "concurrent_host_CPU_load_measured": False, "timing_caveat": "Shared host CPU load may affect writer throughput; not isolated end-to-end performance",
        "performance_scope": "nine-frame synchronous native write/encode/close only; validation timings separate",
        "whole_preparation_throughput_verified": False, "accuracy_verified": False}
    started = time.perf_counter()
    def expired(*_): raise TimeoutError("Actual-data native writer gate exceeded240s")
    alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try: procedural.write_report(path, report); run_gate(procedural.load_native(root), output, report, path)
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter()-started; procedural.write_report(path, report)


if __name__ == "__main__": main()
