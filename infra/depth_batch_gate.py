"""CPU-only paired native depth-writer benchmark, not an accuracy experiment.

Identical procedural camera-Z arrays and exact official PNG encoders/validators
compare single-frame writes with synchronous batches8. No compression changes,
payload shortcuts, reduced validation or production adoption occur.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from world_reward.data import sha256

STAGE = "own_native_depth_batch_write_parity_and_throughput"
COMPACT_STAGE = "own_native_depth_batch_compact_write_parity_and_throughput"
REVISION = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
SOURCE_RELATIVE = "reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py"
SOURCE_SHA = "1429760952205d35c87157c05941defa20dc450f2b014441ca7cd5d39b45b0c5"
WIDTH, HEIGHT, FRAMES, BATCH, WORKERS, BUDGET = 1536, 1152, 16, 8, 8, 240
SCALE = .83
CAMERA = "own_camera"
ALIGNMENT_METHOD = "own_fixed_positive_scale_benchmark"
IDENTITY = {"source": "own_procedural_depth_not_challenge", "fixed_scale": SCALE, "shift": 0.}


def write_report(path, report):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n"); temporary.replace(path)


def load_native(root):
    vendor = root/"vendor/video_to_data"; source = vendor/SOURCE_RELATIVE
    if source.is_symlink() or not source.is_file() or sha256(source) != SOURCE_SHA:
        raise ValueError("Exact pinned native depth-writer source SHA required")
    env = os.environ | {"GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1"}
    command = ["git", "-c", f"safe.directory={vendor}", "-C", str(vendor)]
    if (subprocess.check_output(command+["rev-parse", "HEAD"], text=True, env=env).strip() != REVISION
            or subprocess.check_output(command+["status", "--porcelain", "--untracked-files=all"], text=True, env=env).strip()):
        raise ValueError("Require clean immutable native checkout")
    spec = importlib.util.spec_from_file_location("wr_pinned_depth_h5_batch_gate", source)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def procedural_depths(width=WIDTH, height=HEIGHT, frames=FRAMES, *, seed=1709, invalid_background=False):
    """Seeded smooth depth with correlated texture; fixed positive metre range."""
    from scipy.ndimage import zoom
    rng = np.random.default_rng(seed)
    coarse = rng.standard_normal((48, 64)).astype(np.float32)
    texture = zoom(coarse, (height/48, width/64), order=1, prefilter=False)
    y, x = np.mgrid[:height, :width].astype(np.float32); x /= width; y /= height
    base = 2.1+.45*x+.22*y+.11*np.sin(13*x+9*y)+.008*texture
    depths = [(base+.025*i+.004*np.sin(31*x-17*y+i*.13)).astype(np.float32) for i in range(frames)]
    if invalid_background:
        for i, depth in enumerate(depths):
            support = ((x-(.50+.015*np.sin(i*.4)))/.36)**2+((y-.52)/.42)**2 <= 1
            support &= ~((x > .57+.012*np.sin(i*.3)) & (y > .61))
            depth[~support] = 0.
    if any(not np.isfinite(d).all() or np.min(d) < 0 or (not invalid_background and np.min(d) == 0)
            or not np.any(d > 0) or np.max(d) >= 65.535 for d in depths):
        raise ValueError("Procedural raw depth must fit native nonnegative uint16-mm range with positive support")
    return depths


def records_for(native, depths, *, count_positive=False):
    return [native.DepthFrameRecord(i, depth, depth*np.float32(SCALE), SCALE, 0.,
        int(np.count_nonzero(depth > 0)) if count_positive else int(depth.size)) for i, depth in enumerate(depths)]


def run_write(native, path, records, mode):
    if mode not in ("single", "batch8"): raise ValueError("Fixed writer mode required")
    names = [f"{i:06d}" for i in range(len(records))]; started = time.perf_counter()
    with native.MHRDepthH5Writer(path, {CAMERA: names}, alignment_method=ALIGNMENT_METHOD,
            alignment_input_identity=IDENTITY, encoding_workers=WORKERS) as writer:
        if mode == "single":
            for r in records:
                writer.write_frame(CAMERA, r.index, r.raw_depth_m, r.aligned_depth_m,
                    scale=r.scale, shift=r.shift, valid_count=r.valid_count)
        else:
            for start in range(0, len(records), BATCH): writer.write_frames(CAMERA, records[start:start+BATCH])
        writer.mark_complete()
    return time.perf_counter()-started


def validate_and_receipt(native, path, records):
    """Official exhaustive validation, then exact payload/metadata identities."""
    started = time.perf_counter()
    validation = native.validate_depth_h5(path, expected_cameras=[CAMERA], expected_alignment_method=ALIGNMENT_METHOD,
        expected_alignment_input_identity=IDENTITY, validation_workers=WORKERS)
    elapsed = time.perf_counter()-started
    if (validation["validation_mode"] != "exhaustive" or validation["frame_counts"] != {CAMERA: len(records)}
            or tuple(validation["frame_shapes"][CAMERA]) != records[0].raw_depth_m.shape):
        raise ValueError("Official exhaustive validation coverage/shape mismatch")
    import h5py
    with h5py.File(path, "r") as handle:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in handle[f"frame_names/{CAMERA}"][:]]
        if names != [f"{i:06d}" for i in range(len(records))] or not bool(handle.attrs["complete"]):
            raise ValueError("Original frame coverage/order/completion mismatch")
        meta = {}
        for name, dtype, expected in (("scale", np.float32, [r.scale for r in records]),
                ("shift", np.float32, [r.shift for r in records]), ("valid_count", np.int32, [r.valid_count for r in records])):
            actual = handle[f"alignment/{CAMERA}/{name}"][:]
            if actual.dtype != np.dtype(dtype) or not np.array_equal(actual, np.asarray(expected, dtype=dtype)):
                raise ValueError("Native alignment metadata changed: "+name)
            meta[name] = {"dtype": str(actual.dtype), "values": actual.tolist(), "sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        payloads = {}
        for kind in ("raw", "aligned"):
            dataset = handle[f"{kind}/{CAMERA}"]
            if dataset.shape != (len(records),): raise ValueError("Native payload count mismatch")
            payloads[kind] = []
            for index, record in enumerate(records):
                payload = dataset[index]; decoded = native.decode_depth_png_uint16(payload)
                original = record.raw_depth_m if kind == "raw" else record.aligned_depth_m
                expected = (original.astype(np.float32)*1000.).astype(np.uint16)
                if not np.array_equal(decoded, expected): raise ValueError("Native uint16-mm quantization changed")
                payloads[kind].append({"sha256": hashlib.sha256(payload.tobytes()).hexdigest(), "bytes": int(payload.size)})
    return elapsed, {"frame_names": names, "alignment_metadata": meta, "encoded_payloads": payloads,
        "validation_mode": validation["validation_mode"], "frame_counts": validation["frame_counts"], "frame_shapes": validation["frame_shapes"]}


def run_gate(native, output, report, path, *, depths=None, count_positive=False):
    depths = procedural_depths() if depths is None else depths
    records = records_for(native, depths, count_positive=count_positive)
    report["source_array_sha256"] = [hashlib.sha256(d.tobytes()).hexdigest() for d in depths]
    if count_positive:
        report.update(valid_count_per_frame=[r.valid_count for r in records],
                      invalid_zero_count_per_frame=[int(np.count_nonzero(d == 0)) for d in depths])
    with tempfile.TemporaryDirectory(prefix="depth-batch-", dir=output) as temporary:
        temporary = Path(temporary)
        for mode in ("single", "batch8"):
            target = temporary/f"warmup-{mode}.h5"; run_write(native, target, records[:2], mode); validate_and_receipt(native, target, records[:2])
        report["warmup"] = {"frames_per_mode": 2, "modes": ["single", "batch8"], "excluded_from_timings": True}
        write_report(path, report)
        for repeat, order in enumerate((("single", "batch8"), ("batch8", "single"))):
            trial = {"repeat": repeat, "order": list(order), "modes": {}, "payload_byte_parity": False}
            report["trials"].append(trial); write_report(path, report)
            receipts = {}
            for mode in order:
                target = temporary/f"repeat{repeat}-{mode}.h5"
                report.update(phase="write", active_mode=mode, active_repeat=repeat); write_report(path, report)
                trial["modes"][mode] = {"write_seconds": run_write(native, target, records, mode)}; write_report(path, report)
                report["phase"] = "official_exhaustive_validation"; write_report(path, report)
                elapsed, receipt = validate_and_receipt(native, target, records)
                receipts[mode] = receipt
                trial["modes"][mode].update(validation_seconds=elapsed, native_payload_receipt=receipt); write_report(path, report)
            if receipts["single"] != receipts["batch8"]: raise ValueError("Encoded byte/order/metadata parity failed")
            trial["payload_byte_parity"] = True; write_report(path, report)
    report["temporary_H5_removed"] = True
    if report["source_array_sha256"] != [hashlib.sha256(d.tobytes()).hexdigest() for d in depths]:
        raise ValueError("Native encoding changed original procedural source arrays")
    single = float(np.median([t["modes"]["single"]["write_seconds"] for t in report["trials"]]))
    batch = float(np.median([t["modes"]["batch8"]["write_seconds"] for t in report["trials"]]))
    report.update(median_single_write_seconds=single, median_batch8_write_seconds=batch, median_write_speedup=single/batch,
        payload_byte_parity=True, source_arrays_unchanged=True, phase="complete")
    if single/batch < 1.25: raise ValueError("Predeclared median writer speedup below1.25; no adoption")
    report["status"] = "pass"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", choices=("original", "compact"), default="original")
    args = parser.parse_args(argv); compact = args.protocol == "compact"
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Require isolated remote CPU network-none")
    root = Path(os.environ["WR_ROOT"]); output = root/"results"/("depth-batch-compact-v1" if compact else "depth-batch-v1"); path = output/"report.json"
    if output.is_symlink() or not output.is_dir() or any(output.iterdir()): raise FileExistsError("Require exclusively reserved benchmark output")
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image): raise ValueError("Immutable benchmark source/image required")
    report = {"stage": COMPACT_STAGE if compact else STAGE, "protocol": args.protocol,
        "status": "fail", "phase": "integrity", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "native_source_revision": REVISION, "native_source_sha256": SOURCE_SHA,
        "challenge_inputs_used": False, "ground_truth_used": False, "adoption_performed": False,
        "device": "cpu", "budget_seconds": BUDGET, "frames_per_trial": BATCH if compact else FRAMES, "original_grid": [HEIGHT, WIDTH],
        "batch_size": BATCH, "encoding_workers": WORKERS, "validation_workers": WORKERS, "trials": [],
        "scale": SCALE, "shift": 0., "depth_quantization": "uint16 millimetres native truncation; not accuracy",
        "compression_or_validation_changed": False, "concurrent_host_CPU_load_measured": False,
        "timing_caveat": "Shared host load may affect throughput; this is not isolated whole-pipeline performance",
        "min_median_write_speedup": 1.25, "warmup_frames_per_mode": 2,
        "procedural_seed": 1711 if compact else 1709, "procedural_invalid_zero_background": compact,
        "performance_scope": "paired native write/encode/close only; validation timings separate",
        "whole_preparation_throughput_verified": False, "timed_partial_tail_batch_tested": False}
    start = time.perf_counter()
    def expired(*args): raise TimeoutError("Whole depth-batch gate exceeded240s")
    alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        write_report(path, report); native = load_native(root)
        if compact:
            depths = procedural_depths(width=WIDTH, height=HEIGHT, frames=BATCH, seed=1711, invalid_background=True)
            run_gate(native, output, report, path, depths=depths, count_positive=True)
        else: run_gate(native, output, report, path)
    except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter()-start; write_report(path, report)
    print(json.dumps({k: report[k] for k in ("stage", "status", "median_write_speedup", "elapsed_seconds")}))


if __name__ == "__main__": main()
