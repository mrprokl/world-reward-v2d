"""Private RoboTAP adapter: publish full RGB and standard oracle initial queries.

Original future tracks/visibility are decoded privately but NEVER serialized to
public files. This is not Track1 automatic readiness or leakage-free validation.
"""
import argparse
import builtins
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import importlib
import platform
import re
import signal
import time

import numpy as np
import robotap_boots_acquire as source

ROOT = source.ROOT
JOB = "run_robotap_boots_public"
INPUT = "validation/robotap_boots_v1"
PUBLIC_NAMESPACE = "public_v2"
OUTPUT = INPUT + "/" + PUBLIC_NAMESPACE
PINS = "configs/robotap_boots_acquisition_pins.json"
FILES = ("infra/robotap_boots_public.py", "infra/run_robotap_boots_public.sh", "infra/robotap_boots_acquire.py", "configs/robotap_boots_protocol.json", PINS)
PICKLES = tuple(f"eval_private/pickles/robotap/robotap_split{i}.pkl" for i in range(5))
BUDGET = 180
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
MAX_VIDEO_BYTES = 2 * 1024**3
TAP_SOURCE = {"url": "https://raw.githubusercontent.com/google-deepmind/tapnet/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid/evaluation_datasets.py", "bytes": 24538, "sha256": "90cd01e53e23f6d489d3a6cd840cfd93fed4c1a6f4a0dfd373933cc164f0e570"}


ORIGINAL_PUBLIC_FAILURE = {"bytes": 3080, "sha256": "bbf26a80fe4e52905273705fea34aa26d08a89c86f4ce6e5ec33305c54a9902d"}
MEDIAPY_DECODER_SOURCE = {
    "repository": "google/mediapy", "version": "1.2.7", "revision": "a1b47c721f821ecb34623d861c48460a1f078692",
    "source_url": "https://raw.githubusercontent.com/google/mediapy/a1b47c721f821ecb34623d861c48460a1f078692/mediapy/__init__.py",
    "source_bytes": 73425, "source_sha256": "279aa5b1c1cf1d5b2e2025f76c8594df6312fdc65e9431636448926271eccca2",
    "license": "Apache-2.0", "license_bytes": 11358, "license_sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "allowed_global": "mediapy._VideoArray", "mediapy_package_imported": False,
}


class MediaPyVideoArray(np.ndarray):
    """Only the primary `_VideoArray` wrapper, with inherited ndarray pickle."""
    def __new__(cls, input_array, metadata=None):
        obj = np.asarray(input_array).view(cls); obj.metadata = metadata; return obj
    def __array_finalize__(self, obj):
        if obj is not None: self.metadata = getattr(obj, "metadata", None)


def array_digest(array):
    digest = hashlib.sha256()
    iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"], order="C", buffersize=1024 * 1024)
    for chunk in iterator: digest.update(chunk.tobytes())
    return digest.hexdigest()


def plain_video(video):
    if type(video) is not MediaPyVideoArray: return video
    original = (video.dtype, video.shape, video.strides, video.__array_interface__["data"][0])
    plain = np.asarray(video)
    if (type(plain) is not np.ndarray or (plain.dtype, plain.shape, plain.strides, plain.__array_interface__["data"][0]) != original
            or not np.shares_memory(plain, video) or array_digest(plain) != array_digest(video)):
        raise ValueError("Exact original MediaPy ndarray storage/bytes required")
    return plain


def record(value):
    if (type(value) is not dict or set(value) != {"sha256", "bytes"}
            or type(value["bytes"]) is not int or value["bytes"] <= 0
            or type(value["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
        raise ValueError("Independent positive byte/SHA identity required")


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError("Duplicate JSON keys forbidden")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON forbidden")))


def validate_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "report", "retention_receipt", "pickles", "protocol"}
            or pins["schema"] != "world-reward-robotap-boots-acquisition-pins-v1"):
        raise ValueError("Independently committed acquisition pins required")
    original = pins["report"]
    if type(original) is not dict or set(original) != {"sha256", "bytes", "producer_revision", "script_sha256"}:
        raise ValueError("Complete original acquisition producer required")
    record({k: original[k] for k in ("sha256", "bytes")})
    for key, length in (("producer_revision", 40), ("script_sha256", 64)):
        if type(original[key]) is not str or re.fullmatch(fr"[0-9a-f]{{{length}}}", original[key]) is None:
            raise ValueError("Frozen actual acquisition producer identity required")
    for key in ("retention_receipt", "protocol"): record(pins[key])
    if type(pins["pickles"]) is not dict or set(pins["pickles"]) != set(PICKLES):
        raise ValueError("Exact original five split pickle identities required")
    for value in pins["pickles"].values(): record(value)
    if pins["protocol"] != {"sha256": source.PROTOCOL_SHA256, "bytes": source.PROTOCOL_BYTES}:
        raise ValueError("Original protocol changed")


def check_pinned(path, expected):
    actual = source.identity(path)
    if actual != {k: expected[k] for k in ("sha256", "bytes")}: raise ValueError("Original pinned bytes differ")
    return actual


def original_failure(root):
    path = root / "public_v1/report.json"; check_pinned(path, ORIGINAL_PUBLIC_FAILURE)
    value = strict_json(path.read_bytes())
    if (value.get("status") != "fail" or value.get("error_type") != "UnpicklingError"
            or value.get("producer_revision") != "449c50f583070da52e997ef9a04e0cdc27912b8a"
            or value.get("source_after_reverified") is not True
            or "frozen_selection_before_future_label_access" in value or "public_files" in value):
        raise ValueError("Original failed preselection public receipt required")
    return ORIGINAL_PUBLIC_FAILURE


def acquire_binding(root, pins):
    """All five hashes and sealed source receipts verified BEFORE any unpickle."""
    validate_pins(pins); root = source.canonical(root)
    check_pinned(root / "report.json", pins["report"])
    report = strict_json((root / "report.json").read_bytes())
    if (report.get("status") != "pass" or report.get("stage") != source.STAGE
            or report.get("producer_revision") != pins["report"]["producer_revision"]
            or report.get("script_sha256") != pins["report"]["script_sha256"]
            or report.get("source_and_protocol_after_reverified") is not True
            or report.get("original_pickles_unmodified") is not True
            or report.get("pickle_or_rgb_or_gt_decoded") is not False
            or report.get("inference_performed") is not False or report.get("evaluation_performed") is not False
            or report.get("challenge_inputs_used") is not False or report.get("gpu_used") is not False
            or report.get("disposable_archive_removed_after_successful_retention_receipt") is not True
            or report.get("filename_selection", {}).get("pickle_member_names") != [name.split("pickles/", 1)[1] for name in PICKLES]):
        raise ValueError("Successful original opaque acquisition required")
    receipt_path = root / "eval_private/retention-receipt.json"
    check_pinned(receipt_path, pins["retention_receipt"])
    receipt = strict_json(receipt_path.read_bytes())
    if receipt.get("schema") != "world-reward-robotap-opaque-retention-v1" or receipt.get("no_unpickle_or_decode") is not True:
        raise ValueError("Original sealed opaque retention receipt required")
    for name in PICKLES:
        if (receipt.get("retained_files", {}).get(name) != pins["pickles"][name]
                or report.get("retained_files", {}).get(name) != pins["pickles"][name]):
            raise ValueError("Actual acquisition/retention inventories disagree")
        check_pinned(root / name, pins["pickles"][name])
    return {"acquisition_report": {k: pins["report"][k] for k in ("sha256", "bytes")}, "retention_receipt": pins["retention_receipt"], "pickles": pins["pickles"]}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("mediapy", "_VideoArray"): return MediaPyVideoArray
        if module == "builtins" and name in ("set", "frozenset", "slice"):
            return getattr(builtins, name)
        if module == "numpy" and name in ("dtype", "ndarray"): return getattr(np, name)
        if module in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(importlib.import_module("numpy._core.multiarray" if np.__version__.split(".")[0] >= "2" else "numpy.core.multiarray"), name)
        raise pickle.UnpicklingError("Nonallowlisted pickle global forbidden")
    def persistent_load(self, _): raise pickle.UnpicklingError("Persistent pickle references forbidden")


def read_private(path, pin):
    path = source.canonical(path); check_pinned(path, pin); before = source.FIELDS(path.lstat())
    with path.open("rb") as stream:
        result = RestrictedUnpickler(stream).load()
        if stream.read(1): raise ValueError("Trailing pickle payload forbidden")
    if source.FIELDS(path.lstat()) != before: raise ValueError("Private original changed during decode")
    return result


def first_key(data):
    if (type(data) is not dict or not data or any(type(key) is not str or not key
            or not key.isascii() or any(ord(c) < 32 or ord(c) == 127 for c in key) for key in data)):
        raise ValueError("Original ASCII video identity dictionary required")
    return sorted(data)[0]


def initial_queries(example):
    if type(example) is not dict or set(example) != {"video", "points", "occluded"}:
        raise ValueError("Exact original video/point/occlusion record required")
    video = plain_video(example["video"]); points, occluded = example["points"], example["occluded"]
    if (type(video) is not np.ndarray or video.dtype != np.uint8 or video.ndim != 4 or video.shape[-1] != 3
            or min(video.shape) <= 0 or video.nbytes > MAX_VIDEO_BYTES
            or type(points) is not np.ndarray or points.dtype != np.float32 or points.ndim != 3
            or points.shape[1:] != (video.shape[0], 2) or points.shape[0] == 0
            or type(occluded) is not np.ndarray or occluded.dtype != np.bool_ or occluded.shape != points.shape[:2]):
        raise ValueError("Original full RGB/normalized FP32 track/bool occlusion shapes required")
    indices, unavailable, queries = [], [], []
    # First32 ORIGINAL indices, then standard first-visible query: no replacements.
    for index in range(min(32, len(points))):
        visible = np.flatnonzero(~occluded[index])
        if not len(visible): unavailable.append(index); continue
        frame = int(visible[0]); xy = points[index, frame].astype(np.float64)
        if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
            raise ValueError("Original first-visible normalized query invalid")
        indices.append(index); queries.append((frame, xy[1] * video.shape[1], xy[0] * video.shape[2]))
    if not queries: raise ValueError("First32 original tracks contain no visible initial query; no fallback")
    return video, np.asarray(queries, dtype=np.float64), np.asarray(indices, dtype=np.int64), unavailable


def publish(original, output, pins, deadline, persist):
    acquire_binding(original, pins); selected = []; seen = set()
    for name in PICKLES[:3]:
        source.check_deadline(deadline); data = read_private(original / name, pins["pickles"][name]); key = first_key(data)
        if key in seen: raise ValueError("First lexicographic identities across splits must differ; no fallback")
        seen.add(key); selected.append((name, key, data[key])); del data; gc.collect()
    # All three identities frozen before accessing any future track/visibility.
    selection = [{"pickle_file": name, "video_key": key} for name, key, _ in selected]
    persist(selection); records = []
    for index, (name, key, example) in enumerate(selected):
        source.check_deadline(deadline); video, queries, point_indices, unavailable = initial_queries(example)
        target = output / "inputs" / f"video_{index:03d}.npz"
        with source.private_writer(target) as stream:
            np.savez(stream, video=video, query_points=queries, point_indices=point_indices)
        target.chmod(0o444)
        with np.load(target, allow_pickle=False) as reread:
            if set(reread.files) != {"video", "query_points", "point_indices"} or any(not np.array_equal(reread[k], v) for k, v in (("video", video), ("query_points", queries), ("point_indices", point_indices))):
                raise ValueError("Saved public fullvideo/query-only arrays changed")
        records.append({"file": target.name, **source.identity(target), "source_pickle": name, "video_key": key,
            "frames": int(video.shape[0]), "height": int(video.shape[1]), "width": int(video.shape[2]),
            "point_indices": point_indices.tolist(), "unavailable_original_indices": unavailable,
            "query_count": len(queries), "all_original_frames_retained": True})
    # main's finally performs the single full original post-check, including failures.
    return {"schema": "world-reward-robotap-boots-public-v1", "videos": records, "selection": selection,
        "initial_query_is_external_oracle": True, "query_format": "t,y,x; normalized_xy multiplied by original W,H; no half-pixel offset",
        "future_tracks_or_visibility_public": False, "frame_crop_or_resize": False,
        "training_overlap_verified": False, "challenge_overlap_verified": False, "full_hoi_accuracy_verified": False,
        "tap_query_source": TAP_SOURCE, "public_namespace": PUBLIC_NAMESPACE, "serialization_decoder": MEDIAPY_DECODER_SOURCE}


def source_binding(root, code, revision):
    root, code = map(source.canonical, (root, code))
    if (root != ROOT or re.fullmatch(r"[0-9a-f]{40}", revision) is None or code != root / "jobs" / revision / JOB / "code"
            or Path(__file__) != code / FILES[0]): raise ValueError("Exact immutable public-adapter namespace required")
    markers = {n: source.identity(code.parent / n, readonly=False) for n in ("revision", "source-sha256")}
    if (code.parent / "revision").read_bytes() != (revision + "\n").encode(): raise ValueError("Original producer marker differs")
    if (re.fullmatch(b"[0-9a-f]{64}\n", (code.parent / "source-sha256").read_bytes()) is None
            or Path(source.__file__).resolve() != code / FILES[2]): raise ValueError("Actual complete source helper/markers required")
    return {"files": {n: source.identity(code / n) for n in FILES}, "markers": markers}


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("Private decoding remains on Azure Linux")
    root, code, revision = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    before = source_binding(root, code, revision); protocol = source.read_protocol(code / FILES[3])
    if os.environ.get("WR_IMAGE_ID") != IMAGE or os.environ.get("WR_AZURE_VM02_VERIFIED") != "1":
        raise ValueError("Original CPU image and host-verified Azure VM02 required")
    check_pinned(code / PINS, before["files"][PINS])
    pins = strict_json((code / PINS).read_bytes()); check_pinned(code / PINS, before["files"][PINS]); validate_pins(pins)
    if os.geteuid() != 1000: raise ValueError("Acquisition-owner UID1000 private adapter required")
    output = source.canonical(root / OUTPUT)
    if (os.environ.get("WR_ROBOTAP_PUBLIC_RESERVED") != "1" or not output.is_dir()
            or {p.name for p in output.iterdir()} != {".container.cid"}):
        raise ValueError("Fresh reserved output with only owned container CID required")
    cid = source.canonical(output / ".container.cid")
    if not cid.is_file() or re.fullmatch(b"[0-9a-f]{64}\n?", cid.read_bytes()) is None:
        raise ValueError("Exact owned container CID required")
    original_failure(root / INPUT)
    (output / "inputs").mkdir(mode=0o755); report = {"stage": "external_robotap_oracle_initial_query_public_adapter", "status": "fail", "producer_revision": revision,
        "source_before": before, "serialization_decoder": MEDIAPY_DECODER_SOURCE, "public_namespace": PUBLIC_NAMESPACE,
        "original_public_v1_failure_identity": ORIGINAL_PUBLIC_FAILURE, "original_public_v1_failure_preserved": True, "serialization_compatibility_correction_only": True, "image_id": IMAGE, "azure_vm02_verified_by_host_wrapper": True, "private_future_labels_decoded_by_adapter": True, "future_labels_available_to_inference": False,
        "inference_performed": False, "evaluation_performed": False, "gpu_used": False, "challenge_inputs_used": False, "budget_seconds": BUDGET}
    started = time.monotonic(); error = None
    def expired(*_): raise TimeoutError("Fixed public adapter exceeded180s")
    old = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try:
        def freeze(selection):
            source.save_bytes(output / "selection.json", (json.dumps(selection, indent=2) + "\n").encode(), 0o400)
            report["frozen_selection_before_future_label_access"] = selection
            report["selection_receipt"] = source.identity(output / "selection.json")
        manifest = publish(root / INPUT, output, pins, started + BUDGET, freeze)
        source.save_bytes(output / "inputs/manifest.json", (json.dumps(manifest, indent=2) + "\n").encode(), 0o444)
        report.update(public_files={p.name: source.identity(p) for p in sorted((output / "inputs").iterdir())}, actual_full_frame_matrix=[r["frames"] for r in manifest["videos"]], initial_queries_are_external_oracles=True)
    except Exception as caught:
        error = caught; report.update(error_type=type(caught).__name__, error="Private adapter failed; no inference/evaluation performed",
            cohort_decode_failed_before_query_selection="frozen_selection_before_future_label_access" not in report)
    finally:
        try:
            after = source_binding(root, code, revision); report["source_after"] = after
            if after != before: raise ValueError("Frozen adapter source changed")
            original_failure(root / INPUT); acquire_binding(root / INPUT, pins); report.update(source_after_reverified=True, originals_after_reverified=True)
        except Exception as caught:
            if error is None: error = caught; report["error_type"] = type(caught).__name__
            report["final_integrity_recheck_failed"] = True
        report.update(status="pass" if error is None else "fail", elapsed_seconds=time.monotonic() - started)
        source.save_bytes(output / "report.json", (json.dumps(report, indent=2) + "\n").encode(), 0o400)
        signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term)
    if error is not None: raise RuntimeError("Private RoboTAP adapter failed; inspect sealed tiny receipt") from None


if __name__ == "__main__": main()
