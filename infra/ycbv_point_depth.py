"""Azure RGB-only MoGe2 initialization: exactly three frozen YCBV frame zeroes.

This is an execution/camera-consistency preflight, not a 96-frame prediction,
held-out quality result or metric-scale verification. Only the host reads the
independently pinned acquisition receipt; the GPU receives RGB and public pins.
"""
import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import time

import tudl_holdout_inputs as files

ROOT = Path("/srv/scenesmith/world-reward")
BASE = "validation/ycbv_point_pose_v1"
JOB, OUTPUT = "run_ycbv_point_depth", "depth_init_v1"
PIN_FILE = "configs/ycbv_point_input_pins.json"
IMAGE = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
REVISION = "5c2c4aa229800355648cd268040aa814f8dc94f0"
SCHEMA, PINS_SCHEMA = "world-reward-ycbv-point-rgb-v1", "public_ycbv_point_inputs_pins_v1"
SELECTION = "first_three_sorted_scene_directories_first_96_contiguous_RGB_names_before_private_annotations"
ATTRIBUTION = "YCB-Video: Yu Xiang et al.; BOP conversion: Hodan et al."
STAGE, BUDGET = "public_ycbv_three_frame_zero_native_MoGe2_preflight", 300
WIDTH, HEIGHT, SCENES = 640, 480, (48, 49, 50)
GEOMETRY_SHA = "2f8d5de7d671af16d25fe13c555fd73855d8447bc086bb23aa42e859b303fb05"
SOURCE_FILES = ("infra/ycbv_point_depth.py", "infra/run_ycbv_point_depth.sh",
    "infra/tudl_holdout_inputs.py", "infra/object_synthetic_observations.py",
    "src/world_reward/__init__.py", "src/world_reward/data.py", "src/world_reward/pointmap.py")
FROZEN = {"infra/tudl_holdout_inputs.py": "40ea32a5a85bb53682f1ec8d3aaf4459d8b314498aca67e03b32adf685e0268e",
    "infra/object_synthetic_observations.py": "b3fe54859f478f35ca754cd407d09ea961a46c662246f96e0a7a6b1d59b89940",
    "src/world_reward/pointmap.py": "6c526d76bc97406f9581482f42174b73236237a26029851c5127c08af2917d65",
    "src/world_reward/data.py": "c8200e28900a88394f2821c558cd3693225ae8547331f12f2aff7d225c008a30"}
ACQUISITION_FILES = ("infra/ycbv_point_acquire.py", "infra/run_ycbv_point_acquire.sh",
    "infra/tudl_acquire.py", "configs/ycbv_point_protocol.json")  # Host only.


def validate_pins(pins):
    if type(pins) is not dict or set(pins) != {"schema", "manifest", "acquisition_report"} or pins["schema"] != PINS_SCHEMA:
        raise ValueError("Independent committed public YCBV pins required")
    files._identity_record(pins["manifest"])
    receipt = pins["acquisition_report"]
    if type(receipt) is not dict or set(receipt) != {"bytes", "sha256", "producer_revision", "script_sha256"}:
        raise ValueError("Exact acquisition identity and producer required")
    files._identity_record({k: receipt[k] for k in ("bytes", "sha256")})
    for key, length in (("producer_revision", 40), ("script_sha256", 64)):
        if type(receipt[key]) is not str or re.fullmatch(fr"[0-9a-f]{{{length}}}", receipt[key]) is None:
            raise ValueError("Full immutable acquisition source identity required")


def filenames():
    return tuple(f"scene_{scene:06d}_frame_{frame:06d}.png" for scene in SCENES for frame in range(96))


def public_inputs(directory, pins):
    """Hash all 288 original RGBs without decoding or opening any sibling file."""
    validate_pins(pins); directory = files._canonical(directory)
    names = {"manifest.json", *filenames()}
    if not directory.is_dir() or directory.parts[-3:] != ("validation", "ycbv_point_pose_v1", "inputs") or {p.name for p in directory.iterdir()} != names:
        raise ValueError("Exactly 289 original public files required")
    manifest_id = files.identity(directory / "manifest.json")
    if manifest_id != pins["manifest"]: raise ValueError("Manifest bytes differ before JSON")
    raw = (directory / "manifest.json").read_bytes()
    if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != manifest_id:
        raise ValueError("Manifest changed before JSON interpretation")
    manifest = files.strict_json(raw)
    expected = {"schema": SCHEMA, "revision": REVISION, "license": "MIT", "selection": SELECTION, "attribution": ATTRIBUTION}
    if (type(manifest) is not dict or set(manifest) != {*expected, "images"}
            or any(type(manifest.get(k)) is not str or manifest[k] != v for k, v in expected.items())
            or type(manifest["images"]) is not list or len(manifest["images"]) != 288):
        raise ValueError("Frozen MIT RGB-only cohort required")
    records = []; identities = {}
    for index, row in enumerate(manifest["images"]):
        scene, frame = SCENES[index // 96], index % 96; name = filenames()[index]
        if (type(row) is not dict or set(row) != {"scene_id", "frame_id", "file", "sha256", "width", "height"}
                or any(type(row.get(k)) is not int for k in ("scene_id", "frame_id", "width", "height"))
                or (row["scene_id"], row["frame_id"], row["width"], row["height"]) != (scene, frame, WIDTH, HEIGHT)
                or type(row["file"]) is not str or row["file"] != name
                or type(row["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None):
            raise ValueError("Original scene/frame/order/grid fields required; private fields forbidden")
        identities[name] = files.identity(directory / name)
        if identities[name]["sha256"] != row["sha256"]: raise ValueError("Original RGB SHA differs")
        records.append({**row, "path": directory / name})
    if files.identity(directory / "manifest.json") != manifest_id or {p.name for p in directory.iterdir()} != names:
        raise ValueError("Frozen public cohort changed during validation")
    return records, {"manifest": manifest_id, "RGB_identities": identities}


def marker_identity(path):
    path = files._canonical(path); before = files._state(path)
    if not stat.S_ISREG(before[2]) or before[3] != 1 or before[4] <= 0: raise ValueError("Original regular dispatch marker required")
    value = path.read_bytes()
    if before != files._state(path): raise ValueError("Original marker changed")
    return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}


def bound_source(root, code, revision, *, container=False):
    root, code = files._canonical(root), files._canonical(code)
    if root != ROOT or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None or code != root / "jobs" / revision / JOB / "code":
        raise ValueError("Exact immutable dispatch namespace required")
    if Path(__file__).resolve() != code / SOURCE_FILES[0] or Path(files.__file__).resolve() != code / "infra/tudl_holdout_inputs.py":
        raise ValueError("Original bound execution/readonly helper required")
    helpers = {name: files.identity(code / name) for name in (*SOURCE_FILES, PIN_FILE)}
    if any(helpers[name]["sha256"] != sha for name, sha in FROZEN.items()): raise ValueError("Original numerical/asset helper changed")
    markers = {name: marker_identity(code.parent / name) for name in ("revision", "source-sha256")}
    if (code.parent / "revision").read_bytes() != (revision + "\n").encode() or re.fullmatch(b"[0-9a-f]{64}\n", (code.parent / "source-sha256").read_bytes()) is None:
        raise ValueError("Original dispatch marker contents differ")
    if container:
        for directory in ("infra", "src", "configs"):
            expected = {n for n in (*SOURCE_FILES, PIN_FILE) if n.startswith(directory + "/")}
            if {str(p.relative_to(code)) for p in (code / directory).rglob("*") if p.is_file()} != expected:
                raise ValueError("Only the exact blind helper/public pin closure may be mounted")
        if any((root / name).exists() for name in (BASE + "/eval_private", BASE + "/report.json", "data", "vendor")):
            raise ValueError("Private truth/acquisition/recipes/challenge inputs must not be mounted")
    return {"files": helpers, "markers": markers}


def host_proof(root, code, revision):
    """Host-only acquisition/source proof; never access private annotation values."""
    source = bound_source(root, code, revision); pins = files.strict_json((code / PIN_FILE).read_bytes()); validate_pins(pins)
    pin = pins["acquisition_report"]; path = root / BASE / "report.json"
    receipt_id = files.identity(path)
    if receipt_id != {k: pin[k] for k in ("bytes", "sha256")}: raise ValueError("Independent acquisition bytes differ before JSON")
    raw = path.read_bytes()
    if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != receipt_id:
        raise ValueError("Acquisition receipt changed before JSON interpretation")
    report = files.strict_json(raw); expected = {
        "stage": "external_ycbv_contiguous_rgb_only_acquisition", "status": "pass", "phase": "complete",
        "producer_revision": pin["producer_revision"], "script_sha256": pin["script_sha256"],
        "dataset_revision": REVISION, "license": "MIT", "image_id": IMAGE, "device": "cpu",
        "gpu_used": False, "inference_performed": False, "challenge_inputs_used": False,
        "models_downloaded": False, "train_downloaded": False, "sparse_test_downloaded": False,
        "private_annotations_exported_as_inference_inputs": False, "selection_before_private_annotation_values": True,
        "selected_frames": 288, "all_instances_retained": True, "disposable_archives_removed": True, "source_rehashed_after": True}
    if type(report) is not dict or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()) or report.get("public_manifest") != pins["manifest"]:
        raise ValueError("Successful original CPU acquisition provenance required")
    original = root / "jobs" / pin["producer_revision"] / "run_ycbv_point_acquire" / "code"
    actual = {"files": {n: files.identity(original / n) for n in ACQUISITION_FILES},
        "markers": {n: marker_identity(original.parent / n) for n in ("revision", "source-sha256")}}
    if (actual != report.get("source_helpers") or actual["files"][ACQUISITION_FILES[0]]["sha256"] != pin["script_sha256"]
            or (original.parent / "revision").read_bytes() != (pin["producer_revision"] + "\n").encode()
            or re.fullmatch(b"[0-9a-f]{64}\n", (original.parent / "source-sha256").read_bytes()) is None):
        raise ValueError("Actual acquisition producer helpers/markers differ")
    _, public = public_inputs(root / BASE / "inputs", pins)
    if files.identity(path) != receipt_id: raise ValueError("Original acquisition receipt changed")
    return hashlib.sha256(json.dumps({"source": source, "pins": pins, "receipt": receipt_id,
        "acquisition_source": actual, "public": public}, sort_keys=True).encode()).hexdigest()


def moge_bindings(root, code):
    import object_synthetic_observations as native
    if Path(native.__file__).resolve() != code / "infra/object_synthetic_observations.py": raise ValueError("Original bound model asset helper required")
    model, receipt, asset = native.model_asset(root)
    files._canonical(model); before = files._state(model)
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    spec = util.find_spec("moge")
    if spec is None or not spec.submodule_search_locations: raise ValueError("Pinned installed MoGe missing")
    directory = Path(next(iter(spec.submodule_search_locations)))
    source = native.installed_source(directory, files.strict_json(direct) if direct else {})
    python_files = {str(p.relative_to(directory)): native.identity(p) for p in sorted(directory.rglob("*.py"))}
    if python_files.get("utils/geometry_torch.py", {}).get("sha256") != GEOMETRY_SHA or files._state(model) != before:
        raise ValueError("Original native focal solver/checkpoint changed")
    return model, {"acquisition_report": receipt, "model_asset": asset, "model_state": list(before),
        "model_source": source, "python_source_files": python_files, "moge_version": distribution.version,
        "focal_geometry_source_sha256": GEOMETRY_SHA}


def validate_prediction_arrays(arrays):
    import numpy as np
    from world_reward.pointmap import validate_camera_pointmap
    if type(arrays) is not dict or set(arrays) != {"depth", "points", "mask", "intrinsics", "frame_index"}: raise ValueError("Exact untouched native output arrays required")
    shapes = {"depth": (HEIGHT, WIDTH), "points": (HEIGHT, WIDTH, 3), "mask": (HEIGHT, WIDTH), "intrinsics": (3, 3), "frame_index": ()}
    for key, value in arrays.items():
        dtype = np.bool_ if key == "mask" else np.int64 if key == "frame_index" else np.float32
        if type(value) is not np.ndarray or value.shape != shapes[key] or value.dtype != dtype:
            raise ValueError("Original native full-grid dtype/shape required")
    if arrays["frame_index"].item() != 0: raise ValueError("Only original frame zero is preregistered")
    focal = float(np.hypot(WIDTH, HEIGHT)); K = np.array([[focal, 0., WIDTH / 2], [0., focal, HEIGHT / 2], [0., 0., 1.]])
    return validate_camera_pointmap(arrays["depth"], arrays["points"], arrays["mask"], arrays["intrinsics"], K)


def array_identities(arrays):
    return {key: {"shape": list(value.shape), "dtype": value.dtype.str,
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()} for key, value in arrays.items()}


def exclusive(path, mode):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    os.fchmod(fd, mode)  # Permissions precede the first bytes even under umask 0.
    return os.fdopen(fd, "wb")


def run(root, code, revision, out, report, persist):
    import numpy as np
    pins = files.strict_json((code / PIN_FILE).read_bytes()); records, public = public_inputs(root / BASE / "inputs", pins)
    report.update(input_pins=pins, input_pin_identity=files.identity(code / PIN_FILE), input_manifest=public["manifest"], public_RGB_identities=public["RGB_identities"]); persist()
    started = time.perf_counter(); report["model_read_started"] = True; signal.alarm(BUDGET); binding = None
    try:
        model_path, binding = moge_bindings(root, code); report.update(MoGe_bindings=binding, phase="model_load"); persist()
        import torch
        from moge.model import v2 as module
        from moge.utils import geometry_torch
        from PIL import Image
        if not torch.cuda.is_available() or module.recover_focal_shift is not geometry_torch.recover_focal_shift:
            raise ValueError("Original native CUDA/focal solver required; no fallback")
        report.update(torch_version=torch.__version__, CUDA_version=torch.version.cuda, numpy_version=np.__version__, pillow_version=metadata.version("pillow"), native_focal_solver_modified=False)
        network = module.MoGeModel.from_pretrained(str(model_path)).cuda().eval()
        fov = float(np.degrees(2 * np.arctan(WIDTH / (2 * float(np.hypot(WIDTH, HEIGHT))))))
        report["native_infer_kwargs"] = {"fov_x": fov}; torch.cuda.synchronize()
        for record in records[::96]:
            report.update(phase="frame_zero", active_scene=record["scene_id"]); persist()
            with Image.open(record["path"]) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT): raise ValueError("Original RGB640x480 PNG required")
                rgb = np.asarray(image).copy()
            if rgb.dtype != np.uint8 or rgb.shape != (HEIGHT, WIDTH, 3) or files.identity(record["path"]) != public["RGB_identities"][record["file"]]: raise ValueError("Original RGB bytes/grid changed")
            tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
            report["native_calls_attempted"] += 1; persist()
            with torch.inference_mode(): predicted = network.infer(tensor[None], fov_x=fov)
            report["native_calls_returned"] += 1
            arrays = {key: predicted[key][0].cpu().numpy() for key in ("depth", "points", "mask", "intrinsics")}
            arrays["frame_index"] = np.array(0, dtype=np.int64); checks = validate_prediction_arrays(arrays)
            torch.cuda.synchronize(); report["native_calls_completed"] += 1
            original = array_identities(arrays); path = out / (Path(record["file"]).stem + ".npz")
            with exclusive(path, 0o444) as stream: np.savez_compressed(stream, **arrays)
            with np.load(path, allow_pickle=False) as archive: saved = {k: archive[k] for k in archive.files}
            validate_prediction_arrays(saved)
            if array_identities(saved) != original: raise ValueError("Saved native bytes changed")
            report["outputs"].append({"file": path.name, "scene_id": record["scene_id"], "frame_id": 0,
                **files.identity(path), "rgb_sha256": record["sha256"], "decoded_RGB_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "arrays": original, "pointmap_checks": checks}); persist()
            del arrays, saved, predicted, tensor, rgb
        if module.recover_focal_shift is not geometry_torch.recover_focal_shift: raise ValueError("Original native focal solver changed")
        if any(report[k] != 3 for k in ("native_calls_attempted", "native_calls_returned", "native_calls_completed")) or len(report["outputs"]) != 3: raise ValueError("Exactly three complete original native calls required")
        names = {"report.json", ".container.cid", *[row["file"] for row in report["outputs"]]}
        if {p.name for p in out.iterdir()} != names: raise ValueError("Only three immutable NPZs and receipt/CID allowed")
        report.update(status="pass", phase="complete", actual_native_inference=True, saved_native_arrays_byte_exact=True, outputs_completed=3)
    finally:
        # Original bindings are checked even after a failed native call; no
        # failed run is rescued or any partial output claimed to be complete.
        checks_ok = True
        try:
            report["all_public_RGB_after_reverified"] = (public_inputs(root / BASE / "inputs", pins)[1] == public
                and files.identity(code / PIN_FILE) == report["input_pin_identity"])
        except Exception: report["all_public_RGB_after_reverified"] = False
        checks_ok &= report["all_public_RGB_after_reverified"]
        if binding is not None:
            try: report["model_assets_after_reverified"] = moge_bindings(root, code)[1] == binding
            except Exception: report["model_assets_after_reverified"] = False
            checks_ok &= report["model_assets_after_reverified"]
        report["GPU_budget_elapsed_seconds"] = time.perf_counter() - started
        report["GPU_budget_includes_model_hash_load_decode_infer_export_reverification"] = True
        signal.alarm(0)
        if not checks_ok: raise ValueError("Original public/model bindings failed post-verification")
    if report["GPU_budget_elapsed_seconds"] > BUDGET: raise TimeoutError("Preregistered model-read GPU budget exceeded")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument("--host-proof", action="store_true"); args = parser.parse_args(argv)
    root, code, revision = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    if args.host_proof: print(host_proof(root, code, revision)); return
    if platform.system() != "Linux" or os.getuid() != 1000 or os.environ.get("WR_IMAGE_ID") != IMAGE or os.environ.get("WR_AZURE_VM02_VERIFIED") != "1" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise ValueError("Exact Azure VM02 image/UID1000/network-none required")
    proof = os.environ.get("WR_YCBV_HOST_PROOF_SHA256", "")
    if re.fullmatch(r"[0-9a-f]{64}", proof) is None: raise ValueError("Independent host acquisition/source proof required")
    before = bound_source(root, code, revision, container=True); out = files._canonical(root / BASE / OUTPUT)
    if not out.is_dir() or {p.name for p in out.iterdir()} != {".container.cid"}: raise ValueError("Exclusive fresh initialization namespace required")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision,
        "script_sha256": before["files"][SOURCE_FILES[0]]["sha256"], "source_helpers": before, "image_id": IMAGE,
        "host_acquisition_source_proof_sha256": proof, "acquisition_receipt_checked_host_only": True,
        "private_truth_read": False, "acquisition_receipt_mounted": False, "source_camera_calibration_used": False,
        "sensor_depth_used": False, "human_scale_used": False, "DA3_used": False, "challenge_inputs_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "metric_scale_accuracy_verified": False,
        "accuracy_verified": False, "heldout_performance_verified": False, "challenge_overlap_verified": False,
        "full_trajectories_predicted": False, "full_HOI_verified": False, "CARI4D_victory_verified": False,
        "adoption_performed": False, "model_read_started": False, "native_calls_attempted": 0,
        "native_calls_returned": 0, "native_calls_completed": 0, "outputs": [], "budget_seconds": BUDGET,
        "whole_pilot_GPU_budget_seconds": 3600, "init_budget_must_accumulate_into_pilot": True,
        "native_invalid_pixels_filled": False, "native_validity_preserved": True, "network": "none"}
    started = time.perf_counter(); previous = {}
    def expired(*_): raise TimeoutError("Frozen initialization time budget exhausted")
    with exclusive(out / "report.json", 0o400) as stream:
        def persist():
            stream.seek(0); stream.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode()); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        try:
            previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
            persist(); run(root, code, revision, out, report, persist)
        except Exception as exc:
            report.update(status="fail", error_type=type(exc).__name__, error="Frozen RGB-only preflight failed; paths, URLs and private values omitted")
        finally:
            signal.alarm(0)
            for s, handler in previous.items(): signal.signal(s, handler)
            report["elapsed_seconds"] = time.perf_counter() - started
            try: report["sources_after_reverified"] = bound_source(root, code, revision, container=True) == before
            except Exception: report["sources_after_reverified"] = False
            if not report["sources_after_reverified"]: report["status"] = "fail"
            persist()
    if report["status"] != "pass": raise SystemExit(1)


if __name__ == "__main__": main()
