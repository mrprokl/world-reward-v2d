"""Frozen whole-native-support anchor on twelve NEW TUM RGB frames.

RGB only enters both native backends. Genuine original model/source/loader
helpers are imported unchanged; their TUD-L input functions are never called.
No private rendering, geometry, masks, calibration or scores enter this job.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import tum_rgbd_depth_inputs as inputs
import whole_support_anchor_contract as contract
import rgbd_anchor_contract as grid_native
import tudl_anchor_infer as native

ROOT = Path("/srv/scenesmith/world-reward")
BASE = inputs.BASE
PIN_FILE = "configs/tum_rgbd_depth_input_pins.json"
STAGE = "public_tum_rgbd_whole_support_anchored_depth_predictions"
BUDGET = 300
IMAGE_ID = native.IMAGE_ID
GRID = contract.Grid(640, 480, ((800., 0., 320.), (0., 800., 240.), (0., 0., 1.)))
OUTPUT = "anchor_predictions_v1"
SOURCE_FILES = ("infra/tum_rgbd_depth_infer.py", "infra/tum_rgbd_depth_inputs.py", "infra/run_tum_rgbd_depth_infer.sh",
    "infra/whole_support_anchor_contract.py", "infra/rgbd_anchor_contract.py", *native.SOURCE_FILES)
FROZEN_HELPERS = {"infra/tudl_anchor_infer.py": "2ed48a897fb8f161ce761083d3014188aca0298464bcb5d833ba8c4101beeb16",
    "infra/rgbd_anchor_contract.py": "d3541ce4deb02eff4c7e698425f045bfb246292a89b42d5b77ca2941216d0269",
    "infra/whole_support_anchor_contract.py": "967140d43b6ca34ca2d5f2dbeaa39e9562c0851dded77ce6436907b25be1d484"}


def bound_source(root, code, revision, executing):
    root, code = Path(root), Path(code)
    if (root != ROOT or inputs._canonical(root) != root or not root.is_dir()
            or type(revision) is not str or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or code != root / "jobs" / revision / "run_tum_rgbd_depth_infer/code"
            or inputs._canonical(code) != code or not code.is_dir() or Path(executing) != code / SOURCE_FILES[0]):
        raise ValueError("Actual immutable whole-support prediction producer namespace required")
    modules = {"infra/tum_rgbd_depth_inputs.py": inputs, "infra/whole_support_anchor_contract.py": contract, "infra/rgbd_anchor_contract.py": grid_native,
        "infra/tudl_anchor_infer.py": native, "infra/tudl_holdout_inputs.py": native.inputs,
        "infra/tudl_infer.py": native.moge_native, "infra/object_synthetic_observations.py": native.moge_assets,
        "infra/da3_metric_infer.py": native.da3, "infra/da3_metric_acquire.py": native.da3.acquisition}
    if any(Path(module.__file__).resolve() != code / name for name, module in modules.items()):
        raise ValueError("Genuine numerical/asset helpers must resolve in frozen code")
    if (Path(grid_native.validate_camera_pointmap.__code__.co_filename).resolve() != code / "src/world_reward/pointmap.py"
            or Path(inputs.identity.__code__.co_filename).resolve() != code / "infra/tudl_holdout_inputs.py"
            or Path(native.moge_assets.sha256.__code__.co_filename).resolve() != code / "src/world_reward/data.py"):
        raise ValueError("Genuine original pointmap/hash primitives required")
    for directory in ("infra", "src"):
        expected = {name for name in SOURCE_FILES if name.startswith(directory + "/")}
        actual = {str(path.relative_to(code)) for path in (code / directory).rglob("*") if path.is_file()}
        if actual != expected or any(path.is_symlink() for path in (code / directory).rglob("*")):
            raise ValueError("Only the complete blind inference primitive source closure may be exposed")
    if {str(path.relative_to(code)) for path in (code / "configs").rglob("*") if path.is_file()} != {PIN_FILE}:
        raise ValueError("Only one independent public input pin may be exposed; no recipe or other configuration")
    helpers = {name: inputs.identity(code / name) for name in SOURCE_FILES}
    if any(helpers[name]["sha256"] != digest for name, digest in FROZEN_HELPERS.items()):
        raise ValueError("Original native/grid helper bytes changed; no retune or patch")
    return helpers


def load_public(root, code):
    path = code / PIN_FILE; before = inputs.identity(path)
    pins = inputs.strict_json(path.read_bytes()); records, manifest = inputs.public_inputs(root / BASE / "inputs", pins)
    if inputs.identity(path) != before: raise ValueError("Independent committed new TUM pins changed")
    return records, {"input_pin_identity": before, "input_manifest": manifest, "input_pins": pins,
        "public_records": [{key: value for key, value in row.items() if key != "path"} for row in records],
        "ordered_frames": [list(pair) for pair in inputs.ordered_frames(pins)]}


def read_rgb(record):
    from PIL import Image
    before = inputs.identity(record["path"])
    if before["sha256"] != record["sha256"]:
        raise ValueError("Pinned original RGB changed before decode")
    with Image.open(record["path"]) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (GRID.width, GRID.height):
            raise ValueError("Exact TUM RGB PNG640x480 required")
        rgb = np.asarray(image).copy()
    if rgb.dtype != np.uint8 or rgb.shape != (*GRID.shape, 3) or inputs.identity(record["path"]) != before:
        raise ValueError("Original RGB grid/bytes changed during decode")
    return rgb, hashlib.sha256(rgb.tobytes()).hexdigest()


def verify_bindings(root, code, revision, report):
    if bound_source(root, code, revision, Path(__file__)) != report["source_helpers"]:
        raise ValueError("Frozen whole-support producer closure changed")
    if "input_pins" in report:
        _, public = load_public(root, code)
        if any(report.get(key) != value for key, value in public.items()): raise ValueError("Frozen public RGB/pins changed")
    if "MoGe_bindings" in report and native.moge_bindings(root)[1] != report["MoGe_bindings"]:
        raise ValueError("Original MoGe model/source changed")
    if "DA3_assets" in report and native.da3.pinned_assets(root)[0] != report["DA3_assets"]:
        raise ValueError("Original DA3 model/source changed")
    if "DA3_dependency" in report and native.da3_dependency(root) != report["DA3_dependency"]:
        raise ValueError("Original native dependency changed")
    report["source_assets_after_reverified"] = True


def cleanup_failed_outputs(output, report):
    """Remove only this fresh failed job's named arrays; keep original receipt."""
    if report["status"] == "pass": return
    expected = {Path(name).stem + ".npz" for name in inputs.filenames(report["input_pins"])} if "input_pins" in report else set()
    removed = 0
    for path in output.iterdir():
        if path.name == "report.json": continue
        if path.name not in expected or path.is_symlink() or not path.is_file():
            raise ValueError("Unexpected failed-job artifact preserved; no broad cleanup")
        path.unlink(); removed += 1
    report.update(partial_prediction_arrays_removed=True, partial_prediction_array_count_removed=removed)


def run(root, code, revision, output, report, persist):
    records, public = load_public(root, code)
    model_path, binding = native.moge_bindings(root)
    assets, source = native.da3.pinned_assets(root); dependency = native.da3_dependency(root)
    report.update(**public, MoGe_bindings=binding, DA3_assets=assets, DA3_dependency=dependency, phase="MoGe_model_load"); persist()
    import torch
    from moge.model import v2 as module
    from moge.utils import geometry_torch
    if not torch.cuda.is_available(): raise RuntimeError("Actual remote CUDA required; no CPU/local fallback")
    original = geometry_torch.recover_focal_shift
    if module.recover_focal_shift is not original: raise ValueError("Exact original native focal callback alias required")
    native.seed(torch); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    report.update(torch=torch.__version__, CUDA=torch.version.cuda)
    network = module.MoGeModel.from_pretrained(str(model_path)).cuda().eval()
    torch.cuda.reset_peak_memory_stats(); baselines = []; decoded = []; originals = []
    for record in records:
        report.update(phase="MoGe_fixed_prior", active_file=record["file"]); persist()
        rgb, digest = read_rgb(record); tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float() / 255
        native.seed(torch); report["MoGe_attempts"] += 1; persist()
        prediction, _ = contract.checked_moge_infer(torch, GRID, module, original, network, tensor,
            report["native_focal_solver_calls"], {"file": record["file"], "scene_id": record["scene_id"],
                "frame_id": record["frame_id"], "pass": "fixed_prior"})
        report["MoGe_returns"] += 1
        arrays, _ = contract.camera_arrays(GRID, *(prediction[key][0].detach().cpu().numpy()
            for key in ("depth", "points", "mask", "intrinsics")))
        contract.validate_baseline(GRID, arrays); torch.cuda.synchronize(); report["MoGe_calls_completed"] += 1
        baselines.append({key: value.copy() for key, value in arrays.items()}); decoded.append(digest)
        originals.append(contract.array_identities(arrays)); persist(); del prediction, tensor, arrays
    del network; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    report.update(phase="DA3_model_load", MoGe_model_released_before_DA3_load=True); persist()
    network, processor, postprocessor = native.load_da3(root, torch, assets, source, dependency, report)
    raw_depths = []; ratios = []; details = []
    for index, record in enumerate(records):
        report.update(phase="DA3_native", active_file=record["file"]); persist()
        rgb, digest = read_rgb(record)
        if digest != decoded[index]: raise ValueError("Both backends must consume byte-identical decoded RGB")
        report["DA3_attempts"] += 1; persist()
        depth, detail = native.da3_frame(torch, network, processor, postprocessor, rgb)
        report["DA3_returns"] += 1; report["DA3_calls_completed"] += 1
        diagnostic = contract.whole_support_ratio(GRID, baselines[index]["depth"], baselines[index]["validity"], depth)
        diagnostic.update(scene_id=record["scene_id"], frame_id=record["frame_id"])
        raw_depths.append(depth); ratios.append(diagnostic); details.append(detail)
        report["whole_support_diagnostics"] = ratios.copy(); persist()
        contract.require_support(diagnostic)
    del network, processor, postprocessor; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    coefficients = contract.scene_anchors(GRID, ratios, list(inputs.ordered_frames(report["input_pins"])))
    report.update(phase="immutable_output", scene_anchor_coefficients=[coefficients[scene] for scene in (1, 2, 3)],
        scene_ratios=[[row["ratio_median"] for row in ratios[start:start + 4]] for start in (0, 4, 8)])
    for index, record in enumerate(records):
        scene, frame = record["scene_id"], record["frame_id"]; coefficient = coefficients[scene]
        arrays = contract.candidate_arrays(GRID, baselines[index], raw_depths[index], coefficient, scene, frame)
        if contract.array_identities(baselines[index]) != originals[index]: raise ValueError("Unchanged MoGe baseline bytes lost")
        checks = contract.validate_prediction_arrays(GRID, arrays, scene, frame, coefficient)
        path = output / (Path(record["file"]).stem + ".npz")
        with path.open("xb") as stream: np.savez_compressed(stream, **arrays)
        path.chmod(0o444)
        with np.load(path, allow_pickle=False) as archive: saved = {key: archive[key].copy() for key in archive.files}
        contract.validate_prediction_arrays(GRID, saved, scene, frame, coefficient)
        identities = contract.array_identities(arrays)
        if contract.array_identities(saved) != identities: raise ValueError("Saved complete prediction arrays differ")
        report["outputs"].append({"file": path.name, "scene_id": scene, "frame_id": frame, **inputs.identity(path),
            "rgb_sha256": record["sha256"], "decoded_RGB_sha256": decoded[index], "arrays": identities,
            "original_MoGe_arrays": originals[index], "pointmap_checks": checks, "scene_anchor": coefficient,
            "whole_support": ratios[index], **details[index]}); persist()
    verify_bindings(root, code, revision, report)
    calls = report["native_focal_solver_calls"]
    if (any(report[key] != 12 for key in ("MoGe_attempts", "MoGe_returns", "MoGe_calls_completed", "DA3_attempts", "DA3_returns", "DA3_calls_completed"))
            or len(calls) != 12 or len(report["outputs"]) != 12
            or any(any(call.get(key) != expected for key, expected in {"file": record["file"], "scene_id": record["scene_id"],
                "frame_id": record["frame_id"], "pass": "fixed_prior", "focal_prior_supplied": True,
                "original_solver_returned": True}.items()) or type(call.get("native_nearest64_valid_pixels")) is not int
                or not 2 <= call["native_nearest64_valid_pixels"] <= 4096 for call, record in zip(calls, records))):
        raise ValueError("Exactly twelve validated original native calls per backend required")
    if {path.name for path in output.iterdir()} != {"report.json", *[row["file"] for row in report["outputs"]]}:
        raise ValueError("Only twelve frozen prediction NPZs and one receipt allowed")
    for row in report["outputs"]:
        if inputs.identity(output / row["file"]) != {key: row[key] for key in ("sha256", "bytes")}:
            raise ValueError("Frozen complete prediction artifact changed")
    report.update(status="pass", phase="complete", outputs_completed=12, original_frame_coverage_verified=True,
        sources_assets_rechecked=True, saved_arrays_reread_verified=True, baseline_MoGe_bytes_preserved=True,
        actual_native_inference=True, native_camera_support_verified=True,
        peak_GPU_allocated_bytes=int(torch.cuda.max_memory_allocated())); report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote Linux CUDA inference with network none required")
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    helpers = bound_source(root, code, revision, Path(__file__)); output = root / BASE / OUTPUT
    if image != IMAGE_ID or inputs._canonical(output) != output or not output.is_dir() or any(output.iterdir()):
        raise ValueError("Exact original VM02 image and fresh reserved prediction output required")
    report = {"stage": STAGE, "status": "fail", "phase": "integrity", "producer_revision": revision, "image_id": image,
        "script_sha256": helpers[SOURCE_FILES[0]]["sha256"], "source_helpers": helpers, "budget_seconds": BUDGET,
        "network": "none", "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False, "training_overlap_verified": False,
        "full_license_closure_verified": False, "metric_scale_accuracy_verified": False, "calibration_accuracy_verified": False,
        "new_TUM_recordings_relative_to_TUDL": True, "different_object_identity_fully_proven": False,
        "sparse_early_frames_not_complete_temporal_validation": True, "real_world_generalization_verified": False, "human_contact_or_full_HOI_evaluated": False,
        "human_or_object_geometry_changed": False, "per_frame_anchor_fit": False, "offset_fit": False,
        "semantic_background_exclusion_performed": False, "support_diagnostics_recorded_before_gate": True,
        "support_rule": "median_of_four_per_image_median_MoGe_Z_div_native_DA3_Z_on_all_original_MoGe_valid_pixels_per_scene",
        "minimum_pairs": contract.MIN_PAIRS, "minimum_grid_coverage": contract.MIN_COVERAGE,
        "dispersion_recorded_not_filtered": True, "semantic_mask_or_confidence_weight_used": False, "fixed_camera_K": GRID.K.tolist(),
        "prediction_pixel_convention": "pixel_centres_plus_0.5", "candidate_validity": "exact_unchanged_native_MoGe_validity_no_drop_or_expansion",
        "apply_mask": False, "force_projection": True, "native_camera_support_verified": False,
        "native_DA3_arguments": native.da3.NATIVE_ARGUMENTS, "checkpoint_states_loaded": 0, "checkpoint_load_strict": False,
        "native_sky_correction_unchanged": True, "native_sky_threshold": .3, "native_sky_quantile": .99,
        "metric_depth_scaling_applied_once": True, "metric_depth_factor_rule": "actual_processed_focal_mean_divided_by_300",
        "postprocess": "metric_camera_Z_then_bilinear_align_corners_false_then_one_scene_anchor_then_fixed_K_plus_0.5",
        "confidence_validity_exclusion": False, "pointmap_geometry_filled": False, "seed": 0,
        "RNG_seed_reset_before_each_forward": True, "native_focal_solver_calls": [], "outputs": [], "whole_support_diagnostics": [],
        "MoGe_attempts": 0, "MoGe_returns": 0, "MoGe_calls_completed": 0, "DA3_attempts": 0, "DA3_returns": 0, "DA3_calls_completed": 0}
    started = time.perf_counter(); path = output / "report.json"; post_error = None
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole public TUM native-support anchor inference exceeded300s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, code, revision, output, report, persist)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            try: verify_bindings(root, code, revision, report)
            except Exception as error:
                report.update(status="fail", source_assets_after_reverified=False, source_postcheck_error_type=type(error).__name__,
                    source_postcheck_error=str(error)); post_error = error
            finally:
                try: cleanup_failed_outputs(output, report)
                except Exception as error:
                    report.update(status="fail", failed_output_cleanup_error_type=type(error).__name__, failed_output_cleanup_error=str(error))
                    if post_error is None: post_error = error
                signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o400)
            if post_error is not None: raise post_error


if __name__ == "__main__": main()
