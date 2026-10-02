"""Direct-head representation probe of a NEW frame-zero shared identity.

Original predicted poses are retained; original predicted geometry is not the
target. Twelve fixed frames establish ABI fidelity only, never identity quality
or full-episode validity. No inverse fit, conversion, alignment or GT is used.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

import cari_converter as convert
from cari_target_reseal import (FAILED_SHA, SEALED_SHA, failed_source, regular,
                               sealed_inputs, source_chain, strict_torch)
from world_reward.data import sha256

STAGE = "world_reward_cari_direct_head_shared_identity_representation_probe"
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
RESEAL_SHA = "314f80e8bbb19e9883c75ba262c13417cd34754c0eba364516f1060073c9e825"
INDICES = (0, 72, 143, 215, 287, 359, 430, 502, 574, 646, 717, 789)
BUDGET = 180
VERTICES = 18439
NATIVE_MAX_POINT_MM = .01


def shared_inputs(params):
    """Choose identity before any geometry/error, with no mutation or repair."""
    if set(params) != set(convert.PARAMETER_DIMS): raise ValueError("Exact seven native blocks required")
    for key, dim in convert.PARAMETER_DIMS.items():
        value = params[key]
        if (not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value)
                or value.dtype != np.float32 or value.shape != (790, dim) or not np.isfinite(value).all()):
            raise ValueError("Every original native block must be complete finite FP32")
    if np.any(params["mhr_face"] != 0): raise ValueError("Original face expressions must be zero")
    selected = {key: value[list(INDICES)].copy() for key, value in params.items()}
    for key in ("mhr_shape", "mhr_scale"):
        selected[key] = np.repeat(params[key][:1], len(INDICES), axis=0)
    return selected


def validate_direct(target, model_params, selected):
    if (target.dtype != np.float32 or target.shape != (len(INDICES), VERTICES, 3)
            or model_params.dtype != np.float32 or model_params.shape != (len(INDICES), 204)
            or not np.isfinite(target).all() or not np.isfinite(model_params).all()):
        raise ValueError("Complete finite direct FP32 vertices and 204 parameters required")
    if any(row.tobytes() != model_params[0, 136:].tobytes() for row in model_params[:, 136:]):
        raise ValueError("Direct 68 native scales must be bit-identical across every frame")
    for key in ("mhr_shape", "mhr_scale"):
        if any(row.tobytes() != selected[key][0].tobytes() for row in selected[key]):
            raise ValueError("Explicit fixed frame-zero identity required")


def freeze(out, target, model_params, selected):
    validate_direct(target, model_params, selected)
    tp, pp = out / "target_f32.npy", out / "direct_parameters.npz"
    with tp.open("xb") as stream: np.save(stream, target, allow_pickle=False)
    tp.chmod(0o444)
    with pp.open("xb") as stream:
        np.savez_compressed(stream, frame_index=np.asarray(INDICES, np.int64), model_parameters=model_params,
            shape=selected["mhr_shape"][0], scale_pca=selected["mhr_scale"][0], expression=selected["mhr_face"])
    pp.chmod(0o444)
    target = np.load(tp, mmap_mode="r", allow_pickle=False)
    with np.load(pp, allow_pickle=False) as file: saved = {k: file[k].copy() for k in file.files}
    if (target.flags.writeable or not np.array_equal(saved["model_parameters"], model_params)
            or not np.array_equal(saved["frame_index"], INDICES)):
        raise ValueError("Saved prediction identity differs before independent replay")
    return target, saved, [(tp, sha256(tp)), (pp, sha256(pp))]


def residuals(target, recovered, *, units):
    if (units not in ("m", "mm") or recovered.shape != target.shape
            or recovered.dtype not in (np.float32, np.float64) or not np.isfinite(recovered).all()):
        raise ValueError("Finite complete independent replay and explicit units required")
    delta = recovered.astype(np.float64) * (1000. if units == "m" else 1.) - target.astype(np.float64)*1000.
    error = np.linalg.norm(delta, axis=-1)
    return {"per_frame_mean_mm": error.mean(axis=1).tolist(), "max_point_mm": float(error.max())}


def preserved_failures(root):
    first = failed_source(root)
    second = regular(root / "outputs/episode_000000/cari_target_reseal_refined_v1/report.json")
    convert._require_hash(second, RESEAL_SHA)
    receipt = json.loads(second.read_text())
    expected = {"stage": "world_reward_cari_native_target_strict_reseal", "status": "fail",
        "phase": "paired_target_comparison", "error": "Fresh worker source/version/settings/target identity differs",
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "original_conversion_rerun": False, "solver_calls": 0}
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in expected.items()):
        raise ValueError("Exact unchanged failed reseal receipt required")
    return [(first, FAILED_SHA), (second, RESEAL_SHA)]


def run(root, out, report, persist):
    original, params, receipt, frozen = sealed_inputs(root)
    del original  # Original errors/poses/geometry never select the new identity.
    frozen += preserved_failures(root)
    assets, vendor, more = source_chain(root, params, receipt); frozen += more
    selected = shared_inputs(params)
    source_ids = {k: convert.canonical_array_identity(v) for k, v in params.items()}
    selected_ids = {k: convert.canonical_array_identity(v) for k, v in selected.items()}
    report.update(native_parameter_identities=source_ids, selected_native_parameter_identities=selected_ids,
        inference_source_identity=receipt["inference_source_identity"], body_assets=receipt["body_assets"],
        original_decoder_identity=receipt["decoder_identity"], shared_identity_frame_index=0); persist()
    tool = regular(root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py")
    model = regular(root / "weights/mhr/mhr_model.pt")
    convert._require_hash(tool, convert.CONVERTER_SHA256); convert._require_hash(model, convert.REFERENCE_MODEL_SHA256)
    frozen += [(tool, convert.CONVERTER_SHA256), (model, convert.REFERENCE_MODEL_SHA256)]
    torch = strict_torch()
    actual, _ = convert.validate_native_bundle(torch.load(more[1][0], map_location="cpu", weights_only=False), 790)
    if any(not np.array_equal(actual[k], params[k]) for k in params): raise ValueError("Original bundle parameters differ")
    del actual
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    from lib_mhr.mhr_layer import MHRLayer
    layerpath = Path(sys.modules[MHRLayer.__module__].__file__).resolve()
    if not layerpath.is_relative_to(native / "lib_mhr"): raise ValueError("Original MHRLayer source path differs")
    tensor_inputs = {k: torch.tensor(v.copy(), device="cuda", dtype=torch.float32) for k, v in selected.items()}
    report.update(phase="direct_head_decode", torch=torch.__version__, CUDA=torch.version.cuda,
        settings={"seed": 0, "deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TF32": False, "jit_optimized_execution": False, "dtype": "float32", "CPU_threads": 4}); persist()
    with torch.no_grad(), torch.jit.optimized_execution(False):
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),
            checkpoint_path=assets / "model.ckpt", buffer_path=root / "never_use_unverified_direct_buffer.pt",
            mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
        if layer.decoder_identity() != receipt["decoder_identity"]: raise ValueError("Original decoder assets differ")
        backend = layer.backend
        context = backend._vertices_context(tensor_inputs, detach_fixed=False)
        trans, body_pose, shape, scale = backend._mutable_vertices_inputs(tensor_inputs, context)
        head = context.head
        headpath = Path(sys.modules[type(head).__module__].__file__).resolve()
        if (head.enable_hand_model or not headpath.is_relative_to(Path("/workspace/v2d_sam3d_body/lib/sam_3d_body"))):
            raise ValueError("Exact full-body checkpoint head required")
        report.update(native_layer_sha256=sha256(layerpath), native_head_sha256=sha256(headpath)); persist()
        targets, direct = [], []
        for i in range(len(INDICES)):
            verts, controls = head.mhr_forward(global_trans=trans[i:i+1]*context.flip,
                global_rot=context.global_rot[i:i+1], body_pose_params=body_pose[i:i+1],
                hand_pose_params=context.hand[i:i+1], scale_params=scale[i:i+1], shape_params=shape[i:i+1],
                expr_params=context.face[i:i+1], return_model_params=True)
            report["actual_direct_head_calls"] += 1
            expected_scales = head.scale_mean[None, :] + scale[i:i+1] @ head.scale_comps
            if not torch.equal(controls[:, 136:], expected_scales): raise ValueError("Native head scale expansion differs")
            targets.append((verts*context.flip).cpu().numpy().copy()); direct.append(controls.cpu().numpy().copy())
        target, direct = np.concatenate(targets), np.concatenate(direct)
        target, saved, new_files = freeze(out, target, direct, selected); frozen += new_files
        report.update(phase="independent_native_replay", predictions_frozen_before_replay=True,
            target_vertices_identity=convert.canonical_array_identity(target),
            direct_model_parameters_identity=convert.canonical_array_identity(saved["model_parameters"]),
            file_inventory=[{"file": p.name, "sha256": h, "bytes": p.stat().st_size} for p, h in new_files]); persist()
        recovered = layer.mhr_forward_vertices(tensor_inputs).cpu().numpy().copy()
        report["actual_native_vertices_only_calls"] += 1
        report["native_vs_direct"] = residuals(target, recovered, units="m"); persist()
        if report["native_vs_direct"]["max_point_mm"] > NATIVE_MAX_POINT_MM:
            raise ValueError("Native translation/pose preservation exceeds fixed .01mm point bound")
        report.update(phase="independent_official_reference"); persist()
        spec = importlib.util.spec_from_file_location("world_reward_direct_reference", tool)
        converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
        if Path(converter.MHR.run.__code__.co_filename).resolve() != tool.resolve():
            raise ValueError("Pinned official reference callback required")
        reference = converter.MHR(str(model), "cuda", chunk=16, precision="float32")
        if (reference.mdtype != torch.float32 or reference.dtype != torch.float64
                or reference.chunk != 16 or reference.device.type != "cuda"):
            raise ValueError("Exact FP32 reference model with FP64 residual arithmetic required")
        z = np.concatenate([saved["model_parameters"][0, 136:], saved["shape"]])[None]
        recovered_mm, _ = reference.run(torch.tensor(saved["model_parameters"][:, :136], device="cuda", dtype=torch.float64),
                                        torch.tensor(z, device="cuda", dtype=torch.float64))
        report["actual_official_reference_calls"] += 1
        report["reference_vs_direct"] = residuals(target, recovered_mm.cpu().numpy(), units="mm"); persist()
    for path, digest in frozen: convert._require_hash(regular(path), digest)
    if (any(convert.canonical_array_identity(v) != source_ids[k] for k, v in params.items())
            or any(convert.canonical_array_identity(v) != selected_ids[k] for k, v in selected.items())):
        raise ValueError("Original or selected native controls changed")
    source_chain(root, params, receipt)
    if (report["actual_direct_head_calls"] != 12 or report["actual_native_vertices_only_calls"] != 1
            or report["actual_official_reference_calls"] != 1): raise ValueError("Exact predeclared execution counts required")
    if max(report["reference_vs_direct"]["per_frame_mean_mm"]) > 2.:
        raise ValueError("Independent reference exceeds unchanged2mm representation bound")
    report.update(status="pass", phase="complete", source_bindings_runtime_verified=True,
        direct_representation_fidelity_verified=True, original_native_parameters_unchanged=True,
        actual_total_native_head_calls=13)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != IMAGE):
        raise ValueError("Canonical Azure offline root, source revision and pinned image required")
    out = root / "outputs/episode_000000/cari_direct_shared_probe_refined_v1"
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty new diagnostic output required")
    start = time.perf_counter()
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "script_sha256": sha256(Path(__file__)),
        "producer_revision": revision, "image_id": image, "budget_seconds": BUDGET, "network": "none",
        "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
        "research_only": True, "submission_produced": False, "adoption_authorized": False, "accuracy_verified": False,
        "new_identity_source": True, "identity_selection_rule": "native_frame_zero_protocol_order_not_error",
        "original_native_geometry_recovered": False, "strict_reproducibility_verified": False,
        "historical_target_recovered": False, "full_frame_fit_verified": False, "original_conversion_rerun": False,
        "old_target_read": False, "solver_calls": 0, "alignment_calls": 0, "source_bindings_runtime_verified": False,
        "direct_representation_fidelity_verified": False, "frames": 12, "original_frames": 790, "frame_indices": list(INDICES),
        "native_max_point_mm_bound": NATIVE_MAX_POINT_MM, "reference_mean_mm_bound": 2.,
        "reference_max_point_is_diagnostic_only": True, "actual_direct_head_calls": 0,
        "actual_native_vertices_only_calls": 0, "actual_official_reference_calls": 0,
        "reference_precision": "float32", "reference_residual_dtype": "float64", "reference_model_chunk": 16,
        "official_converter_sha256": convert.CONVERTER_SHA256, "reference_model_sha256": convert.REFERENCE_MODEL_SHA256,
        "runtime_helper_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            ("cari_target_reseal.py", "cari_identity_probe_runtime.py", "cari_converter.py")},
        "sealed_report_sha256": SEALED_SHA, "failed_probe_report_sha256": FAILED_SHA, "failed_reseal_report_sha256": RESEAL_SHA}
    with (out / "report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Direct shared representation probe exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, out, report, persist)
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error)); persist(); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "direct_representation_fidelity_verified")}))


if __name__ == "__main__": main()
