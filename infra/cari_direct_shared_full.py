"""Full-coverage direct ABI replay of a NEW frame-zero shared identity.

The passed twelve-frame probe authorizes only this representation experiment.
No historical target is recovered, no identity quality is established, and no
solver, upstream conversion, alignment, or submission is performed.
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

import cari_direct_shared_probe as probe
from world_reward.data import sha256

STAGE = "world_reward_cari_direct_head_shared_identity_full_representation"
PROBE_SHA = "d22d052a8a631dda57e3b60a0790d4eaf963f808496a806326e3a3e68a14ec67"
PROBE_SCRIPT_SHA = "29db01f3f6a93fdab8e18138997e6fbe642aa7183cb0086bcd98f5091859b1c5"
PROBE_REVISION = "0c0e547db14c6f82717754c2146cbc19adbd3ddf"
FRAMES, CHUNK, BUDGET = 790, 16, 300
SETTINGS = {"seed": 0, "deterministic_algorithms": True, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TF32": False, "jit_optimized_execution": False, "dtype": "float32", "CPU_threads": 4}


def chunks():
    return [slice(start, min(start+CHUNK, FRAMES)) for start in range(0, FRAMES, CHUNK)]


def full_inputs(params):
    probe.shared_inputs(params)  # Exact original790/FP32/zero-expression validation, without mutation.
    result = {key: value.copy() for key, value in params.items()}
    for key in ("mhr_shape", "mhr_scale"): result[key] = np.repeat(params[key][:1], FRAMES, axis=0)
    return result


def require_fields(report, expected):
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Actual source-bound passed twelve-frame direct probe required")


def prerequisite(root, original):
    path = probe.regular(root / "outputs/episode_000000/cari_direct_shared_probe_refined_v1/report.json")
    probe.convert._require_hash(path, PROBE_SHA); prior = json.loads(path.read_text())
    require_fields(prior, {"stage": probe.STAGE, "status": "pass", "phase": "complete",
        "producer_revision": PROBE_REVISION, "script_sha256": PROBE_SCRIPT_SHA, "image_id": probe.IMAGE,
        "input_track": "track_1", "network": "none", "ground_truth_used": False, "hand_labeled_test": False,
        "oracle_modes": [], "new_identity_source": True, "accuracy_verified": False, "adoption_authorized": False,
        "submission_produced": False, "original_conversion_rerun": False, "old_target_read": False,
        "original_native_geometry_recovered": False, "historical_target_recovered": False,
        "strict_reproducibility_verified": False, "full_frame_fit_verified": False, "solver_calls": 0,
        "alignment_calls": 0, "frames": 12, "original_frames": FRAMES, "frame_indices": list(probe.INDICES),
        "identity_selection_rule": "native_frame_zero_protocol_order_not_error", "shared_identity_frame_index": 0,
        "settings": SETTINGS, "actual_direct_head_calls": 12, "actual_native_vertices_only_calls": 1,
        "actual_official_reference_calls": 1, "actual_total_native_head_calls": 13,
        "source_bindings_runtime_verified": True, "direct_representation_fidelity_verified": True,
        "original_native_parameters_unchanged": True, "predictions_frozen_before_replay": True,
        "native_parameter_identities": original["native_parameter_identities"],
        "inference_source_identity": original["inference_source_identity"], "body_assets": original["body_assets"],
        "original_decoder_identity": original["decoder_identity"],
        "native_max_point_mm_bound": probe.NATIVE_MAX_POINT_MM, "reference_mean_mm_bound": 2.,
        "reference_max_point_is_diagnostic_only": True, "reference_precision": "float32",
        "reference_residual_dtype": "float64", "reference_model_chunk": 16,
        "official_converter_sha256": probe.convert.CONVERTER_SHA256,
        "reference_model_sha256": probe.convert.REFERENCE_MODEL_SHA256,
        "sealed_report_sha256": probe.SEALED_SHA, "failed_probe_report_sha256": probe.FAILED_SHA,
        "failed_reseal_report_sha256": probe.RESEAL_SHA})
    probe.convert._require_hash(Path(probe.__file__), PROBE_SCRIPT_SHA)
    helpers = {name: sha256(Path(probe.__file__).with_name(name)) for name in
               ("cari_target_reseal.py", "cari_identity_probe_runtime.py", "cari_converter.py")}
    require_fields(prior, {"runtime_helper_sha256": helpers})
    for key in ("native_vs_direct", "reference_vs_direct"):
        values = prior.get(key, {}); means = np.asarray(values.get("per_frame_mean_mm"), dtype=np.float64)
        maximum = values.get("max_point_mm")
        if (means.shape != (12,) or not np.isfinite(means).all() or np.any(means < 0)
                or type(maximum) is not float or not np.isfinite(maximum) or maximum < means.max()
                or (key == "native_vs_direct" and maximum > probe.NATIVE_MAX_POINT_MM)
                or (key == "reference_vs_direct" and means.max() > 2.)):
            raise ValueError("Passed probe receipt does not satisfy unchanged numeric gates")
    return prior, [(path, PROBE_SHA), (Path(probe.__file__), PROBE_SCRIPT_SHA)]


def freeze(out, target, controls, shared):
    if (target.dtype != np.float32 or target.shape != (FRAMES, probe.VERTICES, 3)
            or controls.dtype != np.float32 or controls.shape != (FRAMES, 204)
            or not np.isfinite(target).all() or not np.isfinite(controls).all()
            or any(row.tobytes() != controls[0, 136:].tobytes() for row in controls[:, 136:])):
        raise ValueError("Every direct790 frame finite FP32 with byte-constant68 scales required")
    for key in ("mhr_shape", "mhr_scale"):
        if any(row.tobytes() != shared[key][0].tobytes() for row in shared[key]):
            raise ValueError("Explicit frame-zero shared identity required")
    if np.any(shared["mhr_face"] != 0): raise ValueError("Zero expressions required")
    tp, pp = out / "target_f32.npy", out / "direct_parameters.npz"
    with tp.open("xb") as file: np.save(file, target, allow_pickle=False)
    tp.chmod(0o444)
    with pp.open("xb") as file:
        np.savez_compressed(file, frame_index=np.arange(FRAMES, dtype=np.int64), pose=controls[:, :136],
            scales=controls[0, 136:], shape=shared["mhr_shape"][0], scale_pca=shared["mhr_scale"][0],
            expression=shared["mhr_face"])
    pp.chmod(0o444)
    saved_target = np.load(tp, mmap_mode="r", allow_pickle=False)
    with np.load(pp, allow_pickle=False) as file: saved = {k: file[k].copy() for k in file.files}
    if (saved_target.flags.writeable or not np.array_equal(saved["pose"], controls[:, :136])
            or not np.array_equal(saved["scales"], controls[0, 136:])
            or not np.array_equal(saved["frame_index"], np.arange(FRAMES))
            or saved["shape"].tobytes() != shared["mhr_shape"][0].tobytes()
            or saved["scale_pca"].tobytes() != shared["mhr_scale"][0].tobytes()
            or saved["expression"].tobytes() != shared["mhr_face"].tobytes()):
        raise ValueError("Full saved predictions changed before independent replay")
    return saved_target, saved, [(tp, sha256(tp)), (pp, sha256(pp))]


def append_residual(report, key, target, recovered, units):
    values = probe.residuals(target, recovered, units=units)
    report[key]["per_frame_mean_mm"].extend(values["per_frame_mean_mm"])
    report[key]["max_point_mm"] = max(report[key]["max_point_mm"], values["max_point_mm"])
    return values


def run(root, out, report, persist):
    original, params, receipt, frozen = probe.sealed_inputs(root); del original
    prior, more = prerequisite(root, receipt); frozen += more + probe.preserved_failures(root)
    assets, vendor, more = probe.source_chain(root, params, receipt); frozen += more
    shared = full_inputs(params)
    identities = {key: probe.convert.canonical_array_identity(value) for key, value in params.items()}
    shared_ids = {key: probe.convert.canonical_array_identity(value) for key, value in shared.items()}
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    layerpath = probe.regular(native / "lib_mhr/mhr_layer.py")
    headpath = probe.regular(Path("/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py"))
    tool = probe.regular(root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py")
    model = probe.regular(root / "weights/mhr/mhr_model.pt")
    for path, digest in ((layerpath, prior["native_layer_sha256"]), (headpath, prior["native_head_sha256"]),
                         (tool, probe.convert.CONVERTER_SHA256), (model, probe.convert.REFERENCE_MODEL_SHA256)):
        probe.convert._require_hash(path, digest); frozen.append((path, digest))
    report.update(native_parameter_identities=identities, shared_native_parameter_identities=shared_ids,
        inference_source_identity=receipt["inference_source_identity"], body_assets=receipt["body_assets"],
        decoder_identity=receipt["decoder_identity"], native_layer_sha256=prior["native_layer_sha256"],
        native_head_sha256=prior["native_head_sha256"]); persist()
    torch = probe.strict_torch()
    if torch.__version__ != prior["torch"] or torch.version.cuda != prior["CUDA"]:
        raise ValueError("Same audited probe Torch/CUDA runtime required")
    actual, _ = probe.convert.validate_native_bundle(torch.load(more[1][0], map_location="cpu", weights_only=False), FRAMES)
    if any(not np.array_equal(actual[key], params[key]) for key in params): raise ValueError("Original bundle parameters differ")
    del actual
    os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    from lib_mhr.mhr_layer import MHRLayer
    if Path(sys.modules[MHRLayer.__module__].__file__).resolve() != layerpath:
        raise ValueError("Original audited MHRLayer source required")
    report.update(phase="direct_head_full_decode", torch=torch.__version__, CUDA=torch.version.cuda); persist()
    def tensors(selection):
        return {key: torch.tensor(value[selection].copy(), device="cuda", dtype=torch.float32) for key, value in shared.items()}
    with torch.no_grad(), torch.jit.optimized_execution(False):
        layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),
            checkpoint_path=assets / "model.ckpt", buffer_path=root / "never_use_unverified_direct_buffer.pt",
            mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
        if layer.decoder_identity() != receipt["decoder_identity"]: raise ValueError("Same checkpoint decoder identity required")
        target = np.empty((FRAMES, probe.VERTICES, 3), np.float32); controls = np.empty((FRAMES, 204), np.float32)
        for selection in chunks():
            inputs = tensors(selection); backend = layer.backend
            context = backend._vertices_context(inputs, detach_fixed=False)
            trans, body_pose, shape, scale = backend._mutable_vertices_inputs(inputs, context)
            if (context.head.enable_hand_model or Path(sys.modules[type(context.head).__module__].__file__).resolve() != headpath):
                raise ValueError("Exact audited full-body head required")
            vertices, direct = context.head.mhr_forward(global_trans=trans*context.flip,
                global_rot=context.global_rot, body_pose_params=body_pose, hand_pose_params=context.hand,
                scale_params=scale, shape_params=shape, expr_params=context.face, return_model_params=True)
            report["actual_direct_head_calls"] += 1
            if not torch.equal(direct[:, 136:], context.head.scale_mean[None, :] + scale @ context.head.scale_comps):
                raise ValueError("Native checkpoint scale expansion differs")
            target[selection] = (vertices*context.flip).cpu().numpy(); controls[selection] = direct.cpu().numpy()
            report["direct_frames_complete"] = selection.stop; persist()
        target, saved, new_files = freeze(out, target, controls, shared); frozen += new_files
        report.update(phase="independent_native_full_replay", predictions_frozen_before_replay=True,
            target_vertices_identity=probe.convert.canonical_array_identity(target),
            direct_model_parameters_identity=probe.convert.canonical_array_identity(controls),
            file_inventory=[{"file": path.name, "sha256": h, "bytes": path.stat().st_size} for path, h in new_files]); persist()
        for selection in chunks():
            recovered = layer.mhr_forward_vertices(tensors(selection)).cpu().numpy()
            report["actual_native_vertices_only_calls"] += 1
            values = append_residual(report, "native_vs_direct", target[selection], recovered, "m"); persist()
            if values["max_point_mm"] > probe.NATIVE_MAX_POINT_MM:
                raise ValueError("Native/direct translation fidelity exceeds fixed .01mm point bound")
        report.update(phase="independent_official_full_replay"); persist()
        spec = importlib.util.spec_from_file_location("world_reward_full_direct_reference", tool)
        converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
        if Path(converter.MHR.run.__code__.co_filename).resolve() != tool:
            raise ValueError("Only pinned official reference forward required")
        reference = converter.MHR(str(model), "cuda", chunk=CHUNK, precision="float32")
        if (reference.mdtype != torch.float32 or reference.dtype != torch.float64
                or reference.chunk != CHUNK or reference.device.type != "cuda"):
            raise ValueError("Exact FP32 official reference with FP64 residuals required")
        identity = torch.tensor(np.concatenate([saved["scales"], saved["shape"]])[None], device="cuda", dtype=torch.float64)
        for selection in chunks():
            vertices_mm, _ = reference.run(torch.tensor(saved["pose"][selection], device="cuda", dtype=torch.float64), identity)
            report["actual_official_reference_calls"] += 1
            values = append_residual(report, "reference_vs_direct", target[selection], vertices_mm.cpu().numpy(), "mm"); persist()
            if max(values["per_frame_mean_mm"]) > 2.:
                raise ValueError("An original frame exceeds unchanged2mm reference representation bound")
    for path, digest in frozen: probe.convert._require_hash(probe.regular(path), digest)
    if (any(probe.convert.canonical_array_identity(v) != identities[k] for k, v in params.items())
            or any(probe.convert.canonical_array_identity(v) != shared_ids[k] for k, v in shared.items())):
        raise ValueError("Original or shared inputs changed")
    probe.source_chain(root, params, receipt)
    if (any(report[key] != len(chunks()) for key in
            ("actual_direct_head_calls", "actual_native_vertices_only_calls", "actual_official_reference_calls"))
            or any(len(report[key]["per_frame_mean_mm"]) != FRAMES for key in ("native_vs_direct", "reference_vs_direct"))):
        raise ValueError("All790frames and fifty16+tail6 calls per pass required")
    report.update(status="pass", phase="complete", source_bindings_runtime_verified=True,
        original_native_parameters_unchanged=True, full_original_frame_coverage_verified=True,
        full_frame_representation_fidelity_verified=True, actual_total_native_head_calls=100)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != probe.IMAGE):
        raise ValueError("Canonical pinned-image Azure offline runtime required")
    out = root / "outputs/episode_000000/cari_direct_shared_full_v1"
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty full representation output required")
    start = time.perf_counter()
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "producer_revision": revision,
        "image_id": image, "script_sha256": sha256(Path(__file__)), "budget_seconds": BUDGET, "settings": SETTINGS,
        "network": "none", "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False,
        "oracle_modes": [], "research_only": True, "adoption_authorized": False, "accuracy_verified": False,
        "submission_produced": False, "original_conversion_rerun": False, "new_identity_source": True,
        "identity_selection_rule": "native_frame_zero_protocol_order_not_error", "shared_identity_frame_index": 0,
        "old_target_read": False, "historical_target_recovered": False, "original_native_geometry_recovered": False,
        "strict_reproducibility_verified": False, "solver_calls": 0, "alignment_calls": 0, "full_frame_fit_verified": False,
        "full_original_frame_coverage_verified": False, "full_frame_representation_fidelity_verified": False,
        "source_bindings_runtime_verified": False, "frames": FRAMES, "chunk_size": CHUNK, "tail_frames": FRAMES%CHUNK,
        "native_max_point_mm_bound": probe.NATIVE_MAX_POINT_MM, "reference_mean_mm_bound": 2.,
        "reference_max_point_is_diagnostic_only": True, "reference_precision": "float32", "reference_residual_dtype": "float64",
        "actual_direct_head_calls": 0, "actual_native_vertices_only_calls": 0, "actual_official_reference_calls": 0,
        "native_vs_direct": {"per_frame_mean_mm": [], "max_point_mm": 0.},
        "reference_vs_direct": {"per_frame_mean_mm": [], "max_point_mm": 0.},
        "direct_probe_report_sha256": PROBE_SHA, "direct_probe_script_sha256": PROBE_SCRIPT_SHA,
        "sealed_report_sha256": probe.SEALED_SHA, "failed_probe_report_sha256": probe.FAILED_SHA,
        "failed_reseal_report_sha256": probe.RESEAL_SHA, "official_converter_sha256": probe.convert.CONVERTER_SHA256,
        "reference_model_sha256": probe.convert.REFERENCE_MODEL_SHA256}
    with (out / "report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Full direct shared representation exceeded300s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, out, report, persist)
        except BaseException as error:
            report.update(error_type=type(error).__name__, error=str(error)); persist(); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "full_frame_representation_fidelity_verified")}))


if __name__ == "__main__": main()
