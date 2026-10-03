"""Decode unchanged refined96 predictions into native Track1 parameters.

Direct204 controls are obtained from the original MHR head, not a layout guess
or a mesh fit. The aligned fixed object frame is retained. This engineering
export is NOT a full501 submission, accuracy evaluation or CARI4D win claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sys
import time

import numpy as np
import body_smoke as body
import cari96_inputs as inputs
import cari96_prepare as geometry
import cari96_refine as lineage
from cari_converter import validate_native_bundle
from world_reward.contracts import Reconstruction, require_rigid_transforms
from world_reward.submission import Track1Episode

BASE = "validation/cari96_export_v1"
STAGE = "public_cari96_refined_direct_native_export"
FRAMES, CHUNK, BUDGET = 96, 16, 180
IMAGE = geometry.IMAGE
CALLS = ("native_geometry", "native_direct", "native_replay", "reference")
TRAJECTORY_KEYS = {"pose", "scales", "shape", "expression", "object_rotation", "object_translation",
                   "object_scale", "object_vertices", "object_faces", "camera_K", "frame_index"}


def array(value, shape, dtype):
    if (type(value) is not np.ndarray or value.shape != shape or value.dtype != np.dtype(dtype)
            or not np.isfinite(value).all()): raise ValueError("Exact complete finite array required")
    return value


def same_arrays(left, right):
    return (set(left) == set(right) and all(isinstance(right[k], np.ndarray) and not np.ma.isMaskedArray(right[k]) and left[k].shape == right[k].shape
        and left[k].dtype == right[k].dtype and left[k].tobytes() == right[k].tobytes() for k in left))


def validate_refined(source, refined, mesh):
    metadata = lineage.validate_result(source, refined)
    params, poses = validate_native_bundle(refined, FRAMES)
    lineage.validate_source_bundle(refined, mesh)
    for key, dimension in inputs.PARAMETER_DIMS.items(): array(params[key], (FRAMES, dimension), np.float32)
    array(poses, (FRAMES, 4, 4), np.float32)
    if not same_arrays({"contact": source["pr"]["contact_logits"]}, {"contact": refined["pr"]["contact_logits"]}):
        raise ValueError("Refined native contact predictions changed")
    return params, poses, metadata


def require_refined_report(report, metadata, pins, mesh):
    lineage.validate_refinement_report(report)
    if (report.get("metadata") != metadata
            or report.get("bundle_sha256") != pins["refined_files"]["refined.pth"]["sha256"]
            or report.get("bundle_bytes") != pins["refined_files"]["refined.pth"]["bytes"]
            or report.get("object_mesh_sha256") != inputs.sha256(mesh)
            or report.get("output_files") != {"refined.pth": pins["refined_files"]["refined.pth"]}):
        raise ValueError("Refined metadata/mesh/output chain differs")


def object_roundtrip(vertices, faces, poses):
    array(vertices, (len(vertices), 3), np.float32)
    array(faces, (len(faces), 3), np.int64)
    require_rigid_transforms(poses, FRAMES)
    if not len(vertices) or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Complete nonempty native object mesh required")
    maximum = 0.
    for pose in poses:
        homogeneous = np.c_[vertices.astype(np.float64), np.ones(len(vertices))] @ pose.astype(np.float64).T
        packed = vertices.astype(np.float64) @ pose[:3, :3].astype(np.float64).T + pose[:3, 3]
        if not np.isfinite(packed).all() or np.any(packed[:, 2] <= 0): raise ValueError("All object vertices must remain in front of camera")
        maximum = max(maximum, float(np.linalg.norm(homogeneous[:, :3] - packed, axis=1).max()))
    if maximum > 1e-5: raise ValueError("Fixed aligned mesh/pose schema roundtrip failed")
    return dict(frames=FRAMES, vertices_per_frame=len(vertices), max_point_error_m=maximum,
                aligned_local_frame_retained=True, object_scale=1., additional_frame_transform=False)


def trajectory(controls, params, poses, vertices, faces, K):
    array(controls, (FRAMES, 204), np.float32)
    array(K, (3, 3), np.float64)
    if not np.array_equal(K, [[1920., 0., 768.], [0., 1920., 576.], [0., 0., 1.]]):
        raise ValueError("Original inferred camera must remain unchanged")
    if any(row.tobytes() != controls[0, 136:].tobytes() for row in controls[:, 136:]):
        raise ValueError("Refined native68 scales must be byte-constant; no identity re-selection")
    data = dict(pose=controls[:, :136].copy(), scales=controls[0, 136:].copy(), shape=params["mhr_shape"][0].copy(),
        expression=np.zeros(72, np.float32), object_rotation=poses[:, :3, :3].copy(), object_translation=poses[:, :3, 3].copy(),
        object_scale=np.asarray(1., np.float32), object_vertices=vertices.copy(), object_faces=faces.copy(),
        camera_K=K.copy(), frame_index=np.arange(FRAMES, dtype=np.int64))
    rec = Reconstruction(*(data[k] for k in ("pose", "scales", "shape", "object_rotation", "object_translation", "object_scale")))
    Track1Episode(rec, data["object_vertices"], data["object_faces"], data["expression"],
        dict(input_track="track_1", ground_truth_used=False, hand_labeled_test=False, oracle_modes=[]), FRAMES).validate()
    return data


def freeze(out, data, params, target, joints, keypoints, faces):
    native = {k: v.copy() for k, v in params.items()}
    native.update(mhr_joints=joints.copy(), mhr_keypoints=keypoints.copy(), human_faces=faces.copy(), frame_index=data["frame_index"].copy())
    for name, values in (("trajectory.npz", data), ("native_parameters.npz", native)):
        with (out / name).open("xb") as stream: np.savez_compressed(stream, **values)
        (out / name).chmod(0o444)
        with np.load(out / name, allow_pickle=False) as stored:
            if not same_arrays(values, {k: stored[k] for k in stored.files}): raise ValueError("Frozen parameter bytes changed")
    with (out / "target.npy").open("xb") as stream: np.save(stream, target, allow_pickle=False)
    (out / "target.npy").chmod(0o444)
    reread = np.load(out / "target.npy", mmap_mode="r", allow_pickle=False)
    if reread.flags.writeable or not same_arrays({"v": target}, {"v": reread}): raise ValueError("Frozen native geometry changed")
    return {name: lineage.identity(out / name) for name in ("trajectory.npz", "native_parameters.npz", "target.npy")}


def call(report, name, function, persist):
    report[name + "_attempts"] += 1; persist()
    value = function()
    report[name + "_returns"] += 1
    return value


def validated(report, name, persist):
    report[name + "_validated"] += 1; persist()


def source_helpers(code, producer):
    rows = producer.get("source_helpers")
    if type(rows) is not dict or not rows: raise ValueError("Refined producer source inventory required")
    for name, row in rows.items():
        path = code / name
        if not path.is_relative_to(code / "infra") or lineage.identity(path) != row:
            raise ValueError("Unchanged refined helper source required")
    if rows.get("infra/cari96_refine.py", {}).get("sha256") != producer["script_sha256"]:
        raise ValueError("Actual refined driver source differs")


def run(root, out, code, report, persist):
    paths = [code / ("configs/cari96_" + name + "_pins.json") for name in ("refined", "forward", "prepare", "input")]
    config_ids = {p: lineage.identity(p) for p in paths}
    pins, forward_pins, prepare_pins, input_pins = [json.loads(p.read_text()) for p in paths]
    refined_report, frozen = lineage.pinned_report(root, lineage.BASE, pins, "refined")
    if set(pins["refined_files"]) != {"report.json", "refined.pth"}: raise ValueError("Only complete new refined96 output is accepted")
    forward, prepared, chain = lineage.source_inputs(root, forward_pins, prepare_pins, input_pins); frozen.update(chain)
    if (refined_report.get("forward_report_sha256") != forward_pins["forward"]["sha256"]
            or refined_report.get("prepare_report_sha256") != prepare_pins["prepare"]["sha256"]
            or refined_report.get("source_bundle_sha256") != forward_pins["forward_files"]["coconet.pth"]["sha256"]):
        raise ValueError("Refined/forward/prepared chain differs")
    source_helpers(code, refined_report)
    assets, hashes = body._body_assets(root); source_id = body._source_identity(root)
    if any(row.get("body_assets") != hashes or row.get("inference_source_identity") != source_id for row in (refined_report, forward, prepared)):
        raise ValueError("Same original native decoder assets/source required")
    vendor = root / "vendor/video_to_data"; body._pinned_checkout(vendor, body.UPSTREAM_REVISION)
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"; model = root / "weights/mhr/mhr_model.pt"
    for path, digest in ((native / "lib_mhr/mhr_layer.py", geometry.LAYER_SHA), (tool, geometry.CONVERTER_SHA),
                         (model, geometry.REFERENCE_SHA), (native / lineage.contract.OPTIMIZER_RELATIVE_PATH, lineage.contract.OPTIMIZER_SHA256)):
        geometry.binding(path, digest); frozen[path] = lineage.identity(path, immutable=False)
    for name, row in hashes.items(): frozen[assets / name] = row
    asset_receipt = root / "results/cari-refinement-assets.json"
    lineage.contract.require_asset_receipt(json.loads(inputs.regular(asset_receipt).read_text()))
    frozen[asset_receipt] = lineage.identity(asset_receipt, immutable=False)
    for name, expected in lineage.contract.REFINEMENT_ASSETS.items():
        path = root / "weights/cari4d/refinement" / name; row = lineage.identity(path, immutable=False)
        if row != {k: expected[k] for k in ("sha256", "bytes")}: raise ValueError("Original refinement asset changed")
        frozen[path] = row
    helpers = {str(p.relative_to(code)): lineage.identity(p) for p in
        (Path(__file__), code / "infra/run_cari96_export.sh", Path(geometry.__file__), Path(lineage.__file__),
         Path(lineage.contract.__file__), Path(inputs.__file__), Path(body.__file__),
         Path(sys.modules[validate_native_bundle.__module__].__file__), Path(sys.modules[Track1Episode.__module__].__file__),
         Path(sys.modules[Reconstruction.__module__].__file__))}
    report.update(phase="native_decode", refined_pins=config_ids[paths[0]], source_helpers=helpers, body_assets=hashes,
        inference_source_identity=source_id, decoder_identity=prepared["decoder_identity"],
        refined_report_sha256=pins["refined"]["sha256"], refined_bundle_sha256=pins["refined_files"]["refined.pth"]["sha256"]); persist()
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise ValueError("Fresh native process required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__) != "2.5.1+cu124" or torch.version.cuda != "12.4": raise ValueError("Pinned native H100 runtime required")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    report["runtime"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__, python=platform.python_version(),
        deterministic_algorithms=True, warn_only=False, jit_optimized_execution=False, tf32=False, seed=0, chunk=CHUNK)
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0")
    from lib_mhr.mhr_layer import MHRLayer
    from learning.training import mhr_opt_refineout as optimizer
    if Path(optimizer.__file__).resolve() != native / lineage.contract.OPTIMIZER_RELATIVE_PATH or Path(sys.modules[MHRLayer.__module__].__file__).resolve() != native / "lib_mhr/mhr_layer.py":
        raise ValueError("Actual unchanged native decoder/mesh loader source required")
    source = torch.load(root / lineage.FORWARD / "coconet.pth", map_location="cpu", weights_only=False)
    refined = torch.load(root / lineage.BASE / "refined.pth", map_location="cpu", weights_only=False)
    mesh = root / lineage.PREPARE / "inputs/export/episode_000015/object_mesh/output_aligned.glb"
    params, poses, metadata = validate_refined(source, refined, mesh)
    require_refined_report(refined_report, metadata, pins, mesh)
    fingerprints = [lineage.fingerprint(value) for value in (source, refined)]
    vertices, object_faces = optimizer._load_object_vertices(mesh)
    report["object_roundtrip"] = object_roundtrip(vertices, object_faces, poses)
    K = np.asarray(json.loads((mesh.parent.parent / "wild_export.json").read_text())["intrinsics"], np.float64)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"), checkpoint_path=assets / "model.ckpt",
        buffer_path=out / "never_use_unverified_buffers.pt", mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
    if layer.decoder_identity() != prepared["decoder_identity"]: raise ValueError("Refined native decoder identity differs")
    target = np.empty((FRAMES, 18439, 3), np.float32); joints = np.empty((FRAMES, 127, 3), np.float32)
    kp = np.empty((FRAMES, 70, 3), np.float32); controls = np.empty((FRAMES, 204), np.float32)
    tensors = lambda selection: {k: torch.tensor(v[selection].copy(), dtype=torch.float32, device="cuda") for k, v in params.items()}
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for start in range(0, FRAMES, CHUNK):
            selection = slice(start, start + CHUNK); t = tensors(selection)
            decoded = call(report, "native_geometry", lambda: layer.mhr_forward(t), persist)
            target[selection] = geometry.checked_geometry(decoded.vertices.cpu().numpy(), CHUNK, 18439)
            joints[selection] = geometry.checked_geometry(decoded.joints.cpu().numpy(), CHUNK, 127)
            kp[selection] = geometry.checked_geometry(decoded.keypoints.cpu().numpy(), CHUNK, 70)
            human_faces = decoded.faces.cpu().numpy()
            if (human_faces.dtype != np.int64 or human_faces.shape != (36874, 3)
                    or np.any(human_faces < 0) or np.any(human_faces >= 18439)
                    or hashlib.sha256(human_faces.astype("<i4").tobytes()).hexdigest() != geometry.FACE_SHA):
                raise ValueError("Exact original human topology required")
            validated(report, "native_geometry", persist)
            context = layer.backend._vertices_context(t, detach_fixed=False)
            trans, pose, shape, scale = layer.backend._mutable_vertices_inputs(t, context)
            if context.head.enable_hand_model: raise ValueError("Original body head required")
            dv, dc = call(report, "native_direct", lambda: context.head.mhr_forward(global_trans=trans * context.flip,
                global_rot=context.global_rot, body_pose_params=pose, hand_pose_params=context.hand, scale_params=scale,
                shape_params=shape, expr_params=context.face, return_model_params=True), persist)
            torch.cuda.synchronize(); controls[selection] = array(dc.cpu().numpy(), (CHUNK, 204), np.float32)
            if (geometry.residuals(target[selection], (dv * context.flip).cpu().numpy(), "m")["max_point_mm"] > .01
                    or not torch.equal(dc[:, 136:], context.head.scale_mean[None, :] + scale @ context.head.scale_comps)):
                raise ValueError("Direct native controls fail .01mm geometry/PCA fidelity")
            validated(report, "native_direct", persist)
        data = trajectory(controls, params, poses, vertices, object_faces, K)
        frozen_outputs = freeze(out, data, params, target, joints, kp, human_faces)
        shutil.copyfile(mesh, out / "object_aligned.glb"); (out / "object_aligned.glb").chmod(0o444)
        frozen_outputs["object_aligned.glb"] = lineage.identity(out / "object_aligned.glb")
        if frozen_outputs["object_aligned.glb"] != lineage.identity(mesh): raise ValueError("Fixed aligned GLB bytes changed")
        report.update(phase="frozen_native_replay", predictions_frozen_before_replays=True, output_files=frozen_outputs); persist()
        with np.load(out / "trajectory.npz", allow_pickle=False) as stream: stored = {k: stream[k] for k in stream.files}
        with np.load(out / "native_parameters.npz", allow_pickle=False) as stream:
            native_artifact = {k: stream[k] for k in stream.files}
            stored_native = {k: native_artifact[k] for k in inputs.PARAMETER_DIMS}
        if not same_arrays(params, stored_native): raise ValueError("Stored refined native blocks differ")
        saved = np.load(out / "target.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, FRAMES, CHUNK):
            selection = slice(start, start + CHUNK)
            recovered = call(report, "native_replay", lambda: layer.mhr_forward({k: torch.tensor(v[selection].copy(),
                dtype=torch.float32, device="cuda") for k, v in stored_native.items()}), persist)
            torch.cuda.synchronize()
            for expected, actual in ((saved[selection], recovered.vertices),
                    (native_artifact["mhr_joints"][selection], recovered.joints),
                    (native_artifact["mhr_keypoints"][selection], recovered.keypoints)):
                if geometry.residuals(expected, actual.cpu().numpy(), "m")["max_point_mm"] > .01:
                    raise ValueError("Stored native V/J/KP replay fails .01mm point fidelity")
            if not np.array_equal(recovered.faces.cpu().numpy(), native_artifact["human_faces"]):
                raise ValueError("Stored native replay topology changed")
            validated(report, "native_replay", persist)
        spec = importlib.util.spec_from_file_location("cari96_official_reference", tool)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        reference = module.MHR(str(model), "cuda", chunk=CHUNK, precision="float32")
        if (reference.mdtype != torch.float32 or reference.dtype != torch.float64 or reference.chunk != CHUNK
                or reference.device.type != "cuda" or Path(module.MHR.run.__code__.co_filename).resolve() != tool):
            raise ValueError("Original FP32 model/FP64 residual reference required")
        z = torch.tensor(np.r_[stored["scales"], stored["shape"]][None], dtype=torch.float64, device="cuda")
        report["phase"] = "official_reference_replay"; persist(); errors = []; maximum = 0.
        for start in range(0, FRAMES, CHUNK):
            selection = slice(start, start + CHUNK)
            v, _ = call(report, "reference", lambda: reference.run(torch.tensor(stored["pose"][selection], dtype=torch.float64, device="cuda"), z), persist)
            torch.cuda.synchronize(); values = geometry.residuals(saved[selection], v.cpu().numpy(), "mm")
            maximum = max(maximum, values["max_point_mm"])
            if max(values["per_frame_mean_mm"]) > 2.: raise ValueError("A stored frame fails unchanged2mm official fidelity")
            errors.extend(values["per_frame_mean_mm"]); validated(report, "reference", persist)
    if ([lineage.fingerprint(value) for value in (source, refined)] != fingerprints
            or {p: lineage.identity(p, immutable=False) for p in frozen} != frozen
            or {p: lineage.identity(p) for p in config_ids} != config_ids
            or {k: lineage.identity(code / k) for k in helpers} != helpers
            or body._source_identity(root) != source_id or body._body_assets(root)[1] != hashes
            or {k: lineage.identity(out / k) for k in frozen_outputs} != frozen_outputs):
        raise ValueError("Frozen native source/prediction/assets/exports changed")
    source_helpers(code, refined_report)
    if any(report[name + suffix] != 6 for name in CALLS for suffix in ("_attempts", "_returns", "_validated")):
        raise ValueError("Complete six-chunk native/direct/stored/reference coverage required")
    report.update(status="pass", phase="complete", frames=FRAMES, source_frames=501, original_frame_indices=list(range(FRAMES)),
        unchanged_refined_predictions_verified=True, raw_masks_contacts_object_pose_unchanged=True,
        source_inputs_assets_rehashed=True, frozen_outputs_rehashed_after_reference=True,
        reference_per_frame_mean_mm=errors, reference_max_point_mm=maximum, native_Track1Episode_schema_verified=True,
        aligned_object_mesh_sha256=inputs.sha256(mesh), full501_submission_eligible=False)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision, out = os.environ["WR_CODE_REVISION"], root / BASE
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or os.geteuid() != 1000
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.environ["WR_IMAGE_ID"] != IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or out.resolve() != out.absolute()
            or not out.is_dir() or any(out.iterdir())): raise ValueError("Fresh offline exclusive native96 output required")
    report = dict(stage=STAGE, status="fail", phase="public_integrity", producer_revision=revision,
        script_sha256=inputs.sha256(Path(__file__)), image_id=IMAGE, input_track="track_1", network="none",
        ground_truth_used=False, private_truth_read=False, hand_labeled_test=False, oracle_modes=[],
        learned_inference_calls=0, optimizer_calls=0, converter_LM_calls=0, quality_verified=False,
        adoption_authorized=False, submission_produced=False, budget_seconds=BUDGET)
    report.update({name + suffix: 0 for name in CALLS for suffix in ("_attempts", "_returns", "_validated")})
    receipt, started = out / "report.json", time.perf_counter()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started
            stream.seek(0); json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Native96 direct export exceeded180s")
        old = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, out, code, report, persist)
        except BaseException as error: report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally: signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o444)


if __name__ == "__main__": main()
