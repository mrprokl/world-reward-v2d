"""Full-original-timeline direct native export of unchanged refined predictions.

The original MHR head supplies direct204 controls; the official MHR consumer
replays them without fitting. Identity, inferred camera, aligned object mesh
and native object poses are retained. This is not Parquet assembly, scoring,
license clearance or evidence of a victory over CARI4D.
"""
from __future__ import annotations

from dataclasses import asdict
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
import cari_clip_inputs as public
import cari_full_refine as lineage
import cari_shared_prepare as native_contract
import cari96_prepare as geometry
from cari_converter import validate_native_bundle
from world_reward.contracts import Reconstruction, require_rigid_transforms
from world_reward.data import sha256
from world_reward.shared_identity import NATIVE_PARAMETER_DIMS, validate_native_parameters
from world_reward.submission import Track1Episode
from world_reward.timeline import finite_chunks

STAGE = "world_reward_native_cari_shared_full_video_direct_export"
BUDGET, CHUNK = 600, 16
IMAGE = geometry.IMAGE
CALLS = native_contract.CALLS
OUTPUTS = {"trajectory.npz", "native_parameters.npz", "target.npy", "object_aligned.glb"}
TRAJECTORY_KEYS = {"pose", "scales", "shape", "expression", "object_rotation", "object_translation",
                   "object_scale", "object_vertices", "object_faces", "camera_K", "frame_index"}


def output_relative(episode):
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Explicit Track1 episode integer in 0..29 required")
    return f"outputs/episode_{episode:06d}/cari_shared_export_v1"


def parser():
    result = native_contract.parser()
    result.description = __doc__
    return result


def same_arrays(left, right):
    return (set(left) == set(right) and all(type(right[key]) is np.ndarray
        and left[key].shape == right[key].shape and left[key].dtype == right[key].dtype
        and left[key].tobytes(order="C") == right[key].tobytes(order="C") for key in left))


def validate_refined(source, refined, mesh, count):
    metadata = lineage.validate_result(source, refined, count)
    lineage.validate_source_bundle(source, mesh, count)
    params, poses = validate_native_bundle(refined, count)
    validate_native_parameters(params, count, require_shared_identity=True)
    native_contract.checked_array(poses, (count, 4, 4))
    if not same_arrays({"contact": source["pr"]["contact_logits"]}, {"contact": refined["pr"]["contact_logits"]}):
        raise ValueError("Original native contact predictions changed")
    return params, poses, metadata


def object_roundtrip(vertices, faces, poses, count):
    native_contract.checked_array(vertices, (len(vertices), 3))
    native_contract.checked_array(faces, (len(faces), 3), np.int64)
    require_rigid_transforms(poses, count)
    if not len(vertices) or not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Complete nonempty original aligned object mesh required")
    maximum = 0.
    homogeneous_vertices = np.c_[vertices.astype(np.float64), np.ones(len(vertices))]
    for pose in poses:
        homogeneous = homogeneous_vertices @ pose.astype(np.float64).T
        packed = vertices.astype(np.float64) @ pose[:3, :3].astype(np.float64).T + pose[:3, 3]
        if not np.isfinite(packed).all() or np.any(packed[:, 2] <= 0):
            raise ValueError("Every original object vertex must remain in front of camera")
        maximum = max(maximum, float(np.linalg.norm(homogeneous[:, :3] - packed, axis=1).max()))
    if maximum > 1e-5:
        raise ValueError("Aligned local object frame/pose schema roundtrip failed")
    return dict(frames=count, vertices_per_frame=len(vertices), max_point_error_m=maximum,
        aligned_local_frame_retained=True, object_scale=1., additional_frame_transform=False)


def trajectory(controls, params, poses, vertices, faces, K, spec):
    count = spec.total_frames
    validate_native_parameters(params, count, require_shared_identity=True)
    native_contract.checked_array(controls, (count, 204))
    native_contract.checked_array(K, (3, 3), np.float64)
    if not np.array_equal(K, public.inferred_camera(spec)):
        raise ValueError("Original inferred RGB-size camera must remain unchanged")
    if any(row.tobytes() != controls[0, 136:].tobytes() for row in controls[:, 136:]):
        raise ValueError("Expanded native68 scale controls must stay byte-constant")
    data = dict(pose=controls[:, :136].copy(), scales=controls[0, 136:].copy(), shape=params["mhr_shape"][0].copy(),
        expression=np.zeros(72, np.float32), object_rotation=poses[:, :3, :3].copy(), object_translation=poses[:, :3, 3].copy(),
        object_scale=np.asarray(1., np.float32), object_vertices=vertices.copy(), object_faces=faces.copy(),
        camera_K=K.copy(), frame_index=np.arange(count, dtype=np.int64))
    rec = Reconstruction(*(data[key] for key in ("pose", "scales", "shape", "object_rotation", "object_translation", "object_scale")))
    Track1Episode(rec, data["object_vertices"], data["object_faces"], data["expression"],
        dict(input_track="track_1", ground_truth_used=False, hand_labeled_test=False, oracle_modes=[]), count).validate()
    return data


def freeze(out, data, params, target, joints, keypoints, faces, mesh):
    native = {key: value.copy() for key, value in params.items()}
    native.update(mhr_joints=joints.copy(), mhr_keypoints=keypoints.copy(), human_faces=faces.copy(),
                  frame_index=data["frame_index"].copy())
    for name, values in (("trajectory.npz", data), ("native_parameters.npz", native)):
        with (out / name).open("xb") as stream:
            np.savez_compressed(stream, **values)
        (out / name).chmod(0o444)
        with np.load(out / name, allow_pickle=False) as stream:
            if not same_arrays(values, {key: stream[key] for key in stream.files}):
                raise ValueError("Frozen native parameter bytes changed")
    with (out / "target.npy").open("xb") as stream:
        np.save(stream, target, allow_pickle=False)
    (out / "target.npy").chmod(0o444)
    saved = np.load(out / "target.npy", mmap_mode="r", allow_pickle=False)
    if (saved.flags.writeable or saved.dtype != target.dtype or saved.shape != target.shape
            or saved.tobytes(order="C") != target.tobytes(order="C")):
        raise ValueError("Frozen native target geometry bytes changed")
    with mesh.open("rb") as source, (out / "object_aligned.glb").open("xb") as destination:
        shutil.copyfileobj(source, destination)
    (out / "object_aligned.glb").chmod(0o444)
    if public.identity(out / "object_aligned.glb") != public.identity(mesh):
        raise ValueError("Fixed original aligned GLB bytes changed")
    with np.load(out / "trajectory.npz", allow_pickle=False) as stream:
        stored = {key: stream[key] for key in stream.files}
    with np.load(out / "native_parameters.npz", allow_pickle=False) as stream:
        stored_native = {key: stream[key] for key in stream.files}
    frozen = {name: public.identity(out / name) for name in sorted(OUTPUTS)}
    return stored, stored_native, saved, frozen


def export_geometry(params, poses, vertices, object_faces, K, spec, out, mesh,
                    native_decode, native_direct, reference_decode, report, persist,
                    *, vertex_count=18439, joint_count=127, keypoint_count=70,
                    face_count=36874, face_sha=geometry.FACE_SHA):
    """Testable four-route execution over every original frame, including tail.

    Runtime binds unchanged native callbacks. Test-only smaller topology is not
    exposed by the command-line interface. No identity selection or fit occurs.
    """
    if not out.is_dir() or any(path.name != "report.json" for path in out.iterdir()):
        raise ValueError("Fresh native export output required before any decode")
    count = spec.total_frames
    validate_native_parameters(params, count, require_shared_identity=True)
    native_contract.checked_array(poses, (count, 4, 4))
    chunks = finite_chunks(count, CHUNK)
    before = native_contract.array_bytes(params)
    fixed = native_contract.array_bytes({"poses": poses, "vertices": vertices, "faces": object_faces, "K": K})
    report["object_roundtrip"] = object_roundtrip(vertices, object_faces, poses, count)
    target = np.empty((count, vertex_count, 3), np.float32)
    joints = np.empty((count, joint_count, 3), np.float32)
    keypoints = np.empty((count, keypoint_count, 3), np.float32)
    controls = np.empty((count, 204), np.float32)
    report.update(phase="native_refined_decode", original_frame_indices=list(range(count)),
        chunk_counts=[selection.stop - selection.start for selection in chunks], identity_reselection_performed=False)
    persist()
    direct_max = 0.
    owned = lambda selection: {key: value[selection].copy() for key, value in params.items()}
    faces = None
    for selection in chunks:
        actual = selection.stop - selection.start
        decoded = native_contract.call(report, "native_geometry", lambda: native_contract.owned_call(native_decode, owned(selection)), persist)
        target[selection] = geometry.checked_geometry(decoded["vertices"], actual, vertex_count)
        joints[selection] = geometry.checked_geometry(decoded["joints"], actual, joint_count)
        keypoints[selection] = geometry.checked_geometry(decoded["keypoints"], actual, keypoint_count)
        current_faces = native_contract.checked_faces(decoded["faces"], vertices=vertex_count, count=face_count, digest=face_sha)
        if faces is not None and not np.array_equal(current_faces, faces):
            raise ValueError("Native human topology changed across original frames")
        faces = current_faces.copy()
        native_contract.validated(report, "native_geometry", persist)
        direct, dc, pca = native_contract.call(report, "native_direct", lambda: native_contract.owned_call(native_direct, owned(selection)), persist)
        native_contract.checked_array(dc, (actual, 204)); native_contract.checked_array(pca, (actual, 68))
        direct = geometry.checked_geometry(direct, actual, vertex_count)
        error = geometry.residuals(target[selection], direct, "m")["max_point_mm"]
        direct_max = max(direct_max, error)
        if error > .01 or dc[:, 136:].tobytes() != pca.tobytes():
            raise ValueError("Native direct controls fail unchanged .01mm geometry/PCA fidelity")
        controls[selection] = dc
        native_contract.validated(report, "native_direct", persist)
    data = trajectory(controls, params, poses, vertices, object_faces, K, spec)
    stored, stored_native, saved, frozen = freeze(out, data, params, target, joints, keypoints, faces, mesh)
    saved_params = {key: stored_native[key] for key in NATIVE_PARAMETER_DIMS}
    if not same_arrays(params, saved_params):
        raise ValueError("Stored refined native blocks changed")
    report.update(phase="frozen_native_replay", predictions_frozen_before_replays=True, output_files=frozen)
    persist()
    replay_max = 0.
    for selection in chunks:
        actual = selection.stop - selection.start
        decoded = native_contract.call(report, "native_replay", lambda: native_contract.owned_call(native_decode,
            {key: value[selection].copy() for key, value in saved_params.items()}), persist)
        for name, expected, dimension in (("vertices", saved[selection], vertex_count),
                ("joints", stored_native["mhr_joints"][selection], joint_count),
                ("keypoints", stored_native["mhr_keypoints"][selection], keypoint_count)):
            value = geometry.checked_geometry(decoded[name], actual, dimension)
            error = geometry.residuals(expected, value, "m")["max_point_mm"]
            replay_max = max(replay_max, error)
            if error > .01:
                raise ValueError("Stored native V/J/KP replay fails unchanged .01mm fidelity")
        if not np.array_equal(native_contract.checked_faces(decoded["faces"], vertices=vertex_count,
                count=face_count, digest=face_sha), stored_native["human_faces"]):
            raise ValueError("Stored native topology changed")
        native_contract.validated(report, "native_replay", persist)
    report.update(phase="official_reference_replay", reference_settings=dict(precision="float32", residual_dtype="float64",
        chunk=CHUNK, device="cuda", mean_point_gate_mm=2., native_max_point_gate_mm=.01))
    persist()
    errors = []; maximum = 0.
    for selection in chunks:
        owned_direct = {key: stored[key].copy() for key in ("pose", "scales", "shape", "frame_index")}
        recovered = native_contract.call(report, "reference", lambda: native_contract.owned_call(
            lambda values: reference_decode(values, selection), owned_direct), persist)
        if type(recovered) is not np.ndarray or recovered.dtype not in (np.float32, np.float64):
            raise ValueError("Exact unmasked official FP32/FP64 geometry required")
        values = geometry.residuals(saved[selection], recovered, "mm")
        if max(values["per_frame_mean_mm"]) > 2.:
            raise ValueError("An original stored frame exceeds unchanged2mm official fidelity")
        errors.extend(values["per_frame_mean_mm"]); maximum = max(maximum, values["max_point_mm"])
        native_contract.validated(report, "reference", persist)
    if (native_contract.array_bytes(params) != before
            or native_contract.array_bytes({"poses": poses, "vertices": vertices, "faces": object_faces, "K": K}) != fixed
            or {path.name for path in out.iterdir() if path.name != "report.json"} != OUTPUTS
            or any((out / name).stat().st_mode & 0o222 for name in frozen)
            or {name: public.identity(out / name) for name in frozen} != frozen):
        raise ValueError("Original refined parameters or frozen native exports changed")
    if any(report[name + suffix] != len(chunks) for name in CALLS for suffix in ("_attempts", "_returns", "_validated")):
        raise ValueError("Every actual full-timeline native/direct/saved/reference chunk required")
    report.update(phase="geometry_routes_complete", frames=count, unchanged_refined_predictions_verified=True,
        reference_per_frame_mean_mm=errors, reference_max_point_mm=maximum, native_direct_max_point_mm=direct_max,
        native_replay_max_point_mm=replay_max, stored_native_replay_verified=True,
        native_Track1Episode_schema_verified=True, original_frame_coverage_verified=True,
        frozen_outputs_rehashed_after_reference=True, aligned_object_mesh_sha256=sha256(mesh))
    return frozen


def source_helpers(code):
    names = ("infra/cari_full_export.py", "infra/run_cari_full_export.sh", "infra/cari_full_refine.py",
        "infra/cari_shared_prepare.py", "infra/run_cari_shared_prepare.sh", "infra/cari_clip_inputs.py",
        "infra/cari96_prepare.py", "infra/cari96_inputs.py", "infra/cari_refine.py", "infra/cari_converter.py", "infra/body_smoke.py",
        "src/world_reward/shared_identity.py", "src/world_reward/timeline.py", "src/world_reward/data.py",
        "src/world_reward/contracts.py", "src/world_reward/submission.py")
    return {name: lineage.identity(code / name) for name in names}


def validate_export_report(report, spec):
    """Full-N consumer gate; native execution is not accuracy or eligibility."""
    if type(spec) is not public.PublicClipSpec:
        raise ValueError("Explicit original full public clip spec required")
    chunks = finite_chunks(spec.total_frames, CHUNK)
    expected = dict(stage=STAGE, status="pass", phase="complete", episode_index=spec.episode_index,
        clip_spec=asdict(spec), image_id=IMAGE, frames=spec.total_frames,
        original_frame_indices=list(range(spec.total_frames)), chunk_counts=[s.stop - s.start for s in chunks],
        input_track="track_1", ground_truth_used=False, ground_truth_read=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], network="none", budget_seconds=BUDGET,
        learned_inference_calls=0, optimizer_calls=0, converter_LM_calls=0, identity_reselection_performed=False,
        quality_verified=False, adoption_authorized=False, submission_produced=False, submission_eligible=False,
        final_Parquet_produced=False, predictions_frozen_before_replays=True,
        unchanged_refined_predictions_verified=True, stored_native_replay_verified=True,
        native_Track1Episode_schema_verified=True, original_frame_coverage_verified=True,
        frozen_outputs_rehashed_after_reference=True, source_inputs_assets_rehashed=True, source_helpers_rehashed=True,
        raw_masks_contacts_object_pose_unchanged=True, full_original_native_export_verified=True)
    expected.update({name + suffix: len(chunks) for name in CALLS for suffix in ("_attempts", "_returns", "_validated")})
    if type(report) is not dict or any(type(report.get(key)) is not type(value) or report[key] != value for key, value in expected.items()):
        raise ValueError("Complete full original native export proof required")
    errors = report.get("reference_per_frame_mean_mm")
    if (type(errors) is not list or len(errors) != spec.total_frames
            or any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 2. for value in errors)):
        raise ValueError("Every original frame must pass unchanged2mm official fidelity")
    for key in ("native_direct_max_point_mm", "native_replay_max_point_mm"):
        value = report.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= .01:
            raise ValueError("Unchanged .01mm native maximum-point fidelity required")
    files = report.get("output_files")
    if type(files) is not dict or set(files) != OUTPUTS:
        raise ValueError("Exact four frozen full-native artifacts required")
    for row in files.values():
        public._receipt(row)
    for key in ("refined_report_sha256", "refined_bundle_sha256", "aligned_object_mesh_sha256"):
        if type(report.get(key)) is not str or not re.fullmatch(r"[0-9a-f]{64}", report[key]):
            raise ValueError("Complete original refined source/output SHA chain required")


def run(root, out, code, episode, report, persist):
    pins_path = code / f"configs/cari_clip_{episode:06d}_shared_refined_pins.json"
    pin_id = lineage.identity(pins_path); pins = json.loads(pins_path.read_text())
    spec = public.PublicClipSpec(**pins["clip_spec"])
    if spec.episode_index != episode:
        raise ValueError("Explicit episode differs from frozen refined clip")
    historical_path=code/f"configs/cari_clip_{episode:06d}_historical_source_pins.json"
    historical=None
    if historical_path.exists():
        from cari_historical_source import verify_historical_source
        historical,historical_proof=verify_historical_source(root,historical_path)
        report["historical_source_binding"]=historical_proof
        chain=lineage.verify_refined_artifacts(root,code,spec,pins,source_code=historical)
    else:
        chain=lineage.verify_refined_artifacts(root,code,spec,pins)
    producer = chain["report"]; lineage.validate_refinement_report(producer, spec)
    frozen = {Path(path): row for path, row in chain["bindings"].items()}
    original_pin = code / f"configs/cari_clip_{episode:06d}_input_pins.json"
    original_pin_id = lineage.identity(original_pin)
    original = public.verify_public_inputs(root, spec, json.loads(original_pin.read_text()))
    for name, row in original["source_files"].items():
        frozen[root / name] = row
    assets, hashes = body._body_assets(root); source_id = body._source_identity(root)
    if (producer.get("body_assets") != hashes or producer.get("inference_source_identity") != source_id
            or (assets / "mhr_buffers.pt").exists()):
        raise ValueError("Same original native decoder assets/source required")
    vendor = root / "vendor/video_to_data"; body._pinned_checkout(vendor, body.UPSTREAM_REVISION)
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    tool = root / "vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"; model = root / "weights/mhr/mhr_model.pt"
    for path, digest in ((native / "lib_mhr/mhr_layer.py", geometry.LAYER_SHA), (tool, geometry.CONVERTER_SHA),
            (model, geometry.REFERENCE_SHA), (native / lineage.contract.OPTIMIZER_RELATIVE_PATH, lineage.contract.OPTIMIZER_SHA256)):
        geometry.binding(path, digest); frozen[path] = public.identity(path)
    for name, row in hashes.items():
        frozen[assets / name] = row
    acquisition = root / "results/weights-acquisition.json"
    frozen[acquisition] = public.identity(acquisition)
    refinement_receipt = root / "results/cari-refinement-assets.json"
    lineage.contract.require_asset_receipt(json.loads(refinement_receipt.read_text()))
    frozen[refinement_receipt] = public.identity(refinement_receipt)
    for name, expected in lineage.contract.REFINEMENT_ASSETS.items():
        path = root / "weights/cari4d/refinement" / name
        frozen[path] = public.identity(path)
        if frozen[path] != {key: expected[key] for key in ("sha256", "bytes")}:
            raise ValueError("Original refinement auxiliary asset changed")
    helpers = source_helpers(code)
    report.update(phase="native_refined_decode", clip_spec=asdict(spec), input_pins=original_pin_id, refined_pins=pin_id,
        refined_report_sha256=pins["refined"]["sha256"], refined_bundle_sha256=pins["refined_files"]["refined.pth"]["sha256"],
        source_helpers=helpers, body_assets=hashes, inference_source_identity=source_id,
        source_files=original["source_files"], decoder_identity=producer["decoder_identity"])
    persist()
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Fresh explicit native CUDA runtime required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__) != "2.5.1+cu124" or torch.version.cuda != "12.4":
        raise ValueError("Pinned original native H100 runtime required")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    report["runtime"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__,
        python=platform.python_version(), deterministic_algorithms=True, warn_only=False,
        jit_optimized_execution=False, tf32=False, cudnn_benchmark=False, seed=0, chunk=CHUNK)
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0")
    from lib_mhr.mhr_layer import MHRLayer
    from learning.training import mhr_opt_refineout as optimizer
    if (Path(optimizer.__file__).resolve() != native / lineage.contract.OPTIMIZER_RELATIVE_PATH
            or Path(sys.modules[MHRLayer.__module__].__file__).resolve() != native / "lib_mhr/mhr_layer.py"):
        raise ValueError("Actual unchanged native decoder/mesh-loader source required")
    source_path = root / f"outputs/episode_{episode:06d}/cari_shared_forward_v1/coconet.pth"
    refined_path = chain["directory"] / "refined.pth"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    refined = torch.load(refined_path, map_location="cpu", weights_only=False)
    mesh = original["paths"]["mesh"]
    params, poses, metadata = validate_refined(source, refined, mesh, spec.total_frames)
    if (producer.get("metadata") != metadata or producer.get("object_mesh_sha256") != sha256(mesh)
            or producer.get("bundle_sha256") != pins["refined_files"]["refined.pth"]["sha256"]):
        raise ValueError("Actual refined bundle/mesh/metadata chain differs")
    fingerprints = [lineage.fingerprint(value) for value in (source, refined)]
    vertices, object_faces = optimizer._load_object_vertices(mesh)
    K = np.asarray(original["wild"]["intrinsics"], np.float64)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"), checkpoint_path=assets / "model.ckpt",
        buffer_path=out / "never_use_unverified_buffers.pt", mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
    if layer.decoder_identity() != producer["decoder_identity"]:
        raise ValueError("Prepared/refined native decoder identity differs")
    tensors = lambda params: {key: torch.tensor(value.copy(), device="cuda", dtype=torch.float32) for key, value in params.items()}
    def native_decode(params):
        decoded = layer.mhr_forward(tensors(params)); torch.cuda.synchronize()
        return dict(vertices=decoded.vertices.cpu().numpy(), joints=decoded.joints.cpu().numpy(),
                    keypoints=decoded.keypoints.cpu().numpy(), faces=decoded.faces.cpu().numpy())
    def native_direct(params):
        t = tensors(params); context = layer.backend._vertices_context(t, detach_fixed=False)
        trans, pose, shape, scale = layer.backend._mutable_vertices_inputs(t, context)
        if context.head.enable_hand_model:
            raise ValueError("Original body-only native head required")
        vertices, controls = context.head.mhr_forward(global_trans=trans * context.flip, global_rot=context.global_rot,
            body_pose_params=pose, hand_pose_params=context.hand, scale_params=scale, shape_params=shape,
            expr_params=context.face, return_model_params=True)
        pca = context.head.scale_mean[None, :] + scale @ context.head.scale_comps; torch.cuda.synchronize()
        return (vertices * context.flip).cpu().numpy(), controls.cpu().numpy(), pca.cpu().numpy()
    module_spec = importlib.util.spec_from_file_location("cari_full_official_reference", tool)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    reference = module.MHR(str(model), "cuda", chunk=CHUNK, precision="float32")
    if (Path(module.MHR.run.__code__.co_filename).resolve() != tool or reference.mdtype != torch.float32
            or reference.dtype != torch.float64 or reference.chunk != CHUNK or reference.device.type != "cuda"):
        raise ValueError("Exact original FP32-model/FP64-residual official consumer required")
    def reference_decode(stored, selection):
        pose = torch.tensor(stored["pose"][selection], device="cuda", dtype=torch.float64)
        shared = torch.tensor(np.r_[stored["scales"], stored["shape"]][None], device="cuda", dtype=torch.float64)
        vertices, _ = reference.run(pose, shared); torch.cuda.synchronize(); return vertices.cpu().numpy()
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        outputs = export_geometry(params, poses, vertices, object_faces, K, spec, out, mesh,
            native_decode, native_direct, reference_decode, report, persist)
    if ([lineage.fingerprint(value) for value in (source, refined)] != fingerprints
            or {path: public.identity(path) for path in frozen} != frozen
            or lineage.identity(pins_path) != pin_id or lineage.identity(original_pin) != original_pin_id
            or source_helpers(code) != helpers or body._source_identity(root) != source_id or body._body_assets(root)[1] != hashes
            or {name: public.identity(out / name) for name in outputs} != outputs):
        raise ValueError("Frozen full native source/prediction/assets/exports changed")
    # Recheck complete producer inventories, not just the files opened above.
    if historical is not None:
        if verify_historical_source(root,historical_path)!=(historical,historical_proof):
            raise ValueError("Original historical source changed after replay")
        lineage.verify_refined_artifacts(root,code,spec,pins,source_code=historical)
    else:
        lineage.verify_refined_artifacts(root,code,spec,pins)
    report.update(status="pass", phase="complete", source_inputs_assets_rehashed=True, source_helpers_rehashed=True,
        raw_masks_contacts_object_pose_unchanged=True, full_original_native_export_verified=True,
        final_Parquet_produced=False, quality_verified=False, submission_eligible=False)
    validate_export_report(report, spec)


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); code = Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    out = root / output_relative(args.episode)
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or os.geteuid() != 1000
            or {path.name for path in Path("/sys/class/net").iterdir()} != {"lo"} or os.environ["WR_IMAGE_ID"] != IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or root.resolve() != root.absolute() or code.resolve() != code.absolute()
            or not out.is_dir() or any(out.iterdir()) or out.resolve() != out.absolute()):
        raise ValueError("Fresh source-bound offline Azure full native export required")
    report = dict(stage=STAGE, status="fail", phase="public_integrity", episode_index=args.episode, producer_revision=revision,
        script_sha256=sha256(Path(__file__)), image_id=IMAGE, input_track="track_1", network="none",
        ground_truth_used=False, ground_truth_read=False, private_truth_read=False, hand_labeled_test=False, oracle_modes=[],
        learned_inference_calls=0, optimizer_calls=0, converter_LM_calls=0, identity_reselection_performed=False,
        quality_verified=False, adoption_authorized=False, submission_produced=False, submission_eligible=False,
        final_Parquet_produced=False, budget_seconds=BUDGET)
    report.update({name + suffix: 0 for name in CALLS for suffix in ("_attempts", "_returns", "_validated")})
    receipt = out / "report.json"; started = time.perf_counter()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_):
            raise TimeoutError("Full native direct export exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(root, out, code, args.episode, report, persist)
        except BaseException as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o444)


if __name__ == "__main__":
    main()
