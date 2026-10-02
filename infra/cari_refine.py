"""Offline public MHR post-CoCoNet refinement, not a challenge accuracy gate.

Run the pinned parity optimizer unchanged: full clip, 300 requested steps
(301 actual updates at this revision). Root/translation, hands, identity,
internal body translations, object shape/scale/rotation and camera are fixed.
Only body rotation controls and object translation are optimized. The frozen
CoCoNet bundle supplies automatic masks/contact predictions, never GT.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

from cari_converter import (require_report, require_full_forward_report,
                            validate_native_bundle, require_aligned_object_metadata)
from cari_runner import UPSTREAM_REVISION, CHECKPOINT_SHA256
from world_reward.data import sha256
from world_reward.contracts import require_rigid_transforms
from world_reward.mesh_geometry import normalize_degenerate_faces

STAGE = "world_reward_native_cari_full_refinement"
NUM_STEPS = 300
BATCH_SIZE = 0
MAX_SECONDS = 7200
OPTIMIZER_RELATIVE_PATH = "learning/training/mhr_opt_refineout.py"
OPTIMIZER_SHA256 = "84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b"
REFINEMENT_ASSETS = {
    "mhr_collision_proxy_4000v.npz": {
        "bytes": 126142, "sha256": "a026fe82599609ee82814b4c18eb0335fa70925740dea5e1c1e1c0930d5edcf7",
        "revision": UPSTREAM_REVISION,
    },
    "mhr_hand_surface_spec.npz": {
        "bytes": 47140, "sha256": "65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8",
        "revision": UPSTREAM_REVISION,
    },
}
FIXED_BLOCKS = ("mhr_global_rot6d", "mhr_trans", "mhr_hand", "mhr_shape", "mhr_scale", "mhr_face")
OPTIMIZED_PARAMETERS = ["object_translation", "mhr_body_pose_cont_rotation_controls"]
FIXED_PARAMETERS = ["object_rotation", *FIXED_BLOCKS, "mhr_body_pose_cont_internal_translations"]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(30), default=15)
    return parser


def require_asset_receipt(receipt: Mapping) -> None:
    """Verify two exact upstream LFS assets; no first-seen hashes accepted."""
    if not isinstance(receipt, Mapping) or receipt.get("upstream_revision") != UPSTREAM_REVISION:
        raise ValueError("Refinement assets require the pinned upstream revision")
    records = receipt.get("assets")
    if not isinstance(records, list) or len(records) != len(REFINEMENT_ASSETS):
        raise ValueError("Require exactly two refinement asset records")
    names = [record.get("filename") for record in records if isinstance(record, Mapping)]
    if len(names) != 2 or set(names) != set(REFINEMENT_ASSETS):
        raise ValueError("Refinement asset names must be unique and exact")
    for record in records:
        expected = REFINEMENT_ASSETS[record["filename"]]
        if type(record.get("bytes")) is not int or any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Refinement asset bytes/hash/revision differ from the release")


def require_full_refinement_report(report: Mapping) -> None:
    """Pure consumer gate: a partial optimizer run is never a final bundle."""
    require_report(report, STAGE)
    metadata = report.get("metadata")
    if (type(report.get("frames")) is not int or report["frames"] < 1
            or type(report.get("episode_index")) is not int or not 0 <= report["episode_index"] < 30
            or report.get("checkpoint_sha256") != CHECKPOINT_SHA256
            or report.get("native_refinement_verified") is not True
            or report.get("ground_truth_read") is not False
            or report.get("network") != "none" or report.get("phase") != "complete"
            or not isinstance(report.get("producer_revision"), str)
            or re.fullmatch(r"[0-9a-f]{40}", report["producer_revision"]) is None
            or not isinstance(report.get("image_id"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", report["image_id"]) is None
            or report.get("optimizer_sha256") != OPTIMIZER_SHA256
            or report.get("refinement_assets") != REFINEMENT_ASSETS
            or not isinstance(report.get("inference_source_identity"), Mapping)
            or not report.get("inference_source_identity")
            or not isinstance(report.get("body_assets"), Mapping) or not report.get("body_assets")
            or not isinstance(metadata, Mapping)
            or metadata.get("native_refinement_verified") is not True
            or metadata.get("full_original_frame_coverage_verified") is not True
            or metadata.get("frozen_parameters_bit_identical") is not True
            or metadata.get("requested_steps") != NUM_STEPS
            or type(metadata.get("requested_steps")) is not int
            or metadata.get("effective_optimizer_updates") != NUM_STEPS + 1
            or type(metadata.get("effective_optimizer_updates")) is not int
            or metadata.get("batch_size") != BATCH_SIZE or type(metadata.get("batch_size")) is not int):
        raise ValueError("Require actual full-frame, unchanged public parity refinement")
    for name in ("forward_report_sha256", "inputs_report_sha256", "source_bundle_sha256", "bundle_sha256"):
        value = report.get(name)
        _required_hash(value)


def _bit_identical(a, b) -> bool:
    left, right = np.asarray(a), np.asarray(b)
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def validate_refined_bundle(source: Mapping, result: Mapping, count: int) -> dict:
    """Check optimized/fixed ABI and coverage; no quality or score inference."""
    old, old_pose = validate_native_bundle(source, count)
    new, new_pose = validate_native_bundle(result, count)
    require_aligned_object_metadata(source["metadata"])
    require_aligned_object_metadata(result["metadata"])
    if source["metadata"].get("object_mesh") != result["metadata"].get("object_mesh"):
        raise ValueError("Refinement changed the object mesh/camera gauge")
    for name in FIXED_BLOCKS:
        if not _bit_identical(old[name], new[name]):
            raise ValueError(f"Frozen native parameter changed: {name}")
    if (not _bit_identical(old["mhr_body_pose_cont"][:, 254:], new["mhr_body_pose_cont"][:, 254:])
            or not _bit_identical(old_pose[:, :3, :3], new_pose[:, :3, :3])):
        raise ValueError("Frozen internal translations or object rotations changed")
    postopt = result.get("postopt", {})
    cfg = postopt.get("config", {})
    required_cfg = {"num_steps": NUM_STEPS, "batch_size": BATCH_SIZE, "frame_start": 0, "frame_limit": 0,
                    "freeze_object_rotation": True, "freeze_body_internal_translations": True}
    if (postopt.get("mode") != "smplh_parity" or postopt.get("frame_indices") != list(range(count))
            or postopt.get("resolved_batch_size") != count
            or postopt.get("batch_sampling") != "full_clip_v1"
            or postopt.get("optimized_parameters") != OPTIMIZED_PARAMETERS
            or postopt.get("fixed_parameters") != FIXED_PARAMETERS
            or not isinstance(cfg, Mapping)
            or any(type(cfg.get(k)) is not type(v) or cfg.get(k) != v for k, v in required_cfg.items())):
        raise ValueError("Refinement changed the full-clip public parity protocol")
    history = postopt.get("history")
    if not isinstance(history, list) or not history or history[0].get("iter") != 0 or history[-1].get("iter") != NUM_STEPS:
        raise ValueError("Refinement did not finish every requested step")
    for item in history:
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in item.values()):
            raise ValueError("Nonfinite or invalid optimizer history")
    diagnostics = postopt.get("final_diagnostics")
    if not isinstance(diagnostics, Mapping) or not diagnostics or not all(
            isinstance(v, (int, float)) and np.isfinite(v) for v in diagnostics.values()):
        raise ValueError("Final optimizer diagnostics are missing or nonfinite")
    return {"native_refinement_verified": True, "full_original_frame_coverage_verified": True,
            "frozen_parameters_bit_identical": True, "requested_steps": NUM_STEPS,
            "effective_optimizer_updates": NUM_STEPS + 1, "batch_size": BATCH_SIZE,
            "optimized_parameters": OPTIMIZED_PARAMETERS, "fixed_parameters": FIXED_PARAMETERS}


def verify_aligned_geometry(source_vertices, source_faces, aligned_vertices, aligned_faces, transform) -> dict:
    """Match every oriented triangle once after A, including cavity surfaces.

    Allow only the existing 1e-5 m GLB float32 frame-change roundoff. No
    welding, simplification, inversion, scale change or historical byte claim.
    Cyclic corner/face/vertex reorder is allowed; reversed winding is not.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    require_rigid_transforms(np.asarray(transform)[None], 1)
    left_ids, _ = normalize_degenerate_faces(source_vertices, source_faces)
    right_ids, _ = normalize_degenerate_faces(aligned_vertices, aligned_faces)
    if not len(left_ids) or len(left_ids) != len(right_ids):
        raise ValueError("Aligned export lost meaningful source triangles")
    a = np.asarray(source_vertices, dtype=np.float64)[np.asarray(source_faces)[left_ids]]
    a = a @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
    b = np.asarray(aligned_vertices, dtype=np.float64)[np.asarray(aligned_faces)[right_ids]]
    options = cKDTree(b.mean(axis=1)).query_ball_point(a.mean(axis=1), r=1e-5)
    rows, columns, errors = [], [], {}
    for i, candidates in enumerate(options):
        for j in candidates:
            error = min(float(np.linalg.norm(a[i] - np.roll(b[j], shift, axis=0), axis=-1).max()) for shift in range(3))
            if error <= 1e-5:
                rows.append(i); columns.append(j); errors[i, j] = error
    graph = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(a), len(b)))
    match = maximum_bipartite_matching(graph, perm_type="column")
    if np.any(match < 0):
        raise ValueError("Aligned export changed oriented triangle geometry")
    return {"oriented_triangles_verified": len(a), "max_vertex_roundtrip_error_m": max(errors[i, int(j)] for i, j in enumerate(match)),
            "roundoff_limit_m": 1e-5, "historical_aligned_mesh_byte_identity_proven": False,
            "geometry_modified": False, "additional_scale_applied": False}


def _required_hash(value) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Require an exact frozen SHA-256")
    return value


def _file(path: Path, root: Path, digest: str | None = None) -> Path:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Frozen regular artifact required: {path}")
    if digest is not None and sha256(path) != digest:
        raise ValueError(f"Frozen artifact changed: {path}")
    return path


def frozen_inputs(root: Path, episode: int) -> dict:
    """Read reports and their artifacts only, not RGB, challenge GT or labels."""
    base = root / f"outputs/episode_{episode:06d}"
    forward_path, inputs_path = base / "cari_forward/report.json", base / "cari_inputs/report.json"
    forward = json.loads(_file(forward_path, base).read_text())
    require_full_forward_report(forward)
    inputs = json.loads(_file(inputs_path, base, _required_hash(forward.get("inputs_report_sha256"))).read_text())
    require_report(inputs, "world_reward_native_cari_inputs")
    for record in (forward, inputs):
        if type(record.get("episode_index", episode)) is not int or record.get("episode_index", episode) != episode:
            raise ValueError("Frozen producer belongs to another episode")
    _required_hash(inputs.get("input_sha256"))
    count = inputs.get("frames")
    if type(count) is not int or count < 1 or inputs.get("original_frame_coverage_verified") is not True:
        raise ValueError("Full original coverage is required")
    reports = {"forward": forward, "inputs": inputs}
    for key, relative, stage in (
        ("body", "body_full/report.json", "sam3d_body_full_video_initializer"),
        ("adapter", "body_full/cari_adapter/report.json", "native_cari_body_adapter_full_video"),
        ("object", "object_pose_full/report.json", "fixed_scale_full_object_pose_initializer"),
    ):
        record = json.loads(_file(base / relative, base, _required_hash(inputs["input_report_sha256"][key])).read_text())
        require_report(record, stage)
        if type(record.get("episode_index", episode)) is not int or record.get("episode_index", episode) != episode:
            raise ValueError("Frozen dependency belongs to another episode")
        reports[key] = record
    if (reports["body"].get("total_video_frames") != count or reports["adapter"].get("frames") != count
            or reports["object"].get("original_frame_coverage_verified") is not True
            or reports["body"].get("frame_indices") != list(range(count))
            or reports["body"].get("mhr_geometry_forward_verified") is not True
            or reports["adapter"].get("body_report_sha256") != inputs["input_report_sha256"]["body"]
            or reports["body"].get("input_sha256") != inputs.get("input_sha256")
            or reports["object"].get("input_sha256") != inputs.get("input_sha256")):
        raise ValueError("Frozen human/object provenance or coverage differ")
    bundle = _file(base / "cari_forward/coconet.pth", base, _required_hash(forward.get("bundle_sha256")))
    _file(base / "object_pose_full/geometry_and_poses.npz", base,
          _required_hash(reports["object"].get("geometry_and_poses_sha256")))
    _file(base / "body_full/cari_adapter/canonical_initializer.pkl", base,
          _required_hash(reports["adapter"].get("canonical_initializer_sha256")))
    export = base / f"cari_inputs/export/episode_{episode:06d}"
    if Path(inputs["export_seq"]).resolve() != export.resolve():
        raise ValueError("Export escapes the selected episode")
    _file(export / "wild_export.json", base, _required_hash(inputs["file_sha256"]["wild_export"]))
    return {"base": base, "bundle": bundle, "mesh": _file(export / "object_mesh/output_aligned.glb", base),
            "count": count, "reports": reports, "forward_path": forward_path, "inputs_path": inputs_path}


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Linux GPU container with network none")
    args = _argument_parser().parse_args()
    root = Path(os.environ["WR_ROOT"])
    output = root / f"outputs/episode_{args.episode:06d}/cari_refined"
    if output.is_symlink() or not output.is_dir() or list(output.iterdir()):
        raise RuntimeError("Wrapper must reserve a new empty refinement output")
    started = time.perf_counter()
    report = {"stage": STAGE, "status": "running", "episode_index": args.episode,
              "input_track": "track_1", "ground_truth_used": False, "ground_truth_read": False,
              "hand_labeled_test": False, "oracle_modes": [], "network": "none", "native_refinement_verified": False,
              "submission_eligible": False, "challenge_performance_verified": False, "adoption_performed": False,
              "optimizer_sha256": OPTIMIZER_SHA256, "script_sha256": sha256(Path(__file__)),
              "producer_revision": os.environ["WR_CODE_REVISION"], "image_id": os.environ["WR_IMAGE_ID"],
              "max_seconds": MAX_SECONDS, "requested_steps": NUM_STEPS, "effective_optimizer_updates": NUM_STEPS + 1}
    report_path = output / "report.json"
    def persist():
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    def timeout_handler(*_):
        raise TimeoutError("Full refinement exceeded its explicit time budget")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, timeout_handler)
    signal.alarm(MAX_SECONDS)
    persist()
    try:
        report["phase"] = "frozen_input_and_source_checks"; persist()
        selected = frozen_inputs(root, args.episode)
        import torch
        from body_smoke import _body_assets, _pinned_checkout, _source_identity
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU or altered-protocol fallback")
        vendor = root / "vendor/video_to_data"
        _pinned_checkout(vendor, UPSTREAM_REVISION)
        native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
        _file(native / OPTIMIZER_RELATIVE_PATH, vendor, OPTIMIZER_SHA256)
        source_identity = _source_identity(root)
        if any(r.get("inference_source_identity") != source_identity for r in
               (selected["reports"]["forward"], selected["reports"]["body"], selected["reports"]["adapter"])):
            raise ValueError("Native decoder differs from original frozen producer source")
        assets, body_hashes = _body_assets(root)
        if body_hashes != selected["reports"]["body"]["body_assets"]:
            raise ValueError("Original Body asset identity changed")
        receipt_path = _file(root / "results/cari-refinement-assets.json", root)
        receipt = json.loads(receipt_path.read_text()); require_asset_receipt(receipt)
        asset_paths = {}
        for name, expected in REFINEMENT_ASSETS.items():
            path = _file(root / "weights/cari4d/refinement" / name, root, expected["sha256"])
            if path.stat().st_size != expected["bytes"]:
                raise ValueError("Refinement asset byte size changed")
            asset_paths[name] = path
        sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
        os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0")
        from learning.training import mhr_opt_refineout as optimizer
        from lib_mhr.mhr_layer import MHRLayer
        if Path(optimizer.__file__).resolve() != (native / OPTIMIZER_RELATIVE_PATH).resolve():
            raise ValueError("Native optimizer resolved outside pinned source")
        cfg = optimizer.MHRParityPostOptConfig(
            penetration_collision_proxy_path=str(asset_paths["mhr_collision_proxy_4000v.npz"]),
            hand_surface_spec_path=str(asset_paths["mhr_hand_surface_spec.npz"]), report_every=100,
        )
        layer = MHRLayer.from_mhr_assets(
            mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"), checkpoint_path=assets / "model.ckpt",
            buffer_path=output / "never_use_compact_buffers.pt", mhr_model_path=assets / "assets/mhr_model.pt", device="cuda",
        )
        decoder = layer.decoder_identity()
        if decoder != selected["reports"]["adapter"]["decoder_identity"]:
            raise ValueError("Original decoder asset identity differs")
        from lib_mhr.collision_proxy import load_mhr_collision_proxy
        from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
        human_faces = layer.mesh_faces(device="cuda").detach().cpu().numpy()
        proxy = load_mhr_collision_proxy(human_faces, asset_paths["mhr_collision_proxy_4000v.npz"])
        hand_spec = load_mhr_hand_surface_spec(asset_paths["mhr_hand_surface_spec.npz"], faces=human_faces)
        if proxy.vertex_count != 4000 or hand_spec.mhr_model_sha256 != body_hashes["assets/mhr_model.pt"]["sha256"]:
            raise ValueError("Refinement proxy or hand spec differs from original MHR model")
        report["asset_topology"] = {"collision_proxy_vertices": proxy.vertex_count,
                                    "hand_vertices_per_side": int(hand_spec.vertex_indices.shape[1]),
                                    "hand_spec_model_sha256": hand_spec.mhr_model_sha256}
        source = torch.load(selected["bundle"], map_location="cpu", weights_only=False)
        validate_native_bundle(source, selected["count"])
        require_aligned_object_metadata(source["metadata"])
        if Path(source["metadata"]["object_mesh"]).resolve() != selected["mesh"].resolve():
            raise ValueError("CoCoNet poses and refinement mesh differ")
        vertices, faces = optimizer._load_object_vertices(selected["mesh"])
        with np.load(selected["base"] / "object_pose_full/geometry_and_poses.npz", allow_pickle=False) as packed:
            packed_vertices, packed_faces = packed["vertices"], packed["faces"]
            if packed["object_scale"].shape != () or float(packed["object_scale"]) != 1.0:
                raise ValueError("Object scale must remain baked exactly once")
            wild = json.loads((selected["mesh"].parent.parent / "wild_export.json").read_text())
            report["aligned_mesh_structural_roundtrip"] = verify_aligned_geometry(
                packed_vertices, packed_faces, vertices, faces, np.asarray(wild["source_object_mesh_to_aligned_transform"], dtype=np.float64))
        report.update(frames=selected["count"], forward_report_sha256=sha256(selected["forward_path"]),
                      inputs_report_sha256=sha256(selected["inputs_path"]), source_bundle_sha256=sha256(selected["bundle"]),
                      checkpoint_sha256=CHECKPOINT_SHA256, inference_source_identity=source_identity,
                      body_assets=body_hashes, decoder_identity=decoder, refinement_assets=REFINEMENT_ASSETS,
                      assets_receipt_sha256=sha256(receipt_path), object_mesh_sha256=sha256(selected["mesh"]), config=asdict(cfg))
        report["phase"] = "native_full_clip_refinement"; persist()
        original_hub = torch.hub.load
        def forbidden_hub(*_, **__):
            raise RuntimeError("Native refinement must not load any Hub model")
        torch.hub.load = forbidden_hub
        try:
            result = optimizer.run_postopt_smplh_parity(source, vertices, faces, cfg, mhr_layer=layer)
        finally:
            torch.hub.load = original_hub
        torch.cuda.synchronize()
        if result["postopt"]["config"] != asdict(cfg):
            raise ValueError("Native optimizer altered the requested configuration")
        report["metadata"] = validate_refined_bundle(source, result, selected["count"])
        # Frozen CoCoNet file must remain untouched; write only this new namespace.
        _file(selected["bundle"], selected["base"], report["source_bundle_sha256"])
        destination = output / "refined.pth"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        torch.save(result, destination, pickle_protocol=4)
        report.update(bundle_sha256=sha256(destination), native_refinement_verified=True, status="pass", phase="complete")
        require_full_refinement_report(report)
        persist()
    except BaseException as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        persist()
        raise
    finally:
        signal.alarm(0)
    print(json.dumps({"stage": STAGE, "status": "pass", "frames": report["frames"],
                      "elapsed_seconds": report["elapsed_seconds"], "challenge_performance_verified": False}))


if __name__ == "__main__":
    main()
