"""One unchanged native full96 parity refinement; no quality or submission.

Only the new constrained forward and its complete prepared snapshot are used.
The native optimizer executes 300 requested/301 actual updates. No identity,
camera, mesh, root, hand, or object-rotation fit is added by this driver.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import sys
import time

import numpy as np
import body_smoke as body
import cari96_inputs as inputs
import cari_refine as contract
from cari_converter import validate_native_bundle, require_aligned_object_metadata

BASE = "validation/cari96_refined_v1"
PREPARE = "validation/cari96_public_v1"
FORWARD = "validation/cari96_forward_v1"
STAGE = "public_cari96_unchanged_native_parity_refinement"
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BUDGET, FRAMES = 1200, 96
PREPARE_STAGE = "public_cari96_shared_initializer_native_abi"
FORWARD_STAGE = "public_cari96_constrained_coconet_forward"
LAYER_SHA = "a753ab8e730b6730fca275384fab629859311983292a407390d88c66ffe68c23"


def identity(path, immutable=True):
    path = inputs.regular(path)
    if immutable and path.stat().st_mode & 0o222: raise ValueError("Immutable producer file required")
    return inputs.identity(path)


def pinned_report(root, base, pins, role):
    """Exact manifest first, then complete directory inventory; no fallback."""
    if (type(pins) is not dict or set(pins) != {"schema", role, role + "_files"}
            or pins["schema"] != f"world-reward-cari96-{role}-pins-v1"):
        raise ValueError("Explicit stage-specific immutable pins required")
    proof, files = pins[role], pins[role + "_files"]
    if (type(proof) is not dict or set(proof) != {"sha256", "bytes", "producer_revision", "script_sha256"}
            or type(files) is not dict or not files or files.get("report.json") != {k: proof[k] for k in ("sha256", "bytes")}):
        raise ValueError("Complete producer inventory including its report required")
    for key in ("producer_revision", "script_sha256"):
        if type(proof[key]) is not str or not re.fullmatch(r"[0-9a-f]{40}" if key == "producer_revision" else r"[0-9a-f]{64}", proof[key]):
            raise ValueError("Exact producer source identity required")
    expected = {}
    for name, row in files.items():
        if type(name) is not str: raise ValueError("Regular relative filename required")
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or str(relative) != name
                or type(row) is not dict or set(row) != {"sha256", "bytes"}
                or type(row["bytes"]) is not int or row["bytes"] <= 0
                or type(row["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Safe regular file identities required")
        path = Path(root) / base / name
        if identity(path) != row: raise ValueError("Frozen producer file changed")
        expected[path] = row
    folder = Path(root) / base
    actual = set()
    for p in folder.rglob("*"):
        if p.is_symlink(): raise ValueError("Producer symlink forbidden")
        if p.is_file(): actual.add(str(p.relative_to(folder)))
        elif not p.is_dir(): raise ValueError("Producer special file forbidden")
    if actual != set(files): raise ValueError("Producer inventory differs; no caches or hidden predictions")
    report = json.loads((folder / "report.json").read_text())
    if any(report.get(k) != proof[k] for k in ("producer_revision", "script_sha256")):
        raise ValueError("Report producer identity differs")
    return report, expected


def require_report(report, stage):
    required = dict(stage=stage, status="pass", phase="complete", image_id=IMAGE, frames=FRAMES,
        input_track="track_1", ground_truth_used=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], network="none")
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in required.items()):
        raise ValueError("Complete public-only native96 producer required")


def source_inputs(root, forward_pins, prepare_pins, source_pins):
    prepare, frozen = pinned_report(root, PREPARE, prepare_pins, "prepare")
    forward, forward_files = pinned_report(root, FORWARD, forward_pins, "forward"); frozen.update(forward_files)
    require_report(prepare, PREPARE_STAGE); require_report(forward, FORWARD_STAGE)
    prepare_outputs = {k: v for k, v in prepare_pins["prepare_files"].items() if k != "report.json"}
    if prepare.get("output_files") != prepare_outputs: raise ValueError("Prepare output receipt differs")
    if (forward.get("prepare_report_sha256") != prepare_pins["prepare"]["sha256"]
            or forward.get("prepare_files") != prepare_pins["prepare_files"]
            or forward.get("checkpoint_sha256") != contract.CHECKPOINT_SHA256
            or forward.get("bundle_sha256") != forward_pins["forward_files"].get("coconet.pth", {}).get("sha256")
            or set(forward_pins["forward_files"]) != {"report.json", "coconet.pth"}):
        raise ValueError("Forward/prepare/checkpoint chain differs")
    for key in ("shared_identity_verified", "raw_prediction_bytes_preserved", "native_global_unchanged", "hub_restored"):
        if forward.get(key) is not True: raise ValueError("Actual constrained native forward proof missing")
    for key in ("forward_attempts", "forward_returns", "forward_validated", "composition_hook_calls",
                "composition_delegate_calls", "composition_delegate_returns", "composition_verified_calls"):
        if type(forward.get(key)) is not int or forward[key] != 1: raise ValueError("Exactly one complete96 forward required")
    inputs.validate_pins(source_pins)
    manifest_path = Path(root) / PREPARE / "inputs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs._no_oracle(manifest)
    if (manifest.get("stage") != "world_reward_public_cari96_inputs_snapshot" or manifest.get("status") != "pass"
            or manifest.get("frames") != FRAMES or manifest.get("original_frame_indices") != list(range(FRAMES))
            or manifest.get("source_frames") != 501 or manifest.get("source_files") != source_pins["source_files"]
            or manifest.get("source_inputs_report_sha256") != source_pins["source_files"][inputs.REPORT]["sha256"]
            or manifest.get("initializer_modified") is not False or manifest.get("ground_truth_read") is not False
            or prepare.get("snapshot_manifest_sha256") != inputs.sha256(manifest_path)):
        raise ValueError("Original public501 to real96 lineage differs")
    snapshot_outputs = {k.removeprefix("inputs/"): v for k, v in prepare_outputs.items()
                        if k.startswith("inputs/") and k != "inputs/manifest.json"}
    if manifest.get("output_files") != snapshot_outputs: raise ValueError("Complete snapshot output chain differs")
    if (prepare.get("shared_identity_verified") is not True or prepare.get("original_initializer_unchanged") is not True
            or prepare.get("source_inputs_assets_rehashed") is not True
            or prepare.get("frozen_outputs_rehashed_after_reference") is not True):
        raise ValueError("Prepared shared decoder/source verification required")
    for key in ("native_geometry_calls", "native_direct_calls", "native_replay_calls", "reference_calls"):
        if type(prepare.get(key)) is not int or prepare[key] != 6: raise ValueError("All96 prepared native/direct/reference chunks required")
    errors = np.asarray(prepare.get("reference_per_frame_mean_mm"))
    if errors.shape != (FRAMES,) or not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > 2):
        raise ValueError("Every prepared frame must pass unchanged2mm representation gate")
    return forward, prepare, frozen


def fingerprint(value):
    """Fingerprint complete source mapping without a duplicate geometry copy."""
    digest = hashlib.sha256()
    def visit(v):
        if np.ma.isMaskedArray(v): raise ValueError("Masked source arrays forbidden")
        if hasattr(v, "detach") and hasattr(v, "cpu"): v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject: raise ValueError("Object source arrays forbidden")
            digest.update(str((v.shape, v.dtype.str)).encode()); digest.update(v.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v): digest.update(str(key).encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(str(type(v)).encode())
            for x in v: visit(x)
        elif isinstance(v, (str, int, float, bool, type(None), np.generic)):
            digest.update(repr(v).encode())
        else: raise ValueError("Unsupported opaque native source value")
    visit(value)
    return digest.hexdigest()


def validate_source_bundle(bundle, mesh):
    params, pose = validate_native_bundle(bundle, FRAMES)
    require_aligned_object_metadata(bundle["metadata"])
    if (bundle.get("schema") != "cari4d.mhr_wild_inference.v1"
            or bundle["metadata"].get("object_mesh") != str(mesh)
            or bundle["metadata"].get("materialized_input_cache") is not None):
        raise ValueError("New aligned no-cache native bundle required")
    for key in ("mhr_shape", "mhr_scale"):
        if any(row.tobytes() != params[key][0].tobytes() for row in params[key]):
            raise ValueError("Shared native identity must be byte-constant before refinement")
    if np.count_nonzero(params["mhr_face"]): raise ValueError("Expressions must remain zero")
    if any(a.dtype != np.float32 for a in params.values()) or pose.dtype != np.float32:
        raise ValueError("Unchanged native float32 predictions required")
    observations = bundle.get("observations", {})
    for key in ("human_mask", "object_mask", "postopt_human_mask", "postopt_object_mask"):
        a = np.asarray(observations.get(key))
        if (np.ma.isMaskedArray(observations.get(key)) or a.ndim != 3 or a.shape[0] != FRAMES
                or min(a.shape[1:]) < 1 or a.dtype != (np.float32 if key.startswith("postopt_") else np.bool_)
                or not np.isfinite(a).all() or not np.isin(a, [0, 1]).all()
                or (key.startswith("postopt_") and a.shape[1:] != (256, 256))):
            raise ValueError("Complete automatic native silhouette observations required")
    contact = np.asarray(bundle["pr"].get("contact_logits"))
    if (np.ma.isMaskedArray(bundle["pr"].get("contact_logits")) or contact.shape != (FRAMES, 2)
            or contact.dtype != np.float32 or not np.isfinite(contact).all()):
        raise ValueError("Complete native contact logits required")
    return params, pose


def validate_result(source, result):
    metadata = contract.validate_refined_bundle(source, result, FRAMES)
    if set(result) != set(source) | {"postopt"}:
        raise ValueError("Native refinement may only add its postopt record")
    for key in set(source) - {"pr"}:
        if key not in source or key not in result or fingerprint(source[key]) != fingerprint(result[key]):
            raise ValueError("Refinement altered frozen raw inputs, camera, topology or observations")
    if set(result["pr"]) != set(source["pr"]) | {"pose_abs_postopt"}:
        raise ValueError("Native refinement may only add its final object-pose copy")
    for key in set(source["pr"]) - {"mhr_body_pose_cont", "pose_abs"}:
        if fingerprint(source["pr"][key]) != fingerprint(result["pr"][key]):
            raise ValueError("Refinement altered a nonoptimized prediction, including contact")
    if fingerprint(result["pr"]["pose_abs_postopt"]) != fingerprint(result["pr"]["pose_abs"]):
        raise ValueError("Native final object-pose copies differ")
    return metadata


def invoke_optimizer(source, vertices, faces, cfg, layer, callback, report, persist):
    before = fingerprint(source)
    report["optimizer_attempts"] += 1; persist()
    result = callback(source, vertices, faces, cfg, mhr_layer=layer)
    report["optimizer_returns"] += 1; persist()
    if fingerprint(source) != before: raise ValueError("Native optimizer modified its frozen input")
    if result.get("postopt", {}).get("config") != asdict(cfg): raise ValueError("Native optimizer altered its exact configuration")
    report["metadata"] = validate_result(source, result)
    report["optimizer_validated"] += 1; persist()
    return result


def validate_refinement_report(report):
    """Pure downstream integrity gate; execution is not an accuracy result."""
    require_report(report, STAGE)
    expected = dict(optimizer_attempts=1, optimizer_returns=1, optimizer_validated=1,
        requested_steps=300, effective_optimizer_updates=301, source_frames=501,
        original_frame_indices=list(range(FRAMES)), source_inputs_assets_rehashed=True,
        saved_bundle_reloaded_verified=True, frozen_raw_inputs_byte_preserved=True,
        object_mesh_unchanged=True, native_refinement_verified=True, ground_truth_read=False,
        learned_inference_calls=0, quality_verified=False, adoption_authorized=False,
        submission_produced=False, numerical_bit_determinism_claimed=False,
        optimizer_sha256=contract.OPTIMIZER_SHA256, refinement_assets=contract.REFINEMENT_ASSETS)
    if any(type(report.get(k)) is not type(v) or report[k] != v for k, v in expected.items()):
        raise ValueError("Exactly one unchanged complete96 native refinement required")
    meta = report.get("metadata", {})
    if (meta.get("frozen_parameters_bit_identical") is not True
            or type(meta.get("effective_optimizer_updates")) is not int or meta["effective_optimizer_updates"] != 301
            or meta.get("optimized_parameters") != contract.OPTIMIZED_PARAMETERS
            or meta.get("fixed_parameters") != contract.FIXED_PARAMETERS):
        raise ValueError("Refinement parameter/update proof incomplete")
    for key in ("forward_report_sha256", "prepare_report_sha256", "source_bundle_sha256", "bundle_sha256", "object_mesh_sha256"):
        if type(report.get(key)) is not str or not re.fullmatch(r"[0-9a-f]{64}", report[key]):
            raise ValueError("Complete source/output SHA chain required")
    if (type(report.get("bundle_bytes")) is not int or report["bundle_bytes"] <= 0
            or report.get("output_files") != {"refined.pth": {"sha256": report["bundle_sha256"], "bytes": report["bundle_bytes"]}}):
        raise ValueError("Exactly one frozen refined bundle required")


def run(root, out, code, args, report, persist):
    config_paths = [Path(args.forward_pins), Path(args.prepare_pins), code / "configs/cari96_input_pins.json"]
    config_ids = {}
    for path in config_paths:
        if not path.is_relative_to(code / "configs"): raise ValueError("Immutable bundled configs required")
        config_ids[path] = identity(path)
    forward, prepare, frozen = source_inputs(root, *(json.loads(p.read_text()) for p in config_paths))
    report.update(phase="source_and_asset_audit", forward_report_sha256=frozen[root / FORWARD / "report.json"]["sha256"],
        prepare_report_sha256=frozen[root / PREPARE / "report.json"]["sha256"]); persist()
    vendor = root / "vendor/video_to_data"; body._pinned_checkout(vendor, body.UPSTREAM_REVISION)
    assets, hashes = body._body_assets(root); source_id = body._source_identity(root)
    if (hashes != prepare.get("body_assets") or hashes != forward.get("body_assets")
            or source_id != prepare.get("inference_source_identity") or source_id != forward.get("inference_source_identity")):
        raise ValueError("Original Body checkpoint/source identity differs")
    native = vendor / "reconstruction/modules/v2d_cari4d/lib/cari4d"
    optimizer_path = native / contract.OPTIMIZER_RELATIVE_PATH
    if inputs.sha256(inputs.regular(optimizer_path)) != contract.OPTIMIZER_SHA256: raise ValueError("Pinned unchanged optimizer required")
    layer_path = native / "lib_mhr/mhr_layer.py"
    if inputs.sha256(inputs.regular(layer_path)) != LAYER_SHA: raise ValueError("Pinned unchanged MHRLayer required")
    asset_receipt = root / "results/cari-refinement-assets.json"
    contract.require_asset_receipt(json.loads(inputs.regular(asset_receipt).read_text()))
    frozen[asset_receipt] = identity(asset_receipt, immutable=False)
    asset_paths = {}
    for name, expected in contract.REFINEMENT_ASSETS.items():
        path = root / "weights/cari4d/refinement" / name
        row = identity(path, immutable=False)
        if row != {k: expected[k] for k in ("sha256", "bytes")}: raise ValueError("Pinned refinement asset differs")
        asset_paths[name] = path; frozen[path] = row
    for name, row in hashes.items(): frozen[assets / name] = row
    frozen[optimizer_path] = identity(optimizer_path, immutable=False)
    frozen[layer_path] = identity(layer_path, immutable=False)
    acquisition = root / "results/weights-acquisition.json"
    frozen[acquisition] = identity(acquisition, immutable=False)
    helpers = {str(p.relative_to(code)): identity(p) for p in
        (Path(__file__), code / "infra/run_cari96_refine.sh", code / "infra/cari_refine.py", code / "infra/cari96_inputs.py")}
    report.update(source_helpers=helpers, body_assets=hashes, inference_source_identity=source_id,
                  optimizer_sha256=contract.OPTIMIZER_SHA256, refinement_assets=contract.REFINEMENT_ASSETS); persist()
    if "torch" in sys.modules: raise ValueError("Fresh native process required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__) != "2.5.1+cu124" or torch.version.cuda != "12.4":
        raise ValueError("Pinned native CUDA runtime required")
    if torch.are_deterministic_algorithms_enabled(): raise ValueError("No new strict backward policy may be imposed")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    report["runtime"] = dict(torch=str(torch.__version__), cuda=torch.version.cuda, python=platform.python_version(),
        numpy=np.__version__, deterministic_algorithms=False, tf32=False, seed=0, threads=4)
    sys.path[:0] = [str(native), "/workspace/v2d_sam3d_body/lib"]
    os.environ.update(MHR_ASSETS_ROOT=str(root / "weights/cari4d/sam3d_body"), MOMENTUM_ENABLED="0")
    from learning.training import mhr_opt_refineout as optimizer
    from lib_mhr.mhr_layer import MHRLayer
    if (Path(optimizer.__file__).resolve() != optimizer_path
            or Path(sys.modules[MHRLayer.__module__].__file__).resolve() != layer_path):
        raise ValueError("Native optimizer/decoder source path differs")
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"), checkpoint_path=assets / "model.ckpt",
        buffer_path=out / "never_use_unverified_buffers.pt", mhr_model_path=assets / "assets/mhr_model.pt", device="cuda")
    if layer.decoder_identity() != prepare["decoder_identity"]: raise ValueError("Prepared native decoder identity differs")
    from lib_mhr.collision_proxy import load_mhr_collision_proxy
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    human_faces = layer.mesh_faces(device="cuda").detach().cpu().numpy()
    proxy = load_mhr_collision_proxy(human_faces, asset_paths["mhr_collision_proxy_4000v.npz"])
    hand_spec = load_mhr_hand_surface_spec(asset_paths["mhr_hand_surface_spec.npz"], faces=human_faces)
    if proxy.vertex_count != 4000 or hand_spec.mhr_model_sha256 != hashes["assets/mhr_model.pt"]["sha256"]:
        raise ValueError("Refinement topology/model asset differs")
    source = torch.load(root / FORWARD / "coconet.pth", map_location="cpu", weights_only=False)
    mesh = root / PREPARE / "inputs/export/episode_000015/object_mesh/output_aligned.glb"
    validate_source_bundle(source, mesh)
    report["empty_observation_frames"] = {k: int(np.count_nonzero(~np.asarray(source["observations"][k]).reshape(FRAMES, -1).any(1)))
        for k in ("human_mask", "object_mask", "postopt_human_mask", "postopt_object_mask")}
    vertices, faces = optimizer._load_object_vertices(mesh)
    if faces is None: raise ValueError("Full original object triangles required")
    mesh_id = fingerprint({"vertices": vertices, "faces": faces})
    cfg = optimizer.MHRParityPostOptConfig(penetration_collision_proxy_path=str(asset_paths["mhr_collision_proxy_4000v.npz"]),
        hand_surface_spec_path=str(asset_paths["mhr_hand_surface_spec.npz"]), report_every=100)
    report.update(phase="native_full96_refinement", config=asdict(cfg), decoder_identity=layer.decoder_identity(),
                  object_mesh_sha256=inputs.sha256(mesh), source_bundle_sha256=inputs.sha256(root / FORWARD / "coconet.pth")); persist()
    result = invoke_optimizer(source, vertices, faces, cfg, layer, optimizer.run_postopt_smplh_parity, report, persist)
    torch.cuda.synchronize()
    if fingerprint({"vertices": vertices, "faces": faces}) != mesh_id: raise ValueError("Native mesh input changed")
    destination = out / "refined.pth"
    if destination.exists(): raise FileExistsError(destination)
    torch.save(result, destination, pickle_protocol=4); destination.chmod(0o444)
    saved = torch.load(destination, map_location="cpu", weights_only=False)
    if validate_result(source, saved) != report["metadata"] or fingerprint(saved) != fingerprint(result):
        raise ValueError("Stored full96 refined bundle changed")
    for path, row in frozen.items():
        if identity(path, immutable=False) != row: raise ValueError("Frozen input/asset changed during refinement")
    for path, row in config_ids.items():
        if identity(path) != row: raise ValueError("Bundled pin config changed")
    if ({k: identity(code / k) for k in helpers} != helpers or body._source_identity(root) != source_id
            or body._body_assets(root)[1] != hashes): raise ValueError("Source helper/native source changed")
    if any(report[k] != 1 for k in ("optimizer_attempts", "optimizer_returns", "optimizer_validated")):
        raise ValueError("One actual full96 native refinement required")
    report.update(status="pass", phase="complete", frames=FRAMES, source_inputs_assets_rehashed=True,
        saved_bundle_reloaded_verified=True, frozen_raw_inputs_byte_preserved=True, object_mesh_unchanged=True,
        bundle_sha256=inputs.sha256(destination), bundle_bytes=destination.stat().st_size,
        output_files={"refined.pth": identity(destination)}, native_refinement_verified=True,
        effective_optimizer_updates=report["metadata"]["effective_optimizer_updates"], source_frames=501,
        original_frame_indices=list(range(FRAMES)))
    validate_refinement_report(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--forward-pins", type=Path, required=True)
    parser.add_argument("--prepare-pins", type=Path, required=True)
    args = parser.parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"])
    revision, out = os.environ["WR_CODE_REVISION"], root / BASE
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or os.geteuid() != 1000
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.environ["WR_IMAGE_ID"] != IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or out.resolve() != out.absolute()
            or not out.is_dir() or any(out.iterdir())):
        raise ValueError("Fresh canonical source-bound offline Azure output required")
    report = dict(stage=STAGE, status="fail", phase="public_integrity", producer_revision=revision,
        script_sha256=inputs.sha256(Path(__file__)), image_id=IMAGE, input_track="track_1", network="none",
        ground_truth_used=False, ground_truth_read=False, private_truth_read=False, hand_labeled_test=False, oracle_modes=[],
        optimizer_attempts=0, optimizer_returns=0, optimizer_validated=0, requested_steps=300,
        effective_optimizer_updates=None, budget_seconds=BUDGET, learned_inference_calls=0,
        quality_verified=False, adoption_authorized=False, submission_produced=False, numerical_bit_determinism_claimed=False)
    path, started = out / "report.json", time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started
            stream.seek(0); json.dump(report, stream, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Unchanged native96 refinement exceeded1200s")
        old = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, out, code, args, report, persist); report["effective_optimizer_updates"] = report["metadata"]["effective_optimizer_updates"]
        except BaseException as error: report.update(status="fail", error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
