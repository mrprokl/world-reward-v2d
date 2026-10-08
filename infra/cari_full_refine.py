"""One unchanged native full-video parity refinement of the shared forward.

The full original timeline is optimized once with native defaults: 300 requested
steps / 301 updates, batch=0. Only body rotation controls and object translation
move; identity, hands, roots, camera and actual aligned mesh remain fixed. This
is execution/representation validation, not held-out accuracy or a submission.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from world_reward.artifact_paths import episode_relative, pin_path as artifact_pin_path
import platform
import re
import signal
import sys
import time

import numpy as np
import body_smoke as body
import cari_clip_inputs as inputs
import cari_refine as contract
from cari_converter import validate_native_bundle, require_aligned_object_metadata

STAGE = "world_reward_native_cari_shared_full_video_refinement"
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BUDGET = 7200
LAYER_SHA = "a753ab8e730b6730fca275384fab629859311983292a407390d88c66ffe68c23"
HISTORY_STEPS = (0, 100, 200, 300)
BODY_ASSET_RELATIVE = "weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3"


def output_relative(episode):
    if type(episode) is not int or not 0 <= episode < 30:
        raise ValueError("Explicit Track1 episode integer in 0..29 required")
    return episode_relative(episode) + "/cari_shared_refined_v1"


def parser():
    class OneEpisode(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if getattr(namespace, self.dest, None) is not None:
                parser.error("Exactly one explicit episode is required")
            setattr(namespace, self.dest, value)
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--episode", type=int, required=True, choices=range(30), action=OneEpisode)
    return result


def identity(path, immutable=True):
    path = inputs.public.regular(path)
    if immutable and path.stat().st_mode & 0o222:
        raise ValueError("Immutable regular producer/config/helper required")
    return inputs.identity(path)


def fingerprint(value):
    """Complete byte fingerprint, including signed zero and opaque-type refusal."""
    digest = hashlib.sha256()
    def visit(v):
        if np.ma.isMaskedArray(v): raise ValueError("Masked native source arrays forbidden")
        if hasattr(v, "detach") and hasattr(v, "cpu"): v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject: raise ValueError("Object native source arrays forbidden")
            digest.update(str((v.shape, v.dtype.str)).encode()); digest.update(v.tobytes(order="C"))
        elif isinstance(v, dict):
            if any(type(key) is not str for key in v): raise ValueError("String native mapping keys required")
            digest.update(b"dict")
            for key in sorted(v): digest.update(repr(key).encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(str(type(v)).encode())
            for item in v: visit(item)
        elif isinstance(v, (str, int, float, bool, type(None), np.generic)):
            digest.update(str(type(v)).encode()); digest.update(repr(v).encode())
        else: raise ValueError("Unsupported opaque native source value")
    visit(value)
    return digest.hexdigest()


def validate_source_bundle(bundle, mesh, count):
    """Count-explicit native ABI; retain real zero masks during occlusion."""
    params, pose = validate_native_bundle(bundle, count)
    require_aligned_object_metadata(bundle["metadata"])
    if (bundle.get("schema") != "cari4d.mhr_wild_inference.v1" or "postopt" in bundle
            or bundle["metadata"].get("object_mesh") != str(mesh)
            or bundle["metadata"].get("materialized_input_cache") is not None):
        raise ValueError("Fresh shared native forward with original aligned mesh required")
    if any(array.dtype != np.float32 for array in params.values()) or pose.dtype != np.float32:
        raise ValueError("Unchanged native float32 predictions required")
    for key in ("mhr_shape", "mhr_scale"):
        if any(row.tobytes() != params[key][0].tobytes() for row in params[key]):
            raise ValueError("Shared clip identity must remain byte-constant")
    if np.count_nonzero(params["mhr_face"]): raise ValueError("Expressions must remain zero")
    observations = bundle.get("observations", {})
    for key in ("human_mask", "object_mask", "postopt_human_mask", "postopt_object_mask"):
        value = observations.get(key); array = np.asarray(value)
        dtype = np.float32 if key.startswith("postopt_") else np.bool_
        if (np.ma.isMaskedArray(value) or array.ndim != 3 or array.shape[0] != count
                or min(array.shape[1:]) < 1 or array.dtype != dtype or not np.isfinite(array).all()
                or not np.isin(array, [0, 1]).all()
                or key.startswith("postopt_") and array.shape[1:] != (256, 256)):
            raise ValueError("Complete automatic native silhouette observations required")
    value = bundle["pr"].get("contact_logits"); contact = np.asarray(value)
    if (np.ma.isMaskedArray(value) or contact.shape != (count, 2)
            or contact.dtype != np.float32 or not np.isfinite(contact).all()):
        raise ValueError("Complete original native contact logits required")
    return params, pose


def validate_result(source, result, count):
    metadata = contract.validate_refined_bundle(source, result, count)
    if set(result) != set(source) | {"postopt"}:
        raise ValueError("Native refinement may only add its postopt record")
    for key in set(source) - {"pr"}:
        if fingerprint(source[key]) != fingerprint(result[key]):
            raise ValueError("Refinement altered frozen raw inputs, topology, camera or observations")
    if set(result["pr"]) != set(source["pr"]) | {"pose_abs_postopt"}:
        raise ValueError("Native refinement may only add its final object-pose copy")
    for key in set(source["pr"]) - {"mhr_body_pose_cont", "pose_abs"}:
        if fingerprint(source["pr"][key]) != fingerprint(result["pr"][key]):
            raise ValueError("Refinement altered a nonoptimized prediction, including contact")
    if fingerprint(result["pr"]["pose_abs_postopt"]) != fingerprint(result["pr"]["pose_abs"]):
        raise ValueError("Native final object-pose copies differ")
    history = result["postopt"]["history"]
    # The pinned optimizer writes float(step), not integer iteration records.
    # Preserve those original records; no post-hoc integer conversion/repair.
    if (len(history) != len(HISTORY_STEPS) or any(type(row.get("iter")) is not float for row in history)
            or [row["iter"] for row in history] != [float(step) for step in HISTORY_STEPS]):
        raise ValueError("All four native reporting steps 0/100/200/300 required")
    if any(type(row.get("batch_start")) is not float or row["batch_start"]!=0.
           or type(row.get("batch_size")) is not float or row["batch_size"]!=float(count) for row in history):
        raise ValueError("Every actual native reported update must use the entire original timeline")
    return metadata


def invoke_optimizer(source, vertices, faces, cfg, layer, callback, count, report, persist):
    before = fingerprint(source)
    report["optimizer_attempts"] += 1; persist()
    result = callback(source, vertices, faces, cfg, mhr_layer=layer)
    report["optimizer_returns"] += 1; persist()
    if fingerprint(source) != before: raise ValueError("Native optimizer modified its frozen input")
    if result.get("postopt", {}).get("config") != asdict(cfg):
        raise ValueError("Native optimizer altered its exact original configuration")
    report["metadata"] = validate_result(source, result, count)
    report["optimizer_validated"] += 1; persist()
    return result


def validate_refinement_report(report, spec):
    if type(spec) is not inputs.PublicClipSpec: raise ValueError("Explicit full public clip spec required")
    expected = dict(stage=STAGE, status="pass", phase="complete", image_id=IMAGE,
        episode_index=spec.episode_index, clip_spec=asdict(spec), frames=spec.total_frames,
        source_frames=spec.total_frames, original_frame_indices=list(range(spec.total_frames)),
        input_track="track_1", ground_truth_used=False, ground_truth_read=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], network="none", requested_steps=300,
        effective_optimizer_updates=301, optimizer_attempts=1, optimizer_returns=1, optimizer_validated=1,
        source_inputs_assets_rehashed=True, source_helpers_rehashed=True, saved_bundle_reloaded_verified=True,
        frozen_raw_inputs_byte_preserved=True, object_mesh_unchanged=True, native_refinement_verified=True,
        learned_inference_calls=0, quality_verified=False, adoption_authorized=False, submission_produced=False,
        numerical_bit_determinism_claimed=False, optimizer_sha256=contract.OPTIMIZER_SHA256,
        refinement_assets=contract.REFINEMENT_ASSETS, checkpoint_sha256=contract.CHECKPOINT_SHA256,
        budget_seconds=BUDGET, history_steps=list(HISTORY_STEPS))
    if type(report) is not dict or any(type(report.get(k)) is not type(v) or report[k] != v for k,v in expected.items()):
        raise ValueError("Exactly one unchanged complete full-timeline native refinement required")
    metadata = report.get("metadata", {})
    required_meta = dict(native_refinement_verified=True, full_original_frame_coverage_verified=True,
        frozen_parameters_bit_identical=True, requested_steps=300, effective_optimizer_updates=301, batch_size=0,
        optimized_parameters=contract.OPTIMIZED_PARAMETERS, fixed_parameters=contract.FIXED_PARAMETERS)
    if any(type(metadata.get(k)) is not type(v) or metadata[k] != v for k,v in required_meta.items()):
        raise ValueError("Native fixed-parameter/full-batch/update proof required")
    for key in ("forward_report_sha256", "prepare_report_sha256", "input_report_sha256", "source_bundle_sha256",
                "bundle_sha256", "object_mesh_sha256", "input_sha256"):
        if type(report.get(key)) is not str or not re.fullmatch(r"[0-9a-f]{64}", report[key]):
            raise ValueError("Complete exact input/output/source SHA chain required")
    if (type(report.get("bundle_bytes")) is not int or report["bundle_bytes"] <= 0
            or report.get("output_files") != {"refined.pth": {"sha256":report["bundle_sha256"], "bytes":report["bundle_bytes"]}}):
        raise ValueError("Exactly one frozen refined output bundle required")


def source_helpers(code):
    names = ("infra/cari_full_refine.py", "infra/run_cari_full_refine.sh", "infra/cari_full_forward.py",
             "infra/run_cari_full_forward.sh", "infra/cari_refine.py", "infra/cari_converter.py",
             "infra/cari_clip_inputs.py", "src/world_reward/artifact_paths.py", "infra/cari96_inputs.py", "infra/body_smoke.py")
    return {name:identity(code/name) for name in names}


def pinned_refined_report(root, spec, pins):
    """Verify exact stage inventory before interpreting its frozen report."""
    if type(spec) is not inputs.PublicClipSpec: raise ValueError("Explicit full public clip spec required")
    if (type(pins) is not dict or set(pins) != {"schema", "clip_spec", "refined", "refined_files"}
            or pins["schema"] != "world-reward-cari-shared-refined-pins-v1"
            or pins["clip_spec"] != asdict(spec) or type(pins["clip_spec"]) is not dict):
        raise ValueError("Explicit immutable full-timeline refined pins required")
    if inputs.PublicClipSpec(**pins["clip_spec"]) != spec:
        raise ValueError("Refined clip spec differs")
    proof, files = pins["refined"], pins["refined_files"]
    inputs._receipt(proof, producer=True)
    if type(files) is not dict or set(files) != {"report.json", "refined.pth"}:
        raise ValueError("Exactly two frozen refined producer files required")
    if files["report.json"] != {key:proof[key] for key in ("sha256", "bytes")}:
        raise ValueError("Refined producer proof differs from the inventory")
    directory = root/output_relative(spec.episode_index)
    paths=tuple(directory.iterdir())
    observed={path.name for path in paths}
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Producer symlink, subdirectory or special file forbidden")
    if observed != set(files): raise ValueError("Complete exclusive refined inventory required")
    frozen = {}
    for name,row in files.items():
        inputs._receipt(row)
        if identity(directory/name) != row: raise ValueError("Frozen refined artifact changed")
        frozen[directory/name] = row
    report = json.loads((directory/"report.json").read_text())
    if any(report.get(key) != proof[key] for key in ("producer_revision", "script_sha256")):
        raise ValueError("Refined report/source producer differs")
    validate_refinement_report(report,spec)
    if report["output_files"] != {"refined.pth":files["refined.pth"]}:
        raise ValueError("Refined saved-bundle receipt differs")
    return dict(directory=directory,report=report,files=dict(files),bindings=frozen)


def verify_refined_artifacts(root, code, spec, pins, source_code=None):
    """Hash/JSON-only full-chain gate; current code owns consumer pins.

    Explicit historical source_code must be independently authenticated by the
    caller. It is read as readonly provenance only, never imported or executed.
    Default behavior remains exact current-code equivalence.
    """
    import cari_full_forward as forward
    selected = pinned_refined_report(root,spec,pins)
    fp = artifact_pin_path(code, spec.episode_index, "shared_forward")
    preceding=(forward.verify_forward_artifacts(root,code,spec,json.loads(identity_and_read(fp)))if source_code is None else
        forward.verify_forward_artifacts(root,code,spec,json.loads(identity_and_read(fp)),source_code=source_code))
    source_code=code if source_code is None else source_code
    prepared = preceding["prepare"]
    report,fr,pr = selected["report"],preceding["report"],prepared["report"]
    if (report.get("forward_report_sha256") != identity(preceding["directory"]/"report.json")["sha256"]
            or report.get("prepare_report_sha256") != identity(prepared["directory"]/"report.json")["sha256"]
            or report.get("source_bundle_sha256") != identity(preceding["directory"]/"coconet.pth")["sha256"]
            or report.get("input_report_sha256") != pr.get("input_report_sha256")
            or report.get("input_sha256") != pr.get("input_sha256")
            or report.get("input_dataset_revision") != inputs.DATASET_REVISION
            or report.get("source_public_files") != pr.get("source_files")
            or report.get("original_input_paths") != pr.get("original_input_paths")
            or report.get("body_assets") != pr.get("body_assets") or report.get("body_assets") != fr.get("body_assets")
            or report.get("inference_source_identity") != pr.get("inference_source_identity")
            or report.get("inference_source_identity") != fr.get("inference_source_identity")
            or report.get("decoder_identity") != pr.get("decoder_identity")):
        raise ValueError("Exact shared refined/forward/prepare/public/model lineage differs")
    mesh=str(root/inputs.relative_paths(spec)["mesh"])
    if report.get("object_mesh_sha256")!=pr["source_files"][str(Path(mesh).relative_to(root))]["sha256"]:
        raise ValueError("Refinement must preserve the exact original aligned mesh")
    helpers = source_helpers(source_code)
    if (report.get("source_helpers") != helpers
            or report.get("script_sha256") != helpers["infra/cari_full_refine.py"]["sha256"]):
        raise ValueError("Refinement helper source changed")
    ownpin=artifact_pin_path(code, spec.episode_index, "shared_refined")
    if json.loads(identity_and_read(ownpin))!=pins:raise ValueError("Explicit refined pins differ from bundled config")
    frozen = {Path(path):row for path,row in preceding["bindings"].items()};frozen.update(selected["bindings"])
    native=root/"vendor/video_to_data/reconstruction/modules/v2d_cari4d/lib/cari4d"
    consumed={native/contract.OPTIMIZER_RELATIVE_PATH,native/"lib_mhr/mhr_layer.py",
        root/"results/cari-refinement-assets.json",root/"results/weights-acquisition.json",
        *(root/"weights/cari4d/refinement"/name for name in contract.REFINEMENT_ASSETS),
        *(root/BODY_ASSET_RELATIVE/name for name in pr["body_assets"])}
    bound=report.get("refinement_bindings")
    if type(bound) is not dict or set(bound)!={str(path) for path in consumed}:
        raise ValueError("Complete exact consumed refinement source/model/asset binding required")
    for path in consumed:
        row=bound[str(path)];inputs._receipt(row)
        if identity(path,immutable=False)!=row:raise ValueError("Consumed native refinement asset/source changed")
        frozen[path]=row
    if (bound[str(native/contract.OPTIMIZER_RELATIVE_PATH)]["sha256"]!=contract.OPTIMIZER_SHA256
            or bound[str(native/"lib_mhr/mhr_layer.py")]["sha256"]!=LAYER_SHA
            or any(bound[str(root/BODY_ASSET_RELATIVE/name)]!=row for name,row in pr["body_assets"].items())
            or any(bound[str(root/"weights/cari4d/refinement"/name)]!={k:row[k] for k in ("sha256","bytes")}
                   for name,row in contract.REFINEMENT_ASSETS.items())):
        raise ValueError("Consumed native source/model/refinement asset identity differs from pinned constants")
    contract.require_asset_receipt(json.loads((root/"results/cari-refinement-assets.json").read_text()))
    for path in (fp,ownpin):
        frozen[path]=identity(path)
    frozen.update({source_code/name:row for name,row in helpers.items()})
    selected.update(bindings=frozen,forward=preceding,prepare=prepared)
    return selected


def identity_and_read(path):
    identity(path)
    return path.read_text()


def run(root, out, code, episode, report, persist):
    import cari_full_forward as forward
    pin = artifact_pin_path(code, episode, "shared_forward")
    pin_id=identity(pin);pins=json.loads(pin.read_text());spec=inputs.PublicClipSpec(**pins["clip_spec"])
    if spec.episode_index!=episode:raise ValueError("Explicit episode differs from pinned full clip")
    selected=forward.verify_forward_artifacts(root,code,spec,pins)
    fr,pr=selected["report"],selected["prepare"]["report"]
    frozen={Path(path):row for path,row in selected["bindings"].items()};frozen[pin]=pin_id
    mesh=root/inputs.relative_paths(spec)["mesh"]
    if identity(mesh,immutable=False)!=pr["source_files"][str(mesh.relative_to(root))]:
        raise ValueError("Exact original aligned object mesh required")
    vendor=root/"vendor/video_to_data";body._pinned_checkout(vendor,body.UPSTREAM_REVISION)
    assets,hashes=body._body_assets(root);source_id=body._source_identity(root)
    if (hashes!=pr.get("body_assets") or hashes!=fr.get("body_assets")
            or source_id!=pr.get("inference_source_identity") or source_id!=fr.get("inference_source_identity")
            or assets!=root/BODY_ASSET_RELATIVE or (assets/"mhr_buffers.pt").exists()):
        raise ValueError("Unchanged original native model/source assets required")
    native=vendor/"reconstruction/modules/v2d_cari4d/lib/cari4d"
    optimizer_path=native/contract.OPTIMIZER_RELATIVE_PATH;layer_path=native/"lib_mhr/mhr_layer.py"
    for path,digest in ((optimizer_path,contract.OPTIMIZER_SHA256),(layer_path,LAYER_SHA)):
        row=identity(path,immutable=False)
        if row["sha256"]!=digest:raise ValueError("Pinned unchanged optimizer/native decoder required")
        frozen[path]=row
    asset_receipt=root/"results/cari-refinement-assets.json";frozen[asset_receipt]=identity(asset_receipt,immutable=False)
    contract.require_asset_receipt(json.loads(asset_receipt.read_text()))
    asset_paths={}
    for name,expected in contract.REFINEMENT_ASSETS.items():
        path=root/"weights/cari4d/refinement"/name;row=identity(path,immutable=False)
        if row!={key:expected[key] for key in ("sha256","bytes")}:
            raise ValueError("Exact pinned collision/hand asset required")
        asset_paths[name]=path;frozen[path]=row
    frozen.update({assets/name:row for name,row in hashes.items()})
    frozen[root/"results/weights-acquisition.json"]=identity(root/"results/weights-acquisition.json",immutable=False)
    helpers=source_helpers(code)
    consumed={optimizer_path,layer_path,asset_receipt,root/"results/weights-acquisition.json",
        *asset_paths.values(),*(assets/name for name in hashes)}
    report.update(phase="source_and_asset_audit",episode_index=episode,clip_spec=asdict(spec),frames=spec.total_frames,
        source_frames=spec.total_frames,original_frame_indices=list(range(spec.total_frames)),
        forward_report_sha256=identity(selected["directory"]/"report.json")["sha256"],
        prepare_report_sha256=identity(selected["prepare"]["directory"]/"report.json")["sha256"],
        input_report_sha256=pr["input_report_sha256"],input_sha256=pr["input_sha256"],input_dataset_revision=inputs.DATASET_REVISION,
        original_input_paths=pr["original_input_paths"],source_public_files=pr["source_files"],source_helpers=helpers,
        body_assets=hashes,inference_source_identity=source_id,optimizer_sha256=contract.OPTIMIZER_SHA256,
        refinement_assets=contract.REFINEMENT_ASSETS,checkpoint_sha256=contract.CHECKPOINT_SHA256,
        refinement_bindings={str(path):frozen[path] for path in consumed});persist()
    if "torch" in sys.modules:raise ValueError("Fresh native runtime required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__)!="2.5.1+cu124" or torch.version.cuda!="12.4":
        raise ValueError("Pinned native H100 CUDA runtime required")
    if torch.are_deterministic_algorithms_enabled():raise ValueError("No new strict backward policy may be imposed")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report["runtime"]=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=platform.python_version(),
        numpy=np.__version__,deterministic_algorithms=False,tf32=False,seed=0,threads=4)
    sys.path[:0]=[str(native),"/workspace/v2d_sam3d_body/lib"]
    os.environ.update(MHR_ASSETS_ROOT=str(root/"weights/cari4d/sam3d_body"),MOMENTUM_ENABLED="0")
    from learning.training import mhr_opt_refineout as optimizer
    from lib_mhr.mhr_layer import MHRLayer
    if (Path(optimizer.__file__).resolve()!=optimizer_path
            or Path(sys.modules[MHRLayer.__module__].__file__).resolve()!=layer_path):
        raise ValueError("Actual pinned optimizer/native decoder import required")
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=Path("/workspace/v2d_sam3d_body/lib"),checkpoint_path=assets/"model.ckpt",
        buffer_path=out/"never_use_unverified_buffer.pt",mhr_model_path=assets/"assets/mhr_model.pt",device="cuda")
    if layer.decoder_identity()!=pr["decoder_identity"]:raise ValueError("Prepared native decoder identity differs")
    from lib_mhr.collision_proxy import load_mhr_collision_proxy
    from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
    human_faces=layer.mesh_faces(device="cuda").detach().cpu().numpy()
    proxy=load_mhr_collision_proxy(human_faces,asset_paths["mhr_collision_proxy_4000v.npz"])
    hand_spec=load_mhr_hand_surface_spec(asset_paths["mhr_hand_surface_spec.npz"],faces=human_faces)
    if proxy.vertex_count!=4000 or hand_spec.mhr_model_sha256!=hashes["assets/mhr_model.pt"]["sha256"]:
        raise ValueError("Native full collision/hand topology/model binding required")
    source=torch.load(selected["directory"]/"coconet.pth",map_location="cpu",weights_only=False)
    validate_source_bundle(source,mesh,spec.total_frames)
    report["empty_observation_frames"]={key:int(np.count_nonzero(~np.asarray(source["observations"][key]).reshape(spec.total_frames,-1).any(1)))
        for key in ("human_mask","object_mask","postopt_human_mask","postopt_object_mask")}
    vertices,faces=optimizer._load_object_vertices(mesh)
    if faces is None:raise ValueError("All actual original object triangles required")
    mesh_id=fingerprint(dict(vertices=vertices,faces=faces))
    cfg=optimizer.MHRParityPostOptConfig(penetration_collision_proxy_path=str(asset_paths["mhr_collision_proxy_4000v.npz"]),
        hand_surface_spec_path=str(asset_paths["mhr_hand_surface_spec.npz"]),report_every=100)
    report.update(phase="native_full_video_refinement",config=asdict(cfg),decoder_identity=layer.decoder_identity(),
        object_mesh_sha256=identity(mesh,immutable=False)["sha256"],
        source_bundle_sha256=identity(selected["directory"]/"coconet.pth")["sha256"]);persist()
    result=invoke_optimizer(source,vertices,faces,cfg,layer,optimizer.run_postopt_smplh_parity,spec.total_frames,report,persist)
    torch.cuda.synchronize()
    if fingerprint(dict(vertices=vertices,faces=faces))!=mesh_id:raise ValueError("Native object mesh inputs changed")
    destination=out/"refined.pth"
    if destination.exists():raise FileExistsError(destination)
    torch.save(result,destination,pickle_protocol=4);destination.chmod(0o444)
    saved=torch.load(destination,map_location="cpu",weights_only=False)
    if validate_result(source,saved,spec.total_frames)!=report["metadata"] or fingerprint(saved)!=fingerprint(result):
        raise ValueError("Stored full-timeline native refined bundle changed")
    if (any(identity(path,immutable=False)!=row for path,row in frozen.items()) or source_helpers(code)!=helpers
            or body._source_identity(root)!=source_id or body._body_assets(root)[1]!=hashes
            or {path.name for path in out.iterdir()}!={"report.json","refined.pth"}):
        raise ValueError("Frozen original inputs, native assets, configs, helpers or output inventory changed")
    report.update(status="pass",phase="complete",source_inputs_assets_rehashed=True,source_helpers_rehashed=True,
        saved_bundle_reloaded_verified=True,frozen_raw_inputs_byte_preserved=True,object_mesh_unchanged=True,
        bundle_sha256=identity(destination)["sha256"],bundle_bytes=destination.stat().st_size,
        output_files={"refined.pth":identity(destination)},native_refinement_verified=True,
        effective_optimizer_updates=report["metadata"]["effective_optimizer_updates"],history_steps=list(HISTORY_STEPS))
    validate_refinement_report(report,spec)


def main(argv=None):
    args=parser().parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    out=root/output_relative(args.episode)
    if (platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or os.geteuid()!=1000
            or {path.name for path in Path("/sys/class/net").iterdir()}!={"lo"} or os.environ["WR_IMAGE_ID"]!=IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}",revision) or root.resolve()!=root.absolute() or code.resolve()!=code.absolute()
            or not out.is_dir() or any(out.iterdir()) or out.resolve()!=out.absolute()):
        raise ValueError("Fresh source-bound offline Azure full refinement output required")
    report=dict(stage=STAGE,status="fail",phase="public_input_audit",episode_index=args.episode,producer_revision=revision,
        script_sha256=inputs.public.sha256(Path(__file__)),image_id=IMAGE,input_track="track_1",network="none",
        ground_truth_used=False,ground_truth_read=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],
        optimizer_attempts=0,optimizer_returns=0,optimizer_validated=0,requested_steps=300,effective_optimizer_updates=None,
        budget_seconds=BUDGET,learned_inference_calls=0,quality_verified=False,adoption_authorized=False,
        submission_produced=False,numerical_bit_determinism_claimed=False)
    receipt=out/"report.json";started=time.perf_counter()
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Unchanged full native refinement exceeded7200s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,out,code,args.episode,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o444)


if __name__=="__main__":main()
