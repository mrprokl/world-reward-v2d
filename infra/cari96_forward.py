"""One constrained96 native CoCoNet window; no refinement, quality or inverse fit.

The unchanged native callable runs in ordinary CUDA/AMP. Only its composition
lookup is isolated; raw network deltas and the original globals remain intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import body_smoke as body
import cari96_inputs as inputs
import cari96_native as constrained
from cari_runner import build_cari_runtime_environment, CHECKPOINT_SHA256, CHECKPOINT_REVISION, CHECKPOINT_RELATIVE_PATH, CONFIG_RELATIVE_PATH
from world_reward.contracts import require_rigid_transforms

BASE = "validation/cari96_forward_v3"
PREPARE = "validation/cari96_public_v1"
STAGE, BUDGET = "public_cari96_constrained_coconet_forward", 360
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_HASHES = {"dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
               "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"}
SOURCES = {constrained.NATIVE_RELATIVE_PATH: constrained.NATIVE_SOURCE_SHA256,
    CONFIG_RELATIVE_PATH: "98846f0cbf556ec93460d01ed8bf85f5f5fc7645b1eadc8357ab84616bf54b87",
    constrained.COMPOSITION_RELATIVE_PATH: "af81329513074fac404676358775b0c1310fac9d522b23d1ce03d9f2de2f451b",
    "lib_mhr/mhr_layer.py": "a753ab8e730b6730fca275384fab629859311983292a407390d88c66ffe68c23",
    "lib_mhr/delta.py": "da495861dc55b266dabaf74a06fc683b74d2354f557827914cfc926b0daa2518",
    "learning/inference.py": "1d4fb02b3bf8345eee2510c96c4f2edcc7f7d17ae298fc7d23a392c56ebace9f",
    "lib_mhr/mhr_supervision.py": "e7f4c5f8991e312eec69a760204758d44edaf9f29f9456ec181d288241686635",
    "learning/training/training_config.py": "d2acefefeea13f4257f8810865fd492079abdca30e7d9198c95eb0b7037fe4cf"}
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
CAMERA = "front_stereo_camera_left"
EXPORT = "inputs/export/episode_000015"
PREPARE_FILES = {"report.json", "shared_initializer.pkl", "direct_parameters.npz", "target.npy", "inputs/manifest.json",
    "inputs/aligned_depth.h5", "inputs/own_object_poses.pkl", EXPORT + "/edex", EXPORT + "/wild_export.json",
    EXPORT + "/object_mesh/output_aligned.glb", *(EXPORT + f"/{kind}/{CAMERA}.h5" for kind in ("images", "human_masks", "object_masks"))}


def identity(path, *, immutable=False):
    path = inputs.regular(path)
    if immutable and path.stat().st_mode & 0o222: raise ValueError("Frozen producer file must be readonly")
    return inputs.identity(path)


def _receipt(value, *, producer=False):
    keys = {"sha256", "bytes"} | ({"producer_revision", "script_sha256"} if producer else set())
    if (type(value) is not dict or set(value) != keys or type(value["bytes"]) is not int or value["bytes"] <= 0
            or type(value["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Exact SHA/size identity required")
    if producer and (type(value["producer_revision"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", value["producer_revision"])
            or type(value["script_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["script_sha256"])):
        raise ValueError("Exact producer/source identity required")


def validate_prepare_pins(pins):
    if (type(pins) is not dict or set(pins) != {"schema", "prepare", "prepare_files"}
            or pins["schema"] != "world-reward-cari96-prepare-pins-v1"
            or type(pins["prepare_files"]) is not dict or set(pins["prepare_files"]) != PREPARE_FILES):
        raise ValueError("Complete explicit96 preparation pins required")
    _receipt(pins["prepare"], producer=True)
    for value in pins["prepare_files"].values(): _receipt(value)
    if pins["prepare_files"]["report.json"] != {k:pins["prepare"][k] for k in ("sha256", "bytes")}:
        raise ValueError("Preparation report pin differs")


def prepared_inputs(root, pins):
    validate_prepare_pins(pins); directory = root / PREPARE
    observed = {str(p.relative_to(directory)): identity(p, immutable=True) for p in directory.rglob("*") if p.is_file()}
    if observed != pins["prepare_files"]: raise ValueError("Complete frozen preparation inventory differs")
    report = json.loads((directory / "report.json").read_text())
    expected = dict(stage="public_cari96_shared_initializer_native_abi", status="pass", phase="complete", frames=96,
        image_id=IMAGE, input_track="track_1", ground_truth_used=False, private_truth_read=False, hand_labeled_test=False,
        oracle_modes=[], network="none", quality_verified=False, adoption_authorized=False, submission_produced=False,
        shared_identity_verified=True, original_initializer_unchanged=True, source_inputs_assets_rehashed=True,
        stored_initializer_native_replay_verified=True, learned_inference_calls=0, optimizer_calls=0,
        native_geometry_calls=6, native_direct_calls=6, native_replay_calls=6, reference_calls=6,
        native_geometry_attempts=6,native_direct_attempts=6,native_replay_attempts=6,reference_attempts=6,
        source_helpers_rehashed=True,frozen_outputs_rehashed_after_reference=True)
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value: raise ValueError("Complete passing native preparation required: " + key)
    if any(report.get(k) != pins["prepare"][k] for k in ("producer_revision", "script_sha256")):
        raise ValueError("Preparation producer identity differs")
    if report.get("output_files") != {p:v for p,v in observed.items() if p != "report.json"}: raise ValueError("All frozen preparation outputs must match generator receipt")
    helpers = report.get("source_helpers")
    expected_helpers = {"infra/cari96_prepare.py","infra/cari96_inputs.py","infra/body_smoke.py","infra/run_cari96_prepare.sh"}
    code = Path(__file__).parents[1]
    if type(helpers) is not dict or set(helpers) != expected_helpers or {p:identity(code/p,immutable=True) for p in helpers} != helpers:
        raise ValueError("Exact unchanged preparation generator helper inventory required")
    bindings = report.get("source_bindings")
    if type(bindings) is not list or len(bindings) != 3 or any(type(x) is not dict or set(x) != {"path","sha256","bytes"} for x in bindings):
        raise ValueError("Original native/reference generation source identities required")
    if [x["sha256"] for x in bindings] != [SOURCES["lib_mhr/mhr_layer.py"],"c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e","352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"]:
        raise ValueError("Native/reference generation source differs")
    links = {"shared_initializer_sha256":"shared_initializer.pkl", "direct_parameters_sha256":"direct_parameters.npz",
             "target_sha256":"target.npy", "snapshot_manifest_sha256":"inputs/manifest.json"}
    if any(report.get(k) != observed[p]["sha256"] for k,p in links.items()): raise ValueError("Preparation artifact chain differs")
    errors = report.get("reference_per_frame_mean_mm")
    if type(errors) is not list or len(errors) != 96 or any(type(x) not in (int,float) or not np.isfinite(x) or not 0 <= x <= 2 for x in errors):
        raise ValueError("All96 unchanged reference-fidelity gates required")
    manifest = json.loads((directory / "inputs/manifest.json").read_text())
    expected_snapshot = dict(stage="world_reward_public_cari96_inputs_snapshot", status="pass", frames=96, source_frames=501,
        source_episode_index=15, original_frame_indices=list(range(96)), ground_truth_used=False, ground_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], no_padding_or_reencoding=True, initializer_modified=False,
        camera_name=CAMERA, stored_payload_and_attribute_reread_verified=True)
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k,v in expected_snapshot.items()):
        raise ValueError("Exact96 original-frame snapshot required")
    original_pins_path = code/"configs/cari96_input_pins.json"
    source_pins = json.loads(original_pins_path.read_text()); inputs.validate_pins(source_pins)
    if report.get("input_pins") != identity(original_pins_path,immutable=True) or manifest.get("source_files") != source_pins["source_files"] or manifest.get("source_inputs_report_sha256") != source_pins["source_files"][inputs.REPORT]["sha256"]:
        raise ValueError("Exact501 original-input source lineage differs")
    output_files = {p.removeprefix("inputs/"):v for p,v in observed.items() if p.startswith("inputs/") and p != "inputs/manifest.json"}
    if manifest.get("output_files") != output_files: raise ValueError("Snapshot file inventory differs")
    import joblib
    initializer = joblib.load(directory / "shared_initializer.pkl")
    validate_initializer(initializer)
    object_poses = joblib.load(directory / "inputs/own_object_poses.pkl")
    if object_poses.get("frames") != initializer["frames"]: raise ValueError("Object frame coverage differs")
    pose = _array(object_poses.get("obj_pose_world"), (96,4,4)); require_rigid_transforms(pose,96)
    return directory, report, initializer, object_poses, observed


def _array(value, shape):
    if type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Exact finite complete native FP32 array required")
    return value


def validate_initializer(value):
    if type(value) is not dict or value.get("body_model") != "mhr" or value.get("frames") != [f"{i:06d}" for i in range(96)] or value.get("kids") != [0]:
        raise ValueError("Complete stored96 initializer required")
    for key,dim in inputs.PARAMETER_DIMS.items(): _array(value.get(key),(96,dim))
    root_basis(value["mhr_global_rot6d"])
    for key,dim in (("mhr_joints",127),("mhr_keypoints",70)): _array(value.get(key),(96,dim,3))
    for key in constrained.IDENTITY_DIMS:
        if any(row.tobytes() != value[key][0].tobytes() for row in value[key]): raise ValueError("Shared identity must precede native cache/render construction")
    if np.any(value["mhr_trans"][:,2] <= 0) or np.any(value["mhr_face"]): raise ValueError("Original positive translation/zero expression required")
    expected = dict(human_identity_clip_constant=True, original_geometry_recovered=False, source_frames=501,
                    original_frame_indices=list(range(96)), ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    metadata = value.get("metadata",{})
    if any(type(metadata.get(k)) is not type(v) or metadata[k] != v for k,v in expected.items()): raise ValueError("Explicit constrained initializer provenance required")


def root_basis(value):
    basis = value.astype(np.float64).reshape(96,3,2)
    first = basis[:,:,0]; norm = np.linalg.norm(first,axis=1)
    if np.any(norm <= 1e-8): raise ValueError("Degenerate original native root basis")
    first = first / norm[:,None]
    second = basis[:,:,1] - np.sum(first*basis[:,:,1],axis=1)[:,None]*first
    if np.any(np.linalg.norm(second,axis=1) <= 1e-8): raise ValueError("Degenerate original native root basis")


def validate_bundle(bundle, initializer, object_poses, hook):
    if (type(bundle) is not dict or bundle.get("schema") != "cari4d.mhr_wild_inference.v1" or bundle.get("gt") != {}
            or bundle.get("frames") != initializer["frames"] or bundle.get("kid") != 0
            or bundle.get("frame_meta") != [{"frame":name,"src_frame":i,"kid":0} for i,name in enumerate(initializer["frames"])]):
        raise ValueError("Exact native96 no-GT bundle/timeline required")
    metadata = bundle.get("metadata", {})
    expected = dict(ground_truth_used=False, window_length=96, window_stride=96, overlap_policy="first_occurrence",
        materialized_input_cache=None, materialized_input_cache_identity=None, depth_backend="moge2")
    if any(type(metadata.get(k)) is not type(v) or metadata[k] != v for k,v in expected.items()): raise ValueError("Native one-window/no-cache metadata differs")
    if bundle.get("checkpoint",{}).get("step") != 200000: raise ValueError("Original step200000 checkpoint required")
    cfg = bundle.get("config", {})
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k,v in dict(clip_len=96,enable_amp=True,pred_mhr_shape=True,pred_mhr_scale=False,body_model="mhr",mhr_joint_supervision_mode="body12_freeze_hand_face_all_losses").items()):
        raise ValueError("Unchanged commercial native model configuration required")
    for prefix in ("pr","pr_initial","in"):
        params = bundle.get(prefix,{})
        for key,dim in inputs.PARAMETER_DIMS.items(): _array(params.get(key),(96,dim))
        root_basis(params["mhr_global_rot6d"])
        require_rigid_transforms(_array(params.get("pose_abs"),(96,4,4)),96)
    if bundle["pr_initial"]["pose_abs"].tobytes() != bundle["pr"]["pose_abs"].tobytes() or bundle["pr"]["pose_abs_1st_delta"].tobytes() != bundle["pr"]["pose_abs"].tobytes(): raise ValueError("Native object-pose copies changed")
    for key in inputs.PARAMETER_DIMS:
        if bundle["in"][key].tobytes() != initializer[key].tobytes() or bundle["pr_initial"][key].tobytes() != bundle["pr"][key].tobytes():
            raise ValueError("Native input/initial-prediction blocks changed")
    for key in constrained.IDENTITY_DIMS:
        if bundle["pr"][key].tobytes() != initializer[key].tobytes(): raise ValueError("Constrained identity changed")
    if bundle["in"]["pose_abs"].tobytes() != object_poses["obj_pose_world"].tobytes(): raise ValueError("Inferred object trajectory/basis changed")
    faces = bundle.get("faces")
    if (type(faces) is not np.ndarray or faces.dtype != np.int32 or faces.shape != (36874,3)
            or hashlib.sha256(faces.astype("<i4",copy=False).tobytes()).hexdigest() != FACE_SHA):
        raise ValueError("Actual native topology differs")
    neutral = _array(bundle.get("mhr_neutral_height_init"),(96,))
    spatial = _array(bundle.get("mhr_spatial_scale"),(96,))
    diameter = _array(bundle.get("mesh_diameter"),(96,))
    if (np.any(neutral <= 0) or np.any(spatial <= 0) or np.any(diameter <= 0)
            or not np.allclose(neutral*spatial,2.,atol=1e-6,rtol=1e-6)
            or any(row.tobytes()!=neutral[0].tobytes() for row in neutral)):
        raise ValueError("Native shared neutral-height/spatial normalization differs")
    K = _array(bundle.get("K_rois"),(96,3,3));boxes = _array(bundle.get("bboxes"),(96,4))
    if np.any(K[:,:2,:2].diagonal(axis1=1,axis2=2) <= 0) or not np.array_equal(K[:,2],np.broadcast_to(np.array([0,0,1],np.float32),(96,3))) or np.any(boxes[:,2:] <= boxes[:,:2]):
        raise ValueError("Native camera crop matrices/bboxes invalid")
    contact = _array(bundle["pr"].get("contact_logits"),(96,2))
    if not np.array_equal(contact,bundle["pr_initial"].get("contact_logits")): raise ValueError("Native contact logits changed")
    observations = bundle.get("observations",{})
    for key in ("human_mask","object_mask"):
        a = observations.get(key)
        if type(a) is not np.ndarray or a.dtype != np.bool_ or a.shape != (96,224,224): raise ValueError("Native observed mask coverage differs")
    fullK = _array(observations.get("K_full"),(96,3,3))
    expectedK = np.array([[1920,0,768],[0,1920,576],[0,0,1]],np.float32)
    if not np.array_equal(fullK,np.broadcast_to(expectedK,(96,3,3))): raise ValueError("Original full camera changed")
    _array(observations.get("postopt_K_rois"),(96,3,3))
    if observations.get("postopt_crop_contract") != "cari4d.smplh_postopt_full_resolution_crop.v1": raise ValueError("Native full-resolution postopt crop contract differs")
    for key in ("postopt_human_mask","postopt_object_mask"):
        a = observations.get(key)
        if type(a) is not np.ndarray or a.dtype != np.float32 or a.shape != (96,256,256) or not np.isfinite(a).all() or not np.isin(a,[0,1]).all():
            raise ValueError("Native postopt masks invalid")
    raw = bundle.get("raw",{})
    if type(raw) is not dict or not {"rot","trans","delta_mhr_shape"}.issubset(raw) or "delta_mhr_scale" in raw:
        raise ValueError("Original commercial raw network outputs required")
    for key,value in raw.items():
        if key not in {"rot","trans"} and not key.startswith("delta_mhr_"): raise ValueError("Unexpected native raw key")
        if type(value) is not np.ndarray or value.dtype != np.float32 or value.ndim != 2 or value.shape[0] != 96 or not np.isfinite(value).all():
            raise ValueError("Complete unmodified native raw FP32 output required")
    validate_frozen_supervision(bundle,initializer)
    if any(type(hook.get(k)) is not int or hook[k] != 1 for k in constrained.COUNTERS): raise ValueError("Exactly one native composition/delegate/verification required")
    if any(hook.get(k) is not True for k in ("raw_prediction_bytes_preserved","initializer_bytes_preserved","native_compose_global_unchanged","native_function_code_unchanged")):
        raise ValueError("Native raw/global preservation evidence required")
    return {k:{"shape":list(v.shape),"sha256":hashlib.sha256(v.tobytes()).hexdigest()} for k,v in raw.items()}


def validate_frozen_supervision(bundle, initializer):
    """Exact upstream FP32 init+zero operation, not signed-zero restoration.

    The pinned supervision layer zeros hand deltas; face has no head. Native
    composition nevertheless adds positive zero to both blocks. -0 -> +0 is
    an IEEE representation change, not a changed expression or hand pose.
    Verify that exact operation and retain the untouched native output bytes.
    """
    raw = bundle["raw"]
    for key,dimension in (("mhr_hand",108),("mhr_face",72)):
        delta_key = "delta_"+key
        if key == "mhr_hand" and delta_key not in raw:
            raise ValueError("Original frozen native hand delta must be present")
        if delta_key in raw:
            delta = _array(raw[delta_key],(96,dimension))
            if delta.tobytes() != np.zeros_like(delta).tobytes():
                raise ValueError("Native frozen hand/face delta differs from exact zeros_like")
        expected = initializer[key]+np.zeros_like(initializer[key])
        actual = _array(bundle["pr"][key],(96,dimension))
        if (actual.tobytes() != expected.tobytes() or not np.array_equal(actual,initializer[key])):
            raise ValueError("Native frozen hand/face composition differs from exact FP32 init+zero")


def config_receipt(cfg):
    """Log active inference settings only; hash untouched training metadata.

    The native config includes an unused Infinity gradient-clip default. Keep
    it unchanged in the source/bundle, but do not emit nonstandard JSON numbers
    or rewrite it to fit our finite receipt format.
    """
    keys = ("clip_len","enable_amp","pred_mhr_shape","pred_mhr_scale","body_model","mhr_joint_supervision_mode")
    active = {key:cfg[key] for key in keys}
    json.dumps(active,allow_nan=False)
    digest = hashlib.sha256(json.dumps(cfg,sort_keys=True,allow_nan=True).encode()).hexdigest()
    return active,digest


def validate_forward_report(report):
    expected = dict(stage=STAGE,status="pass",phase="complete",frames=96,source_frames=501,original_frame_indices=list(range(96)),
        input_track="track_1",ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",
        forward_attempts=1,forward_returns=1,forward_validated=1,shared_identity_verified=True,raw_prediction_bytes_preserved=True,
        native_global_unchanged=True,hub_restored=True,source_inputs_assets_rehashed=True,quality_verified=False,adoption_authorized=False,
        refinement_performed=False,unchanged_original_CARI_method=False,MHR_strict_scope_performed=False,exact_deterministic_forward_claim=False,submission_produced=False,stored_bundle_reread_verified=True)
    if any(type(report.get(k)) is not type(v) or report[k] != v for k,v in expected.items()): raise ValueError("Complete constrained native forward report required")
    if any(type(report.get(k)) is not int or report[k] != 1 for k in constrained.COUNTERS): raise ValueError("Complete one-window hook evidence required")


def capture_native_raw(torch, cloned, report):
    """Instrument the sole isolated composition lookup; all other globals intact."""
    delegate = cloned.__globals__["compose_mhr_output"]
    def capture(prediction,batch):
        result = delegate(prediction,batch)
        saved = {}
        for key,value in prediction.items():
            if key in {"rot","trans"} or key.startswith("delta_mhr_"):
                a = constrained._numpy(torch,value).reshape(96,-1)
                saved[key] = {"shape":list(a.shape),"sha256":hashlib.sha256(a.tobytes()).hexdigest()}
        report["original_native_raw"] = saved
        return result
    cloned.__globals__["compose_mhr_output"] = capture
    return cloned


def local_dino_hub(torch, repository, report):
    """Install one offline source guard; caller always restores the exact loader."""
    original = torch.hub.load
    def load(repo_or_dir,*arguments,**kwargs):
        if repo_or_dir not in ("facebookresearch/dinov2","facebookresearch/dinov2:main") or kwargs.get("source","github") != "github":
            raise ValueError("Unexpected native DINO Hub source")
        model = kwargs.get("model", arguments[0] if arguments else None)
        if model not in ("dinov2_vitb14","dinov2_vits14"): raise ValueError("Unexpected native DINO model")
        report["hub_attempts"] += 1; kwargs["source"] = "local"
        result = original(str(repository),*arguments,**kwargs); report["hub_returns"] += 1
        return result
    torch.hub.load = load
    return original


def asset_bindings(root, prepared):
    source = body._source_identity(root); directory, assets = body._body_assets(root)
    if source != prepared.get("inference_source_identity") or assets != prepared.get("body_assets") or (directory/"mhr_buffers.pt").exists():
        raise ValueError("Original decoder assets/source differ from prepared initializer")
    vendor = root/"vendor/video_to_data"; body._pinned_checkout(vendor,constrained.UPSTREAM_REVISION)
    native = vendor/"reconstruction/modules/v2d_cari4d/lib/cari4d"
    files = {str(native/p):identity(native/p) for p in SOURCES}
    if any(files[str(native/p)]["sha256"] != digest for p,digest in SOURCES.items()): raise ValueError("Pinned original native source/config differs")
    acquisition = root/"results/weights-acquisition.json"; data = json.loads(acquisition.read_text())
    records = [row for row in data["assets"] if row["repo_id"] == "nvidia/cari4d_commercial"]
    checkpoint_dir = root/"weights/cari4d/cari4d"
    if len(records) != 1 or records[0]["revision"] != CHECKPOINT_REVISION or records[0]["path"] != str(checkpoint_dir):
        raise ValueError("Pinned commercial checkpoint acquisition required")
    checkpoint = checkpoint_dir/CHECKPOINT_RELATIVE_PATH
    if identity(checkpoint)["sha256"] != CHECKPOINT_SHA256: raise ValueError("Original commercial checkpoint differs")
    files[str(checkpoint)] = identity(checkpoint)
    home = root/"weights/cari4d/sam3d_body/torch_home"; repository = home/"hub/facebookresearch_dinov2_main"
    body._pinned_checkout(repository,DINOV2_REVISION)
    auxiliary = root/"results/auxiliary-assets.json"; aux = json.loads(auxiliary.read_text())
    if aux.get("source_revisions",{}).get("dinov2") != DINOV2_REVISION: raise ValueError("Pinned local DINO acquisition required")
    for name,digest in DINO_HASHES.items():
        rows = [row for row in aux["checkpoints"] if row["filename"] == name]
        path = home/"hub/checkpoints"/name
        if len(rows) != 1 or rows[0]["sha256"] != digest or identity(path) != {k:rows[0][k] for k in ("sha256","bytes")}:
            raise ValueError("Exact local DINO checkpoint differs")
        files[str(path)] = identity(path)
    files[str(acquisition)] = identity(acquisition); files[str(auxiliary)] = identity(auxiliary)
    for p in sorted(repository.rglob("*.py")): files[str(p)] = identity(p)
    return native,checkpoint,home,repository,source,assets,files


def run(root,out,code,report,persist):
    pins_path = code/"configs/cari96_prepare_pins.json"; pin_id = identity(pins_path,immutable=True)
    pins = json.loads(pins_path.read_text()); directory,prepared,initializer,poses,prepared_files = prepared_inputs(root,pins)
    # Literal provenance sources are included in the static immutable code closure.
    prepare_source = Path(__file__).with_name("cari96_prepare.py")
    if identity(prepare_source,immutable=True)["sha256"] != pins["prepare"]["script_sha256"]:
        raise ValueError("Current unchanged preparation source differs from producer")
    native,checkpoint,home,repository,source,assets,files = asset_bindings(root,prepared)
    helpers = {str(p):identity(p,immutable=True) for p in (Path(__file__),prepare_source,Path(constrained.__file__),Path(inputs.__file__),Path(body.__file__),Path(__file__).with_name("cari_runner.py"))}
    report.update(phase="public_inputs_audited",prepare_report_sha256=pins["prepare"]["sha256"],prepare_files=prepared_files,
        snapshot_manifest_sha256=prepared["snapshot_manifest_sha256"],body_assets=assets,inference_source_identity=source,
        decoder_identity=prepared["decoder_identity"],checkpoint_sha256=CHECKPOINT_SHA256,config_sha256=SOURCES[CONFIG_RELATIVE_PATH],composition_source_sha256=SOURCES[constrained.COMPOSITION_RELATIVE_PATH]);persist()
    if "torch" in sys.modules or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise ValueError("Fresh explicit CUDA runtime required")
    import torch
    if not torch.cuda.is_available() or str(torch.__version__) != "2.5.1+cu124" or torch.version.cuda != "12.4": raise ValueError("Pinned CUDA image required")
    np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    environment = build_cari_runtime_environment(str(native),str(root/"weights/cari4d/sam3d_body"),str(home),"/workspace/v2d_sam3d_body/lib")
    os.environ.update(environment);os.environ["MPLCONFIGDIR"]="/tmp/cari96-matplotlib"
    sys.path[:0] = environment["PYTHONPATH"].split(":")
    original = local_dino_hub(torch,repository,report)
    try:
        from tools import run_mhr_wild_inference as module
        if Path(module.__file__).resolve() != native/constrained.NATIVE_RELATIVE_PATH: raise ValueError("Native tools namespace resolved incorrectly")
        if module.compose_mhr_output.__code__.co_filename != str(native/constrained.COMPOSITION_RELATIVE_PATH): raise ValueError("Original composition import differs")
        report.update(phase="native_constrained_forward");persist()
        cloned = constrained.clone_native_forward(torch,module.run_mhr_wild_inference,report,
            source_bytes=(native/constrained.NATIVE_RELATIVE_PATH).read_bytes())
        cloned = capture_native_raw(torch,cloned,report)
        report["forward_attempts"] += 1; persist()
        result = cloned(directory/EXPORT,directory/"inputs/aligned_depth.h5",directory/"shared_initializer.pkl",
            directory/"inputs/own_object_poses.pkl",native/CONFIG_RELATIVE_PATH,checkpoint,out/"coconet.pth",
            stride=96,render_batch_size=32,crop_workers=8,crop_buffer_count=2,input_cache=None,use_input_cache=False,
            device_name="cuda",overwrite=False,wandb_run_path=None,offline_supervision_contract=True)
        report["forward_returns"] += 1;torch.cuda.synchronize()
        if Path(result).resolve() != out/"coconet.pth": raise ValueError("Native output path differs")
        bundle = torch.load(result,map_location="cpu",weights_only=False)
        raw = validate_bundle(bundle,initializer,poses,report)
        if raw != report.get("original_native_raw"): raise ValueError("Stored raw deltas differ from untouched native predictions")
        report["forward_validated"] += 1
        paths = dict(depth_source=directory/"inputs/aligned_depth.h5",mhr_init_source=directory/"shared_initializer.pkl",foundationpose_source=directory/"inputs/own_object_poses.pkl",object_mesh=directory/EXPORT/"object_mesh/output_aligned.glb")
        if any(bundle["metadata"].get(k) != str(v) for k,v in paths.items()): raise ValueError("Native consumed-input paths differ")
        if any(bundle.get("checkpoint",{}).get(k) != v for k,v in dict(path=str(checkpoint),step=200000).items()): raise ValueError("Native checkpoint provenance differs")
        if torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled(): raise ValueError("Native ordinary CUDA policy changed")
        active_config,config_digest = config_receipt(bundle["config"])
        report.update(native_raw=raw,native_metadata=bundle["metadata"],native_config=active_config,
            native_config_fingerprint=config_digest,native_frozen_hand_face_operation="unchanged_fp32_init_plus_exact_zero",
            source_files=files,helper_files=helpers)
    finally:
        torch.hub.load = original; report["hub_restored"] = torch.hub.load is original;persist()
    if not report["hub_attempts"] or report["hub_returns"] != report["hub_attempts"]: raise ValueError("All actual offline DINO loads must return")
    if prepared_inputs(root,pins)[4] != prepared_files or identity(pins_path,immutable=True) != pin_id or asset_bindings(root,prepared)[-1] != files:
        raise ValueError("Input/source/asset bindings changed during native forward")
    if any(identity(Path(p),immutable=True) != row for p,row in helpers.items()): raise ValueError("Frozen producer source changed")
    (out/"coconet.pth").chmod(0o444)
    frozen_bundle = identity(out/"coconet.pth",immutable=True)
    reread = torch.load(out/"coconet.pth",map_location="cpu",weights_only=False)
    if validate_bundle(reread,initializer,poses,report) != report["original_native_raw"] or identity(out/"coconet.pth",immutable=True) != frozen_bundle:
        raise ValueError("Frozen bundle reread/raw preservation failed")
    report["stored_bundle_reread_verified"] = True
    report.update(status="pass",phase="complete",frames=96,source_frames=501,original_frame_indices=list(range(96)),
        bundle_sha256=inputs.sha256(out/"coconet.pth"),bundle_bytes=(out/"coconet.pth").stat().st_size,
        shared_identity_verified=True,native_global_unchanged=report["native_compose_global_unchanged"],source_inputs_assets_rehashed=True,submission_produced=False)
    validate_forward_report(report)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);rev=os.environ["WR_CODE_REVISION"];out=root/BASE
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or os.geteuid() != 1000
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.environ["WR_IMAGE_ID"] != IMAGE
            or not re.fullmatch(r"[0-9a-f]{40}",rev) or not out.is_dir() or any(out.iterdir()) or out.resolve()!=out.absolute()):
        raise ValueError("Fresh offline source-bound forward output required")
    report = dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=rev,script_sha256=inputs.sha256(Path(__file__)),image_id=IMAGE,
        input_track="track_1",ground_truth_used=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",budget_seconds=BUDGET,
        quality_verified=False,adoption_authorized=False,refinement_performed=False,unchanged_original_CARI_method=False,
        MHR_strict_scope_performed=False,exact_deterministic_forward_claim=False,native_enable_amp=True,
        deterministic_algorithms=False,warn_only=False,TF32=False,forward_attempts=0,forward_returns=0,forward_validated=0,hub_attempts=0,hub_returns=0)
    path=out/"report.json";started=time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Constrained native96 forward exceeded360s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,out,code,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist()
            for p in out.rglob("*"):
                if p.is_file():p.chmod(0o444)


if __name__ == "__main__": main()
