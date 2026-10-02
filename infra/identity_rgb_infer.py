"""Freeze public RGB observations, then test a first-RGB shared human identity.

The second branch changes only shape45/scale28 within a five-frame clip.
Neither private truth, inverse fitting, learned camera, alpha nor object shape
is accessible here. Reference replay establishes ABI fidelity, not accuracy.
"""
import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np

import identity_rgb_masks as masks
import hand_synthetic_infer as human
import hand_synthetic_render as regions_helper
import joint_rgb_infer as joint
import object_synthetic_observations as depth_model
from camera_render import raster_camera_mesh
from world_reward.data import sha256
from world_reward.pointmap import validate_camera_pointmap

BASE = "validation/identity_rgb_v2"
# The same immutable official tool/model as native conversion, without loading
# the inverse-fitting driver merely to obtain two data-independent file pins.
CONVERTER_SHA256 = "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e"
REFERENCE_MODEL_SHA256 = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
STAGE = "public_identity_rgb_raw_and_first_frame_shared_identity_predictions"
CLIPS, FRAMES, WIDTH, HEIGHT, VERTICES, JOINTS, BUDGET = 3, 5, 1024, 768, 18439, 127, 600
CAMERA_K = np.array([[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]])
IMAGE_ID = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BLOCKS = {"global_rot": (3,), "body_pose_params": (133,), "hand_pose_params": (108,),
          "shape_params": (45,), "scale_params": (28,), "expr_params": (72,),
          "pred_cam_t": (3,), "mhr_model_params": (204,)}
RAW_KEYS = frozenset(BLOCKS) | {"raw_depth", "raw_points", "validity", "rendered_depth", "silhouette",
    "human_mask", "object_mask", "raw_vertices_camera_m", "raw_joints_camera_m", "human_faces", "camera_K", "clip_index", "frame_index"}
PAIR_KEYS = frozenset({"raw_vertices_camera_m", "shared_vertices_camera_m", "raw_joints_camera_m", "shared_joints_camera_m",
    "human_faces", "hand_mask_left", "hand_mask_right", "camera_K", "pred_cam_t", "raw_model_controls", "shared_model_controls",
    "raw_shape_params", "shared_shape_params", "raw_scale_params", "shared_scale_params", "expression", "clip_index", "frame_index"})


def helper_identities():
    return {name: sha256(Path(module.__file__)) for name, module in {
        "mask": masks, "mask_shared": masks.masks, "body_loader": human,
        "body": human.body, "MoGe": depth_model, "depth_camera": joint, "regions": regions_helper,
    }.items()} | {"inference": sha256(Path(__file__)), "camera_render": sha256(Path(__file__).with_name("camera_render.py"))}


def regular(path, expected=None, immutable=False):
    path = Path(path)
    if path.resolve() != path.absolute() or not path.is_file() or (immutable and path.stat().st_mode & 0o222):
        raise ValueError("Require canonical regular frozen public artifact")
    actual = masks.masks.identity(path)
    if expected is not None and actual["sha256"] != expected: raise ValueError("Public artifact SHA differs")
    return actual


def public_inputs(root):
    """Validate the entire automatic-mask producer before reading observations."""
    root = Path(root); base = root/BASE; records, manifest = masks.validate_inputs(base/"inputs")
    path = base/"automatic_masks/report.json"; receipt = regular(path, immutable=True); report = json.loads(path.read_text())
    expected = dict(stage=masks.STAGE, status="pass", phase="complete", frames=15, network="none", private_truth_read=False,
        ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        actual_automatic_inference_verified=True, all_cases_retained=True, human_query="person.", object_query="bottle.",
        input_manifest_sha256=manifest["sha256"], input_manifest_bytes=manifest["bytes"],
        script_sha256=sha256(Path(masks.__file__)), shared_mask_helper_sha256=sha256(Path(masks.masks.__file__)),
        detector_revision=masks.masks.DETECTOR_REVISION, sam2_weights_revision=masks.masks.SAM2_REVISION)
    if (not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items())
            or not re.fullmatch("[0-9a-f]{40}", str(report.get("producer_revision", "")))
            or not re.fullmatch("sha256:[0-9a-f]{64}", str(report.get("image_id", "")))):
        raise ValueError("Require current completed automatic masks with no fallback or labels")
    masks.completed(report)
    if report.get("model_assets") != {name: {"sha256": value[0], "bytes": value[1]} for name, value in masks.masks.ASSETS.items()}:
        raise ValueError("Automatic masks did not bind all exact detector/SAM2 assets")
    names = {"report.json"}
    for record, row in zip(records, report["records"]):
        if row.get("rgb_sha256") != record["sha256"]: raise ValueError("Mask/RGB identities differ")
        for label, _ in masks.QUERIES:
            filename = row[label+"_mask_file"]; names.add(filename); target = base/"automatic_masks"/filename
            identity = regular(target, row[label+"_mask_sha256"], immutable=True)
            if identity["bytes"] != row.get(label+"_mask_bytes"): raise ValueError("Mask size differs")
            record[label+"_mask_path"] = target
    if {p.name for p in (base/"automatic_masks").iterdir()} != names: raise ValueError("Unexpected public mask artifacts")
    return records, {"public_manifest_sha256": manifest["sha256"], "public_inputs_sha256": manifest["sha256"], "public_inputs_bytes": manifest["bytes"],
                     "mask_report_sha256": receipt["sha256"], "mask_report_bytes": receipt["bytes"]}


def validate_producer_bindings(root, report):
    """CPU-only source/asset integrity; does not certify receipt numerical truth."""
    root = Path(root); model = report.get("body_model", {})
    if (report.get("script_sha256") != sha256(Path(__file__)) or report.get("helper_source_sha256") != helper_identities()
            or report.get("model_sha256") != REFERENCE_MODEL_SHA256 or report.get("converter_sha256") != CONVERTER_SHA256
            or model.get("body_revision") != human.body.BODY_REVISION or model.get("upstream_revision") != human.body.UPSTREAM_REVISION
            or model.get("dinov3_revision") != human.body.DINOV3_REVISION
            or model.get("body_assets", {}).get("model.ckpt") != {"sha256": human.BODY_SHA, "bytes": human.BODY_BYTES}
            or model.get("checkpoint_loading", {}).get("mode") != "strict_network_and_head_state_with_explicit_asset_buffer_retention"
            or model.get("checkpoint_loading", {}).get("unexpected_keys") != []
            or report.get("sources_assets_rechecked") is not True or report.get("hand_regions_native_verified") is not True):
        raise ValueError("Actual pinned producer/model/source binding differs")
    if human.body._source_identity(root) != model.get("inference_source_identity") or human.body._body_assets(root)[1] != model["body_assets"]:
        raise ValueError("Current original native Body source/assets differ")
    if depth_model.model_asset(root)[1:] != (report.get("acquisition_report"), report.get("MoGe_model_asset")):
        raise ValueError("Original MoGe model/receipt differs")
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    spec = importlib.util.find_spec("moge")
    if spec is None or not spec.submodule_search_locations: raise ValueError("Installed pinned MoGe package missing")
    directory = Path(next(iter(spec.submodule_search_locations)))
    if (depth_model.installed_source(directory, json.loads(direct) if direct else {}) != report.get("MoGe_source")
            or regular(directory/"utils/geometry_torch.py")["sha256"] != joint.GEOMETRY_SHA):
        raise ValueError("Actual MoGe focal/source identity differs")
    regular(root/"weights/mhr/mhr_model.pt", REFERENCE_MODEL_SHA256)
    regular(root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py", CONVERTER_SHA256)
    path = root/"results/mhr-finger-semantics-v4.json"; regular(path, report.get("semantic_report_sha256"))
    semantic = json.loads(path.read_text()); regions_helper.require_semantic_report(semantic)
    if report.get("joint_names") != semantic["joint_names"] or report.get("image_id") != IMAGE_ID or semantic.get("source_image_id") != IMAGE_ID:
        raise ValueError("Native region names/reference semantic image identity differs")
    return True


def array(value, shape, dtype, *, finite=True):
    result = np.asarray(value)
    if np.ma.isMaskedArray(value) or result.shape != shape or result.dtype != np.dtype(dtype) or (finite and not np.isfinite(result).all()):
        raise ValueError("Exact array shape/dtype/finite contract differs")
    return result


def common(data, record):
    for key in ("clip_index", "frame_index"):
        if int(array(data[key], (), "int64")) != record[key]: raise ValueError("Original frame mapping differs")
    if not np.array_equal(array(data["camera_K"], (3, 3), "float64"), CAMERA_K): raise ValueError("Fixed public camera differs")
    faces = array(data["human_faces"], (36874, 3), "int64")
    if np.any(faces < 0) or np.any(faces >= VERTICES): raise ValueError("Native topology indices differ")


def validate_raw(data, record):
    if set(data) != RAW_KEYS: raise ValueError("Exact raw observation schema required")
    common(data, record)
    for key, shape in BLOCKS.items(): array(data[key], shape, "float32")
    if np.any(data["expr_params"]) or np.any(data["mhr_model_params"][:3]) or data["pred_cam_t"][2] <= 0:
        raise ValueError("Root-relative native controls/zero expression/positive camera required")
    array(data["raw_vertices_camera_m"], (VERTICES, 3), "float32"); array(data["raw_joints_camera_m"], (JOINTS, 3), "float32")
    for key in ("validity", "silhouette", "human_mask", "object_mask"): array(data[key], (HEIGHT, WIDTH), "bool")
    depth = array(data["raw_depth"], (HEIGHT, WIDTH), "float32", finite=False)
    points = array(data["raw_points"], (HEIGHT, WIDTH, 3), "float32", finite=False)
    normalized = np.diag([1/WIDTH, 1/HEIGHT, 1.]) @ CAMERA_K
    validate_camera_pointmap(depth, points, data["validity"], normalized, CAMERA_K)
    rendered = array(data["rendered_depth"], (HEIGHT, WIDTH), "float32", finite=False)
    silhouette = data["silhouette"]
    if (not silhouette.any() or not np.isfinite(rendered[silhouette]).all() or np.any(rendered[silhouette] <= 0)
            or not np.isnan(rendered[~silhouette]).all() or not data["human_mask"].any() or not data["object_mask"].any()):
        raise ValueError("Native rendered depth/automatic mask support differs; no fill")
    return data


def validate_pair(data, record, raw=None):
    if set(data) != PAIR_KEYS: raise ValueError("Exact paired geometry schema required")
    common(data, record)
    for mode in ("raw", "shared"):
        array(data[mode+"_vertices_camera_m"], (VERTICES, 3), "float32")
        array(data[mode+"_joints_camera_m"], (JOINTS, 3), "float32")
        array(data[mode+"_model_controls"], (204,), "float32")
        array(data[mode+"_shape_params"], (45,), "float32"); array(data[mode+"_scale_params"], (28,), "float32")
    left, right = (array(data["hand_mask_"+side], (VERTICES,), "bool") for side in ("left", "right"))
    if left.sum() < 50 or right.sum() < 50 or np.any(left & right): raise ValueError("Actual disjoint LBS hand regions required")
    array(data["pred_cam_t"], (3,), "float32"); array(data["expression"], (72,), "float32")
    if (np.any(data["expression"]) or np.any(data["shared_model_controls"][:3])
            or not np.array_equal(data["raw_model_controls"][:136], data["shared_model_controls"][:136])):
        raise ValueError("Shared branch changed root/body/fingers or zero expression")
    if raw is not None:
        for pair, original in (("raw_model_controls", "mhr_model_params"), ("raw_shape_params", "shape_params"),
                ("raw_scale_params", "scale_params"), ("expression", "expr_params"), ("pred_cam_t", "pred_cam_t"),
                ("raw_vertices_camera_m", "raw_vertices_camera_m"), ("raw_joints_camera_m", "raw_joints_camera_m"), ("human_faces", "human_faces")):
            if not np.array_equal(data[pair], raw[original]): raise ValueError("Paired raw source changed after freeze")
    return data


def save(output, folder, record, data):
    relative = folder+"/"+Path(record["file"]).stem+".npz"; target = output/relative
    with target.open("xb") as stream: np.savez_compressed(stream, **data)
    target.chmod(0o444)
    return {k: record[k] for k in ("file", "clip_index", "frame_index")} | {
        "artifact": relative, "rgb_sha256": record["sha256"], **regular(target, immutable=True)}


def frozen_rows(output, rows, records, folder, validator):
    if len(rows) != 15 or {p.name for p in (output/folder).iterdir()} != {Path(r["file"]).stem+".npz" for r in records}:
        raise ValueError("Every original frozen observation required; no extras")
    result = []
    for row, record in zip(rows, records):
        if row["artifact"] != folder+"/"+Path(record["file"]).stem+".npz" or row["rgb_sha256"] != record["sha256"]:
            raise ValueError("Artifact original frame identity differs")
        target = output/row["artifact"]
        if regular(target, row["sha256"], immutable=True)["bytes"] != row["bytes"]: raise ValueError("Frozen artifact size differs")
        with np.load(target, allow_pickle=False) as archive: values = {k: archive[k] for k in archive.files}
        result.append(validator(values, record))
    return result


def validate_region_masks(pairs, report):
    expected = report.get("hand_region_sha256", {})
    if set(expected) != {"left", "right"}: raise ValueError("Native LBS region identities missing")
    for pair in pairs:
        for side in ("left", "right"):
            if hashlib.sha256(pair["hand_mask_"+side].tobytes()).hexdigest() != expected[side]:
                raise ValueError("Frozen paired mask differs from actual native LBS region")


def shared_decode(torch, head, raw, first):
    """Native metres, root-zero controls; apply original camera translation once."""
    tensor = lambda value: torch.as_tensor(value, device="cuda")[None]
    with torch.inference_mode():
        vertices, joints, controls = head.mhr_forward(global_trans=tensor(raw["global_rot"])*0,
            global_rot=tensor(raw["global_rot"]), body_pose_params=tensor(raw["body_pose_params"]),
            hand_pose_params=tensor(raw["hand_pose_params"]), scale_params=tensor(first["scale_params"]),
            shape_params=tensor(first["shape_params"]), expr_params=tensor(np.zeros(72, np.float32)),
            return_keypoints=False, return_joint_coords=True, return_model_params=True, return_joint_rotations=False)
    cpu = lambda value: value[0].detach().cpu().numpy()
    flip = np.array([1., -1., -1.], np.float32)
    return cpu(vertices)*flip+raw["pred_cam_t"], cpu(joints)*flip+raw["pred_cam_t"], cpu(controls)


def reference_errors(vertices_mm, joints_m, shared_vertices, shared_joints, translations):
    """Reference already flips axes: F64 mm target and translation exactly once."""
    v = array(vertices_mm, (15, VERTICES, 3), "float64") + translations.astype(np.float64)[:, None]*1000
    j = array(joints_m, (15, JOINTS, 3), "float64") + translations.astype(np.float64)[:, None]
    error = np.linalg.norm(v-shared_vertices.astype(np.float64)*1000, axis=2)
    means = error.mean(1)
    if np.any(means > 2.): raise ValueError("Independent reference forward mean exceeds2mm")
    return {"per_frame_mean_mm": means.tolist(), "max_point_mm_diagnostic": float(error.max()),
            "max_joint_distance_mm_diagnostic": float(np.linalg.norm(j-shared_joints.astype(np.float64), axis=2).max()*1000)}


def run(root, report, path, persist):
    records, public = public_inputs(root); helpers = helper_identities(); output = path.parent
    asset_path, acquisition, asset = depth_model.model_asset(root)
    model_path = root/"weights/mhr/mhr_model.pt"; tool = root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    regular(model_path, REFERENCE_MODEL_SHA256); regular(tool, CONVERTER_SHA256)
    semantic_path = root/"results/mhr-finger-semantics-v4.json"; semantic_receipt = regular(semantic_path)
    semantic = json.loads(semantic_path.read_text()); regions_helper.require_semantic_report(semantic)
    if report["image_id"] != IMAGE_ID or semantic.get("source_image_id") != IMAGE_ID:
        raise ValueError("Require actual original native/reference semantic image")
    report.update(**public, helper_source_sha256=helpers, script_sha256=sha256(Path(__file__)), acquisition_report=acquisition, MoGe_model_asset=asset,
        semantic_report_sha256=semantic_receipt["sha256"], model_sha256=REFERENCE_MODEL_SHA256, converter_sha256=CONVERTER_SHA256, phase="model_load"); persist()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import moge
    from moge.model import v2 as depth_module
    from moge.model.v2 import MoGeModel
    from moge.utils import geometry_torch
    from PIL import Image
    if not torch.cuda.is_available(): raise RuntimeError("Actual offline CUDA required")
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    source = depth_model.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
    if sha256(Path(geometry_torch.__file__)) != joint.GEOMETRY_SHA or depth_module.recover_focal_shift is not geometry_torch.recover_focal_shift:
        raise ValueError("Native MoGe focal solver source/ABI differs")
    model, estimator, faces, body_source = human.load_model(root, torch)
    joints = model.head_pose.mhr.get_joint_names(); lbs = model.head_pose.mhr.get_lbsw()
    if joints != semantic["joint_names"] or not isinstance(lbs, tuple) or len(lbs) != 2: raise ValueError("Actual native LBS/name semantics differ")
    regions = regions_helper.lbs_regions(*(v.detach().cpu().numpy() for v in lbs), joints)
    report.update(body_model=body_source, MoGe_source=source, focal_geometry_source_sha256=joint.GEOMETRY_SHA,
        torch_version=str(torch.__version__), joint_names=joints, hand_regions_native_verified=True,
        hand_region_sha256={name: hashlib.sha256(regions[side]["vertex_mask"].tobytes()).hexdigest()
                           for name, side in (("left", "l"), ("right", "r"))},
        hand_regions_source="actual_native_head_MHR_get_lbsw_and_joint_names", phase="raw_inference"); persist()
    network = MoGeModel.from_pretrained(str(asset_path)).cuda().eval()
    camera = torch.as_tensor(CAMERA_K.astype(np.float32), device="cuda")[None]
    for record in records:
        report.update(active_file=record["file"]); persist()
        with Image.open(record["path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (WIDTH, HEIGHT): raise ValueError("Original public RGB decode differs")
            rgb = np.asarray(image).copy()
        hm, om = (joint.read_mask(record[label+"_mask_path"], Image) for label in ("human", "object"))
        box, prompt = human.derived_bbox(rgb, hm)
        with torch.inference_mode(): predictions = estimator.process_one_image(img=rgb, bboxes=box[None], masks=prompt, cam_int=camera, inference_type="body")
        if not isinstance(predictions, list) or len(predictions) != 1: raise ValueError("Exactly one actual Body prediction required")
        values, parity = human.decode_prediction(torch, model, predictions[0], "body"); report["body_calls_completed"] += 1
        if not np.isclose(float(values["focal_length"]), 1280., atol=1e-4, rtol=1e-6): raise ValueError("Body ignored fixed camera")
        tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
        depth = joint.checked_depth_infer(torch, depth_module, network, tensor, float(np.degrees(2*np.arctan(WIDTH/2560))),
            report["native_focal_solver_calls"], {k: record[k] for k in ("file", "clip_index", "frame_index")})
        report["MoGe_calls_completed"] += 1
        z, points, valid, normalized = (depth[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics"))
        checks = joint.pointmap_contract(z, points, valid, normalized, CAMERA_K)
        silhouette, rendered = raster_camera_mesh(values["vertices_camera_m"], faces, CAMERA_K, WIDTH, HEIGHT)
        raw = {k: values[k] for k in BLOCKS} | dict(raw_depth=z, raw_points=points, validity=valid,
            rendered_depth=rendered.cpu().numpy(), silhouette=silhouette.cpu().numpy(), human_mask=hm > 0, object_mask=om > 0,
            raw_vertices_camera_m=values["vertices_camera_m"], raw_joints_camera_m=values["joints_camera_m"], human_faces=faces,
            camera_K=CAMERA_K.copy(), clip_index=np.array(record["clip_index"], np.int64), frame_index=np.array(record["frame_index"], np.int64))
        validate_raw(raw, record); report["raw_outputs"].append(save(output, "raw", record, raw))
        report["calls"].append(dict(file=record["file"], bbox_xyxy=box.tolist(), native_forward_errors=parity,
            pointmap_checks=checks, decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())); persist()
        del predictions, depth, tensor, raw, values
    del network; torch.cuda.empty_cache()
    raw_frames = frozen_rows(output, report["raw_outputs"], records, "raw", validate_raw)
    report.update(raw_frozen_before_shared=True, phase="shared_native_decode"); persist()
    pairs = []
    for record, raw in zip(records, raw_frames):
        first = raw_frames[record["clip_index"]*FRAMES]
        v, j, controls = shared_decode(torch, model.head_pose, raw, first); report["shared_head_calls_completed"] += 1
        pair = dict(raw_vertices_camera_m=raw["raw_vertices_camera_m"], shared_vertices_camera_m=v,
            raw_joints_camera_m=raw["raw_joints_camera_m"], shared_joints_camera_m=j, human_faces=faces,
            hand_mask_left=regions["l"]["vertex_mask"], hand_mask_right=regions["r"]["vertex_mask"], camera_K=CAMERA_K.copy(),
            pred_cam_t=raw["pred_cam_t"], raw_model_controls=raw["mhr_model_params"], shared_model_controls=controls,
            raw_shape_params=raw["shape_params"], shared_shape_params=first["shape_params"], raw_scale_params=raw["scale_params"],
            shared_scale_params=first["scale_params"], expression=np.zeros(72, np.float32), clip_index=raw["clip_index"], frame_index=raw["frame_index"])
        validate_pair(pair, record, raw); pairs.append(pair); persist()
    for start in range(0, 15, FRAMES):
        for row in pairs[start:start+FRAMES]:
            if not np.array_equal(row["shared_model_controls"][136:], pairs[start]["shared_model_controls"][136:]): raise ValueError("Shared68 scales changed within clip")
    for record, pair in zip(records, pairs): report["paired_outputs"].append(save(output, "paired", record, pair))
    pairs = frozen_rows(output, report["paired_outputs"], records, "paired", validate_pair)
    validate_region_masks(pairs, report)
    report.update(paired_frozen_before_reference=True); persist()
    report.update(phase="independent_reference_forward"); persist()
    spec = importlib.util.spec_from_file_location("world_reward_identity_official_reference", tool)
    if spec is None or spec.loader is None: raise ValueError("Pinned official reference import failed")
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    reference = official.MHR(str(model_path), "cuda", chunk=16, precision="float32")
    if not np.array_equal(reference.model.character_torch.mesh.faces.detach().cpu().numpy(), faces): raise ValueError("Reference/native topology differs")
    controls = np.stack([p["shared_model_controls"] for p in pairs]); shapes = np.stack([p["shared_shape_params"] for p in pairs])
    vertices_mm, joints_m = reference.run(torch.tensor(controls[:, :136], dtype=torch.float64, device="cuda"),
        torch.tensor(np.concatenate((controls[:, 136:], shapes), axis=1), dtype=torch.float64, device="cuda"))
    report["official_reference_calls"] += 1
    report["reference_fidelity"] = reference_errors(vertices_mm.detach().cpu().numpy(), joints_m.detach().cpu().numpy(),
        np.stack([p["shared_vertices_camera_m"] for p in pairs]), np.stack([p["shared_joints_camera_m"] for p in pairs]), np.stack([p["pred_cam_t"] for p in pairs]))
    report["official_reference_per_frame_mean_mm"] = report["reference_fidelity"]["per_frame_mean_mm"]
    frozen_rows(output, report["raw_outputs"], records, "raw", validate_raw)
    frozen_rows(output, report["paired_outputs"], records, "paired", validate_pair)
    if (public_inputs(root) != (records, public) or helper_identities() != helpers or depth_model.model_asset(root)[1:] != (acquisition, asset)
            or human.body._source_identity(root) != body_source["inference_source_identity"]
            or human.body._body_assets(root)[1] != body_source["body_assets"]
            or depth_model.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {}) != source):
        raise ValueError("Source/model/public artifacts changed during inference")
    regular(model_path, REFERENCE_MODEL_SHA256); regular(tool, CONVERTER_SHA256); regular(semantic_path, semantic_receipt["sha256"])
    if {p.name for p in output.iterdir()} != {"raw", "paired", "report.json"}: raise ValueError("Unexpected inference outputs")
    report.update(status="pass", phase="complete", frames=15, all_cases_retained=True, sources_assets_rechecked=True,
        conversion_fidelity_verified=True, actual_body_inference=True, actual_MoGe_inference=True, actual_shared_native_forward=True,
        file_inventory=[r["artifact"] for r in report["raw_outputs"]+report["paired_outputs"]]+["report.json"])
    report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}: raise RuntimeError("Require remote Linux CUDA network-none")
    root = Path(os.environ["WR_ROOT"]); output = root/BASE/"predictions_v1"; path = output/"report.json"
    if root != Path("/srv/scenesmith/world-reward") or output.resolve() != output.absolute() or not output.is_dir() or any(output.iterdir()):
        raise FileExistsError("Require exclusively reserved fresh inference output")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch("[0-9a-f]{40}", revision) or image != IMAGE_ID: raise ValueError("Immutable source/original semantic image identities required")
    (output/"raw").mkdir(); (output/"paired").mkdir()
    report = dict(stage=STAGE, status="fail", phase="public_integrity", producer_revision=revision, image_id=image, network="none",
        private_truth_read=False, challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        adoption_performed=False, body_inference_type="body", camera_K=CAMERA_K.tolist(), focal_fitted=False, scale_fit=False,
        identity_source="first_original_RGB_shape45_and_scale28_per_five_frame_clip", expression_zero=True,
        camera_translation_rule="root_zero_native_and_official_YZ_flip_then_original_pred_cam_t_once", inverse_fit_performed=False,
        budget_seconds=BUDGET, raw_outputs=[], paired_outputs=[], calls=[], native_focal_solver_calls=[],
        body_calls_completed=0, MoGe_calls_completed=0, shared_head_calls_completed=0, official_reference_calls=0,
        raw_frozen_before_shared=False, accuracy_verified=False, full_HOI_verified=False)
    started = time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole identity inference exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, report, path, persist)
        except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
