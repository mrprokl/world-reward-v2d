"""Freeze fresh D96 public Body/MoGe and first-RGB identity observations only.

No private scene, DWPose, refitting or quality selection is accessible. Native
308 keypoints remain distinct from reference127 joints and prediction70 joints.
Official replay proves representation fidelity, not geometric accuracy.
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
import sys
import time

import numpy as np
import identity_rgb_infer as native
import keypoint_rgb_masks as masks
from world_reward.data import sha256

BASE = "validation/keypoint_rgb_v1"
STAGE = "public_keypoint_rgb_body_depth_first_rgb_identity_baseline"
MANIFEST_SHA = "082549b5a1f8a4a7d687bc17ec6847f3628d6d4230186053951caa2454d2979d"
MASK_SHA = "d9cae7c4c7131b599ebcafc58a415b9f763286806f80d4a6a46ee114249d8568"
MASK_REVISION = "42f457fc46fbeb814befb48b8104403b06922f63"
MASK_SCRIPT_SHA = "d7269a1bd7d1041b902b9278770b0fcd25d1a4455001fa964deba985898760f7"
KEYPOINTS, BUDGET = 308, 600
RAW_EXTRA = {"raw_keypoints_camera_m", "raw_joint_global_rotations"}
PAIR_EXTRA = {"raw_keypoints_camera_m", "shared_keypoints_camera_m", "raw_joint_global_rotations", "shared_joint_global_rotations"}
RAW_KEYS, PAIR_KEYS = native.RAW_KEYS | RAW_EXTRA, native.PAIR_KEYS | PAIR_EXTRA
CAMERA_K = native.CAMERA_K.copy()


def helper_identities():
    return native.helper_identities() | {"baseline": sha256(Path(__file__)), "fresh_mask": sha256(Path(masks.__file__))}


def public_inputs(root):
    """Fresh full-public whitelist; never call the old cohort's public reader."""
    base = Path(root)/BASE; records, manifest = masks.validate_inputs(base/"inputs")
    if manifest["sha256"] != MANIFEST_SHA or manifest["bytes"] != 2199: raise ValueError("Exact fresh RGB manifest differs")
    path = base/"automatic_masks/report.json"; receipt = native.regular(path, MASK_SHA, immutable=True); report = json.loads(path.read_text())
    expected = dict(stage=masks.STAGE, status="pass", phase="complete", frames=15, network="none", private_truth_read=False,
        ground_truth_used=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        actual_automatic_inference_verified=True, all_cases_retained=True, human_query="person.", object_query="bottle.",
        input_manifest_sha256=MANIFEST_SHA, input_manifest_bytes=2199, producer_revision=MASK_REVISION,
        script_sha256=MASK_SCRIPT_SHA, shared_mask_helper_sha256=sha256(Path(masks.masks.__file__)),
        detector_revision=masks.masks.DETECTOR_REVISION, sam2_weights_revision=masks.masks.SAM2_REVISION)
    if (sha256(Path(masks.__file__)) != MASK_SCRIPT_SHA or not isinstance(report, dict)
            or any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items())):
        raise ValueError("Actual frozen fresh automatic masks/source provenance differs")
    masks.completed(report); assets = report.get("model_assets")
    if (not isinstance(assets, dict) or set(assets) != set(masks.masks.ASSETS)
            or any(not isinstance(assets[n], dict) or set(assets[n]) != {"path", "sha256", "bytes"}
                   or assets[n]["path"] != str(Path(root)/"weights"/n) or assets[n]["sha256"] != pin[0]
                   or type(assets[n]["bytes"]) is not int or assets[n]["bytes"] != pin[1] for n, pin in masks.masks.ASSETS.items())):
        raise ValueError("Exact detector/SAM2 asset receipt required")
    names = {"report.json"}
    for record, row in zip(records, report["records"]):
        if row.get("rgb_sha256") != record["sha256"]: raise ValueError("Automatic masks/RGB hashes differ")
        native.regular(record["path"], record["sha256"], immutable=True)
        for label, _ in masks.QUERIES:
            name = row[label+"_mask_file"]; target = base/"automatic_masks"/name; names.add(name)
            if native.regular(target, row[label+"_mask_sha256"], immutable=True)["bytes"] != row[label+"_mask_bytes"]:
                raise ValueError("Frozen automatic mask bytes differ")
            record[label+"_mask_path"] = target
    if {p.name for p in (base/"automatic_masks").iterdir()} != names: raise ValueError("Unexpected public mask artifacts")
    return records, {"public_manifest_sha256": MANIFEST_SHA, "public_inputs_sha256": MANIFEST_SHA, "public_inputs_bytes": 2199,
                     "mask_report_sha256": MASK_SHA, "mask_report_bytes": receipt["bytes"]}


def rotations(value):
    result = native.array(value, (127, 3, 3), "float32")
    if (not np.allclose(result @ result.swapaxes(-1, -2), np.eye(3), atol=1e-4, rtol=0)
            or not np.allclose(np.linalg.det(result), 1., atol=1e-4, rtol=0)):
        raise ValueError("Native global joint rotations must be proper SO3")
    return result


def validate_raw(data, record):
    if set(data) != RAW_KEYS: raise ValueError("Exact extended raw308 observation schema required")
    native.validate_raw({k: data[k] for k in native.RAW_KEYS}, record)
    native.array(data["raw_keypoints_camera_m"], (KEYPOINTS, 3), "float32"); rotations(data["raw_joint_global_rotations"])
    return data


def validate_pair(data, record, raw=None):
    if set(data) != PAIR_KEYS: raise ValueError("Exact extended paired308 schema required")
    native.validate_pair({k: data[k] for k in native.PAIR_KEYS}, record,
                         None if raw is None else {k: raw[k] for k in native.RAW_KEYS})
    for mode in ("raw", "shared"):
        native.array(data[mode+"_keypoints_camera_m"], (KEYPOINTS, 3), "float32"); rotations(data[mode+"_joint_global_rotations"])
    if raw is not None and any(data[k].tobytes() != raw[k].tobytes() for k in RAW_EXTRA):
        raise ValueError("Frozen raw native keypoints/rotations changed")
    return data


def validate_clip_constants(pairs, raw_frames):
    if len(pairs) != 15 or len(raw_frames) != 15: raise ValueError("Every original native frame required")
    for index, pair in enumerate(pairs):
        for source in (pair, raw_frames[index]):
            if (int(native.array(source["clip_index"], (), "int64")), int(native.array(source["frame_index"], (), "int64"))) != divmod(index, 5):
                raise ValueError("Original complete clip/frame ordering required")
        first = raw_frames[(index//5)*5]; anchor = pairs[(index//5)*5]
        for key, original in (("shared_shape_params", "shape_params"), ("shared_scale_params", "scale_params")):
            if pair[key].tobytes() != first[original].tobytes(): raise ValueError("Shared identity must be exact first RGB, not fit/average")
        if pair["shared_model_controls"][136:].tobytes() != anchor["shared_model_controls"][136:].tobytes():
            raise ValueError("Expanded native68 scales must remain byte-constant per clip")


def raw_keypoint_decode(torch, head, prediction, values):
    with torch.inference_mode(): vertices, keypoints, joints, controls, global_rotations = native.human.body._native_forward_from_blocks(head, prediction)
    expected = ((1, 18439, 3), (1, 308, 3), (1, 127, 3), (1, 204), (1, 127, 3, 3))
    if any(tuple(x.shape) != s or not torch.isfinite(x).all() for x, s in zip((vertices, keypoints, joints, controls, global_rotations), expected)):
        raise ValueError("Actual native V/308KP/127joint/204control ABI differs")
    cpu = lambda x: x[0].detach().float().cpu().numpy()
    flip = np.array([1., -1., -1.], np.float32); t = values["pred_cam_t"]
    v, kp, j, c, r = cpu(vertices)*flip+t, cpu(keypoints)*flip+t, cpu(joints)*flip+t, cpu(controls), cpu(global_rotations)
    if (np.linalg.norm(v-values["vertices_camera_m"], axis=-1).max() > 1e-5
            or np.linalg.norm(j-values["joints_camera_m"], axis=-1).max() > 1e-5 or np.abs(c-values["mhr_model_params"]).max() > 1e-5):
        raise ValueError("Additional raw keypoint decode changed native geometry/controls")
    rotations(r); return kp, r


def shared_decode(torch, head, raw, first):
    tensor = lambda v: torch.as_tensor(v, device="cuda")[None]
    with torch.inference_mode():
        values = head.mhr_forward(global_trans=tensor(raw["global_rot"])*0, global_rot=tensor(raw["global_rot"]),
            body_pose_params=tensor(raw["body_pose_params"]), hand_pose_params=tensor(raw["hand_pose_params"]),
            scale_params=tensor(first["scale_params"]), shape_params=tensor(first["shape_params"]),
            expr_params=tensor(np.zeros(72, np.float32)), return_keypoints=True, return_joint_coords=True,
            return_model_params=True, return_joint_rotations=True)
    expected = ((1, 18439, 3), (1, 308, 3), (1, 127, 3), (1, 204), (1, 127, 3, 3))
    if not isinstance(values, tuple) or len(values) != 5 or any(tuple(v.shape) != s or not torch.isfinite(v).all() for v, s in zip(values, expected)):
        raise ValueError("Actual complete shared native forward ABI differs")
    cpu = lambda v: v[0].detach().float().cpu().numpy(); v, kp, j, c, r = (cpu(x) for x in values)
    flip = np.array([1., -1., -1.], np.float32); t = raw["pred_cam_t"]; rotations(r)
    return v*flip+t, kp*flip+t, j*flip+t, c, r


def validate_producer_bindings(root, report):
    """CPU file/source proof; numerical fidelity still requires actual execution."""
    model = report.get("body_model", {}); load = model.get("checkpoint_loading", {})
    if (report.get("script_sha256") != sha256(Path(__file__)) or report.get("helper_source_sha256") != helper_identities()
            or report.get("model_sha256") != native.REFERENCE_MODEL_SHA256 or report.get("converter_sha256") != native.CONVERTER_SHA256
            or model.get("body_revision") != native.human.body.BODY_REVISION or model.get("upstream_revision") != native.human.body.UPSTREAM_REVISION
            or model.get("dinov3_revision") != native.human.body.DINOV3_REVISION
            or model.get("body_assets", {}).get("model.ckpt") != {"sha256": native.human.BODY_SHA, "bytes": native.human.BODY_BYTES}
            or load.get("mode") != "strict_network_and_head_state_with_explicit_asset_buffer_retention" or load.get("unexpected_keys") != []
            or not isinstance(load.get("retained_mhr_asset_buffer_names"), list) or len(load["retained_mhr_asset_buffer_names"]) != 113
            or len(set(load["retained_mhr_asset_buffer_names"])) != 113 or type(load.get("parameter_tensors_loaded")) is not int
            or load["parameter_tensors_loaded"] != 1101
            or native.human.body._source_identity(root) != model.get("inference_source_identity")
            or native.human.body._body_assets(root)[1] != model.get("body_assets")):
        raise ValueError("Exact native Body checkpoint/source binding differs")
    if native.depth_model.model_asset(root)[1:] != (report.get("acquisition_report"), report.get("MoGe_model_asset")):
        raise ValueError("MoGe acquisition/source asset differs")
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json"); spec = importlib.util.find_spec("moge")
    if spec is None or not spec.submodule_search_locations: raise ValueError("Installed native MoGe source missing")
    directory = Path(next(iter(spec.submodule_search_locations)))
    if (native.depth_model.installed_source(directory, json.loads(direct) if direct else {}) != report.get("MoGe_source")
            or sha256(directory/"utils/geometry_torch.py") != native.joint.GEOMETRY_SHA):
        raise ValueError("Actual MoGe source/focal solver differs")
    for path, pin in ((root/"weights/mhr/mhr_model.pt", native.REFERENCE_MODEL_SHA256),
                      (root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py", native.CONVERTER_SHA256)):
        native.regular(path, pin)
    semantic_path = root/"results/mhr-finger-semantics-v4.json"; native.regular(semantic_path, report["semantic_report_sha256"])
    semantic = json.loads(semantic_path.read_text()); native.regions_helper.require_semantic_report(semantic)
    if (report.get("joint_names") != semantic["joint_names"] or report.get("image_id") != native.IMAGE_ID
            or semantic.get("source_image_id") != native.IMAGE_ID): raise ValueError("Reference/semantic native image or joint names differ")
    return True


def run(root, report, path, persist):
    records, public = public_inputs(root); helpers = helper_identities(); output = path.parent
    asset_path, acquisition, asset = native.depth_model.model_asset(root)
    model_path = root/"weights/mhr/mhr_model.pt"; tool = root/"vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
    native.regular(model_path, native.REFERENCE_MODEL_SHA256); native.regular(tool, native.CONVERTER_SHA256)
    semantic_path = root/"results/mhr-finger-semantics-v4.json"; semantic_receipt = native.regular(semantic_path)
    semantic = json.loads(semantic_path.read_text()); native.regions_helper.require_semantic_report(semantic)
    if report["image_id"] != native.IMAGE_ID or semantic.get("source_image_id") != native.IMAGE_ID: raise ValueError("Actual native semantic image required")
    report.update(**public, helper_source_sha256=helpers, script_sha256=sha256(Path(__file__)), acquisition_report=acquisition,
        MoGe_model_asset=asset, semantic_report_sha256=semantic_receipt["sha256"], model_sha256=native.REFERENCE_MODEL_SHA256,
        converter_sha256=native.CONVERTER_SHA256, phase="model_load"); persist()
    if "torch" in sys.modules: raise ValueError("CUBLAS setup must precede Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import moge
    from moge.model import v2 as depth_module
    from moge.model.v2 import MoGeModel
    from moge.utils import geometry_torch
    from PIL import Image
    if not torch.cuda.is_available(): raise RuntimeError("Actual offline CUDA required")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    distribution = metadata.distribution("moge"); direct = distribution.read_text("direct_url.json")
    source = native.depth_model.installed_source(Path(moge.__file__).parent, json.loads(direct) if direct else {})
    if sha256(Path(geometry_torch.__file__)) != native.joint.GEOMETRY_SHA or depth_module.recover_focal_shift is not geometry_torch.recover_focal_shift:
        raise ValueError("Original MoGe focal solver/source differs")
    model, estimator, faces, body_source = native.human.load_model(root, torch)
    joints = model.head_pose.mhr.get_joint_names(); lbs = model.head_pose.mhr.get_lbsw()
    if joints != semantic["joint_names"] or not isinstance(lbs, tuple) or len(lbs) != 2: raise ValueError("Actual native LBS/joint metadata differs")
    regions = native.regions_helper.lbs_regions(*(v.detach().cpu().numpy() for v in lbs), joints)
    report.update(body_model=body_source, MoGe_source=source, focal_geometry_source_sha256=native.joint.GEOMETRY_SHA,
        torch_version=str(torch.__version__), seed=0, TF32=False, CUBLAS_WORKSPACE_CONFIG=":4096:8", joint_names=joints,
        hand_regions_native_verified=True, hand_region_sha256={n: hashlib.sha256(regions[s]["vertex_mask"].tobytes()).hexdigest()
        for n, s in (("left", "l"), ("right", "r"))}, phase="raw_inference"); persist()
    network = MoGeModel.from_pretrained(str(asset_path)).cuda().eval(); camera = torch.as_tensor(CAMERA_K.astype(np.float32), device="cuda")[None]
    for record in records:
        report.update(active_file=record["file"]); persist()
        with Image.open(record["path"]) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (1024, 768): raise ValueError("Original RGB decode differs")
            rgb = np.asarray(image).copy()
        hm, om = (native.joint.read_mask(record[label+"_mask_path"], Image) for label in ("human", "object"))
        box, prompt = native.human.derived_bbox(rgb, hm)
        with torch.inference_mode(): predictions = estimator.process_one_image(img=rgb, bboxes=box[None], masks=prompt, cam_int=camera, inference_type="body")
        if not isinstance(predictions, list) or len(predictions) != 1: raise ValueError("Exactly one native Body actor required")
        values, parity = native.human.decode_prediction(torch, model, predictions[0], "body")
        report["body_calls_completed"] += 1; report["raw_parity_head_calls_completed"] += 1
        kp, r = raw_keypoint_decode(torch, model.head_pose, predictions[0], values); report["raw_keypoint_head_calls_completed"] += 1
        if not np.isclose(float(values["focal_length"]), 1280., atol=1e-4, rtol=1e-6): raise ValueError("Body ignored fixed camera")
        tensor = torch.from_numpy(rgb).cuda().permute(2, 0, 1).float()/255
        depth = native.joint.checked_depth_infer(torch, depth_module, network, tensor, float(np.degrees(2*np.arctan(1024/2560))),
            report["native_focal_solver_calls"], {k: record[k] for k in ("file", "clip_index", "frame_index")}); report["MoGe_calls_completed"] += 1
        z, points, valid, normalized = (depth[k][0].detach().cpu().numpy() for k in ("depth", "points", "mask", "intrinsics"))
        checks = native.joint.pointmap_contract(z, points, valid, normalized, CAMERA_K)
        silhouette, rendered = native.raster_camera_mesh(values["vertices_camera_m"], faces, CAMERA_K, 1024, 768)
        raw = {k: values[k] for k in native.BLOCKS} | dict(raw_depth=z, raw_points=points, validity=valid,
            rendered_depth=rendered.cpu().numpy(), silhouette=silhouette.cpu().numpy(), human_mask=hm > 0, object_mask=om > 0,
            raw_vertices_camera_m=values["vertices_camera_m"], raw_joints_camera_m=values["joints_camera_m"], raw_keypoints_camera_m=kp,
            raw_joint_global_rotations=r, human_faces=faces, camera_K=CAMERA_K.copy(),
            clip_index=np.array(record["clip_index"], np.int64), frame_index=np.array(record["frame_index"], np.int64))
        validate_raw(raw, record); report["raw_outputs"].append(native.save(output, "raw", record, raw))
        report["calls"].append(dict(file=record["file"], bbox_xyxy=box.tolist(), native_forward_errors=parity,
            pointmap_checks=checks, decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())); persist()
        del predictions, depth, tensor, raw, values
    del network; torch.cuda.empty_cache()
    raw_frames = native.frozen_rows(output, report["raw_outputs"], records, "raw", validate_raw)
    report.update(raw_frozen_before_shared=True, phase="shared_native_decode"); persist(); pairs = []
    for record, raw in zip(records, raw_frames):
        first = raw_frames[record["clip_index"]*5]; v, kp, j, controls, r = shared_decode(torch, model.head_pose, raw, first)
        report["shared_head_calls_completed"] += 1
        pair = dict(raw_vertices_camera_m=raw["raw_vertices_camera_m"], shared_vertices_camera_m=v, raw_keypoints_camera_m=raw["raw_keypoints_camera_m"],
            shared_keypoints_camera_m=kp, raw_joints_camera_m=raw["raw_joints_camera_m"], shared_joints_camera_m=j,
            raw_joint_global_rotations=raw["raw_joint_global_rotations"], shared_joint_global_rotations=r, human_faces=faces,
            hand_mask_left=regions["l"]["vertex_mask"], hand_mask_right=regions["r"]["vertex_mask"], camera_K=CAMERA_K.copy(), pred_cam_t=raw["pred_cam_t"],
            raw_model_controls=raw["mhr_model_params"], shared_model_controls=controls, raw_shape_params=raw["shape_params"], shared_shape_params=first["shape_params"],
            raw_scale_params=raw["scale_params"], shared_scale_params=first["scale_params"], expression=np.zeros(72, np.float32), clip_index=raw["clip_index"], frame_index=raw["frame_index"])
        validate_pair(pair, record, raw); pairs.append(pair)
    validate_clip_constants(pairs, raw_frames)
    for record, pair in zip(records, pairs): report["paired_outputs"].append(native.save(output, "paired", record, pair))
    pairs = native.frozen_rows(output, report["paired_outputs"], records, "paired", validate_pair)
    native.validate_region_masks(pairs, report); validate_clip_constants(pairs, raw_frames)
    report.update(paired_frozen_before_reference=True, phase="independent_reference_forward"); persist()
    spec = importlib.util.spec_from_file_location("world_reward_keypoint_official_reference", tool); official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    reference = official.MHR(str(model_path), "cuda", chunk=16, precision="float32")
    if not np.array_equal(reference.model.character_torch.mesh.faces.detach().cpu().numpy(), faces): raise ValueError("Reference/native topology differs")
    controls = np.stack([p["shared_model_controls"] for p in pairs]); shapes = np.stack([p["shared_shape_params"] for p in pairs])
    vertices_mm, joints_m = reference.run(torch.tensor(controls[:, :136], dtype=torch.float64, device="cuda"),
        torch.tensor(np.c_[controls[:, 136:], shapes], dtype=torch.float64, device="cuda")); report["official_reference_calls"] += 1
    report["reference_fidelity"] = native.reference_errors(vertices_mm.detach().cpu().numpy(), joints_m.detach().cpu().numpy(),
        np.stack([p["shared_vertices_camera_m"] for p in pairs]), np.stack([p["shared_joints_camera_m"] for p in pairs]), np.stack([p["pred_cam_t"] for p in pairs]))
    report["official_reference_per_frame_mean_mm"] = report["reference_fidelity"]["per_frame_mean_mm"]
    native.frozen_rows(output, report["raw_outputs"], records, "raw", validate_raw)
    native.frozen_rows(output, report["paired_outputs"], records, "paired", validate_pair)
    if public_inputs(root) != (records, public) or helper_identities() != helpers: raise ValueError("Frozen public/source bytes changed")
    validate_producer_bindings(root, report)
    if {p.name for p in output.iterdir()} != {"raw", "paired", "report.json"}: raise ValueError("Unexpected baseline artifacts")
    counts = {k: report[k] for k in ("body_calls_completed", "MoGe_calls_completed", "raw_parity_head_calls_completed", "raw_keypoint_head_calls_completed", "shared_head_calls_completed")}
    if any(v != 15 for v in counts.values()) or report["official_reference_calls"] != 1: raise ValueError("Exact15 native call coverage required")
    report.update(status="pass", phase="complete", frames=15, all_cases_retained=True, sources_assets_rechecked=True,
        conversion_fidelity_verified=True, actual_body_inference=True, actual_MoGe_inference=True, actual_shared_native_forward=True,
        file_inventory=[r["artifact"] for r in report["raw_outputs"]+report["paired_outputs"]]+["report.json"])
    report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); output = root/BASE/"baseline_v1"; revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if (platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"} or os.geteuid() != 1000
            or root != Path("/srv/scenesmith/world-reward") or output.resolve() != output.absolute() or not output.is_dir() or any(output.iterdir())
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or image != native.IMAGE_ID): raise ValueError("Fresh reserved offline native baseline required")
    (output/"raw").mkdir(); (output/"paired").mkdir(); path = output/"report.json"
    report = dict(stage=STAGE, status="fail", phase="public_integrity", producer_revision=revision, image_id=image, network="none",
        private_truth_read=False, challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        body_inference_type="body", camera_K=CAMERA_K.tolist(), focal_fitted=False, scale_fit=False, inverse_fit_performed=False,
        identity_source="first_original_RGB_shape45_and_scale28_per_five_frame_clip", expression_zero=True,
        camera_translation_rule="root_zero_native_and_official_YZ_flip_then_original_pred_cam_t_once", keypoint_count=308, joint_count=127,
        budget_seconds=BUDGET, raw_outputs=[], paired_outputs=[], calls=[], native_focal_solver_calls=[], body_calls_completed=0,
        MoGe_calls_completed=0, raw_parity_head_calls_completed=0, raw_keypoint_head_calls_completed=0, shared_head_calls_completed=0,
        official_reference_calls=0, raw_frozen_before_shared=False, paired_frozen_before_reference=False, accuracy_verified=False,
        full_HOI_verified=False, adoption_performed=False, DWPose_inference_performed=False)
    started = time.perf_counter()
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole fresh baseline exceeded600s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); run(root, report, path, persist)
        except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
