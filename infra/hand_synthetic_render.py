"""Render-only own MHR RGB validation: truth stays outside public inputs.

Procedural diffuse materials are not photorealistic validation data. No model
inference, masks for inference, challenge records, or quality claims occur here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
from world_reward.data import sha256
import mhr_finger_semantics_gate as semantics
from camera_render import _opencv_camera, project_camera_points, PYTORCH3D_REVISION

WIDTH, HEIGHT = 1024, 768
CASE_NAMES = ("neutral", "left_bend", "right_bend", "oblique_bimanual", "left_occluded", "edge_crop")
FLEXION_NAMES = ("thumb2_rz", "thumb3_rz", "index1_rz", "index2_rz", "index3_rz", "middle1_rz", "middle2_rz", "middle3_rz",
                 "ring1_rz", "ring2_rz", "ring3_rz", "pinky1_rz", "pinky2_rz", "pinky3_rz")


def public_manifest(records):
    if len(records) != 6: raise ValueError("Require all six frozen RGB cases")
    images = []
    for index, digest in enumerate(records):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Require PNG SHA256 digests")
        images.append({"file": f"case_{index:03d}.png", "sha256": digest, "width": WIDTH, "height": HEIGHT})
    return {"schema": "world-reward-hands-rgb-inputs-v1", "images": images}


def require_semantic_report(report):
    expected = {"stage": "own_reference_mhr_finger_semantics_support", "status": "pass", "phase_A_verified": True,
                "phase_B_verified": True, "named_finger_joint_partition_verified": True,
                "excluded_joint_invariance_verified": True, "correctives_skeleton_bit_identical": True,
                "model_sha256": semantics.MODEL_SHA, "body_checkpoint_sha256": semantics.BODY_SHA,
                "body_revision": semantics.BODY_REVISION, "network": "none", "challenge_inputs_used": False,
                "script_sha256": sha256(Path(semantics.__file__)), "deterministic_algorithms": True,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF32": False}
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise ValueError("Require actual passing SHA-bound current semantic producer, not same snapshot path")


def camera_from_neutral(vertices):
    """Manufacture a fixed camera from own neutral extent, not inferred evidence."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or len(v) < 3:
        raise ValueError("Require finite own neutral vertices")
    center = (v.min(0) + v.max(0)) / 2
    f = float(np.hypot(WIDTH, HEIGHT))
    relative = v - center
    distance = max(np.max(f*np.abs(relative[:, 0])/(WIDTH*.40)-relative[:, 2]),
                   np.max(f*np.abs(relative[:, 1])/(HEIGHT*.40)-relative[:, 2])) + .15
    if distance <= 0: raise ValueError("Own neutral mesh has no camera extent")
    K = np.array([[f, 0., WIDTH/2], [0., f, HEIGHT/2], [0., 0., 1.]])
    return K, np.array([-center[0], -center[1], distance-center[2]])


def named_controls(names, limits):
    """Actual getter limits constrain a fixed deterministic flexion recipe."""
    bounds = np.asarray(limits)
    if (len(names) != 249 or len(set(names)) != len(names) or bounds.shape != (249, 2)
            or bounds.dtype.kind != "f" or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Require actual [249,2] min/max parameter limits")
    if np.any(bounds[:204, 0] > 0) or np.any(bounds[:204, 1] < 0):
        raise ValueError("Own exact neutral controls must obey every exposed bound")
    controls = np.zeros((6, 204), dtype=np.float32)
    changed = []
    for side, rows in (("l", (1, 3, 4, 5)), ("r", (2, 3, 4, 5))):
        for suffix in FLEXION_NAMES:
            name = f"{side}_{suffix}"
            if name not in names[:204]: raise ValueError(f"Required named own flexion control absent: {name}")
            column = names.index(name); lower, upper = bounds[column]
            # Positive curl direction if allowed; otherwise the negative range.
            value = min(.45, float(upper)*.4) if upper > .05 else max(-.45, float(lower)*.4)
            if not np.isfinite(value) or abs(value) < .02 or not lower <= value <= upper:
                raise ValueError(f"Named own flexion range insufficient: {name}")
            controls[list(rows), column] = value
            changed.append({"name": name, "column": column, "value_rad": value, "limits": [float(lower), float(upper)]})
    return controls, changed


def lbs_regions(indices, weights, joint_names):
    indices, weights = np.asarray(indices), np.asarray(weights)
    if (indices.shape != weights.shape or indices.ndim != 2 or indices.shape[0] != 18439
            or indices.dtype.kind not in "iu" or weights.dtype.kind != "f" or not np.isfinite(weights).all()
            or np.any(indices < 0) or np.any(indices >= len(joint_names)) or np.any(weights < 0)
            or not np.allclose(weights.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Require actual normalized [18439,K] LBS joint indices/weights")
    result = {}
    for side in "lr":
        joints = [j for j, name in enumerate(joint_names)
                  if name == f"{side}_wrist" or re.fullmatch(fr"{side}_(thumb|index|middle|ring|pinky)([0-3]|_?null)", name)]
        if not joints: raise ValueError("Actual semantic hand LBS joints absent")
        mass = np.where(np.isin(indices, joints), weights, 0).sum(1)
        region = mass >= .5
        if np.count_nonzero(region) < 50: raise ValueError("Actual LBS hand region unexpectedly absent")
        result[side] = {"joint_indices": joints, "vertex_mask": region, "weight_mass": mass}
    if np.any(result["l"]["vertex_mask"] & result["r"]["vertex_mask"]):
        raise ValueError("Dominant LBS hand regions overlap")
    return result


def render_rgb(torch, vertices, faces, colors, K):
    from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer, PointLights,
                                   RasterizationSettings, SoftPhongShader, TexturesVertex)
    from pytorch3d.structures import Meshes
    mesh = Meshes(verts=[torch.as_tensor(vertices, device="cuda", dtype=torch.float32)],
                  faces=[torch.as_tensor(faces, device="cuda", dtype=torch.int64)],
                  textures=TexturesVertex(verts_features=torch.as_tensor(colors, device="cuda", dtype=torch.float32)[None]))
    camera = _opencv_camera(torch, K, WIDTH, HEIGHT)
    settings = RasterizationSettings(image_size=(HEIGHT, WIDTH), blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, cull_backfaces=False, clip_barycentric_coords=False,
        cull_to_frustum=False, max_faces_per_bin=len(faces))
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=settings)
    lights = PointLights(device="cuda", location=((0., -.8, 0.),), ambient_color=((.35,)*3,),
                         diffuse_color=((.65,)*3,), specular_color=((0.,)*3,))
    shader = SoftPhongShader(device="cuda", cameras=camera, lights=lights,
        materials=Materials(device="cuda", specular_color=((0.,)*3,)),
        blend_params=BlendParams(background_color=(.72, .72, .72)))
    fragments = rasterizer(mesh)
    rgba = shader(fragments, mesh)
    pixels = torch.round(rgba[0, ..., :3].clamp(0, 1)*255).to(torch.uint8).cpu().numpy()
    return pixels, fragments.pix_to_face[0, ..., 0].cpu().numpy(), fragments.zbuf[0, ..., 0].cpu().numpy()


def visibility_preflight(vertices, faces, regions, face_image, K):
    evidence = {}
    for side in "lr":
        mask = regions[side]["vertex_mask"]
        pixels = project_camera_points(vertices[mask], K)
        span = np.ptp(pixels, axis=0)
        region_faces = np.flatnonzero(np.count_nonzero(mask[faces], axis=1) >= 2)
        visible = np.count_nonzero(np.isin(face_image, region_faces))
        evidence[side] = {"projected_span_px": span.tolist(), "visible_hand_pixels": int(visible)}
        if np.any(span < 24) or visible < 64 or np.any(pixels < 0) or np.any(pixels >= [WIDTH, HEIGHT]):
            raise ValueError("Own first-two-case hand anatomy/visibility preflight failed, not model accuracy")
    return evidence


def camera_scene(vertices, faces, regions, translation, case_index):
    """Known synthesis transforms/occluder; never an inference observation API."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec([0., .45 if case_index == 3 else 0., 0.]).as_matrix()
    t = np.asarray(translation).copy()
    if case_index == 5: t[0] -= t[2]*.28
    camera = np.asarray(vertices) @ R.T + t
    colors = np.tile([.75, .75, .73], (len(camera), 1))
    # Plain gray upper/lower garment bands; no finger/side identity colors.
    low, high = vertices[:, 1].min(), vertices[:, 1].max()
    height = (vertices[:, 1]-low)/(high-low)
    clothing = (height > .24) & (height < .80)
    clothing &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    colors[clothing] = [.35, .35, .35]
    scene, topology = camera.copy(), np.asarray(faces).copy()
    occluder = np.empty((0, 3), dtype=float)
    if case_index == 4:
        hand = camera[regions["l"]["vertex_mask"]]; center = (hand.min(0)+hand.max(0))/2
        extent = np.maximum(np.ptp(hand, axis=0)*[.8, .8, .3], [.04, .04, .025])
        center[2] = hand[:, 2].min()-.08
        corners = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                            [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]])
        occluder = center + corners*extent/2
        box_faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
                              [1,2,6], [1,6,5], [2,3,7], [2,7,6], [3,0,4], [3,4,7]])
        topology = np.concatenate((topology, box_faces+len(camera)))
        scene = np.concatenate((scene, occluder)); colors = np.concatenate((colors, np.tile([.28]*3, (8, 1))))
    if not np.isfinite(scene).all() or np.min(scene[:, 2]) <= .01:
        raise ValueError("Own scene crosses camera near plane")
    return camera, scene, topology, colors, R, t, occluder


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA renderer with network none")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", ""); image = os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable renderer revision and image ID")
    destination = root / "validation/hands_rgb_v1"
    reserved = os.environ.get("WR_RENDER_OUTPUT_RESERVED") == "1"
    if destination.is_symlink() or (destination.exists() and
            (not reserved or not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError("Synthetic dataset is frozen; never overwrite")
    if reserved and not destination.is_dir(): raise ValueError("Reserved dataset must be a new empty mount")
    if any(p.is_symlink() for p in destination.parents if p.is_relative_to(root)):
        raise ValueError("Require regular validation parents")
    if not reserved: destination.mkdir()
    public = destination / "inputs"; private = destination / "eval_private"
    public.mkdir(); private.mkdir(mode=0o700)
    with (private / "render-report.json").open("x") as handle:
        started = time.perf_counter()
        report = {"stage": "own_mhr_rendered_rgb_validation", "status": "fail", "phase": "prerequisites",
                  "script_sha256": sha256(Path(__file__)), "code_revision": revision, "image_id": image,
                  "model_sha256": semantics.MODEL_SHA, "body_checkpoint_sha256": semantics.BODY_SHA,
                  "network": "none", "challenge_inputs_used": False, "synthetic_truth_used_for_rendering_only": True,
                  "inference_performed": False, "accuracy_verified": False, "photorealism_verified": False,
                  "hand_labels_for_test_records_created": False, "budget_seconds": 120, "cases": [],
                  "fixture": {"case_names": list(CASE_NAMES), "width": WIDTH, "height": HEIGHT,
                              "shared_identity": "zero45", "shared_scales": "zero68", "expression": "zero72",
                              "camera_rule": "one neutral whole-mesh extent, no per-case adjustment",
                              "RGB_material": "plain gray garments/white diffuse skin, no identity colors",
                              "randomness": "none", "visibility_gates": {"first_cases": [0, 1], "minimum_span_px": 24, "minimum_pixels": 64}},
                  "coordinate_rule": "rawcm*diag(1,-1,-1)/100 then own camera R,t exactly once"}
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started
            handle.seek(0); handle.write(json.dumps(report, allow_nan=False)+"\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("Render-only synthesis exceeded120s")
        previous_alarm = signal.signal(signal.SIGALRM, expired); previous_term = signal.signal(signal.SIGTERM, expired); signal.alarm(120)
        try:
            persist()
            prerequisite = root / "results/mhr-finger-semantics-v3.json"; prerequisite_sha = sha256(prerequisite)
            semantics.regular_hash(prerequisite, root, prerequisite_sha)
            semantic = json.loads(prerequisite.read_text()); require_semantic_report(semantic)
            model_path = root / "weights/mhr/mhr_model.pt"
            semantics.regular_hash(model_path, root, semantics.MODEL_SHA, 696110248)
            if "torch" in __import__("sys").modules: raise RuntimeError("Require CUBLAS setup before torch import")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            semantics.strict_reference_runtime(torch)
            import pytorch3d
            from PIL import Image
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9":
                raise RuntimeError("Require CUDA and audited PyTorch3D0.7.9")
            model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            names, joints = model.get_parameter_names(), model.get_joint_names()
            getter = model.get_parameter_limits().detach().cpu().numpy()
            controls, changed = named_controls(names, getter)
            faces = model.character_torch.mesh.faces.detach().cpu().numpy()
            if faces.shape != (36874, 3) or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= 18439):
                raise ValueError("Actual reference topology differs")
            if joints != semantic["joint_names"] or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72):
                raise ValueError("Actual reference names/external ABI differ from semantic prerequisite")
            lbs = model.get_lbsw()
            if not isinstance(lbs, tuple) or len(lbs) != 2: raise ValueError("Require exported real LBS indices/weights tuple")
            indices, weights = (v.detach().cpu().numpy() for v in lbs)
            regions = lbs_regions(indices, weights, joints)
            identity = torch.as_tensor(semantics.identity_rows(np.zeros(45, np.float32), 6), device="cuda")
            expression = torch.zeros(6, 72, device="cuda")
            params = torch.as_tensor(controls, device="cuda"); original = params.clone()
            with torch.inference_mode():
                raw_v, skeleton = model(identity, params, expression, True)
            torch.cuda.synchronize()
            if not torch.equal(params, original) or torch.count_nonzero(identity) or torch.count_nonzero(expression):
                raise ValueError("Rendering modified fixed own truth controls")
            raw_v, skeleton = raw_v.cpu().numpy(), skeleton.cpu().numpy()
            semantics.check_geometry(raw_v, skeleton, 127)
            flip = np.diag([1., -1., -1.]); canonical = raw_v @ flip / 100
            K, translation = camera_from_neutral(canonical[0])
            report.update(phase="rendering", prerequisite_report_sha256=prerequisite_sha, torch=torch.__version__,
                          pytorch3d=pytorch3d.__version__, pytorch3d_revision=PYTORCH3D_REVISION,
                          camera_helper_sha256=sha256(Path(__file__).with_name("camera_render.py")),
                          joint_names=joints, parameter_names=names, named_changed_controls=changed,
                          parameter_limits_shape=list(getter.shape), LBS_shape=list(indices.shape))
            with (private / "rig.npz").open("xb") as stream:
                np.savez_compressed(stream, faces=faces, lbs_indices=indices, lbs_weights=weights,
                    hand_vertex_mask_left=regions["l"]["vertex_mask"], hand_vertex_mask_right=regions["r"]["vertex_mask"],
                    hand_weight_mass_left=regions["l"]["weight_mass"], hand_weight_mass_right=regions["r"]["weight_mass"], parameter_limits=getter)
            report["rig_sha256"] = sha256(private / "rig.npz"); persist()
            images = []
            from scipy.spatial.transform import Rotation
            for index in range(6):
                report["active_case"] = index; persist()
                camera, scene, topology, colors, R, t, occluder = camera_scene(canonical[index], faces, regions, translation, index)
                with torch.inference_mode(): rgb, face_image, depth = render_rgb(torch, scene, topology, colors, K)
                torch.cuda.synchronize()
                if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8: raise ValueError("Invalid rendered RGB grid")
                camera_joints = skeleton[index, :, :3] @ flip @ R.T / 100 + t
                orientation = Rotation.from_quat(skeleton[index, :, 3:7]).as_matrix()
                orientation = R[None] @ flip[None] @ orientation
                filename = f"case_{index:03d}"
                truth = private / (filename+".npz")
                with truth.open("xb") as stream:
                    np.savez_compressed(stream, pose=controls[index, :136], scales=controls[index, 136:], shape=np.zeros(45, np.float32),
                        expression=np.zeros(72, np.float32), vertices_camera_m=camera, joints_camera_m=camera_joints,
                        global_quaternion_xyzw=Rotation.from_matrix(orientation).as_quat(), global_scale=skeleton[index, :, 7],
                        camera_K=K, camera_R=R, camera_t=t, occluder_vertices_camera_m=occluder,
                        visible_face_indices=face_image, scene_depth_m=depth)
                record = {"case_index": index, "truth_sha256": sha256(truth), "phase": "preflight"}
                report["cases"].append(record); persist()
                if index < 2: record["visibility"] = visibility_preflight(camera, faces, regions, face_image, K)
                with (public / (filename+".png")).open("xb") as stream: Image.fromarray(rgb).save(stream, format="PNG")
                images.append(sha256(public / (filename+".png")))
                record.update(phase="rendered", rgb_sha256=images[-1], decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                persist(); print(json.dumps({"phase": "rendered", "case_index": index}), flush=True)
                if time.perf_counter()-started > 120: raise TimeoutError("Synthetic rendering exceeded120s")
            semantics.regular_hash(prerequisite, root, prerequisite_sha); semantics.regular_hash(model_path, root, semantics.MODEL_SHA, 696110248)
            with (public / "manifest.json").open("x") as stream: stream.write(json.dumps(public_manifest(images))+"\n")
            report.update(status="pass", phase="complete", public_manifest_sha256=sha256(public / "manifest.json"),
                          all_truth_private=True, generated_GT_masks_supplied_to_inference=False)
            report.pop("active_case", None)
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}; raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term)
            persist(); print(json.dumps({k: report[k] for k in ("stage", "status", "phase", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()
