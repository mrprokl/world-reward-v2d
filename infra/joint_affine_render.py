"""New independent human/bottle RGB cohort for shared affine grounding research.

All manufacturing identity, camera and object offsets stay in eval_private.
Public inputs contain only 18 ordered RGB identities. This is diffuse synthetic
data, not photorealistic evidence, inference, parameter tuning or adoption.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
import hand_synthetic_render as render
from joint_rgb_render import bottle_mesh, TRUTH_KEYS
from world_reward.data import sha256

WIDTH, HEIGHT, CLIPS, FRAMES, BUDGET = 1024, 768, 3, 6, 180
SCHEMA = "world-reward-joint-affine-rgb-v1"
FOCALS = (1024., 1408., 1792.)
SHAPES = (-.4, 0., .4)
BOTTLE_SCALES = (1.1, .8, 1.25)
CLOTH_COLORS = ((.28, .39, .47), (.44, .32, .25), (.31, .42, .29))
OFFSETS = ((-.055, .015, 0.), (.25, .015, -.25), (-.035, .015, .25))


def public_manifest(digests):
    if len(digests) != CLIPS*FRAMES or any(not isinstance(d, str) or not re.fullmatch("[0-9a-f]{64}", d) for d in digests):
        raise ValueError("Require all18 ordered RGB SHA256 identities")
    return {"schema": SCHEMA, "images": [dict(file=f"clip_{c:02d}_frame_{f:03d}.png", sha256=digests[c*FRAMES+f],
        width=WIDTH, height=HEIGHT) for c in range(CLIPS) for f in range(FRAMES)]}


def named_controls(names, limits):
    """Fresh prescribed named arm/finger controls, bounded without clipping.

    Global body yaw is an explicit known synthesis camera-frame rigid transform;
    no guessed native root column is used. Shape45 is clip-constant and every
    pose's68 scale controls remain zero.
    """
    bounds = np.asarray(limits)
    if (not isinstance(names, list) or len(names) != 249 or len(set(names)) != 249
            or any(not isinstance(n, str) or not n for n in names) or np.ma.isMaskedArray(limits)
            or bounds.shape != (249, 2) or bounds.dtype.kind != "f" or np.isnan(bounds).any()
            or np.any(bounds[:, 0] > bounds[:, 1]) or np.any(bounds[:204, 0] > 0) or np.any(bounds[:204, 1] < 0)):
        raise ValueError("Require actual unique249 names/bounds and legal neutral controls")
    controls = np.zeros((CLIPS*FRAMES, 204), np.float32); shape = np.zeros((CLIPS*FRAMES, 45), np.float32); changes = []
    for clip in range(CLIPS):
        if not bounds[204, 0] <= SHAPES[clip] <= bounds[204, 1]:
            raise ValueError("Fresh fixed identity coefficient exceeds actual reference bounds")
        shape[clip*FRAMES:(clip+1)*FRAMES, 0] = SHAPES[clip]
        side = "r" if clip == 1 else "l"
        for frame in range(FRAMES):
            recipe = [(f"{side}_uparm_ry", .16+.04*frame), (f"{side}_elbow_bend", .26+.05*frame)]
            recipe += [(f"{side}_{finger}1_rz", .22+.025*frame) for finger in ("index", "middle", "ring", "pinky")]
            for name, value in recipe:
                if name not in names[:136]: raise ValueError("Required actual pose name absent: "+name)
                column = names.index(name)
                if not bounds[column, 0] <= value <= bounds[column, 1]:
                    raise ValueError("Fresh prescribed control exceeds actual bounds: "+name)
                controls[clip*FRAMES+frame, column] = value
                changes.append(dict(clip_index=clip, frame_index=frame, name=name, column=column, value=value))
    return controls, shape, changes


def fixed_camera(neutral):
    """One known framing translation for new identities, never per-frame fit."""
    v = np.asarray(neutral)
    if np.ma.isMaskedArray(neutral) or v.shape != (CLIPS, 18439, 3) or v.dtype.kind != "f" or not np.isfinite(v).all():
        raise ValueError("Require the three own finite neutral identity meshes")
    points = v.reshape(-1, 3); center = (points.min(0)+points.max(0))/2; relative = points-center
    distance = max(np.max(max(FOCALS)*np.abs(relative[:, 0])/(WIDTH*.31)-relative[:, 2]),
                   np.max(max(FOCALS)*np.abs(relative[:, 1])/(HEIGHT*.31)-relative[:, 2]))+.35
    if not np.isfinite(distance) or distance <= 0: raise ValueError("Invalid own neutral camera extent")
    return np.array([-center[0], -center[1], distance-center[2]])


def scene(human, human_faces, regions, translation, clip, frame):
    """Fresh known yaw/depth/context; shared bottle scale is baked once."""
    from scipy.spatial.transform import Rotation
    if type(clip) is not int or type(frame) is not int or not 0 <= clip < CLIPS or not 0 <= frame < FRAMES:
        raise ValueError("Require one of the18 prescribed scene indices")
    R = Rotation.from_rotvec([0., (-.25, .20, -.12)[clip]+.045*(frame-2.5), 0.]).as_matrix()
    t = np.asarray(translation)+[.012*(frame-2.5), 0., .03*(frame-2.5)]
    hv = human@R.T+t; side = "r" if clip == 1 else "l"
    hand = hv[regions[side]["vertex_mask"]]
    center = (hand.min(0)+hand.max(0))/2 + np.asarray(OFFSETS[clip])
    canonical, faces = bottle_mesh(); canonical = canonical*BOTTLE_SCALES[clip]
    object_R = Rotation.from_rotvec([.03*(frame-2.5), .04*frame, -.07*(frame-2.5)]).as_matrix()
    ov = canonical@object_R.T+center
    hc = np.tile([.75, .75, .73], (len(hv), 1)); low, high = human[:, 1].min(), human[:, 1].max()
    if high <= low: raise ValueError("Own human has no clothing extent")
    band = (human[:, 1]-low)/(high-low); cloth = (band>.24)&(band<.80)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    hc[cloth] = CLOTH_COLORS[clip]
    oc = np.tile([.25, .40, .32], (len(ov), 1)); oc[np.arange(len(ov)) >= 160] = [.30, .30, .29]
    vertices, topology, colors = np.r_[hv, ov], np.r_[human_faces, faces+len(hv)], np.r_[hc, oc]
    if not np.isfinite(vertices).all() or vertices[:, 2].min() <= .01:
        raise ValueError("New own scene crosses the camera near plane")
    K = np.array([[FOCALS[clip], 0., WIDTH/2], [0., FOCALS[clip], HEIGHT/2], [0., 0., 1.]])
    return hv, ov, faces, vertices, topology, colors, K


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require offline remote CUDA synthesis")
    root = Path(os.environ["WR_ROOT"]); dest = root/"validation/joint_affine_rgb_v1"
    if dest.is_symlink() or dest.resolve() != dest.absolute() or not dest.is_dir() or any(dest.iterdir()):
        raise FileExistsError("Require exclusively reserved empty J3 root")
    revision, image = os.environ.get("WR_CODE_REVISION", ""), os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable renderer source and image")
    public, private = dest/"inputs", dest/"eval_private"; public.mkdir(); private.mkdir(mode=0o700)
    report = dict(stage="own_joint_affine_human_object_rgb_render", status="fail", phase="prerequisites", cases=[],
        budget_seconds=BUDGET, code_revision=revision, image_id=image, script_sha256=sha256(Path(__file__)),
        render_helper_sha256=sha256(Path(render.__file__)), bottle_helper_sha256=sha256(Path(bottle_mesh.__code__.co_filename)),
        network="none", challenge_inputs_used=False, synthetic_truth_used_for_rendering_only=True, inference_performed=False,
        accuracy_verified=False, photorealism_verified=False, all_truth_private=True, generated_GT_masks_supplied_to_inference=False,
        true_focals=list(FOCALS), shape_first_coefficient=list(SHAPES), fixed_scales="zero68", facial_expression="zero72",
        object_scale_baked_once=list(BOTTLE_SCALES), clip_contexts=["near", "distant_lateral_front", "back_partially_occluded"],
        camera_framing="one neutral union extent across fixed identities; no per-frame camera fit", truth_keys=sorted(TRUTH_KEYS))
    start = time.perf_counter()
    with (private/"render-report.json").open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole new J3 synthesis exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); prereq = root/"results/mhr-finger-semantics-v4.json"; prerequisite_sha = sha256(prereq)
            semantic = json.loads(prereq.read_text()); render.require_semantic_report(semantic)
            model_path = root/"weights/mhr/mhr_model.pt"
            render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, 696110248)
            if "torch" in __import__("sys").modules: raise RuntimeError("CUBLAS setup must precede Torch import")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Require actual audited CUDA raster")
            with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            names, joints = model.get_parameter_names(), model.get_joint_names()
            controls, shape, changes = named_controls(names, model.get_parameter_limits().cpu().numpy())
            faces = model.character_torch.mesh.faces.cpu().numpy()
            if (faces.shape != (36874, 3) or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= 18439)
                    or joints != semantic["joint_names"] or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72)):
                raise ValueError("Pinned reference topology/metadata ABI differs")
            regions = render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()), joints)
            p, identity = torch.as_tensor(controls, device="cuda"), torch.as_tensor(shape, device="cuda")
            before_p, before_identity = p.clone(), identity.clone(); expression = torch.zeros(CLIPS*FRAMES, 72, device="cuda")
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                neutral, neutral_sk = model(identity[::FRAMES], torch.zeros(CLIPS, 204, device="cuda"), expression[::FRAMES], True)
                raw, skeleton = model(identity, p, expression, True)
            torch.cuda.synchronize()
            if not torch.equal(p, before_p) or not torch.equal(identity, before_identity) or torch.count_nonzero(expression):
                raise ValueError("Reference changed supplied fixed own identity/controls")
            render.semantics.check_geometry(raw.cpu().numpy(), skeleton.cpu().numpy(), 127)
            render.semantics.check_geometry(neutral.cpu().numpy(), neutral_sk.cpu().numpy(), 127)
            flip = np.diag([1., -1., -1.]); human = raw.cpu().numpy()@flip/100
            translation = fixed_camera(neutral.cpu().numpy()@flip/100)
            report.update(phase="rendering", model_sha256=render.semantics.MODEL_SHA, semantic_report_sha256=prerequisite_sha,
                named_controls=changes, parameter_names=names, joint_names=joints, torch=torch.__version__, pytorch3d=pytorch3d.__version__,
                pytorch3d_revision=render.PYTORCH3D_REVISION, camera_helper_sha256=sha256(Path(__file__).with_name("camera_render.py")))
            with (private/"rig.npz").open("xb") as handle:
                np.savez_compressed(handle, native_controls=controls, identity_coefficients=shape, faces=faces,
                    parameter_limits=model.get_parameter_limits().cpu().numpy(), neutral_camera_translation=translation)
            report["rig_sha256"] = sha256(private/"rig.npz"); digests = []; persist()
            for clip in range(CLIPS):
                for frame in range(FRAMES):
                    report.update(active_clip=clip, active_frame=frame); persist()
                    hv, ov, of, sv, sf, colors, K = scene(human[clip*FRAMES+frame], faces, regions, translation, clip, frame)
                    with torch.inference_mode(): rgb, face, depth = render.render_rgb(torch, sv, sf, colors, K)
                    torch.cuda.synchronize()
                    if (rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8 or face.shape != (HEIGHT, WIDTH)
                            or np.any(face >= len(sf)) or np.count_nonzero((face >= 0)&(face < len(faces))) < 64
                            or np.count_nonzero(face >= len(faces)) < 64):
                        raise ValueError("Each own rendered entity requires visible original-grid support")
                    depth = np.where(face >= 0, depth, np.nan).astype(np.float32)
                    if not np.isfinite(depth[face >= 0]).all() or np.any(depth[face >= 0] <= 0): raise ValueError("Invalid rendered camera-Z")
                    stem = f"clip_{clip:02d}_frame_{frame:03d}"; truth = private/(stem+".npz")
                    with truth.open("xb") as handle:
                        np.savez_compressed(handle, human_vertices_camera_m=hv, human_faces=faces, object_vertices_camera_m=ov,
                            object_faces=of, camera_K=K, scene_depth_m=depth, visible_face_indices=face,
                            clip_index=np.array(clip, np.int64), frame_index=np.array(frame, np.int64))
                    with (public/(stem+".png")).open("xb") as handle: Image.fromarray(rgb).save(handle, format="PNG")
                    digests.append(sha256(public/(stem+".png")))
                    report["cases"].append(dict(clip_index=clip, frame_index=frame, file=stem+".png", truth_sha256=sha256(truth), rgb_sha256=digests[-1])); persist()
            render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, 696110248)
            if sha256(prereq) != prerequisite_sha: raise ValueError("Semantic producer changed during rendering")
            with (public/"manifest.json").open("x") as handle: json.dump(public_manifest(digests), handle)
            report.update(status="pass", phase="complete", frames=CLIPS*FRAMES, public_manifest_sha256=sha256(public/"manifest.json"))
            report.pop("active_clip", None); report.pop("active_frame", None)
        except Exception as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist()


if __name__ == "__main__": main()
