"""H98 fresh fixed-focal scenes; manufacturing truth is evaluation-private.

Preparation only: no predicted keypoints, reconstruction, accuracy or adoption.
Recipes and framing are fixed before any new observations. The bottle receives
one fixed manufacturing scale, never fitted or changed across frames.
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
import hand_synthetic_render as render
import identity_rgb_render as primitives
from world_reward.data import sha256
import root5_rgb_protocol as protocol

COHORT=protocol.COHORT
BASE=COHORT.base
SCHEMA=COHORT.schema
STAGE="own_fresh_root5_rgb_render"
WIDTH,HEIGHT,CLIPS,FRAMES=COHORT.width,COHORT.height,COHORT.clips,COHORT.frames
BUDGET=120
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
FOCALS = (1280., 1280., 1280.)
SHAPES=((-0.19,-0.07),(0.11,0.24),(0.43,-0.09))
SCALES=(-0.035,0.020,0.055)
SIDES=("l","r","l")
YAWS=(0.23,-0.17,0.08)
BOTTLE_DEPTH_OFFSETS=(0.11,0.04,-0.04,-0.095,-0.14)
BOTTLE_SCALE=1.04
TRUTH_KEYS = primitives.TRUTH_KEYS


def sha256_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def helper_hashes():
    return {"render": sha256(Path(render.__file__)), "primitives": sha256(Path(primitives.__file__)),
            "joint": sha256(Path(__file__).with_name("joint_rgb_render.py")),
            "camera": sha256(Path(__file__).with_name("camera_render.py")),
            "protocol":sha256(Path(protocol.__file__))}


def public_manifest(digests):
    return protocol.public_manifest(digests,COHORT)


def named_controls(names, limits):
    bounds = np.asarray(limits)
    if (not isinstance(names, list) or len(names) != 249 or len(set(names)) != 249
            or any(not isinstance(n, str) or not n for n in names) or np.ma.isMaskedArray(limits)
            or bounds.shape != (249, 2) or bounds.dtype.kind != "f" or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Actual unique249 native names/bounds required")
    sb = bounds[136:204]; locked = np.all(sb == 0, axis=1)
    patterns = np.broadcast_to(np.asarray(SCALES, np.float32)[:, None], (3, 68)).copy(); patterns[:, locked] = 0
    if locked.all() or np.any(patterns < sb[:, 0]) or np.any(patterns > sb[:, 1]) or len({r.tobytes() for r in patterns}) != 3:
        raise ValueError("Three frozen legal free-scale identities required; no clipping")
    controls = np.zeros((15, 204), np.float32); identities = np.zeros((15, 45), np.float32); changes = []
    for clip, side in enumerate(SIDES):
        other = "l" if side == "r" else "r"
        for frame in range(FRAMES):
            row = clip*FRAMES+frame; controls[row, 136:] = patterns[clip]; identities[row, :2] = SHAPES[clip]
            recipe = [(f"{side}_uparm_ry", .16+.045*frame), (f"{side}_elbow_bend", .27+.055*frame),
                      (f"{side}_wrist_ry", .03-.018*frame), (f"{other}_uparm_ry", -.015*(frame-2))]
            recipe += [(f"{side}_{finger}1_rz", .11+.030*frame) for finger in ("index", "middle", "ring", "pinky")]
            for name, value in recipe:
                if name not in names[:136]: raise ValueError("Required named motion control absent: "+name)
                column = names.index(name); controls[row, column] = value
                changes.append(dict(clip_index=clip, frame_index=frame, name=name, column=column, value=value))
    full = np.c_[controls, identities]; neutral = full[::FRAMES].copy(); neutral[:, :136] = 0
    if np.any(full < bounds[:, 0]) or np.any(full > bounds[:, 1]) or np.any(neutral < bounds[:, 0]) or np.any(neutral > bounds[:, 1]):
        raise ValueError("Every animated/neutral249 value must obey actual bounds")
    return controls, identities, changes


def fixed_framing(neutral):
    values = np.asarray(neutral)
    if np.ma.isMaskedArray(neutral) or values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError("Three finite native neutral identities required")
    points = values.reshape(-1, 3); center = (points.min(0)+points.max(0))/2; relative = points-center
    distance = max(np.max(1280*np.abs(relative[:, 0])/(WIDTH*.29)-relative[:, 2]),
                   np.max(1280*np.abs(relative[:, 1])/(HEIGHT*.29)-relative[:, 2]))+.4
    if not np.isfinite(distance) or distance <= 0: raise ValueError("Positive fixed manufacturing camera distance required")
    return center, float(distance), float(relative[:, 1].max()+.08)


def scene(human, human_faces, regions, center, distance, floor, clip, frame):
    from scipy.spatial.transform import Rotation
    if type(clip) is not int or type(frame) is not int or not 0 <= clip < 3 or not 0 <= frame < 5:
        raise ValueError("One prescribed clip/frame required")
    actor = (human-center) @ Rotation.from_rotvec([0., YAWS[clip]+.028*(frame-2), 0.]).as_matrix().T
    actor += [.016*(frame-2), .006*np.sin(.9*frame), -.024*(frame-2)]
    hand = actor[regions[SIDES[clip]]["vertex_mask"]]
    if len(hand) < 50: raise ValueError("Actual active hand LBS region absent")
    bottle, of = primitives.bottle_mesh(); bottle *= BOTTLE_SCALE
    bottle = bottle @ Rotation.from_rotvec([-.035*(frame-2), .055*frame, .045*(frame-2)]).as_matrix().T
    bottle += (hand.min(0)+hand.max(0))/2+[.048*(-1 if SIDES[clip] == "l" else 1), -.014, BOTTLE_DEPTH_OFFSETS[frame]]
    hc = np.tile([.70,.67,.64], (len(actor), 1)); low, high = human[:, 1].min(), human[:, 1].max()
    if high <= low: raise ValueError("Native human physical extent absent")
    band = (human[:, 1]-low)/(high-low); cloth = (band > .23) & (band < .81)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    hc[cloth] = np.c_[.32+.025*np.sin(17*human[cloth,0]), .29+.03*np.cos(14*human[cloth,1]), .36+.02*np.sin(10*human[cloth,2])]
    oc = np.tile([.37,.30,.20], (len(bottle), 1)); oc[160:] = [.30,.28,.25]
    bv, bf, bc = primitives.background(distance, floor)
    vertices = np.r_[actor, bottle, bv]+[0., 0., distance]
    faces = np.r_[human_faces, of+len(actor), bf+len(actor)+len(bottle)]
    if not np.isfinite(vertices).all() or vertices[:, 2].min() <= .01: raise ValueError("Entire scene crosses camera near plane")
    K=np.array(COHORT.fixed_K,np.float64)
    return vertices[:len(actor)], vertices[len(actor):len(actor)+len(bottle)], of, vertices, faces, np.r_[hc, oc, bc], K


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote offline CUDA preparation required")
    root = Path(os.environ["WR_ROOT"]); dest = root/BASE; revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or dest.resolve() != dest.absolute() or not dest.is_dir() or any(dest.iterdir())
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or image != IMAGE or os.geteuid() != 1000):
        raise ValueError("Empty canonical reserved source/image-bound destination required")
    public, private = dest/"inputs", dest/"eval_private"; public.mkdir(); private.mkdir(mode=0o700)
    helpers = helper_hashes(); source_sha = sha256(Path(__file__))
    report = dict(stage=STAGE, status="fail", phase="prerequisites", frames=15, cases=[], budget_seconds=BUDGET,
        code_revision=revision, image_id=image, script_sha256=source_sha, helper_source_sha256=helpers,
        challenge_inputs_used=False, synthetic_truth_used_for_rendering_only=True, inference_performed=False,
        predictions_performed=False, quality_verified=False, accuracy_verified=False, photorealism_verified=False,
        all_truth_private=True, identity_clip_constant=True, true_camera_private=True, true_focals=FOCALS,
        inference_prior_focal=1280., shape_first_two_coefficients=SHAPES, scale68_scalar_candidates=SCALES,
        interaction_sides=SIDES,yaw_base=YAWS,bottle_depth_offsets=BOTTLE_DEPTH_OFFSETS,bottle_manufacturing_scale=BOTTLE_SCALE,
        actual_reference_forward_calls=0,reference_forward_attempts=0,reference_forward_returns=0,
        scale_pattern_rule="fixed scalar on nonlocked controls; actual [0,0] scales exactly zero; no clipping",
        scene_depth_scope="human/object foreground only; background IDs=-1/depth=NaN",truth_keys=sorted(TRUTH_KEYS),
        truth_human_faces_dtype="int32",truth_object_faces_dtype="int64",independent_fresh_cohort=True,
        license_clearance_verified=False,training_overlap_excluded=False)
    started = time.perf_counter(); receipt = private/"render-report.json"
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole fresh keypoint render exceeded120s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); prereq = root/"results/mhr-finger-semantics-v4.json"; prereq_sha = sha256(prereq)
            render.semantics.regular_hash(prereq, root, prereq_sha); semantic = json.loads(prereq.read_text()); render.require_semantic_report(semantic)
            if semantic.get("source_image_id") != image: raise ValueError("Reference image differs from semantic producer")
            with (private/"semantic-report.json").open("xb") as handle: handle.write(prereq.read_bytes())
            (private/"semantic-report.json").chmod(0o400)
            model_path = root/"weights/mhr/mhr_model.pt"; render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, 696110248)
            if "torch" in sys.modules: raise RuntimeError("CUBLAS setup must precede Torch")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Actual pinned CUDA renderer required")
            with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            names, joints = model.get_parameter_names(), model.get_joint_names(); bounds = model.get_parameter_limits().cpu().numpy()
            controls, identities, changes = named_controls(names, bounds); faces = model.character_torch.mesh.faces.cpu().numpy()
            if (faces.shape != (36874, 3) or faces.dtype!=np.int32 or np.any(faces < 0) or np.any(faces >= 18439)
                    or joints != semantic["joint_names"] or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72)):
                raise ValueError("Actual reference topology/names/coefficient ABI differs")
            report.update(human_faces_sha256=sha256_array(faces),human_faces_shape=list(faces.shape))
            regions = render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()), joints)
            neutral = controls[::FRAMES].copy(); neutral[:, :136] = 0
            p = torch.tensor(controls, device="cuda"); ids = torch.tensor(identities, device="cuda"); before = (p.clone(), ids.clone())
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                report["reference_forward_attempts"]+=1;persist()
                raw,sk=model(ids,p,torch.zeros(15,72,device="cuda"),True)
                report["reference_forward_returns"]+=1;persist()
                report["reference_forward_attempts"]+=1;persist()
                base,base_sk=model(ids[::FRAMES],torch.tensor(neutral,device="cuda"),torch.zeros(3,72,device="cuda"),True)
                report["reference_forward_returns"]+=1;persist()
            torch.cuda.synchronize()
            if not torch.equal(p, before[0]) or not torch.equal(ids, before[1]): raise ValueError("Native model mutated manufacturing inputs")
            render.semantics.check_geometry(raw.cpu().numpy(), sk.cpu().numpy(), 127); render.semantics.check_geometry(base.cpu().numpy(),base_sk.cpu().numpy(),127)
            report["actual_reference_forward_calls"]=2;persist()
            flip = np.diag([1., -1., -1.]); humans = raw.cpu().numpy() @ flip/100; center, distance, floor = fixed_framing(base.cpu().numpy() @ flip/100)
            with (private/"rig.npz").open("xb") as handle: np.savez_compressed(handle, controls=controls, shape45=identities, parameter_limits=bounds)
            (private/"rig.npz").chmod(0o400)
            report.update(phase="rendering", parameter_names=names, named_controls=changes, scale68_patterns=controls[::5, 136:].tolist(),
                zero_locked_scale_indices=np.flatnonzero(np.all(bounds[136:204] == 0, axis=1)).tolist(), rig_sha256=sha256(private/"rig.npz"),
                model_sha256=render.semantics.MODEL_SHA, semantic_report_sha256=prereq_sha, torch=str(torch.__version__),
                pytorch3d_revision=render.PYTORCH3D_REVISION, coordinate_rule="rawcm*diag(1,-1,-1)/100 then own yaw/translation exactly once")
            digests = []; persist()
            for index in range(15):
                clip, frame = divmod(index, FRAMES); report.update(active_clip=clip, active_frame=frame); persist()
                hv, ov, of, sv, sf, colors, K = scene(humans[index], faces, regions, center, distance, floor, clip, frame)
                pixels = render.project_camera_points(hv, K)
                if np.any(pixels < 8) or np.any(pixels >= [WIDTH-8, HEIGHT-8]): raise ValueError("Frozen actor framing failed; no camera repair")
                with torch.inference_mode(): rgb, face, depth = render.render_rgb(torch, sv, sf, colors, K)
                torch.cuda.synchronize(); face, depth = primitives.foreground_truth(face, depth, faces, of)
                human_pixels = int(np.count_nonzero((face >= 0) & (face < len(faces))))
                object_pixels = int(np.count_nonzero(face >= len(faces)))
                if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8: raise ValueError("Original RGB grid differs")
                stem = f"clip_{clip:02d}_frame_{frame:03d}"; truth, png = private/(stem+".npz"), public/(stem+".png")
                with truth.open("xb") as handle: np.savez_compressed(handle, human_vertices_camera_m=hv, human_faces=faces,
                    object_vertices_camera_m=ov, object_faces=of, camera_K=K, scene_depth_m=depth, visible_face_indices=face,
                    clip_index=np.array(clip, np.int64), frame_index=np.array(frame, np.int64))
                truth.chmod(0o400)
                with png.open("xb") as handle: Image.fromarray(rgb).save(handle, format="PNG")
                png.chmod(0o444); digests.append(sha256(png)); report["cases"].append(dict(file=png.name,human_faces_sha256=sha256_array(faces),object_faces_sha256=sha256_array(of),clip_index=clip,
                    frame_index=frame, rgb_sha256=digests[-1], truth_sha256=sha256(truth),
                    visible_human_pixels=human_pixels, visible_object_pixels=object_pixels)); persist()
            if len(set(digests)) != 15: raise ValueError("All15 new moving RGBs must differ; no cloning")
            render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, 696110248)
            if sha256(prereq) != prereq_sha or helper_hashes() != helpers or sha256(Path(__file__)) != source_sha:
                raise ValueError("Frozen source/model/semantic evidence changed")
            with (public/"manifest.json").open("x")as handle:json.dump(public_manifest(digests),handle)
            (public/"manifest.json").chmod(0o444)
            expected = {"rig.npz", "semantic-report.json", "render-report.json", *[f"clip_{c:02d}_frame_{f:03d}.npz" for c in range(3) for f in range(5)]}
            if {p.name for p in private.iterdir()} != expected or {p.name for p in public.iterdir()} != {"manifest.json", *[r["file"] for r in report["cases"]]}:
                raise ValueError("Exact15 private/public immutable inventory required")
            protocol.public_inputs(public,COHORT)
            report.update(status="pass", phase="complete", actual_MHR_reference_used=True, public_manifest_sha256=sha256(public/"manifest.json"),
                          public_manifest_bytes=(public/"manifest.json").stat().st_size)
            report.pop("active_clip", None); report.pop("active_frame", None)
        except BaseException as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o400)


if __name__ == "__main__": main()
