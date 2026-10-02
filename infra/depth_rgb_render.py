"""Render fresh D88 RGB/depth truth; no inference, human or HOI quality claim.

Only fifteen original RGBs and their byte manifest are public. Native MHR,
object geometry, camera, scene-Z and visible IDs remain evaluation-private.
"""
import argparse
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
import rgb_cohort_protocol as protocol
from world_reward.data import sha256

BASE, WIDTH, HEIGHT = protocol.BASE, protocol.WIDTH, protocol.HEIGHT
CLIPS, FRAMES, BUDGET = protocol.CLIPS, protocol.FRAMES, 120
STAGE = "own_fresh_depth_rgb_render"
IMAGE_ID = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
TRUTH_KEYS = primitives.TRUTH_KEYS


def helper_hashes():
    return {"protocol": sha256(Path(protocol.__file__)), "primitives": sha256(Path(primitives.__file__)),
            "render": sha256(Path(render.__file__)), "joint": sha256(Path(__file__).with_name("joint_rgb_render.py")),
            "camera": sha256(Path(__file__).with_name("camera_render.py"))}


def fixed_framing(neutral):
    values = np.asarray(neutral)
    if np.ma.isMaskedArray(neutral) or values.ndim != 3 or values.shape[0] != CLIPS or values.shape[2] != 3 or not np.isfinite(values).all():
        raise ValueError("Three finite neutral native identities required")
    points = values.reshape(-1, 3); center = (points.min(0)+points.max(0))/2; relative = points-center
    distance = max(np.max(max(protocol.D88.focals)*np.abs(relative[:, 0])/(WIDTH*.29)-relative[:, 2]),
                   np.max(max(protocol.D88.focals)*np.abs(relative[:, 1])/(HEIGHT*.29)-relative[:, 2]))+.4
    return center, float(distance), float(relative[:, 1].max()+.08)


def scene(human, human_faces, regions, center, distance, floor, clip, frame):
    from scipy.spatial.transform import Rotation
    if type(clip) is not int or type(frame) is not int or not 0 <= clip < CLIPS or not 0 <= frame < FRAMES:
        raise ValueError("One prescribed D88 clip/frame required")
    actor = (human-center) @ Rotation.from_rotvec([0., (.12, -.19, .09)[clip]+.04*(frame-2), 0.]).as_matrix().T
    actor += [.012*(frame-2), .006*np.sin(.8*frame), .021*(frame-2)]
    hand = actor[regions["l" if clip == 1 else "r"]["vertex_mask"]]
    if len(hand) < 50: raise ValueError("Native moving hand region absent")
    bottle, of = primitives.bottle_mesh(); bottle *= 1.12
    bottle = bottle @ Rotation.from_rotvec([.035*(frame-2), .075*frame, -.045*(frame-2)]).as_matrix().T
    bottle += (hand.min(0)+hand.max(0))/2+[.032*(-1 if clip == 1 else 1), .018, protocol.D88.bottle_depth_offsets[frame]]
    hc = np.tile([.75, .73, .70], (len(actor), 1)); low, high = human[:, 1].min(), human[:, 1].max()
    if high <= low: raise ValueError("Native human extent absent")
    band = (human[:, 1]-low)/(high-low); cloth = (band > .23) & (band < .81)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    hc[cloth] = np.c_[.31+.026*np.sin(15*human[cloth, 0]), .35+.022*np.cos(11*human[cloth, 1]), .38+.019*np.sin(8*human[cloth, 2])]
    oc = np.tile([.23, .41, .30], (len(bottle), 1)); oc[160:] = [.28, .29, .27]
    bv, bf, bc = primitives.background(distance, floor)
    vertices = np.r_[actor, bottle, bv]+[0., 0., distance]
    faces = np.r_[human_faces, of+len(actor), bf+len(actor)+len(bottle)]
    if not np.isfinite(vertices).all() or vertices[:, 2].min() <= .01: raise ValueError("Entire D88 scene crosses near plane")
    focal = protocol.D88.focals[clip]; K = np.array([[focal, 0., WIDTH/2], [0., focal, HEIGHT/2], [0., 0., 1.]])
    return vertices[:len(actor)], vertices[len(actor):len(actor)+len(bottle)], of, vertices, faces, np.r_[hc, oc, bc], K


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote offline CUDA renderer required")
    root = Path(os.environ["WR_ROOT"]); dest = root/BASE; revision, image = os.environ["WR_CODE_REVISION"], os.environ["WR_IMAGE_ID"]
    if (root != Path("/srv/scenesmith/world-reward") or dest.resolve() != dest.absolute() or not dest.is_dir() or any(dest.iterdir())
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != IMAGE_ID):
        raise ValueError("Empty canonical source/image-bound D88 destination required")
    public, private = dest/"inputs", dest/"eval_private"; public.mkdir(); private.mkdir(mode=0o700)
    helpers = helper_hashes(); report = dict(stage=STAGE, status="fail", phase="prerequisites", frames=CLIPS*FRAMES, cases=[],
        budget_seconds=BUDGET, code_revision=revision, image_id=image, script_sha256=sha256(Path(__file__)), helper_source_sha256=helpers,
        protocol=protocol.D88.to_dict(), protocol_sha256=protocol.D88.digest(), challenge_inputs_used=False,
        synthetic_truth_used_for_rendering_only=True, inference_performed=False, accuracy_verified=False, full_HOI_verified=False,
        photorealism_verified=False, all_truth_private=True, identity_clip_constant=True, actual_reference_forward_calls=0,
        scene_depth_scope="human/object foreground only; room IDs=-1/depth=NaN", background_face_ids_removed_from_truth=True,
        scale_pattern_rule="fixed D88 scalar on all nonlocked scales; actual [0,0] controls exactly zero", truth_keys=sorted(TRUTH_KEYS))
    started = time.perf_counter(); receipt = private/"render-report.json"
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole D88 render exceeds120s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); prereq = root/"results/mhr-finger-semantics-v4.json"; prereq_sha = sha256(prereq)
            render.semantics.regular_hash(prereq, root, prereq_sha); semantic = json.loads(prereq.read_text()); render.require_semantic_report(semantic)
            if semantic.get("source_image_id") != image: raise ValueError("D88 image differs from native semantic source")
            with (private/"semantic-report.json").open("xb") as handle: handle.write(prereq.read_bytes())
            (private/"semantic-report.json").chmod(0o400)
            model_path = root/"weights/mhr/mhr_model.pt"; render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, 696110248)
            if "torch" in sys.modules: raise RuntimeError("CUBLAS must be configured before Torch")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__ != "0.7.9": raise RuntimeError("Actual pinned CUDA renderer required")
            with torch.jit.optimized_execution(False): model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            names, joints = model.get_parameter_names(), model.get_joint_names(); bounds = model.get_parameter_limits().cpu().numpy()
            controls, identity, changes = protocol.named_controls(names, bounds); faces = model.character_torch.mesh.faces.cpu().numpy()
            if (faces.shape != (36874, 3) or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= 18439)
                    or joints != semantic["joint_names"] or (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72)):
                raise ValueError("Actual native topology/joint/shape ABI differs")
            report.update(parameter_names=names, named_controls=changes, scale68_patterns=controls[::FRAMES, 136:].tolist(),
                zero_locked_scale_indices=np.flatnonzero(np.all(bounds[136:204] == 0, axis=1)).tolist(),
                model_sha256=render.semantics.MODEL_SHA, semantic_report_sha256=prereq_sha); persist()
            regions = render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()), joints)
            neutral = controls[::FRAMES].copy(); neutral[:, :136] = 0
            p = torch.tensor(controls, device="cuda"); ids = torch.tensor(identity, device="cuda"); before = (p.clone(), ids.clone())
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                raw, skeleton = model(ids, p, torch.zeros(len(controls), 72, device="cuda"), True); report["actual_reference_forward_calls"] += 1
                base, sk = model(ids[::FRAMES], torch.tensor(neutral, device="cuda"), torch.zeros(CLIPS, 72, device="cuda"), True)
                report["actual_reference_forward_calls"] += 1
            torch.cuda.synchronize()
            if not torch.equal(p, before[0]) or not torch.equal(ids, before[1]): raise ValueError("Native reference mutated manufacturing inputs")
            render.semantics.check_geometry(raw.cpu().numpy(), skeleton.cpu().numpy(), 127); render.semantics.check_geometry(base.cpu().numpy(), sk.cpu().numpy(), 127)
            flip = np.diag([1., -1., -1.]); humans = raw.cpu().numpy() @ flip/100; center, distance, floor = fixed_framing(base.cpu().numpy() @ flip/100)
            with (private/"rig.npz").open("xb") as handle: np.savez_compressed(handle, controls=controls, shape45=identity, parameter_limits=bounds)
            (private/"rig.npz").chmod(0o400); report.update(phase="rendering", rig_sha256=sha256(private/"rig.npz"), torch=str(torch.__version__),
                pytorch3d_revision=render.PYTORCH3D_REVISION); digests = []; persist()
            for index in range(CLIPS*FRAMES):
                clip, frame = divmod(index, FRAMES); report.update(active_clip=clip, active_frame=frame); persist()
                hv, ov, of, sv, sf, colors, K = scene(humans[index], faces, regions, center, distance, floor, clip, frame)
                pixels = render.project_camera_points(hv, K)
                if np.any(pixels < 8) or np.any(pixels >= [WIDTH-8, HEIGHT-8]): raise ValueError("Fixed D88 actor framing fails")
                with torch.inference_mode(): rgb, face, depth = render.render_rgb(torch, sv, sf, colors, K)
                torch.cuda.synchronize(); face, depth = primitives.foreground_truth(face, depth, faces, of)
                if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8: raise ValueError("Original RGB grid differs")
                stem = f"clip_{clip:02d}_frame_{frame:03d}"; truth, png = private/(stem+".npz"), public/(stem+".png")
                with truth.open("xb") as handle: np.savez_compressed(handle, human_vertices_camera_m=hv, human_faces=faces,
                    object_vertices_camera_m=ov, object_faces=of, camera_K=K, scene_depth_m=depth, visible_face_indices=face,
                    clip_index=np.array(clip, np.int64), frame_index=np.array(frame, np.int64))
                truth.chmod(0o400)
                with png.open("xb") as handle: Image.fromarray(rgb).save(handle, format="PNG")
                png.chmod(0o444); digests.append(sha256(png)); report["cases"].append(dict(clip_index=clip, frame_index=frame,
                    file=png.name, rgb_sha256=digests[-1], truth_sha256=sha256(truth))); persist()
            if len(set(digests)) != CLIPS*FRAMES: raise ValueError("Every new D88 RGB must differ; no cloning")
            render.semantics.regular_hash(model_path, root, render.semantics.MODEL_SHA, 696110248)
            if sha256(prereq) != prereq_sha or helper_hashes() != helpers: raise ValueError("Frozen source/model/semantic inputs changed")
            with (public/"manifest.json").open("x") as handle: json.dump(protocol.public_manifest(digests), handle)
            (public/"manifest.json").chmod(0o444)
            expected = {"rig.npz", "semantic-report.json", "render-report.json", *[f"clip_{c:02d}_frame_{f:03d}.npz" for c in range(CLIPS) for f in range(FRAMES)]}
            if {p.name for p in private.iterdir()} != expected: raise ValueError("Exact fifteen truths/rig/semantic/report inventory required")
            records, public_receipt = protocol.public_inputs(public)
            if [r["sha256"] for r in records] != digests: raise ValueError("Published original RGB identity differs")
            report.update(status="pass", phase="complete", actual_MHR_reference_used=True,
                public_manifest_sha256=public_receipt["sha256"], public_manifest_bytes=public_receipt["bytes"])
            report.pop("active_clip", None); report.pop("active_frame", None)
        except BaseException as error: report.update(error_type=type(error).__name__, error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); receipt.chmod(0o400)


if __name__ == "__main__": main()
