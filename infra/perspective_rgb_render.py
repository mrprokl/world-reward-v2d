"""Fresh procedural perspective RGB: manufacturing calibration stays private.

Six open-front rooms supply parallel-line/vertical cues; three weak controls
are uniform-background human-only or empty. This is not photorealistic data or
evidence that a learned calibration model succeeds. No cohort filtering occurs.
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
from joint_rgb_render import bottle_mesh
from world_reward.data import sha256

WIDTH, HEIGHT, BUDGET = 1024, 768, 120
SCHEMA = "world-reward-perspective-rgb-v1"
FOCALS = (1024., 1408., 1792.)
PITCH_ROLL = ((-.12, .08), (.16, -.09))
SHAPES = (.21, -.18, .36)
ACTOR_SIZES = (.94, 1.03, 1.08)
TRUTH_KEYS = {"camera_K", "gravity", "strong"}


def public_manifest(digests):
    if len(digests) != 9 or any(not isinstance(x, str) or not re.fullmatch("[0-9a-f]{64}", x) for x in digests):
        raise ValueError("Exactly nine ordered RGB identities required")
    return {"schema": SCHEMA, "images": [{"file": f"case_{i:02d}.png", "sha256": d,
        "width": WIDTH, "height": HEIGHT} for i, d in enumerate(digests)]}


def camera_rotation(pitch, roll):
    """Proper world-to-camera Rz(roll) Rx(pitch), radians; world Y points down."""
    cx, sx, cz, sz = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]]) @ np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]])


def camera_truth():
    rotations = [camera_rotation(pitch, roll) for _ in FOCALS for pitch, roll in PITCH_ROLL]
    rotations += [camera_rotation(.04, -.05), camera_rotation(-.03, .02), camera_rotation(0., 0.)]
    cameras = np.array([[[f, 0., WIDTH/2], [0., f, HEIGHT/2], [0., 0., 1.]]
                       for f in (*[f for f in FOCALS for _ in PITCH_ROLL], *FOCALS)], np.float64)
    gravity = np.array([R @ [0., -1., 0.] for R in rotations], np.float64)
    return cameras, np.array(rotations), gravity, np.array([True]*6+[False]*3, bool)


def named_controls(names, limits):
    bounds = np.asarray(limits)
    if (not isinstance(names, list) or len(names) != 249 or len(set(names)) != 249
            or np.ma.isMaskedArray(limits) or bounds.shape != (249, 2) or bounds.dtype.kind != "f"
            or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1])
            or np.any(bounds[:204, 0] > 0) or np.any(bounds[:204, 1] < 0)):
        raise ValueError("Actual unique parameter names/bounds with legal neutral required")
    controls = np.zeros((9, 204), np.float32); identity = np.zeros((9, 45), np.float32); changes = []
    for i in range(9):
        group = i//2 if i < 6 else i-6
        if not bounds[204, 0] <= SHAPES[group] <= bounds[204, 1]:
            raise ValueError("Fresh identity coefficient exceeds actual bound")
        identity[i, 0] = SHAPES[group]
        side = "r" if i % 2 else "l"
        recipe = [(f"{side}_uparm_ry", .135+.017*(i%3)), (f"{side}_elbow_bend", .335+.019*(i%3))]
        recipe += [(f"{side}_{finger}1_rz", .155+.013*(i%3)) for finger in ("index", "middle", "ring", "pinky")]
        for name, value in recipe:
            if name not in names[:136]: raise ValueError("Required actual named pose control absent: "+name)
            column = names.index(name)
            if not bounds[column, 0] <= value <= bounds[column, 1]: raise ValueError("Fresh pose exceeds actual bound")
            controls[i, column] = value; changes.append({"case_index": i, "name": name, "column": column, "value": value})
    return controls, identity, changes


def fixed_framing(neutral, rotations):
    points = np.asarray(neutral, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3 or not np.isfinite(points).all(): raise ValueError("Finite own neutral identity meshes required")
    center = (points.min(0)+points.max(0))/2; relative = points-center
    rotated = np.concatenate([relative @ R.T for R in rotations])
    distance = max(np.max(max(FOCALS)*np.abs(rotated[:, 0])/(WIDTH*.28)-rotated[:, 2]),
                   np.max(max(FOCALS)*np.abs(rotated[:, 1])/(HEIGHT*.28)-rotated[:, 2]))+.45
    if not np.isfinite(distance) or distance <= 0: raise ValueError("Invalid fixed neutral framing")
    return center, float(distance), float(relative[:, 1].max()+.06)


def tiled_quad(origin, u, v, color, nu=12, nv=12):
    """Own tessellated planar tiles; no scene/focal-dependent color labels."""
    origin, u, v, color = (np.asarray(x, np.float64) for x in (origin, u, v, color))
    vertices, faces, colors = [], [], []
    for i in range(nu):
        for j in range(nv):
            n = len(vertices)
            vertices.extend([origin+u*a/nu+v*b/nv for a, b in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))])
            faces.extend([[n,n+1,n+2],[n,n+2,n+3]])
            tint = .045*((i+j)%2-.5)+.012*np.sin(.8*i+1.1*j)
            colors.extend([np.clip(color+tint, .08, .9)]*4)
    return np.array(vertices), np.array(faces, np.int64), np.array(colors)


def join_meshes(meshes):
    vertices, faces, colors, offset = [], [], [], 0
    for v, f, c in meshes:
        vertices.append(v); faces.append(f+offset); colors.append(c); offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(colors)


def room_mesh(distance, floor):
    front, back, halfwidth, ceiling = 2.5-distance, 4.8, 3.2, -2.2
    specs = [([-halfwidth,floor,front], [2*halfwidth,0,0], [0,0,back-front], [.59,.55,.48]),
        ([-halfwidth,ceiling,front], [0,floor-ceiling,0], [0,0,back-front], [.67,.70,.73]),
        ([halfwidth,ceiling,front], [0,0,back-front], [0,floor-ceiling,0], [.70,.68,.64]),
        ([-halfwidth,ceiling,back], [2*halfwidth,0,0], [0,floor-ceiling,0], [.65,.70,.66]),
        ([-halfwidth,ceiling,front], [0,0,back-front], [2*halfwidth,0,0], [.75,.74,.70])]
    meshes = [tiled_quad(*spec) for spec in specs]
    # Two upright, closed own box furnishings retain independent vertical/depth cues.
    for x, z, w, h, d in ((-1.55,.45,.8,.74,.66),(1.45,1.6,.68,1.08,.62)):
        o = np.array([x-w/2,floor-h,z-d/2]); a,b,c=np.array([w,0,0]),np.array([0,h,0]),np.array([0,0,d])
        for origin,u,v in ((o,a,b),(o+c,b,a),(o,c,a),(o+b,a,c),(o,b,c),(o+a,c,b)):
            meshes.append(tiled_quad(origin,u,v,[.35,.40,.44],2,2))
    return join_meshes(meshes)


def scene(human, human_faces, regions, center, distance, floor, index, R):
    from scipy.spatial.transform import Rotation
    actor = human-center
    actor = actor @ Rotation.from_rotvec([0., -.11+.031*index, 0.]).as_matrix().T
    hand = actor[regions["r" if index%2 else "l"]["vertex_mask"]]
    bottle, bottle_faces = bottle_mesh(); bottle = bottle*1.075
    bottle += (hand.min(0)+hand.max(0))/2+[.065*(-1 if index%2 == 0 else 1),.014,-.09]
    hc = np.tile([.74,.73,.71],(len(actor),1)); low,high=human[:,1].min(),human[:,1].max()
    cloth = ((human[:,1]-low)/(high-low)>.24)&((human[:,1]-low)/(high-low)<.80)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    hc[cloth]=[.34,.35,.36]
    meshes = [(actor,human_faces,hc)]
    if index < 6:
        meshes += [(bottle,bottle_faces,np.tile([.26,.41,.33],(len(bottle),1))),room_mesh(distance,floor)]
    vertices, faces, colors = join_meshes(meshes)
    camera = vertices @ R.T + [0.,0.,distance]
    if not np.isfinite(camera).all() or camera[:,2].min()<=.01: raise ValueError("Entire own scene must have positive camera Z")
    return camera,faces,colors,len(human_faces),len(bottle_faces) if index<6 else 0


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Remote offline CUDA-only synthesis required")
    root=Path(os.environ["WR_ROOT"]); revision,image=os.environ["WR_CODE_REVISION"],os.environ["WR_IMAGE_ID"]
    dest=root/"validation/perspective_rgb_v1"
    if (root!=Path("/srv/scenesmith/world-reward") or dest.resolve()!=dest.absolute() or not dest.is_dir() or any(dest.iterdir())
            or not re.fullmatch("[0-9a-f]{40}",revision) or not re.fullmatch("sha256:[0-9a-f]{64}",image)):
        raise ValueError("Canonical reserved root and immutable source/image required")
    public,private=dest/"inputs",dest/"eval_private";public.mkdir();private.mkdir(mode=0o700)
    report={"stage":"own_procedural_perspective_rgb_render","status":"fail","cases":[],"budget_seconds":BUDGET,
        "script_sha256":sha256(Path(__file__)),"render_helper_sha256":sha256(Path(render.__file__)),
        "joint_helper_sha256":sha256(Path(__file__).with_name("joint_rgb_render.py")),
        "camera_helper_sha256":sha256(Path(__file__).with_name("camera_render.py")),"code_revision":revision,"image_id":image,
        "challenge_inputs_used":False,"synthetic_truth_used_for_rendering_only":True,"inference_performed":False,
        "accuracy_verified":False,"photorealism_verified":False,"camera_framing":"one neutral union translation for all9",
        "strong_cases":6,"weak_cases":3,"weak_design":["empty_uniform","human_uniform_background","human_uniform_background"],
        "actor_sizes_baked_once":ACTOR_SIZES,"raw_model_scale_controls":"zero68","shape_first_coefficient":SHAPES,
        "licenses":{"reference_mhr":"Apache-2.0","own_geometry_materials":"procedural_owned_not_external_dataset"}}
    start=time.perf_counter()
    with (private/"render-report.json").open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-start;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole frozen perspective render exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist();prereq=root/"results/mhr-finger-semantics-v4.json";prereq_sha=sha256(prereq)
            render.semantics.regular_hash(prereq,root,prereq_sha);semantic=json.loads(prereq.read_text());render.require_semantic_report(semantic)
            if semantic.get("source_image_id")!=image:raise ValueError("Render must use the semantic-provenance image")
            model_path=root/"weights/mhr/mhr_model.pt"
            render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            if "torch" in sys.modules:raise RuntimeError("CUBLAS configuration must precede Torch import")
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__!="0.7.9":raise RuntimeError("Actual pinned CUDA raster required")
            K,rotations,gravity,strong=camera_truth()
            with torch.jit.optimized_execution(False):model=torch.jit.load(str(model_path),map_location="cuda").float().eval()
            names,joints=model.get_parameter_names(),model.get_joint_names();controls,identity,changes=named_controls(names,model.get_parameter_limits().cpu().numpy())
            faces=model.character_torch.mesh.faces.cpu().numpy()
            if (faces.shape!=(36874,3) or faces.dtype.kind not in "iu" or np.any(faces<0) or np.any(faces>=18439)
                    or joints!=semantic["joint_names"] or (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())!=(45,72)):
                raise ValueError("Actual MHR names/topology/coefficient ABI changed")
            regions=render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()),joints)
            with torch.inference_mode(),torch.jit.optimized_execution(False):
                raw,state=model(torch.as_tensor(identity,device="cuda"),torch.as_tensor(controls,device="cuda"),torch.zeros(9,72,device="cuda"),True)
                neutral,_=model(torch.as_tensor(identity,device="cuda"),torch.zeros(9,204,device="cuda"),torch.zeros(9,72,device="cuda"),True)
            torch.cuda.synchronize();render.semantics.check_geometry(raw.cpu().numpy(),state.cpu().numpy(),127)
            sizes=np.array([ACTOR_SIZES[i//2 if i<6 else i-6] for i in range(9)])
            humans=raw.cpu().numpy()@np.diag([1.,-1.,-1.])/100*sizes[:,None,None]
            neutral=neutral.cpu().numpy()@np.diag([1.,-1.,-1.])/100*sizes[:,None,None]
            center,distance,floor=fixed_framing(neutral,rotations)
            report.update(model_sha256=render.semantics.MODEL_SHA,semantic_report_sha256=prereq_sha,
                pytorch3d_revision=render.PYTORCH3D_REVISION,torch=torch.__version__,parameter_names=names,named_controls=changes,
                camera_rotations=rotations.tolist(),world_up_vector=[0.,-1.,0.],fixed_distance_m=distance);persist()
            digests=[]
            for index in range(9):
                report["active_case"]=index;persist()
                if index==6:rgb=np.full((HEIGHT,WIDTH,3),184,np.uint8)
                else:
                    vertices,topology,colors,hf,bf=scene(humans[index],faces,regions,center,distance,floor,index,rotations[index])
                    bodypixels=render.project_camera_points(vertices[:18439],K[index])
                    if np.any(bodypixels<16) or np.any(bodypixels>=[WIDTH-16,HEIGHT-16]):raise ValueError("Fixed central actor framing failed")
                    with torch.inference_mode():rgb,face,_=render.render_rgb(torch,vertices,topology,colors,K[index])
                    torch.cuda.synchronize()
                    if np.count_nonzero((face>=0)&(face<hf))<64:raise ValueError("Own human RGB support absent")
                    if index<6 and (np.count_nonzero((face>=hf)&(face<hf+bf))<16 or np.count_nonzero(face>=hf+bf)<1024):
                        raise ValueError("Strong own room/bottle RGB support absent")
                if rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8:raise ValueError("Original RGB grid contract failed")
                filename=f"case_{index:02d}.png";path=public/filename
                with path.open("xb") as handle:Image.fromarray(rgb).save(handle,format="PNG")
                path.chmod(0o444);digests.append(sha256(path));report["cases"].append({"file":filename,"rgb_sha256":digests[-1]});persist()
            truthpath=private/"calibration_truth.npz"
            with truthpath.open("xb") as handle:np.savez_compressed(handle,camera_K=K,gravity=gravity,strong=strong)
            truthpath.chmod(0o400)
            render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            if sha256(prereq)!=prereq_sha:raise ValueError("Semantic provenance changed during render")
            with (public/"manifest.json").open("x") as handle:json.dump(public_manifest(digests),handle)
            (public/"manifest.json").chmod(0o444)
            report.update(status="pass",actual_MHR_reference_used=True,actual_reference_forward_calls=2,
                truth_sha256=sha256(truthpath),public_manifest_sha256=sha256(public/"manifest.json"));persist()
        except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));persist();raise
        finally:signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist()


if __name__=="__main__":main()
