"""Fresh moving MHR identities/bottle RGB; all manufacturing truth is private.

Three shape45/scale68 identities are genuinely clip-constant native controls,
not resized meshes or cloned RGB. Diffuse procedural scenes are not a claim of
photorealism, accuracy, full temporal HOI validation or submission improvement.
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
from joint_rgb_render import bottle_mesh, TRUTH_KEYS
from world_reward.data import sha256

WIDTH,HEIGHT,CLIPS,FRAMES,BUDGET=1024,768,3,5,120
SCHEMA="world-reward-identity-rgb-v1"
BASE="validation/identity_rgb_v2"
FOCALS=(1160.,1480.,1720.)
SHAPES=((-0.27,.12),(.32,-.15),(.49,.08))
SCALES=(-.04,.03,.07)
SCALE_PATTERN_RULE="fixed_scalar_on_all_nonlocked_controls_exact_zero_for_native_zero_locked_controls"


def public_manifest(digests):
    if len(digests)!=CLIPS*FRAMES or any(not isinstance(d,str) or not re.fullmatch(r"[0-9a-f]{64}",d) for d in digests):
        raise ValueError("All fifteen ordered RGB hashes required")
    return {"schema":SCHEMA,"images":[dict(file=f"clip_{c:02d}_frame_{f:03d}.png",sha256=digests[c*FRAMES+f],width=WIDTH,height=HEIGHT)
        for c in range(CLIPS) for f in range(FRAMES)]}


def scale_pattern(limits):
    """Keep the frozen scalars; only native [0,0] controls remain exactly zero."""
    bounds=np.asarray(limits)
    if (np.ma.isMaskedArray(limits) or bounds.shape!=(68,2) or bounds.dtype.kind!="f" or np.isnan(bounds).any()
            or np.any(bounds[:,0]>bounds[:,1]) or np.any(bounds[:,0]>0) or np.any(bounds[:,1]<0)):
        raise ValueError("Actual68 scale bounds must permit exact neutral zero")
    locked=(bounds[:,0]==0)&(bounds[:,1]==0)
    if locked.all():raise ValueError("Native scales have no free control for three distinct identities")
    pattern=np.broadcast_to(np.asarray(SCALES,np.float32)[:,None],(CLIPS,68)).copy()
    pattern[:,locked]=0
    if np.any(pattern<bounds[:,0]) or np.any(pattern>bounds[:,1]):
        raise ValueError("Frozen nonlocked native scale pattern exceeds actual bounds; no clipping")
    if len({row.tobytes() for row in pattern})!=CLIPS:
        raise ValueError("Three distinct legal native scale identities required")
    return pattern


def named_controls(names,limits):
    """Frozen named articulation, shape45 and native documented last68 scales."""
    bounds=np.asarray(limits)
    if (not isinstance(names,list) or len(names)!=249 or len(set(names))!=249 or any(not isinstance(n,str) or not n for n in names)
            or np.ma.isMaskedArray(limits) or bounds.shape!=(249,2) or bounds.dtype.kind!="f" or np.isnan(bounds).any()
            or np.any(bounds[:,0]>bounds[:,1]) or np.any(bounds[:204,0]>0) or np.any(bounds[:204,1]<0)):
        raise ValueError("Actual unique249 parameter names/legal bounds required")
    patterns=scale_pattern(bounds[136:204])
    controls=np.zeros((CLIPS*FRAMES,204),np.float32);identity=np.zeros((CLIPS*FRAMES,45),np.float32);changes=[]
    for clip in range(CLIPS):
        if np.any(np.asarray(SHAPES[clip])<bounds[204:206,0]) or np.any(np.asarray(SHAPES[clip])>bounds[204:206,1]):
            raise ValueError("Frozen shape coefficients exceed actual bounds")
        side="r" if clip==1 else "l"
        for frame in range(FRAMES):
            row=clip*FRAMES+frame;controls[row,136:204]=patterns[clip];identity[row,:2]=SHAPES[clip]
            other="l" if side=="r" else "r"
            recipe=[(f"{side}_uparm_ry",.18+.055*frame),(f"{side}_elbow_bend",.27+.055*frame),
                    (f"{side}_wrist_ry",-.045+.02*frame),(f"{other}_uparm_ry",.025*(frame-2))]
            recipe += [(f"{side}_{finger}1_rz",.12+.04*frame) for finger in ("index","middle","ring","pinky")]
            for name,value in recipe:
                if name not in names[:136]:raise ValueError("Required actual named motion control absent: "+name)
                column=names.index(name)
                if not bounds[column,0]<=value<=bounds[column,1]:raise ValueError("Fixed articulation exceeds actual bounds: "+name)
                controls[row,column]=value;changes.append(dict(clip_index=clip,frame_index=frame,name=name,column=column,value=value))
    full=np.c_[controls,identity]
    if np.any(full<bounds[:,0]) or np.any(full>bounds[:,1]):
        raise ValueError("Complete animated249 controls/identity violate actual bounds")
    neutral=full[::FRAMES].copy();neutral[:,:136]=0
    if np.any(neutral<bounds[:,0]) or np.any(neutral>bounds[:,1]):
        raise ValueError("Complete neutral249 controls/identity violate actual bounds")
    return controls,identity,changes


def fixed_framing(neutral):
    value=np.asarray(neutral)
    if np.ma.isMaskedArray(neutral) or value.ndim!=3 or value.shape[0]!=CLIPS or value.shape[2]!=3 or not np.isfinite(value).all():
        raise ValueError("Three finite native neutral identity meshes required")
    points=value.reshape(-1,3);center=(points.min(0)+points.max(0))/2;relative=points-center
    distance=max(np.max(max(FOCALS)*np.abs(relative[:,0])/(WIDTH*.29)-relative[:,2]),
                 np.max(max(FOCALS)*np.abs(relative[:,1])/(HEIGHT*.29)-relative[:,2]))+.4
    return center,float(distance),float(relative[:,1].max()+.08)


def background(distance,floor):
    """Own varied depth-facing tiles/floor, geometry independent of clip identity."""
    vertices=[];faces=[];colors=[]
    for origin,u,v,color in (([-3,-2.2,4.5],[6,0,0],[0,floor+2.2,0],[.62,.67,.69]),
                            ([-3,floor,2.4-distance],[6,0,0],[0,0,distance+2.1],[.55,.50,.43])):
        origin,u,v,color=(np.asarray(x,float) for x in (origin,u,v,color))
        for i in range(10):
            for j in range(8):
                n=len(vertices);vertices.extend([origin+u*a/10+v*b/8 for a,b in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))])
                faces.extend([[n,n+1,n+2],[n,n+2,n+3]])
                tint=.032*((i+2*j)%3-1)+.008*np.cos(.7*i+.9*j)
                colors.extend([color+tint]*4)
    return np.asarray(vertices),np.asarray(faces,np.int64),np.asarray(colors)


def scene(human,human_faces,regions,center,distance,floor,clip,frame):
    from scipy.spatial.transform import Rotation
    if type(clip) is not int or type(frame) is not int or not 0<=clip<CLIPS or not 0<=frame<FRAMES:
        raise ValueError("One prescribed clip/frame required")
    actor=(human-center)@Rotation.from_rotvec([0.,(-.17,.14,-.08)[clip]+.05*(frame-2),0.]).as_matrix().T
    actor += [.014*(frame-2),.005*np.sin(frame),.025*(frame-2)]
    hand=actor[regions["r" if clip==1 else "l"]["vertex_mask"]]
    if len(hand)<50:raise ValueError("Semantic moving hand region absent")
    bottle,of=bottle_mesh();bottle*=1.12
    bottle=bottle@Rotation.from_rotvec([.04*(frame-2),.09*frame,-.055*(frame-2)]).as_matrix().T
    bottle += (hand.min(0)+hand.max(0))/2+[.035*(-1 if clip!=1 else 1),.022,(-.105,-.045,.015,.060,.105)[frame]]
    hc=np.tile([.75,.73,.70],(len(actor),1));low,high=human[:,1].min(),human[:,1].max()
    band=(human[:,1]-low)/(high-low);cloth=(band>.23)&(band<.81)
    cloth &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    # Continuous garment texture is not an identity/side label; materials same all clips.
    hc[cloth]=np.c_[.30+.025*np.sin(16*human[cloth,0]),.34+.025*np.cos(12*human[cloth,1]),.37+.02*np.sin(9*human[cloth,2])]
    oc=np.tile([.23,.41,.30],(len(bottle),1));oc[160:]=[.28,.29,.27]
    bv,bf,bc=background(distance,floor)
    vertices=np.r_[actor,bottle,bv]+[0.,0.,distance]
    faces=np.r_[human_faces,of+len(actor),bf+len(actor)+len(bottle)];colors=np.r_[hc,oc,bc]
    if not np.isfinite(vertices).all() or vertices[:,2].min()<=.01:raise ValueError("Entire own scene crosses camera near plane")
    K=np.array([[FOCALS[clip],0.,WIDTH/2],[0.,FOCALS[clip],HEIGHT/2],[0.,0.,1.]])
    return vertices[:len(actor)],vertices[len(actor):len(actor)+len(bottle)],of,vertices,faces,colors,K


def foreground_truth(face,depth,human_faces,object_faces):
    """Retain real foreground occlusion; room IDs never masquerade as objects."""
    if face.shape!=(HEIGHT,WIDTH) or depth.shape!=face.shape or face.dtype.kind not in "iu":raise ValueError("Original raster grids required")
    selected=(face>=0)&(face<len(human_faces)+len(object_faces))
    human=(face>=0)&(face<len(human_faces));obj=(face>=len(human_faces))&selected
    if human.sum()<64 or obj.sum()<64:raise ValueError("Both own rendered entities need visible support")
    if not np.isfinite(depth[selected]).all() or np.any(depth[selected]<=0):raise ValueError("Foreground camera-Z invalid")
    return np.where(selected,face,-1).astype(np.int64),np.where(selected,depth,np.nan).astype(np.float32)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!="Linux" or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}:raise RuntimeError("Remote offline CUDA render required")
    root=Path(os.environ["WR_ROOT"]);dest=root/BASE;revision,image=os.environ["WR_CODE_REVISION"],os.environ["WR_IMAGE_ID"]
    if (root!=Path("/srv/scenesmith/world-reward") or dest.resolve()!=dest.absolute() or not dest.is_dir() or any(dest.iterdir())
            or not re.fullmatch(r"[0-9a-f]{40}",revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}",image)):
        raise ValueError("Canonical empty reserved source/image-bound destination required")
    public,private=dest/"inputs",dest/"eval_private";public.mkdir();private.mkdir(mode=0o700)
    report=dict(stage="own_moving_identity_rgb_render",status="fail",phase="prerequisites",cases=[],budget_seconds=BUDGET,
        code_revision=revision,image_id=image,script_sha256=sha256(Path(__file__)),render_helper_sha256=sha256(Path(render.__file__)),
        joint_helper_sha256=sha256(Path(__file__).with_name("joint_rgb_render.py")),camera_helper_sha256=sha256(Path(__file__).with_name("camera_render.py")),
        challenge_inputs_used=False,synthetic_truth_used_for_rendering_only=True,inference_performed=False,accuracy_verified=False,
        photorealism_verified=False,all_truth_private=True,true_focals=list(FOCALS),inference_fixed_focal=1280.,
        shape_first_two_coefficients=SHAPES,scale68_scalar_candidates=SCALES,scale68_pattern_rule=SCALE_PATTERN_RULE,
        scale68_patterns=[],identity_clip_constant=True,truth_keys=sorted(TRUTH_KEYS),
        scene_depth_scope="human/object visible foreground only; background IDs=-1/depth=NaN",background_face_ids_removed_from_truth=True,
        truth_array_contract={"human_vertices_camera_m":"float64[18439,3]","human_faces":"integer[36874,3]",
            "object_vertices_camera_m":"float64[194,3]","object_faces":"int64[384,3]","camera_K":"float64[3,3]",
            "scene_depth_m":"float32[768,1024] NaNoutside","visible_face_indices":"int64[768,1024] -1outside",
            "clip_index":"int64 scalar","frame_index":"int64 scalar"},actual_reference_forward_calls=0)
    started=time.perf_counter();receipt=private/"render-report.json"
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole fresh identity render exceeds120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist();prereq=root/"results/mhr-finger-semantics-v4.json";prereq_sha=sha256(prereq)
            render.semantics.regular_hash(prereq,root,prereq_sha);semantic=json.loads(prereq.read_text());render.require_semantic_report(semantic)
            if semantic.get("source_image_id")!=image:raise ValueError("Renderer image must match semantic producer")
            with (private/"semantic-report.json").open("xb") as handle:handle.write(prereq.read_bytes())
            (private/"semantic-report.json").chmod(0o400)
            model_path=root/"weights/mhr/mhr_model.pt";render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            if "torch" in sys.modules:raise RuntimeError("CUBLAS setup precedes Torch import")
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__!="0.7.9":raise RuntimeError("Actual pinned CUDA renderer required")
            with torch.jit.optimized_execution(False):model=torch.jit.load(str(model_path),map_location="cuda").float().eval()
            names,joints=model.get_parameter_names(),model.get_joint_names();limits=model.get_parameter_limits().cpu().numpy()
            controls,identity,changes=named_controls(names,limits);faces=model.character_torch.mesh.faces.cpu().numpy()
            report.update(scale68_patterns=controls[::FRAMES,136:204].tolist(),scale_control_names=names[136:204],
                scale_control_bounds=limits[136:204].tolist(),
                zero_locked_scale_indices=np.flatnonzero(np.all(limits[136:204]==0,axis=1)).tolist())
            persist()
            if (faces.shape!=(36874,3) or faces.dtype.kind not in "iu" or np.any(faces<0) or np.any(faces>=18439) or joints!=semantic["joint_names"]
                    or (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())!=(45,72)):
                raise ValueError("Actual reference topology/joint ABI differs")
            regions=render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()),joints)
            neutral=controls[::FRAMES].copy();neutral[:,:136]=0
            p=torch.tensor(controls,device="cuda");ids=torch.tensor(identity,device="cuda");before=(p.clone(),ids.clone())
            with torch.inference_mode(),torch.jit.optimized_execution(False):
                raw,skeleton=model(ids,p,torch.zeros(15,72,device="cuda"),True);report["actual_reference_forward_calls"]+=1
                base,base_skeleton=model(ids[::FRAMES],torch.tensor(neutral,device="cuda"),torch.zeros(3,72,device="cuda"),True);report["actual_reference_forward_calls"]+=1
            torch.cuda.synchronize()
            if not torch.equal(p,before[0]) or not torch.equal(ids,before[1]):raise ValueError("Native reference mutated supplied identity/controls")
            render.semantics.check_geometry(raw.cpu().numpy(),skeleton.cpu().numpy(),127);render.semantics.check_geometry(base.cpu().numpy(),base_skeleton.cpu().numpy(),127)
            flip=np.diag([1.,-1.,-1.]);humans=raw.cpu().numpy()@flip/100;center,distance,floor=fixed_framing(base.cpu().numpy()@flip/100)
            report.update(phase="rendering",model_sha256=render.semantics.MODEL_SHA,semantic_report_sha256=prereq_sha,
                parameter_names=names,named_controls=changes,scale_control_names=names[136:204],scale_control_bounds=limits[136:204].tolist(),
                torch=torch.__version__,pytorch3d_revision=render.PYTORCH3D_REVISION)
            with (private/"rig.npz").open("xb") as handle:np.savez_compressed(handle,controls=controls,shape45=identity,parameter_limits=limits)
            (private/"rig.npz").chmod(0o400);report["rig_sha256"]=sha256(private/"rig.npz");digests=[];persist()
            for clip in range(CLIPS):
                for frame in range(FRAMES):
                    report.update(active_clip=clip,active_frame=frame);persist()
                    hv,ov,of,sv,sf,colors,K=scene(humans[clip*FRAMES+frame],faces,regions,center,distance,floor,clip,frame)
                    pixels=render.project_camera_points(hv,K)
                    if np.any(pixels<8) or np.any(pixels>=[WIDTH-8,HEIGHT-8]):raise ValueError("Fixed own actor framing fails")
                    with torch.inference_mode():rgb,face,depth=render.render_rgb(torch,sv,sf,colors,K)
                    torch.cuda.synchronize();face,depth=foreground_truth(face,depth,faces,of)
                    if rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8:raise ValueError("Original RGB grid invalid")
                    stem=f"clip_{clip:02d}_frame_{frame:03d}";truth=private/(stem+".npz");png=public/(stem+".png")
                    with truth.open("xb") as handle:np.savez_compressed(handle,human_vertices_camera_m=hv,human_faces=faces,
                        object_vertices_camera_m=ov,object_faces=of,camera_K=K,scene_depth_m=depth,visible_face_indices=face,
                        clip_index=np.array(clip,np.int64),frame_index=np.array(frame,np.int64))
                    truth.chmod(0o400)
                    with png.open("xb") as handle:Image.fromarray(rgb).save(handle,format="PNG")
                    png.chmod(0o444);digests.append(sha256(png));report["cases"].append(dict(clip_index=clip,frame_index=frame,file=png.name,rgb_sha256=digests[-1],truth_sha256=sha256(truth)));persist()
            if len(set(digests))!=15:raise ValueError("All new moving RGBs must differ; no cloning")
            render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            if sha256(prereq)!=prereq_sha:raise ValueError("Semantic source receipt changed")
            with (public/"manifest.json").open("x") as handle:json.dump(public_manifest(digests),handle)
            (public/"manifest.json").chmod(0o444)
            report.update(status="pass",phase="complete",frames=15,public_manifest_sha256=sha256(public/"manifest.json"),actual_MHR_reference_used=True)
            report.pop("active_clip",None);report.pop("active_frame",None)
        except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o400)


if __name__=="__main__":main()
