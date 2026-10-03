"""Own 2×2×2 factorial RGB manufacture, not inference or accuracy validation.

Morphology, diffuse appearance and object occlusion vary independently. The
three native poses and the one camera are identical across appearance/occlusion.
All factor labels, native reference geometry and keypoints remain private.
"""
import argparse
import ast
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
import factorial_rgb_protocol as protocol
import hand_synthetic_render as render
import identity_rgb_render as primitives
from world_reward.data import sha256

COHORT = protocol.COHORT
BASE, STAGE, BUDGET = COHORT.base, "own_factorial_rgb_render", 120
WIDTH, HEIGHT, GROUPS, FRAMES = COHORT.width, COHORT.height, COHORT.groups, COHORT.frames
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
SHAPES, SCALES = ((.31,.09),(-.28,.18)), (.04,-.03)
BOTTLE_SCALE, OCCLUSION_OFFSETS = 1.45, (-.25,.25)
BODY_SHA, BODY_BYTES = render.semantics.BODY_SHA, 2109129346
BODY_RELATIVE = render.semantics.BODY_RELATIVE
BUNDLED_RIG_SHA, BUNDLED_RIG_BYTES = render.semantics.MODEL_SHA, 696110248
HEAD_SHA = "62af48b1f33462bc445d7f342009fcbceabd221f2a6e9bcd1d3739d582bc9fd6"
NAMES_SHA = "695c2c7d472e32757c480114fdb054d54ee4af53f69b2a6e040b00a55b270dc9"
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
PACKAGE = "vendor/video_to_data/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body"
TRUTH_KEYS = {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces", "camera_K",
    "scene_depth_m", "visible_face_indices", "group_index", "frame_index", "human_joints_camera_m", "human_keypoints_camera_m"}


def array_sha(value): return hashlib.sha256(value.tobytes()).hexdigest()


def helper_hashes():
    return dict(render=sha256(Path(render.__file__)), primitives=sha256(Path(primitives.__file__)),
        joint=sha256(Path(__file__).with_name("joint_rgb_render.py")), camera=sha256(Path(__file__).with_name("camera_render.py")),
        protocol=sha256(Path(protocol.__file__)))


def factors(group):
    if type(group) is not int or not 0 <= group < GROUPS: raise ValueError("Original factorial group required")
    return group//4, (group//2)%2, group%2


def named_controls(names, limits):
    bounds = np.asarray(limits)
    if (np.ma.isMaskedArray(limits) or not isinstance(names,list) or len(names)!=249 or len(set(names))!=249
        or any(type(n) is not str or not n for n in names) or bounds.shape!=(249,2) or bounds.dtype.kind!="f"
        or np.isnan(bounds).any() or np.any(bounds[:,0]>bounds[:,1])): raise ValueError("Actual unique249 names/legal bounds required")
    scales = np.broadcast_to(np.asarray(SCALES,np.float32)[:,None],(2,68)).copy()
    locked = np.all(bounds[136:204]==0,axis=1); scales[:,locked]=0
    if locked.all() or np.any(scales<bounds[136:204,0]) or np.any(scales>bounds[136:204,1]):
        raise ValueError("Fixed scale pattern must obey native limits; never clip")
    controls, identities, changes = np.zeros((6,204),np.float32), np.zeros((6,45),np.float32), []
    for morphology in range(2):
        for frame in range(FRAMES):
            row=morphology*FRAMES+frame;controls[row,136:]=scales[morphology];identities[row,:2]=SHAPES[morphology]
            recipe=[("l_uparm_ry",.19+.04*frame),("l_elbow_bend",.29+.055*frame),
                    ("l_wrist_ry",.01-.015*frame),("r_uparm_ry",-.025*(frame-1))]
            recipe += [(f"l_{finger}1_rz",.12+.025*frame) for finger in ("index","middle","ring","pinky")]
            for name,value in recipe:
                if name not in names[:136]:raise ValueError("Prescribed native articulation absent: "+name)
                column=names.index(name);controls[row,column]=value
                changes.append(dict(morphology_index=morphology,frame_index=frame,name=name,column=column,value=value))
    full=np.c_[controls,identities];neutral=full[::FRAMES].copy();neutral[:,:136]=0
    if (np.any(full<bounds[:,0]) or np.any(full>bounds[:,1]) or np.any(neutral<bounds[:,0]) or np.any(neutral>bounds[:,1])):
        raise ValueError("All prescribed animated/neutral249 values must be legal; no adaptation")
    return controls,identities,changes


def fixed_framing(neutral):
    v=np.asarray(neutral)
    if np.ma.isMaskedArray(neutral) or v.ndim!=3 or v.shape[0]!=2 or v.shape[2]!=3 or not np.isfinite(v).all():
        raise ValueError("Two finite own neutral identities required")
    points=v.reshape(-1,3);center=(points.min(0)+points.max(0))/2;relative=points-center
    distance=max(np.max(1280*np.abs(relative[:,0])/(WIDTH*.29)-relative[:,2]),
                 np.max(1280*np.abs(relative[:,1])/(HEIGHT*.29)-relative[:,2]))+.4
    if not np.isfinite(distance) or distance<=0:raise ValueError("Positive common camera distance required")
    return center,float(distance),float(relative[:,1].max()+.08)


def reference_landmarks(vertices_cm, skeleton, mapping):
    """Exact head ABI: map native V/J in metres BEFORE camera transforms.

    Mapping weights are checkpoint metadata, not assumed affine/unit-sum and
    never renormalized. The rig/geometry comes only from our reference forward.
    """
    v,s,m=np.asarray(vertices_cm),np.asarray(skeleton),np.asarray(mapping)
    if any(np.ma.isMaskedArray(x) for x in (vertices_cm,skeleton,mapping)):raise ValueError("Unmasked reference arrays required")
    if (v.ndim!=3 or v.shape[1:]!=(18439,3) or s.shape!=(len(v),127,8) or m.shape!=(308,18566)
        or any(x.dtype!=np.float32 or not np.isfinite(x).all() for x in (v,s,m))):raise ValueError("Actual F32 native reference/mapping ABI required")
    joints=s[...,:3]/np.float32(100.)
    combined=np.concatenate((v/np.float32(100.),joints),axis=1)
    # Same batched matrix contraction/reshape as pinned MHRHead L259–269.
    keypoints=(m@combined.transpose(1,0,2).reshape(18566,-1)).reshape(308,len(v),3).transpose(1,0,2)
    if not np.isfinite(keypoints).all():raise ValueError("Mapped own reference keypoints invalid")
    return joints,keypoints.copy()


def load_mapping(root,torch):
    root=Path(root);checkpoint=root/BODY_RELATIVE/"model.ckpt"
    render.semantics.regular_hash(checkpoint,root,BODY_SHA,BODY_BYTES)
    head=root/PACKAGE/"models/heads/mhr_head.py";names=root/PACKAGE/"metadata/mhr70.py"
    render.semantics.regular_hash(head,root,HEAD_SHA);render.semantics.regular_hash(names,root,NAMES_SHA)
    tree=ast.parse(names.read_text());name_values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="mhr_names" for t in n.targets)]
    if len(name_values)!=1 or len(name_values[0])!=70 or any(type(n)is not str for n in name_values[0]):
        raise ValueError("Actual source semantic70 names required")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    state=payload.get("state_dict",payload) if isinstance(payload,dict) else None
    value=state.get("head_pose.keypoint_mapping") if isinstance(state,dict) else None
    if not torch.is_tensor(value) or value.dtype!=torch.float32 or tuple(value.shape)!=(308,18566) or not torch.isfinite(value).all():
        raise ValueError("Exact filled head_pose.keypoint_mapping required")
    mapping=value.detach().cpu().numpy().copy();del payload,state,value
    bundled=root/BODY_RELATIVE/"assets/mhr_model.pt"
    acquisition=root/"results/weights-acquisition.json";render.semantics.regular_hash(acquisition,root,sha256(acquisition))
    acquired=json.loads(acquisition.read_text());rows=[row for row in acquired.get("assets",[])
        if row.get("repo_id")=="facebook/sam-3d-body-dinov3"]
    if (len(rows)!=1 or rows[0].get("revision")!=render.semantics.BODY_REVISION
        or Path(rows[0].get("path","")).resolve()!=(root/BODY_RELATIVE).resolve()):
        raise ValueError("Exact pinned existing Body rig acquisition required")
    render.semantics.regular_hash(bundled,root,BUNDLED_RIG_SHA,BUNDLED_RIG_BYTES)
    return mapping,dict(body_checkpoint_sha256=BODY_SHA,body_checkpoint_bytes=BODY_BYTES,
        head_source_sha256=HEAD_SHA,names_source_sha256=NAMES_SHA,keypoint_mapping_sha256=array_sha(mapping),
        keypoint_mapping_shape=[308,18566],keypoint_mapping_dtype="float32",first70_names=name_values[0],
        normalized_or_modified=False,source="head_pose.keypoint_mapping",contract="map concat(native_vertices_cm,native_joint_xyz_cm)/100 before camera basis/R/T",
        bundled_rig_sha256=BUNDLED_RIG_SHA,bundled_rig_bytes=BUNDLED_RIG_BYTES,
        acquisition_report_sha256=sha256(acquisition))


def bundled_parity(vertices,skeleton,bundled_vertices,bundled_skeleton):
    render.semantics.check_geometry(vertices,skeleton,127);render.semantics.check_geometry(bundled_vertices,bundled_skeleton,127)
    if vertices.shape!=bundled_vertices.shape or skeleton.shape!=bundled_skeleton.shape:raise ValueError("Identical bundled/reference batches required")
    errors=dict(vertices_m=float(np.linalg.norm((vertices.astype(np.float64)-bundled_vertices)/100,axis=-1).max()),
        joints_m=float(np.linalg.norm((skeleton[...,:3].astype(np.float64)-bundled_skeleton[...,:3])/100,axis=-1).max()))
    if any(not np.isfinite(v) or v>1e-5 for v in errors.values()):raise ValueError("Bundled mapping rig differs from reference geometry beyond1e-5m")
    return errors


def transform_points(points,center,distance,frame):
    from scipy.spatial.transform import Rotation
    if type(frame)is not int or not 0<=frame<FRAMES:raise ValueError("Prescribed pose index required")
    R=Rotation.from_rotvec([0.,.13+.035*(frame-1),0.]).as_matrix()
    return (points-center)@R.T+[.012*(frame-1),0.,distance]


def scene(human,faces,elbow,center,distance,floor,group,frame):
    morphology,appearance,occlusion=factors(group)
    actor=transform_points(human,center,distance,frame);ep=transform_points(np.asarray(elbow)[None],center,distance,frame)[0]
    bottle,of=primitives.bottle_mesh();bottle=bottle*BOTTLE_SCALE
    # Native bottle is upright; fixed axis orientation/centering for both depth orders.
    bottle+=ep-(bottle.min(0)+bottle.max(0))/2+[0.,0.,OCCLUSION_OFFSETS[occlusion]]
    hc=np.tile([.70,.68,.65],(len(actor),1));low,high=human[:,1].min(),human[:,1].max()
    if high<=low:raise ValueError("Physical own human extent required")
    cloth=(human[:,1]-low)/(high-low);cloth=(cloth>.23)&(cloth<.81)
    if appearance==0:hc[cloth]=[.31,.33,.36]
    else:
        stripe=(np.floor((human[cloth,0]+human[cloth,1])*22).astype(np.int64)%2).astype(bool)
        hc[cloth]=np.where(stripe[:,None],[.49,.45,.39],[.20,.27,.35])
    bv,bf,bc=primitives.background(distance,floor);oc=np.tile([.29,.40,.31],(len(bottle),1))
    vertices=np.r_[actor,bottle,bv];topology=np.r_[faces,of+len(actor),bf+len(actor)+len(bottle)]
    vertices[len(actor)+len(bottle):]+=[0.,0.,distance]
    if not np.isfinite(vertices).all() or vertices[:,2].min()<=.01:raise ValueError("Entire own scene crosses camera near plane")
    return actor,bottle,of,vertices,topology,np.r_[hc,oc,bc],np.asarray(COHORT.fixed_K,np.float64)


def occlusion_evidence(front,back,face_count):
    a,b=np.asarray(front),np.asarray(back)
    if a.shape!=(HEIGHT,WIDTH) or b.shape!=a.shape or a.dtype.kind not in "iu" or b.dtype.kind not in "iu":
        raise ValueError("Both original-grid paired face images required")
    human_a=(a>=0)&(a<face_count);human_b=(b>=0)&(b<face_count)
    newly_visible=int(np.count_nonzero(human_b&~human_a))
    if newly_visible<64:raise ValueError("Front/back factor does not expose64 extra human pixels; no fixture repair")
    return dict(newly_visible_human_pixels=newly_visible,front_human_pixels=int(human_a.sum()),back_human_pixels=int(human_b.sum()))


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!="Linux" or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}:raise RuntimeError("Remote offline CUDA manufacture required")
    root=Path(os.environ["WR_ROOT"]);dest=root/BASE;revision,image=os.environ["WR_CODE_REVISION"],os.environ["WR_IMAGE_ID"]
    if (root!=Path("/srv/scenesmith/world-reward") or dest.resolve()!=dest.absolute() or not dest.is_dir() or any(dest.iterdir())
        or any(p.is_symlink()for p in(dest,*dest.parents)) or os.geteuid()!=1000 or image!=IMAGE or not re.fullmatch("[0-9a-f]{40}",revision)):
        raise ValueError("Empty canonical reserved source/image-bound destination required")
    public,private=dest/"inputs",dest/"eval_private";public.mkdir();private.mkdir(mode=0o700)
    started=time.perf_counter();helpers=helper_hashes();source_sha=sha256(Path(__file__));receipt=private/"render-report.json"
    report=dict(stage=STAGE,status="fail",phase="prerequisites",frames=24,groups=8,cases=[],budget_seconds=BUDGET,
        code_revision=revision,image_id=image,script_sha256=source_sha,helper_source_sha256=helpers,network="none",
        challenge_inputs_used=False,inference_performed=False,optimizer_performed=False,quality_verified=False,
        accuracy_verified=False,photorealism_verified=False,all_truth_private=True,synthetic_truth_used_for_rendering_only=True,
        true_camera_private=True,license_clearance_verified=False,training_overlap_excluded=False,actual_reference_forward_calls=0,
        reference_forward_attempts=0,reference_forward_returns=0,raster_attempts=0,raster_returns=0,
        bundled_forward_attempts=0,bundled_forward_returns=0,bundled_forward_validated=0,
        morphology_shape_first_two=SHAPES,morphology_scale68_scalars=SCALES,bottle_manufacturing_scale=BOTTLE_SCALE,
        occlusion_Z_offsets=OCCLUSION_OFFSETS,factor_order="morphology-major,appearance-middle,occlusion-minor",
        camera_rule="one camera from both native neutral identities; no per-case adjustment",truth_keys=sorted(TRUTH_KEYS))
    with receipt.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Factorial RGB manufacture exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:
            persist();prereq=root/"results/mhr-finger-semantics-v4.json";prereq_sha=sha256(prereq)
            render.semantics.regular_hash(prereq,root,prereq_sha);semantic=json.loads(prereq.read_text());render.require_semantic_report(semantic)
            if semantic.get("source_image_id")!=image:raise ValueError("Actual semantic/reference image differs")
            (private/"semantic-report.json").write_bytes(prereq.read_bytes());(private/"semantic-report.json").chmod(0o400)
            model_path=root/"weights/mhr/mhr_model.pt";render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            if "torch"in sys.modules:raise RuntimeError("Strict CUBLAS setup must precede Torch")
            os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__!="0.7.9":raise RuntimeError("Pinned native CUDA raster required")
            mapping,mapping_id=load_mapping(root,torch)
            with torch.jit.optimized_execution(False):model=torch.jit.load(str(model_path),map_location="cuda").float().eval()
            names,joints=model.get_parameter_names(),model.get_joint_names();bounds=model.get_parameter_limits().cpu().numpy()
            controls,identities,changes=named_controls(names,bounds);faces=model.character_torch.mesh.faces.cpu().numpy()
            if (faces.dtype!=np.int32 or faces.shape!=(36874,3) or faces.min()<0 or faces.max()>=18439 or array_sha(faces)!=FACE_SHA
                or joints!=semantic["joint_names"] or (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())!=(45,72)):
                raise ValueError("Actual source-bound native topology/joints/identity ABI differs")
            regions=render.lbs_regions(*(x.cpu().numpy()for x in model.get_lbsw()),joints)
            neutral=controls[::FRAMES].copy();neutral[:,:136]=0;p=torch.tensor(controls,device="cuda");ids=torch.tensor(identities,device="cuda")
            with torch.inference_mode(),torch.jit.optimized_execution(False):
                report["reference_forward_attempts"]+=1;persist();raw,sk=model(ids,p,torch.zeros(6,72,device="cuda"),True)
                report["reference_forward_returns"]+=1;persist();report["reference_forward_attempts"]+=1;persist()
                nv,ns=model(ids[::FRAMES],torch.tensor(neutral,device="cuda"),torch.zeros(2,72,device="cuda"),True)
                report["reference_forward_returns"]+=1;persist()
            torch.cuda.synchronize();raw,sk,nv,ns=(v.cpu().numpy()for v in(raw,sk,nv,ns))
            render.semantics.check_geometry(raw,sk,127);render.semantics.check_geometry(nv,ns,127)
            if not np.array_equal(p.cpu().numpy(),controls) or not np.array_equal(ids.cpu().numpy(),identities):raise ValueError("Own native model mutated manufacturing parameters")
            with torch.jit.optimized_execution(False):bundled=torch.jit.load(str(root/BODY_RELATIVE/"assets/mhr_model.pt"),map_location="cuda").float().eval()
            bundled_faces=bundled.character_torch.mesh.faces.cpu().numpy()
            if bundled.get_joint_names()!=joints or bundled_faces.shape!=faces.shape or not np.array_equal(bundled_faces,faces):
                raise ValueError("Mapping/checkpoint rig and reference joint names/topology differ")
            with torch.inference_mode(),torch.jit.optimized_execution(False):
                report["bundled_forward_attempts"]+=1;persist();bv,bs=bundled(ids,p,torch.zeros(6,72,device="cuda"),True)
                report["bundled_forward_returns"]+=1;persist()
            torch.cuda.synchronize();errors=bundled_parity(raw,sk,bv.cpu().numpy(),bs.cpu().numpy())
            if not np.array_equal(p.cpu().numpy(),controls) or not np.array_equal(ids.cpu().numpy(),identities):raise ValueError("Bundled rig mutated own249 parameters")
            report.update(bundled_forward_validated=1,bundled_reference_parity=errors,bundled_reference_same_joint_names=True,
                bundled_reference_same_topology=True,actual_total_native_forward_calls=3);del bundled,bv,bs
            j3d,k3d=reference_landmarks(raw,sk,mapping);flip=np.array([1.,-1.,-1.]);humans=raw/100*flip;j3d=j3d*flip;k3d=k3d*flip
            center,distance,floor=fixed_framing(nv/100*flip)
            with(private/"rig.npz").open("xb")as handle:np.savez_compressed(handle,controls=controls,shape45=identities,parameter_limits=bounds,
                hand_mask_left=regions["l"]["vertex_mask"],hand_mask_right=regions["r"]["vertex_mask"])
            (private/"rig.npz").chmod(0o400)
            report.update(phase="rendering",actual_reference_forward_calls=2,actual_reference_geometry_count=6,
                model_sha256=render.semantics.MODEL_SHA,semantic_report_sha256=prereq_sha,rig_sha256=sha256(private/"rig.npz"),
                mapping=mapping_id,parameter_names=names,joint_names=joints,named_controls=changes,human_faces_sha256=FACE_SHA,
                human_faces_dtype="int32",torch=str(torch.__version__),pytorch3d_revision=render.PYTORCH3D_REVISION,
                coordinate_rule="map reference nativeV/J cm/100 before diag(1,-1,-1), then identical own yaw/camera translation")
            digests=[];pair_faces={};occlusions=[]
            for group in range(GROUPS):
                morphology,appearance,occlusion=factors(group)
                for frame in range(FRAMES):
                    index=morphology*FRAMES+frame;report.update(active_group=group,active_frame=frame);persist()
                    hv,ov,of,sv,sf,colors,K=scene(humans[index],faces,k3d[index,7],center,distance,floor,group,frame)
                    pixels=render.project_camera_points(hv,K)
                    if np.any(pixels<8) or np.any(pixels>=[WIDTH-8,HEIGHT-8]):raise ValueError("Frozen full-body framing failed; no camera adjustment")
                    report["raster_attempts"]+=1;persist()
                    with torch.inference_mode():rgb,face,depth=render.render_rgb(torch,sv,sf,colors,K)
                    report["raster_returns"]+=1;torch.cuda.synchronize();face,depth=primitives.foreground_truth(face,depth,faces,of)
                    if rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8:raise ValueError("Original RGB grid required")
                    pair=(morphology,appearance,frame)
                    if occlusion==0:pair_faces[pair]=face
                    else:occlusions.append(dict(morphology_index=morphology,appearance_index=appearance,frame_index=frame,
                        **occlusion_evidence(pair_faces.pop(pair),face,len(faces))))
                    stem=COHORT.frame_name(group*FRAMES+frame)[:-4];truth,png=private/(stem+".npz"),public/(stem+".png")
                    with truth.open("xb")as handle:np.savez_compressed(handle,human_vertices_camera_m=hv,human_faces=faces,
                        object_vertices_camera_m=ov,object_faces=of,camera_K=K,scene_depth_m=depth,visible_face_indices=face,
                        group_index=np.array(group,np.int64),frame_index=np.array(frame,np.int64),
                        human_joints_camera_m=transform_points(j3d[index],center,distance,frame),
                        human_keypoints_camera_m=transform_points(k3d[index],center,distance,frame))
                    truth.chmod(0o400)
                    with png.open("xb")as handle:Image.fromarray(rgb).save(handle,format="PNG")
                    png.chmod(0o444);digests.append(sha256(png));report["cases"].append(dict(file=png.name,rgb_sha256=digests[-1],
                        truth_sha256=sha256(truth),group_index=group,frame_index=frame,morphology_index=morphology,
                        appearance_index=appearance,occlusion_index=occlusion,visible_human_pixels=int(np.count_nonzero((face>=0)&(face<len(faces)))),
                        visible_object_pixels=int(np.count_nonzero(face>=len(faces)))));persist()
            if len(set(digests))!=24 or pair_faces:raise ValueError("All24 unique factor RGBs and complete occlusion pairs required")
            render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            render.semantics.regular_hash(root/BODY_RELATIVE/"model.ckpt",root,BODY_SHA,BODY_BYTES)
            render.semantics.regular_hash(root/BODY_RELATIVE/"assets/mhr_model.pt",root,mapping_id["bundled_rig_sha256"],mapping_id["bundled_rig_bytes"])
            render.semantics.regular_hash(root/"results/weights-acquisition.json",root,mapping_id["acquisition_report_sha256"])
            render.semantics.regular_hash(root/PACKAGE/"models/heads/mhr_head.py",root,HEAD_SHA)
            render.semantics.regular_hash(root/PACKAGE/"metadata/mhr70.py",root,NAMES_SHA)
            if sha256(prereq)!=prereq_sha or helper_hashes()!=helpers or sha256(Path(__file__))!=source_sha:raise ValueError("Source/model/semantic bytes changed")
            with(public/"manifest.json").open("x")as handle:json.dump(protocol.public_manifest(digests),handle)
            (public/"manifest.json").chmod(0o444);protocol.public_inputs(public)
            expected={"rig.npz","semantic-report.json","render-report.json",*[COHORT.frame_name(i)[:-4]+".npz" for i in range(24)]}
            if {p.name for p in private.iterdir()}!=expected:raise ValueError("Exact private manufacturing inventory required")
            report.update(status="pass",phase="complete",actual_MHR_reference_used=True,all_factors_geometry_independent=True,
                occlusion_evidence=occlusions,public_manifest_sha256=sha256(public/"manifest.json"),public_manifest_bytes=(public/"manifest.json").stat().st_size)
            report.pop("active_group",None);report.pop("active_frame",None)
        except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o400)


if __name__=="__main__":main()
