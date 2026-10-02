"""New own human/bottle RGB cohort; synthesis truth is private, never inference."""
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
from world_reward.data import sha256

WIDTH, HEIGHT, CLIPS, FRAMES = 1024, 768, 3, 3
FOCALS = (1280., 960., 1600.)
SCHEMA = "world-reward-joint-rgb-v1"
TRUTH_KEYS = {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces",
              "camera_K", "scene_depth_m", "visible_face_indices", "clip_index", "frame_index"}


def public_manifest(digests):
    if len(digests) != 9 or any(not isinstance(d, str) or not re.fullmatch("[0-9a-f]{64}", d) for d in digests):
        raise ValueError("Require nine ordered RGB SHA256 digests")
    return {"schema": SCHEMA, "images": [{"file": f"clip_{c:02d}_frame_{f:03d}.png",
        "sha256": digests[c*FRAMES+f], "width": WIDTH, "height": HEIGHT}
        for c in range(CLIPS) for f in range(FRAMES)]}


def named_controls(names, bounds):
    bounds = np.asarray(bounds)
    if (len(names) != 249 or len(set(names)) != 249 or bounds.shape != (249, 2)
            or bounds.dtype.kind != "f" or np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1])
            or np.any(bounds[:204, 0] > 0) or np.any(bounds[:204, 1] < 0)):
        raise ValueError("Require actual unique names/limits and legal neutral controls")
    poses = np.zeros((9, 204), np.float32); changes = []
    for clip in range(CLIPS):
        side = "l" if clip != 1 else "r"
        for frame in range(FRAMES):
            row = clip*FRAMES+frame
            recipe = [(f"{side}_uparm_ry", .10+.025*frame), (f"{side}_elbow_bend", .30+.04*frame)]
            recipe += [(f"{side}_{finger}1_rz", .18+.035*frame) for finger in ("index", "middle", "ring", "pinky")]
            for name, value in recipe:
                if name not in names[:204]: raise ValueError("Required real named control absent: "+name)
                column = names.index(name)
                if not bounds[column, 0] <= value <= bounds[column, 1]:
                    raise ValueError("Frozen control exceeds actual limit: "+name)
                poses[row, column] = value
                changes.append({"clip_index": clip, "frame_index": frame, "name": name, "column": column, "value": value})
    return poses, changes


def bottle_mesh():
    """Closed own asymmetric body/shoulder/neck/cap profile, no downloaded asset."""
    levels = np.array([[-.17,.052],[-.155,.061],[.065,.065],[.105,.030],[.155,.027],[.170,.030]])
    angle = np.arange(32)*2*np.pi/32
    vertices = []
    for y, radius in levels:
        r = radius*(1+.065*np.sin(angle)+.045*np.cos(2*angle))
        vertices.extend(np.column_stack((r*np.cos(angle), np.full(32,y), r*.90*np.sin(angle))))
    vertices.extend([[0,levels[0,0],0],[0,levels[-1,0],0]])
    faces = []
    for ring in range(len(levels)-1):
        for i in range(32):
            a,b=ring*32+i,ring*32+(i+1)%32; c,d=a+32,b+32
            faces.extend([[a,c,d],[a,d,b]])
    for i in range(32):
        faces.extend([[192,i,(i+1)%32],[193,160+(i+1)%32,160+i]])
    v,f=np.asarray(vertices,float),np.asarray(faces,np.int64)
    if np.einsum('ij,ij->i',v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]])).sum()<0: f=f[:,::-1].copy()
    return v,f


def fixed_camera(neutral):
    neutral=np.asarray(neutral)
    if neutral.ndim!=2 or neutral.shape[1]!=3 or len(neutral)<3 or not np.isfinite(neutral).all():
        raise ValueError('Require finite own neutral geometry')
    center=(neutral.min(0)+neutral.max(0))/2; relative=neutral-center
    distance=max(np.max(1600*np.abs(relative[:,0])/(WIDTH*.34)-relative[:,2]),
                 np.max(1600*np.abs(relative[:,1])/(HEIGHT*.34)-relative[:,2]))+.25
    if not np.isfinite(distance) or distance<=0: raise ValueError('Own neutral camera extent invalid')
    return np.array([-center[0],-center[1],distance-center[2]])


def scene(human, human_faces, regions, translation, clip, frame):
    from scipy.spatial.transform import Rotation
    R=Rotation.from_rotvec([0.,(-.08,.06,-.04)[clip]+.025*frame,0.]).as_matrix()
    t=np.asarray(translation)+[.008*(frame-1),0.,0.]
    hv=human@R.T+t; side="r" if clip==1 else "l"
    hand=hv[regions[side]["vertex_mask"]]; center=(hand.min(0)+hand.max(0))/2
    bv,bf=bottle_mesh()
    # Known own synthesis geometry only: front/back depth offsets, never inference inputs.
    center += [.055*(-1 if side=="l" else 1),.015,(0.,-.16,.16)[clip]]
    ov=bv@Rotation.from_rotvec([.025*frame,0.,.06*(frame-1)]).as_matrix().T+center
    hc=np.tile([.75,.75,.73],(len(hv),1)); low,high=human[:,1].min(),human[:,1].max()
    band=(human[:,1]-low)/(high-low); clothing=(band>.24)&(band<.80)
    clothing &= ~regions["l"]["vertex_mask"] & ~regions["r"]["vertex_mask"]
    hc[clothing]=[.35,.35,.35]
    oc=np.tile([.24,.42,.31],(len(ov),1)); oc[np.arange(len(ov))>=160]=[.29,.29,.28]
    vertices=np.r_[hv,ov]; faces=np.r_[human_faces,bf+len(hv)]; colors=np.r_[hc,oc]
    if not np.isfinite(vertices).all() or vertices[:,2].min()<=.01: raise ValueError("Own joint scene crosses near plane")
    K=np.array([[FOCALS[clip],0.,WIDTH/2],[0.,FOCALS[clip],HEIGHT/2],[0.,0.,1.]])
    return hv,ov,bf,vertices,faces,colors,K


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    if platform.system()!="Linux" or {p.name for p in Path('/sys/class/net').iterdir()}!={'lo'}:
        raise RuntimeError("Require remote CUDA network-none synthesis")
    root=Path(os.environ['WR_ROOT']); dest=root/'validation/joint_rgb_v1'
    if dest.is_symlink() or not dest.is_dir() or any(dest.iterdir()): raise FileExistsError("Require exclusive reserved J1 root")
    public,private=dest/'inputs',dest/'eval_private'; public.mkdir(); private.mkdir(mode=0o700)
    report={'stage':'own_joint_human_object_rgb_render','status':'fail','cases':[],'budget_seconds':120,
        'script_sha256':sha256(Path(__file__)),'render_helper_sha256':sha256(Path(render.__file__)),
        'camera_helper_sha256':sha256(Path(__file__).with_name('camera_render.py')),
        'code_revision':os.environ['WR_CODE_REVISION'],'image_id':os.environ['WR_IMAGE_ID'],
        'challenge_inputs_used':False,'synthetic_truth_used_for_rendering_only':True,'inference_performed':False,
        'photorealism_verified':False,'accuracy_verified':False,'true_focals':FOCALS,'inference_prior_focal':1280.,
        'camera_framing':'one neutral whole-body extent, same translation all clips/frames',
        'clip_contexts':['near_hand','bottle_front','bottle_back'],'fixed_identity_each_clip':'zero45','fixed_scales':'zero68'}
    start=time.perf_counter()
    if not re.fullmatch('[0-9a-f]{40}',report['code_revision']) or not re.fullmatch('sha256:[0-9a-f]{64}',report['image_id']):
        raise ValueError('Require immutable render source/image identities')
    with (private/'render-report.json').open('x') as stream:
        def persist():
            report['elapsed_seconds']=time.perf_counter()-start; stream.seek(0); json.dump(report,stream,allow_nan=False)
            stream.write('\n'); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError('Frozen120s joint render deadline')
        signal.signal(signal.SIGALRM,expired); signal.signal(signal.SIGTERM,expired); signal.alarm(120)
        try:
            persist(); prereq=root/'results/mhr-finger-semantics-v4.json'; ph=sha256(prereq)
            render.require_semantic_report(json.loads(prereq.read_text()))
            model_path=root/'weights/mhr/mhr_model.pt'
            render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            import torch
            import pytorch3d
            from PIL import Image
            render.semantics.strict_reference_runtime(torch)
            if not torch.cuda.is_available() or pytorch3d.__version__!='0.7.9': raise RuntimeError('Require actual CUDA audited raster')
            with torch.jit.optimized_execution(False): model=torch.jit.load(str(model_path),map_location='cuda').float().eval()
            names,joints=model.get_parameter_names(),model.get_joint_names()
            controls,changes=named_controls(names,model.get_parameter_limits().cpu().numpy())
            faces=model.character_torch.mesh.faces.cpu().numpy()
            if (faces.shape!=(36874,3) or faces.dtype.kind not in 'iu' or np.any(faces<0) or np.any(faces>=18439)
                    or joints!=json.loads(prereq.read_text())['joint_names']): raise ValueError('Reference topology/names ABI differs')
            if (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())!=(45,72): raise ValueError('Reference coefficient ABI differs')
            regions=render.lbs_regions(*(x.cpu().numpy() for x in model.get_lbsw()),joints)
            with torch.inference_mode(),torch.jit.optimized_execution(False):
                neutral,_=model(torch.zeros(1,45,device='cuda'),torch.zeros(1,204,device='cuda'),torch.zeros(1,72,device='cuda'),True)
                raw,skeleton=model(torch.zeros(9,45,device='cuda'),torch.as_tensor(controls,device='cuda'),torch.zeros(9,72,device='cuda'),True)
            torch.cuda.synchronize(); render.semantics.check_geometry(raw.cpu().numpy(),skeleton.cpu().numpy(),127)
            human=raw.cpu().numpy()@np.diag([1.,-1.,-1.])/100
            translation=fixed_camera(neutral[0].cpu().numpy()@np.diag([1.,-1.,-1.])/100)
            report.update(model_sha256=render.semantics.MODEL_SHA,semantic_report_sha256=ph,
                named_controls=changes,parameter_names=names,pytorch3d_revision=render.PYTORCH3D_REVISION)
            digests=[]
            for clip in range(CLIPS):
                for frame in range(FRAMES):
                    hv,ov,of,sv,sf,colors,K=scene(human[clip*FRAMES+frame],faces,regions,translation,clip,frame)
                    with torch.inference_mode(): rgb,face,depth=render.render_rgb(torch,sv,sf,colors,K)
                    torch.cuda.synchronize()
                    if (rgb.shape!=(HEIGHT,WIDTH,3) or rgb.dtype!=np.uint8 or np.any(face>=len(sf))
                            or np.count_nonzero((face>=0)&(face<len(faces)))<64 or np.count_nonzero(face>=len(faces))<64):
                        raise ValueError('Both rendered entities require visible original RGB support')
                    depth=np.where(face>=0,depth,np.nan).astype(np.float32)
                    if not np.isfinite(depth[face>=0]).all() or np.any(depth[face>=0]<=0): raise ValueError('Invalid scene cameraZ')
                    stem=f'clip_{clip:02d}_frame_{frame:03d}'; truth=private/(stem+'.npz')
                    with truth.open('xb') as handle: np.savez_compressed(handle,human_vertices_camera_m=hv,human_faces=faces,
                        object_vertices_camera_m=ov,object_faces=of,camera_K=K,scene_depth_m=depth,
                        visible_face_indices=face,clip_index=np.array(clip),frame_index=np.array(frame))
                    with (public/(stem+'.png')).open('xb') as handle: Image.fromarray(rgb).save(handle,format='PNG')
                    digests.append(sha256(public/(stem+'.png')))
                    report['cases'].append({'clip_index':clip,'frame_index':frame,'file':stem+'.png',
                        'truth_sha256':sha256(truth),'rgb_sha256':digests[-1]}); persist()
            render.semantics.regular_hash(model_path,root,render.semantics.MODEL_SHA,696110248)
            if sha256(prereq)!=ph: raise ValueError('Reference semantic provenance changed')
            with (public/'manifest.json').open('x') as handle: json.dump(public_manifest(digests),handle)
            report.update(status='pass',public_manifest_sha256=sha256(public/'manifest.json'))
        except Exception as exc: report.update(error_type=type(exc).__name__,error=str(exc)); raise
        finally: signal.alarm(0); persist()


if __name__=='__main__': main()
