"""One private procedural MHR near-grasp capability, never RGB/test inference.

Fresh neutral controls and the own fixed 194V/384F bottle are the only geometry
sources. A legal, bounded finger/object solve may FAIL; no old poses, shrinking,
surface repair or less conservative collision proof is a fallback. Positive
0.5mm-gap distal surface proximity is not touching, force closure, biomechanics,
motion, rendered evidence, or challenge performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys
import time

import numpy as np
from world_reward.cross_surface import audit_closed_surface, audit_cross_surfaces

ROOT=Path("/srv/scenesmith/world-reward")
OUTPUT="validation/own_grasp_capability_v1"
IMAGE="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
MODEL_SHA="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
MODEL_BYTES=696110248
BUDGET=300
MAX_FORWARDS=100
SEARCH_STATES=86
GAP=.0005
TOLERANCE=1e-8
SIDE="l"
FINGER=re.compile(r"([lr])_(thumb|index|middle|ring|pinky)([0-3])_r([xyz])")
JOINT=re.compile(r"([lr])_(thumb|index|middle|ring|pinky)(?:[0-3]|_?null)")
HELPERS=("infra/own_grasp_capability.py","infra/run_own_grasp_capability.sh",
    "src/world_reward/cross_surface.py","src/world_reward/__init__.py")


def bottle_mesh():
    """Exact own asymmetric geometry; no asset import or scale adaptation."""
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


def array_id(value):
    value=np.ascontiguousarray(value)
    digest=hashlib.sha256(str(value.dtype).encode()+repr(value.shape).encode())
    digest.update(value.tobytes());return digest.hexdigest()


def identity(path,immutable=False):
    path=Path(path)
    if not path.is_absolute() or path.resolve()!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError("Canonical nonsymlink source file required")
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size<=0 or immutable and before.st_mode&0o222:
        raise ValueError("Nonempty regular source / immutable code required")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    after=path.lstat()
    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_mode)!=(
            after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_mode):
        raise ValueError("Source changed while hashing")
    return dict(sha256=digest.hexdigest(),bytes=after.st_size)


def bound_code(root,code,revision):
    root,code=Path(root),Path(code)
    if (not re.fullmatch("[0-9a-f]{40}",revision) or code.parent.parent.parent!=root/"jobs"
            or code.parent.parent.name!=revision or code.parent.name!="run_own_grasp_capability"
            or code.name!="code" or code.resolve()!=code or not code.is_dir()
            or any(p.is_symlink() for p in (code,*code.parents))):
        raise ValueError("Exact immutable Azure jobs/revision/bundle/code namespace required")
    return code


def exact_mount(text,path,readonly):
    matches=[]
    for line in text.splitlines():
        fields=line.split();sep=fields.index("-") if "-" in fields else -1
        if sep<6 or len(fields)<sep+4:raise ValueError("Malformed kernel mountinfo")
        mount=re.sub(r"\\(040|011|012|134)",lambda m:chr(int(m[1],8)),fields[4])
        if re.search(r"\\[0-9]",mount):raise ValueError("Unsupported mount path escape")
        if mount==str(path):
            opts=fields[5].split(",");mode="ro" if readonly else "rw"
            if mode not in opts or ("rw" if readonly else "ro") in opts:raise ValueError("Exact source/output mount mode differs")
            matches.append(dict(mount_point=mount,mount_options=opts))
    if len(matches)!=1:raise ValueError("Exactly one explicitly bound source/output mount required")
    return matches[0]


def legal_controls(controls,bounds):
    if np.ma.isMaskedArray(controls) or np.ma.isMaskedArray(bounds):raise ValueError("Masked native controls/bounds forbidden")
    q,b=np.asarray(controls),np.asarray(bounds)
    if (q.shape!=(204,) or q.dtype!=np.float32 or not np.isfinite(q).all()
            or b.shape!=(249,2) or b.dtype.kind!="f" or np.isnan(b).any() or np.any(b[:,0]>b[:,1])):
        raise ValueError("Exact native204 and actual249 dense bounds required")
    full=np.r_[q,np.zeros(45,np.float32)]
    if np.any(full<b[:,0]) or np.any(full>b[:,1]) or q[136:].tobytes()!=np.zeros(68,np.float32).tobytes():
        raise ValueError("Every manufactured native249 control must be legal; scales/shape remain exact zero")
    return q


def native_input_identity(controls,identity_coeffs,expression):
    """Pure exact-byte snapshot for every delegated original forward input."""
    q,i,e=map(np.asarray,(controls,identity_coeffs,expression))
    if (q.shape!=(204,) or i.shape!=(1,45) or e.shape!=(1,72)
            or any(np.ma.isMaskedArray(v) for v in (controls,identity_coeffs,expression))
            or any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in (q,i,e))
            or i.tobytes()!=np.zeros((1,45),np.float32).tobytes()
            or e.tobytes()!=np.zeros((1,72),np.float32).tobytes()):
        raise ValueError("Every native identity/expression must be literal fresh finite float32 zero")
    return dict(q=array_id(q),identity=array_id(i),expression=array_id(e))


def clear_failed_payloads(out):
    """Remove only this exclusive attempt's own disposable private payloads."""
    for name in ("controls.npz","geometry.npz"):
        path=Path(out)/name
        if path.is_file() and not path.is_symlink():path.unlink()


def native_metadata_identity(names,joints,parents,transform,bounds,faces,lbs_indices,lbs_weights):
    return dict(parameter_names=list(names),joint_names=list(joints),parents_sha256=array_id(parents),
        transform_sha256=array_id(transform),bounds_sha256=array_id(bounds),faces_sha256=array_id(faces),
        lbs_sha256=[array_id(lbs_indices),array_id(lbs_weights)])


def metadata(names,joints,parents,transform,bounds,lbs_indices,lbs_weights,faces):
    """Actual getter/tree/map/LBS semantics, never guessed anatomical vertex IDs."""
    if any(np.ma.isMaskedArray(v) for v in (parents,transform,bounds,lbs_indices,lbs_weights,faces)):
        raise ValueError("Masked original native metadata forbidden")
    p,m,idx,w,f=map(np.asarray,(parents,transform,lbs_indices,lbs_weights,faces))
    if (type(names) is not list or len(names)!=249 or len(set(names))!=249 or type(joints) is not list
            or len(joints)!=127 or len(set(joints))!=127 or any(type(n) is not str or not n for n in names+joints)
            or p.shape!=(127,) or p.dtype.kind not in "iu" or np.any(p< -1) or np.any(p>=127)
            or np.count_nonzero(p==-1)!=1 or m.shape!=(889,249) or m.dtype.kind!="f" or not np.isfinite(m).all()
            or idx.shape!=w.shape or idx.ndim!=2 or idx.shape[0]!=18439 or idx.dtype.kind not in "iu"
            or w.dtype.kind!="f" or not np.isfinite(w).all() or np.any(idx<0) or np.any(idx>=127)
            or np.any(w<0) or not np.allclose(w.sum(1),1.,atol=1e-5,rtol=0)
            or f.shape!=(36874,3) or f.dtype.kind not in "iu" or np.any(f<0) or np.any(f>=18439)):
        raise ValueError("Original native names/tree/889x249 map/full topology/normalized LBS ABI required")
    legal_controls(np.zeros(204,np.float32),bounds)
    chains=[]
    for first in range(127):
        chain=set();current=first
        while current!=-1:
            if current in chain:raise ValueError("Cyclic original joint parent tree")
            chain.add(current);current=int(p[current])
        chains.append(chain)
    matches=[FINGER.fullmatch(n) for n in names[:204]]
    if [i for i,x in enumerate(matches) if x]!=list(range(68,122)):
        raise ValueError("Original semantic finger columns must partition exactly68:122")
    active=[];distal={};records=[]
    for finger in ("thumb","index"):
        stem=f"{SIDE}_{finger}3";wrist=f"{SIDE}_wrist"
        if stem not in joints or wrist not in joints:raise ValueError("Original named distal joint/wrist absent")
        joint=joints.index(stem);closure=[j for j,c in enumerate(chains) if joint in c]
        if any(not JOINT.fullmatch(joints[j]) or JOINT.fullmatch(joints[j]).groups()[:2]!=(SIDE,finger) for j in closure):
            raise ValueError("Distal closure escapes the original named finger")
        mass=np.where(np.isin(idx,closure),w,0.).sum(1)
        patch=np.flatnonzero(np.all(mass[f]>=.5,axis=1))
        if len(patch)<3:raise ValueError("No full original distal triangle patch supported by dominant LBS")
        distal[finger]=dict(joint=joint,parent=int(p[joint]),closure=closure,faces=patch,mass=mass)
    for col,match in enumerate(matches):
        if not match or match.groups()[:2] not in ((SIDE,"thumb"),(SIDE,"index")):continue
        affected=np.flatnonzero(m[:,col]!=0);local=sorted(set((affected//7).tolist()))
        if (not len(affected) or np.any(~np.isin(affected%7,[3,4,5])) or any(
                not JOINT.fullmatch(joints[j]) or JOINT.fullmatch(joints[j]).groups()[:2]!=match.groups()[:2]
                or joints.index(f"{SIDE}_wrist") not in chains[j] for j in local)):
            raise ValueError("Original named control must change only its finger's local rotations")
        lower,upper=np.asarray(bounds)[col]
        if not np.isfinite([lower,upper]).all():raise ValueError("Finite actual finger bounds required for bounded manufacture")
        if lower<upper:active.append(col)
        records.append(dict(column=col,name=names[col],limits=[float(lower),float(upper)],local_joints=local))
    if len(active)<2:raise ValueError("Too few original legal free thumb/index controls")
    return dict(active=np.asarray(active,np.int64),distal=distal,control_semantics=records,
        wrist=joints.index(f"{SIDE}_wrist"),names=names,joints=joints)


def centers_normals(vertices,faces):
    t=np.asarray(vertices,np.float64)[np.asarray(faces)]
    normals=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);area=np.linalg.norm(normals,axis=1)
    if not np.isfinite(t).all() or np.any(area<=0):raise ValueError("Original nondegenerate finite triangles required")
    return t.mean(1),normals/area[:,None]


def select_patches(vertices,skeleton,faces,semantic,bv,bf):
    """Deterministic anatomical distal faces + actual opposed neck triangles.

    Scores use this freshly decoded neutral mesh only. Face centroids are exact
    interior barycentric surface points; no hand bounding box is a grasp target.
    """
    centers,normals=centers_normals(vertices,faces);bc,bn=centers_normals(bv,bf)
    chosen={}
    for finger,record in semantic["distal"].items():
        axis=skeleton[record["joint"],:3]-skeleton[record["parent"],:3]
        length=np.linalg.norm(axis)
        if not np.isfinite(length) or length<=1e-6:raise ValueError("Original distal anatomical axis unavailable")
        candidates=record["faces"];projection=centers[candidates]@(axis/length)
        order=np.lexsort((candidates,-projection));chosen[finger]=candidates[order[:min(64,max(3,len(candidates)//2))]]
    # Ring3→4 side triangles only, the unmodified sloping/asymmetric neck.
    neck=np.arange(3*64,4*64);best=None
    for a in neck:
        for b in neck:
            if a>=b or bn[a]@bn[b]>-.95:continue
            od=bc[b]-bc[a];distance=np.linalg.norm(od);direction=od/distance
            if bn[a]@(-direction)<.8 or bn[b]@direction<.8:continue
            for h in chosen["thumb"]:
                hd=centers[chosen["index"]]-centers[h];lengths=np.linalg.norm(hd,axis=1)
                valid=lengths>1e-6;unit=hd/np.maximum(lengths[:,None],1e-6)
                opposing=(normals[h]@unit.T>=.5)&(np.einsum("ij,ij->i",normals[chosen["index"]],-unit)>=.5)
                scores=np.abs(lengths-(distance+2*GAP))+.005*(1+normals[chosen["index"]]@normals[h])
                for j in np.flatnonzero(valid&opposing):
                    item=(float(scores[j]),int(h),int(chosen["index"][j]),int(a),int(b))
                    if best is None or item<best:best=item
    if best is None:raise ValueError("No actual inward-facing distal thumb/index and opposed original neck surface pair")
    _,h,i,a,b=best;human=np.array([h,i],np.int64);object_faces=np.array([a,b],np.int64)
    hd=centers[i]-centers[h];hx=hd/np.linalg.norm(hd)
    # Positive bottle axis points toward wrist: the unchanged long body extends away.
    hy=skeleton[semantic["wrist"],:3]-centers[human].mean(0);hy-=hx*(hy@hx)
    if np.linalg.norm(hy)<=1e-6:raise ValueError("Anatomical wrist-to-distal grasp frame is singular")
    hy/=np.linalg.norm(hy);hz=np.cross(hx,hy)
    ox=bc[b]-bc[a];ox/=np.linalg.norm(ox);oy=np.array([0.,1.,0.]);oy-=ox*(oy@ox);oy/=np.linalg.norm(oy)
    R=np.column_stack((hx,hy,hz))@np.column_stack((ox,oy,np.cross(ox,oy))).T
    t=centers[human].mean(0)-bc[object_faces].mean(0)@R.T
    return dict(human_faces=human,object_faces=object_faces,R=R,t=t,
        neutral_gap_m=float(np.linalg.norm(hd)),original_neck_span_m=float(np.linalg.norm(bc[b]-bc[a])),selection_score=best[0])


def contact_evidence(hv,hf,ov,of,patches):
    hp,hn=centers_normals(hv,hf[patches["human_faces"]]);op,on=centers_normals(ov,of[patches["object_faces"]])
    delta=hp-op;distance=np.linalg.norm(delta,axis=1);signed=np.einsum("ij,ij->i",delta,on)
    alignment=np.einsum("ij,ij->i",hn,-on);opposition=float(on[0]@on[1])
    valid=(np.isfinite(distance).all() and np.all((distance>TOLERANCE)&(distance<=.002))
        and np.all((signed>=GAP/2)&(signed<=GAP*1.5)) and np.all(alignment>=.9) and opposition<=-.95
        and np.all(np.linalg.norm(delta-GAP*on,axis=1)<=.0005))
    return dict(passed=bool(valid),human_faces=patches["human_faces"].tolist(),object_faces=patches["object_faces"].tolist(),
        human_barycentric=[[1/3]*3]*2,object_barycentric=[[1/3]*3]*2,surface_pair_distance_m=distance.tolist(),
        signed_outward_gap_m=signed.tolist(),opposed_human_object_normal_cosine=alignment.tolist(),
        object_opposed_normal_dot=opposition,target_positive_gap_m=GAP,proximity_only=True,
        touching_certified=False,force_closure_verified=False)


def feasible_step(current,delta,lower,upper,max_component=.05):
    """Scale one proposed legal update; never clip/repair a native control."""
    q,d,lo,hi=map(lambda x:np.asarray(x,np.float64),(current,delta,lower,upper))
    if (q.shape!=d.shape or lo.shape!=q.shape or hi.shape!=q.shape or not np.isfinite([q,d,lo,hi]).all()
            or np.any(lo>hi) or np.any(q<lo) or np.any(q>hi)):
        raise ValueError("Finite already-legal bounded update required")
    scale=min(1.,max_component/max(float(np.abs(d).max()),max_component))
    for value,step,l,u in zip(q,d,lo,hi):
        if step>0:scale=min(scale,(u-value)/step*.99)
        elif step<0:scale=min(scale,(l-value)/step*.99)
    if scale<=1e-9:raise ValueError("Legal surface solve is blocked at an actual anatomical bound")
    result=(q+scale*d).astype(np.float32)
    if np.any(result<lo) or np.any(result>hi):raise ValueError("Float32 update exceeds actual bounds; no clipping")
    return result


def five_point_offsets(value,lower,upper):
    left,right=value-lower,upper-value
    if min(left,right)>=.002:return np.array([-2,-1,1,2],float)*.001,"central"
    sign=1. if right>=left else -1.;h=min(.001,max(left,right)/4)
    if not np.isfinite(h) or h<1e-5:raise ValueError("Actual native bound has no resolvable five-point derivative room")
    return sign*np.arange(1,5)*h,"one_sided"


def five_point_derivative(base,values,offsets,mode):
    values=np.asarray(values,np.float64);base=np.asarray(base,np.float64);o=np.asarray(offsets)
    if values.shape!=(4,*base.shape) or not np.isfinite(values).all():raise ValueError("Four actual finite FD forwards required")
    if mode=="central":return (values[0]-8*values[1]+8*values[2]-values[3])/(12*o[2])
    if mode=="one_sided":return (-25*base+48*values[0]-36*values[1]+16*values[2]-3*values[3])/(12*o[0])
    raise ValueError("Declared five-point stencil required")


def require_closed(certificate):
    if (type(certificate) is not dict or certificate.get("status")!="pass" or certificate.get("full_original_faces_retained") is not True
            or certificate.get("mesh",{}).get("embedding_verified") is not True
            or certificate.get("embedding",{}).get("verified") is not True
            or type(certificate.get("tolerance_m")) is not float or certificate["tolerance_m"]!=TOLERANCE
            or type(certificate.get("max_candidate_pairs")) is not int or certificate["max_candidate_pairs"]!=2_000_000
            or certificate.get("exact_arithmetic_proof") is not False):
        raise ValueError("Fresh full original neutral closed/outward/self-embedded surface prerequisite failed")


def require_cross(certificate):
    if (type(certificate) is not dict or certificate.get("status")!="pass" or certificate.get("nonpenetration_certified") is not True
            or certificate.get("full_original_faces_retained") is not True or certificate.get("broadphase_complete") is not True
            or certificate.get("sampled_collision_test_used") is not False or certificate.get("touching_certified") is not False
            or certificate.get("force_closure_verified") is not False or certificate.get("exact_arithmetic_proof") is not False
            or any(certificate.get("intersection_counts",{}).values()) or len(certificate.get("embedding",[]))!=2
            or not all(row.get("verified") is True for row in certificate["embedding"])
            or type(certificate.get("containment")) is not list or len(certificate["containment"])!=2
            or any(type(rows) is not list or not rows or any(type(row) is not dict or row.get("state")!="outside" for row in rows)
                for rows in certificate["containment"])
            or set(certificate.get("intersection_counts",{}))!={"proper_intersection","coplanar_overlap","boundary_contact","ambiguous"}
            or any(type(v) is not int for v in certificate["intersection_counts"].values())
            or type(certificate.get("tolerance_m")) is not float or certificate["tolerance_m"]!=TOLERANCE
            or type(certificate.get("proximity_m")) is not float or certificate["proximity_m"]!=.002
            or type(certificate.get("max_candidate_pairs")) is not int or certificate["max_candidate_pairs"]!=2_000_000):
        raise ValueError("Full original human/object cross-triangle and containment certificate failed")


def require_original_coverage(certificate,human_faces=36874,object_faces=384):
    meshes=certificate.get("meshes")
    if type(meshes) is not list or len(meshes)!=2:
        raise ValueError("Both original full surface inventories required")
    for record,nv,nf in zip(meshes,(18439,194),(human_faces,object_faces)):
        if (type(record.get("vertices")) is not int or record["vertices"]!=nv
                or type(record.get("faces")) is not int or record["faces"]!=nf
                or type(record.get("original_face_coverage")) is not int or record["original_face_coverage"]!=nf
                or record.get("topology_closed_oriented") is not True or record.get("embedding_verified") is not True):
            raise ValueError("Actual full original human/bottle topology/coverage required")


class NativeLedger:
    def __init__(self,report,persist):self.report=report;self.persist=persist;self.report["native_calls"]=[]
    def call(self,phase,invoke,validate):
        if len(self.report["native_calls"])>=MAX_FORWARDS:raise RuntimeError("Frozen100 actual native forward budget exhausted")
        row=dict(index=len(self.report["native_calls"])+1,phase=phase,batch_size=1,attempted=True,returned=False,validated=False)
        self.report["native_calls"].append(row);self.persist()
        result=invoke();row["returned"]=True;self.persist();validate(result);row["validated"]=True;self.persist();return result
    def complete(self):
        rows=self.report["native_calls"]
        if not rows or any(not row["returned"] or not row["validated"] for row in rows):raise ValueError("Every attempted actual native call must complete")
        self.report.update(native_attempts=len(rows),native_returns=len(rows),native_validated=len(rows))


def _strict_runtime(torch):
    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8" or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled() or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark or torch.is_inference_mode_enabled()
            or not torch.is_grad_enabled()):raise ValueError("Actual strict differentiable native CUDA required")


def run(root,code,out,report,persist,started):
    bound_code(root,code,report["producer_revision"])
    mount_text=Path("/proc/self/mountinfo").read_text()
    mounts={str(path):exact_mount(mount_text,path,ro) for path,ro in ((code,True),(root/"weights/mhr/mhr_model.pt",True),(out,False))}
    model_path=root/"weights/mhr/mhr_model.pt";source=identity(model_path)
    if source!=dict(sha256=MODEL_SHA,bytes=MODEL_BYTES):raise ValueError("Exact pinned original reference MHR model required")
    helpers={name:identity(code/name,True) for name in HELPERS};report.update(model=source,source_helpers=helpers,exact_mounts=mounts);persist()
    if "torch" in sys.modules:raise ValueError("Fresh CUBLAS-before-Torch private runtime required")
    import torch
    if str(torch.__version__)!="2.5.1+cu124" or torch.version.cuda!="12.4" or not torch.cuda.is_available():raise ValueError("Pinned native CUDA ABI required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    _strict_runtime(torch)
    with torch.jit.optimized_execution(False):model=torch.jit.load(str(model_path),map_location="cuda").float().eval()
    model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):raise ValueError("Native weights must remain frozen")
    names,joints=list(model.get_parameter_names()),list(model.get_joint_names())
    if (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())!=(45,72):raise ValueError("Native coefficient ABI differs")
    schema=model._c._get_method("forward").schema
    if ([(a.name,str(a.type)) for a in schema.arguments[1:]]!=[("identity_coeffs","Tensor"),("model_parameters","Tensor"),
            ("face_expr_coeffs","Tensor"),("apply_correctives","bool")] or str(schema.returns[0].type)!="Tuple[Tensor, Tensor]"):
        raise ValueError("Actual original MHRDemo forward ABI required")
    cpu=lambda value:value.detach().cpu().numpy().copy()
    bounds=cpu(model.get_parameter_limits());faces=cpu(model.character_torch.mesh.faces)
    parents=cpu(model.character_torch.skeleton.joint_parents);transform=cpu(model.get_parameter_transform())
    lbs=model.get_lbsw()
    if not isinstance(lbs,tuple) or len(lbs)!=2:raise ValueError("Original exported LBS index/weight tuple required")
    semantic=metadata(names,joints,parents,transform,bounds,*map(cpu,lbs),faces)
    if (names!=list(model.character_torch.parameter_transform.parameter_names) or joints!=list(model.character_torch.skeleton.joint_names)
            or not np.array_equal(transform,cpu(model.character_torch.parameter_transform.parameter_transform))):
        raise ValueError("Getter metadata differs from the actual original native submodule")
    frozen_metadata=native_metadata_identity(names,joints,parents,transform,bounds,faces,*map(cpu,lbs))
    report.update(metadata=frozen_metadata,active_controls=semantic["control_semantics"],runtime=dict(torch=str(torch.__version__),
        CUDA=torch.version.cuda,deterministic_algorithms=True,warn_only=False,JIT_optimized=False,TF32=False,seed=0,threads=4));persist()
    ledger=NativeLedger(report,persist);flip=torch.tensor([1.,-1.,-1.],device="cuda")
    zeroid=torch.zeros(1,45,device="cuda");zeroexpr=torch.zeros(1,72,device="cuda")
    def native(q,phase):
        _strict_runtime(torch);legal_controls(cpu(q),bounds)
        before=native_input_identity(cpu(q),cpu(zeroid),cpu(zeroexpr))
        fixed=np.ones(204,bool);fixed[semantic["active"]]=False
        if cpu(q)[fixed].tobytes()!=np.zeros(204,np.float32)[fixed].tobytes():raise ValueError("Nonactive original controls changed")
        def invoke():
            with torch.jit.optimized_execution(False):result=model(zeroid,q[None],zeroexpr,True)
            torch.cuda.synchronize();return result
        def validate(result):
            if (not isinstance(result,tuple) or len(result)!=2 or tuple(result[0].shape)!=(1,18439,3)
                    or tuple(result[1].shape)!=(1,127,8) or any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in result)
                    or torch.any(result[1][...,7]<=0) or torch.max(torch.abs(torch.linalg.vector_norm(result[1][...,3:7],dim=-1)-1))>1e-5):
                raise ValueError("Actual full original native geometry/global skeleton ABI differs")
            if before!=native_input_identity(cpu(q),cpu(zeroid),cpu(zeroexpr)):
                raise ValueError("Actual native forward mutated original controls/identity/expression")
            _strict_runtime(torch)
        v,s=ledger.call(phase,invoke,validate);return v[0]*flip/100,s[0]
    q=torch.zeros(204,device="cuda",requires_grad=True);hv,sk=native(q,"fresh_neutral")
    report["phase"]="neutral_full_surface_prerequisite";persist()
    report["neutral_surface"]=audit_closed_surface(cpu(hv),faces,tolerance_m=TOLERANCE,budget_seconds=30.)
    require_closed(report["neutral_surface"])
    if (report["neutral_surface"]["mesh"].get("vertices")!=18439 or report["neutral_surface"]["mesh"].get("original_face_coverage")!=36874):
        raise ValueError("Neutral full native18439/36874 topology inventory differs")
    bv,bf=bottle_mesh();report["bottle_surface"]=audit_closed_surface(bv,bf,tolerance_m=TOLERANCE,budget_seconds=30.)
    require_closed(report["bottle_surface"])
    if (report["bottle_surface"]["mesh"].get("vertices")!=194 or report["bottle_surface"]["mesh"].get("original_face_coverage")!=384):
        raise ValueError("Original own bottle194/384 inventory differs")
    if time.monotonic()-started>75:raise TimeoutError("Neutral original-surface prerequisites exceeded reserved75s")
    skm=cpu(sk);skm[:,:3]=skm[:,:3]*np.array([1.,-1.,-1.])/100
    patches=select_patches(cpu(hv),skm,faces,semantic,bv,bf)
    report["surface_selection"]={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in patches.items()};persist()
    human_index=torch.tensor(faces[patches["human_faces"]],device="cuda");active=semantic["active"]
    hp=hv[human_index].mean(1).reshape(-1)
    jac=torch.stack([torch.autograd.grad(value,q,retain_graph=True)[0] for value in hp])[:,active]
    if not torch.isfinite(jac).all() or float(torch.linalg.vector_norm(jac,dim=0).max())<=1e-5:raise ValueError("Fresh original native fingertip autograd absent/nonfinite/zero")
    col=int(active[int(torch.linalg.vector_norm(jac,dim=0).argmax())]);analytic=cpu(jac[:,list(active).index(col)])
    offsets,mode=five_point_offsets(0.,*bounds[col]);values=[]
    for offset in offsets:
        probe=torch.zeros(204,device="cuda");probe[col]=float(offset)
        pv,_=native(probe,"five_point_native_FD");values.append(cpu(pv[human_index].mean(1).reshape(-1)))
    observed=five_point_derivative(cpu(hp),values,offsets,mode);error=float(np.linalg.norm(observed-analytic))
    if error>5e-5+.02*np.linalg.norm(analytic):raise ValueError("Fresh native five-point finite difference disagrees with autograd")
    report["native_gradient_probe"]=dict(column=col,name=names[col],offsets_rad=offsets.tolist(),stencil=mode,
        analytic_m_per_rad=analytic.tolist(),five_point_m_per_rad=observed.tolist(),error_norm_m_per_rad=error,verified=True)
    del hv,sk,hp,jac,q
    R0=torch.tensor(patches["R"],device="cuda",dtype=torch.float32);t0=torch.tensor(patches["t"],device="cuda",dtype=torch.float32)
    oc,on=centers_normals(bv,bf[patches["object_faces"]]);oc=torch.tensor(oc,device="cuda",dtype=torch.float32)
    on=torch.tensor(on,device="cuda",dtype=torch.float32);aidx=torch.tensor(active,device="cuda")
    x=torch.zeros(len(active)+6,device="cuda",requires_grad=True);best=None;losses=[];report["phase"]="bounded_surface_solve";persist()
    def evaluate(x):
        controls=torch.zeros(204,device="cuda").scatter(0,aidx,x[:len(active)])
        human,skeleton=native(controls,"surface_objective")
        tri=human[human_index];hp=tri.mean(1);hn=torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        hn=hn/torch.linalg.vector_norm(hn,dim=1,keepdim=True)
        r=x[len(active):len(active)+3];z=r[0]*0
        skew=torch.stack((z,-r[2],r[1],r[2],z,-r[0],-r[1],r[0],z)).reshape(3,3)
        R=torch.matrix_exp(skew)@R0;t=t0+x[-3:];op=oc@R.T+t;normal=on@R.T
        residual=torch.cat(((hp-op-GAP*normal).reshape(-1)/.002,(hn+normal).reshape(-1)/.25,
            .02*x[:len(active)]/.5,.01*x[-6:]))
        return residual,controls,human,skeleton,R,t
    for step in range(SEARCH_STATES):
        if time.monotonic()-started>220:raise TimeoutError("Surface search exceeded reserved220s; final proof cannot be skipped")
        residual,controls,human,skeleton,R,t=evaluate(x)
        if not torch.isfinite(residual).all():raise ValueError("Actual differentiable surface residual nonfinite")
        loss=float(residual.square().sum().detach());losses.append(loss)
        if best is None or loss<best[0]:best=(loss,cpu(x),step)
        evidence=contact_evidence(cpu(human),faces,bv@cpu(R).T+cpu(t),bf,patches)
        if evidence["passed"]:break
        if step+1==SEARCH_STATES:break
        J=torch.stack([torch.autograd.grad(value,x,retain_graph=True)[0] for value in residual])
        if not torch.isfinite(J).all():raise ValueError("Actual native/object surface Jacobian nonfinite")
        delta=torch.linalg.solve(J.T@J+.001*torch.eye(len(x),device="cuda"),-J.T@residual)
        lo=np.r_[bounds[active,0],[-math.pi]*3,[-.1]*3];hi=np.r_[bounds[active,1],[math.pi]*3,[.1]*3]
        updated=feasible_step(cpu(x),cpu(delta),lo,hi)
        x=torch.tensor(updated,device="cuda",requires_grad=True);report.update(evaluated_states=step+1,adam_used=False);persist()
    report.update(evaluated_states=len(losses),best_evaluated_state=best[2],objective_losses=losses);persist()
    x=torch.tensor(best[1],device="cuda",requires_grad=True)
    residual,controls,human,skeleton,R,t=evaluate(x)
    evidence=contact_evidence(cpu(human),faces,bv@cpu(R).T+cpu(t),bf,patches);report["distal_surface_proximity"]=evidence
    if not evidence["passed"]:raise ValueError("Original legal thumb/index surfaces did not reach the fixed bottle gap/opposed normals")
    geometry=dict(human_vertices_m=cpu(human),human_faces=faces.astype(np.int64),object_vertices_local_m=bv,object_faces=bf,
        object_R=cpu(R),object_translation_m=cpu(t),object_vertices_m=bv@cpu(R).T+cpu(t),global_skeleton_native=cpu(skeleton))
    report["phase"]="full_cross_surface_certificate";persist()
    report["cross_surface"]=audit_cross_surfaces(geometry["human_vertices_m"],faces,geometry["object_vertices_m"],bf,
        tolerance_m=TOLERANCE,proximity_m=.002,budget_seconds=30.)
    require_cross(report["cross_surface"])
    require_original_coverage(report["cross_surface"])
    saved_controls=dict(model_parameters=cpu(controls),identity=np.zeros(45,np.float32),scales=np.zeros(68,np.float32),
        expression=np.zeros(72,np.float32),human_patch_faces=patches["human_faces"],object_patch_faces=patches["object_faces"])
    for name,arrays in (("controls.npz",saved_controls),("geometry.npz",geometry)):
        with (out/name).open("xb") as stream:np.savez_compressed(stream,**arrays)
        (out/name).chmod(0o400)
    def reread(name):
        with np.load(out/name,allow_pickle=False) as stream:return {key:stream[key].copy() for key in stream.files}
    c,g=reread("controls.npz"),reread("geometry.npz")
    for actual,expected in ((c,saved_controls),(g,geometry)):
        if set(actual)!=set(expected) or any(array_id(actual[k])!=array_id(expected[k]) for k in expected):raise ValueError("Frozen private geometry/control bytes changed")
    legal_controls(c["model_parameters"],bounds)
    replay,_=native(torch.tensor(c["model_parameters"],device="cuda"),"stored_final_native_replay")
    if array_id(cpu(replay))!=array_id(g["human_vertices_m"]):raise ValueError("Frozen original native geometry changed on final replay")
    report["saved_cross_surface"]=audit_cross_surfaces(g["human_vertices_m"],g["human_faces"],g["object_vertices_m"],g["object_faces"],
        tolerance_m=TOLERANCE,proximity_m=.002,budget_seconds=30.)
    require_cross(report["saved_cross_surface"]);require_original_coverage(report["saved_cross_surface"]);ledger.complete()
    if (identity(model_path)!=source or {name:identity(code/name,True) for name in HELPERS}!=helpers
            or not np.array_equal(cpu(model.get_parameter_limits()),bounds) or array_id(cpu(model.character_torch.mesh.faces))!=array_id(faces)
            or list(model.get_parameter_names())!=names or list(model.get_joint_names())!=joints
            or array_id(cpu(model.character_torch.skeleton.joint_parents))!=frozen_metadata["parents_sha256"]
            or array_id(cpu(model.get_parameter_transform()))!=frozen_metadata["transform_sha256"]
            or [array_id(cpu(v)) for v in model.get_lbsw()]!=frozen_metadata["lbs_sha256"]
            or (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())!=(45,72)
            or list(model.character_torch.skeleton.joint_names)!=joints
            or list(model.character_torch.parameter_transform.parameter_names)!=names
            or array_id(cpu(model.character_torch.parameter_transform.parameter_transform))!=frozen_metadata["transform_sha256"]
            or {str(path):exact_mount(Path("/proc/self/mountinfo").read_text(),path,ro)
                for path,ro in ((code,True),(model_path,True),(out,False))}!=mounts):
        raise ValueError("Original source/model/metadata changed")
    report.update(status="pass",phase="complete",private_outputs={name:identity(out/name,True) for name in ("controls.npz","geometry.npz")},
        saved_native_replay_verified=True,source_model_helpers_rehashed=True,one_private_procedural_state_certified=True)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);out=root/OUTPUT;revision=os.environ["WR_CODE_REVISION"]
    bound_code(root,code,revision)
    if (platform.system()!="Linux" or root!=ROOT or os.geteuid()!=1000 or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}
            or os.environ.get("WR_IMAGE_ID")!=IMAGE or not re.fullmatch("[0-9a-f]{40}",revision) or not out.is_dir() or any(out.iterdir())
            or any(p.resolve()!=p or any(q.is_symlink() for q in (p,*p.parents)) for p in (root,code,out))
            or out.stat().st_mode&0o077 or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise ValueError("Fresh private-only source-bound offline Azure capability runtime required")
    if Path(__file__).resolve()!=code/"infra/own_grasp_capability.py":raise ValueError("Actual source-bound entrypoint required")
    report=dict(stage="world_reward_own_native_surface_grasp_capability",status="fail",phase="source_integrity",producer_revision=revision,
        script_sha256=identity(Path(__file__),True)["sha256"],image_id=IMAGE,network="none",budget_seconds=BUDGET,max_native_forwards=MAX_FORWARDS,
        own_fresh_procedural_geometry_only=True,historical_poses_results_read=False,challenge_inputs_used=False,ground_truth_read=False,
        hand_labeled_test=False,RGB_produced=False,public_manifest_produced=False,quality_verified=False,challenge_performance_verified=False,
        force_closure_verified=False,biomechanics_verified=False,touching_certified=False,motion_verified=False,adoption_authorized=False,
        fixed_identity="zero45",fixed_scales="zero68",fixed_expression="zero72",object_scale=1.,target_positive_gap_m=GAP,
        geometry_frame="native_cm_to_metres_proper_diag_1_minus1_minus1",native_calls=[])
    started=time.monotonic();receipt=out/"report.json"
    with receipt.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.monotonic()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Frozen300s private capability deadline")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,code,out,report,persist,started)
        except BaseException as error:
            report.update(status="fail",error_type=type(error).__name__,error=str(error))
            clear_failed_payloads(out)
            raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();receipt.chmod(0o400)


if __name__=="__main__":main()
