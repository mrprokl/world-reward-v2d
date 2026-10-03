"""Native XY/Euler5 autograd capability on already-scored public RGB evidence.

No optimizer, ranking, new predictions, proxies or private truth. Fixed remainder
controls are retained exactly; dense remainder bounds are not a new ABI claim.
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
import keypoint_rgb_public as public
from world_reward.data import sha256

native=public.native
OUT="results/root5-native-capability-v1"
STAGE="public_native_root5_autograd_capability"
BUDGET=120
BOUNDS=(.30,.30,.30/np.sqrt(3.),.30/np.sqrt(3.),.30/np.sqrt(3.))


def root_contract(controls,euler,translation,raw,pair,limits):
    c=native.array(controls,(204,),"float32");r=native.array(euler,(3,),"float32");t=native.array(translation,(3,),"float32")
    bounds=np.asarray(limits)
    if (np.ma.isMaskedArray(limits)or bounds.shape!=(3,2)or bounds.dtype.kind!="f"or np.isnan(bounds).any()or np.any(bounds[:,0]>bounds[:,1])):
        raise ValueError("Actual rootEuler-only bounds required")
    if (np.any(c[:3])or c[3:6].tobytes()!=r.tobytes()or c[6:].tobytes()!=pair["shared_model_controls"][6:].tobytes()
            or t[2:].tobytes()!=raw["pred_cam_t"][2:].tobytes()or np.any(r<bounds[:,0])or np.any(r>bounds[:,1])):
        raise ValueError("Only bounded nativeEuler/XY may differ; Z/remainder controls must be byte-fixed")


def parity(values,pair):
    if len(values)!=5:raise ValueError("Complete native V/KP/J/controls/rotation tuple required")
    errors={name:float(np.linalg.norm(a.astype(np.float64)-pair[key].astype(np.float64),axis=-1).max())for name,a,key in zip(
        ("vertices_m","keypoints_m","joints_m"),values[:3],("shared_vertices_camera_m","shared_keypoints_camera_m","shared_joints_camera_m"))}
    errors["controls"]=float(np.abs(values[3].astype(np.float64)-pair["shared_model_controls"]).max())
    errors["rotations"]=float(np.abs(values[4].astype(np.float64)-pair["shared_joint_global_rotations"]).max())
    if any(not np.isfinite(e)or e>1e-5 for e in errors.values()):raise ValueError("Initial shared native parity exceeds1e-5")
    public.baseline.rotations(values[4]);return errors


def jacobian_evidence(jacobian,points):
    j=np.asarray(jacobian)
    if (np.ma.isMaskedArray(jacobian)or type(points)is not int or not 6<=points<=10 or j.shape!=(points*2,5)
            or j.dtype.kind!="f"or not np.isfinite(j).all()):raise ValueError("Observation-only finite2Nby5 Jacobian required")
    singular=np.linalg.svd(j.astype(np.float64),compute_uv=False);ratio=float(singular[-1]/singular[0])if singular[0]>0 else 0.
    evidence=dict(singular_values=singular.tolist(),minimum_maximum_ratio=ratio,observation_rows=points*2,columns=5,prior_rows_included=False)
    return evidence


def analytic_xy_evidence(jacobian,points,camera_K):
    p=np.asarray(points);K=np.asarray(camera_K);j=np.asarray(jacobian)
    if (any(np.ma.isMaskedArray(v)for v in(jacobian,points,camera_K))or p.ndim!=2 or p.shape[1]!=3
            or not 6<=len(p)<=10 or j.shape!=(2*len(p),5)or K.shape!=(3,3)
            or any(v.dtype.kind!="f"or not np.isfinite(v).all()for v in(p,K,j))or np.any(p[:,2]<=0)
            or K[0,0]<=0 or K[1,1]<=0 or K[0,1]!=0 or K[1,0]!=0
            or not np.array_equal(K[2],np.array([0.,0.,1.]))):
        raise ValueError("Finite pinhole points/K and root5 Jacobian required")
    expected=np.zeros((2*len(p),2),np.float64)
    expected[::2,0]=BOUNDS[0]*float(K[0,0])/p[:,2].astype(np.float64)
    expected[1::2,1]=BOUNDS[1]*float(K[1,1])/p[:,2].astype(np.float64)
    error=float(np.abs(j[:,:2]-expected).max())
    if not np.allclose(j[:,:2],expected,rtol=1e-5,atol=1e-6):
        raise ValueError("Native autograd XY differs from analytic zero-latent pinhole Jacobian")
    return dict(maximum_absolute_error=error,rtol=1e-5,atol=1e-6,latent_is_zero=True,additional_native_calls=0)


def strict(torch):
    if (not torch.are_deterministic_algorithms_enabled()or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark
            or torch.is_inference_mode_enabled()or not torch.is_grad_enabled()or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise ValueError("Strict differentiable native CUDA state required; no kernel relaxation")
    if max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>32*1024**3:raise MemoryError("32GiB envelope exceeded")


def native_forward(torch,head,raw,pair,report,persist):
    strict(torch)
    if head.enable_hand_model is not False:raise ValueError("Exact full-body head requires enable_hand_model=False")
    tensor=lambda value:torch.tensor(value.copy(),device="cuda",dtype=torch.float32)[None]
    latent=torch.zeros(5,device="cuda",dtype=torch.float32,requires_grad=True)
    delta=torch.tanh(latent)*torch.tensor(BOUNDS,device="cuda",dtype=torch.float32)
    euler=tensor(raw["global_rot"])+delta[2:][None]
    origin=tensor(raw["pred_cam_t"])[0]
    translation=torch.stack((origin[0]+delta[0],origin[1]+delta[1],origin[2]))
    report["attempted_native_heads"]+=1;persist()
    values=head.mhr_forward(global_trans=torch.zeros_like(euler),global_rot=euler,
        body_pose_params=tensor(raw["body_pose_params"]),hand_pose_params=tensor(raw["hand_pose_params"]),
        shape_params=tensor(pair["shared_shape_params"]),scale_params=tensor(pair["shared_scale_params"]),expr_params=tensor(raw["expr_params"]),
        return_keypoints=True,return_joint_coords=True,return_model_params=True,return_joint_rotations=True)
    report["returned_native_heads"]+=1;persist()
    expected=((1,native.VERTICES,3),(1,308,3),(1,127,3),(1,204),(1,127,3,3))
    if not isinstance(values,tuple) or len(values)!=5 or any(tuple(v.shape)!=s or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v,s in zip(values,expected)):
        raise ValueError("Actual complete native float32 ABI differs")
    flip=torch.tensor([1.,-1.,-1.],device="cuda",dtype=torch.float32)
    camera=tuple(v[0]*flip+translation for v in values[:3])+(values[3][0],values[4][0])
    if any(torch.any(v[:,2]<=0)for v in camera[:3]):raise ValueError("Every camera point requires positiveZ; no clipping")
    strict(torch);return camera,latent,euler[0],translation


def perform(root,report,persist):
    records,raw,pairs,dw,lineage=public.public_predictions(root)
    helpers=public.baseline.helper_identities()|dict(public=sha256(Path(public.__file__)),capability=sha256(Path(__file__)))
    report.update(lineage=lineage,source_helpers=helpers,phase="native_model_load");persist()
    if"torch"in sys.modules:raise ValueError("CUBLAS environment must precede Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    if not torch.cuda.is_available()or str(torch.__version__)!="2.5.1+cu124"or torch.version.cuda!="12.4":raise ValueError("Exact pinned native CUDA runtime required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats()
    model,estimator,faces,body_source=native.human.load_model(root,torch);model.eval();model.requires_grad_(False);head=model.head_pose
    if (any(p.requires_grad for p in model.parameters())or head.enable_hand_model is not False
            or not np.array_equal(faces,pairs[0]["human_faces"])or tuple(head.scale_mean.shape)!=(68,)or tuple(head.scale_comps.shape)!=(28,68)):
        raise ValueError("Frozen full-body native head/scale/topology required")
    if body_source!=json.loads((root/public.BASE/"baseline_v1/report.json").read_text())["body_model"]:
        raise ValueError("Native source/checkpoint changed from frozen baseline")
    limits=head.mhr.get_parameter_limits().detach().cpu().numpy()
    if limits.shape!=(249,2):raise ValueError("Actual native249 root metadata required")
    root_limits=limits[3:6].copy();head_path=Path(sys.modules[type(head).__module__].__file__);head_source=native.regular(head_path)
    report.update(body_model=body_source,native_head_source=head_source,torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,
        CUBLAS_WORKSPACE_CONFIG=":4096:8",seed=0,threads=4,TF32=False,deterministic_algorithms=True,warn_only=False,
        root_Euler_bounds=root_limits.tolist(),phase="native_parity_and_first_observation_jacobian");persist()
    for i,(record,original,pair,observation)in enumerate(zip(records,raw,pairs,dw)):
        report.update(active_file=record["file"]);persist()
        if np.any(original["expr_params"]):raise ValueError("Expression remains exactzero")
        values,latent,euler,translation=native_forward(torch,head,original,pair,report,persist)
        cpu=lambda v:v.detach().cpu().numpy().copy()
        arrays=tuple(cpu(v)for v in values);root_contract(arrays[3],cpu(euler),cpu(translation),original,pair,root_limits)
        row=dict(file=record["file"],initial_parity=parity(arrays,pair),Z_byte_unchanged=True,fixed_remainder_byte_unchanged=True)
        report["validated_native_heads"]+=1;report["records"].append(row);persist()
        if i==0:
            observed,indices,valid=public.policy.training_observations(observation["keypoints"][0],observation["scores"][0])
            selected=values[1][torch.tensor(indices,device="cuda")];K=torch.tensor(pair["camera_K"].astype(np.float32),device="cuda")
            residual=(selected[:,:2]/selected[:,2:]*K.diag()[:2]+K[:2,2])-torch.tensor(observed.astype(np.float32),device="cuda")
            gradients=[]
            for scalar in residual.reshape(-1):
                report["attempted_autograd_rows"]+=1;persist()
                gradient=torch.autograd.grad(scalar,latent,retain_graph=True,create_graph=False)[0]
                gradients.append(cpu(gradient));report["completed_autograd_rows"]+=1;persist()
            jacobian=np.stack(gradients)
            report["analytic_XY_jacobian"]=analytic_xy_evidence(jacobian,cpu(selected),cpu(K))
            evidence=jacobian_evidence(jacobian,len(observed));report["observation_jacobian"]=evidence;persist()
            if evidence["minimum_maximum_ratio"]<1e-5:raise ValueError("Native root5 observation Jacobian underidentified; no prior rank rescue")
            strict(torch)
        del values,latent,euler,translation,arrays
    final=public.public_predictions(root)[-1]
    if (final!=lineage or native.regular(head_path)!=head_source
            or public.baseline.helper_identities()|dict(public=sha256(Path(public.__file__)),capability=sha256(Path(__file__)))!=helpers):
        raise ValueError("Frozen source/model/input evidence changed")
    if report["attempted_native_heads"]!=15 or report["returned_native_heads"]!=15 or report["validated_native_heads"]!=15:
        raise ValueError("All15 exact native baseline heads required")
    report.update(status="pass",phase="complete",frames=15,all_cases_retained=True,source_inputs_assets_rehashed=True,
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    report.pop("active_file",None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));out=root/OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or out.resolve()!=out.absolute()or os.geteuid()!=1000
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.environ.get("WR_IMAGE_ID")!=native.IMAGE_ID
            or not re.fullmatch("[0-9a-f]{40}",revision)or not out.is_dir()or any(out.iterdir())):
        raise ValueError("Fresh reserved offline native capability output required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=native.IMAGE_ID,network="none",device="cuda",
        script_sha256=sha256(Path(__file__)),budget_seconds=BUDGET,attempted_native_heads=0,returned_native_heads=0,validated_native_heads=0,
        attempted_autograd_rows=0,completed_autograd_rows=0,optimizer_updates=0,final_candidates=0,records=[],
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        predictions_modified=False,quality_verified=False,accuracy_verified=False,adoption_authorized=False,
        old_scored_public_cohort_reused=True,dense_fixed_remainder_bounds_checked=False,root_bounds_only=True,
        numerical_reproducibility_verified=False,source_inputs_assets_rehashed=False,all_cases_retained=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole native root5 capability exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();perform(root,report,persist)
        except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
