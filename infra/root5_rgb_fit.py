"""H98 fixed-depth native XY/Euler5 fit on fresh public RGB observations.

This produces frozen candidates, not quality/adoption decisions. The object
proxy is manufactured once from the unfitted shared baseline. The silhouette
safeguard is computed only after every final candidate is frozen; it never
chooses optimization states or changes the RGB training observations.
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
import root5_rgb_public as public
from world_reward import metric_alignment, root_refit as policy
from world_reward.data import sha256

native=public.native
baseline=public.baseline
BASE=public.protocol.COHORT.base
OUT=BASE+"/root_fit_v2"
STAGE="public_root5_rgb_native_fixed_depth_refit"
BUDGET=300
BOUNDS=np.array((.30,.30,.30/np.sqrt(3.),.30/np.sqrt(3.),.30/np.sqrt(3.)),np.float64)
OPTIMIZER=dict(evaluated_states_per_frame=60,updates_per_frame=59,learning_rate=.01,betas=[.9,.999],eps=1e-8)
REPLAY_OUT=BASE+"/root_replay_v1"
REPLAY_STAGE="public_root5_rgb_native_fixed_depth_replay"
REPLAY_BUDGET=120


def helper_identities():
    return public.helper_identities(fit_source=Path(__file__))



object_sample=public.object_sample
validate_proxy=public.validate_proxy
frozen_proxies=public.frozen_proxies
validate_candidate=public.validate_candidate
frozen_candidates=public.frozen_candidates


def optimization_schedule(evaluate,update):
    losses=[]
    for i in range(60):
        loss=float(evaluate(i))
        if not np.isfinite(loss)or loss<0:raise ValueError("All60 evaluated objectives must be finite/nonnegative")
        losses.append(loss)
        if i<59:update(i)
    return losses,policy.best_evaluated(np.asarray(losses,np.float64))


validate_schedule=public.validate_schedule
validate_trace=public.validate_schedule


def strict_forward(torch):
    if (not torch.are_deterministic_algorithms_enabled()or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark
            or torch.is_inference_mode_enabled()or not torch.is_grad_enabled()or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8"):
        raise ValueError("Strict native differentiable CUDA required; no kernel suppression")
    if max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())>32*1024**3:raise MemoryError("32GiB envelope exceeded")


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
    baseline.rotations(values[4]);return errors


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
    if not np.allclose(j[:,:2],expected,rtol=1e-5,atol=1e-6):raise ValueError("Native XY autograd differs from analytic zero-latent projection")
    return dict(maximum_absolute_error=error,rtol=1e-5,atol=1e-6,latent_is_zero=True,additional_native_calls=0)


def differentiable_native(torch,head,fixed,pair,latent,limits,report,row,persist,kind):
    strict_forward(torch)
    if head.enable_hand_model is not False:raise ValueError("Native head must be full-body only")
    delta=torch.tanh(latent)*torch.tensor(BOUNDS,device="cuda",dtype=torch.float32)
    origin=fixed["pred_cam_t"][0];translation=torch.stack((origin[0]+delta[0],origin[1]+delta[1],origin[2]))
    euler=fixed["global_rot"]+delta[2:][None]
    report["counters"][kind+"_attempted"]+=1;row["native_heads_attempted"]+=1;persist()
    values=head.mhr_forward(global_trans=torch.zeros_like(euler),global_rot=euler,body_pose_params=fixed["body_pose_params"],
        hand_pose_params=fixed["hand_pose_params"],shape_params=fixed["shape_params"],scale_params=fixed["scale_params"],
        expr_params=fixed["expr_params"],return_keypoints=True,return_joint_coords=True,return_model_params=True,return_joint_rotations=True)
    report["counters"][kind+"_returned"]+=1;row["native_heads_returned"]+=1;persist()
    shapes=((1,native.VERTICES,3),(1,308,3),(1,127,3),(1,204),(1,127,3,3))
    if not isinstance(values,tuple)or len(values)!=5 or any(tuple(v.shape)!=s or v.dtype!=torch.float32 or not torch.isfinite(v).all()for v,s in zip(values,shapes)):
        raise ValueError("Complete native float32 V/KP/J/controls/SO3 ABI differs")
    c=values[3][0]
    if (torch.any(c[:3])or not torch.equal(c[3:6],euler[0])or not torch.equal(c[6:],fixed["model_controls"][6:])
            or not torch.equal(translation[2:],origin[2:])or torch.any(euler[0]<limits[:,0])or torch.any(euler[0]>limits[:,1])):
        raise ValueError("Native rootEuler-only bounds/Z/fixed remainder contract violated")
    flip=torch.tensor([1.,-1.,-1.],device="cuda",dtype=torch.float32)
    arrays=tuple(v[0]*flip+translation for v in values[:3])+(c,values[4][0])
    if any(torch.any(v[:,2]<=0)for v in arrays[:3]):raise ValueError("Every camera point requires positiveZ; no clipping")
    R=arrays[4]
    if not torch.allclose(R@R.transpose(-1,-2),torch.eye(3,device="cuda"),atol=1e-4,rtol=0)or not torch.allclose(torch.linalg.det(R),torch.ones(127,device="cuda"),atol=1e-4,rtol=0):
        raise ValueError("Native global joint rotations must be proper SO3")
    strict_forward(torch);report["counters"][kind+"_validated"]+=1;row["native_heads_validated"]+=1;persist()
    return arrays,delta,euler[0],translation


def fit_frame(torch,head,limits,raw,pair,observation,record,report,persist):
    target,indices,validity=policy.training_observations(observation["keypoints"][0],observation["scores"][0])
    fixed={k:torch.tensor(raw[k].copy(),device="cuda",dtype=torch.float32)[None]for k in native.BLOCKS if k!="mhr_model_params"}
    fixed["shape_params"]=torch.tensor(pair["shared_shape_params"].copy(),device="cuda")[None]
    fixed["scale_params"]=torch.tensor(pair["shared_scale_params"].copy(),device="cuda")[None]
    fixed["model_controls"]=torch.tensor(pair["shared_model_controls"].copy(),device="cuda")
    u=torch.zeros(5,device="cuda",dtype=torch.float32,requires_grad=True)
    optimizer=torch.optim.Adam([u],lr=.01,betas=(.9,.999),eps=1e-8)
    K=torch.tensor(pair["camera_K"].astype(np.float32),device="cuda");observed=torch.tensor(target.astype(np.float32),device="cuda")
    selected=torch.tensor(indices,device="cuda");cpu=lambda x:x.detach().cpu().numpy().copy()
    row=dict(file=record["file"],training_validity=validity.tolist(),training_MHR_indices=indices.tolist(),training_points=len(target),
        evaluated_losses=[],evaluated_latents=[],evaluated_projected_points=[],evaluated_physical_deltas=[],update_gradients=[],
        native_heads_attempted=0,native_heads_returned=0,native_heads_validated=0,adam_updates=0,best_evaluated_index=None,observation_compute_dtype="float32",
        heldout_COCO_indices=list(policy.HELDOUT_COCO),heldout_used_for_fit=False)
    report["fit_records"].append(row);persist();state={}
    def evaluate(index):
        arrays,delta,euler,translation=differentiable_native(torch,head,fixed,pair,u,limits,report,row,persist,"objective_native_heads")
        keypoints=arrays[1][selected];projected=keypoints[:,:2]/keypoints[:,2:]*K.diag()[:2]+K[:2,2];residual=projected-observed
        if index==0:
            row["initial_parity"]=parity(tuple(cpu(x)for x in arrays),pair)
            root_contract(cpu(arrays[3]),cpu(euler),cpu(translation),raw,pair,limits.detach().cpu().numpy());persist()
            gradients=[]
            for scalar in residual.reshape(-1):
                report["counters"]["initial_jacobian_rows_attempted"]+=1;persist()
                gradients.append(cpu(torch.autograd.grad(scalar,u,retain_graph=True,create_graph=False)[0]))
                report["counters"]["initial_jacobian_rows_completed"]+=1;persist()
            J=np.stack(gradients);row["initial_observation_jacobian_values"]=J.tolist();row["analytic_XY_jacobian"]=analytic_xy_evidence(J,cpu(keypoints),cpu(K))
            row["initial_observation_jacobian"]=jacobian_evidence(J,len(target));persist()
            if row["initial_observation_jacobian"]["minimum_maximum_ratio"]<1e-5:raise ValueError("Root5 observation Jacobian underidentified; no prior rescue")
        q=torch.linalg.vector_norm(residual,dim=-1)/5.
        data=torch.where(q<=1,.5*q*q,q-.5).mean();loss=data+.5*torch.square(delta/.15).sum();total=float(loss.detach().cpu())
        strict_forward(torch)
        if not np.isfinite(total)or total<0:raise ValueError("Finite native objective required")
        row["evaluated_losses"].append(total);row["evaluated_latents"].append(cpu(u).tolist())
        row["evaluated_projected_points"].append(cpu(projected).tolist());row["evaluated_physical_deltas"].append(cpu(delta).tolist())
        if "best_loss"not in state or total<state["best_loss"]:
            state.update(best_loss=total,best_latent=u.detach().clone(),best_index=index,best_delta=cpu(delta))
        state["loss"]=loss;persist();return total
    def update(_):
        optimizer.zero_grad(set_to_none=True);report["counters"]["optimizer_backwards_attempted"]+=1;persist()
        state["loss"].backward();report["counters"]["optimizer_backwards_returned"]+=1;report["native_backwards"]+=1
        if u.grad is None or not torch.isfinite(u.grad).all():raise ValueError("Finite native gradient required; no kernel relaxation")
        row["update_gradients"].append(cpu(u.grad).tolist());optimizer.step()
        report["counters"]["adam_updates"]+=1;report["optimizer_updates"]+=1;row["adam_updates"]+=1;strict_forward(torch);persist()
    losses,best=optimization_schedule(evaluate,update)
    if best!=state["best_index"]:raise ValueError("First-tie best evaluated state changed")
    arrays,delta,euler,translation=differentiable_native(torch,head,fixed,pair,state["best_latent"],limits,report,row,persist,"final_native_heads")
    if cpu(delta).tobytes()!=state["best_delta"].tobytes():raise ValueError("Selected native state changed on final replay")
    candidate={k:raw[k].copy()for k in native.BLOCKS}
    candidate.update(global_rot=cpu(euler),pred_cam_t=cpu(translation),shape_params=pair["shared_shape_params"].copy(),scale_params=pair["shared_scale_params"].copy(),
        mhr_model_params=cpu(arrays[3]),vertices_camera_m=cpu(arrays[0]),keypoints_camera_m=cpu(arrays[1]),joints_camera_m=cpu(arrays[2]),
        joint_global_rotations=cpu(arrays[4]),human_faces=pair["human_faces"].copy(),hand_mask_left=pair["hand_mask_left"].copy(),hand_mask_right=pair["hand_mask_right"].copy(),
        camera_K=pair["camera_K"].copy(),clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy(),latent=cpu(state["best_latent"]),physical_delta=cpu(delta))
    validate_candidate(candidate,record,raw,pair)
    row.update(best_evaluated_index=best,selected_total_objective=losses[best],selected_physical_delta=cpu(delta).tolist());persist()
    return candidate


def mask_iou(predicted,human,object_mask):
    arrays=[np.asarray(v)for v in(predicted,human,object_mask)]
    if any(np.ma.isMaskedArray(v)for v in(predicted,human,object_mask))or any(v.shape!=(native.HEIGHT,native.WIDTH)or v.dtype!=np.bool_ for v in arrays):
        raise ValueError("Original automatic boolean masks required")
    p,h,o=arrays;region=~o;p=p&region;h=h&region
    if int(h.sum())<32:raise ValueError("Every observed human needs32 nonobject pixels; no frame drop")
    union=int((p|h).sum())
    if union<=0:raise ValueError("Silhouette union invalid")
    return float((p&h).sum()/union)


def silhouette_safeguard(before,after):
    a=np.asarray(before);b=np.asarray(after)
    if any(np.ma.isMaskedArray(v)for v in(before,after))or a.shape!=(15,)or b.shape!=(15,)or any(v.dtype.kind!="f"or not np.isfinite(v).all()or np.any(v<0)or np.any(v>1)for v in(a,b)):
        raise ValueError("Complete15 finite full-grid silhouette IoUs required")
    change=b-a
    return dict(baseline_iou=a.tolist(),candidate_iou=b.tolist(),per_frame_delta=change.tolist(),per_clip_mean_delta=change.reshape(3,5).mean(1).tolist(),
        worst_frame_delta=float(change.min()),worst_clip_mean_delta=float(change.reshape(3,5).mean(1).min()),limit=-.01,
        safeguard_pass=bool(np.all(change>=-.01)and np.all(change.reshape(3,5).mean(1)>=-.01)),used_for_state_selection=False,
        candidate_export_blocked_by_safeguard=False,accuracy_verified=False)


def perform(root,out,report,persist,pins):
    records,raw,pairs,dw,lineage=public.public_predictions(root,pins);helpers=helper_identities()
    report.update(lineage=lineage,source_helpers=helpers,phase="native_model_load");persist()
    if "torch"in sys.modules:raise ValueError("CUBLAS setup precedes Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    import pytorch3d
    if not torch.cuda.is_available()or str(torch.__version__)!="2.5.1+cu124"or torch.version.cuda!="12.4"or pytorch3d.__version__!="0.7.9":raise ValueError("Exact pinned native CUDA/renderer runtime required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats();model,estimator,faces,body_source=native.human.load_model(root,torch);model.eval();model.requires_grad_(False)
    head=model.head_pose
    if (head.enable_hand_model is not False or any(p.requires_grad for p in model.parameters())or not np.array_equal(faces,pairs[0]["human_faces"])
            or tuple(head.scale_mean.shape)!=(68,)or tuple(head.scale_comps.shape)!=(28,68)):raise ValueError("Frozen full-body native head/rig required")
    frozen=json.loads((root/BASE/public.observe.FOLDERS["baseline"]/"report.json").read_text())
    if body_source!=frozen["body_model"]:raise ValueError("Native source/checkpoint differs from baseline")
    bounds=head.mhr.get_parameter_limits().detach().clone()
    if tuple(bounds.shape)!=(249,2)or torch.isnan(bounds).any()or torch.any(bounds[:,0]>bounds[:,1]):raise ValueError("Actual native root-only metadata required")
    limits=bounds[3:6].clone();source=Path(sys.modules[type(head).__module__].__file__);head_source=native.regular(source)
    report.update(native_head_source=head_source,body_model=body_source,torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,
        pytorch3d_version=pytorch3d.__version__,seed=0,TF32=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",deterministic_algorithms=True,warn_only=False,threads=4,
        root_bounds=limits.detach().cpu().numpy().tolist(),phase="common_object_proxy");persist()
    del model,estimator;torch.cuda.empty_cache();rendered=[];before_iou=[]
    for record,pair,original in zip(records,pairs,raw):
        strict_forward(torch);report["counters"]["proxy_rasters_attempted"]+=1;persist()
        mask,depth=native.raster_camera_mesh(pair["shared_vertices_camera_m"],pair["human_faces"],pair["camera_K"],native.WIDTH,native.HEIGHT)
        torch.cuda.synchronize();strict_forward(torch);report["counters"]["proxy_rasters_completed"]+=1
        m,d=mask.cpu().numpy(),depth.cpu().numpy();rendered.append((m,d));before_iou.append(mask_iou(m,original["human_mask"],original["object_mask"]));persist()
    for clip in range(3):
        indices=range(clip*5,clip*5+5);visibility=[rendered[i][0]&raw[i]["human_mask"]&~raw[i]["object_mask"]&raw[i]["validity"]for i in indices]
        alignment=metric_alignment.fit_shared_depth_scale([raw[i]["raw_depth"]for i in indices],[rendered[i][1]for i in indices],visibility,
            list(range(5)),min_correspondences_per_frame=32,min_supported_frames=5)
        if alignment.supported_frames!=5:raise ValueError("Every shared-human frame must support the common proxy scale")
        report["proxy_alignments"].append(alignment.to_dict())
        for i in indices:
            points,pixels=object_sample(raw[i],alignment.shared_scale)
            proxy=dict(object_points_camera_m=points,pixel_indices=pixels,shared_depth_scale=np.asarray(alignment.shared_scale,np.float64),
                clip_index=raw[i]["clip_index"].copy(),frame_index=raw[i]["frame_index"].copy())
            validate_proxy(proxy,records[i],raw[i]);report["proxy_outputs"].append(native.save(out,"proxies",records[i],proxy))
    frozen_proxies(root,records,report["proxy_outputs"],raw,pairs)
    report.update(object_proxies_frozen_before_fit=True,baseline_silhouette_iou=before_iou,phase="native_root5_optimization");persist()
    for record,original,pair,observation in zip(records,raw,pairs,dw):
        report.update(active_file=record["file"]);persist();candidate=fit_frame(torch,head,limits,original,pair,observation,record,report,persist)
        report["candidate_outputs"].append(native.save(out,"candidates",record,candidate));persist()
    candidates=frozen_candidates(root,records,report["candidate_outputs"],raw,pairs)
    report.update(all_candidates_frozen=True,native_arrays_verified=True,phase="final_silhouette_safeguard");persist();after_iou=[]
    for record,original,candidate in zip(records,raw,candidates):
        strict_forward(torch);report["counters"]["safeguard_rasters_attempted"]+=1;persist()
        mask,depth=native.raster_camera_mesh(candidate["vertices_camera_m"],candidate["human_faces"],candidate["camera_K"],native.WIDTH,native.HEIGHT)
        torch.cuda.synchronize();strict_forward(torch);report["counters"]["safeguard_rasters_completed"]+=1
        after_iou.append(mask_iou(mask.cpu().numpy(),original["human_mask"],original["object_mask"]));persist()
    report["silhouette_safeguard"]=silhouette_safeguard(np.asarray(before_iou),np.asarray(after_iou));persist()
    frozen_candidates(root,records,report["candidate_outputs"],raw,pairs);frozen_proxies(root,records,report["proxy_outputs"],raw,pairs)
    if public.public_predictions(root,pins)[-1]!=lineage or helper_identities()!=helpers or native.regular(source)!=head_source:raise ValueError("Frozen public/source/model evidence changed")
    expected=dict(proxy_rasters_attempted=15,proxy_rasters_completed=15,safeguard_rasters_attempted=15,safeguard_rasters_completed=15,
        objective_native_heads_attempted=900,objective_native_heads_returned=900,objective_native_heads_validated=900,
        final_native_heads_attempted=15,final_native_heads_returned=15,final_native_heads_validated=15,
        optimizer_backwards_attempted=885,optimizer_backwards_returned=885,adam_updates=885)
    if any(report["counters"][k]!=v for k,v in expected.items()):raise ValueError("Exact15-frame60/59/915native/30raster protocol required")
    if {p.name for p in out.iterdir()}!={"proxies","candidates","report.json"}:raise ValueError("Exact frozen output inventory required")
    report.update(status="pass",phase="complete",frames=15,inputs_unchanged=True,all_cases_retained=True,
        source_inputs_assets_rehashed=True,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved());report.pop("active_file",None)


def replay_perform(root,report,persist,pins,fit_pins):
    if (not isinstance(fit_pins,dict)or set(fit_pins)!={"sha256","producer_revision"}
            or not re.fullmatch("[0-9a-f]{64}",str(fit_pins.get("sha256","")))
            or not re.fullmatch("[0-9a-f]{40}",str(fit_pins.get("producer_revision","")))):
        raise ValueError("Explicit frozen completed fit report pin required")
    records,raw,pairs,dw,lineage=public.public_predictions(root,pins);path=root/OUT/"report.json"
    receipt=native.regular(path,fit_pins["sha256"],immutable=True);producer=json.loads(path.read_text())
    required=dict(stage=STAGE,status="pass",phase="complete",producer_revision=fit_pins["producer_revision"],image_id=native.IMAGE_ID,
        script_sha256=sha256(Path(__file__)),source_helpers=helper_identities(),network="none",device="cuda",budget_seconds=300,
        optimizer=OPTIMIZER,lineage=lineage,private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,
        oracle_modes=[],quality_verified=False,accuracy_verified=False,adoption_authorized=False,
        previous_failed_fit=public.previous_fit_failure(root),previous_failure_rewritten=False,all_candidates_frozen=True,
        native_arrays_verified=True,object_proxies_frozen_before_fit=True,inputs_unchanged=True,all_cases_retained=True,root_bounds_only=True,
        dense_fixed_remainder_bounds_checked=False,camera_Z_fixed=True,intrinsics_fixed=True,shared_identity_constant=True,body_hands_fixed=True)
    if any(type(producer.get(k))is not type(v)or producer[k]!=v for k,v in required.items()):raise ValueError("Exact completed source-bound root5 fit required before replay")
    expected=dict(proxy_rasters_attempted=15,proxy_rasters_completed=15,safeguard_rasters_attempted=15,safeguard_rasters_completed=15,
        objective_native_heads_attempted=900,objective_native_heads_returned=900,objective_native_heads_validated=900,
        final_native_heads_attempted=15,final_native_heads_returned=15,final_native_heads_validated=15,
        optimizer_backwards_attempted=885,optimizer_backwards_returned=885,adam_updates=885)
    if any(producer.get("counters",{}).get(k)!=v or type(producer["counters"][k])is not int for k,v in expected.items()):raise ValueError("Every preregistered native/update/raster call required")
    candidates=frozen_candidates(root,records,producer["candidate_outputs"],raw,pairs);frozen_proxies(root,records,producer["proxy_outputs"],raw,pairs)
    rows=producer.get("fit_records")
    if not isinstance(rows,list)or len(rows)!=15:raise ValueError("All15 ordered fit traces required")
    for record,observation,candidate,row in zip(records,dw,candidates,rows):
        if row.get("file")!=record["file"]:raise ValueError("Original complete trace order required")
        report["trace_checks"].append(validate_trace(row,observation,candidate));persist()
    report.update(fit_report=receipt,lineage=lineage,source_helpers=helper_identities(),all_training_traces_verified=True,phase="native_best_replay");persist()
    if "torch"in sys.modules:raise ValueError("Replay CUBLAS setup precedes Torch")
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    if not torch.cuda.is_available()or str(torch.__version__)!="2.5.1+cu124"or torch.version.cuda!="12.4":raise ValueError("Exact native replay CUDA runtime required")
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats();model,estimator,faces,body_source=native.human.load_model(root,torch);model.eval();model.requires_grad_(False);head=model.head_pose
    if (body_source!=producer["body_model"]or head.enable_hand_model is not False or any(p.requires_grad for p in model.parameters())
            or not np.array_equal(faces,pairs[0]["human_faces"])):raise ValueError("Frozen replay model/native topology differs")
    bounds=head.mhr.get_parameter_limits().detach().clone()
    if tuple(bounds.shape)!=(249,2)or torch.isnan(bounds).any()or torch.any(bounds[:,0]>bounds[:,1]):raise ValueError("Actual root bounds replay metadata required")
    limits=bounds[3:6];head_path=Path(sys.modules[type(head).__module__].__file__);head_source=native.regular(head_path)
    if head_source!=producer["native_head_source"]:raise ValueError("Actual native head source differs from fit")
    report.update(body_model=body_source,native_head_source=head_source,torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,
                  deterministic_algorithms=True,warn_only=False,TF32=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",seed=0,threads=4);persist()
    for record,original,pair,candidate in zip(records,raw,pairs,candidates):
        row=dict(file=record["file"],native_heads_attempted=0,native_heads_returned=0,native_heads_validated=0);report["records"].append(row);persist()
        fixed={k:torch.tensor(original[k].copy(),device="cuda",dtype=torch.float32)[None]for k in native.BLOCKS if k!="mhr_model_params"}
        fixed["shape_params"]=torch.tensor(pair["shared_shape_params"].copy(),device="cuda")[None]
        fixed["scale_params"]=torch.tensor(pair["shared_scale_params"].copy(),device="cuda")[None]
        fixed["model_controls"]=torch.tensor(pair["shared_model_controls"].copy(),device="cuda")
        latent=torch.tensor(candidate["latent"].copy(),device="cuda",dtype=torch.float32,requires_grad=True)
        values,delta,euler,translation=differentiable_native(torch,head,fixed,pair,latent,limits,report,row,persist,"final_native_heads")
        cpu=lambda x:x.detach().cpu().numpy().copy();arrays=tuple(cpu(v)for v in values)
        comparison=dict(shared_vertices_camera_m=candidate["vertices_camera_m"],shared_keypoints_camera_m=candidate["keypoints_camera_m"],
            shared_joints_camera_m=candidate["joints_camera_m"],shared_model_controls=candidate["mhr_model_params"],shared_joint_global_rotations=candidate["joint_global_rotations"])
        row["replay_parity"]=parity(arrays,comparison)
        root_contract(arrays[3],cpu(euler),cpu(translation),original,pair,limits.cpu().numpy())
        if (cpu(delta).tobytes()!=candidate["physical_delta"].tobytes()or cpu(euler).tobytes()!=candidate["global_rot"].tobytes()
                or cpu(translation).tobytes()!=candidate["pred_cam_t"].tobytes()or arrays[3].tobytes()!=candidate["mhr_model_params"].tobytes()):
            raise ValueError("Frozen selected controls/Euler/translation/delta changed on replay")
        persist();del values,arrays,latent,delta,euler,translation
    if any(report["counters"][k]!=15 for k in("final_native_heads_attempted","final_native_heads_returned","final_native_heads_validated")):
        raise ValueError("Exactly15 selected-state native replays required")
    frozen_candidates(root,records,producer["candidate_outputs"],raw,pairs);frozen_proxies(root,records,producer["proxy_outputs"],raw,pairs)
    if (native.regular(path,fit_pins["sha256"],immutable=True)!=receipt or public.public_predictions(root,pins)[-1]!=lineage
            or helper_identities()!=report["source_helpers"]or native.regular(head_path)!=head_source):raise ValueError("Replay input/model/source bytes changed")
    report.update(status="pass",phase="complete",all_cases_retained=True,all_native_best_replays_verified=True,predictions_native_replay_verified=True,source_inputs_assets_rehashed=True,
        optimizer_updates=0,native_backwards=0,additional_training_forwards=0,frames=15,max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),max_gpu_reserved_bytes=torch.cuda.max_memory_reserved())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument("operation",choices=("fit","replay"))
    parser.add_argument("--public-pins",type=Path,required=True);parser.add_argument("--fit-pins",type=Path);args=parser.parse_args(argv)
    operation=args.operation;budget=BUDGET if operation=="fit"else REPLAY_BUDGET
    if(operation=="replay")!=(args.fit_pins is not None):parser.error("--fit-pins is required only for replay")
    code=Path(os.environ.get("WR_CODE",""))
    if not code.is_absolute()or code.resolve()!=code.absolute()or args.public_pins!=code/"configs/root5_rgb_public_pins_v1.json":
        raise ValueError("Public receipt pins must be the immutable source-bundle configuration")
    if args.fit_pins is not None and args.fit_pins!=code/"configs/root5_rgb_fit_pins_v1.json":raise ValueError("Fit receipt pin must be the immutable source-bundle configuration")
    pin_identity=native.regular(args.public_pins,immutable=True);pins=json.loads(args.public_pins.read_text());public.validate_pins(pins)
    fit_pins=None;fit_pin_identity=None
    if args.fit_pins is not None:fit_pin_identity=native.regular(args.fit_pins,immutable=True);fit_pins=json.loads(args.fit_pins.read_text())
    root=Path(os.environ.get("WR_ROOT",""));out=root/(OUT if operation=="fit"else REPLAY_OUT);revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or out.resolve()!=out.absolute()or os.geteuid()!=1000
            or{p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.environ.get("WR_IMAGE_ID")!=native.IMAGE_ID
            or not re.fullmatch("[0-9a-f]{40}",revision)or not out.is_dir()or any(out.iterdir())or any(p.is_symlink()for p in(out,*out.parents))):
        raise ValueError("Fresh reserved offline root5 fitting output required")
    if operation=="fit":(out/"proxies").mkdir();(out/"candidates").mkdir()
    started=time.perf_counter();path=out/"report.json"
    keys=("proxy_rasters_attempted","proxy_rasters_completed","safeguard_rasters_attempted","safeguard_rasters_completed",
        "objective_native_heads_attempted","objective_native_heads_returned","objective_native_heads_validated",
        "final_native_heads_attempted","final_native_heads_returned","final_native_heads_validated","initial_jacobian_rows_attempted",
        "initial_jacobian_rows_completed","optimizer_backwards_attempted","optimizer_backwards_returned","adam_updates")
    report=dict(stage=STAGE if operation=="fit"else REPLAY_STAGE,status="fail",phase="public_integrity",operation=operation,producer_revision=revision,image_id=native.IMAGE_ID,script_sha256=sha256(Path(__file__)),
        network="none",device="cuda",budget_seconds=budget,optimizer=OPTIMIZER if operation=="fit"else None,training_COCO_indices=list(policy.TRAIN_COCO),
        training_MHR_indices=list(policy.TRAIN_MHR),heldout_COCO_indices=list(policy.HELDOUT_COCO),initial_jacobian_includes_priors=False,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],quality_verified=False,
        accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,shared_identity_constant=True,body_hands_fixed=True,camera_Z_fixed=True,
        intrinsics_fixed=True,root_bounds_only=True,dense_fixed_remainder_bounds_checked=False,numerical_reproducibility_verified=False,
        object_proxies_frozen_before_fit=False,all_candidates_frozen=False,native_arrays_verified=False,inputs_unchanged=False,all_cases_retained=False,
        public_pins_file=pin_identity,fit_pins_file=fit_pin_identity,candidate_outputs=[],proxy_outputs=[],proxy_alignments=[],fit_records=[],records=[],trace_checks=[],
        all_training_traces_verified=False,all_native_best_replays_verified=False,predictions_native_replay_verified=False,
        native_backwards=0,optimizer_updates=0,source_inputs_assets_rehashed=False,counters={k:0 for k in keys})
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n")
            stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole public native root5 operation exceeded declared budget")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(budget)
        try:
            persist()
            previous_failure=public.previous_fit_failure(root)
            report.update(previous_failed_fit=previous_failure,previous_failure_rewritten=False);persist()
            if operation=="fit":perform(root,out,report,persist,pins)
            else:replay_perform(root,report,persist,pins,fit_pins)
            if public.previous_fit_failure(root)!=previous_failure:raise ValueError("Original zero-call bootstrap failure changed")
            if native.regular(args.public_pins,immutable=True)!=pin_identity or(args.fit_pins is not None and native.regular(args.fit_pins,immutable=True)!=fit_pin_identity):
                raise ValueError("Immutable explicit source/prediction pin configuration changed")
        except Exception as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
