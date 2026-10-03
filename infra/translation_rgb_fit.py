"""H97 CPU external-camera translation fit; frozen native body/identity retained.

No native/model forwards, root rotations, shape changes or object resampling.
This public RGB consistency experiment does not establish private 3D accuracy.
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
import translation_rgb_public as public
from world_reward import translation_refit as policy
from world_reward.data import sha256

native=public.native
BUDGET=60


def fit_frame(raw,pair,observation,record,report,persist):
    observed,indices,valid=policy.training_observations(observation["keypoints"][0],observation["scores"][0])
    points=pair["shared_keypoints_camera_m"].astype(np.float64)[indices]
    u=np.zeros(3,np.float64);first=np.zeros(3,np.float64);second=np.zeros(3,np.float64)
    row=dict(file=record["file"],training_points=len(observed),training_MHR_indices=indices.tolist(),training_validity=valid.tolist(),
        evaluated_losses=[],evaluated_latents=[],cpu_evaluations=0,adam_updates=0,heldout_COCO_indices=list(policy.HELDOUT_COCO),heldout_used_for_fit=False)
    report["fit_records"].append(row);persist();best_loss=None;best_u=None
    for step in range(60):
        projected,jacobian=policy.project_and_jacobian(points,raw["pred_cam_t"],u,pair["camera_K"])
        if step==0:
            row["initial_observation_jacobian"]=policy.jacobian_evidence(jacobian,len(observed))
            report["counters"]["initial_jacobian_rows"]+=2*len(observed)
        result,gradient=policy.objective_and_gradient(projected,observed,policy.bounded_delta(u),jacobian,u)
        loss=result["total"]
        if not np.isfinite(loss)or loss<0 or gradient.shape!=(3,)or not np.isfinite(gradient).all():raise ValueError("Finite evaluated translation loss/gradient required")
        row["evaluated_losses"].append(float(loss));row["evaluated_latents"].append(u.tolist());row["cpu_evaluations"]+=1;report["counters"]["evaluated_states"]+=1
        if best_loss is None or loss<best_loss:best_loss=float(loss);best_u=u.copy();best_index=step
        if step<59:
            u,first,second=policy.adam_step(u,gradient,first,second,step+1)
            row["adam_updates"]+=1;report["counters"]["adam_updates"]+=1
        persist()
    best=policy.best_evaluated(np.asarray(row["evaluated_losses"],np.float64))
    if best!=best_index:raise ValueError("First-tie evaluated-state selection changed")
    delta=policy.bounded_delta(best_u);translation=policy.camera_translation(raw["pred_cam_t"],delta)
    vertices,keypoints,joints=public.translated_geometry(pair,raw["pred_cam_t"],translation)
    candidate={k:raw[k].copy()for k in native.BLOCKS}
    candidate.update(shape_params=pair["shared_shape_params"].copy(),scale_params=pair["shared_scale_params"].copy(),
        mhr_model_params=pair["shared_model_controls"].copy(),pred_cam_t=translation.astype(np.float32),
        vertices_camera_m=vertices,keypoints_camera_m=keypoints,joints_camera_m=joints,
        joint_global_rotations=pair["shared_joint_global_rotations"].copy(),human_faces=pair["human_faces"].copy(),
        camera_K=pair["camera_K"].copy(),hand_mask_left=pair["hand_mask_left"].copy(),hand_mask_right=pair["hand_mask_right"].copy(),
        clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy(),latent=best_u.copy(),physical_delta=delta)
    public.validate_candidate(candidate,record,raw,pair)
    row.update(best_evaluated_index=best,selected_total_objective=best_loss,selected_physical_delta=delta.tolist())
    persist();return candidate


def perform(root,out,report,persist):
    if"torch"in sys.modules:raise ValueError("Translation-only CPU trial must not import Torch")
    records,raw,pairs,dw,proxies,lineage=public.public_predictions(root);helpers=public.helper_identities()
    report.update(lineage=lineage,source_helpers=helpers,phase="external_translation_optimization");persist()
    for record,original,pair,observation in zip(records,raw,pairs,dw):
        report.update(active_file=record["file"]);persist()
        data=fit_frame(original,pair,observation,record,report,persist)
        report["candidate_outputs"].append(native.save(out,"candidates",record,data));persist()
    public.frozen_candidates(root,records,report["candidate_outputs"],raw,pairs)
    report.update(all_candidates_frozen=True,phase="final_public_audit");persist()
    final=public.public_predictions(root)[-1]
    if final!=lineage or public.helper_identities()!=helpers:raise ValueError("Frozen observations/source/models/proxies changed")
    if (report["counters"]["evaluated_states"]!=900 or report["counters"]["adam_updates"]!=885
            or len(report["fit_records"])!=15 or {p.name for p in out.iterdir()}!={"report.json","candidates"}):
        raise ValueError("Exact complete H97 evaluated-state/export protocol required")
    if"torch"in sys.modules:raise ValueError("Translation-only CPU trial must not import Torch")
    report.update(status="pass",phase="complete",frames=15,all_cases_retained=True,inputs_unchanged=True)
    report.pop("active_file",None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));out=root/public.OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or out.resolve()!=out.absolute()or os.geteuid()!=1000
            or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.environ.get("WR_IMAGE_ID")!=native.IMAGE_ID
            or os.environ.get("CUDA_VISIBLE_DEVICES")!=""or not re.fullmatch("[0-9a-f]{40}",revision)or not out.is_dir()or any(out.iterdir())):
        raise ValueError("Fresh reserved offline CPU translation trial required")
    (out/"candidates").mkdir();started=time.perf_counter();path=out/"report.json"
    report=dict(stage=public.STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=native.IMAGE_ID,
        script_sha256=sha256(Path(__file__)),network="none",device="cpu",budget_seconds=BUDGET,optimizer=public.OPTIMIZER,
        training_COCO_indices=list(policy.TRAIN_COCO),training_MHR_indices=list(policy.TRAIN_MHR),initial_jacobian_includes_priors=False,
        object_proxies_reused=True,proxies_recomputed=False,methodological_prior_cohort_reuse=True,native_controls_unchanged=True,
        geometry_translation_precision="float64 policy deltaT then one float32 geometry export; pred_cam_t storedfloat32",
        shared_identity_constant=True,body_hands_fixed=True,camera_fixed=True,gpu_used=False,Torch_imported=False,
        native_forward_calls=0,private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,
        all_candidates_frozen=False,inputs_unchanged=False,all_cases_retained=False,candidate_outputs=[],fit_records=[],
        counters=dict(evaluated_states=0,adam_updates=0,initial_jacobian_rows=0,native_forward_calls=0,proxy_rasters=0))
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Whole public translation trial exceeded60s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();perform(root,out,report,persist)
        except Exception as error:report.update(error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
