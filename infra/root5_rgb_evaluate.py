"""One H98 private scoring after complete public traces and native-best replay.

The unchanged camera-PVE/hand-vector metrics receive all15 frames even when
an automatic silhouette safeguard fails. No truth is passed to the producers,
no fitting occurs here, and a synthetic result never authorizes adoption.
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
import root_rgb_metrics as metrics
import root5_rgb_public as public
from world_reward.data import sha256

COHORT=public.protocol.COHORT
BASE=COHORT.base
OUT=BASE+"/quality_v1"
STAGE="private_root5_rgb_paired_fixed_depth_quality"
REPLAY_OUT=BASE+"/root_replay_v1"
REPLAY_STAGE="public_root5_rgb_native_fixed_depth_replay"
IMAGE="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BUDGET=120
RENDER_REVISION="077a03910d3568634c3b0d0e0b8ae0978303ce28"
RENDER_SHA="3c3ac975eda3f0b7975a7f0ef210075d89b26b2c91f910d46e14945a93150a2a"
RENDER_BYTES=30085
RENDER_SOURCE_SHA="608ce90af1ed555e59bcdd8b4c984c86dc1f68c51e0be53b74fcc6392c64a265"
MANIFEST_SHA="16d909d206ea9a71f402c01a699e69d63603e5b1366e1e24d55ac208846f466b"
HUMAN_FACES_SHA="f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
TRUTH_KEYS=metrics.TRUTH_KEYS
regular,require_fields,floating=metrics.regular,metrics.require_fields,metrics.floating
frame_metrics,aggregate=metrics.frame_metrics,metrics.aggregate
visible_object_median,heldout_diagnostics=metrics.visible_object_median,metrics.heldout_diagnostics
OPTIMIZER=dict(evaluated_states_per_frame=60,updates_per_frame=59,learning_rate=.01,betas=[.9,.999],eps=1e-8)
QUALITY_KEYS={"schema","public_pins_sha256",*[stage+suffix for stage in("fit","replay")for suffix in("_revision","_report_sha256","_report_bytes","_script_sha256")]}


def load_quality_pins(path,public_path):
    identity=regular(path);value=json.loads(Path(path).read_text())
    if not isinstance(value,dict)or set(value)!=QUALITY_KEYS or value.get("schema")!="world-reward-root5-quality-pins-v1":
        raise ValueError("Complete explicit quality/replay pins required; no fallback")
    for name in("public_pins_sha256","fit_report_sha256","fit_script_sha256","replay_report_sha256","replay_script_sha256"):
        if type(value[name])is not str or not re.fullmatch("[0-9a-f]{64}",value[name]):raise ValueError("Exact source/receipt SHA required")
    for stage in("fit","replay"):
        if (type(value[stage+"_revision"])is not str or not re.fullmatch("[0-9a-f]{40}",value[stage+"_revision"])
                or type(value[stage+"_report_bytes"])is not int or value[stage+"_report_bytes"]<=0):raise ValueError("Immutable producer/positive receipt bytes required")
    if regular(public_path)["sha256"]!=value["public_pins_sha256"]:raise ValueError("Public pins/config bytes differ")
    return value,identity


def producer_source(root,revision,stem,name):
    if not re.fullmatch("[0-9a-f]{40}",revision)or not re.fullmatch("run_root5_rgb_[a-z]+",stem)or not re.fullmatch(r"[a-z0-9_]+\.py",name):
        raise ValueError("Explicit canonical source-only producer required")
    return Path(root)/"jobs"/revision/stem/"code"/"infra"/name


def source_bindings(root,pins):
    fit=producer_source(root,pins["fit_revision"],"run_root5_rgb_fit","root5_rgb_fit.py")
    replay=producer_source(root,pins["replay_revision"],"run_root5_rgb_fit","root5_rgb_fit.py")
    return dict(quality=regular(Path(__file__)),metrics=regular(Path(metrics.__file__)),public=regular(Path(public.__file__)),
        fit=regular(fit,pins["fit_script_sha256"]),replay=regular(replay,pins["replay_script_sha256"]))


def parity_errors(row,key):
    errors=row.get(key)
    if not isinstance(errors,dict)or set(errors)!={"vertices_m","keypoints_m","joints_m","controls","rotations"}:
        raise ValueError("Complete native V/KP/J/control/SO3 parity evidence required")
    values=floating(list(errors.values()),(5,),"native replay errors")
    if np.any(values<0)or np.any(values>1e-5):raise ValueError("Native representation parity exceeds frozen1e-5")


def safeguard_decision(value):
    if not isinstance(value,dict):raise ValueError("Frozen15-frame public silhouette safeguard required")
    before=floating(value.get("baseline_iou"),(15,),"baseline silhouette IoU")
    after=floating(value.get("candidate_iou"),(15,),"candidate silhouette IoU")
    if np.any(before<0)or np.any(before>1)or np.any(after<0)or np.any(after>1):raise ValueError("Finite IoUs in[0,1] required")
    delta=after-before;clip=delta.reshape(3,5).mean(1)
    expected=dict(baseline_iou=before.tolist(),candidate_iou=after.tolist(),per_frame_delta=delta.tolist(),per_clip_mean_delta=clip.tolist(),
        worst_frame_delta=float(delta.min()),worst_clip_mean_delta=float(clip.min()),limit=-.01,
        safeguard_pass=bool(np.all(delta>=-.01)and np.all(clip>=-.01)),used_for_state_selection=False,
        candidate_export_blocked_by_safeguard=False,accuracy_verified=False)
    require_fields(value,expected)
    return expected


def audit_replay(root,pins,records,producer,lineage,trace_checks,source):
    path=Path(root)/REPLAY_OUT/"report.json";receipt=regular(path,pins["replay_report_sha256"])
    if receipt["bytes"]!=pins["replay_report_bytes"]:raise ValueError("Exact replay receipt byte count required")
    replay=json.loads(path.read_text())
    if any(k in replay for k in("error","error_type")):raise ValueError("Passing native replay cannot retain an error")
    require_fields(replay,dict(stage=REPLAY_STAGE,status="pass",phase="complete",operation="replay",producer_revision=pins["replay_revision"],
        image_id=IMAGE,script_sha256=pins["replay_script_sha256"],network="none",device="cuda",frames=15,all_cases_retained=True,
        predictions_native_replay_verified=True,source_inputs_assets_rehashed=True,all_training_traces_verified=True,
        all_native_best_replays_verified=True,optimizer_updates=0,additional_training_forwards=0,native_backwards=0,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        shared_identity_constant=True,body_hands_fixed=True,camera_Z_fixed=True,intrinsics_fixed=True,root_bounds_only=True,
        dense_fixed_remainder_bounds_checked=False,lineage=lineage,accuracy_verified=False,quality_verified=False,adoption_authorized=False,
        deterministic_algorithms=True,warn_only=False,TF32=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",seed=0,threads=4,
        body_model=producer["body_model"],native_head_source=producer["native_head_source"],trace_checks=trace_checks,
        source_helpers=public.helper_identities(fit_source=Path(source)),torch_version="2.5.1+cu124",CUDA_version="12.4"))
    if replay.get("fit_report")!=regular(Path(root)/public.OUT/"report.json",pins["fit_report_sha256"]):raise ValueError("Replay did not bind exact fit receipt")
    require_fields(replay.get("counters"),dict(final_native_heads_attempted=15,final_native_heads_returned=15,final_native_heads_validated=15,
        objective_native_heads_attempted=0,objective_native_heads_returned=0,objective_native_heads_validated=0,optimizer_backwards_attempted=0,
        optimizer_backwards_returned=0,adam_updates=0,proxy_rasters_attempted=0,proxy_rasters_completed=0,safeguard_rasters_attempted=0,safeguard_rasters_completed=0))
    rows=replay.get("records")
    if not isinstance(rows,list)or len(rows)!=15:raise ValueError("All15 independent selected-state native replays required")
    for record,row in zip(records,rows):
        require_fields(row,dict(file=record["file"],native_heads_attempted=1,native_heads_returned=1,native_heads_validated=1));parity_errors(row,"replay_parity")
    if {p.name for p in path.parent.iterdir()}!={"report.json"}:raise ValueError("Replay namespace must contain only its frozen receipt")
    regular(source,pins["replay_script_sha256"])
    return receipt


def paired_xy_diagnostic(jacobian,points,K):
    # The pair is not the native zero-forward tensor used for the strict gate.
    # Initial parity is bounded, not bit equality; this check is diagnostic only.
    J=np.asarray(jacobian);p=np.asarray(points);camera=np.asarray(K)
    expected=np.zeros((2*len(p),2));expected[::2,0]=.30*camera[0,0]/p[:,2]
    expected[1::2,1]=.30*camera[1,1]/p[:,2]
    return dict(maximum_absolute_error=float(np.abs(J[:,:2]-expected).max()),
                native_zero_points_are_identical_to_pair_verified=False,used_as_gate=False)


def public_fit(root,public_pins,pins):
    records,raw,pairs,dw,lineage=public.public_predictions(root,public_pins);bindings=source_bindings(root,pins)
    path=Path(root)/public.OUT/"report.json";receipt=regular(path,pins["fit_report_sha256"])
    if receipt["bytes"]!=pins["fit_report_bytes"]:raise ValueError("Frozen fit receipt byte count differs")
    failed=public.previous_fit_failure(root)
    producer=json.loads(path.read_text());fit_source=bindings["fit"]["path"]
    if any(k in producer for k in("error","error_type")):raise ValueError("Passing fit cannot retain an error")
    require_fields(producer,dict(stage="public_root5_rgb_native_fixed_depth_refit",status="pass",phase="complete",operation="fit",producer_revision=pins["fit_revision"],
        script_sha256=pins["fit_script_sha256"],source_helpers=public.helper_identities(fit_source=Path(fit_source)),lineage=lineage,
        image_id=IMAGE,network="none",device="cuda",budget_seconds=300,optimizer=OPTIMIZER,frames=15,all_cases_retained=True,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        shared_identity_constant=True,body_hands_fixed=True,camera_Z_fixed=True,intrinsics_fixed=True,root_bounds_only=True,
        dense_fixed_remainder_bounds_checked=False,inputs_unchanged=True,native_arrays_verified=True,all_candidates_frozen=True,
        object_proxies_frozen_before_fit=True,initial_jacobian_includes_priors=False,training_COCO_indices=list(metrics.policy.TRAIN_COCO),
        training_MHR_indices=list(metrics.policy.TRAIN_MHR),accuracy_verified=False,quality_verified=False,adoption_authorized=False,
        source_inputs_assets_rehashed=True,optimizer_updates=885,native_backwards=885,torch_version="2.5.1+cu124",CUDA_version="12.4",
        previous_failed_fit=failed,previous_failure_rewritten=False,pytorch3d_version="0.7.9",seed=0,TF32=False,CUBLAS_WORKSPACE_CONFIG=":4096:8",deterministic_algorithms=True,warn_only=False,threads=4))
    expected=dict(objective_native_heads_attempted=900,objective_native_heads_returned=900,objective_native_heads_validated=900,
        final_native_heads_attempted=15,final_native_heads_returned=15,final_native_heads_validated=15,optimizer_backwards_attempted=885,
        optimizer_backwards_returned=885,adam_updates=885,proxy_rasters_attempted=15,proxy_rasters_completed=15,safeguard_rasters_attempted=15,safeguard_rasters_completed=15)
    require_fields(producer.get("counters"),expected)
    candidates=public.frozen_candidates(root,records,producer.get("candidate_outputs"),raw,pairs)
    proxies=public.frozen_proxies(root,records,producer.get("proxy_outputs"),raw,pairs)
    rows=producer.get("fit_records")
    if not all(len(v)==15 for v in(records,raw,pairs,dw,candidates,proxies,rows or[])):raise ValueError("All15 frames/traces/proxies/candidates required")
    limits=np.asarray(producer.get("root_bounds"))
    if limits.shape!=(3,2)or limits.dtype.kind!="f"or np.isnan(limits).any()or np.any(limits[:,0]>limits[:,1]):raise ValueError("Native root-only limit metadata required")
    for candidate in candidates:
        if np.any(candidate["global_rot"]<limits[:,0])or np.any(candidate["global_rot"]>limits[:,1]):raise ValueError("Candidate native Euler exceeds root-only limits")
    observed=0;traces=[]
    for record,row,observation,candidate,pair in zip(records,rows,dw,candidates,pairs):
        require_fields(row,dict(file=record["file"],native_heads_attempted=61,native_heads_returned=61,native_heads_validated=61))
        parity_errors(row,"initial_parity");traces.append(public.validate_schedule(row,observation,candidate))
        evidence=row.get("initial_observation_jacobian",{});n=traces[-1]["observation_points"];observed+=2*n
        require_fields(evidence,dict(columns=5,observation_rows=2*n,prior_rows_included=False))
        singular=floating(evidence.get("singular_values"),(5,),"observation singular values")
        if np.any(singular<0)or np.any(np.diff(singular)>0)or singular[0]<=0 or singular[-1]/singular[0]<1e-5:
            raise ValueError("Unregularized5DOF observation rank insufficient")
        analytic=row.get("analytic_XY_jacobian",{})
        require_fields(analytic,dict(rtol=1e-5,atol=1e-6,latent_is_zero=True,additional_native_calls=0))
        error=analytic.get("maximum_absolute_error")
        if type(error)is not float or not np.isfinite(error)or error<0:raise ValueError("Finite analytic XY fidelity evidence required")
        _,indices,_=metrics.policy.training_observations(observation["keypoints"][0],observation["scores"][0])
        points=pair["shared_keypoints_camera_m"][indices];J=np.asarray(row["initial_observation_jacobian_values"],np.float64)
        paired_xy_diagnostic(J,points,pair["camera_K"])
    require_fields(producer["counters"],dict(initial_jacobian_rows_attempted=observed,initial_jacobian_rows_completed=observed))
    if not 180<=observed<=300:raise ValueError("Complete training observation coverage required")
    guard=safeguard_decision(producer.get("silhouette_safeguard"))
    replay=audit_replay(root,pins,records,producer,lineage,traces,bindings["replay"]["path"])
    # Public lineage contains host-writable native assets mounted read-only.
    # Its own audited validator, not the strict result-file validator, rehashes it.
    frozen=[receipt,replay,*bindings.values()]
    for key in("candidate_outputs","proxy_outputs"):
        for row in producer[key]:
            identity=regular(Path(root)/public.OUT/row["artifact"],row["sha256"])
            if identity["bytes"]!=row["bytes"]:raise ValueError("Exact frozen predicted byte count required")
            frozen.append(identity)
    if {p.name for p in path.parent.iterdir()}!={"report.json","candidates","proxies"}:raise ValueError("Exact frozen fit inventory required")
    return records,pairs,candidates,proxies,dw,producer,frozen,guard


def validate_truth(data,record,faces):
    if set(data)!=TRUTH_KEYS:raise ValueError("Exactly9 manufacturing truth fields required")
    floating(data["human_vertices_camera_m"],(18439,3),"human truth");floating(data["object_vertices_camera_m"],(194,3),"object truth")
    hf,of=data["human_faces"],data["object_faces"]
    if (type(faces)is not np.ndarray or faces.dtype!=np.int64 or faces.shape!=(36874,3)
        or type(hf)is not np.ndarray or hf.dtype!=np.int32 or hf.shape!=(36874,3)or not np.array_equal(hf,faces)
        or hashlib.sha256(hf.tobytes()).hexdigest()!=HUMAN_FACES_SHA or np.any(hf<0)or np.any(hf>=18439)
        or type(of)is not np.ndarray or of.dtype!=np.int64 or of.shape!=(384,3)or np.any(of<0)or np.any(of>=194)
        or np.any(of[:,0]==of[:,1])or np.any(of[:,1]==of[:,2])or np.any(of[:,0]==of[:,2])):raise ValueError("Exact native typed truth topology required")
    for name in("clip_index","frame_index"):
        value=data[name]
        if type(value)is not np.ndarray or value.dtype!=np.int64 or value.shape!=()or value.item()!=record[name]:raise ValueError("Original private identity required")
    K=data["camera_K"]
    if type(K)is not np.ndarray or K.dtype!=np.float64 or not np.array_equal(K,np.asarray(COHORT.fixed_K)):raise ValueError("Frozen isolated camera required")
    depth,ids=data["scene_depth_m"],data["visible_face_indices"]
    if (type(depth)is not np.ndarray or depth.dtype!=np.float32 or depth.shape!=(768,1024)or type(ids)is not np.ndarray or ids.dtype!=np.int64
        or ids.shape!=depth.shape or np.any(ids<-1)or np.any(ids>=len(hf)+len(of))or not np.isnan(depth[ids<0]).all()
        or not np.isfinite(depth[ids>=0]).all()or np.any(depth[ids>=0]<=0)
        or np.count_nonzero((ids>=0)&(ids<len(hf)))<64 or np.count_nonzero(ids>=len(hf))<64):raise ValueError("Complete finite visible foreground truth required")


def private_truth(root,records,frozen):
    private=Path(root)/BASE/"eval_private";path=private/"render-report.json";receipt=regular(path,RENDER_SHA)
    if receipt["bytes"]!=RENDER_BYTES:raise ValueError("Frozen render receipt size differs")
    render=json.loads(path.read_text())
    require_fields(render,dict(stage="own_fresh_root5_rgb_render",status="pass",phase="complete",frames=15,code_revision=RENDER_REVISION,
        script_sha256=RENDER_SOURCE_SHA,image_id=IMAGE,actual_MHR_reference_used=True,actual_reference_forward_calls=2,
        reference_forward_attempts=2,reference_forward_returns=2,challenge_inputs_used=False,synthetic_truth_used_for_rendering_only=True,
        inference_performed=False,predictions_performed=False,accuracy_verified=False,quality_verified=False,photorealism_verified=False,
        all_truth_private=True,identity_clip_constant=True,true_camera_private=True,independent_fresh_cohort=True,
        scene_depth_scope="human/object foreground only; background IDs=-1/depth=NaN",public_manifest_sha256=MANIFEST_SHA,
        public_manifest_bytes=2196,truth_keys=sorted(TRUTH_KEYS),truth_human_faces_dtype="int32",truth_object_faces_dtype="int64",
        human_faces_sha256=HUMAN_FACES_SHA,model_sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"))
    helpers={k:sha256(Path(__file__).with_name(name))for k,name in dict(render="hand_synthetic_render.py",primitives="identity_rgb_render.py",
        joint="joint_rgb_render.py",camera="camera_render.py",protocol="root5_rgb_protocol.py").items()}
    require_fields(render,dict(helper_source_sha256=helpers));frozen.append(receipt)
    source=producer_source(root,RENDER_REVISION,"run_root5_rgb_prepare","root5_rgb_render.py");frozen.append(regular(source,RENDER_SOURCE_SHA))
    semantic_path=private/"semantic-report.json";frozen.append(regular(semantic_path,render.get("semantic_report_sha256")))
    import hand_synthetic_render as regions
    semantic=json.loads(semantic_path.read_text());regions.require_semantic_report(semantic)
    if semantic.get("source_image_id")!=IMAGE:raise ValueError("Private renderer semantic source image differs")
    rig_path=private/"rig.npz";frozen.append(regular(rig_path,render.get("rig_sha256")))
    with np.load(rig_path,allow_pickle=False)as saved:rig={k:saved[k]for k in saved.files}
    validate_rig(rig,render)
    cases=render.get("cases")
    if not isinstance(cases,list)or len(cases)!=15:raise ValueError("All15 private cases required")
    expected={"render-report.json","semantic-report.json","rig.npz",*[Path(r["file"]).stem+".npz"for r in records]}
    if {p.name for p in private.iterdir()}!=expected:raise ValueError("Private exact15-frame inventory changed")
    return private,cases


def validate_rig(rig,render):
    if set(rig)!={"controls","shape45","parameter_limits"}:raise ValueError("Original own manufacturing rig required")
    controls=floating(rig["controls"],(15,204),"manufacturing controls");shapes=floating(rig["shape45"],(15,45),"manufacturing identities")
    bounds=rig["parameter_limits"];names=render.get("parameter_names")
    if(type(bounds)is not np.ndarray or bounds.shape!=(249,2)or bounds.dtype.kind!="f"or np.isnan(bounds).any()or np.any(bounds[:,0]>bounds[:,1])
        or rig["controls"].dtype!=np.float32 or rig["shape45"].dtype!=np.float32 or not isinstance(names,list)or len(names)!=249
        or len(set(names))!=249 or any(type(n)is not str or not n for n in names)
        or np.any(np.c_[controls,shapes]<bounds[:,0])or np.any(np.c_[controls,shapes]>bounds[:,1])):raise ValueError("Complete249 manufacturing values/names/bounds required")
    for clip in range(3):
        for i in range(clip*5,(clip+1)*5):
            if shapes[i].tobytes()!=shapes[clip*5].tobytes()or controls[i,136:].tobytes()!=controls[clip*5,136:].tobytes():raise ValueError("Manufacturing identity must remain clip-constant")


def run(root,report,public_path,quality_path,persist=lambda:None):
    if "torch"in sys.modules:raise ValueError("CPU quality cannot import Torch")
    public_pins=public.load_pins(public_path);pins,config=load_quality_pins(quality_path,public_path)
    records,pairs,candidates,proxies,dw,producer,frozen,guard=public_fit(root,public_pins,pins)
    bindings=source_bindings(root,pins)
    frozen.extend((config,regular(public_path)))
    report.update(phase="public_predictions_audited",predictions_frozen_before_private=True,native_best_replay_verified_before_private=True,
        fit_report_sha256=pins["fit_report_sha256"],replay_report_sha256=pins["replay_report_sha256"],public_lineage=producer["lineage"],
        silhouette_safeguard=guard,private_truth_used_for_evaluation_only=True);persist()
    private,cases=private_truth(root,records,frozen);scores=[]
    for record,pair,candidate,proxy,observation,case in zip(records,pairs,candidates,proxies,dw,cases):
        require_fields(case,dict(file=record["file"],rgb_sha256=record["sha256"],clip_index=record["clip_index"],frame_index=record["frame_index"],human_faces_sha256=HUMAN_FACES_SHA))
        path=private/(Path(record["file"]).stem+".npz");frozen.append(regular(path,case.get("truth_sha256")))
        with np.load(path,allow_pickle=False)as saved:truth={k:saved[k]for k in saved.files}
        validate_truth(truth,record,pair["human_faces"])
        if hashlib.sha256(truth["object_faces"].tobytes()).hexdigest()!=case.get("object_faces_sha256"):raise ValueError("Typed object face bytes changed")
        median,count=visible_object_median(truth)
        values=frame_metrics(dict(raw=pair["raw_vertices_camera_m"],baseline=pair["shared_vertices_camera_m"],fitted=candidate["vertices_camera_m"]),
            truth["human_vertices_camera_m"],proxy["object_points_camera_m"],median,pair["hand_mask_left"],pair["hand_mask_right"])
        scores.append(dict(clip_index=record["clip_index"],frame_index=record["frame_index"],true_visible_object_pixels=count,
            heldout_RGB_diagnostics=heldout_diagnostics(pair,candidate,observation),**values))
    clips,decision=aggregate(scores);decision["gates"]["no_automatic_human_silhouette_degradation_over_1pp"]=guard["safeguard_pass"]
    decision["synthetic_root_refit_hypothesis_supported"]=all(decision["gates"].values())
    for identity in frozen:
        if regular(identity["path"],identity["sha256"])["bytes"]!=identity["bytes"]:raise ValueError("Frozen public/private bytes changed")
    if public.public_predictions(root,public_pins)[-1]!=producer["lineage"]:
        raise ValueError("Public source/model/input lineage changed")
    if source_bindings(root,pins)!=bindings:raise ValueError("Frozen quality/public/metric source changed")
    if "torch"in sys.modules:raise ValueError("CPU quality must not import Torch")
    report.update(status="pass",phase="complete",frame_metrics=scores,clip_metrics=clips,decision=decision,all_frames_scored=True,
        all_cases_retained=True,no_gt_alignment=True,wrists_or_joint_truth_used=False,private_render_report_sha256=RENDER_SHA,
        diagnostic_centering_not_primary_alignment=True,private_optimizer_or_model_calls=0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--public-pins",required=True);parser.add_argument("--quality-pins",required=True);args=parser.parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));code=Path(os.environ.get("WR_CODE",""));out=root/OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if(platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or not code.is_absolute()or os.geteuid()!=1000
        or{p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.environ.get("WR_IMAGE_ID")!=IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES")!=""
        or not re.fullmatch("[0-9a-f]{40}",revision)or not out.is_dir()or any(out.iterdir())or out.resolve()!=out.absolute()
        or any(p.is_symlink()for p in(out,*out.parents))):raise ValueError("Fresh canonical offline CPU quality output required")
    for name in(args.public_pins,args.quality_pins):
        path=Path(name)
        if path.parent!=code/"configs"or path.resolve()!=path.absolute():raise ValueError("Explicit immutable code/configs pins required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=IMAGE,script_sha256=sha256(Path(__file__)),
        network="none",budget_seconds=BUDGET,private_truth_used_for_evaluation_only=False,predictions_frozen_before_private=False,
        native_best_replay_verified_before_private=False,challenge_inputs_used=False,accuracy_verified=False,adoption_authorized=False,
        full_HOI_verified=False,real_domain_verified=False,rigid_object_mesh_or_contact_verified=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n")
            stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Private root5 quality exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,report,args.public_pins,args.quality_pins,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
