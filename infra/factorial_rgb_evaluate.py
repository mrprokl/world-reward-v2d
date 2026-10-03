"""One frozen H99 private diagnostic, never fitting or a challenge score.

All24 public observations must pass their original contracts before any own
reference label is opened. Factor contrasts are descriptive, equally weighted;
the first-human full-vertex Sim3 is a separate three-pose gauge diagnostic.
"""
import argparse
from dataclasses import asdict
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
import factorial_rgb_observe as public
import joint_rgb_evaluate as alignment
from root_rgb_metrics import regular, floating, require_fields
from world_reward.data import sha256

COHORT = public.COHORT
BASE, OUT = COHORT.base, COHORT.base+"/quality_v1"
STAGE, BUDGET = "private_factorial_rgb_observation_diagnostic", 120
IMAGE = public.native.IMAGE_ID
PIN_SCHEMA = "world-reward-factorial-quality-pins-v1"
ROLES = ("render", "body", "dwpose")
PIN_KEYS = {"schema", "public_pins_sha256", *ROLES}
PRODUCER_KEYS = {"producer_revision", "report_sha256", "report_bytes", "script_sha256"}
COCO_MHR = (0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14)
LIMBS_COCO = ((5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16))
TRUTH_KEYS = {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces", "camera_K",
    "scene_depth_m", "visible_face_indices", "group_index", "frame_index", "human_joints_camera_m", "human_keypoints_camera_m"}
FACE_SHA = "f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
BODY_SHA = "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
HEAD_SHA = "62af48b1f33462bc445d7f342009fcbceabd221f2a6e9bcd1d3739d582bc9fd6"
NAMES_SHA = "695c2c7d472e32757c480114fdb054d54ee4af53f69b2a6e040b00a55b270dc9"
RULE = dict(relative_detector_gain_required=.10, supporting_groups_required=6, groups=8, frames_per_group=3,
    weighting="equal indices within frame, equal3 poses within group, equal8 groups",
    paired_validity="same COCO17 indices with native DWPose score>0; no confidence weights or GT visibility",
    every_frame_needs_positive_pair=True, both_appearance_levels_required=True, both_occlusion_levels_required=True)


def load_pins(path, public_path):
    receipt=regular(path);value=json.loads(Path(path).read_text())
    if not isinstance(value,dict) or set(value)!=PIN_KEYS or value.get("schema")!=PIN_SCHEMA:
        raise ValueError("Exact explicit completed diagnostic producer pins required")
    if type(value["public_pins_sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}",value["public_pins_sha256"]):
        raise ValueError("Immutable public configuration SHA required")
    if regular(public_path)["sha256"]!=value["public_pins_sha256"]:raise ValueError("Public configuration changed")
    for role in ROLES:
        row=value[role]
        if not isinstance(row,dict) or set(row)!=PRODUCER_KEYS:raise ValueError("Exact producer identity required")
        for name,size in (("producer_revision",40),("report_sha256",64),("script_sha256",64)):
            if type(row[name]) is not str or not re.fullmatch("[0-9a-f]{"+str(size)+"}",row[name]):raise ValueError("Canonical completed source/receipt required")
        if type(row["report_bytes"]) is not int or row["report_bytes"]<=0:raise ValueError("Positive completed receipt size required")
    return value,receipt


def producer_source(root,pin,role):
    stem,name=("run_factorial_rgb_prepare","factorial_rgb_render.py") if role=="render" else ("run_factorial_rgb_observe","factorial_rgb_observe.py")
    return Path(root)/"jobs"/pin["producer_revision"]/stem/"code"/"infra"/name


def quality_sources():
    return [regular(Path(__file__)),regular(Path(alignment.__file__))]


def pinned_report(path,pin):
    receipt=regular(path,pin["report_sha256"])
    if receipt["bytes"]!=pin["report_bytes"]:raise ValueError("Completed receipt size changed")
    report=json.loads(Path(path).read_text())
    if not isinstance(report,dict) or any(k in report for k in ("error","error_type")):raise ValueError("Complete passing receipt only")
    return report,receipt


def public_observations(root,pins,quality):
    records,inputs=public.public_inputs(root,pins);reports={};arrays={};frozen=[]
    for role,validator in (("body",public.validate_body),("dwpose",public.validate_dw)):
        pin=quality[role];source=regular(producer_source(root,pin,role),pin["script_sha256"]);frozen.append(source)
        path=Path(root)/BASE/public.FOLDERS[role]/"report.json";report,receipt=pinned_report(path,pin);frozen.append(receipt)
        helpers=public.source_identity() | {"observer":pin["script_sha256"]}
        require_fields(report,dict(stage=public.STAGES[role],status="pass",phase="complete",cohort=asdict(COHORT),frames=24,
            producer_revision=pin["producer_revision"],image_id=IMAGE,script_sha256=pin["script_sha256"],source_helpers=helpers,
            network="none",budget_seconds=public.BUDGETS[role],private_truth_read=False,ground_truth_used=False,
            challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],fitting_performed=False,
            accuracy_verified=False,quality_verified=False,adoption_authorized=False,all_cases_retained=True,source_inputs_assets_rehashed=True))
        expected_inputs=inputs if role=="body" else public.public_inputs(root,pins,("human",))[1]
        require_fields(report,dict(public_inputs=expected_inputs))
        if role=="body":
            require_fields(report,dict(actual_body_inference=True,native_arrays_verified=True,body_inference_type="body",
                camera_K=[list(r)for r in COHORT.fixed_K],keypoint_count=308,joint_count=127,
                identity_policy="ordinary framewise prediction; no shared identity/adoption",
                **{k:24 for k in ("body_attempts","body_calls_completed","parity_head_attempts","parity_heads_completed","keypoint_head_attempts","keypoint_heads_completed")}))
            model=report.get("body_model")
            if not isinstance(model,dict):raise ValueError("Actual original Body loader evidence required")
            loading=model.get("checkpoint_loading")
            require_fields(loading,dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention",parameter_tensors_loaded=1101,unexpected_keys=[]))
            retained=loading.get("retained_mhr_asset_buffer_names")
            if not isinstance(retained,list)or len(retained)!=113 or len(set(retained))!=113:raise ValueError("Exact113 immutable rig-buffer retention required")
            require_fields(model,dict(inference_source_identity=public.native.human.body._source_identity(root),body_assets=public.native.human.body._body_assets(root)[1]))
            semantic_path=Path(root)/"results/mhr-finger-semantics-v4.json";semantic=public.native.regular(semantic_path)
            public.native.regions_helper.require_semantic_report(json.loads(semantic_path.read_text()))
            if semantic["sha256"]!=report.get("semantic_report_sha256"):raise ValueError("Body semantic provenance changed")
            if json.loads(semantic_path.read_text()).get("source_image_id")!=IMAGE:raise ValueError("Semantic producer image changed")
            for row in report["records"]:
                errors=row.get("native_forward_errors")
                if not isinstance(errors,dict) or set(errors)!={"vertices_m","joints_m","keypoints_m","controls"}:raise ValueError("Full native block parity required")
                numbers=floating(list(errors.values()),(4,),"native parity")
                if np.any(numbers<0)or np.any(numbers>1e-5):raise ValueError("Native block fidelity failed")
        else:
            require_fields(report,dict(actual_sessions=1,native_cpu_abi_verified=True,private_prefix_packages_installed=True,private_prefix_removed=True,
                native_source_modified=False,own_feed_cast=False,channel_swap=False,full_image_fallback=False,raw_scores_clamped=False,
                confidence_threshold_applied=False,wrapper_neck134_used=False,global_image_modified=False,gpu_used=False,
                capability_evidence=public.dw_helper.validate_smoke(root),assets=public.smoke.validate_assets(root),dependencies=public.smoke.dependency_identity()))
            calls=report.get("calls")
            if not isinstance(calls,list) or len(calls)!=24 or any(c.get("run_completed")is not True for c in calls):raise ValueError("All24 actual native CPU calls required")
        values=public.frozen_artifacts(path.parent,report.get("records"),records,validator)
        for row,data in zip(report["records"],values):
            frozen.append(regular(path.parent/row["prediction_file"],row["prediction"]["sha256"]))
            if role=="dwpose":
                for key in ("keypoints","scores","validity","bbox"):
                    require_fields(row,{key:public.smoke.array_identity(data[key])})
        reports[role]=report;arrays[role]=values
    return records,arrays,reports,inputs,frozen


def validate_truth(data,record,faces):
    if set(data)!=TRUTH_KEYS:raise ValueError("Exact11 private manufacturing fields required")
    for key,count in (("human_vertices_camera_m",18439),("object_vertices_camera_m",194),("human_joints_camera_m",127),("human_keypoints_camera_m",308)):
        value=floating(data[key],(count,3),key)
        if data[key].dtype!=np.float64 or np.any(value[:,2]<=0):raise ValueError("Original F64 positive camera truth required")
    hf,of=data["human_faces"],data["object_faces"]
    if (type(faces)is not np.ndarray or faces.dtype!=np.int64 or faces.shape!=(36874,3)
        or type(hf)is not np.ndarray or hf.dtype!=np.int32 or hf.shape!=(36874,3)or not np.array_equal(hf,faces)
        or hashlib.sha256(hf.tobytes()).hexdigest()!=FACE_SHA or np.any(hf<0)or np.any(hf>=18439)
        or type(of)is not np.ndarray or of.dtype!=np.int64 or of.shape!=(384,3)or np.any(of<0)or np.any(of>=194)
        or np.any(of[:,0]==of[:,1])or np.any(of[:,1]==of[:,2])or np.any(of[:,0]==of[:,2])):raise ValueError("Native typed topology changed")
    for name in ("group_index","frame_index"):
        if type(data[name])is not np.ndarray or data[name].dtype!=np.int64 or data[name].shape!=()or data[name].item()!=record[name]:raise ValueError("Original private frame identity required")
    if data["camera_K"].dtype!=np.float64 or not np.array_equal(data["camera_K"],np.asarray(COHORT.fixed_K)):raise ValueError("Common fixed manufacturing camera required")
    depth,ids=data["scene_depth_m"],data["visible_face_indices"]
    if (depth.dtype!=np.float32 or depth.shape!=(768,1024)or ids.dtype!=np.int64 or ids.shape!=depth.shape
        or np.any(ids<-1)or np.any(ids>=len(hf)+len(of))or not np.isnan(depth[ids<0]).all()
        or not np.isfinite(depth[ids>=0]).all()or np.any(depth[ids>=0]<=0)
        or np.count_nonzero((ids>=0)&(ids<len(hf)))<64 or np.count_nonzero(ids>=len(hf))<64):raise ValueError("Complete foreground manufacturing evidence required")


def private_truth(root,records,pin,frozen):
    private=Path(root)/BASE/"eval_private";render,receipt=pinned_report(private/"render-report.json",pin);frozen.append(receipt)
    frozen.append(regular(producer_source(root,pin,"render"),pin["script_sha256"]))
    require_fields(render,dict(stage="own_factorial_rgb_render",status="pass",phase="complete",frames=24,groups=8,code_revision=pin["producer_revision"],
        image_id=IMAGE,script_sha256=pin["script_sha256"],network="none",challenge_inputs_used=False,inference_performed=False,
        optimizer_performed=False,quality_verified=False,accuracy_verified=False,photorealism_verified=False,all_truth_private=True,
        synthetic_truth_used_for_rendering_only=True,true_camera_private=True,actual_MHR_reference_used=True,
        actual_reference_forward_calls=2,reference_forward_attempts=2,reference_forward_returns=2,
        bundled_forward_attempts=1,bundled_forward_returns=1,bundled_forward_validated=1,actual_total_native_forward_calls=3,
        bundled_reference_same_joint_names=True,bundled_reference_same_topology=True,
        raster_attempts=24,raster_returns=24,all_factors_geometry_independent=True,truth_keys=sorted(TRUTH_KEYS),
        public_manifest_sha256=public.protocol.identity(Path(root)/BASE/"inputs/manifest.json")["sha256"]))
    mapping=render.get("mapping")
    require_fields(mapping,dict(body_checkpoint_sha256=BODY_SHA,body_checkpoint_bytes=2109129346,head_source_sha256=HEAD_SHA,
        names_source_sha256=NAMES_SHA,keypoint_mapping_shape=[308,18566],keypoint_mapping_dtype="float32",normalized_or_modified=False,
        source="head_pose.keypoint_mapping",bundled_rig_sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc",bundled_rig_bytes=696110248))
    if not re.fullmatch("[0-9a-f]{64}",str(mapping.get("keypoint_mapping_sha256",""))):raise ValueError("Actual native mapping byte identity required")
    # The renderer owns source/bundled-rig compatibility; quality does not run a model.
    parity=render.get("bundled_reference_parity")
    if not isinstance(parity,dict) or set(parity)!={"vertices_m","joints_m"}:raise ValueError("Gated same-parameter bundled/reference parity required")
    values=floating(list(parity.values()),(2,),"same-rig parity")
    if np.any(values<0)or np.any(values>1e-5):raise ValueError("Native bundled/reference parity failed")
    occlusions=render.get("occlusion_evidence")
    if not isinstance(occlusions,list)or len(occlusions)!=12:raise ValueError("All12 manufactured front/back pairs required")
    for row,(m,a,f) in zip(occlusions,((m,a,f)for m in range(2)for a in range(2)for f in range(3))):
        require_fields(row,dict(morphology_index=m,appearance_index=a,frame_index=f))
        if type(row.get("newly_visible_human_pixels"))is not int or row["newly_visible_human_pixels"]<64:raise ValueError("Frozen occlusion factor was not useful")
    cases=render.get("cases")
    if not isinstance(cases,list)or len(cases)!=24:raise ValueError("All24 own reference records required")
    expected={"render-report.json","semantic-report.json","rig.npz",*[Path(r["file"]).stem+".npz"for r in records]}
    if {p.name for p in private.iterdir()}!=expected:raise ValueError("Exact private24 inventory required")
    for name,key in (("rig.npz","rig_sha256"),("semantic-report.json","semantic_report_sha256")):
        frozen.append(regular(private/name,render.get(key)))
    return private,cases,render


def project(points,K):
    p=floating(points,(len(points),3),"projected points")
    if np.any(p[:,2]<=0)or not np.array_equal(K,np.asarray(COHORT.fixed_K)):raise ValueError("Common positiveZ pinhole required")
    return p[:,:2]/p[:,2,None]*np.diag(K)[:2]+K[:2,2]


def geometry_metrics(prediction,truth):
    p,q=floating(prediction,(len(prediction),3),"prediction"),floating(truth,(len(prediction),3),"truth")
    delta=p-q;centroid=delta.mean(0);centered=delta-centroid
    rms2=float(np.mean(np.sum(delta*delta,axis=1)));centered2=float(np.mean(np.sum(centered*centered,axis=1)))
    if not np.isclose(rms2,np.dot(centroid,centroid)+centered2,atol=1e-12,rtol=1e-12):raise ValueError("RMS bias decomposition failed")
    return dict(pve_cm=float(np.linalg.norm(delta,axis=1).mean()*100),rms_cm=float(np.sqrt(rms2)*100),
        centroid_error_xyz_cm=(centroid*100).tolist(),centroid_error_cm=float(np.linalg.norm(centroid)*100),
        centered_pve_cm=float(np.linalg.norm(centered,axis=1).mean()*100),centered_rms_cm=float(np.sqrt(centered2)*100))


def frame_metrics(body,dw,truth):
    indices=np.asarray(COCO_MHR);p=body["keypoints_camera_m"][indices];q=truth["human_keypoints_camera_m"][indices];K=truth["camera_K"]
    bp,tq=project(p,K),project(q,K);body_errors=np.linalg.norm(bp-tq,axis=1)
    valid=dw["validity"][0,:17];detector_errors=np.linalg.norm(dw["keypoints"][0,:17]-tq,axis=1)
    lengths=lambda v:np.asarray([np.linalg.norm(v[a]-v[b])for a,b in LIMBS_COCO])
    true_lengths=lengths(q)
    if np.any(true_lengths<=1e-8):raise ValueError("Derived anatomical limbs must be nondegenerate")
    return dict(body_geometry=geometry_metrics(body["vertices_camera_m"],truth["human_vertices_camera_m"]),
        body_joint_pve_cm=geometry_metrics(body["joints_camera_m"],truth["human_joints_camera_m"])["pve_cm"],
        body_COCO17_keypoint_pve_cm=geometry_metrics(p,q)["pve_cm"],limb_length_relative_error=(lengths(p)/true_lengths-1).tolist(),
        body_all17_reprojection_px=float(body_errors.mean()),body_COCO17_reprojection_px=body_errors.tolist(),
        native_DW_positive_COCO17=valid.tolist(),native_DW_positive_count=int(valid.sum()),
        body_paired_reprojection_px=float(body_errors[valid].mean())if valid.any()else None,
        dwpose_paired_reprojection_px=float(detector_errors[valid].mean())if valid.any()else None,
        dwpose_COCO17_errors_px=[float(v)if ok else None for v,ok in zip(detector_errors,valid)])


def factor_index(group):return group//4,(group//2)%2,group%2


def aggregate(rows):
    if len(rows)!=24:raise ValueError("Every24 diagnostic frame required")
    groups=[]
    for i,row in enumerate(rows):require_fields(row,dict(group_index=i//3,frame_index=i%3))
    for group in range(8):
        cases=rows[group*3:(group+1)*3];paired=all(c["native_DW_positive_count"]>0 for c in cases)
        before=float(np.mean([c["body_paired_reprojection_px"]for c in cases]))if paired else None
        after=float(np.mean([c["dwpose_paired_reprojection_px"]for c in cases]))if paired else None
        gain=(before-after)/before if paired and before>0 else None
        m,a,o=factor_index(group)
        groups.append(dict(group_index=group,morphology_index=m,appearance_index=a,occlusion_index=o,
            body_paired_reprojection_px=before,dwpose_paired_reprojection_px=after,relative_detector_gain=gain,
            native_DW_positive_counts=[c["native_DW_positive_count"]for c in cases],
            body_all17_reprojection_px=float(np.mean([c["body_all17_reprojection_px"]for c in cases])),
            body_pve_cm=float(np.mean([c["body_geometry"]["pve_cm"]for c in cases])),
            body_centered_pve_cm=float(np.mean([c["body_geometry"]["centered_pve_cm"]for c in cases])),
            body_signed_centroid_Z_cm=float(np.mean([c["body_geometry"]["centroid_error_xyz_cm"][2]for c in cases])),
            first_human_sim3_pve_cm=float(np.mean([c["first_human_sim3_geometry"]["pve_cm"]for c in cases]))))
    supporting=[g["group_index"]for g in groups if g["relative_detector_gain"]is not None and g["relative_detector_gain"]>=.10]
    coverage=all(g["relative_detector_gain"]is not None for g in groups)
    level_gates={name:all(any(factor_index(i)[column]==level for i in supporting)for level in (0,1))for name,column in (("both_appearance_levels",1),("both_occlusion_levels",2))}
    gates=dict(every24_frame_has_positive_pair_and_all8_gains_defined=coverage,six_of_eight_groups_gain_at_least_10pct=len(supporting)>=6,**level_gates)
    contrasts=[]
    for factor,column in (("appearance",1),("occlusion",2)):
        for group in range(8):
            if factor_index(group)[column]!=0:continue
            other=group+(2 if factor=="appearance"else 1)
            for frame in range(3):
                a,b=rows[group*3+frame],rows[other*3+frame]
                common=np.asarray(a["native_DW_positive_COCO17"]) & np.asarray(b["native_DW_positive_COCO17"])
                paired={}
                for mode,key in (("body","body_COCO17_reprojection_px"),("dwpose","dwpose_COCO17_errors_px")):
                    before=np.asarray(a[key],dtype=np.float64)[common];after=np.asarray(b[key],dtype=np.float64)[common]
                    paired[mode+"_common_positive_reprojection_level1_minus_level0_px"]=float(after.mean()-before.mean())if common.any()else None
                contrasts.append(dict(factor=factor,level0_group=group,level1_group=other,frame_index=frame,
                    body_all17_reprojection_level1_minus_level0_px=b["body_all17_reprojection_px"]-a["body_all17_reprojection_px"],
                    common_native_positive_count=int(common.sum()),**paired,
                    paired_native_positive_subsets_identical=a["native_DW_positive_COCO17"]==b["native_DW_positive_COCO17"]))
    decision=dict(rule=RULE,gates=gates,supporting_groups=supporting,
        automatic_prompt_pilot_evidence_supported=all(gates.values()),adoption_authorized=False,accuracy_verified=False,
        full_HOI_verified=False,CARI4D_superiority_verified=False,interpretation="finite manufactured-cohort diagnosis; not independent people/statistical significance")
    return groups,contrasts,decision


def run(root,report,public_path,quality_path,persist=lambda:None):
    pins=public.validate_pins(json.loads(Path(public_path).read_text()),"body");quality,qreceipt=load_pins(quality_path,public_path)
    records,arrays,producers,inputs,frozen=public_observations(root,pins,quality)
    frozen.extend((qreceipt,regular(public_path),*quality_sources()))
    if "torch"in sys.modules:raise ValueError("No Torch/model runtime may precede private quality")
    report.update(predictions_frozen_before_private=True,all24_public_observations_verified=True,phase="private_diagnostic",rule=RULE,
        private_truth_read=True,private_truth_used_for_evaluation_only=True);persist()
    private,cases,render=private_truth(root,records,quality["render"],frozen);rows=[];transforms={};paired_geometry={}
    for record,case,body,dw in zip(records,cases,arrays["body"],arrays["dwpose"]):
        group,frame=record["group_index"],record["frame_index"];m,a,o=factor_index(group)
        require_fields(case,dict(file=record["file"],rgb_sha256=record["sha256"],group_index=group,frame_index=frame,morphology_index=m,appearance_index=a,occlusion_index=o))
        path=private/(Path(record["file"]).stem+".npz");frozen.append(regular(path,case.get("truth_sha256")))
        with np.load(path,allow_pickle=False)as saved:truth={k:saved[k]for k in saved.files}
        validate_truth(truth,record,body["human_faces"])
        hashes=tuple(hashlib.sha256(truth[k].tobytes()).hexdigest()for k in("human_vertices_camera_m","human_joints_camera_m","human_keypoints_camera_m","camera_K"))
        key=(m,frame)
        if key in paired_geometry and paired_geometry[key]!=hashes:raise ValueError("Appearance/occlusion must not change human geometry/camera")
        paired_geometry[key]=hashes
        if frame==0:transforms[group]=alignment.fit_human_sim3(body["vertices_camera_m"],truth["human_vertices_camera_m"])
        transformed=alignment.transform(body["vertices_camera_m"],transforms[group])
        rows.append(dict(group_index=group,frame_index=frame,**frame_metrics(body,dw,truth),
            first_human_sim3_geometry=geometry_metrics(transformed,truth["human_vertices_camera_m"])))
    groups,contrasts,decision=aggregate(rows)
    for receipt in frozen:
        if regular(receipt["path"],receipt["sha256"])["bytes"]!=receipt["bytes"]:raise ValueError("Frozen source/public/private bytes changed")
    if public.public_inputs(root,pins)[1]!=inputs:raise ValueError("Public source/input lineage changed")
    # The original validators rehash host-writable model assets, mounted read-only;
    # result-file mode checks must not be imposed on those canonical assets.
    if public.native.human.body._source_identity(root)!=producers["body"]["body_model"]["inference_source_identity"]or public.native.human.body._body_assets(root)[1]!=producers["body"]["body_model"]["body_assets"]:raise ValueError("Body sources/assets changed")
    if public.smoke.validate_assets(root)!=producers["dwpose"]["assets"]or public.dw_helper.validate_smoke(root)!=producers["dwpose"]["capability_evidence"]:raise ValueError("Native detector assets/capability changed")
    semantic=public.native.regular(Path(root)/"results/mhr-finger-semantics-v4.json")
    if semantic["sha256"]!=producers["body"]["semantic_report_sha256"]:raise ValueError("Canonical semantic receipt changed")
    if "torch"in sys.modules:raise ValueError("Private CPU quality must not import or execute Torch")
    report.update(status="pass",phase="complete",frame_metrics=rows,group_metrics=groups,factor_contrasts=contrasts,decision=decision,
        macro_group_means={k:float(np.mean([g[k]for g in groups]))if all(g[k]is not None for g in groups)else None for k in
            ("body_paired_reprojection_px","dwpose_paired_reprojection_px","body_all17_reprojection_px","body_pve_cm","body_centered_pve_cm","body_signed_centroid_Z_cm","first_human_sim3_pve_cm")},
        first_human_sim3_transforms=transforms,first_human_sim3_scope="one proper positive full18439 firsthuman transform/group; same transform3poses, NOT official subset/HOI score",
        private_truth_used_for_evaluation_only=True,all_frames_scored=True,all_cases_retained=True,all_factors_geometry_independent_verified=True,
        derived_landmarks_definition="frozen checkpoint mapper on own reference V/J; not measured independent anatomical truth",
        no_GT_visibility_selection=True,private_optimizer_or_model_calls=0,source_inputs_assets_rehashed=True,
        accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,CARI4D_superiority_verified=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);parser.add_argument("--public-pins",type=Path,required=True);parser.add_argument("--quality-pins",type=Path,required=True);args=parser.parse_args(argv)
    root=Path(os.environ.get("WR_ROOT",""));code=Path(os.environ.get("WR_CODE",""));revision=os.environ.get("WR_CODE_REVISION","");out=root/OUT
    if (platform.system()!="Linux"or root!=Path("/srv/scenesmith/world-reward")or not code.is_absolute()or os.geteuid()!=1000
        or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.environ.get("WR_IMAGE_ID")!=IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES")!=""
        or not re.fullmatch("[0-9a-f]{40}",revision)or not out.is_dir()or any(out.iterdir())or out.resolve()!=out.absolute()
        or any(p.is_symlink()for p in(out,*out.parents))):raise ValueError("Fresh canonical offline CPU diagnostic required")
    if args.public_pins!=code/"configs/factorial_rgb_public_pins_v1.json"or args.quality_pins!=code/"configs/factorial_rgb_quality_pins_v1.json":raise ValueError("Explicit immutable source-bundle configurations required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=IMAGE,script_sha256=sha256(Path(__file__)),
        network="none",budget_seconds=BUDGET,predictions_frozen_before_private=False,private_truth_read=False,private_truth_used_for_evaluation_only=False,challenge_inputs_used=False,
        accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,CARI4D_superiority_verified=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Private factorial diagnostic exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,report,args.public_pins,args.quality_pins,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
