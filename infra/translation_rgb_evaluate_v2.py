"""H97 v2 fixes only native int32 truth-face ABI; frozen fit and gates retained."""
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
import translation_rgb_evaluate as old
from world_reward.data import sha256

policy=old.policy
BASE,IMAGE,BUDGET=old.BASE,old.IMAGE,old.BUDGET
VERTICES,WIDTH,HEIGHT=old.VERTICES,old.WIDTH,old.HEIGHT
TRUTH_KEYS,MANIFEST_SHA=old.TRUTH_KEYS,old.MANIFEST_SHA
RENDER_SHA,RENDER_SOURCE_SHA=old.RENDER_SHA,old.RENDER_SOURCE_SHA
OUT=BASE+"/translation_quality_v2"
STAGE=old.STAGE+"_v2"
FIT_REVISION="2799e8d2c468aff9230b422beea33779e640f325"
FIT_REPORT_SHA="a9108a7409912783926eb4fffe6103a97a4c304d3cb57c8a21d72e1b2664a812"
FAILED_REPORT_SHA="733b7ba34bf8703782a8aa703a6fdc5fed379372b582807962287aeabedca38b"
FAILED_REPORT_BYTES=35320
OLD_SOURCE_SHA="b45e34d071c36724eb5acedf61441e9ba5df3a19924febfb26674a886c67bba0"
PUBLIC_SOURCE_SHA="2e4ac121796923e43f4b006e2ea3f1344adfba3f44766f822a707c05e42ee80d"
POLICY_SOURCE_SHA="d67ddb9ec748a1b6046d43428645e1daf30cff9ded86a7ed5d7157ff9fb002b0"
FIT_SOURCE_SHA="4248a26be67b89267f0780bd9565e60eab1c80f4e5fbefd800d901060f3f74bf"
HUMAN_FACES_SHA="f6748e290ef37fbb6877c4cc5bd7287105db9e98252b0ba170ae9ac3c45eacd6"
regular,require_fields,floating=old.regular,old.require_fields,old.floating
render_source=old.render_source


def fit_source(root):
    return Path(root)/"jobs"/FIT_REVISION/"run_translation_rgb_fit"/"code"/"infra"/"translation_rgb_fit.py"


def source_bindings(root):
    import translation_rgb_public as public
    bindings={"v1_quality":regular(Path(old.__file__),OLD_SOURCE_SHA),
        "public_consumer":regular(Path(public.__file__),PUBLIC_SOURCE_SHA),
        "policy":regular(Path(old.policy.__file__),POLICY_SOURCE_SHA),
        "frozen_fit":regular(fit_source(root),FIT_SOURCE_SHA)}
    bindings["v2_quality"]=regular(Path(__file__))
    return bindings


def previous_failure(root):
    path=Path(root)/BASE/"translation_quality_v1/report.json";receipt=regular(path,FAILED_REPORT_SHA)
    if receipt["bytes"]!=FAILED_REPORT_BYTES:raise ValueError("Original failed quality receipt size differs")
    row=json.loads(path.read_text())
    require_fields(row,dict(stage=old.STAGE,status="fail",phase="public_predictions_audited",producer_revision=FIT_REVISION,
        image_id=IMAGE,network="none",script_sha256=OLD_SOURCE_SHA,error_type="ValueError",error="Truth topology differs",
        predictions_frozen_before_private=True,private_truth_used_for_evaluation_only=True,fit_report_sha256=FIT_REPORT_SHA,
        challenge_inputs_used=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,
        methodological_followup_not_independent_replication=True))
    if any(k in row for k in ("frame_metrics","clip_metrics","decision")):raise ValueError("No previous quality scores or decision allowed")
    if {p.name for p in path.parent.iterdir()}!={"report.json"}:raise ValueError("Original failed quality namespace changed")
    return receipt


def public_fit(root):
    import translation_rgb_public as public
    records,raw,pairs,dw,proxies,lineage=public.public_predictions(root)
    path=Path(root)/public.OUT/"report.json";receipt=regular(path,FIT_REPORT_SHA);report=json.loads(path.read_text())
    if {p.name for p in path.parent.iterdir()}!={"report.json","candidates"}:raise ValueError("Exact frozen translation inventory required")
    require_fields(report,dict(stage=public.STAGE,status="pass",phase="complete",device="cpu",network="none",
        producer_revision=FIT_REVISION,image_id=IMAGE,
        script_sha256=regular(fit_source(root))["sha256"],source_helpers=public.helper_identities(fit_source(root)),lineage=lineage,
        optimizer=public.OPTIMIZER,private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,
        hand_labeled_test=False,oracle_modes=[],object_proxies_reused=True,proxies_recomputed=False,
        all_candidates_frozen=True,inputs_unchanged=True,native_controls_unchanged=True,gpu_used=False,Torch_imported=False,
        native_forward_calls=0,shared_identity_constant=True,body_hands_fixed=True,camera_fixed=True,
        methodological_prior_cohort_reuse=True,initial_jacobian_includes_priors=False,
        training_COCO_indices=list(policy.TRAIN_COCO),training_MHR_indices=list(policy.TRAIN_MHR),full_HOI_verified=False,
        accuracy_verified=False,quality_verified=False,adoption_authorized=False,frames=15,all_cases_retained=True))
    counters=report.get("counters")
    require_fields(counters,dict(evaluated_states=900,adam_updates=885,native_forward_calls=0,proxy_rasters=0))
    total_rows=counters.get("initial_jacobian_rows")
    if type(total_rows)is not int or not 180<=total_rows<=300:raise ValueError("Complete original observation coverage required")
    candidates=public.frozen_candidates(root,records,report.get("candidate_outputs"),raw,pairs)
    if not all(len(x)==15 for x in (records,raw,pairs,dw,proxies,candidates)):raise ValueError("All15 frames required")
    rows=report.get("fit_records")
    if not isinstance(rows,list)or len(rows)!=15:raise ValueError("All15 evaluated schedules required")
    observed=0
    for record,row,candidate,observation,original,pair in zip(records,rows,candidates,dw,raw,pairs):
        require_fields(row,dict(file=record["file"]))
        observed+=public.validate_schedule(row,original,pair,observation,candidate)
    if observed!=total_rows:raise ValueError("Jacobian row coverage differs")
    frozen=[receipt,previous_failure(root),*lineage["frozen_files"]]
    for row in report["candidate_outputs"]:
        identity=regular(Path(root)/public.OUT/row["artifact"],row["sha256"])
        if identity["bytes"]!=row["bytes"]:raise ValueError("Candidate byte count differs")
        frozen.append(identity)
    regular(render_source(root,"keypoint_rgb_render.py"),RENDER_SOURCE_SHA)
    for row in frozen:
        if regular(row["path"],row["sha256"])["bytes"]!=row["bytes"]:raise ValueError("Public lineage changed")
    return records,pairs,candidates,proxies,dw,report,frozen


def validate_truth(data, record, faces):
    if set(data) != TRUTH_KEYS: raise ValueError("Exactly9 foreground-only truth fields required")
    floating(data["human_vertices_camera_m"], (VERTICES, 3), "human")
    floating(data["object_vertices_camera_m"], (194, 3), "object")
    hf, of = data["human_faces"], data["object_faces"]
    if type(faces) is not np.ndarray or faces.dtype!=np.int64 or faces.shape!=(36874,3):raise ValueError("Original baseline int64 topology required")
    if (type(hf) is not np.ndarray or hf.dtype != np.int32 or hf.shape != (36874, 3) or not np.array_equal(hf, faces)
            or np.any(hf < 0) or np.any(hf >= VERTICES) or type(of) is not np.ndarray or of.dtype != np.int64
            or of.shape != (384, 3) or np.any(of < 0) or np.any(of >= 194)
            or np.any(of[:, 0] == of[:, 1]) or np.any(of[:, 1] == of[:, 2]) or np.any(of[:, 0] == of[:, 2])):
        raise ValueError("Truth topology differs")
    for name in ("clip_index", "frame_index"):
        value = data[name]
        if type(value) is not np.ndarray or value.dtype != np.int64 or value.shape != () or value.item() != record[name]:
            raise ValueError("All original private frame identities required")
    K = data["camera_K"]
    if type(K) is not np.ndarray or K.dtype != np.float64 or not np.array_equal(K, [[1280.,0.,512.],[0.,1280.,384.],[0.,0.,1.]]):
        raise ValueError("Preregistered isolated fixed camera differs")
    depth, ids = data["scene_depth_m"], data["visible_face_indices"]
    if (type(depth) is not np.ndarray or depth.dtype != np.float32 or depth.shape != (HEIGHT, WIDTH)
            or type(ids) is not np.ndarray or ids.dtype != np.int64 or ids.shape != depth.shape
            or np.any(ids < -1) or np.any(ids >= len(hf)+len(of)) or not np.isnan(depth[ids < 0]).all()
            or not np.isfinite(depth[ids >= 0]).all() or np.any(depth[ids >= 0] <= 0)
            or np.count_nonzero((ids >= 0) & (ids < len(hf))) < 64 or np.count_nonzero(ids >= len(hf)) < 64):
        raise ValueError("Invalid foreground truth")

    if hashlib.sha256(hf.tobytes()).hexdigest()!=HUMAN_FACES_SHA:raise ValueError("Original int32 face bytes changed")


def run(root,report,persist=lambda:None):
    if "torch" in sys.modules:raise ValueError("CPU quality must not import Torch")
    bindings=source_bindings(root)
    records,pairs,candidates,proxies,dw,producer,frozen=public_fit(root)
    frozen.extend(bindings.values())
    report.update(phase="public_predictions_audited",predictions_frozen_before_private=True,fit_report_sha256=FIT_REPORT_SHA,
        previous_failed_quality_sha256=FAILED_REPORT_SHA,public_lineage=producer["lineage"],source_bindings=bindings,
        native_truth_human_faces_dtype="int32",baseline_human_faces_dtype="int64",native_truth_human_faces_sha256=HUMAN_FACES_SHA,
        dtype_abi_repair_only=True,optimizer_rerun=False,metric_or_threshold_changed=False);persist()
    report["private_truth_used_for_evaluation_only"]=True
    private,cases=old.private_truth(root,records,frozen);scores=[]
    for record,pair,candidate,proxy,observation,case in zip(records,pairs,candidates,proxies,dw,cases):
        require_fields(case,dict(file=record["file"],rgb_sha256=record["sha256"],clip_index=record["clip_index"],frame_index=record["frame_index"]))
        path=private/(Path(record["file"]).stem+".npz");frozen.append(regular(path,case.get("truth_sha256")))
        with np.load(path,allow_pickle=False)as saved:truth={k:saved[k]for k in saved.files}
        validate_truth(truth,record,pair["human_faces"]);median,count=old.visible_object_median(truth)
        values=old.frame_metrics(dict(raw=pair["raw_vertices_camera_m"],baseline=pair["shared_vertices_camera_m"],fitted=candidate["vertices_camera_m"]),
            truth["human_vertices_camera_m"],proxy["object_points_camera_m"],median,pair["hand_mask_left"],pair["hand_mask_right"])
        scores.append(dict(clip_index=record["clip_index"],frame_index=record["frame_index"],true_visible_object_pixels=count,
            heldout_RGB_diagnostics=old.heldout_diagnostics(pair,candidate,observation),**values))
    clips,decision=old.aggregate(scores)
    for row in frozen:
        if regular(row["path"],row["sha256"])["bytes"]!=row["bytes"]:raise ValueError("Public/private bytes changed")
    import translation_rgb_public as public
    if public.helper_identities(fit_source(root))!=producer["source_helpers"]or source_bindings(root)!=bindings:
        raise ValueError("Frozen quality/public/source bindings changed")
    if "torch" in sys.modules:raise ValueError("CPU quality must not import Torch")
    report.update(status="pass",phase="complete",frame_metrics=scores,clip_metrics=clips,decision=decision,all_frames_scored=True,
        all_cases_retained=True,no_gt_alignment=True,wrists_or_joint_truth_used=False,private_render_report_sha256=RENDER_SHA,
        methodological_followup_not_independent_replication=True,private_optimizer_or_model_calls=0)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);out=root/OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if(platform.system()!="Linux"or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}or os.geteuid()!=1000
        or root!=Path("/srv/scenesmith/world-reward")or out.resolve()!=out.absolute()or not out.is_dir()or any(out.iterdir())
        or not re.fullmatch("[0-9a-f]{40}",revision)or os.environ.get("WR_IMAGE_ID")!=IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES")!=""):
        raise ValueError("Fresh canonical offline CPU quality output required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,fit_producer_revision=FIT_REVISION,
        image_id=IMAGE,network="none",script_sha256=sha256(Path(__file__)),budget_seconds=BUDGET,
        private_truth_used_for_evaluation_only=False,predictions_frozen_before_private=False,challenge_inputs_used=False,
        accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,real_domain_verified=False,
        rigid_object_mesh_or_contact_verified=False,methodological_followup_not_independent_replication=True,
        dtype_abi_repair_only=True,optimizer_rerun=False,metric_or_threshold_changed=False)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n")
            stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Translation v2 quality exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
