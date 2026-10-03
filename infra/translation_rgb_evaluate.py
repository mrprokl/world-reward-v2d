"""H97 paired translation quality on the reused D96 cohort; no adoption."""
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
from world_reward.data import sha256
from world_reward import translation_refit as policy

BASE = "validation/keypoint_rgb_v1"
CLIPS, FRAMES, VERTICES, WIDTH, HEIGHT, BUDGET = 3, 5, 18439, 1024, 768, 120
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
RENDER_SHA = "2407c53871f4b7f8e16d1e3420416a65b93bffcb0ea22cafb14b963f4206e158"
RENDER_REVISION = "42f457fc46fbeb814befb48b8104403b06922f63"
RENDER_SOURCE_SHA = "8fc03c813815a8e3fc01cab0e7b3a7d54b255692d25acad0df9814b4afd5c185"
MANIFEST_SHA = "082549b5a1f8a4a7d687bc17ec6847f3628d6d4230186053951caa2454d2979d"
TRUTH_KEYS = {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces",
              "camera_K", "scene_depth_m", "visible_face_indices", "clip_index", "frame_index"}


OUT=BASE+"/translation_quality_v1"
STAGE="private_keypoint_rgb_paired_external_translation_quality"


def require_fields(data, expected):
    if not isinstance(data, dict) or any(type(data.get(k)) is not type(v) or data[k] != v for k, v in expected.items()):
        raise ValueError("Exact frozen producer contract required")


def regular(path, expected=None):
    path = Path(path)
    if (path.resolve() != path.absolute() or not path.is_file() or path.stat().st_mode & 0o222
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Canonical read-only regular artifact required")
    actual = sha256(path)
    if expected is not None and (not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected) or actual != expected):
        raise ValueError("Frozen artifact bytes changed")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def floating(value, shape, name):
    if np.ma.isMaskedArray(value): raise ValueError("Hidden validity forbidden: "+name)
    array = np.asarray(value)
    if array.shape != shape or array.dtype not in (np.float32, np.float64) or not np.isfinite(array).all():
        raise ValueError("Complete finite floating geometry required: "+name)
    return array.astype(np.float64)


def validate_truth(data, record, faces):
    if set(data) != TRUTH_KEYS: raise ValueError("Exactly9 foreground-only truth fields required")
    floating(data["human_vertices_camera_m"], (VERTICES, 3), "human")
    floating(data["object_vertices_camera_m"], (194, 3), "object")
    hf, of = data["human_faces"], data["object_faces"]
    if (type(hf) is not np.ndarray or hf.dtype != np.int64 or hf.shape != (36874, 3) or not np.array_equal(hf, faces)
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


def visible_object_median(truth):
    selected = truth["visible_face_indices"] >= len(truth["human_faces"])
    y, x = np.nonzero(selected); z = truth["scene_depth_m"][selected].astype(np.float64); K = truth["camera_K"]
    points = np.c_[((x+.5-K[0,2])/K[0,0])*z, ((y+.5-K[1,2])/K[1,1])*z, z]
    return np.median(points, axis=0), int(len(points))


def frame_metrics(humans, truth, object_points, object_truth_median, left, right):
    target = floating(truth, (VERTICES, 3), "truth")
    if np.ma.isMaskedArray(object_points): raise ValueError("Hidden object validity forbidden")
    points = np.asarray(object_points)
    if points.ndim != 2 or points.shape[1:] != (3,) or not 32 <= len(points) <= 8192:
        raise ValueError("Fixed public visible-object proxy required")
    points = floating(points, points.shape, "public object proxy")
    if np.any(points[:, 2] <= 0): raise ValueError("Positive camera object depth required")
    q = floating(object_truth_median, (3,), "true visible-object median")
    masks = (left, right)
    if (any(type(m) is not np.ndarray or m.dtype != np.bool_ or m.shape != (VERTICES,) or m.sum() < 50 for m in masks)
            or np.any(left & right)):
        raise ValueError("Actual fixed disjoint LBS hand regions required")
    p = np.median(points, axis=0); result = {}
    for name in ("raw", "baseline", "fitted"):
        v = floating(humans[name], (VERTICES,3), name); difference = v-target; centroid = v.mean(0)-target.mean(0)
        relative = [float(np.linalg.norm((p-v[m].mean(0))-(q-target[m].mean(0)))*100) for m in masks]
        result[name] = {"human_pve_cm": float(np.linalg.norm(difference,axis=1).mean()*100),
            "per_hand_relative_vector_cm": relative, "centroid_error_xyz_cm": (centroid*100).tolist(),
            "centroid_error_cm": float(np.linalg.norm(centroid)*100),
            "centered_human_pve_cm": float(np.linalg.norm(difference-centroid,axis=1).mean()*100),
            "per_hand_vertex_pve_cm": [float(np.linalg.norm(difference[m],axis=1).mean()*100) for m in masks]}
    return result


def heldout_diagnostics(pair, candidate, observation):
    coco=np.array(policy.HELDOUT_COCO); mapped=np.array(policy.COCO_TO_MHR)[coco]
    xy=np.asarray(observation["keypoints"])[0,coco]; scores=np.asarray(observation["scores"])[0,coco]
    if xy.shape!=(7,2) or scores.shape!=(7,) or not np.isfinite(xy).all() or not np.isfinite(scores).all():
        raise ValueError("Invalid heldout observations")
    valid=scores>0; output={"COCO_indices":coco.tolist(),"positive_score_count":int(valid.sum()),"used_for_fit":False,
                            "private_keypoint_truth_used":False}
    K=pair["camera_K"]
    for name,value in (("baseline",pair["shared_keypoints_camera_m"]),("fitted",candidate["keypoints_camera_m"])):
        points=floating(value,(308,3),name+"308 keypoints")[mapped]
        if np.any(points[:,2]<=0):raise ValueError("Invalid projected depth")
        projected=points[:,:2]/points[:,2:]*np.diag(K)[:2]+K[:2,2]
        output[name+"_mean_pixel_error"]=float(np.linalg.norm(projected[valid]-xy[valid],axis=1).mean()) if valid.any() else None
    return output


def render_source(root,name):
    if name not in ("keypoint_rgb_render.py","identity_rgb_render.py","joint_rgb_render.py"):raise ValueError("Bound renderer source required")
    return Path(root)/"jobs"/RENDER_REVISION/"run_keypoint_rgb_prepare"/"code"/"infra"/name


def fit_source(root):
    revision=os.environ.get("WR_CODE_REVISION","")
    if not re.fullmatch("[0-9a-f]{40}",revision):raise ValueError("Immutable current translation source required")
    return Path(root)/"jobs"/revision/"run_translation_rgb_fit"/"code"/"infra"/"translation_rgb_fit.py"


def public_fit(root):
    import translation_rgb_public as public
    records,raw,pairs,dw,proxies,lineage=public.public_predictions(root)
    path=Path(root)/public.OUT/"report.json";receipt=regular(path);report=json.loads(path.read_text())
    if {p.name for p in path.parent.iterdir()}!={"report.json","candidates"}:raise ValueError("Exact frozen translation inventory required")
    require_fields(report,dict(stage=public.STAGE,status="pass",phase="complete",device="cpu",network="none",
        producer_revision=os.environ.get("WR_CODE_REVISION"),image_id=IMAGE,
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
    frozen=[receipt,*lineage["frozen_files"]]
    for row in report["candidate_outputs"]:
        identity=regular(Path(root)/public.OUT/row["artifact"],row["sha256"])
        if identity["bytes"]!=row["bytes"]:raise ValueError("Candidate byte count differs")
        frozen.append(identity)
    regular(render_source(root,"keypoint_rgb_render.py"),RENDER_SOURCE_SHA)
    for row in frozen:
        if regular(row["path"],row["sha256"])["bytes"]!=row["bytes"]:raise ValueError("Public lineage changed")
    return records,pairs,candidates,proxies,dw,report,frozen


def aggregate(scores):
    if len(scores)!=15:raise ValueError("All15 paired frames required")
    for index,row in enumerate(scores):require_fields(row,dict(clip_index=index//5,frame_index=index%5))
    clips=[]
    for clip in range(3):
        rows=scores[clip*5:(clip+1)*5]
        clips.append(dict(clip_index=clip,**{mode:dict(human_pve_cm=float(np.mean([r[mode]["human_pve_cm"]for r in rows])),
            per_hand_relative_vector_cm=np.mean([r[mode]["per_hand_relative_vector_cm"]for r in rows],axis=0).tolist())
            for mode in ("raw","baseline","fitted")}))
    human=np.array([[r[m]["human_pve_cm"]for m in ("baseline","fitted")]for r in clips])
    hands=np.array([[r[m]["per_hand_relative_vector_cm"]for m in ("baseline","fitted")]for r in clips])
    return clips,policy.quality_decision(human,hands)


def private_truth(root,records,frozen):
    private=Path(root)/BASE/"eval_private";path=private/"render-report.json";receipt=regular(path,RENDER_SHA)
    render=json.loads(path.read_text())
    require_fields(render,dict(stage="own_fresh_keypoint_rgb_render",status="pass",phase="complete",frames=15,
        code_revision=RENDER_REVISION,script_sha256=RENDER_SOURCE_SHA,image_id=IMAGE,
        actual_MHR_reference_used=True,actual_reference_forward_calls=2,challenge_inputs_used=False,
        synthetic_truth_used_for_rendering_only=True,inference_performed=False,predictions_performed=False,
        accuracy_verified=False,quality_verified=False,photorealism_verified=False,all_truth_private=True,identity_clip_constant=True,
        true_camera_private=True,scene_depth_scope="human/object foreground only; background IDs=-1/depth=NaN",
        public_manifest_sha256=MANIFEST_SHA,public_manifest_bytes=2199,truth_keys=sorted(TRUTH_KEYS),
        model_sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"))
    hashes={k:regular(render_source(root,n))["sha256"] if k in ("primitives","joint") else sha256(Path(__file__).with_name(n))
        for k,n in dict(render="hand_synthetic_render.py",primitives="identity_rgb_render.py",joint="joint_rgb_render.py",camera="camera_render.py").items()}
    require_fields(render,dict(helper_source_sha256=hashes));frozen.append(receipt)
    semantic_path=private/"semantic-report.json";frozen.append(regular(semantic_path,render.get("semantic_report_sha256")))
    import hand_synthetic_render as regions
    semantic=json.loads(semantic_path.read_text());regions.require_semantic_report(semantic)
    if semantic.get("source_image_id")!=IMAGE:raise ValueError("Reference semantic image differs")
    rig_path=private/"rig.npz";frozen.append(regular(rig_path,render.get("rig_sha256")))
    with np.load(rig_path,allow_pickle=False)as saved:rig={k:saved[k]for k in saved.files}
    if set(rig)!={"controls","shape45","parameter_limits"}:raise ValueError("Original manufacturing rig required")
    controls=floating(rig["controls"],(15,204),"rig controls");shapes=floating(rig["shape45"],(15,45),"rig shapes")
    limits=rig["parameter_limits"];names=render.get("parameter_names")
    if(type(limits)is not np.ndarray or limits.shape!=(249,2)or limits.dtype.kind!="f"or np.isnan(limits).any()or np.any(limits[:,0]>limits[:,1])
        or rig["controls"].dtype!=np.float32 or rig["shape45"].dtype!=np.float32 or not isinstance(names,list)or len(names)!=249 or len(set(names))!=249
        or np.any(np.c_[controls,shapes]<limits[:,0])or np.any(np.c_[controls,shapes]>limits[:,1])):raise ValueError("Manufacturing ABI changed")
    for clip in range(3):
        for index in range(clip*5,(clip+1)*5):
            if shapes[index].tobytes()!=shapes[clip*5].tobytes()or controls[index,136:].tobytes()!=controls[clip*5,136:].tobytes():raise ValueError("Manufactured identity changed")
    cases=render.get("cases")
    if not isinstance(cases,list)or len(cases)!=15:raise ValueError("All15 private original cases required")
    expected={"render-report.json","semantic-report.json","rig.npz",*[Path(r["file"]).stem+".npz"for r in records]}
    if {p.name for p in private.iterdir()}!=expected:raise ValueError("Private inventory differs")
    return private,cases


def run(root,report,persist=lambda:None):
    if "torch" in sys.modules:raise ValueError("CPU quality must not import Torch")
    own_sources={"quality":sha256(Path(__file__)),"translation_refit":sha256(Path(policy.__file__))}
    records,pairs,candidates,proxies,dw,producer,frozen=public_fit(root)
    report.update(phase="public_predictions_audited",predictions_frozen_before_private=True,fit_report_sha256=frozen[0]["sha256"],
                  public_lineage=producer["lineage"],quality_source_sha256=own_sources);persist()
    # Only after complete public candidates/proxies/source checks.
    report["private_truth_used_for_evaluation_only"]=True;private,cases=private_truth(root,records,frozen);scores=[]
    for record,pair,candidate,proxy,observation,case in zip(records,pairs,candidates,proxies,dw,cases):
        require_fields(case,dict(file=record["file"],rgb_sha256=record["sha256"],clip_index=record["clip_index"],frame_index=record["frame_index"]))
        path=private/(Path(record["file"]).stem+".npz");frozen.append(regular(path,case.get("truth_sha256")))
        with np.load(path,allow_pickle=False)as saved:truth={k:saved[k]for k in saved.files}
        validate_truth(truth,record,pair["human_faces"]);median,count=visible_object_median(truth)
        values=frame_metrics(dict(raw=pair["raw_vertices_camera_m"],baseline=pair["shared_vertices_camera_m"],fitted=candidate["vertices_camera_m"]),
            truth["human_vertices_camera_m"],proxy["object_points_camera_m"],median,pair["hand_mask_left"],pair["hand_mask_right"])
        scores.append(dict(clip_index=record["clip_index"],frame_index=record["frame_index"],true_visible_object_pixels=count,
            heldout_RGB_diagnostics=heldout_diagnostics(pair,candidate,observation),**values))
    clips,decision=aggregate(scores)
    for row in frozen:
        if regular(row["path"],row["sha256"])["bytes"]!=row["bytes"]:raise ValueError("Public/private bytes changed")
    import translation_rgb_public as public
    if public.helper_identities(fit_source(root))!=producer["source_helpers"]:raise ValueError("Public source changed")
    if own_sources!={"quality":sha256(Path(__file__)),"translation_refit":sha256(Path(policy.__file__))}:raise ValueError("Quality source changed")
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
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=IMAGE,network="none",
        script_sha256=sha256(Path(__file__)),budget_seconds=BUDGET,private_truth_used_for_evaluation_only=False,
        predictions_frozen_before_private=False,challenge_inputs_used=False,accuracy_verified=False,adoption_authorized=False,
        full_HOI_verified=False,real_domain_verified=False,rigid_object_mesh_or_contact_verified=False,
        methodological_followup_not_independent_replication=True)
    started=time.perf_counter();path=out/"report.json"
    with path.open("x")as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False);stream.write("\n")
            stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Translation quality exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
