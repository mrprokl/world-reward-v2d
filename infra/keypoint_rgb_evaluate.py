"""Frozen D96 camera PVE/hand-centroid quality; no fitting or adoption."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import time

import numpy as np
from world_reward.data import sha256
from world_reward import root_refit as policy

BASE = "validation/keypoint_rgb_v1"
OUT = BASE+"/quality_v1"
STAGE = "private_keypoint_rgb_paired_root_refit_quality"
CLIPS, FRAMES, VERTICES, WIDTH, HEIGHT, BUDGET = 3, 5, 18439, 1024, 768, 120
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
RENDER_SHA = "2407c53871f4b7f8e16d1e3420416a65b93bffcb0ea22cafb14b963f4206e158"
RENDER_REVISION = "42f457fc46fbeb814befb48b8104403b06922f63"
RENDER_SOURCE_SHA = "8fc03c813815a8e3fc01cab0e7b3a7d54b255692d25acad0df9814b4afd5c185"
MANIFEST_SHA = "082549b5a1f8a4a7d687bc17ec6847f3628d6d4230186053951caa2454d2979d"
TRUTH_KEYS = {"human_vertices_camera_m", "human_faces", "object_vertices_camera_m", "object_faces",
              "camera_K", "scene_depth_m", "visible_face_indices", "clip_index", "frame_index"}


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


def producer_source(root, name):
    if name == "keypoint_rgb_fit.py":
        revision=os.environ.get("WR_CODE_REVISION",""); stem="run_keypoint_rgb_fit"
        if not re.fullmatch("[0-9a-f]{40}",revision):raise ValueError("Exact current fit revision required")
    elif name in ("keypoint_rgb_render.py","identity_rgb_render.py","joint_rgb_render.py"):
        revision=RENDER_REVISION;stem="run_keypoint_rgb_prepare"
    else:raise ValueError("Only bound hash-only producer sources accepted")
    return Path(root)/"jobs"/revision/stem/"code"/"infra"/name


def public_fit(root):
    import keypoint_rgb_public as fit
    records, raw, pairs, dw, lineage = fit.public_predictions(root)
    path = Path(root)/BASE/"root_fit_v1/report.json"; identity = regular(path)
    producer = json.loads(path.read_text())
    require_fields(producer, {"stage": fit.STAGE, "status": "pass", "phase": "complete", "network": "none",
        "image_id": IMAGE, "private_truth_read": False, "ground_truth_used": False, "challenge_inputs_used": False,
        "hand_labeled_test": False, "oracle_modes": [], "inputs_unchanged": True,
        "shared_identity_constant": True, "body_hands_fixed": True, "camera_fixed": True,
        "object_proxies_frozen_before_fit": True, "all_candidates_frozen": True, "native_arrays_verified": True,
        "frames": 15, "all_cases_retained": True, "quality_verified": False, "accuracy_verified": False,
        "adoption_authorized": False, "full_HOI_verified": False,
        "training_COCO_indices": list(policy.TRAIN_COCO), "training_MHR_indices": list(policy.TRAIN_MHR),
        "initial_jacobian_includes_priors": False, "optimizer": {"evaluated_states_per_frame":60,"updates_per_frame":59,
            "learning_rate":.01,"betas":[.9,.999],"eps":1e-8},
        "script_sha256": regular(producer_source(root,"keypoint_rgb_fit.py"))["sha256"],
        "source_helpers": fit.helper_identities(producer_source(root,"keypoint_rgb_fit.py")), "lineage": lineage})
    if (not re.fullmatch("[0-9a-f]{40}", str(producer.get("producer_revision", "")))
            or producer["producer_revision"] != os.environ.get("WR_CODE_REVISION")):
        raise ValueError("Fit revision differs")
    require_fields(producer.get("counters"), {"objective_native_heads":900,"final_native_heads":15,"proxy_rasters":15,"adam_updates":885})
    jacobian_rows = producer["counters"].get("initial_jacobian_rows")
    if type(jacobian_rows) is not int or not 180 <= jacobian_rows <= 300:
        raise ValueError("Invalid Jacobian coverage")
    candidates = fit.frozen_candidates(root, records, producer.get("candidate_outputs"), raw, pairs)
    proxies = fit.frozen_proxies(root, records, producer.get("proxy_outputs"), raw, pairs)
    if not all(len(value) == CLIPS*FRAMES for value in (records, raw, pairs, dw, candidates, proxies)):
        raise ValueError("Incomplete frames")
    fit_rows=producer.get("fit_records")
    if not isinstance(fit_rows,list) or len(fit_rows)!=15:raise ValueError("Incomplete optimization")
    observed_rows=0
    for record,row,candidate,observation in zip(records,fit_rows,candidates,dw):
        require_fields(row,{"file":record["file"],"native_forward_calls":60,"adam_updates":59,
            "heldout_COCO_indices":list(policy.HELDOUT_COCO),"heldout_used_for_fit":False})
        target,indices,valid=policy.training_observations(observation["keypoints"][0],observation["scores"][0])
        require_fields(row,{"training_points":len(target),"training_MHR_indices":indices.tolist(),"training_validity":valid.tolist()})
        parity=row.get("initial_parity")
        if not isinstance(parity,dict) or set(parity)!={"vertices","keypoints","joints","controls"}:
            raise ValueError("Missing initial parity")
        errors=floating(list(parity.values()),(4,),"initial native errors")
        if np.any(errors<0) or np.any(errors>1e-5):raise ValueError("Initial native parity failed")
        jac=row.get("initial_observation_jacobian"); observed_rows+=2*len(target)
        require_fields(jac,{"observation_rows":2*len(target),"prior_rows_included":False})
        singular=floating(jac.get("singular_values"),(6,),"observation singular values")
        ratio=singular[-1]/singular[0] if singular[0]>0 else 0.
        if np.any(singular<0) or np.any(np.diff(singular)>0) or ratio<1e-5 or jac.get("minimum_maximum_ratio")!=ratio:
            raise ValueError("Invalid observation Jacobian")
        losses=floating(row.get("evaluated_losses"),(60,),"evaluated objectives");best=policy.best_evaluated(losses)
        require_fields(row,{"best_evaluated_index":best,"selected_total_objective":float(losses[best])})
        delta=floating(row.get("selected_physical_delta"),(6,),"selected update").astype(np.float32)
        if not np.array_equal(delta,candidate["physical_delta"]):raise ValueError("Evaluated selected candidate changed")
    if observed_rows!=jacobian_rows:raise ValueError("Jacobian coverage differs")
    if regular(producer_source(root,"keypoint_rgb_render.py"))["sha256"]!=RENDER_SOURCE_SHA:
        raise ValueError("Renderer source differs")
    frozen = [identity, *lineage["frozen_files"]]
    for key in ("candidate_outputs", "proxy_outputs"):
        for row in producer[key]:
            artifact = Path(root)/BASE/"root_fit_v1"/row["artifact"]
            if regular(artifact, row["sha256"])["bytes"] != row["bytes"]:
                raise ValueError("Candidate/proxy frozen byte count differs")
            frozen.append(regular(artifact, row["sha256"]))
    for row in frozen:
        if regular(row["path"], row["sha256"])["bytes"] != row["bytes"]: raise ValueError("Public lineage bytes differ")
    return records, pairs, candidates, proxies, dw, producer, frozen


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


def aggregate(scores):
    if len(scores) != CLIPS*FRAMES: raise ValueError("All15 paired scores required; no abstention drop")
    clips = []
    for index, row in enumerate(scores):
        require_fields(row, {"clip_index": index//FRAMES, "frame_index": index%FRAMES})
    for clip in range(CLIPS):
        rows = scores[clip*FRAMES:(clip+1)*FRAMES]
        clips.append({"clip_index": clip, **{mode:{"human_pve_cm":float(np.mean([r[mode]["human_pve_cm"] for r in rows])),
            "per_hand_relative_vector_cm":np.mean([r[mode]["per_hand_relative_vector_cm"] for r in rows],axis=0).tolist()}
            for mode in ("raw","baseline","fitted")}})
    human = np.array([[r[m]["human_pve_cm"] for m in ("baseline","fitted")] for r in clips])
    hands = np.array([[r[m]["per_hand_relative_vector_cm"] for m in ("baseline","fitted")] for r in clips])
    return clips, policy.quality_decision(human, hands)


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


def run(root, report, persist=lambda:None):
    records, pairs, candidates, proxies, dw, producer, frozen = public_fit(root)
    report.update(phase="public_predictions_audited", predictions_frozen_before_private=True,
                  fit_report_sha256=frozen[0]["sha256"], public_lineage=producer["lineage"]); persist()
    report["private_truth_used_for_evaluation_only"] = True
    private = Path(root)/BASE/"eval_private"; path = private/"render-report.json"; identity = regular(path, RENDER_SHA)
    render = json.loads(path.read_text())
    require_fields(render, {
        "stage":"own_fresh_keypoint_rgb_render","status":"pass","phase":"complete","frames":15,"code_revision":RENDER_REVISION,
        "script_sha256":RENDER_SOURCE_SHA,"image_id":IMAGE})
    require_fields(render, {"actual_MHR_reference_used":True,"actual_reference_forward_calls":2,
        "challenge_inputs_used":False,"synthetic_truth_used_for_rendering_only":True,"inference_performed":False,
        "predictions_performed":False,"all_truth_private":True,"identity_clip_constant":True,"true_camera_private":True,
        "scene_depth_scope":"human/object foreground only; background IDs=-1/depth=NaN"})
    require_fields(render, {"accuracy_verified":False,"quality_verified":False,"photorealism_verified":False,
        "public_manifest_sha256":MANIFEST_SHA,"public_manifest_bytes":2199,"truth_keys":sorted(TRUTH_KEYS),
        "model_sha256":"352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"})
    render_source=producer_source(root,"keypoint_rgb_render.py")
    if regular(render_source)["sha256"] != RENDER_SOURCE_SHA: raise ValueError("Actual immutable renderer source changed")
    helpers={k:sha256(producer_source(root,name) if k in ("primitives","joint") else Path(__file__).with_name(name)) for k,name in {
        "render":"hand_synthetic_render.py","primitives":"identity_rgb_render.py",
        "joint":"joint_rgb_render.py","camera":"camera_render.py"}.items()}
    require_fields(render,{"helper_source_sha256":helpers})
    frozen.append(identity)
    semantic_path = private/"semantic-report.json"; semantic_id = regular(semantic_path,render.get("semantic_report_sha256"))
    semantic = json.loads(semantic_path.read_text())
    import hand_synthetic_render as regions
    regions.require_semantic_report(semantic)
    if semantic.get("source_image_id") != IMAGE: raise ValueError("Renderer/reference semantic image differs")
    frozen.append(semantic_id)
    rig_path = private/"rig.npz"; rig_id = regular(rig_path,render.get("rig_sha256"))
    with np.load(rig_path,allow_pickle=False) as saved: rig={k:saved[k] for k in saved.files}
    if set(rig)!={"controls","shape45","parameter_limits"}: raise ValueError("Declared own manufacturing rig required")
    controls=floating(rig["controls"],(15,204),"manufacturing controls")
    shapes=floating(rig["shape45"],(15,45),"manufacturing shapes"); bounds=rig["parameter_limits"]
    names=render.get("parameter_names")
    if (rig["controls"].dtype!=np.float32 or rig["shape45"].dtype!=np.float32 or type(bounds) is not np.ndarray
            or bounds.dtype.kind!="f" or bounds.shape!=(249,2) or np.isnan(bounds).any() or np.any(bounds[:,0]>bounds[:,1])
            or not isinstance(names,list) or len(names)!=249 or len(set(names))!=249 or any(not isinstance(n,str) or not n for n in names)
            or np.any(np.c_[controls,shapes]<bounds[:,0]) or np.any(np.c_[controls,shapes]>bounds[:,1])):
        raise ValueError("Invalid manufacturing rig")
    for clip in range(3):
        for index in range(clip*5,(clip+1)*5):
            if shapes[index].tobytes()!=shapes[clip*5].tobytes() or controls[index,136:].tobytes()!=controls[clip*5,136:].tobytes():
                raise ValueError("Identity changed")
    frozen.append(rig_id); cases=render.get("cases")
    if not isinstance(cases,list) or len(cases)!=15: raise ValueError("Every private original case required")
    expected={"render-report.json","semantic-report.json","rig.npz",*[Path(r["file"]).stem+".npz" for r in records]}
    if {p.name for p in private.iterdir()}!=expected: raise ValueError("Exact15private truth inventory required")
    scores=[]
    for record,pair,candidate,proxy,observation,case in zip(records,pairs,candidates,proxies,dw,cases):
        require_fields(case,{"file":record["file"],"rgb_sha256":record["sha256"],"clip_index":record["clip_index"],"frame_index":record["frame_index"]})
        path=private/(Path(record["file"]).stem+".npz"); identity=regular(path,case.get("truth_sha256"))
        with np.load(path,allow_pickle=False) as saved: truth={k:saved[k] for k in saved.files}
        validate_truth(truth,record,pair["human_faces"]); median,count=visible_object_median(truth)
        metrics=frame_metrics({"raw":pair["raw_vertices_camera_m"],"baseline":pair["shared_vertices_camera_m"],
            "fitted":candidate["vertices_camera_m"]},truth["human_vertices_camera_m"],proxy["object_points_camera_m"],median,
            pair["hand_mask_left"],pair["hand_mask_right"])
        scores.append({"clip_index":record["clip_index"],"frame_index":record["frame_index"],"true_visible_object_pixels":count,
                       "heldout_RGB_diagnostics":heldout_diagnostics(pair,candidate,observation),**metrics})
        frozen.append(identity)
    clips, decision=aggregate(scores)
    for identity in frozen:
        if regular(identity["path"],identity["sha256"])["bytes"]!=identity["bytes"]: raise ValueError("Artifact changed")
    import keypoint_rgb_public as fit
    if fit.helper_identities(producer_source(root,"keypoint_rgb_fit.py"))!=producer["source_helpers"]:
        raise ValueError("Helper changed")
    report.update(status="pass",phase="complete",frame_metrics=scores,clip_metrics=clips,decision=decision,
        private_render_report_sha256=RENDER_SHA,all_frames_scored=True,all_cases_retained=True,no_gt_alignment=True,
        diagnostic_centering_not_primary_alignment=True,wrists_or_joint_truth_used=False)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__,allow_abbrev=False).parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);output=root/OUT;revision=os.environ.get("WR_CODE_REVISION","")
    if (platform.system()!="Linux" or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"} or os.geteuid()!=1000
            or root!=Path("/srv/scenesmith/world-reward") or output.resolve()!=output.absolute() or not output.is_dir() or any(output.iterdir())
            or not re.fullmatch("[0-9a-f]{40}",revision) or os.environ.get("WR_IMAGE_ID")!=IMAGE
            or os.environ.get("CUDA_VISIBLE_DEVICES")!=""):
        raise ValueError("Fresh canonical offline CPU-only quality output required")
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=IMAGE,
        script_sha256=sha256(Path(__file__)),network="none",budget_seconds=BUDGET,private_truth_used_for_evaluation_only=False,
        predictions_frozen_before_private=False,challenge_inputs_used=False,accuracy_verified=False,adoption_authorized=False,
        full_HOI_verified=False,real_domain_verified=False,rigid_object_mesh_or_contact_verified=False)
    started=time.perf_counter();path=output/"report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Private paired root-refit quality exceeded120s")
        alarm=signal.signal(signal.SIGALRM,expired);term=signal.signal(signal.SIGTERM,expired);signal.alarm(BUDGET)
        try:persist();run(root,report,persist)
        except BaseException as error:report.update(status="fail",error_type=type(error).__name__,error=str(error));raise
        finally:
            signal.alarm(0);signal.signal(signal.SIGALRM,alarm);signal.signal(signal.SIGTERM,term);persist();path.chmod(0o444)


if __name__=="__main__":main()
