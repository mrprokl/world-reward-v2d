"""H97 CPU public evidence and external-translation-only artifact firewall."""
import json
from pathlib import Path

import numpy as np
import keypoint_rgb_public as previous
from world_reward import translation_refit as policy
from world_reward.data import sha256

native=previous.native
BASE=previous.BASE
OUT=BASE+"/translation_fit_v1"
STAGE="public_keypoint_rgb_external_translation_refit"
FAILED_SHA="aacb82e7c48b5b28ec26f1eb3bb88194c76fbb3c0ce07cc54d6032adb25412d2"
FAILED_REV="52f35c0b0df9a1448adf16c1cff796007fb41198"
FAILED_SCRIPT="d5146d1f89a0fbbd0e4614abd8896a9e13679bd4307535a3cea0b2f46a84583b"
OPTIMIZER=dict(evaluated_states_per_frame=60,updates_per_frame=59,learning_rate=.01,betas=[.9,.999],eps=1e-8)
CANDIDATE_KEYS=previous.CANDIDATE_KEYS


def helper_identities(fit_source=None):
    source=Path(__file__).with_name("translation_rgb_fit.py")if fit_source is None else Path(fit_source)
    return previous.baseline.helper_identities()|dict(previous_public=sha256(Path(previous.__file__)),
        public_consumer=sha256(Path(__file__)),fit=native.regular(source)["sha256"],translation_refit=sha256(Path(policy.__file__)))


def failed_fit(root):
    path=Path(root)/BASE/"root_fit_v1/report.json";receipt=native.regular(path,FAILED_SHA,immutable=True)
    row=json.loads(path.read_text())
    previous.require_fields(row,dict(stage=previous.STAGE,status="fail",phase="native_root_optimization",producer_revision=FAILED_REV,
        script_sha256=FAILED_SCRIPT,image_id=native.IMAGE_ID,network="none",device="cuda",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],object_proxies_frozen_before_fit=True,
        all_candidates_frozen=False,quality_verified=False,accuracy_verified=False,adoption_authorized=False,
        error_type="ValueError",error="Complete249 native bounds/rootEuler mapping violated; no clipping"))
    previous.require_fields(row.get("counters"),dict(proxy_rasters=15,objective_native_heads=0,final_native_heads=0,adam_updates=0,initial_jacobian_rows=0))
    if row.get("candidate_outputs")!=[]or len(row.get("proxy_outputs",[]))!=15:
        raise ValueError("Original stopped root fit with all15proxies required")
    if any((path.parent/"candidates").iterdir())or{p.name for p in path.parent.iterdir()}!={"report.json","candidates","proxies"}:
        raise ValueError("No hidden failed-run prediction or overwritten source")
    return row,receipt


def public_predictions(root):
    records,raw,pairs,dw,lineage=previous.public_predictions(root)
    failed,receipt=failed_fit(root)
    proxies=previous.frozen_proxies(root,records,failed["proxy_outputs"],raw,pairs)
    files=[receipt]
    files.extend(native.regular(Path(root)/previous.OUT/r["artifact"],r["sha256"],immutable=True)for r in failed["proxy_outputs"])
    return records,raw,pairs,dw,proxies,lineage|dict(failed_fit_report_sha256=FAILED_SHA,
        reused_object_proxy_outputs=failed["proxy_outputs"],frozen_files=lineage["frozen_files"]+files)


def translated_geometry(pair,original_translation,new_translation):
    if np.ma.isMaskedArray(original_translation)or np.ma.isMaskedArray(new_translation):raise ValueError("Hidden translation validity forbidden")
    delta=np.asarray(new_translation,dtype=np.float64)-np.asarray(original_translation,dtype=np.float64)
    if delta.shape!=(3,)or not np.isfinite(delta).all():raise ValueError("Finite external camera translation required")
    values=tuple((pair[k].astype(np.float64)+delta).astype(np.float32)for k in
                 ("shared_vertices_camera_m","shared_keypoints_camera_m","shared_joints_camera_m"))
    if any(not np.isfinite(v).all()or np.any(v[:,2]<=0)for v in values):raise ValueError("Full translated geometry must remain finite/positive; no clipping")
    return values


def validate_candidate(data,record,raw,pair):
    if set(data)!=CANDIDATE_KEYS:raise ValueError("Exact translation-only full geometry schema required")
    native.common(data,record)
    for k,shape in native.BLOCKS.items():native.array(data[k],shape,"float32")
    u=native.array(data["latent"],(3,),"float64");d=native.array(data["physical_delta"],(3,),"float64")
    if not np.array_equal(d,policy.bounded_delta(u)):raise ValueError("Frozen evaluated translation latents/delta changed")
    translation=policy.camera_translation(raw["pred_cam_t"],d)
    if data["pred_cam_t"].tobytes()!=translation.astype(np.float32).tobytes():raise ValueError("External camera translation changed")
    for k,expected in (("global_rot",raw["global_rot"]),("body_pose_params",raw["body_pose_params"]),
        ("hand_pose_params",raw["hand_pose_params"]),("expr_params",raw["expr_params"]),("shape_params",pair["shared_shape_params"]),
        ("scale_params",pair["shared_scale_params"]),("mhr_model_params",pair["shared_model_controls"]),
        ("joint_global_rotations",pair["shared_joint_global_rotations"]),("human_faces",pair["human_faces"]),
        ("camera_K",pair["camera_K"]),("hand_mask_left",pair["hand_mask_left"]),("hand_mask_right",pair["hand_mask_right"])):
        if data[k].tobytes()!=expected.tobytes():raise ValueError("No native/body/hand/identity/orientation/scale/camera changes allowed")
    for k,expected,shape in zip(("vertices_camera_m","keypoints_camera_m","joints_camera_m"),
            translated_geometry(pair,raw["pred_cam_t"],translation),((native.VERTICES,3),(308,3),(127,3))):
        value=native.array(data[k],shape,"float32")
        if value.tobytes()!=expected.tobytes()or np.any(value[:,2]<=0):raise ValueError("Rigid translation only, all positive cameraZ; no vertex/depth rescaling")
    previous.baseline.rotations(data["joint_global_rotations"])
    return data


def frozen_candidates(root,records,rows,raw_frames,pairs):
    return native.frozen_rows(Path(root)/OUT,rows,records,"candidates",lambda data,record:
        validate_candidate(data,record,raw_frames[record["clip_index"]*5+record["frame_index"]],pairs[record["clip_index"]*5+record["frame_index"]]))


def validate_schedule(row,raw,pair,observation,candidate):
    """Replay all observed CPU states and59 manual updates before private IO."""
    observed,indices,valid=policy.training_observations(observation["keypoints"][0],observation["scores"][0])
    previous.require_fields(row,dict(training_points=len(observed),training_MHR_indices=indices.tolist(),training_validity=valid.tolist(),
        cpu_evaluations=60,adam_updates=59,heldout_COCO_indices=list(policy.HELDOUT_COCO),heldout_used_for_fit=False))
    latents=native.array(np.asarray(row.get("evaluated_latents")),(60,3),"float64")
    losses=native.array(np.asarray(row.get("evaluated_losses")),(60,),"float64")
    points=pair["shared_keypoints_camera_m"].astype(np.float64)[indices]
    u=np.zeros(3,np.float64);m=u.copy();v=u.copy()
    for i in range(60):
        if not np.array_equal(latents[i],u):raise ValueError("Observed latent differs from exact initial/Adam sequence")
        projected,J=policy.project_and_jacobian(points,raw["pred_cam_t"],u,pair["camera_K"])
        if i==0 and row.get("initial_observation_jacobian")!=policy.jacobian_evidence(J,len(observed)):
            raise ValueError("Actual observation-only analytic Jacobian differs")
        result,g=policy.objective_and_gradient(projected,observed,policy.bounded_delta(u),J,u)
        if losses[i]!=result["total"]:raise ValueError("Observed objective differs from exact public replay")
        if i<59:u,m,v=policy.adam_step(u,g,m,v,i+1)
    best=policy.best_evaluated(losses)
    previous.require_fields(row,dict(best_evaluated_index=best,selected_total_objective=float(losses[best]),
        selected_physical_delta=policy.bounded_delta(latents[best]).tolist()))
    if not np.array_equal(candidate["latent"],latents[best]):raise ValueError("Frozen selected candidate was not an evaluated best state")
    return 2*len(observed)
