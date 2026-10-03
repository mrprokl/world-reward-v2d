"""Unchanged historical paired root metrics, reusable without old stage drivers.

The seven functions below retain exact AST/numerics from the frozen own D96
quality source. Camera-space PVE and hand-centroid vectors use no GT alignment.
"""
import re
from pathlib import Path
import numpy as np
from world_reward.data import sha256
from world_reward import root_refit as policy

CLIPS,FRAMES,VERTICES,WIDTH,HEIGHT=3,5,18439,1024,768
TRUTH_KEYS={"human_vertices_camera_m","human_faces","object_vertices_camera_m","object_faces",
            "camera_K","scene_depth_m","visible_face_indices","clip_index","frame_index"}
LEGACY_SOURCE_SHA="5124edadfeffa6891af1167b2d51b896d96ed9ec005fcf7c8bc67afb694af6e5"


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
