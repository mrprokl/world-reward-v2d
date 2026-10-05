"""Fixed-scale rigid pose initialization from automatic image evidence.

Predicted depth is not GT. Sparse engineering and explicit full-video modes
neither establish accuracy, calibrated scale nor challenge eligibility.
"""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import platform
import sys
import time

from body_smoke import EPISODE, TRACK1_EPISODE_COUNT, _validate_inputs
from camera_render import raster_camera_mesh, silhouette_iou
from world_reward.data import sha256
from world_reward.rigid_alignment import align_observed_points
from world_reward.mesh_geometry import normalize_degenerate_faces
from world_reward.mesh_budget import fit_topology_preserving_budget
from world_reward.pose_selection import select_pose_path


class _QueryFlag(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, False): parser.error('Repeated --query-requalification')
        setattr(namespace, self.dest, True)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=range(TRACK1_EPISODE_COUNT), default=EPISODE)
    parser.add_argument("--full-video", action="store_true")
    parser.add_argument("--mesh-source", choices=('default','volume','conditioned','solid','surface'), default='default')
    parser.add_argument('--query-requalification', action=_QueryFlag, nargs=0, default=False)
    return parser


def _load_solid_mesh(root, episode, input_sha, object_report_path, alignment_path, scale, output, fixed_mesh_path, *, query_requalification=False):
    """Only consume the pinned CPU proposal; no mesh fitting or second scale."""
    from solid_geometry_loader import load, identity, strict_json
    import shutil
    if type(query_requalification) is not bool: raise ValueError('Explicit query profile required')
    options = dict(query_requalification=True) if query_requalification else {}
    pin_path = Path(__file__).resolve().parent.parent / 'configs' / f'solid_mesh_{episode:06d}_pins.json'
    pin_identity = identity(pin_path)  # Missing pins stop before output reservation.
    pins = strict_json(pin_path.read_text())
    values = load(root, episode, input_sha, sha256(object_report_path), sha256(alignment_path), scale, pins=pins, **options)
    if identity(pin_path) != pin_identity:
        raise ValueError('Committed solid mesh pins changed during loading')
    vertices, faces, active_indices, geometry_cleanup, qualified_glb, topology_budget = values
    if ('query_requalification' in topology_budget) != query_requalification:
        raise ValueError('Explicit native query profile differs from frozen CPU receipt')
    source_identity = identity(qualified_glb)
    if not _solid_output_reserved(output):
        output.mkdir(exist_ok=False)
    shutil.copyfile(qualified_glb, fixed_mesh_path)  # Canonical only; vertices are already metric.
    if (sha256(fixed_mesh_path) != source_identity['sha256'] or identity(qualified_glb) != source_identity
            or identity(pin_path) != pin_identity):
        raise ValueError('Solid canonical GLB or committed pins changed during copying')
    topology_budget['committed_pins_sha256'] = pin_identity['sha256']
    return vertices, faces, active_indices, geometry_cleanup, qualified_glb, topology_budget


def _load_surface_mesh(root, episode, input_sha, object_report_path, alignment_path, scale, output, fixed_mesh_path):
    """Inert independently pinned surface; no CPU solver in the GPU process."""
    from surface_geometry_loader import load, identity, strict_json
    import shutil
    pin_path = Path(__file__).resolve().parent.parent / 'configs' / f'surface_mesh_{episode:06d}_pins.json'
    pin = identity(pin_path)
    values = load(root, episode, input_sha, sha256(object_report_path), sha256(alignment_path),
                  scale, pins=strict_json(pin_path.read_bytes()))
    if identity(pin_path) != pin:
        raise ValueError('Committed surface pins changed during loading')
    vertices, faces, active, cleanup, canonical, receipt = values
    receipt['committed_pins_sha256'] = pin['sha256']
    original = identity(canonical)
    if not _solid_output_reserved(output):
        output.mkdir(exist_ok=False)
    shutil.copyfile(canonical, fixed_mesh_path)
    if (sha256(fixed_mesh_path) != original['sha256'] or identity(canonical) != original
            or identity(pin_path) != pin):
        raise ValueError('Canonical surface changed during Azure-only copying')
    return vertices, faces, active, cleanup, canonical, receipt


def _solid_output_reserved(output):
    """One empty, wrapper-owned output; never resume a prediction directory."""
    import stat
    value=os.environ.get('WR_POSE_OUTPUT_RESERVED')
    if value is None:
        return False
    if value!='1':
        raise ValueError('Only explicit solid output reservation is supported')
    info=output.lstat()
    if (output.resolve()!=output or any(p.is_symlink() for p in (output,*output.parents))
            or not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o755
            or info.st_uid!=os.getuid() or any(output.iterdir())):
        raise ValueError('Canonical empty owned solid output required; never resume')
    return True


def main() -> None:
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux GPU container with network none")
    args = _argument_parser().parse_args()
    if args.query_requalification and args.mesh_source != 'solid':
        raise ValueError('Query requalification is solid-only')
    if args.mesh_source in ('solid','surface') and not args.full_video:
        raise ValueError('Qualified geometry requires the complete original video')
    root = Path(os.environ.get("WR_ROOT", "/srv/scenesmith/world-reward"))
    inputs = _validate_inputs(root, episode_index=args.episode)
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation
    import trimesh
    base = root / f"outputs/episode_{args.episode:06d}"
    output = base / ("object_pose_full" if args.full_video else "object_pose_smoke")
    if args.mesh_source=='conditioned':
        output = output.with_name(output.name+'_conditioned')
    elif args.mesh_source in ('solid','surface'):
        output = output.with_name(output.name+'_'+args.mesh_source)
    if os.environ.get('WR_POSE_OUTPUT_RESERVED') is not None and args.mesh_source not in ('solid','surface'):
        raise ValueError('Output reservation requires a qualified representation')
    reserved=_solid_output_reserved(output) if args.mesh_source in ('solid','surface') else False
    if not reserved and (output.exists() or output.is_symlink()):
        raise RuntimeError("Frozen object pose smoke already exists")
    object_dir = base / "object_grounded"
    object_report_path, alignment_path = object_dir / "report.json", base / "scale_smoke/report.json"
    report = json.loads(object_report_path.read_text())
    alignment = json.loads(alignment_path.read_text())
    for record, stage in ((report, "sam3d_objects_grounded_fixed_frame"),
                          (alignment, "predicted_human_anchored_moge2_pointmaps")):
        expected = {"stage": stage, "status": "pass", "episode_index": args.episode, "input_track": "track_1",
                    "input_sha256": inputs["video_sha256"], "ground_truth_used": False,
                    "hand_labeled_test": False, "oracle_modes": []}
        if (any(record.get(key) != value for key, value in expected.items())
                or record["ground_truth_used"] is not False or record["hand_labeled_test"] is not False):
            raise RuntimeError("Object/alignment video-only provenance mismatch")
    if report["scale_source"] != "already_human_anchored_MoGe2_no_second_scalar":
        raise RuntimeError("Never apply a MoGe2 scale to an independent MoGe1 object pose")
    if report["pointmap_grounding"]["alignment_report_sha256"] != sha256(alignment_path):
        raise RuntimeError("Object grounding alignment changed")
    for filename, field in (("object.glb", "object_sha256"), ("transform.json", "transform_sha256"),
                            ("intrinsics.json", "intrinsics_sha256")):
        if sha256(object_dir / filename) != report[field]:
            raise RuntimeError("Frozen object mesh/transform/camera changed")
    transform = json.loads((object_dir / "transform.json").read_text())
    if transform != report["transform"]:
        raise RuntimeError("Object transform does not match inference report")
    scale = np.asarray(transform["scale"], dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError("Object scale must be finite and positive")
    if not np.allclose(scale, scale[0], atol=0, rtol=1e-5):
        raise RuntimeError("Anisotropic generated scale needs explicit fixed-mesh baking; no averaging")
    quaternion = transform["rotation"]
    rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
    translation = np.asarray(transform["translation"], dtype=float)
    camera_dict = json.loads((object_dir / "intrinsics.json").read_text())
    camera = np.array([[camera_dict["fx"], 0, camera_dict["cx"]],
                       [0, camera_dict["fy"], camera_dict["cy"]], [0, 0, 1]])
    height, width = camera_dict["height"], camera_dict["width"]
    sys.path.insert(0, str(root / "vendor/v2d_submission_kit"))
    from v2dlb.mesh_budget import budget_mesh
    import inspect
    import importlib.metadata
    budget_source_path = Path(inspect.getfile(budget_mesh))
    started = time.perf_counter()
    # The GLB is canonical; pose is stored separately. Keep exact official budget
    # arrays for final packing and derive only a non-padding mesh for raster QA.
    fixed_mesh_path = output / "object_fixed_canonical.glb"
    if args.mesh_source=='volume':
        from volume_geometry_loader import load
        from volume_mesh_pin_inventory import strict_json, identity
        import shutil
        # Generic sources are independently frozen in committed, readonly code.
        # Only historical episode0 may retain the exact legacy producer path.
        pin_path = Path(__file__).resolve().parent.parent / "configs" / f"volume_mesh_{args.episode:06d}_pins.json"
        pins = None
        pin_identity = None
        if pin_path.exists() or pin_path.is_symlink():
            pin_identity = identity(pin_path)
            if pin_path.stat().st_mode & 0o222:
                raise ValueError("Immutable committed generic volume mesh pins required")
            pins = strict_json(pin_path.read_text())
        vertices,faces,active_indices,geometry_cleanup,qualified_glb,topology_budget=load(
            root,args.episode,inputs['video_sha256'],sha256(object_report_path),sha256(alignment_path),float(scale[0]),pins=pins)
        if pin_identity is not None:
            if identity(pin_path) != pin_identity:
                raise ValueError("Original committed volume pins changed during loading")
            topology_budget['committed_pins_sha256'] = pin_identity['sha256']
        output.mkdir(exist_ok=False)
        shutil.copyfile(qualified_glb,fixed_mesh_path)  # remote only, canonical scale not applied here
    elif args.mesh_source=='conditioned':
        from conditioned_geometry_loader import load, identity, strict_json
        import shutil
        pin_path = Path(__file__).resolve().parent.parent / 'configs' / f'conditioned_mesh_{args.episode:06d}_pins.json'
        pin_identity = identity(pin_path)  # Missing independent CPU pins FAIL; never use default geometry.
        pins = strict_json(pin_path.read_text())
        vertices,faces,active_indices,geometry_cleanup,qualified_glb,topology_budget=load(
            root,args.episode,inputs['video_sha256'],sha256(object_report_path),sha256(alignment_path),float(scale[0]),pins=pins)
        if identity(pin_path) != pin_identity:
            raise ValueError('Committed conditioned mesh pins changed during loading')
        topology_budget['committed_pins_sha256'] = pin_identity['sha256']
        source_identity = identity(qualified_glb)
        output.mkdir(exist_ok=False)
        shutil.copyfile(qualified_glb,fixed_mesh_path)  # canonical only; metric vertices already scaled by CPU
        if sha256(fixed_mesh_path) != source_identity['sha256'] or identity(qualified_glb) != source_identity:
            raise ValueError('Conditioned canonical GLB changed during remote-only copying')
    elif args.mesh_source=='surface':
        vertices,faces,active_indices,geometry_cleanup,qualified_glb,topology_budget=_load_surface_mesh(
            root,args.episode,inputs['video_sha256'],object_report_path,alignment_path,float(scale[0]),output,fixed_mesh_path)
    elif args.mesh_source=='solid':
        vertices,faces,active_indices,geometry_cleanup,qualified_glb,topology_budget=_load_solid_mesh(
            root,args.episode,inputs['video_sha256'],object_report_path,alignment_path,float(scale[0]),output,fixed_mesh_path,
            **(dict(query_requalification=True) if args.query_requalification else {}))
    else:
        output.mkdir(exist_ok=False)
        source_raw = trimesh.load(object_dir / "object.glb", force="mesh", process=False)
        source_mesh = trimesh.Trimesh(source_raw.vertices, source_raw.faces, process=True)
        fixed_mesh, topology_budget = fit_topology_preserving_budget(source_mesh)
        fixed_mesh.export(fixed_mesh_path)
        # Already <=4096 faces/vertices: official packer only welds/pads.
        vertices, faces = budget_mesh(str(fixed_mesh_path), faces=4096, vertices=4096)
        vertices = vertices * scale[0]  # same grounding gauge, applied exactly once
        active_indices, geometry_cleanup = normalize_degenerate_faces(vertices, faces)
    # Official padding is represented by repeated vertex indices. Replacing
    # only numerically collapsed triangles does not change geometric surfaces.
    # Preserve exact 4096 row budget and never drop a real component/cavity.
    inactive = np.ones(len(faces), dtype=bool)
    inactive[active_indices] = False
    if args.mesh_source=='conditioned' and np.any(faces[inactive] != 0):
        raise ValueError('Conditioned route only permits original all-zero official padding')
    if args.mesh_source in ('solid','surface') and np.any(faces[inactive] != 0):
        raise ValueError('Solid route only permits original all-zero official padding')
    faces = faces.copy()
    faces[inactive] = 0
    if args.mesh_source=='surface':
        from world_reward.surface_pose_geometry import compact_surface
        cv,cf,checked_active,_=compact_surface(vertices,faces,
            canonical_vertex_count=topology_budget['canonical_vertices_count'],
            canonical_face_count=topology_budget['canonical_faces_count'])
        if not np.array_equal(checked_active,active_indices):raise ValueError('Original surface face indices changed')
        mesh=trimesh.Trimesh(cv,cf,process=False)
    elif args.mesh_source in ('volume','conditioned','solid'):
        ids,inverse=np.unique(faces[active_indices],return_inverse=True)
        mesh=trimesh.Trimesh(vertices[ids],inverse.reshape(-1,3),process=False)
    else:
        mesh = trimesh.Trimesh(vertices, faces[active_indices], process=True)
    if args.mesh_source!='surface' and (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0):
        raise RuntimeError("Official-budget geometry lost closed oriented volume; do not use for PEN")
    sampled, _ = trimesh.sample.sample_surface(mesh, 8192, seed=0)
    pointmaps = {record["frame_index"]: record for record in alignment["pointmaps"]}
    evidence = {record["frame_index"]: record for record in alignment["human_evidence"]}
    if (len(pointmaps) != len(alignment["pointmaps"]) or len(evidence) != len(alignment["human_evidence"])
            or sorted(pointmaps) != inputs["indices"] or sorted(evidence) != inputs["indices"]):
        raise RuntimeError("Sparse evidence must retain exact original frame indices")
    full_depth_report_path = base / "depth_full/report.json"
    if args.full_video:
        full_depth = json.loads(full_depth_report_path.read_text())
        full_body = json.loads((base / "body_full/report.json").read_text())
        required = {"stage": "monocular_moge2_full_video", "status": "pass", "episode_index": args.episode, "input_track": "track_1",
                    "input_sha256": inputs["video_sha256"], "ground_truth_used": False, "hand_labeled_test": False,
                    "oracle_modes": [], "total_video_frames": inputs["total_frames"]}
        if (any(full_depth.get(key) != value for key, value in required.items())
                or full_depth["ground_truth_used"] is not False or full_depth["hand_labeled_test"] is not False):
            raise RuntimeError("Full depth original-video provenance mismatch")
        body_required={**required,'stage':'sam3d_body_full_video_initializer'}
        if (any(type(full_body.get(key)) is not type(value) or full_body[key]!=value for key,value in body_required.items())):
            raise RuntimeError("Full body original-video provenance mismatch")
        pointmaps = {record["frame_index"]: record for record in full_depth["frames"]}
        body_frames = {record["frame_index"]: record for record in full_body["frames"]}
        indices = list(range(inputs["total_frames"]))
        if (len(pointmaps) != len(full_depth["frames"]) or sorted(pointmaps) != indices
                or len(body_frames) != len(full_body['frames']) or sorted(body_frames) != indices or full_body["input_sha256"] != inputs["video_sha256"]
                or full_body.get("episode_index") != args.episode):
            raise RuntimeError("Full depth/body do not cover all original video frames")
    else:
        indices = inputs["indices"]
    object_masks=base/'automatic_masks/masks/1'
    if ([p.name for p in sorted(object_masks.glob('*.png'))]!=[f'{index:06d}.png' for index in range(inputs['total_frames'])]
            or any(p.is_symlink() or not p.is_file() for p in object_masks.glob('*.png'))
            or object_masks.resolve()!=object_masks.absolute()):
        raise RuntimeError('Automatic object masks must cover every original frame as regular files')
    candidate_reports, poses_R, poses_t = [], [], []
    # Finite generic orientation hypotheses, not manually supplied object
    # symmetries. All can compete by observed silhouette; no symmetry averaging.
    orientation_hypotheses = Rotation.create_group("O").as_matrix()
    previous_rotation = rotation.copy()
    for index in indices:
        point_record = pointmaps[index]
        mask_path = base / f"automatic_masks/masks/1/{index:06d}.png"
        with Image.open(mask_path) as image:
            mask = np.asarray(image) > 0
        if args.full_video:
            point_path = base / f"depth_full/{index:06d}.npz"
            if (sha256(point_path) != point_record["output_sha256"]
                    or point_record["decoded_rgb_sha256"] != body_frames[index]["decoded_rgb_sha256"]):
                raise RuntimeError("Full depth SHA or decoded RGB mismatch")
            with np.load(point_path, allow_pickle=False) as arrays:
                depth, valid_depth, intrinsic = arrays["depth"], arrays["mask"], arrays["intrinsics"]
                if arrays["frame_index"].ndim != 0 or int(arrays["frame_index"]) != index:
                    raise RuntimeError("Full depth stored original frame index mismatch")
                pixel_K = np.diag([width, height, 1]) @ intrinsic
                if not np.allclose(pixel_K, camera, atol=1e-3, rtol=1e-6) or depth.shape != (height, width):
                    raise RuntimeError("Full depth and object must use identical fixed original camera")
                yy, xx = np.mgrid[:height, :width]
                points = np.stack(((xx + .5 - pixel_K[0, 2]) * depth / pixel_K[0, 0],
                                   (yy + .5 - pixel_K[1, 2]) * depth / pixel_K[1, 1], depth), axis=-1)
                points *= alignment["depth_alignment"]["shared_scale"]
                points[~valid_depth] = np.nan
        else:
            point_path = Path(point_record["pointmap_path"])
            if point_path != alignment_path.parent / f"{index:06d}.npy" or sha256(point_path) != point_record["pointmap_sha256"]:
                raise RuntimeError("Pointmap path/hash mismatch")
            if sha256(Path(point_record["intrinsics_path"])) != point_record["intrinsics_sha256"]:
                raise RuntimeError("Pointmap camera hash mismatch")
            if json.loads(Path(point_record["intrinsics_path"]).read_text()) != camera_dict:
                raise RuntimeError("One fixed RGB-size camera must agree across object and human pointmaps")
            if sha256(mask_path) != evidence[index]["object_mask_sha256"]:
                raise RuntimeError("Automatic object mask changed after alignment")
            points = np.load(point_path, allow_pickle=False)
        if mask.shape != (height, width) or points.shape != (height, width, 3):
            raise RuntimeError("Object mask/pointmap original camera grid mismatch")
        visible = mask & np.isfinite(points).all(-1) & (points[..., 2] > 0)
        observed = points[visible]
        if len(observed) < 40:
            raise RuntimeError("Insufficient inferred visible object points")
        rng = np.random.default_rng(0)
        if len(observed) > 2048:
            observed = observed[rng.choice(len(observed), 2048, replace=False)]
        candidates, rejected = [], []
        seed_rotations = [rotation @ hypothesis for hypothesis in orientation_hypotheses]
        if args.full_video:
            seed_rotations.append(previous_rotation)
        for number, initial_R in enumerate(seed_rotations):
            # Keep the generative translation at frame zero; otherwise seed
            # the surface centre from automatic observed depth, never hand pose.
            initial_t = translation if index == 0 else np.median(observed, axis=0) - mesh.centroid @ initial_R.T
            try:
                initial_mask, _ = raster_camera_mesh(mesh.vertices @ initial_R.T + initial_t, mesh.faces, camera, width, height)
                initial_iou = silhouette_iou(initial_mask.cpu().numpy(), mask)
                fit = align_observed_points(sampled, observed, initial_R, initial_t)
                fitted_mask, _ = raster_camera_mesh(mesh.vertices @ fit.rotation.T + fit.translation, mesh.faces, camera, width, height)
                fitted_iou = silhouette_iou(fitted_mask.cpu().numpy(), mask)
                # Partial depth ICP cannot choose unseen orientation by residual
                # alone. Require non-worse 2D image support before accepting a step.
                accepted = fitted_iou >= initial_iou and fit.final_residual <= fit.initial_residual
                chosen_R, chosen_t = (fit.rotation, fit.translation) if accepted else (initial_R, initial_t)
                chosen_iou = fitted_iou if accepted else initial_iou
                chosen_residual = fit.final_residual if accepted else fit.initial_residual
                candidates.append({"hypothesis_index": number, "initial_silhouette_iou": initial_iou,
                                   "fitted_silhouette_iou": fitted_iou, "icp_accepted_by_image_gate": bool(accepted),
                                   "selected_silhouette_iou": chosen_iou, "selected_depth_residual_m": chosen_residual,
                                   "rotation": chosen_R.tolist(), "translation": chosen_t.tolist(), "icp": fit.to_dict()})
            except ValueError as exc:
                # Candidate numerical/near-plane/underconstraint failure is not
                # permission to bypass global input provenance or CUDA errors.
                rejected.append({"hypothesis_index": number, "reason": str(exc)})
        if not candidates:
            raise RuntimeError(f"No finite supported pose hypothesis at original frame {index}: {rejected}")
        best = max(candidates, key=lambda c: (c["selected_silhouette_iou"], -c["selected_depth_residual_m"], -c["hypothesis_index"]))
        poses_R.append(best["rotation"])
        poses_t.append(best["translation"])
        previous_rotation = np.asarray(best["rotation"])
        candidate_reports.append({"frame_index": index, "visible_point_pixels": int(visible.sum()),
                                  "sampled_observations": len(observed), "selected": best, "candidates": candidates,
                                  "rejected_candidates": rejected, "object_mask_sha256": sha256(mask_path)})
        if args.full_video and (index + 1) % 50 == 0:
            print(json.dumps({"stage": "object_pose_full_progress", "frames_complete": index + 1,
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    temporal_report = None
    if args.full_video:
        hypothesis_count = len(orientation_hypotheses) + 1
        rotations = np.broadcast_to(np.eye(3), (len(indices), hypothesis_count, 3, 3)).copy()
        translations = np.zeros((len(indices), hypothesis_count, 3))
        costs, valid_candidates = np.zeros((len(indices), hypothesis_count)), np.zeros((len(indices), hypothesis_count), bool)
        for frame, frame_report in enumerate(candidate_reports):
            for candidate in frame_report["candidates"]:
                slot = candidate["hypothesis_index"]
                rotations[frame, slot], translations[frame, slot] = candidate["rotation"], candidate["translation"]
                costs[frame, slot], valid_candidates[frame, slot] = 1 - candidate["selected_silhouette_iou"], True
        path = select_pose_path(rotations, translations, costs, valid_candidates=valid_candidates,
                                translation_weight=1., rotation_weight=.1, frame_times=np.asarray(indices))
        poses_R, poses_t = path.rotations, path.translations
        temporal_report = {"method": "Viterbi_without_symmetry_equivalence_or_pose_averaging",
                           "image_cost": "1 - automatic_mask_IoU", "translation_weight": 1., "rotation_weight": .1,
                           "time_units": "original_frame_indices", "candidate_indices": path.candidate_indices.tolist(),
                           "unary_cost": path.unary_cost, "transition_cost": path.transition_cost,
                           "total_cost": path.total_cost, "quality_verified": False}
    with (output / "geometry_and_poses.npz").open("xb") as handle:
        np.savez_compressed(handle, vertices=vertices, faces=faces, frame_index=np.asarray(indices),
                            rotation=np.asarray(poses_R), translation=np.asarray(poses_t), object_scale=np.array(1.))
    result = {"stage": "fixed_scale_full_object_pose_initializer" if args.full_video else "fixed_scale_sparse_object_pose_consistency",
              "status": "pass", "episode_index": args.episode, "execution_verified": True, "original_frame_coverage_verified": args.full_video,
              "candidate_accuracy_validated": False, "full_trajectory_accuracy_verified": False,
              "input_track": "track_1", "input_sha256": inputs["video_sha256"], "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "submission_eligible": False,
              "challenge_performance_verified": False, "metric_scale_accuracy_verified": False,
              "object_report_sha256": sha256(object_report_path), "alignment_report_sha256": sha256(alignment_path),
              "fixed_shape": True, "scale": "generative_grounded_scale_baked_once_into_fixed_vertices; submission_scale=1",
              "mesh_watertight": bool(mesh.is_watertight), "mesh_winding_consistent": bool(mesh.is_winding_consistent),
              "metric_gauge_extent": mesh.extents.tolist(), "official_budget_vertices": len(vertices), "official_budget_faces": len(faces),
              "geometry_cleanup": geometry_cleanup,
              "mesh_source":args.mesh_source,
              "topology_budget": topology_budget, "fixed_canonical_mesh_sha256": sha256(fixed_mesh_path),
              "pose_hypotheses": "24_octahedral_orientations_not_asserted_true_object_symmetries",
              "objective": "maximum_automatic_mask_IoU_then_partial_depth_RMSE; no_GT_or_challenge_metric",
              "frames": candidate_reports, "geometry_and_poses_sha256": sha256(output / "geometry_and_poses.npz"),
              "temporal_selection": temporal_report,
              "full_depth_report_sha256": sha256(full_depth_report_path) if args.full_video else None,
              "budget_source_sha256": sha256(budget_source_path),
              "geometry_library_versions": {name: importlib.metadata.version(name) for name in ("trimesh", "fast-simplification")},
              "elapsed_seconds": time.perf_counter() - started, "script_sha256": sha256(Path(__file__))}
    if args.query_requalification: result['query_requalification'] = topology_budget['query_requalification']
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "status": "pass", "extent": mesh.extents.tolist(),
                      "frames": len(indices), "greedy_silhouette_iou_median": float(np.median([frame["selected"]["selected_silhouette_iou"] for frame in candidate_reports])),
                      "elapsed_seconds": result["elapsed_seconds"], "challenge_performance_verified": False}))


if __name__ == "__main__":
    main()
