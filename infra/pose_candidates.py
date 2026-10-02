"""Same fixed-mesh pose hypotheses/ICP/image gates, with optional batch rasters.

This changes execution scheduling only. No hypotheses, ICP defaults, pixels,
samples, acceptance thresholds, pose averaging or temporal choices change.
Batch-size one retains the original initial-render/ICP/fitted-render order.
Batch-size eight renders initial poses, runs independent ICP sequentially, then
renders fitted poses. Candidate-specific numerical failures retain their
original hypothesis indices; shared-input and whole-batch failures propagate.
"""

from __future__ import annotations

import numpy as np

from camera_render import (
    _finite_array, _intrinsics, _mesh_inputs,
    raster_camera_mesh, raster_camera_mesh_batch, silhouette_iou,
)
from world_reward.rigid_alignment import align_observed_points


NEAR_CLIP_M = 1e-4


def _shared_inputs(vertices, faces, observed_mask, camera_matrix):
    """Reject invalid shared geometry/camera outside candidate rejection scopes.

    Canonical coordinates need not be in front of the camera. The unchanged
    scalar ``_mesh_inputs`` performs camera-Z/float32 checks on each *posed*
    mesh before batching. Return original dtypes, so per-pose matmul roundoff
    is identical to the scalar producer; no batched matmul or repair is used.
    """
    checked = _finite_array(vertices, "canonical_vertices")
    if checked.ndim != 2 or checked.shape[1:] != (3,) or len(checked) < 3:
        raise ValueError("Canonical mesh must contain at least three finite vertices")
    indices = np.asarray(faces)
    if (indices.ndim != 2 or indices.shape[1:] != (3,) or not len(indices)
            or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(checked))):
        raise ValueError("Shared faces must contain valid integer vertex indices")
    if (np.any(indices[:, 0] == indices[:, 1]) or np.any(indices[:, 0] == indices[:, 2])
            or np.any(indices[:, 1] == indices[:, 2])):
        raise ValueError("Shared faces contain repeated vertex indices")
    triangles = checked[indices]
    with np.errstate(over="ignore", invalid="ignore"):
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if not np.isfinite(normals).all() or np.any(np.all(normals == 0, axis=1)):
        raise ValueError("Shared mesh contains zero-area or numerically invalid triangles")
    _intrinsics(camera_matrix)
    mask = np.asarray(observed_mask)
    if mask.ndim != 2 or min(mask.shape, default=0) == 0 or mask.dtype != np.bool_:
        raise ValueError("Observed silhouette must be a nonempty 2D boolean array")
    return np.asarray(vertices), indices, mask, np.asarray(camera_matrix)


def _candidate(number, initial_R, initial_t, initial_iou, fit, fitted_iou) -> dict:
    # Exactly the scalar producer's image-and-residual conjunction/tie rules.
    accepted = fitted_iou >= initial_iou and fit.final_residual <= fit.initial_residual
    chosen_R, chosen_t = (fit.rotation, fit.translation) if accepted else (initial_R, initial_t)
    chosen_iou = fitted_iou if accepted else initial_iou
    chosen_residual = fit.final_residual if accepted else fit.initial_residual
    return {
        "hypothesis_index": number, "initial_silhouette_iou": initial_iou,
        "fitted_silhouette_iou": fitted_iou, "icp_accepted_by_image_gate": bool(accepted),
        "selected_silhouette_iou": chosen_iou, "selected_depth_residual_m": chosen_residual,
        "rotation": chosen_R.tolist(), "translation": chosen_t.tolist(), "icp": fit.to_dict(),
    }


def _mask_numpy(mask_tensor, shape: tuple[int, ...]) -> np.ndarray:
    """One CUDA-to-CPU mask conversion per batch; unexpected outputs fail globally."""
    masks = mask_tensor.cpu().numpy()
    if masks.shape != shape or masks.dtype != np.bool_:
        raise RuntimeError("Raster mask violated the unchanged boolean original-resolution contract")
    return masks


def evaluate_pose_candidates(
    canonical_vertices,
    faces,
    surface_points,
    observed_points,
    observed_mask,
    camera_matrix,
    seed_rotations,
    seed_translations,
    *,
    render_batch_size: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Return original-format ``(candidates, rejected)`` in hypothesis order.

    Seeds are the caller's unchanged 24/25 rotations and per-seed translations;
    no new seed, pruning, RNG or best/temporal selection is introduced. All
    transformations use ``vertices @ R.T + t`` separately, including batch mode.
    ICP remains sequential with its original defaults and source samples.

    Only candidate-scoped ValueError (pose/near-plane/ICP/IoU) is rejected. A
    fitted pose failing its raster gate rejects the *entire* candidate, not
    merely its update. RuntimeError and all whole-batch backend exceptions
    propagate: a GPU/backend failure is never converted into candidate evidence.
    Shared mesh/camera/mask validation is outside these catches. No upstream
    provenance is certified by this numerical orchestration helper.
    """
    if (isinstance(render_batch_size, (bool, np.bool_))
            or not isinstance(render_batch_size, (int, np.integer)) or render_batch_size not in (1, 8)):
        raise ValueError("render_batch_size must be exactly 1 or 8")
    vertices, indices, mask, matrix = _shared_inputs(canonical_vertices, faces, observed_mask, camera_matrix)
    rotations, translations = list(seed_rotations), list(seed_translations)
    if not rotations or len(rotations) != len(translations):
        raise ValueError("Require nonempty, equally sized seed rotation/translation lists")
    seeds = [(np.asarray(rotation), np.asarray(translation))
             for rotation, translation in zip(rotations, translations, strict=True)]
    height, width = mask.shape
    candidates, rejected = {}, {}

    if render_batch_size == 1:
        for number, (initial_R, initial_t) in enumerate(seeds):
            try:
                initial_mask, _ = raster_camera_mesh(vertices @ initial_R.T + initial_t, indices, matrix, width, height)
                initial_iou = silhouette_iou(initial_mask.cpu().numpy(), mask)
                fit = align_observed_points(surface_points, observed_points, initial_R, initial_t)
                fitted_mask, _ = raster_camera_mesh(vertices @ fit.rotation.T + fit.translation, indices, matrix, width, height)
                fitted_iou = silhouette_iou(fitted_mask.cpu().numpy(), mask)
                candidates[number] = _candidate(number, initial_R, initial_t, initial_iou, fit, fitted_iou)
            except ValueError as exc:
                rejected[number] = {"hypothesis_index": number, "reason": str(exc)}
        return list(candidates.values()), list(rejected.values())

    initial_ious, fits = {}, {}

    def render_phase(numbers, *, fitted: bool) -> None:
        # CPU validation is candidate-scoped, exactly the scalar raster's
        # original validator. Keep original posed arrays (not its float32
        # returns), so the batch backend applies the same conversion once.
        valid = []
        for number in numbers:
            R, t = (fits[number].rotation, fits[number].translation) if fitted else seeds[number]
            try:
                posed = vertices @ R.T + t
                _mesh_inputs(posed, indices, matrix, width, height, NEAR_CLIP_M)
            except ValueError as exc:
                rejected[number] = {"hypothesis_index": number, "reason": str(exc)}
            else:
                valid.append((number, posed))
        for start in range(0, len(valid), int(render_batch_size)):
            batch = valid[start:start + int(render_batch_size)]
            # All batch/backend exceptions propagate; no attribution by guessing.
            masks_gpu, depths_gpu = raster_camera_mesh_batch(
                np.stack([posed for _, posed in batch]), indices, matrix, width, height,
            )
            del depths_gpu
            masks = _mask_numpy(masks_gpu, (len(batch), height, width))
            del masks_gpu
            for (number, _), rendered_mask in zip(batch, masks, strict=True):
                try:
                    iou = silhouette_iou(rendered_mask, mask)
                    if fitted:
                        initial_R, initial_t = seeds[number]
                        candidates[number] = _candidate(number, initial_R, initial_t, initial_ious[number], fits[number], iou)
                    else:
                        initial_ious[number] = iou
                except ValueError as exc:
                    rejected[number] = {"hypothesis_index": number, "reason": str(exc)}

    render_phase(range(len(seeds)), fitted=False)
    for number in initial_ious:
        initial_R, initial_t = seeds[number]
        try:
            fits[number] = align_observed_points(surface_points, observed_points, initial_R, initial_t)
        except ValueError as exc:
            rejected[number] = {"hypothesis_index": number, "reason": str(exc)}
    render_phase(fits, fitted=True)
    # Execution phases must not reorder rejected hypotheses or valid slots.
    return ([candidates[number] for number in sorted(candidates)],
            [rejected[number] for number in sorted(rejected)])
