"""Experimental conservative nested M0(pose-only)/M1(shape+pose) selection.

Callers construct paired untouched TEMPORAL BLOCKS, not adjacent frame samples.
Their independence, loss calibration, model fits and provenance are not verified
here. A one-standard-error gain and normalized Schur conditioning are candidate
research heuristics, not statistical identification or challenge validation.
Nothing fits, mutates geometry, reads data or authorizes automatic adoption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


MINIMUM_BLOCKS, MAX_CONDITION_NUMBER = 3, 1e8


@dataclass(frozen=True)
class ShapeSelection:
    decision: Literal["pose_only", "shape_pose_proposal"]
    reasons: tuple[str, ...]
    report: dict


def _array(value, name, shape=None):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.dtype.kind not in "iuf" or not np.isfinite(array).all()
            or (shape is not None and array.shape != shape)):
        raise ValueError(f"{name} requires finite real values with shape {shape}")
    return array.astype(np.float64, copy=True)


def shape_schur_complement(shape_hessian, pose_hessian, cross_hessian):
    """Undamped Hθθ-Hθp solve(Hpp,Hpθ); no pseudo-inverse or PSD repair.

    Five shape controls and arbitrary positive pose dimension; finite symmetric
    Hessians are required. The pose block must be strictly positive and satisfy
    the declared numerical condition cap. Priors/damping, coordinate units and
    statistical information are caller concerns, not silently removed here.
    """
    shape = _array(shape_hessian, "shape_hessian", (5, 5))
    pose = _array(pose_hessian, "pose_hessian")
    if pose.ndim != 2 or pose.shape[0] != pose.shape[1] or not len(pose):
        raise ValueError("pose_hessian must be nonempty square")
    cross = _array(cross_hessian, "cross_hessian", (5, len(pose)))
    if not np.array_equal(shape, shape.T) or not np.array_equal(pose, pose.T):
        raise ValueError("Hessian blocks must be explicitly symmetric; no averaging repair")
    eigenvalues = np.linalg.eigvalsh(pose)
    if (not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0
            or eigenvalues[0] / eigenvalues[-1] < 1 / MAX_CONDITION_NUMBER):
        raise ValueError("Pose Hessian is nonpositive or numerically rank-deficient; no damping/pseudo-inverse")
    with np.errstate(over="ignore", invalid="ignore"):
        schur = shape - cross @ np.linalg.solve(pose, cross.T)
    if not np.isfinite(schur).all():
        raise ValueError("Schur complement exceeds finite numeric range")
    return schur


def select_shape_candidate(pose_only_losses, shape_pose_losses, pose_only_iou, shape_pose_iou,
                           shape_schur_eigenvalues, *, saturated_bounds: bool,
                           pose_modes_ambiguous: bool, heldout_pixels_disjoint: bool) -> ShapeSelection:
    """Default M0 unless every experimental gate admits an M1 proposal.

    Loss vectors are paired block mean squared metric residuals in m² with the
    same observations/weights. Positive gain is M0-M1. IoUs use the same blocks;
    both missing conservatively abstains. Disjoint held-out pixels alone do not
    establish temporal block independence or prevent hyperparameter overfit.
    """
    if any(type(value) is not bool for value in (saturated_bounds, pose_modes_ambiguous, heldout_pixels_disjoint)):
        raise ValueError("Saturation, ambiguity and held-out disjointness require explicit booleans")
    baseline = _array(pose_only_losses, "pose_only_losses")
    candidate = _array(shape_pose_losses, "shape_pose_losses", baseline.shape)
    if baseline.ndim != 1 or len(baseline) < MINIMUM_BLOCKS or (baseline < 0).any() or (candidate < 0).any():
        raise ValueError("Require at least three paired nonnegative temporal-block losses in m²")
    eigenvalues = _array(shape_schur_eigenvalues, "shape_schur_eigenvalues", (5,))
    silhouette_missing = pose_only_iou is None and shape_pose_iou is None
    if (pose_only_iou is None) != (shape_pose_iou is None):
        raise ValueError("Supply both paired IoU vectors or neither")
    if not silhouette_missing:
        iou0, iou1 = (_array(value, name, baseline.shape) for value, name in
                      ((pose_only_iou, "pose_only_iou"), (shape_pose_iou, "shape_pose_iou")))
        if (iou0 < 0).any() or (iou0 > 1).any() or (iou1 < 0).any() or (iou1 > 1).any():
            raise ValueError("Paired block IoUs must lie in [0,1]")
    with np.errstate(over="ignore", invalid="ignore"):
        gains = baseline - candidate
        mean_gain = float(gains.mean())
        # Rescale before variance to avoid spurious overflow for finite losses.
        scale = float(np.abs(gains).max())
        standard_error = float(np.std(gains / scale, ddof=1) / np.sqrt(len(gains)) * scale) if scale else 0.
    if not np.isfinite([mean_gain, standard_error]).all():
        raise ValueError("Paired improvement statistics exceed finite numeric range")
    minimum, maximum = float(eigenvalues.min()), float(eigenvalues.max())
    schur_ok = minimum > 0 and minimum / maximum >= 1 / MAX_CONDITION_NUMBER
    reasons = []
    if not heldout_pixels_disjoint: reasons.append("heldout_pixels_not_disjoint")
    if saturated_bounds: reasons.append("shape_bounds_saturated")
    if pose_modes_ambiguous: reasons.append("pose_modes_ambiguous")
    if not schur_ok: reasons.append("shape_schur_nonpositive_or_ill_conditioned")
    if not mean_gain > standard_error: reasons.append("paired_gain_not_above_one_standard_error")
    if silhouette_missing: reasons.append("paired_silhouette_validation_missing")
    elif float(iou1.mean()) < float(iou0.mean()): reasons.append("mean_heldout_silhouette_regressed")
    decision = "pose_only" if reasons else "shape_pose_proposal"
    report = {"policy": "experimental_nested_shape_one_standard_error_v1", "decision": decision,
              "policy_thresholds_validated": False, "adoption_authorized": False,
              "challenge_performance_verified": False, "loss_units": "metres_squared",
              "validation_unit": "caller_supplied_temporal_block", "paired_blocks": len(gains),
              "block_independence_verified": False, "statistical_calibration_verified": False,
              "mean_paired_gain_m2": mean_gain, "paired_gain_standard_error_m2": standard_error,
              "minimum_blocks": MINIMUM_BLOCKS, "maximum_schur_condition_number": MAX_CONDITION_NUMBER,
              "schur_min_eigenvalue": minimum, "schur_max_eigenvalue": maximum,
              "schur_numerical_condition_proxy_pass": schur_ok, "absolute_information_verified": False,
              "statistical_identifiability_verified": False, "heldout_pixels_disjoint": heldout_pixels_disjoint,
              "saturated_bounds": saturated_bounds, "pose_modes_ambiguous": pose_modes_ambiguous,
              "mean_iou_M0": None if silhouette_missing else float(iou0.mean()),
              "mean_iou_M1": None if silhouette_missing else float(iou1.mean()), "reasons": list(reasons)}
    return ShapeSelection(decision, tuple(reasons), report)
