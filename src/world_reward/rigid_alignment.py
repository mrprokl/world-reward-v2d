"""Bounded-memory rigid ICP from partial observed points to a fixed mesh sample.

The mesh sample must already have physical scale. This fits proper rotation and
translation only, using observed-to-surface nearest neighbours (not forcing hidden
surface points onto visible evidence). No GT, scale, deformation, timestamps or
smoothing are inputs. ICP is local and cannot identify an unknown symmetry; callers
may enumerate externally defined initial rotations and inspect evidence separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RigidAlignment:
    rotation: np.ndarray
    translation: np.ndarray
    status: str
    initial_residual: float
    final_residual: float
    inliers: int
    iterations: int

    def to_dict(self) -> dict:
        return {
            "schema": "world-reward-rigid-partial-icp-v1",
            "interpretation": "Trimmed observed-to-sampled-surface fit; not pose accuracy or symmetry resolution",
            "rotation": self.rotation.tolist(), "translation": self.translation.tolist(),
            "status": self.status, "initial_residual": self.initial_residual,
            "final_residual": self.final_residual, "inliers": self.inliers,
            "iterations": self.iterations, "scale_fitted": False,
        }


def _points(value: object, name: str) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name}: explicit finite points required, not a masked array")
    try:
        points = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: finite real [N,3] points required") from exc
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or not (np.issubdtype(points.dtype, np.integer) or np.issubdtype(points.dtype, np.floating))):
        raise ValueError(f"{name}: finite real nonempty [N,3] points required")
    points = points.astype(np.float64, copy=True)
    if not np.isfinite(points).all():
        raise ValueError(f"{name}: nonfinite points must be explicitly filtered by the caller")
    return points


def _rank_two(points: np.ndarray, name: str) -> None:
    centered = points - points.mean(axis=0)
    if not np.isfinite(centered).all():
        raise ValueError(f"{name}: numeric range overflow")
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 2 or singular[0] <= 0 or singular[1] <= singular[0] * 1e-8:
        raise ValueError(f"{name}: underconstrained point support (rank < 2)")


def _correspondences(
    mesh: np.ndarray, observed: np.ndarray, rotation: np.ndarray,
    translation: np.ndarray, keep: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    transformed = mesh @ rotation.T + translation
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed mesh has nonfinite coordinates")
    # One tree plus O(N_observed) queries, never an all-pairs distance matrix.
    distances, source_indices = cKDTree(transformed).query(observed, k=1)
    if not np.isfinite(distances).all():
        raise ValueError("Nearest-neighbour residual exceeds finite numeric range")
    target_indices = np.argsort(distances, kind="stable")[:keep]
    selected = distances[target_indices]
    maximum = float(selected.max())
    residual = maximum * float(np.sqrt(np.mean((selected / maximum) ** 2))) if maximum else 0.0
    return source_indices[target_indices], target_indices, residual


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _rank_two(source, "Visible source correspondences")
    _rank_two(target, "Observed target correspondences")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    a, b = source - source_center, target - target_center
    normalization = max(float(np.abs(a).max()), float(np.abs(b).max()))
    cross = (a / normalization).T @ (b / normalization)
    u, singular, vt = np.linalg.svd(cross)
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-8:
        raise ValueError("Visible correspondence mapping is underconstrained (rank < 2)")
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    if not np.isfinite(translation).all():
        raise ValueError("Rigid update produced nonfinite translation")
    return rotation, translation


def align_observed_points(
    mesh_points: object, observed_points: object, initial_rotation: object,
    initial_translation: object, *, trim_fraction: float = 0.8,
    min_correspondences: int = 32, max_iterations: int = 30,
    tolerance: float = 1e-6,
) -> RigidAlignment:
    """Trimmed rigid ICP with proper SO(3), keeping initialization if not improved.

    At every step retain ``floor(trim_fraction * N_observed)`` smallest nearest
    surface residuals, requiring >= ``min_correspondences``. Solve Kabsch on their
    corresponding original source points and observed targets; accept only a lower
    trimmed RMSE. Report before/after in input point units and do not interpret the
    residual as known pose error. Empty/nonfinite inputs and rank<2 correspondences
    fail explicitly; planar support is allowed but may remain ambiguous in-plane.
    A large move is not itself rejected, but needs a useful externally supplied
    initialization. Outlier fractions above the declared trimming budget can bias
    the result. No hyperparameter or candidate is inferred from challenge GT.
    """
    mesh = _points(mesh_points, "mesh_points")
    observed = _points(observed_points, "observed_points")
    if (isinstance(trim_fraction, bool) or not isinstance(trim_fraction, Real)
            or not np.isfinite(trim_fraction) or not 0 < trim_fraction <= 1):
        raise ValueError("trim_fraction must be finite in (0,1]")
    for value, name in ((min_correspondences, "min_correspondences"), (max_iterations, "max_iterations")):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    keep = int(np.floor(float(trim_fraction) * len(observed)))
    if keep < min_correspondences:
        raise ValueError("Insufficient observed correspondences after declared trimming")
    _rank_two(mesh, "Mesh sample")
    _rank_two(observed, "Observed sample")
    rotation = np.asarray(initial_rotation, dtype=np.float64).copy()
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("initial_rotation must be a finite proper SO(3) matrix")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("initial_translation must be a finite [3] vector")
    initial_r, initial_t = rotation.copy(), translation.copy()
    source_ids, target_ids, initial_residual = _correspondences(mesh, observed, rotation, translation, keep)
    residual = initial_residual
    iterations = 0
    for _ in range(int(max_iterations)):
        proposal_r, proposal_t = _kabsch(mesh[source_ids], observed[target_ids])
        next_source_ids, next_target_ids, proposed = _correspondences(mesh, observed, proposal_r, proposal_t, keep)
        iterations += 1
        if proposed >= residual:
            break
        improvement = residual - proposed
        rotation, translation, residual = proposal_r, proposal_t, proposed
        source_ids, target_ids = next_source_ids, next_target_ids
        if improvement <= tolerance:
            break
    if residual >= initial_residual:
        rotation, translation, residual, status = initial_r, initial_t, initial_residual, "unchanged"
    else:
        status = "improved"
    return RigidAlignment(rotation, translation, status, initial_residual, residual, keep, iterations)
