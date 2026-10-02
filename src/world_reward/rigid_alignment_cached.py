"""Experimental canonical-tree execution of the unchanged partial rigid ICP.

For proper R, querying ``(observed - t) @ R`` in a static canonical cKDTree
has the same distances as querying observed against ``mesh @ R.T + t``.
Finite precision can change nearest-neighbour ties, trimming ties or stopping
iterations. This is therefore a separate proposal, not an automatic replacement
of the original solver. A frozen parity/throughput gate is required downstream.
One tree is built per alignment, never a clip-wide cache or all-pairs matrix.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

from .rigid_alignment import RigidAlignment, _kabsch, _points, _rank_two


def _cached_correspondences(tree, mesh, observed, rotation, translation, keep):
    """Use inverse rigid queries; preserve stable trimming and original units."""
    # Retain the original transformed-mesh finite gate, not just inverse rays.
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = mesh @ rotation.T + translation
        canonical_observed = (observed - translation) @ rotation
    if not np.isfinite(transformed).all():
        raise ValueError("Transformed mesh has nonfinite coordinates")
    if not np.isfinite(canonical_observed).all():
        raise ValueError("Inverse-transformed observations have nonfinite coordinates")
    distances, source_indices = tree.query(canonical_observed, k=1)
    if not np.isfinite(distances).all():
        raise ValueError("Nearest-neighbour residual exceeds finite numeric range")
    target_indices = np.argsort(distances, kind="stable")[:keep]
    selected = distances[target_indices]
    maximum = float(selected.max())
    residual = maximum * float(np.sqrt(np.mean((selected / maximum) ** 2))) if maximum else 0.0
    return source_indices[target_indices], target_indices, residual


def align_observed_points_cached(
    mesh_points: object, observed_points: object, initial_rotation: object,
    initial_translation: object, *, trim_fraction: float = 0.8,
    min_correspondences: int = 32, max_iterations: int = 30,
    tolerance: float = 1e-6,
) -> RigidAlignment:
    """Same absolute Kabsch/monotone ICP, static canonical nearest-neighbour tree.

    Reuse original point/rank/Kabsch validation and report ABI; keep the exact
    default trimming, support, iteration and stopping controls. Only query
    coordinates and tree lifetime differ. No parameter clipping, scale fit,
    pose composition, symmetries, hidden-to-visible residual or fallback occurs.
    Inputs are copied by the original validators and never modified.
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
    tree = cKDTree(mesh)
    source_ids, target_ids, initial_residual = _cached_correspondences(tree, mesh, observed, rotation, translation, keep)
    residual, iterations = initial_residual, 0
    for _ in range(int(max_iterations)):
        # Absolute canonical->observed transform, exactly as the original solver.
        proposal_r, proposal_t = _kabsch(mesh[source_ids], observed[target_ids])
        next_source_ids, next_target_ids, proposed = _cached_correspondences(
            tree, mesh, observed, proposal_r, proposal_t, keep,
        )
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
