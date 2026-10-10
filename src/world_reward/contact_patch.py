"""Bounded anatomical candidate pools and a soft existential contact residual.

Open-CHOIR (arXiv:2605.20992v4, sections 4.3/7.3) motivates soft surface
correspondences rather than hard frame-wise witnesses. This small primitive is
NOT Open-CHOIR parity: it does not estimate normals, infer contact, build signed
distance fields, enforce friction or correct human articulation. Candidate pools
are an approximation, not contact truth. Their coverage and the residual's
temperature/true-contact bias must be tested on independent development data.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ContactPatchConfig:
    temperature_diameter: float
    max_candidates: int
    development_reference: str

    def __post_init__(self):
        if (type(self.temperature_diameter) not in (int, float)
                or not np.isfinite(self.temperature_diameter) or self.temperature_diameter <= 0
                or type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 8
                or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Positive externally frozen temperature, <=8 candidates and reference required')


def _owned_readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def frozen_interval_candidates(hand_points, source_indices, initial_surface_distances,
                               activations, geometry_supported, max_candidates):
    """Choose a fixed candidate-ID pool for each contiguous active hand interval.

    Distances MUST come from original, pre-fit automatic geometry. Their minimum
    over an interval ranks candidate coverage; it does not qualify actual contact
    or set a contact state. All selected anatomical trajectories retain their
    frame-varying positions. Support is never interpolated or manufactured.

    Returns owned immutable ``(indices[T,2,P], points[T,2,P,3], visible[T,2,P])``.
    Inactive/padding indices are -1; unsupported points are NaN. A known selected
    ID remains recorded when its geometry is unsupported. If the bounded pool
    leaves an active frame without supported geometry, fail instead of silently
    removing that contact constraint. Caller keeps the original activations.
    """
    originals = (hand_points, source_indices, initial_surface_distances,
                 activations, geometry_supported)
    if any(np.ma.isMaskedArray(value) for value in originals):
        raise ValueError('Explicit unmasked anatomical geometry and support required')
    points, indices, distances, active, supported = map(np.asarray, originals)
    if (points.dtype.kind != 'f' or points.ndim != 4 or points.shape[1] != 2
            or points.shape[-1] != 3 or points.shape[0] < 1 or points.shape[2] < 1
            or indices.dtype.kind not in 'iu' or indices.shape != points.shape[1:3]
            or np.any(indices < 0)
            or distances.dtype.kind != 'f' or distances.shape != points.shape[:-1]
            or active.dtype != np.bool_ or active.shape != points.shape[:2]
            or supported.dtype != np.bool_ or supported.shape != points.shape[:-1]
            or not np.isfinite(points[supported]).all()
            or not np.isnan(points[~supported]).all()
            or not np.isfinite(distances[supported]).all() or np.any(distances[supported] < 0)
            or not np.isnan(distances[~supported]).all()
            or type(max_candidates) is not int or not 1 <= max_candidates <= 8):
        raise ValueError('Exact full-T anatomy, nonnegative supported distances and bounded pool required')
    if any(len(np.unique(side)) != len(side) for side in indices):
        raise ValueError('Anatomical source IDs must be unique within each hand')
    # Reject unsigned IDs that cannot survive the signed -1 unknown convention.
    if indices.size and np.max(indices) > np.iinfo(np.int64).max:
        raise ValueError('Anatomical source IDs must fit signed int64')
    count, _, source_count, _ = points.shape
    pool_count = min(source_count, max_candidates)
    selected_ids = np.full((count, 2, pool_count), -1, np.int64)
    selected_points = np.full((count, 2, pool_count, 3), np.nan, np.float64)
    selected_visible = np.zeros((count, 2, pool_count), bool)
    for side in range(2):
        padded = np.r_[False, active[:, side], False]
        changes = np.diff(padded.astype(np.int8))
        for begin, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            score = np.where(supported[begin:end, side], distances[begin:end, side], np.inf).min(axis=0)
            # Input-column permutation cannot change equal-score ID selection.
            order = np.lexsort((indices[side], score))
            chosen = order[np.isfinite(score[order])][:pool_count]
            if not len(chosen) or not supported[begin:end, side][:, chosen].any(axis=1).all():
                raise ValueError('Bounded anatomical pool lacks active-frame coverage; do not self-disable')
            size = len(chosen)
            selected_ids[begin:end, side, :size] = indices[side, chosen]
            selected_points[begin:end, side, :size] = points[begin:end, side][:, chosen]
            selected_visible[begin:end, side, :size] = supported[begin:end, side][:, chosen]
    return tuple(_owned_readonly(value) for value in
                 (selected_ids, selected_points, selected_visible))


def smooth_patch_distances(distances, groups, ngroups, temperature_metres):
    """Softmax-weighted RMS surface distance, one residual per contact group.

    Weights are proportional to ``exp(-d**2 / temperature**2)``. This is a soft
    existential approximation, not a sum attracting every candidate equally.
    It is nonnegative, does not cancel signed gaps, and permits a different
    candidate to realize contact. Finite temperature can give positive residual
    even when one candidate touches exactly; this bias is explicitly NOT removed.

    Groups may be interleaved/ragged, but every index 0..ngroups-1 needs observed
    distances. Singletons are exactly the old point-to-surface residual. Stable
    logarithmic accumulation prevents a distant, exponentially downweighted
    candidate from causing overflow or a false zero for positive supported gaps.
    """
    if np.ma.isMaskedArray(distances) or np.ma.isMaskedArray(groups):
        raise ValueError('Explicit unmasked supported distances and groups required')
    d, group = map(np.asarray, (distances, groups))
    if (d.dtype.kind != 'f' or d.ndim != 1 or not np.isfinite(d).all() or np.any(d < 0)
            or group.dtype.kind not in 'iu' or group.shape != d.shape
            or type(ngroups) is not int or ngroups < 0
            or np.any(group < 0) or np.any(group >= ngroups)
            or type(temperature_metres) not in (int, float)
            or not np.isfinite(temperature_metres) or temperature_metres <= 0):
        raise ValueError('Finite nonnegative distances, contiguous groups and positive temperature required')
    if ngroups == 0:
        if len(d):
            raise ValueError('An empty group set cannot contain observations')
        return _owned_readonly(np.empty(0, np.float64))
    if len(d) < ngroups:
        raise ValueError('Every active contact group needs supported geometry; do not self-disable')
    group = group.astype(np.int64, copy=False)
    counts = np.bincount(group, minlength=ngroups)
    if np.any(counts == 0):
        raise ValueError('Every active contact group needs supported geometry; do not self-disable')
    d = d.astype(np.float64, copy=False)
    minimum = np.full(ngroups, np.inf); np.minimum.at(minimum, group, d)
    maximum = np.zeros(ngroups); np.maximum.at(maximum, group, d)
    m = minimum[group]
    exponent = np.zeros(len(d), np.float64)
    different = d > m
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        exponent[different] = ((d[different]-m[different])/temperature_metres
            * (d[different]/temperature_metres+m[different]/temperature_metres))
    if np.isnan(exponent).any() or np.any(exponent < 0):
        raise ValueError('Soft contact arithmetic exceeds representable finite differences')
    with np.errstate(under='ignore'):
        weight = np.exp(-exponent)
    normalizer = np.bincount(group, weights=weight, minlength=ngroups)
    # Compute log(weight*d**2), not exp(weight)*d**2. The latter can erase
    # representable contributions when the weight alone underflows to zero.
    log_energy = np.full(len(d), -np.inf)
    positive = d > 0
    log_energy[positive] = (2*np.log(d[positive])-exponent[positive]
                            -np.log(normalizer[group[positive]]))
    pivot = np.full(ngroups, -np.inf); np.maximum.at(pivot, group, log_energy)
    finite = np.isfinite(log_energy)
    contribution = np.zeros(len(d))
    with np.errstate(under='ignore'):
        contribution[finite] = np.exp(log_energy[finite]-pivot[group[finite]])
    total = np.bincount(group, weights=contribution, minlength=ngroups)
    residual = np.zeros(ngroups)
    nonzero = total > 0
    with np.errstate(over='ignore', under='ignore'):
        residual[nonzero] = np.exp(.5*(pivot[nonzero]+np.log(total[nonzero])))
    # Weighted RMS is mathematically between min and max. Restore those bounds
    # only for floating-point underflow/round-off, never manufacture absence.
    residual = np.minimum(np.maximum(residual, minimum), maximum)
    singleton = counts == 1
    residual[singleton] = minimum[singleton]
    if not np.isfinite(residual).all():
        raise ValueError('Soft contact residual is not finitely representable')
    return _owned_readonly(residual)
