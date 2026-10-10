"""Opt-in within-frame common-mode covariance for inferred axial depth.

This is an analytic rank-one measurement model, not a new depth predictor or
temporal filter. A frame's supported observations have covariance
``sigma_independent**2 * I + sigma_common**2 * ones``. Only its common mode loses
the false confidence obtained by counting correlated depths independently.
Scales must be frozen on external development data; prediction discrepancies
from a challenge clip are not calibrated depth errors. No geometry, gauge,
timeline, observations or contact activations are changed here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _scale(value, *, positive):
    if (type(value) not in (int, float) or not np.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError('Finite positive independent/nonnegative common scale required')
    return float(value)


def _owned_immutable(value):
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class CommonModeDepthConfig:
    """Externally frozen standard deviation normalized by fixed mesh diameter.

    Zero selects exact legacy independent weighting. This declaration is not
    proof of calibrated uncertainty or leakage-free development provenance.
    """
    sigma_common_diameter: float
    development_reference: str

    def __post_init__(self):
        _scale(self.sigma_common_diameter, positive=False)
        if type(self.development_reference) is not str or not self.development_reference.strip():
            raise ValueError('Explicit external development reference required')


@dataclass(frozen=True, init=False)
class FrameDepthWhitening:
    """Prepared permutation-equivariant symmetric inverse square root.

    Preparation groups actual supported frame IDs once. ``apply`` is O(M) in
    supported observations, needs no dense matrix and never fills missing rows.
    Its squared norm equals the Gaussian GLS quadratic for this covariance.
    Applying a scalar robust loss afterwards defines a robust whitened factor,
    not an exact Gaussian or elliptical robust likelihood. Between-frame error
    correlation is deliberately not modeled by this within-frame operator.
    """
    frame_ids: np.ndarray
    sigma_independent_m: float
    sigma_common_m: float
    _groups: np.ndarray
    _counts: np.ndarray
    _sample_counts: np.ndarray
    _group_sigma: np.ndarray

    def __init__(self, frame_ids, sigma_independent_m, sigma_common_m):
        ids = np.asarray(frame_ids)
        if (np.ma.isMaskedArray(frame_ids) or ids.dtype.kind not in 'iu'
                or ids.ndim != 1 or not len(ids) or np.any(ids < 0)):
            raise ValueError('Nonempty one-dimensional nonnegative integer supported frame IDs required')
        independent = _scale(sigma_independent_m, positive=True)
        common = _scale(sigma_common_m, positive=False)
        _, groups, counts = np.unique(ids, return_inverse=True, return_counts=True)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            group_sigma = np.hypot(independent, np.sqrt(counts.astype(float))*common)
            inverse_independent = np.divide(1., independent)
            inverse_group = np.divide(1., group_sigma)
        if (not np.isfinite(inverse_independent) or not np.isfinite(group_sigma).all()
                or np.any(group_sigma <= 0) or not np.isfinite(inverse_group).all()
                or np.any(inverse_group <= 0)):
            raise ValueError('Covariance scales exceed finite representable positive arithmetic')
        for name, value in (('frame_ids', ids), ('_groups', groups), ('_counts', counts),
                            ('_sample_counts', counts[groups]), ('_group_sigma', group_sigma)):
            object.__setattr__(self, name, _owned_immutable(value))
        object.__setattr__(self, 'sigma_independent_m', independent)
        object.__setattr__(self, 'sigma_common_m', common)

    @property
    def observation_count(self):
        return len(self.frame_ids)

    @property
    def group_count(self):
        return len(self._counts)

    def apply(self, residual_m):
        residual = np.asarray(residual_m)
        if (np.ma.isMaskedArray(residual_m) or residual.dtype.kind != 'f'
                or residual.shape != self.frame_ids.shape or not np.isfinite(residual).all()):
            raise ValueError('Exact supported one-dimensional finite floating depth residuals required')
        residual = residual.astype(np.float64, copy=False)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            if self.sigma_common_m == 0:
                # Preserve the independent route's exact division/order.
                result = residual/self.sigma_independent_m
            else:
                # Divide before summing: an ordinary sum of finite residuals
                # can overflow even when their mean is representable.
                means = np.bincount(self._groups, weights=residual/self._sample_counts,
                                    minlength=self.group_count)
                row_means = means[self._groups]
                result = ((residual-row_means)/self.sigma_independent_m
                          + row_means/self._group_sigma[self._groups])
        if not np.isfinite(result).all():
            raise ValueError('Whitened residuals exceed finite representable arithmetic')
        return result
