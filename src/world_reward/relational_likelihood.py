"""Conditional Gaussian relation evidence, not calibrated identity or contact.

The null independently models EVERY supplied hand/object innovation. A pair
hypothesis replaces only that hand/object product by a joint Gaussian with the
SAME entity means/marginal covariances; a shared two-state HMM marginalizes
independent/dependent regimes. Other entities and common absence/clutter factors
cancel in its Bayes factor. Missing dimensions use Gaussian marginal densities,
never zero-imputed motion. Missingness is conditioned upon/assumed ignorable;
informative occlusion is not solved by this working model.

All parameters and hypothesis priors are external, with no fitted defaults.
Innovations/means must be source-aware causal estimates with common temporal
noise conditioning across both arms. This API does not authenticate training
splits, causality, whiteness, support, physical identities or calibration. Raw
velocities may violate the conditional independence assumption. Hand slots are
not actors; correlated stationary motions need not identify an interaction.
Subtracting the same estimated camera motion does NOT establish independence:
shared estimation error c gives Cov(h-c,o-c)=Var(c) even for independent h,o.
The current OLS/Boots evidence has not qualified this null on external negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _real(value, shape, name, *, finite=True):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.shape != shape or array.dtype.kind != "f"
            or (finite and not np.isfinite(array).all())):
        raise ValueError(f"{name}: floating {shape} required with finite supplied parameters")
    return _owned(np.array(array, dtype=np.float64, copy=True))


def _keys(keys, name):
    if (type(keys) is not tuple or not keys
            or any(type(k) is not str or not k.strip() or len(k) > 256
                   or any(ord(c) < 32 or ord(c) == 127 for c in k) for k in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError(f"{name}: unique bounded keys in original tuple order required")
    return keys


def _support(value, shape):
    array = np.asarray(value)
    if np.ma.isMaskedArray(value) or array.dtype != np.bool_ or array.shape != shape:
        raise ValueError("Per-dimension boolean observed support required")
    return _owned(array)


def _stochastic(value, shape, name):
    array = _real(value, shape, name)
    # Floating normalization check only; never silently renormalize parameters.
    if (np.any(array < 0) or np.any(array > 1)
            or np.any(np.abs(array.sum(axis=-1) - 1.) > 8 * np.finfo(np.float64).eps)):
        raise ValueError(f"{name}: externally normalized nonnegative probabilities required")
    with np.errstate(divide="ignore"):
        return np.log(array)


def _cholesky(covariance):
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("Exactly symmetric supplied covariance required")
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Positive definite supplied covariance required") from error
    if not np.isfinite(chol).all() or np.any(np.diag(chol) <= 0):
        raise ValueError("Covariance factorization is numerically unavailable")
    return chol


def _gaussian_marginal(value, mean, covariance, observed):
    ids = np.flatnonzero(observed)
    if not len(ids):
        return 0.  # Integral of the completely unobserved normalized density.
    chol = _cholesky(covariance[np.ix_(ids, ids)])
    whitened = np.linalg.solve(chol, value[ids] - mean[ids])
    result = -.5 * (len(ids) * math.log(2. * math.pi)
                     + 2. * np.log(np.diag(chol)).sum() + whitened @ whitened)
    if not np.isfinite(result):
        raise ValueError("Gaussian marginal arithmetic overflowed")
    return float(result)


def _logsumexp(value):
    maximum = float(np.max(value))
    if maximum == -np.inf:
        return maximum
    return maximum + math.log(float(np.exp(value - maximum).sum()))


def _forward_ratio(ratios, log_initial, log_transition):
    if not len(ratios) or np.all(ratios == 0):
        return 0.  # Normalized latent law carries no relation evidence.
    alpha = log_initial + np.array([0., ratios[0]])
    for ratio in ratios[1:]:
        alpha = np.array([_logsumexp(alpha + log_transition[:, j]) for j in range(2)])
        alpha += np.array([0., ratio])
    result = _logsumexp(alpha)
    if not math.isfinite(result):
        raise ValueError("Conditional forward likelihood overflowed")
    return result


@dataclass(frozen=True, eq=False)
class RelationalLikelihoodEvidence:
    """Raw model evidence only; no accepted identity, probability or confidence.

    Pair arrays use [H,O,(T-1)]. Relative hypothesis log weights order is null
    first, then the full original row-major Cartesian pair bank. They include
    supplied priors with the common whole-bank factor removed algebraically,
    without posterior normalization. Absolute null/pair likelihoods are only
    diagnostics: their large common term can round away a small Bayes factor.
    Duplicate physical hypotheses must have their prior mass split externally;
    unique textual keys do not certify different physical entities. No model
    handles multiple simultaneous pair relations in one hypothesis.
    """
    frame_index: np.ndarray
    hand_keys: tuple[str, ...]
    object_keys: tuple[str, ...]
    pair_supported: np.ndarray
    dependent_log_ratios: np.ndarray
    log_bayes_factors: np.ndarray
    null_log_likelihood: float
    pair_log_likelihoods: np.ndarray
    relative_hypothesis_log_weights: np.ndarray


def conditional_relational_likelihood(
        frame_index, hand_keys, object_keys, hand_innovations, object_innovations,
        hand_observed, object_observed, *, hand_mean, object_mean,
        hand_covariance, object_covariance, cross_covariance,
        regime_initial, regime_transition, hypothesis_prior, common_log_factors):
    """Forward-logsumexp pair/null comparison on one complete observation bank.

    Original frame_index is int64 arange(T), T>=1. Innovations, external means
    and dimension supports are [entities,T-1,2]; entity covariances [entities,2,2]
    are shared across every pairing. cross_covariance[H,O,2,2] is the hand-object
    block of each SPD joint covariance. HMM initial[2]/transition[2,2] are shared
    normalized laws, with state0 independent and state1 statistically dependent.
    hypothesis_prior[1+H*O] includes a positive explicit null, zero pair priors
    are permitted. common_log_factors[T-1] is ONE finite full-bank absence/clutter
    log factor applied identically to every hypothesis, not a per-pair penalty.

    Unobserved coordinates may be NaN; observed coordinates must be finite.
    If either entity is completely absent in an interval, its emission ratio is
    exactly zero while the latent regime still advances. No observations over
    the whole timeline implies Bayes factor1/logBF0, not an accepted identity.
    Numerical invalidity fails the whole bank; no low-support pair is discarded.
    """
    indices = np.asarray(frame_index)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64 or indices.ndim != 1
            or not len(indices) or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
        raise ValueError("Complete original int64 arange(T) required")
    hand_keys, object_keys = _keys(hand_keys, "hands"), _keys(object_keys, "objects")
    nh, no, steps = len(hand_keys), len(object_keys), len(indices) - 1
    h = _real(hand_innovations, (nh, steps, 2), "hand innovations", finite=False)
    o = _real(object_innovations, (no, steps, 2), "object innovations", finite=False)
    hs, os = _support(hand_observed, h.shape), _support(object_observed, o.shape)
    if not np.isfinite(h[hs]).all() or not np.isfinite(o[os]).all():
        raise ValueError("Observed innovations must be finite; missing stays unobserved")
    hm, om = _real(hand_mean, h.shape, "hand means"), _real(object_mean, o.shape, "object means")
    hc = _real(hand_covariance, (nh, 2, 2), "hand covariances")
    oc = _real(object_covariance, (no, 2, 2), "object covariances")
    cross = _real(cross_covariance, (nh, no, 2, 2), "cross covariances")
    log_initial = _stochastic(regime_initial, (2,), "regime initial")
    log_transition = _stochastic(regime_transition, (2, 2), "regime transitions")
    log_prior = _stochastic(hypothesis_prior, (1 + nh * no,), "hypothesis prior")
    if not np.isfinite(log_prior[0]):
        raise ValueError("Explicit null must have positive supplied prior")
    common = _real(common_log_factors, (steps,), "shared full-bank factors")
    for covariance in (*hc, *oc):
        _cholesky(covariance)
    ratios = np.zeros((nh, no, steps))
    pair_supported = hs.any(axis=-1)[:, None, :] & os.any(axis=-1)[None, :, :]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hl = np.array([[_gaussian_marginal(h[i, t], hm[i, t], hc[i], hs[i, t])
                            for t in range(steps)] for i in range(nh)]).reshape(nh, steps)
            ol = np.array([[_gaussian_marginal(o[j, t], om[j, t], oc[j], os[j, t])
                            for t in range(steps)] for j in range(no)]).reshape(no, steps)
            null = float(hl.sum() + ol.sum() + common.sum())
            for i in range(nh):
                for j in range(no):
                    joint = np.block([[hc[i], cross[i, j]], [cross[i, j].T, oc[j]]])
                    _cholesky(joint)  # Invalid model is not hidden by missing data.
                    if not np.any(cross[i, j]):
                        continue  # Gaussian joint equals product exactly, not cancellation roundoff.
                    for t in np.flatnonzero(pair_supported[i, j]):
                        value = np.r_[h[i, t], o[j, t]]
                        mean = np.r_[hm[i, t], om[j, t]]
                        observed = np.r_[hs[i, t], os[j, t]]
                        ratios[i, j, t] = _gaussian_marginal(value, mean, joint, observed) - hl[i, t] - ol[j, t]
            bf = np.array([_forward_ratio(ratios[i, j], log_initial, log_transition)
                           for i in range(nh) for j in range(no)]).reshape(nh, no)
            pair_likelihoods = null + bf
            log_weights = np.r_[0., bf.ravel()] + log_prior
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError("Conditional likelihood arithmetic unavailable") from error
    if (not math.isfinite(null) or not np.isfinite(ratios).all() or not np.isfinite(bf).all()
            or not np.isfinite(pair_likelihoods).all() or np.isnan(log_weights).any()
            or np.isposinf(log_weights).any()):
        raise ValueError("Conditional likelihood overflowed")
    return RelationalLikelihoodEvidence(
        _owned(indices), hand_keys, object_keys, _owned(pair_supported), _owned(ratios), _owned(bf),
        null, _owned(pair_likelihoods), _owned(log_weights))
