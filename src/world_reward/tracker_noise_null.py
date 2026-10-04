"""External tracking-error dependence diagnostic, not an association model.

Input errors are frozen prediction minus uncertain annotation in original pixel
units. Original point-index parity gives nonsemantic even/odd groups. Adjacent
error increments use only jointly visible endpoints strictly AFTER each original
query; missing samples never become zero or a replacement query. Model B is one
Gaussian4D fitted on all eligible cross-group pairs; A has identical means and
marginal covariances, with its cross block zero. Shared points/times are NOT IID:
this is a weighted composite density diagnostic without calibrated inference.
Heteroskedasticity, shared annotation errors and temporal dependence can explain
gains; they do not prove camera error, physical identity/contact or a universal
null for hand/object innovations. Labels, splits and prediction seals are caller
responsibilities; no I/O, fitting after heldout, alignment or threshold here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _key(value):
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError("Explicit bounded clip key required")
    return value


@dataclass(frozen=True, eq=False)
class TrackerErrorClip:
    clip_key: str
    frame_index: np.ndarray
    point_indices: np.ndarray
    query_frame_index: np.ndarray
    errors: np.ndarray
    observed: np.ndarray

    def __post_init__(self):
        _key(self.clip_key)
        values = (self.frame_index, self.point_indices, self.query_frame_index, self.errors, self.observed)
        if any(np.ma.isMaskedArray(v) for v in values):
            raise ValueError("Unmasked original arrays required")
        frames, points, queries, errors, support = map(np.asarray, values)
        if (frames.dtype != np.int64 or frames.ndim != 1 or len(frames) < 3
                or not np.array_equal(frames, np.arange(len(frames), dtype=np.int64))
                or points.dtype != np.int64 or points.ndim != 1 or len(points) > 32
                or np.any(points < 0) or np.any(points >= 32) or np.any(np.diff(points) <= 0)
                or queries.dtype != np.int64 or queries.shape != points.shape
                or np.any(queries < 0) or np.any(queries >= len(frames))
                or errors.dtype.kind != "f" or errors.shape != (len(points), len(frames), 2)
                or support.dtype != np.bool_ or support.shape != errors.shape[:2]
                or not np.isfinite(errors[support]).all()):
            raise ValueError("Full arange(T), sorted original indices0..31, query frames and supported finite XY errors required")
        for name, value in zip(("frame_index", "point_indices", "query_frame_index", "errors", "observed"), values):
            object.__setattr__(self, name, _owned(np.asarray(value)))


def _increments(clip):
    if type(clip) is not TrackerErrorClip:
        raise ValueError("Explicit frozen TrackerErrorClip required")
    support = clip.observed[:, :-1] & clip.observed[:, 1:]
    support &= np.arange(len(clip.frame_index) - 1)[None, :] > clip.query_frame_index[:, None]
    values = np.full((len(clip.point_indices), len(clip.frame_index) - 1, 2), np.nan)
    with np.errstate(over="raise", invalid="raise"):
        try:
            values[support] = clip.errors[:, 1:][support] - clip.errors[:, :-1][support]
        except FloatingPointError as error:
            raise ValueError("Observed error increments overflowed") from error
    return values, support


def _pairs(clip, lag):
    values, support = _increments(clip)
    even = np.flatnonzero(clip.point_indices % 2 == 0)
    odd = np.flatnonzero(clip.point_indices % 2 == 1)
    span = max(0, len(clip.frame_index) - 1 - lag)
    result = []
    for i in even:
        for j in odd:
            valid = support[i, :span] & support[j, lag:lag + span]
            times = np.flatnonzero(valid)
            rows = np.c_[values[i, times], values[j, times + lag]]
            result.append((int(clip.point_indices[i]), int(clip.point_indices[j]), times, rows))
    return tuple(result)


def _factor(covariance):
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("Symmetric Gaussian covariance required")
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("Fitted covariance is not positive definite; no jitter/fallback") from error
    if not np.isfinite(factor).all() or np.any(np.diag(factor) <= 0):
        raise ValueError("Gaussian covariance factorization unavailable")
    return factor


@dataclass(frozen=True, eq=False)
class TrackerNoiseNull:
    fit_clip_keys: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    independent_covariance: np.ndarray
    fit_sample_counts: tuple[int, ...]
    fit_supported_pair_counts: tuple[int, ...]

    def __post_init__(self):
        mean, covariance, independent = map(np.asarray, (self.mean, self.covariance, self.independent_covariance))
        if (type(self.fit_clip_keys) is not tuple or not self.fit_clip_keys
                or len(set(self.fit_clip_keys)) != len(self.fit_clip_keys)
                or any(_key(k) != k for k in self.fit_clip_keys)
                or mean.shape != (4,) or mean.dtype != np.float64 or not np.isfinite(mean).all()
                or covariance.shape != (4, 4) or covariance.dtype != np.float64
                or independent.shape != (4, 4) or independent.dtype != np.float64
                or not np.isfinite(covariance).all() or not np.isfinite(independent).all()
                or len(self.fit_sample_counts) != len(self.fit_clip_keys)
                or len(self.fit_supported_pair_counts) != len(self.fit_clip_keys)
                or any(type(n) is not int or n <= 0 for n in (*self.fit_sample_counts, *self.fit_supported_pair_counts))):
            raise ValueError("Explicit fitted Gaussian and full fit-clip counts required")
        expected = covariance.copy(); expected[:2, 2:] = 0; expected[2:, :2] = 0
        if not np.array_equal(independent, expected):
            raise ValueError("A must reuse B marginal covariances with cross block zero")
        _factor(covariance); _factor(independent)
        for name, value in (("mean", mean), ("covariance", covariance), ("independent_covariance", independent)):
            object.__setattr__(self, name, _owned(value))


def fit_tracker_noise_null(fit_clips):
    """Weighted Gaussian MLE: equal clip, then eligible pair, then complete interval.

    All deterministic pairs remain in diagnostics; unsupported pairs have no
    density sample, not zero. Every fit clip needs at least one supported pair,
    pooled complete sample count >4, and SPD MLE without shrink/jitter/retry.
    Fit-only weights prevent a longer/more-paired clip dominating by row count;
    repeated rows are not claimed independent statistical samples.
    """
    if (type(fit_clips) is not tuple or not fit_clips
            or any(type(c) is not TrackerErrorClip for c in fit_clips)
            or len({c.clip_key for c in fit_clips}) != len(fit_clips)):
        raise ValueError("Distinct frozen training clips required")
    samples, weights, counts, pair_counts = [], [], [], []
    for clip in fit_clips:
        pairs = _pairs(clip, 0)
        eligible = [rows for _, _, _, rows in pairs if len(rows)]
        if not eligible:
            raise ValueError("Each fit clip needs complete cross-group samples")
        counts.append(sum(len(rows) for rows in eligible)); pair_counts.append(len(eligible))
        for rows in eligible:
            samples.append(rows)
            weights.append(np.full(len(rows), 1. / (len(fit_clips) * len(eligible) * len(rows))))
    if sum(counts) <= 4:
        raise ValueError("More than four complete Gaussian4D samples required")
    values, weight = np.concatenate(samples), np.concatenate(weights)
    mass = float(weight.sum())
    with np.errstate(over="raise", invalid="raise"):
        try:
            mean = (weight[:, None] * values).sum(axis=0) / mass
            centered = values - mean
            # One scalar reduction per unordered entry, mirrored exactly. This
            # computes the MLE, not a post-hoc covariance repair or shrinkage.
            covariance = np.empty((4, 4))
            for i in range(4):
                for j in range(i, 4):
                    covariance[i, j] = covariance[j, i] = np.sum(weight * centered[:, i] * centered[:, j]) / mass
        except FloatingPointError as error:
            raise ValueError("Fit Gaussian arithmetic overflowed") from error
    independent = covariance.copy(); independent[:2, 2:] = 0; independent[2:, :2] = 0
    return TrackerNoiseNull(tuple(c.clip_key for c in fit_clips), mean, covariance, independent,
                            tuple(counts), tuple(pair_counts))


def _log_density(rows, mean, covariance):
    factor = _factor(covariance)
    whitened = np.linalg.solve(factor, (rows - mean).T).T
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = -.5 * (4 * math.log(2 * math.pi) + 2 * np.log(np.diag(factor)).sum()
                            + np.sum(whitened * whitened, axis=1))
        except FloatingPointError as error:
            raise ValueError("Heldout Gaussian density overflowed") from error
    if not np.isfinite(result).all():
        raise ValueError("Heldout Gaussian density unavailable")
    return result


def evaluate_tracker_noise_null(model, clip):
    """Frozen heldout same-model synchronous/unwrapped shifted paired density gain.

    Shift is floor(original T/3) on odd-group increments; no wrap/reindexing.
    Means are equal per supported pair, with raw count-weighted mean separate.
    Full pair*(T-1) denominator retained; shift out-of-range and unavailable
    query/visibility intervals remain counted, not silently dropped. No fit-clip
    reuse, winning thresholds, p-values, identity, contact or model adoption.
    """
    if type(model) is not TrackerNoiseNull or type(clip) is not TrackerErrorClip:
        raise ValueError("Explicit fitted model and frozen heldout clip required")
    if clip.clip_key in model.fit_clip_keys:
        raise ValueError("Heldout clip must not occur in fitting")
    steps = len(clip.frame_index) - 1
    report = dict(clip_key=clip.clip_key, original_frames=len(clip.frame_index),
                  original_point_indices=clip.point_indices.tolist(), query_frame_indices=clip.query_frame_index.tolist(),
                  error_kind="adjacent_prediction_minus_annotation_XY_error_increment",
                  after_query="both endpoints strictly after original query frame",
                  fit_clip_keys=list(model.fit_clip_keys), composite_density_only=True,
                  independent_samples_claimed=False, calibrated_association=False,
                  physical_identity_verified=False, contact_verified=False, adoption=False)
    for name, lag in (("synchronous", 0), ("shifted", len(clip.frame_index) // 3)):
        pairs = _pairs(clip, lag)
        rows, pair_means, gains = [], [], []
        for i, j, times, values in pairs:
            gain = _log_density(values, model.mean, model.covariance) - _log_density(values, model.mean, model.independent_covariance)
            if not np.isfinite(gain).all():
                raise ValueError("Paired density gain overflowed")
            mean = float(gain.mean()) if len(gain) else None
            rows.append(dict(even_original_index=i, odd_original_index=j, complete_samples=len(values),
                             missing_samples=steps - len(values), mean_log_likelihood_gain=mean))
            if len(gain):
                gains.append(gain); pair_means.append(mean)
        count = sum(len(g) for g in gains)
        paired_mean = float(np.mean(pair_means)) if pair_means else None
        sample_mean = float(np.concatenate(gains).mean()) if count else None
        if any(value is not None and not math.isfinite(value) for value in (paired_mean, sample_mean)):
            raise ValueError("Pooled density gain overflowed")
        report[name] = dict(lag_original_frames=lag, original_intervals=steps,
                            cross_group_pairs=len(pairs), supported_pairs=len(pair_means),
                            possible_pair_intervals=len(pairs) * steps, complete_samples=count,
                            missing_samples=len(pairs) * steps - count,
                            out_of_range_shift_samples=len(pairs) * min(lag, steps),
                            mean_log_likelihood_gain=paired_mean,
                            sample_weighted_mean_gain=sample_mean,
                            pairs=rows)
    # Descriptive contrast on identical (original pair, anchor t) support. The
    # shifted odd context still changes: this is NOT a causal/significance gate.
    synchronous, shifted = _pairs(clip, 0), _pairs(clip, len(clip.frame_index) // 3)
    rows, pair_means, deltas = [], [], []
    for sync, shift in zip(synchronous, shifted):
        i, j, sync_times, sync_values = sync
        si, sj, shift_times, shift_values = shift
        if (i, j) != (si, sj):
            raise ValueError("Matched contrast changed original point pair")
        _, sync_ids, shift_ids = np.intersect1d(sync_times, shift_times, assume_unique=True, return_indices=True)
        x, y = sync_values[sync_ids], shift_values[shift_ids]
        gs = _log_density(x, model.mean, model.covariance) - _log_density(x, model.mean, model.independent_covariance)
        gl = _log_density(y, model.mean, model.covariance) - _log_density(y, model.mean, model.independent_covariance)
        delta = gs - gl
        if not np.isfinite(delta).all():
            raise ValueError("Matched anchor contrast overflowed")
        mean = float(delta.mean()) if len(delta) else None
        rows.append(dict(even_original_index=i, odd_original_index=j, complete_samples=len(delta),
                         missing_samples=steps - len(delta), mean_log_likelihood_delta=mean))
        if len(delta):
            deltas.append(delta); pair_means.append(mean)
    count = sum(len(value) for value in deltas)
    pair_mean = float(np.mean(pair_means)) if pair_means else None
    sample_mean = float(np.concatenate(deltas).mean()) if count else None
    if any(value is not None and not math.isfinite(value) for value in (pair_mean, sample_mean)):
        raise ValueError("Matched anchor mean overflowed")
    report["matched_anchor_shift"] = dict(
        descriptive_only=True, contrast="synchronous_gain_minus_shifted_gain_same_pair_and_anchor",
        lag_original_frames=len(clip.frame_index) // 3, original_intervals=steps,
        cross_group_pairs=len(synchronous), supported_pairs=len(pair_means),
        possible_pair_intervals=len(synchronous) * steps, complete_samples=count,
        missing_samples=len(synchronous) * steps - count,
        out_of_range_shift_samples=len(synchronous) * min(len(clip.frame_index) // 3, steps),
        mean_log_likelihood_delta=pair_mean, sample_weighted_mean_delta=sample_mean, pairs=rows)
    return report
