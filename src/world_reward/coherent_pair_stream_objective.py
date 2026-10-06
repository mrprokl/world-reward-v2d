"""Ordered, fixed-population marginal loss/VJP; no I/O, optimizer or FIT.

Each complete image score is consumed once through the original objective.
Only owned CPU scalars/VJPs and diagnostics survive the row. Caller authenticates
image IDs, references and device snapshots; UNKNOWN alternatives are not negatives.
"""
import math

import numpy as np

from . import coherent_pair_marginal_objective as original
from . import coherent_route_scorer as core
from .interaction_tuple_evidence import _require, _sealed


def _cpu(value, backend):
    if backend == 'torch':
        value = value.detach().cpu().numpy()
    result = np.array(value, copy=True)
    _require(result.dtype == np.float64, 'Original FP64 objective result required')
    return result


def ordered_stream_marginal_objective(rows, record_ids, theta, *, alpha=0.,
                                     regularization=0., arm='A', backend='numpy'):
    """Consume exactly ordered ``(record_id, genuine_score, full_mask)`` rows.

    Undefined per-record loss remains NaN; its explicit zero numerator retains
    the record in the fixed denominator. Ridge applies once to theta (A) or
    alpha (B). A requires alpha0. B's frozen geometry is a caller obligation.
    Torch is lazy through the original objective; this return is owned CPU FP64.
    An incomplete, extra, reordered or invalid stream yields no partial result.
    """
    _require(type(record_ids) is tuple and len(record_ids) > 0
             and all(type(x) is str and 0 < len(x) <= 256
                     and all(32 <= ord(c) != 127 for c in x) for x in record_ids)
             and len(set(record_ids)) == len(record_ids),
             'Nonempty unique ordered record IDs required')
    _require(type(theta) is np.ndarray and theta.dtype == np.float64
             and theta.shape == (17,) and np.isfinite(theta).all(),
             'Plain finite FP64 geometry17 required')
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha >= 0
             and type(regularization) in (int, float) and math.isfinite(regularization)
             and regularization >= 0 and type(arm) is str and arm in ('A', 'B')
             and type(backend) is str and backend in ('numpy', 'torch'),
             'Explicit arm/backend and nonnegative finite parameters required')
    _require(arm != 'A' or alpha == 0., 'Geometry arm requires alpha0')
    borrowed = theta; before = core._fingerprint(borrowed); theta = _sealed(theta)
    try:
        iterator = iter(rows)
    except TypeError:
        raise ValueError('One-pass row iterator required') from None
    _require(iterator is rows, 'One-pass row iterator required')
    alpha, lam = float(alpha), float(regularization)
    function = original.marginal_objective if backend == 'numpy' else original.marginal_objective_torch
    parameters = theta if arm == 'A' else np.array([alpha], np.float64)
    gradient = np.zeros(len(parameters), np.float64)
    losses, contributions, counts, references = [], [], [], []
    temperature = device = None; absent = object()
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for expected in record_ids:
            row = next(iterator, absent)
            _require(type(row) is tuple and len(row) == 3 and type(row[0]) is str
                     and row[0] == expected, 'Exact complete ordered row required')
            _, score, mask = row
            value = function((score,), (mask,), theta, alpha=alpha,
                             regularization=0., arm=arm)
            if temperature is None:
                temperature = score.temperature
                if backend == 'torch':
                    device = score.device
            _require(score.temperature == temperature, 'One uniform stream temperature required')
            _require(backend != 'torch' or score.device == device,
                     'One explicit stream scorer device required')
            loss = _cpu(value.loss, backend); derivative = _cpu(value.gradient, backend)
            record_loss = _cpu(value.record_losses, backend)
            _require(loss.shape == () and np.isfinite(loss)
                     and derivative.shape == gradient.shape and np.isfinite(derivative).all()
                     and record_loss.shape == (1,) and not np.isinf(record_loss).any(),
                     'Finite original loss/VJP and explicit record diagnostic required')
            contributions.append(float(loss)); gradient += derivative
            losses.append(float(record_loss[0])); c = value.counts
            counts.append((c['positive_groups'], c['supported_positive_groups'], c['supported_groups']))
            references.append(value.score_references[0])
            del row, score, mask, value, loss, derivative, record_loss
        _require(next(iterator, absent) is absent, 'Extra row outside fixed population')
        loss = math.fsum(contributions)/len(record_ids)
        if lam:
            loss += .5*lam*float(parameters @ parameters)
        gradient = gradient/len(record_ids)+lam*parameters
    _require(math.isfinite(loss) and np.isfinite(gradient).all(), 'Finite full-stream objective/VJP required')
    _require(before == core._fingerprint(borrowed), 'Borrowed geometry changed during stream')
    statuses, total, status = original._counts(counts)
    return original.MarginalObjective(loss, _sealed(gradient), _sealed(np.array(losses, np.float64)),
        statuses, total, status, arm, lam, tuple(references))
