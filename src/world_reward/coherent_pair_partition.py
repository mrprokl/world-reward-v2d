"""Label-free supported-group partition/VJP, not a loss, FIT or contact claim.

All native arrays and exact aliases are validated, never pruned. Torch is lazy;
caller owns device-buffer byte snapshots and source authentication. Only owned
CPU scalars/VJPs leave this primitive. Undefined support is not a valid loss.
"""
from dataclasses import dataclass
import math
import re

import numpy as np

from . import coherent_pair_marginal_objective as original
from . import coherent_route_scorer as core
from .coherent_pair_marginal import MarginalPairScores
from .interaction_tuple_evidence import _require, _sealed


@dataclass(frozen=True, eq=False)
class PartitionResult:
    log_partition: float | None
    gradient: np.ndarray              # Geometry17 A or scalar1 B; owned sealed FP64
    supported_groups: int
    status: str                       # supported / no_supported (undefined)
    score_reference: tuple            # Original distribution/parameter fingerprints
    arm: str
    temperature: float


def _torch_score(score, t):
    shape = tuple(score.supported.shape)
    n, o = len(score.identity['source_person_ids']), len(score.identity['source_object_ids'])
    fields = (('supported', shape, t.bool), ('native_supported', (n, 2, o), t.bool),
        ('native_route_supported', (n, 2, o), t.bool), ('scores_a', shape, t.float64),
        ('scores_b', shape, t.float64), ('native_scores_a', (n, 2, o), t.float64),
        ('native_scores_b', (n, 2, o), t.float64), ('geometry_derivatives_a', (*shape, 17), t.float64),
        ('geometry_derivatives_b', (*shape, 17), t.float64), ('alpha_derivatives_b', shape, t.float64))
    _require(type(score.device) is str and re.fullmatch(r'cpu|cuda:[0-9]+', score.device)
             and len(shape) == 2, 'Explicit device and complete group grid required')
    for name, size, dtype in fields:
        a = getattr(score, name)
        _require(type(a) is t.Tensor and a.dtype == dtype and tuple(a.shape) == size
                 and str(a.device) == score.device and not a.requires_grad,
                 'Original detached FP64/device/support ABI required')
    _require(not bool((score.native_route_supported & ~score.native_supported).any()),
             'Route support cannot invent base support')
    for name, good in (('scores_a', score.supported), ('scores_b', score.supported),
                       ('native_scores_a', score.native_supported), ('native_scores_b', score.native_supported)):
        a = getattr(score, name)
        _require(bool(t.isfinite(a[good]).all() & t.isnan(a[~good]).all()),
                 'Supported finite scores and raw unsupported NaN required')
    for name in ('geometry_derivatives_a', 'geometry_derivatives_b', 'alpha_derivatives_b'):
        a = getattr(score, name)
        _require(bool(t.isfinite(a).all()) and not bool(a[~score.supported].any()),
                 'Finite analytic VJP and zero unsupported derivative required')


def marginal_partition(score, theta, *, alpha=0., arm='A', backend='numpy'):
    """Return logsumexp(S) and its active-parameter derivative, WITHOUT labels.

    Temperature is already inside S; there is no extra division here. Centering
    derivatives requires adding the common row back (unlike a MIL difference).
    Empty supported populations return None/zeros with explicit no_supported.
    B exposes the right alpha0 derivative; A requires alpha0. No regularizer.
    """
    _require(type(theta) is np.ndarray and theta.dtype == np.float64 and theta.shape == (17,)
             and np.isfinite(theta).all(), 'Plain finite FP64 geometry17 required')
    _require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha >= 0
             and type(arm) is str and arm in ('A', 'B') and type(backend) is str
             and backend in ('numpy', 'torch'), 'Explicit arm/backend/nonnegative alpha required')
    _require(arm != 'A' or alpha == 0., 'Geometry arm requires alpha0')
    borrowed = theta; before = core._fingerprint(borrowed); theta = _sealed(theta)
    if backend == 'torch':
        from .coherent_pair_packed_torch import TorchMarginalPairScores
        expected = TorchMarginalPairScores
    else:
        expected = MarginalPairScores
    _require(type(score) is expected, 'Genuine marginal score type required')
    _require(type(score.temperature) is float and math.isfinite(score.temperature) and score.temperature > 0
             and type(score.alpha) is float and score.alpha == float(alpha), 'Exact finite scoring parameters required')
    _require(type(score.parameter_fingerprint) is str and score.parameter_fingerprint ==
             core._fingerprint((theta, score.temperature, float(alpha)))
             and type(score.distribution_fingerprint) is str
             and re.fullmatch('[0-9a-f]{64}', score.distribution_fingerprint), 'Scorer fingerprints required')
    original._identity(score)
    metadata = lambda: core._fingerprint((score.identity, score.distribution_fingerprint,
        score.parameter_fingerprint, score.temperature, score.alpha, getattr(score, 'device', None)))
    state = metadata() if backend == 'torch' else core._fingerprint(score)
    gradient = np.zeros(17 if arm == 'A' else 1, np.float64); logz = None
    if backend == 'numpy':
        original._numpy_score(score); count = int(score.supported.sum())
        if count:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                values = getattr(score, 'scores_'+arm.lower())[score.supported]
                d = score.geometry_derivatives_a[score.supported] if arm == 'A' else score.alpha_derivatives_b[score.supported].reshape(-1, 1)
                offset, correction, weight = original._lse_numpy(values)
                logz = float(offset+correction); gradient = d[0]+weight@(d-d[0])
    else:
        import torch as t
        with t.no_grad():
            _torch_score(score, t); count = int(score.supported.sum())
            if count:
                values = getattr(score, 'scores_'+arm.lower())[score.supported]
                d = score.geometry_derivatives_a[score.supported] if arm == 'A' else score.alpha_derivatives_b[score.supported].reshape(-1, 1)
                centered = d-d[0]; shifted = values-values.max(); weight = t.exp(shifted); total = weight.sum()
                _require(bool(t.isfinite(centered).all()) and bool(t.isfinite(shifted).all())
                         and bool(t.isfinite(total)) and bool(total > 0), 'Device partition overflow')
                z = values.max()+t.log(total); g = d[0]+(weight/total)@centered
                logz = float(z.detach().cpu().numpy()); gradient = np.array(g.detach().cpu().numpy(), copy=True)
    _require((logz is None or math.isfinite(logz)) and type(gradient) is np.ndarray
             and gradient.dtype == np.float64 and gradient.shape == (17 if arm == 'A' else 1,)
             and np.isfinite(gradient).all(), 'Finite partition/VJP required')
    _require(before == core._fingerprint(borrowed) and state ==
             (metadata() if backend == 'torch' else core._fingerprint(score)), 'Borrowed scorer/coefficients changed')
    return PartitionResult(logz, _sealed(gradient), count, 'supported' if count else 'no_supported',
                           (score.distribution_fingerprint, score.parameter_fingerprint), arm, score.temperature)
