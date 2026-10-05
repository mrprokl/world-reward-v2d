"""Limited one-sparse equality reference proposal, never native qualification.

The distinct supported equality rows fix one coefficient each; unconstrained
coefficients are zero. A float64 subtraction/division is stored in float32 ONCE.
The result is a rounded proposal, not a certified real minimum-norm solution.
CPU NumPy ``mean + coefficients @ components`` and literal bounds are diagnosed;
CUDA GEMM/FMA may differ. No fitting, projection, clipping, tolerance or search.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


class UnsupportedScaleReference(ValueError):
    """Outside this narrow policy; no dense/duplicate/QP fallback is attempted."""


def _frozen(value, shape, name, *, finite=True):
    if (not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value)
            or value.dtype != np.float32 or value.shape != shape):
        raise ValueError(name + ': original unmasked float32 shape required')
    result = np.frombuffer(value.tobytes(order='C'), dtype=np.float32).reshape(shape)
    if np.isnan(result).any() or finite and not np.isfinite(result).all():
        raise ValueError(name + ': original finite values required')
    return result


def _identity(name, array):
    return name, tuple(array.shape), array.dtype.str, hashlib.sha256(array.tobytes(order='C')).hexdigest()


@dataclass(frozen=True, eq=False)
class ScaleReferenceProposal:
    coefficients: np.ndarray  # (28,), immutable once-rounded float32 proposal
    cpu_expanded_scale: np.ndarray  # (68,), NumPy CPU float32 expression only
    fixed_coordinates: tuple[int, ...]
    fixed_coefficient_indices: tuple[int, ...]  # -1 denotes already-consistent zero row
    violating_coordinates: tuple[int, ...]
    fixed_residuals_float64: tuple[float, ...]
    input_identities: tuple
    output_identities: tuple
    cpu_literal_bounds_passed: bool
    status: str
    proposal_only: bool = True
    native_expansion_verified: bool = False
    rounded_coefficients_minimum_norm_certified: bool = False
    adoption: bool = False
    policy: str = 'distinct_one_sparse_equalities_float64_ratio_float32_once_other_coefficients_zero'
    expansion_semantics: str = 'CPU_NumPy_float32_mean_plus_coefficients_matmul_components_not_CUDA'


def propose_scale_reference(components, mean, bounds) -> ScaleReferenceProposal:
    """Preserve a failed candidate and literal residuals, never try neighbors.

    Inputs are original components[28,68], mean[68], bounds[68,2], all F32.
    Bounds may be infinite. Every equality (lower==upper) must be a zero row
    already satisfied, or exactly one nonzero component with distinct support.
    All 68 inequalities are tested on the once-rounded CPU expansion. That
    diagnostic does not authorize a native reference or reopen a closed cohort.
    """
    c = _frozen(components, (28, 68), 'components')
    m = _frozen(mean, (68,), 'mean')
    b = _frozen(bounds, (68, 2), 'bounds', finite=False)
    if np.any(b[:, 0] > b[:, 1]):
        raise ValueError('Literal lower bound exceeds upper bound')
    fixed = np.flatnonzero(b[:, 0] == b[:, 1])
    if not len(fixed) or not np.isfinite(b[fixed]).all():
        raise UnsupportedScaleReference('Finite nonempty fixed-coordinate equalities required')
    coefficients = np.zeros(28, np.float32)
    supports, used = [], set()
    for coordinate in fixed:
        support = np.flatnonzero(c[:, coordinate] != 0)
        if not len(support):
            if m[coordinate] != b[coordinate, 0]:
                raise UnsupportedScaleReference('Immutable zero-component coordinate violates fixed bound')
            supports.append(-1)
            continue
        if len(support) != 1:
            raise UnsupportedScaleReference('Dense equality is unsupported; no QP fallback')
        index = int(support[0])
        if index in used:
            raise UnsupportedScaleReference('Duplicate equality coefficient is unsupported')
        used.add(index); supports.append(index)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            ratio = (np.float64(b[coordinate, 0]) - np.float64(m[coordinate])) / np.float64(c[index, coordinate])
            coefficients[index] = np.float32(ratio)
        if not np.isfinite(coefficients[index]):
            raise UnsupportedScaleReference('Single coefficient conversion is outside finite float32')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        expanded = m + coefficients @ c
    violations = np.flatnonzero(~np.isfinite(expanded) | (expanded < b[:, 0]) | (expanded > b[:, 1]))
    coefficients = _frozen(coefficients, (28,), 'proposal')
    # Nonfinite arithmetic is retained as a failed diagnostic, not filled with zero.
    expanded = np.frombuffer(expanded.tobytes(), dtype=np.float32).reshape(68)
    passed = not len(violations)
    return ScaleReferenceProposal(coefficients, expanded, tuple(map(int, fixed)), tuple(supports),
        tuple(map(int, violations)), tuple(float(np.float64(expanded[i]) - np.float64(b[i, 0])) for i in fixed),
        tuple(_identity(n, a) for n, a in (('components', c), ('mean', m), ('bounds', b))),
        tuple(_identity(n, a) for n, a in (('coefficients', coefficients), ('cpu_expanded_scale', expanded))),
        passed, 'PROPOSAL_CPU_BOUNDS_PASS' if passed else 'PROPOSAL_CPU_BOUNDS_FAIL')
