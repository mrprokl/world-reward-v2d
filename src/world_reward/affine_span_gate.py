"""One-prime exact NON-membership certificate for an IEEE affine system.

Given offset + matrix @ x = target over Q, finite F32/F64 inputs are interpreted
as their exact dyadic values. No floating subtraction, fit, SVD or projection is
performed. Full column rank modulo the fixed odd prime proves full column rank
over Q; one higher augmented rank then proves inconsistency over Q. Every other
outcome is INCONCLUSIVE, never a membership or reconstruction certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

PRIME = 2147483647


@dataclass(frozen=True)
class AffineSpanResult:
    status: str
    rank_matrix: int
    rank_augmented: int
    rows: int
    columns: int
    input_identities: tuple
    prime: int = PRIME
    membership_certified: bool = False
    adoption: bool = False
    interpretation: str = 'exact_dyadic_rational_system_single_fixed_prime_no_approximate_fit'

    @property
    def nonmembership_certified(self):
        return self.status == 'EXACT_NOT_IN_AFFINE_SPAN'


def _frozen(value, label):
    if (not isinstance(value, np.ndarray) or np.ma.isMaskedArray(value)
            or value.dtype not in (np.dtype('float32'), np.dtype('float64'))):
        raise ValueError(label + ': finite original unmasked F32/F64 array required')
    frozen = np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)
    if not np.isfinite(frozen).all():
        raise ValueError(label + ': finite original unmasked F32/F64 array required')
    return frozen


def _residue(value):
    # Promotion of every binary32 value to binary64 is exact.
    numerator, denominator = float(value).as_integer_ratio()
    return (numerator % PRIME) * pow(denominator % PRIME, -1, PRIME) % PRIME


def _rank(rows):
    """Gaussian elimination in F_p, no rank tolerance or floating arithmetic."""
    a = [row[:] for row in rows]
    rank = 0
    for column in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inverse = pow(a[rank][column], -1, PRIME)
        a[rank] = [(x * inverse) % PRIME for x in a[rank]]
        for i in range(rank + 1, len(a)):
            factor = a[i][column]
            if factor:
                a[i] = [(x - factor * y) % PRIME for x, y in zip(a[i], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def affine_span_gate(matrix, offset, target):
    """Return a certificate or INCONCLUSIVE; inputs and their row order survive.

    matrix is [M,N], offset/target are [M], M,N>0. Independent original float
    dtypes are allowed. Read-only owned byte snapshots fix the represented data
    before elimination; returned identities contain no caller-owned mutable data.
    Coefficients x are unrestricted rationals, not fitted floating parameters.
    """
    a, o, t = (_frozen(v, label) for v, label in
               ((matrix, 'matrix'), (offset, 'offset'), (target, 'target')))
    if (a.ndim != 2 or min(a.shape) == 0 or o.shape != (a.shape[0],)
            or t.shape != (a.shape[0],)):
        raise ValueError('Nonempty matrix[M,N] and offset/target[M] required')
    rows = [[_residue(x) for x in row] for row in a]
    # Exact field subtraction, NOT rounded target-offset in F32/F64.
    augmented = [row + [(_residue(y) - _residue(x)) % PRIME]
                 for row, x, y in zip(rows, o, t)]
    ra, rab = _rank(rows), _rank(augmented)
    status = 'EXACT_NOT_IN_AFFINE_SPAN' if ra == a.shape[1] and rab == ra + 1 else 'INCONCLUSIVE'
    identities = tuple((label, tuple(value.shape), value.dtype.str,
                        hashlib.sha256(value.tobytes(order='C')).hexdigest())
                       for label, value in (('matrix', a), ('offset', o), ('target', t)))
    return AffineSpanResult(status, ra, rab, a.shape[0], a.shape[1], identities)
