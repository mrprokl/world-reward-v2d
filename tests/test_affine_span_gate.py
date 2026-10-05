"""Independent rational references; manufactured arrays only, no model I/O."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib

import numpy as np
import pytest

from world_reward.affine_span_gate import PRIME, affine_span_gate


def rational_rank(rows):
    """Tiny independent Gauss-Jordan over Q, not the production field rank."""
    rows = [[x if isinstance(x, Fraction) else Fraction(float(x)) for x in row] for row in rows]
    pivot_row = 0
    for j in range(len(rows[0])):
        candidates = [i for i in range(pivot_row, len(rows)) if rows[i][j] != 0]
        if not candidates:
            continue
        i = candidates[-1]
        rows[pivot_row], rows[i] = rows[i], rows[pivot_row]
        divisor = rows[pivot_row][j]
        rows[pivot_row] = [v / divisor for v in rows[pivot_row]]
        for k in range(len(rows)):
            if k != pivot_row:
                factor = rows[k][j]
                rows[k] = [v - factor * w for v, w in zip(rows[k], rows[pivot_row])]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_exact_nonmembership_and_affine_offset(dtype):
    a = np.array([[1, 0], [0, .5], [1, 1]], dtype)
    offset = np.array([.5, -.25, 1.], dtype)
    target = np.array([2.5, 1.25, 7.], dtype)
    result = affine_span_gate(a, offset, target)
    assert result.status == 'EXACT_NOT_IN_AFFINE_SPAN' and result.nonmembership_certified
    assert (result.rank_matrix, result.rank_augmented, result.prime) == (2, 3, 2147483647)
    assert rational_rank(a) == 2
    augmented = [list(row) + [Fraction(float(t)) - Fraction(float(o))] for row, o, t in zip(a, offset, target)]
    assert rational_rank(augmented) == 3
    assert result.membership_certified is False and result.adoption is False


def test_representable_target_is_only_inconclusive_never_membership_pass():
    a = np.array([[1., 0.], [0., 1.], [.5, .25]])
    o = np.array([.5, 2., -1.])
    target = o + a @ np.array([4., 2.])  # all dyadic operations exact here
    result = affine_span_gate(a, o, target)
    assert result.status == 'INCONCLUSIVE' and result.rank_matrix == result.rank_augmented == 2
    assert not result.membership_certified and not result.nonmembership_certified


def test_zero_target_dimension_68_by_28_exact_nonmembership():
    a = np.zeros((68, 28), np.float32)
    a[:28] = np.eye(28, dtype=np.float32)
    offset = np.zeros(68, np.float32)
    offset[28] = np.nextafter(np.float32(0.), np.float32(1.))
    result = affine_span_gate(a, offset, np.zeros(68, np.float64))
    assert result.nonmembership_certified and (result.rank_matrix, result.rank_augmented) == (28, 29)


def test_rank_drop_mod_prime_is_inconclusive_even_when_rationally_inconsistent():
    a = np.array([[float(PRIME)], [0.]], np.float64)
    o = np.zeros(2); t = np.array([0., 1.])
    assert rational_rank(a) == 1 and rational_rank([[float(PRIME), 0.], [0., 1.]]) == 2
    result = affine_span_gate(a, o, t)
    assert result.status == 'INCONCLUSIVE' and (result.rank_matrix, result.rank_augmented) == (0, 1)


def test_singular_over_rationals_does_not_claim_general_nonmembership():
    a = np.array([[1., 1.], [0., 0.], [2., 2.]])
    result = affine_span_gate(a, np.zeros(3), np.array([0., 1., 0.]))
    assert result.status == 'INCONCLUSIVE' and (result.rank_matrix, result.rank_augmented) == (1, 2)


def test_subnormal_and_extreme_dyadics_without_float_difference_overflow():
    tiny = np.nextafter(np.float64(0.), np.float64(1.))
    largest = np.finfo(np.float64).max
    result = affine_span_gate(np.array([[tiny], [0.]]), np.array([0., -largest]), np.array([tiny, largest]))
    assert result.nonmembership_certified and (result.rank_matrix, result.rank_augmented) == (1, 2)
    result = affine_span_gate(np.array([[tiny], [0.]], np.float32), np.zeros(2, np.float32), np.array([0., 1.]))
    assert result.status == 'INCONCLUSIVE'  # literal supplied F32 casts tiny to zero


def test_exact_offset_subtraction_not_rounded_before_field_conversion():
    # 1 - tiny rounds to 1 in F64. Exact dyadic difference must still be used.
    tiny = np.nextafter(np.float64(0.), np.float64(1.))
    a = np.array([[1.], [1.]])
    result = affine_span_gate(a, np.array([tiny, 0.]), np.array([1., 1.]))
    assert result.nonmembership_certified
    assert rational_rank([[1., Fraction(1) - Fraction(float(tiny))], [1., Fraction(1)]]) == 2


def test_signed_zero_and_mixed_original_dtypes():
    a = np.array([[1.], [-0.]], np.float32)
    result = affine_span_gate(a, np.array([-0., 0.], np.float64), np.array([1., 1.], np.float32))
    assert result.nonmembership_certified and result.input_identities[0][2] == '<f4'


def test_permutations_and_exact_dyadic_unit_changes_preserve_certificate():
    a = np.array([[1., 0.], [0., 1.], [.25, .5]])
    o, t = np.array([1., 2., 3.]), np.array([2., 3., 9.])
    baseline = affine_span_gate(a, o, t)
    for rows in ([2, 0, 1], [1, 2, 0]):
        result = affine_span_gate(a[rows, ::-1] * 8, o[rows] * 8, t[rows] * 8)
        assert (result.status, result.rank_matrix, result.rank_augmented) == (baseline.status, 2, 3)


def test_borrowed_inputs_unchanged_and_result_identity_immutable():
    a = np.array([[1., 2.], [0., 1.], [0., 0.]])[:, ::-1]
    o, t = np.zeros(3), np.array([0., 0., 1.])
    before = (a.tobytes(), o.tobytes(), t.tobytes())
    result = affine_span_gate(a, o, t)
    assert before == (a.tobytes(), o.tobytes(), t.tobytes())
    assert tuple(row[3] for row in result.input_identities) == tuple(hashlib.sha256(x).hexdigest() for x in before)
    identity = result.input_identities
    a[:] = 8; o[:] = 9; t[:] = 10
    assert result.input_identities == identity
    with pytest.raises(FrozenInstanceError): result.status = 'PASS'
    with pytest.raises(TypeError): result.input_identities[0][1][0] = 8


@pytest.mark.parametrize('bad', [np.array([[1]], np.int64), np.array([[True]]), np.array([[np.nan]]),
    np.array([[np.inf]]), np.ma.array([[1.]], mask=False), np.array([[1.]], object), [[1.]],
    np.zeros((0, 1)), np.zeros((2, 0)), np.zeros(2), np.zeros((2, 1, 1))])
def test_invalid_matrix_dtype_finites_or_shape_stops(bad):
    with pytest.raises(ValueError): affine_span_gate(bad, np.zeros(2), np.zeros(2))


@pytest.mark.parametrize('bad', [np.zeros((2, 1)), np.zeros(3), np.array([0., np.inf]),
    np.ma.array([0., 0.], mask=False), np.zeros(2, np.int64)])
def test_invalid_vector_contract_stops(bad):
    for o, t in ((bad, np.zeros(2)), (np.zeros(2), bad)):
        with pytest.raises(ValueError): affine_span_gate(np.ones((2, 1)), o, t)


def test_deterministic_small_certificates_agree_with_independent_rational_rank():
    rng = np.random.default_rng(8401)
    for _ in range(24):
        a = rng.integers(-4, 5, (4, 2)).astype(np.float64) / 4
        o, t = rng.integers(-4, 5, (2, 4)).astype(np.float64) / 8
        result = affine_span_gate(a, o, t)
        if result.nonmembership_certified:
            assert rational_rank(a) == 2
            assert rational_rank([list(row) + [Fraction(float(y)) - Fraction(float(x))]
                                  for row, x, y in zip(a, o, t)]) == 3
