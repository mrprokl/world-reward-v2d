"""Data-free source-chart controls, not native/compiler or mesh qualification."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import numpy as np
import pytest

from world_reward.mesh_conditioning_v2 import (
    POLICY_SHA256, prepare_conditioning_v2, sterbenz_subtraction_certified,
)


def mesh():
    v = np.array([[1., -4., 0.], [2., -4., 0.], [1., -2., 2.], [1., -4., 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return v, f


def test_unique_sterbenz_or_zero_axis_choice_then_exact_whole_source_inverse():
    v, f = mesh()
    before = v.tobytes(), f.tobytes()
    c = prepare_conditioning_v2(v, f)
    np.testing.assert_array_equal(c.origin, [1.5, -3., 0.])
    assert c.origin_modes == ("sterbenz_midpoint", "sterbenz_midpoint", "zero")
    assert c.scale == 2. and c.scale_exponent == 1
    assert c.decode(c.encode(v)).tobytes() == v.tobytes()
    assert (v.tobytes(), f.tobytes()) == before
    assert c.diagnostics["all_source_roundtrip_vertices"] == 4
    assert c.diagnostics["policy_sha256"] == POLICY_SHA256
    for k in ("origin_search_performed", "source_arrays_modified", "source_geometry_repaired",
              "physical_geometry_rescaled", "physical_serialization_verified",
              "metric_fidelity_verified", "topology_verified", "native_backend_qualified", "adopted"):
        assert c.diagnostics[k] is False


@pytest.mark.parametrize("origin,values,expected", [
    (2., [1., 2., 4.], True), (2., [np.nextafter(1., 0.), 2., 4.], False),
    (2., [1., 2., np.nextafter(4., np.inf)], False), (-2., [-4., -2., -1.], True),
    (2., [0., 2., 4.], False), (2., [-1., 2., 4.], False), (0., [-2., 0., 1.], True),
])
def test_sufficient_condition_has_exact_boundaries_and_same_sign(origin, values, expected):
    a = np.array(values, np.float64)
    assert sterbenz_subtraction_certified(a, origin) is expected
    if expected:
        for x in a:
            assert Fraction(float(x-origin)) == Fraction(float(x))-Fraction(origin)


def test_sterbenz_subnormal_boundary_is_rational_not_rounded_half():
    tiny = math.ldexp(1., -1074)
    # Rounded y/2 is 2*tiny, but exact y/2 is 2.5*tiny: x=2*tiny is NOT sufficient.
    assert float(5*tiny*.5) == 2*tiny
    assert not sterbenz_subtraction_certified(np.array([2*tiny, 5*tiny]), float(5*tiny))
    assert sterbenz_subtraction_certified(np.array([3*tiny, 5*tiny]), float(5*tiny))


@pytest.mark.parametrize("exponent", [-500, -16, 0, 16, 500])
def test_representable_power_of_two_unit_change_is_exact(exponent):
    v, f = mesh()
    old = prepare_conditioning_v2(v, f)
    scaled = np.ldexp(v, exponent)
    new = prepare_conditioning_v2(scaled, f)
    assert new.origin_modes == old.origin_modes
    np.testing.assert_array_equal(new.origin, np.ldexp(old.origin, exponent))
    assert new.scale_exponent == old.scale_exponent+exponent
    assert new.encode(scaled).tobytes() == old.encode(v).tobytes()
    assert new.decode(new.encode(scaled)).tobytes() == scaled.tobytes()


def test_cross_zero_source_keeps_small_nonzero_and_signed_zero_without_snap():
    v, f = mesh()
    v[:, 0] = [-1., 1., 1e-300, -0.]
    c = prepare_conditioning_v2(v, f)
    assert c.origin[0] == 0 and c.origin_modes[0] == "zero"
    assert c.decode(c.encode(v)).tobytes() == v.tobytes()
    assert c.diagnostics["roundtrip_byte_exact"]


def test_minimal_subnormal_extent_has_exact_ldexp_chart():
    v, f = mesh()
    tiny = math.ldexp(1., -1074)
    v[:] = [[0., 0., 0.], [tiny, 0., 0.], [0., tiny, 0.], [0., 0., tiny]]
    c = prepare_conditioning_v2(v, f)
    assert c.scale_exponent == -1074 and c.scale == tiny
    assert c.origin_modes == ("zero",)*3
    assert c.decode(c.encode(v)).tobytes() == v.tobytes()


def test_range_loss_fails_after_choice_without_alternate_origin():
    v, f = mesh()
    v[0, 2] = math.ldexp(1., -1074)  # Scale=2 would discard this source coordinate.
    with pytest.raises(ValueError, match="underflow"):
        prepare_conditioning_v2(v, f)
    c = prepare_conditioning_v2(*mesh())
    with pytest.raises(ValueError, match="overflow"):
        c.decode(np.full((1, 3), np.finfo(np.float64).max))
    # Nonzero subnormal rounding need not become zero; exact-inverse check is separate.
    v[0, 2] = 3*math.ldexp(1., -1074)
    with pytest.raises(ValueError, match="inverse not exact"):
        prepare_conditioning_v2(v, f)


def test_sterbenz_selected_midpoint_range_failure_never_reselects_zero():
    v, f = mesh()
    v[:, 0] = [1e-300, 2e-300, 1e-300, 1e-300]
    v[:, 1] = [0., math.ldexp(1., 1023), 0., 0.]
    candidate = float(v[:, 0].min()+(v[:, 0].max()-v[:, 0].min())*.5)
    assert sterbenz_subtraction_certified(v[:, 0], candidate)
    with pytest.raises(ValueError, match="underflow"):
        prepare_conditioning_v2(v, f)


def test_orphan_rows_affect_certification_not_bbox_and_are_never_removed():
    v, f = mesh()
    c = prepare_conditioning_v2(v, f)
    extended = np.r_[v, [[-1., -3., .5]]]
    other = prepare_conditioning_v2(extended, f)
    assert other.scale == c.scale
    assert other.origin_modes[0] == "zero" and other.origin[0] == 0
    assert other.diagnostics["all_source_roundtrip_vertices"] == 5
    assert other.encode(extended).shape == extended.shape
    assert other.decode(other.encode(extended)).tobytes() == extended.tobytes()


def test_immutable_copies_and_no_caller_array_alias():
    v, f = mesh()
    c = prepare_conditioning_v2(v, f)
    for a in (c.origin, c.referenced_indices, c.encode(v), c.decode(c.encode(v))):
        assert not a.flags.writeable and not np.shares_memory(a, v)
        with pytest.raises(ValueError):
            a.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        c.scale = 4.
    with pytest.raises(TypeError):
        c.diagnostics["adopted"] = True
    with pytest.raises(ValueError):
        replace(c, origin_modes=("zero",)*3)


@pytest.mark.parametrize("what", ["nan", "f32", "bool_faces", "negative", "bounds", "zero", "overflow"])
def test_malformed_and_unrepresentable_fail_without_repair(what):
    v, f = mesh()
    if what == "nan": v[0, 0] = np.nan
    if what == "f32": v = v.astype(np.float32)
    if what == "bool_faces": f = f.astype(bool)
    if what == "negative": f[0, 0] = -1
    if what == "bounds": f[0, 0] = 4
    if what == "zero": v[:] = 0
    if what == "overflow": v[:, 0] = [-np.finfo(float).max, np.finfo(float).max, 0., 0.]
    with pytest.raises(ValueError):
        prepare_conditioning_v2(v, f)


def test_representable_large_same_sign_bbox_avoids_sum_overflow():
    v, f = mesh()
    v[:] = [[1.5e308, 0., 0.], [1.6e308, 0., 0.], [1.5e308, 1e307, 0.], [1.5e308, 0., 1e307]]
    c = prepare_conditioning_v2(v, f)
    assert c.origin_modes[0] == "sterbenz_midpoint"
    assert c.decode(c.encode(v)).tobytes() == v.tobytes()
