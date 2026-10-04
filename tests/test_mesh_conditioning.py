"""Manufactured charts only: no mesh loader, runtime compiler or quality claim."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
import pytest

from world_reward.mesh_conditioning import MeshConditioningChart, prepare_conditioning


def tetra():
    return (np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., .5]]),
            np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64))


def typed_hash(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def test_source_chart_is_one_immutable_power_of_two_with_exact_small_roundtrip():
    v, f = tetra()
    before = v.tobytes(), f.tobytes()
    chart = prepare_conditioning(v, f)
    assert isinstance(chart, MeshConditioningChart)
    assert chart.scale == 2. and chart.scale_exponent == 1
    np.testing.assert_array_equal(chart.origin, [1., .5, .25])
    encoded = chart.encode(v)
    np.testing.assert_array_equal(encoded, (v-chart.origin)/chart.scale)
    np.testing.assert_array_equal(chart.decode(encoded), v)
    assert chart.diagnostics["roundtrip_byte_exact"]
    assert chart.diagnostics["input_vertices_sha256"] == typed_hash(v)
    assert chart.diagnostics["input_faces_sha256"] == typed_hash(f)
    assert (v.tobytes(), f.tobytes()) == before
    assert isinstance(chart.diagnostics, MappingProxyType)
    assert all(type(item) in (str, int, float, bool) for item in chart.diagnostics.values())
    json.dumps(dict(chart.diagnostics), allow_nan=False)
    for flag in ("source_arrays_modified", "source_geometry_repaired",
                 "physical_serialization_verified", "metric_fidelity_verified",
                 "topology_verified", "adopted"):
        assert chart.diagnostics[flag] is False
    with pytest.raises(FrozenInstanceError):
        chart.scale = 4.
    with pytest.raises(TypeError):
        chart.diagnostics["adopted"] = True


def test_arrays_cannot_alias_or_reenable_writes():
    v, f = tetra()
    chart = prepare_conditioning(v, f)
    encoded = chart.encode(v)
    decoded = chart.decode(encoded)
    for array in (chart.origin, chart.referenced_indices, encoded, decoded):
        assert not array.flags.writeable
        assert not np.shares_memory(array, v) and not np.shares_memory(array, f)
        with pytest.raises(ValueError):
            array.flags.writeable = True
    original_origin = chart.origin.copy()
    v[:] = 999.
    f[:] = 0
    np.testing.assert_array_equal(chart.origin, original_origin)
    assert chart.diagnostics["input_vertices_sha256"] != typed_hash(v)


@pytest.mark.parametrize("width,exponent", [(1., 0), (2., 1), (3., 2), (.25, -2),
                                              (np.nextafter(2., np.inf), 2),
                                              (np.nextafter(2., 0.), 1)])
def test_covering_scale_is_minimal_not_log2_rounded(width, exponent):
    v, f = tetra()
    v *= width/2.
    chart = prepare_conditioning(v, f)
    assert chart.scale_exponent == exponent
    assert chart.scale == math.ldexp(1., exponent)
    assert chart.scale >= width
    assert chart.scale/2 < width
    assert chart.diagnostics["encoded_maximum_absolute_coordinate"] <= .5


@pytest.mark.parametrize("exponent", [-500, -16, 0, 16, 500])
def test_power_of_two_source_unit_changes_preserve_chart_coordinates(exponent):
    v, f = tetra()
    initial = prepare_conditioning(v, f)
    scaled = np.ldexp(v, exponent)
    chart = prepare_conditioning(scaled, f)
    assert chart.scale_exponent == initial.scale_exponent + exponent
    np.testing.assert_array_equal(chart.origin, np.ldexp(initial.origin, exponent))
    np.testing.assert_array_equal(chart.encode(scaled), initial.encode(v))
    assert chart.decode(chart.encode(scaled)).tobytes() == scaled.tobytes()


@pytest.mark.parametrize("angle", [0., .37, 1.2])
def test_fixed_rotation_translation_scale_cage_returns_physical_not_new_gauge(angle):
    v, f = tetra()
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    physical = (v@rotation.T)*8 + np.array([13., -7., 29.])
    chart = prepare_conditioning(physical, f)
    encoded = chart.encode(physical)
    np.testing.assert_allclose(chart.decode(encoded), physical, rtol=0, atol=8e-15)
    np.testing.assert_array_equal(chart.origin, physical.min(axis=0)+(physical.max(axis=0)-physical.min(axis=0))*.5)
    assert chart.diagnostics["encoded_maximum_absolute_coordinate"] <= .5
    # A bbox chart is not claimed rotation-equivariant or a metric calibration.
    assert not chart.diagnostics["metric_fidelity_verified"]


def test_orphan_policy_is_explicit_and_never_deletes_or_renumbers():
    v, f = tetra()
    chart = prepare_conditioning(v, f)
    extended = np.r_[[[1e300, -1e300, 1e300]], v, [[-1e300, 1e300, 0.]]]
    shifted_faces = f + 1
    with_orphans = prepare_conditioning(extended, shifted_faces)
    np.testing.assert_array_equal(with_orphans.origin, chart.origin)
    assert with_orphans.scale == chart.scale
    np.testing.assert_array_equal(with_orphans.referenced_indices, [1, 2, 3, 4])
    assert with_orphans.diagnostics["ignored_orphan_vertices"] == 2
    assert with_orphans.diagnostics["vertices"] == 6
    assert with_orphans.diagnostics["input_vertices_sha256"] == typed_hash(extended)
    assert with_orphans.encode(extended).shape == extended.shape
    np.testing.assert_array_equal(shifted_faces, f+1)


def test_bbox_midpoint_avoids_sum_overflow():
    v = np.array([[1.5e308, 0., 0.], [1.6e308, 0., 0.],
                  [1.5e308, 1e307, 0.], [1.5e308, 0., 1e307]])
    _, f = tetra()
    with np.errstate(over="ignore"):
        assert not np.isfinite(v.min(axis=0)[0] + v.max(axis=0)[0])
    chart = prepare_conditioning(v, f)
    assert np.isfinite(chart.origin).all()
    np.testing.assert_array_equal(chart.decode(chart.encode(v)), v)


def test_finite_roundtrip_loss_is_reported_not_snapped_to_original():
    v, f = tetra()
    v[0] = [1e-300, 0., 0.]
    chart = prepare_conditioning(v, f)
    decoded = chart.decode(chart.encode(v))
    assert decoded[0, 0] == 0 and v[0, 0] == 1e-300
    assert not chart.diagnostics["roundtrip_numerically_exact"]
    assert not chart.diagnostics["roundtrip_byte_exact"]
    assert chart.diagnostics["roundtrip_changed_coordinates"] == 1
    assert chart.diagnostics["roundtrip_maximum_absolute_error"] == 1e-300
    assert chart.diagnostics["decoded_source_sha256"] == typed_hash(decoded)
    assert not chart.diagnostics["source_geometry_repaired"]


def test_signed_zero_roundtrip_bytes_are_measured_separately():
    v = np.array([[-1., -1., -1.], [1., 1., 1.], [-0., 0., 0.], [0., 1., 0.]])
    _, f = tetra()
    chart = prepare_conditioning(v, f)
    assert chart.diagnostics["roundtrip_numerically_exact"]
    assert not chart.diagnostics["roundtrip_byte_exact"]
    assert np.signbit(v[2, 0]) and not np.signbit(chart.decode(chart.encode(v))[2, 0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_even_in_orphan(bad):
    v, f = tetra()
    with pytest.raises(ValueError, match="finite"):
        prepare_conditioning(np.r_[v, [[bad, 0., 0.]]], f)


@pytest.mark.parametrize("bad_faces", [[], [[-1, 1, 2]], [[0, 1, 4]],
                                           [[True, False, True]], [[0., 1., 2.]],
                                           [[0, 1]], [[2**64-1, 1, 2]]])
def test_invalid_faces_fail_before_chart(bad_faces):
    v, _ = tetra()
    with pytest.raises(ValueError):
        prepare_conditioning(v, bad_faces)


def test_repeated_degenerate_faces_are_not_repaired_or_called_topology_valid():
    v, _ = tetra()
    f = np.array([[0, 0, 1], [0, 0, 1]], np.int64)
    chart = prepare_conditioning(v, f)
    assert chart.diagnostics["faces"] == 2 and chart.diagnostics["referenced_vertices"] == 2
    assert not chart.diagnostics["topology_verified"]
    assert chart.diagnostics["input_faces_sha256"] == typed_hash(f)


def test_masks_float32_wrong_grid_zero_extent_and_minimal_midpoint_fail():
    v, f = tetra()
    for invalid in (np.ma.array(v), v.astype(np.float32), v[:, :2], v[:3]):
        with pytest.raises(ValueError):
            prepare_conditioning(invalid, f)
    with pytest.raises(ValueError, match="unmasked"):
        prepare_conditioning(v, np.ma.array(f))
    with pytest.raises(ValueError, match="positive"):
        prepare_conditioning(np.zeros_like(v), f)
    v[:] = 0
    v[1, 0] = np.nextafter(0., 1.)
    with pytest.raises(ValueError, match="midpoint underflow"):
        prepare_conditioning(v, f)


def test_extreme_extent_or_covering_scale_fails_before_invalid_chart():
    v, f = tetra()
    v[0, 0], v[1, 0] = -np.finfo(float).max, np.finfo(float).max
    with pytest.raises(ValueError, match="overflow"):
        prepare_conditioning(v, f)
    v, f = tetra()
    v *= np.finfo(float).max/2
    with pytest.raises(ValueError, match="power-of-two"):
        prepare_conditioning(v, f)


def test_transform_rejects_overflow_nonfinite_mask_and_lost_nonzero_underflow():
    v, f = tetra()
    huge = prepare_conditioning(np.ldexp(v-[1., .5, .25], 1022), f)
    with pytest.raises(ValueError, match="overflow"):
        huge.decode(np.full((1, 3), 4., dtype=float))
    with pytest.raises(ValueError, match="underflow"):
        huge.encode(np.array([[huge.origin[0], np.nextafter(huge.origin[1], np.inf),
                               np.nextafter(0., 1.)]]))
    tiny = prepare_conditioning(np.ldexp(v, -100), f)
    with pytest.raises(ValueError, match="underflow"):
        tiny.decode(np.full((1, 3), np.nextafter(0., 1.)))
    for method in (tiny.encode, tiny.decode):
        for invalid in (np.array([[np.nan, 0., 0.]]), np.ma.array(v), v.astype(np.float32), []):
            with pytest.raises(ValueError):
                method(invalid)


def test_chart_constructor_copies_and_rejects_nonpower_scale_or_nested_diagnostics():
    v, f = tetra()
    chart = prepare_conditioning(v, f)
    with pytest.raises(ValueError):
        replace(chart, scale=3.)
    with pytest.raises(ValueError):
        replace(chart, scale_exponent=True)
    with pytest.raises(ValueError):
        replace(chart, referenced_indices=np.array([1, 0], np.int64))
    with pytest.raises(ValueError):
        replace(chart, diagnostics=MappingProxyType({"bad": []}))
    with pytest.raises(ValueError):
        replace(chart, diagnostics=MappingProxyType({"bad": np.inf}))


def test_strided_and_big_endian_f64_source_preserves_original_hash_contract():
    v, f = tetra()
    storage = np.zeros((len(v), 6))
    storage[:, ::2] = v
    chart = prepare_conditioning(storage[:, ::2], f)
    np.testing.assert_array_equal(chart.decode(chart.encode(v)), v)
    big = v.astype(">f8")
    chart = prepare_conditioning(big, f)
    assert chart.diagnostics["input_vertices_sha256"] == typed_hash(big)
    assert chart.diagnostics["input_vertices_sha256"] != typed_hash(v)
