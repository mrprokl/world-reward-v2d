"""Manufactured precision fixtures; no mesh assets, Trimesh install or inference."""
import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

import world_reward.mesh_serialization as serialization
from world_reward.mesh_serialization import serialization_preflight


def triangle():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), np.array([[0, 1, 2]])


def typed_hash(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def test_valid_triangle_is_immutable_scalar_evidence_not_adoption():
    v, f = triangle()
    before = v.copy(), f.copy()
    v.flags.writeable = f.flags.writeable = False
    report = serialization_preflight(v, f)
    assert isinstance(report, MappingProxyType)
    assert report["phase"] == "prepared" and report["position_weld_admissible"]
    assert all(type(value) in (str, int, bool) for value in report.values())
    json.dumps(dict(report), allow_nan=False)
    with pytest.raises(TypeError):
        report["adopted"] = True
    assert all(report[key] is False for key in (
        "input_arrays_modified", "faces_deleted_or_reordered", "positions_snapped_or_rescaled",
        "geometry_returned", "area_or_length_tolerance_used", "pre_export_geometry_recovered",
        "orientation_verified", "topology_verified", "embedding_verified", "metric_fidelity_verified",
        "simplification_verified", "runtime_library_verified", "challenge_performance_verified", "adopted"))
    assert v.tobytes() == before[0].tobytes() and f.tobytes() == before[1].tobytes()
    assert report["input_vertices_sha256"] == typed_hash(v)
    assert report["input_faces_sha256"] == typed_hash(f)


def test_exact_seams_preserve_every_oriented_triangle_without_snapping():
    v, f = triangle()
    v = np.r_[v, [v[0]]]
    f = np.r_[f, [[3, 2, 1]]]
    report = serialization_preflight(v, f)
    assert report["source_exact_seam_duplicates"] == report["source_exact_seam_groups"] == 1
    assert report["welded_vertices"] == 3 and report["faces"] == 2
    assert report["weld_merged_distinct_float32_positions"] == 0
    assert report["position_weld_admissible"] and report["serialized_triangles_numerically_preserved_by_weld"]
    assert report["serialized_triangles_byte_preserved_by_weld"]
    assert report["serialized_oriented_triangles_sha256"] == report["welded_oriented_triangles_sha256"]
    assert report["representative_original_indices_sha256"] == typed_hash(np.array([0, 1, 2], np.int64))


def test_signed_zero_is_an_exact_seam_not_a_nonexact_merge():
    v, f = triangle()
    v = np.r_[v, [[-0., -0., -0.]]]
    f = np.r_[f, [[3, 1, 2]]]
    before = v.tobytes(), f.tobytes()
    report = serialization_preflight(v, f)
    assert report["source_negative_zero_coordinates"] == 3
    assert report["float32_negative_zero_coordinates"] == 3
    assert report["source_exact_seam_duplicates"] == 1 and report["position_weld_admissible"]
    assert report["serialized_triangles_numerically_preserved_by_weld"]
    assert not report["serialized_triangles_byte_preserved_by_weld"]
    # The byte hashes retain signs even though the numerical positions coincide.
    assert report["serialized_oriented_triangles_sha256"] != report["welded_oriented_triangles_sha256"]
    assert (v.tobytes(), f.tobytes()) == before


def test_orphans_are_not_weld_keys_and_do_not_affect_representatives():
    v, f = triangle()
    original = serialization_preflight(v, f)
    v = np.r_[[[1e300, 1e300, 1e300]], v, [[1e-300, 0., 0.]]]
    f += 1
    report = serialization_preflight(v, f)
    assert report["ignored_orphan_vertices"] == 2 and report["referenced_vertices"] == 3
    assert report["position_weld_admissible"]
    assert report["weld_keys_sha256"] == original["weld_keys_sha256"]
    assert report["representative_original_indices_sha256"] == typed_hash(np.array([1, 2, 3], np.int64))


def test_nonexact_rounded_key_collision_never_returns_a_repaired_mesh():
    v, f = triangle()
    v = np.r_[v, [[1e-9, 0., 0.]]]
    f = np.r_[f, [[3, 1, 2]]]
    report = serialization_preflight(v, f)
    assert report["source_exact_seam_duplicates"] == report["float32_collapsed_distinct_positions"] == 0
    assert report["nonexact_weld_collision_groups"] == report["weld_merged_distinct_float32_positions"] == 1
    assert report["source_triangles_exactly_active"] and report["float32_triangles_exactly_active"]
    assert report["welded_triangles_exactly_active"]
    assert not report["position_weld_admissible"] and not report["serialized_triangles_numerically_preserved_by_weld"]
    assert report["faces"] == 2 and not report["faces_deleted_or_reordered"]


def test_first_representative_is_original_index_not_key_order_or_snap():
    v = np.array([[1e-9, 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    f = np.array([[1, 2, 3], [0, 3, 2]])
    report = serialization_preflight(v, f)
    stored = v.astype(np.float32).astype(np.float64)
    expected = stored[[[0, 2, 3], [0, 3, 2]]]
    assert report["representative_original_indices_sha256"] == typed_hash(np.array([0, 2, 3], np.int64))
    assert report["welded_oriented_triangles_sha256"] == typed_hash(expected)
    snapped = expected.copy()
    snapped[:, 0] = 0.
    assert report["welded_oriented_triangles_sha256"] != typed_hash(snapped)


def test_float64_to_float32_loss_is_distinct_from_weld_collision():
    v = np.array([[1e6, 1e6, 1e6], [1e6 + .01, 1e6, 1e6], [1e6, 1e6 + .01, 1e6]])
    report = serialization_preflight(v, [[0, 1, 2]])
    assert report["source_triangles_exactly_active"]
    assert not report["float32_triangles_exactly_active"] and not report["welded_triangles_exactly_active"]
    assert report["float32_collapsed_distinct_positions"] == 2
    assert report["weld_merged_distinct_float32_positions"] == 0
    assert not report["position_weld_admissible"] and not report["pre_export_geometry_recovered"]


def test_tiny_positive_triangle_is_exactly_active_even_if_default_weld_destroys_it():
    v, f = triangle()
    v *= 2. ** -30
    report = serialization_preflight(v, f)
    assert report["source_triangles_exactly_active"] and report["float32_triangles_exactly_active"]
    assert not report["welded_triangles_exactly_active"]
    assert report["weld_merged_distinct_float32_positions"] == 2
    assert not report["area_or_length_tolerance_used"] and not report["position_weld_admissible"]


@pytest.mark.parametrize("face", [[0, 1, 2], [0, 0, 2]])
def test_exactly_collinear_and_repeated_indices_are_negative_not_filtered(face):
    v = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
    report = serialization_preflight(v, [face])
    assert report["faces"] == 1
    assert not report["source_triangles_exactly_active"] and not report["position_weld_admissible"]


def test_default_keys_use_even_ties_including_negative_and_signed_zero():
    # Powers of two give exact half-integers *after* float32 storage and *1e8.
    values = np.array([1., 3., 5., -1., -3., -5., -0.]) / 512.
    stored = np.column_stack((values, np.zeros((len(values), 2)))).astype(np.float32).astype(np.float64)
    keys = serialization._weld_keys(stored)
    assert keys[:, 0].tolist() == [195312, 585938, 976562, -195312, -585938, -976562, 0]
    assert np.array_equal(keys, np.round(stored * 10**8).astype(np.int64))
    assert not np.array_equal(keys[:, 0], np.sign(stored[:, 0]) * np.floor(np.abs(stored[:, 0] * 1e8) + .5))


def test_int64_key_range_upper_exclusive_lower_inclusive():
    maximum = np.float64(2. ** 63)
    # Exercise the rounded conversion boundary directly, independently of F32.
    with pytest.raises(ValueError, match="int64"):
        serialization._weld_keys(np.array([[maximum / 1e8, 0., 0.]]))
    lower = np.array([[-maximum / 1e8, 0., 0.]])
    assert serialization._weld_keys(lower)[0, 0] == np.iinfo(np.int64).min
    with pytest.raises(ValueError, match="int64"):
        serialization._weld_keys(np.array([[np.nextafter(lower[0, 0], -np.inf), 0., 0.]]))


@pytest.mark.parametrize("sign", [-1., 1.])
def test_referenced_float32_key_boundary_fails_before_undefined_integer_cast(sign):
    v, f = triangle()
    edge = np.float32(2. ** 63 / 1e8)
    v[0, 0] = sign * float(np.nextafter(edge, np.float32(np.inf)))
    before = v.tobytes(), f.tobytes()
    with pytest.raises(ValueError, match="int64"):
        serialization_preflight(v, f)
    assert (v.tobytes(), f.tobytes()) == before


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_rejected_even_when_unreferenced(invalid):
    v, f = triangle()
    v = np.r_[v, [[invalid, 0., 0.]]]
    with pytest.raises(ValueError, match="finite"):
        serialization_preflight(v, f)


def test_referenced_float32_overflow_is_not_removed_as_a_repair():
    v, f = triangle()
    v[0, 0] = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="finite float32"):
        serialization_preflight(v, f)


@pytest.mark.parametrize("f", [[], [[-1, 1, 2]], [[0, 1, 3]], [[0., 1., 2.]], [[False, True, True]], [[0, 1]]])
def test_invalid_face_contract(f):
    v, _ = triangle()
    with pytest.raises(ValueError, match="faces"):
        serialization_preflight(v, f)


@pytest.mark.parametrize("which", ["vertices", "faces"])
def test_masked_geometry_is_not_silently_unmasked(which):
    v, f = triangle()
    if which == "vertices":
        v = np.ma.array(v, mask=False)
    else:
        f = np.ma.array(f, mask=False)
    with pytest.raises(ValueError, match="unmasked"):
        serialization_preflight(v, f)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f8"])
def test_noncontiguous_inputs_and_dtype_bytes_are_unchanged(dtype):
    v = np.zeros((3, 6), dtype=dtype)[:, ::2]
    v[:] = triangle()[0]
    f = np.array([[0, 99, 1, 99, 2, 99]], np.uint32)[:, ::2]
    before = v.tobytes(), f.tobytes(), v.dtype, f.dtype, v.strides, f.strides
    report = serialization_preflight(v, f)
    assert report["position_weld_admissible"]
    assert (v.tobytes(), f.tobytes(), v.dtype, f.dtype, v.strides, f.strides) == before


def test_seeded_policy_matches_independent_native_loop_and_ignores_face_order_for_representatives():
    rng = np.random.default_rng(8401)
    for _ in range(32):
        v = rng.uniform(-2., 2., (12, 3))
        v[8:10] = v[:2]
        v[10] = v[2] + [1e-9, 0., 0.]
        f = np.array([[0, 1, 2], [8, 9, 10], [4, 5, 6]])
        rng.shuffle(f)
        before = v.tobytes(), f.tobytes()
        seen, representatives, triangles = {}, [], []
        stored = v.astype(np.float32).astype(np.float64)
        for index in sorted(set(f.ravel().tolist())):
            key = tuple(np.round(stored[index] * 1e8).astype(np.int64))
            if key not in seen:
                seen[key] = index
                representatives.append(index)
        for face in f:
            triangles.append([stored[seen[tuple(np.round(stored[index] * 1e8).astype(np.int64))]] for index in face])
        report = serialization_preflight(v, f)
        assert report["representative_original_indices_sha256"] == typed_hash(np.array(representatives, np.int64))
        assert report["welded_oriented_triangles_sha256"] == typed_hash(np.asarray(triangles))
        assert (v.tobytes(), f.tobytes()) == before
