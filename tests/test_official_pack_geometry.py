"""Tiny own dual-precision fixtures; no GLB, official code, media or model."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def geometry():
    path = Path(__file__).resolve().parents[1] / "infra/official_pack_geometry.py"
    spec = importlib.util.spec_from_file_location("own_official_pack_geometry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mesh():
    raw = np.array([[.100000000001, .200000000003, .300000000007],
                    [.400000000011, .200000000003, .300000000007],
                    [.100000000001, .500000000013, .300000000007],
                    [.100000000001, .200000000003, .600000000017]], np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return raw, faces


def inputs():
    raw, faces = mesh()
    return [raw.astype(np.float32), faces.copy(), raw, faces.copy(), raw.copy(), faces.copy()]


def test_independent_exact_raw_surface_and_native_quantization(geometry):
    args = inputs()
    before = [value.copy() for value in args]
    proof = geometry.verify_exact_dual_surfaces(*args)
    assert not np.array_equal(args[0].astype(np.float64), args[2])
    assert proof["raw_oriented_triangles_exact"] and proof["native_fp32_quantization_exact"]
    assert proof["source_active_faces"] == proof["native_active_faces"] == proof["active_faces"] == 4
    assert proof["original_nonzero_triangles_preserved"]
    assert proof["additional_scale_or_alignment"] is False and proof["geometry_repaired"] is False
    for old, actual in zip(before, args, strict=True):
        assert old.dtype == actual.dtype and old.tobytes() == actual.tobytes()


def test_permuted_cyclic_weld_and_zero_area_padding_remain_exact(geometry):
    args = inputs()
    perm = np.array([2, 0, 3, 1])
    inverse = np.argsort(perm)
    args[4] = np.concatenate((args[4][perm], args[4][perm][:1]))
    args[5] = np.roll(inverse[args[5][::-1]], 1, axis=1)
    args[5][args[5] == 0] = 4  # Only exact duplicate coordinates are interchangeable.
    args[5] = np.concatenate((args[5], np.array([[0, 0, 0], [1, 1, 2]], np.int64)))
    proof = geometry.verify_exact_dual_surfaces(*args)
    assert proof["active_faces"] == 4 and proof["packed_exact_degenerate_faces"] == 2


@pytest.mark.parametrize("fault", ["epsilon", "near_weld", "shrink", "flip", "delete", "duplicate"])
def test_raw_surface_mutations_fail_even_when_fp32_hides_them(geometry, fault):
    args = inputs()
    if fault in ("epsilon", "near_weld"):
        args[4][1, 0] += 1e-12
        assert np.array_equal(args[4].astype(np.float32), args[0])
    elif fault == "shrink":
        args[4] *= .99
    elif fault == "flip":
        args[5][0] = args[5][0][[1, 0, 2]]
    elif fault == "delete":
        args[5] = args[5][1:]
    else:
        args[5] = np.concatenate((args[5], args[5][:1]))
    with pytest.raises(ValueError, match="exact raw GLB surface"):
        geometry.verify_exact_dual_surfaces(*args)


@pytest.mark.parametrize("fault", ["epsilon", "flip", "delete", "duplicate"])
def test_native_geometry_cannot_bypass_raw_quantization(geometry, fault):
    args = inputs()
    if fault == "epsilon":
        args[0][1, 0] = np.nextafter(args[0][1, 0], np.float32(np.inf))
    elif fault == "flip":
        args[1][0] = args[1][0][[1, 0, 2]]
    elif fault == "delete":
        args[1] = args[1][1:]
    else:
        args[1] = np.concatenate((args[1], args[1][:1]))
    with pytest.raises(ValueError, match="exact raw/packed FP32 quantization"):
        geometry.verify_exact_dual_surfaces(*args)


def test_nonzero_original_triangle_collapsing_in_fp32_fails(geometry):
    raw, faces = mesh()
    raw = np.concatenate((raw, [[1., 0., 0.], [1. + 1e-8, 0., 0.], [1., 1., 0.]]))
    faces = np.concatenate((faces, np.array([[4, 5, 6]], np.int64)))
    with pytest.raises(ValueError, match="collapses or creates"):
        geometry.verify_exact_dual_surfaces(raw.astype(np.float32), faces, raw, faces, raw.copy(), faces.copy())


def test_exact_raw_collinearity_becoming_surface_in_fp32_fails(geometry):
    raw, faces = mesh()
    # Exact dyadic collinearity, broken by independent coordinate rounding.
    delta = 2. ** -26
    raw = np.concatenate((raw, [[1., 1., 0.], [1. + 3 * delta, 1. + 5 * delta, 0.],
                                [1. + 6 * delta, 1. + 10 * delta, 0.]]))
    faces = np.concatenate((faces, np.array([[4, 5, 6]], np.int64)))
    quantized = raw.astype(np.float32)
    assert np.all(np.cross(raw[5] - raw[4], raw[6] - raw[4]) == 0)
    assert np.any(np.cross(quantized[5] - quantized[4], quantized[6] - quantized[4]) != 0)
    with pytest.raises(ValueError, match="collapses or creates"):
        geometry.verify_exact_dual_surfaces(quantized, faces, raw, faces, raw.copy(), faces.copy())


@pytest.mark.parametrize("scale", [1., 1e-100, 1e-200, np.nextafter(0., 1.)])
def test_exact_dyadic_nonzero_surfaces_never_disappear_on_cross_underflow(geometry, scale):
    vertices = np.array([[0., 0., 0.], [scale, 0., 0.], [0., scale, 0.]], np.float64)
    triangles = geometry.canonical_oriented_triangles(vertices, np.array([[0, 1, 2]], np.int64))
    assert triangles.shape == (1, 9)


def test_exact_large_dyadic_classification_avoids_cross_overflow(geometry):
    vertices = np.array([[-1e308, 0., 0.], [1e308, 0., 0.], [0., 1e308, 0.]], np.float64)
    result = geometry.canonical_oriented_triangles(vertices, np.array([[0, 1, 2]], np.int64))
    assert result.shape == (1, 9) and np.isfinite(result).all()


def test_exact_collinear_padding_and_repeated_coordinates_are_not_surfaces(geometry):
    vertices, faces = mesh()
    vertices = np.concatenate((vertices, [[0., 0., 0.], [1e-200, 0., 0.], [2e-200, 0., 0.]]))
    faces = np.concatenate((faces, np.array([[4, 5, 6], [4, 4, 5]], np.int64)))
    assert len(geometry.canonical_oriented_triangles(vertices, faces)) == 4


@pytest.mark.parametrize("position,dtype", [(0, np.float64), (2, np.float32), (4, np.float32),
                                           (1, np.int32), (3, np.uint64), (5, np.float64)])
def test_exact_native_raw_packed_dtype_contracts(geometry, position, dtype):
    args = inputs()
    args[position] = args[position].astype(dtype)
    with pytest.raises(ValueError, match="exact floating vertices and int64"):
        geometry.verify_exact_dual_surfaces(*args)


@pytest.mark.parametrize("position", range(6))
def test_masked_mesh_inputs_fail(geometry, position):
    args = inputs()
    args[position] = np.ma.array(args[position], mask=False)
    with pytest.raises(ValueError, match="unmasked"):
        geometry.verify_exact_dual_surfaces(*args)


@pytest.mark.parametrize("position", [0, 2, 4])
@pytest.mark.parametrize("fault", ["nan", "inf", "empty", "shape"])
def test_invalid_vertex_arrays_fail(geometry, position, fault):
    args = inputs()
    if fault in ("nan", "inf"):
        args[position][0, 0] = np.nan if fault == "nan" else np.inf
    elif fault == "empty":
        args[position] = args[position][:0]
    else:
        args[position] = args[position][:, :2]
    with pytest.raises(ValueError):
        geometry.verify_exact_dual_surfaces(*args)


@pytest.mark.parametrize("position", [1, 3, 5])
@pytest.mark.parametrize("fault", ["negative", "outside", "empty", "shape"])
def test_invalid_face_indices_fail(geometry, position, fault):
    args = inputs()
    if fault == "negative":
        args[position][0, 0] = -1
    elif fault == "outside":
        args[position][0, 0] = 4
    elif fault == "empty":
        args[position] = args[position][:0]
    else:
        args[position] = args[position][:, :2]
    with pytest.raises(ValueError):
        geometry.verify_exact_dual_surfaces(*args)


def test_finite_fp64_cannot_overflow_native_cast(geometry):
    raw, faces = mesh()
    raw *= 1e40
    # The frozen native input can be finite but cannot equal an overflowing cast.
    native = np.zeros(raw.shape, np.float32)
    with pytest.raises(ValueError, match="overflow native FP32"):
        geometry.verify_exact_dual_surfaces(native, faces, raw, faces, raw.copy(), faces.copy())


def test_all_zero_surface_is_not_a_geometry_proof(geometry):
    vertices = np.zeros((3, 3), np.float64)
    faces = np.array([[0, 1, 2]], np.int64)
    with pytest.raises(ValueError, match="nonzero triangle"):
        geometry.canonical_oriented_triangles(vertices, faces)


def test_module_import_is_numpy_free_for_host_preflight(monkeypatch):
    import builtins

    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "scipy", "torch", "trimesh", "joblib"}:
            raise AssertionError("Pure module import must not load numeric/model runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    path = Path(__file__).resolve().parents[1] / "infra/official_pack_geometry.py"
    spec = importlib.util.spec_from_file_location("own_geometry_import_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.verify_exact_dual_surfaces)
