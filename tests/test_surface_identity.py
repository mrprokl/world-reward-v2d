"""Manufactured arrays only; no real loader, packer, media or QEM qualification."""
from dataclasses import FrozenInstanceError
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from world_reward.raw_shape_proposal import AutomaticShapeAnchor, RawShapeProposal
from world_reward.shared_scene import ArtifactRef
from world_reward.surface_identity import SurfaceIdentity, prepare_surface_identity


def triangle(dtype=np.float64):
    return np.array([[-0., 0, 0], [1., 0, 0], [0, 1., 0]], dtype), np.array([[0, 1, 2]], np.int64)


def sheet():
    return np.array([[0., 0, 0], [1, 0, 0], [1, 1, .25], [0, 1, 0]]), np.array([[0, 1, 2], [0, 2, 3]], np.int64)


def tetra():
    return np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_triangle_boundary_identity_all_native_bytes_and_readonly(dtype):
    v, f = triangle(dtype)
    result = prepare_surface_identity(v, f)
    for original, owned in ((v, result.vertices), (f, result.faces)):
        assert original.dtype == owned.dtype and original.tobytes() == owned.tobytes()
        assert not np.shares_memory(original, owned)
    assert result.boundary_loops == ((0, 1, 2),) and result.boundary_components == (0,)
    assert result.component_keys == ("source-array-first-face-0",)
    assert result.face_components.tolist() == [0]
    for array in (result.vertices, result.faces, result.vertex_birth_indices,
                  result.face_birth_indices, result.face_components):
        with pytest.raises(ValueError): array.setflags(write=True)
        with pytest.raises(ValueError): array.flat[0] = 99
    assert result.vertex_birth_indices.tolist() == [0, 1, 2]
    assert result.face_birth_indices.tolist() == [0]
    v[:] = 9; f[:] = 2
    assert result.faces.tolist() == [[0, 1, 2]] and np.signbit(result.vertices[0, 0])
    with pytest.raises(FrozenInstanceError): result.faces = f


def test_curved_sheet_has_one_path_link_per_boundary_vertex():
    v, f = sheet()
    result = SurfaceIdentity(v, f)
    assert result.boundary_loops == ((0, 1, 2, 3),)
    assert result.diagnostics["boundary_edges"] == 4
    assert result.diagnostics["components"][0]["euler_characteristic"] == 1
    assert result.face_components.tolist() == [0, 0]


@pytest.mark.parametrize("reverse", [False, True])
def test_closed_components_need_no_outward_sign_or_positive_volume(reverse):
    v, f = tetra()
    if reverse: f = f[:, ::-1]
    result = SurfaceIdentity(v, f)
    assert result.diagnostics["closed"] and result.boundary_loops == ()
    assert result.diagnostics["components"][0]["euler_characteristic"] == 2
    assert result.faces.tobytes() == f.tobytes()
    assert not result.report()["solid_geometry_certified"]


def test_annulus_keeps_two_oriented_boundary_loops_no_hole_filling():
    angles = np.arange(4) * np.pi / 2
    v = np.array([[r * np.cos(a), r * np.sin(a), 0.] for r in (2., 1.) for a in angles])
    f = np.array([face for i in range(4) for face in (
        (i, (i + 1) % 4, (i + 1) % 4 + 4), (i, (i + 1) % 4 + 4, i + 4))], np.int64)
    result = SurfaceIdentity(v, f)
    assert result.boundary_loops == ((0, 1, 2, 3), (4, 7, 6, 5))
    assert result.diagnostics["components"][0]["euler_characteristic"] == 0
    assert result.faces.tobytes() == f.tobytes()


def test_disconnected_triangle_and_sheet_preserve_first_face_component_order_and_orphans():
    sv, sf = sheet(); tv, tf = triangle()
    v = np.vstack((sv, tv + [4, 0, 0], [[1e300, 2, 3]]))
    f = np.vstack((tf + 4, sf[::-1]))
    result = SurfaceIdentity(v, f)
    assert result.component_keys == ("source-array-first-face-0", "source-array-first-face-1")
    assert result.face_components.tolist() == [0, 1, 1]
    assert result.boundary_loops == ((4, 5, 6), (0, 1, 2, 3))
    assert result.boundary_components == (0, 1)
    assert result.diagnostics["unused_vertices_preserved"] == 1
    assert len(result.vertices) == 8 and result.vertices.tobytes() == v.tobytes()
    assert result.diagnostics["referenced_float32_positions_finite"]


def test_source_indices_not_position_weld_define_components():
    v, f = triangle()
    v = np.vstack((v, v)); f = np.vstack((f, f + 3))
    result = SurfaceIdentity(v, f)
    assert len(result.component_keys) == 2 and len(result.vertices) == 6
    assert result.faces.tobytes() == f.tobytes()
    assert not result.report()["embedding_verified"]  # Coincident surfaces not adjudicated.


def test_tiny_f64_triangle_is_exactly_active_even_when_cross_underflows():
    v, f = triangle(); v *= 2. ** -600
    assert np.all(np.cross(v[1] - v[0], v[2] - v[0]) == 0)
    result = SurfaceIdentity(v, f)
    assert result.diagnostics["every_native_triangle_exactly_active"]
    assert result.diagnostics["native_exact_fallback_faces"] == 1
    assert not result.diagnostics["float32_storage_triangles_exactly_active"]
    assert result.vertices.tobytes() == v.tobytes()  # Diagnostic does not repair or export.


def test_native_float32_subnormal_positive_triangle_is_not_truncated():
    v, f = triangle(np.float32); v *= np.nextafter(np.float32(0), np.float32(1))
    result = SurfaceIdentity(v, f)
    assert result.diagnostics["every_native_triangle_exactly_active"]
    assert result.diagnostics["float32_storage_triangles_exactly_active"]
    assert result.vertices.tobytes() == v.tobytes()


def test_noncollinear_exact_predicate_rejects_zero_face_no_partial_view():
    v, f = sheet(); v = np.vstack((v, [[2, 0, 0]])); f = np.vstack((f, [[0, 1, 4]]))
    before = v.tobytes(), f.tobytes()
    with pytest.raises(ValueError, match="collinear"): SurfaceIdentity(v, f)
    assert (v.tobytes(), f.tobytes()) == before


def test_f64_float32_storage_loss_is_diagnostic_not_changed_native_source():
    v = np.array([[1e6, 1e6, 0], [1e6 + .01, 1e6, 0], [1e6, 1e6 + .01, 0]])
    result = SurfaceIdentity(v, triangle()[1])
    assert result.diagnostics["every_native_triangle_exactly_active"]
    assert result.diagnostics["float32_changes_referenced_positions"]
    assert not result.diagnostics["float32_storage_triangles_exactly_active"]
    assert result.vertices.tobytes() == v.tobytes() and not result.report()["loader_packer_qualified"]


def test_f64_overflow_in_float32_storage_does_not_corrupt_source():
    v, f = triangle(); v *= 1e300
    result = SurfaceIdentity(v, f)
    assert result.diagnostics["every_native_triangle_exactly_active"]
    assert not result.diagnostics["referenced_float32_positions_finite"]
    assert not result.diagnostics["float32_storage_triangles_exactly_active"]
    assert result.vertices.tobytes() == v.tobytes()


@pytest.mark.parametrize("f", [np.array([[0, 1, 2], [0, 1, 3]], np.int64),
    np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], np.int64),
    np.array([[0, 1, 2], [0, 3, 4]], np.int64),
    np.array([[0, 1, 2], [2, 1, 0]], np.int64)])
def test_nonmanifold_orientation_bowtie_or_duplicate_rejects_without_repair(f):
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [-1, 0, 1]])
    before = v.tobytes(), f.tobytes()
    with pytest.raises(ValueError): SurfaceIdentity(v, f)
    assert (v.tobytes(), f.tobytes()) == before


def test_closed_bowtie_vertex_rejects_even_with_all_edge_incidence_two():
    v, f = tetra()
    v = np.vstack((v, v[1:] + [0, 0, -3]))
    second = np.where(f == 0, 0, f + 3)
    with pytest.raises(ValueError, match="Vertex link"):
        SurfaceIdentity(v, np.vstack((f, second)))


@pytest.mark.parametrize("which", ["vertices", "faces"])
def test_all_vertices_and_faces_count_against_budget_even_orphans(which):
    v, f = triangle()
    if which == "vertices": v = np.vstack((v, np.zeros((4094, 3))))
    else: f = np.repeat(f, 4097, axis=0)
    with pytest.raises(ValueError, match="4096"): SurfaceIdentity(v, f)


def test_exact4096_vertex_budget_keeps_all_orphans():
    v, f = triangle(); v = np.vstack((v, np.zeros((4093, 3))))
    result = SurfaceIdentity(v, f)
    assert len(result.vertices) == 4096 and result.diagnostics["unused_vertices_preserved"] == 4093
    assert result.vertex_birth_indices[-1] == 4095


@pytest.mark.parametrize("changes", [
    dict(vertices=[[0., 0, 0], [1, 0, 0], [0, 1, 0]]), dict(vertices=np.zeros((3, 3), np.float16)),
    dict(vertices=np.zeros((3, 3), np.int64)), dict(vertices=np.full((3, 3), np.nan)),
    dict(vertices=np.full((3, 3), np.inf)), dict(vertices=np.zeros((2, 3))),
    dict(vertices=np.zeros((3, 2))), dict(vertices=np.ma.array(np.zeros((3, 3)), mask=False)),
    dict(faces=np.array([[0, 1, 2]], np.int32)), dict(faces=np.array([[0, 1, 2]], np.uint64)),
    dict(faces=np.array([[0., 1., 2.]])), dict(faces=np.array([[-1, 1, 2]], np.int64)),
    dict(faces=np.array([[0, 1, 3]], np.int64)), dict(faces=np.array([[0, 0, 2]], np.int64)),
    dict(faces=np.zeros((0, 3), np.int64)), dict(faces=np.array([0, 1, 2], np.int64)),
    dict(faces=np.ma.array([[0, 1, 2]], mask=False)), dict(declared_references={"source": ()}),
    dict(declared_proposal_sha256=True), dict(declared_proposal_sha256="a" * 63),
])
def test_strict_native_contract_no_dtype_conversion(changes):
    v, f = triangle()
    with pytest.raises(ValueError): SurfaceIdentity(**(dict(vertices=v, faces=f) | changes))


def test_noncontiguous_source_byte_order_and_inward_faces_preserved():
    v, f = tetra(); wide = np.zeros((4, 6)); wide[:, ::2] = v
    v = wide[:, ::2]; f = f[::-1, ::-1]
    result = SurfaceIdentity(v, f)
    assert result.vertices.flags.c_contiguous and result.faces.flags.c_contiguous
    assert result.vertices.tobytes() == v.tobytes() and result.faces.tobytes() == f.tobytes()


def test_raw_shape_admissibility_keeps_declared_refs_gauge_bind_not_authenticate():
    v, f = triangle(np.float32)
    refs = {role: (ArtifactRef(f"declared/{role}", 3, "a" * 64),) for role in ("source", "model", "config")}
    raw = RawShapeProposal(v, f, "native", "native_frame", "native_unit", "unverified",
        AutomaticShapeAnchor(0, refs["source"][0], refs["config"][0], "automatic"), refs)
    result = SurfaceIdentity.from_proposal(raw)
    report = result.report()
    assert report["declared_proposal_sha256"] == raw.report()["proposal_sha256"]
    assert result.vertices.dtype == raw.vertices.dtype and result.vertices.tobytes() == raw.vertices.tobytes()
    assert result.faces.tobytes() == raw.faces.tobytes() and not np.shares_memory(result.vertices, raw.vertices)
    refs.clear()
    assert set(result.declared_references) == {"source", "model", "config"}
    with pytest.raises(TypeError): result.declared_references["source"] = ()
    for key in ("geometry_operations_applied", "area_or_length_tolerance_used", "source_references_authenticated",
                "metric_gauge_verified", "embedding_verified", "solid_geometry_certified", "loader_packer_qualified",
                "reconstruction_accuracy_verified", "adoption_authorized"):
        assert report[key] is False
    json.dumps(report, allow_nan=False)
    report["diagnostics"]["components"][0]["faces"] = 999
    assert result.diagnostics["components"][0]["faces"] == 1
    assert report["vertices"]["sha256"] == hashlib.sha256(v.tobytes()).hexdigest()


def test_native_geometry_drift_changes_hash_not_fitted_or_hidden():
    v, f = triangle(); first = SurfaceIdentity(v, f)
    v[1, 0] = np.nextafter(v[1, 0], np.inf)
    second = SurfaceIdentity(v, f)
    assert first.report()["vertices"]["sha256"] != second.report()["vertices"]["sha256"]
    assert first.faces.tobytes() == second.faces.tobytes()
    a, b, c = [[Fraction.from_float(float(x)) for x in p] for p in first.vertices]
    assert (b[0] - a[0]) * (c[1] - a[1]) != (b[1] - a[1]) * (c[0] - a[0])


def test_from_proposal_never_accepts_fake_metadata_object():
    with pytest.raises(ValueError, match="RawShapeProposal"): SurfaceIdentity.from_proposal(object())


@pytest.mark.parametrize("value", [[ArtifactRef("declared/ref", 1, "a" * 64)], (), (object(),)])
def test_declared_references_are_strict_tuples_of_metadata_only(value):
    refs = {role: (ArtifactRef(f"declared/{role}", 1, "a" * 64),)
            for role in ("source", "model", "config")}
    refs["source"] = value
    with pytest.raises(ValueError): prepare_surface_identity(*triangle(), declared_references=refs)


def test_declared_reference_metadata_is_bounded():
    refs = {role: (ArtifactRef(f"declared/{role}", 1, "a" * 64),) * 128
            for role in ("source", "model", "config")}
    with pytest.raises(ValueError, match="16KB"):
        prepare_surface_identity(*triangle(), declared_references=refs)
