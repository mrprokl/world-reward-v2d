"""Tiny arrays only: no generated assets, files, datasets, models or I/O."""
from dataclasses import FrozenInstanceError, replace
import hashlib

import numpy as np
import pytest

from world_reward.raw_shape_proposal import AutomaticShapeAnchor, RawShapeProposal, SCHEMA
from world_reward.shared_scene import ArtifactRef


def ref(name="source"):
    return ArtifactRef(f"declared/{name}.bin", 12, "a" * 64)


def anchor(**changes):
    return AutomaticShapeAnchor(**(dict(frame_index=0, image=ref("rgb"), mask=ref("mask"),
                                       method="automatic detector native defaults") | changes))


def proposal(**changes):
    args = dict(vertices=np.array([[-0., 0, 0], [2., 0, 0], [0, 1., 0], [4., 5., 6.]], np.float32),
                faces=np.array([[0, 1, 2]], np.int64), backend="opaque native generator",
                canonical_frame="native object frame", linear_unit="native units, not metric",
                gauge_convention="decoder axes unchanged; native scale not applied",
                anchor=anchor(), references={role: (ref(role),) for role in ("source", "model", "config")},
                provenance={"seed": 0, "settings": {"postprocess": False, "tags": ["raw"]}})
    return RawShapeProposal(**(args | changes))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_dtype_bytes_strides_signedzero_and_immutable_owning_copy(dtype):
    vertices = np.arange(24, dtype=dtype).reshape(4, 6)[:, ::2]
    vertices[0, 0] = -0.
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)[::-1]
    raw = proposal(vertices=vertices, faces=faces)
    for before, after in ((vertices, raw.vertices), (faces, raw.faces)):
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes(order="C") == after.tobytes(order="C")
        assert not np.shares_memory(before, after) and not after.flags.writeable
        with pytest.raises(ValueError): after.setflags(write=True)
        with pytest.raises(ValueError): after.flat[0] = 0
    captured = raw.vertices.tobytes()
    vertices[:] = 9; faces[:] = 1
    assert raw.vertices.tobytes() == captured and raw.faces.tolist() == [[0, 2, 3], [0, 1, 2]]
    with pytest.raises(FrozenInstanceError): raw.backend = "different"


def test_every_component_unused_vertex_duplicate_and_degenerate_face_preserved():
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [4, 0, 0], [5, 0, 0],
                  [4, 1, 0], [2, 0, 0], [123, 456, 789]], np.float64)
    f = np.array([[3, 4, 5], [0, 1, 2], [0, 1, 2], [0, 0, 1], [0, 1, 6]], np.int64)
    raw = proposal(vertices=v, faces=f)
    assert raw.vertices.tobytes() == v.tobytes() and raw.faces.tobytes() == f.tobytes()
    d = raw.report()["diagnostics"]
    assert d["referenced_vertices"] == 7 and d["unused_vertices_preserved"] == 1
    assert d["repeated_index_faces_preserved"] == 1 and d["zero_cross_product_faces_fp64"] == 2
    assert d["nonfinite_cross_product_faces_fp64"] == 0


def test_raw_proposal_has_no_packed_vertex_budget_or_hidden_compaction():
    v = np.vstack((proposal().vertices, np.zeros((4100, 3), np.float32)))
    raw = proposal(vertices=v)
    assert len(raw.vertices) == 4104 and raw.report()["diagnostics"]["unused_vertices_preserved"] == 4101


def test_collapsed_one_vertex_proposal_is_retained_not_certified_or_filled():
    raw = proposal(vertices=np.zeros((1, 3)), faces=np.zeros((1, 3), np.int64))
    assert raw.vertices.shape == (1, 3) and raw.faces.tolist() == [[0, 0, 0]]
    assert raw.report()["diagnostics"]["repeated_index_faces_preserved"] == 1
    assert raw.report()["solid_geometry_certified"] is False


def test_numeric_diagnostics_never_normalize_rescale_or_reject_raw_geometry():
    huge = np.array([[0., 0, 0], [1e308, 0, 0], [0, 1e308, 0]], np.float64)
    tiny = huge * 1e-308 * 1e-200
    for v, diagnostic in ((huge, "nonfinite_cross_product_faces_fp64"),
                          (tiny, "zero_cross_product_faces_fp64")):
        raw = proposal(vertices=v)
        assert raw.vertices.tobytes() == v.tobytes()
        assert raw.report()["diagnostics"][diagnostic] == 1
        assert raw.report()["solid_geometry_certified"] is False


def test_provenance_deep_copy_and_reference_role_supports_multiple_weights():
    metadata = {"nested": {"tags": ["unchanged"]}}
    refs = {role: (ref(role),) for role in ("source", "model", "config")}
    refs["model"] += (ref("auxiliary-model"),)
    raw = proposal(provenance=metadata, references=refs)
    metadata["nested"]["tags"].append("changed"); refs["model"] = ()
    assert raw.provenance["nested"]["tags"] == ("unchanged",) and len(raw.references["model"]) == 2
    with pytest.raises(TypeError): raw.provenance["nested"]["tags"] = ()
    with pytest.raises(TypeError): raw.references["source"] = ()
    cloned = replace(raw)
    assert cloned.report() == raw.report() and not np.shares_memory(cloned.vertices, raw.vertices)
    report = raw.report(); report["provenance"]["nested"]["tags"].append("external")
    assert raw.provenance["nested"]["tags"] == ("unchanged",)


def test_all_repeated_collinear_faces_preserved_without_admissibility_or_source_attestation():
    raw = proposal(vertices=np.zeros((2, 3)), faces=np.array([[0, 1, 0], [1, 1, 1]], np.int64),
                   provenance={"caller_claims_solid": True})
    assert raw.faces.tolist() == [[0, 1, 0], [1, 1, 1]]
    report = raw.report()
    assert report["solid_geometry_certified"] is False and report["source_references_authenticated"] is False
    assert report["geometry_operation_scope"] == "this_freeze_only_upstream_unverified"


def test_fingerprints_bind_every_array_dtype_order_reference_gauge_and_anchor():
    raw = proposal(); report = raw.report()
    assert report["schema"] == SCHEMA and report["status"] == "proposal_uncertified"
    assert report["vertices"]["sha256"] == hashlib.sha256(raw.vertices.tobytes()).hexdigest()
    assert report["faces"]["sha256"] == hashlib.sha256(raw.faces.tobytes()).hexdigest()
    assert raw.report() == report
    for alternative in (replace(raw, vertices=raw.vertices.astype(np.float64)),
                        replace(raw, vertices=np.vstack((raw.vertices, [[9, 9, 9]])).astype(np.float32)),
                        replace(raw, faces=np.array([[2, 1, 0]], np.int64)),
                        replace(raw, gauge_convention="different declaration"),
                        replace(raw, anchor=anchor(image=ref("different-image"))),
                        replace(raw, references=raw.references | {"config": (ref("different-config"),)})):
        assert alternative.report()["proposal_sha256"] != report["proposal_sha256"]
    for key in ("source_references_authenticated", "automatic_anchor_authenticated", "metric_gauge_verified",
                "solid_geometry_certified", "reconstruction_accuracy_verified", "adoption_authorized"):
        assert report[key] is False
    assert report["supplied_raw_arrays_byte_preserved"] is True and report["freeze_geometry_operations_applied"] is False
    assert report["source_geometry_operations_verified"] is False


@pytest.mark.parametrize("changes", [
    {"vertices": [[0., 0, 0], [1., 0, 0], [0., 1., 0]]},
    {"vertices": np.zeros((3, 3), np.int64)}, {"vertices": np.zeros((3, 3), np.float16)},
    {"vertices": np.zeros((3, 3), np.complex128)}, {"vertices": np.zeros((3, 2))},
    {"vertices": np.zeros((0, 3))},
    {"vertices": np.full((3, 3), np.nan)}, {"vertices": np.full((3, 3), np.inf)},
    {"vertices": np.ma.array(np.zeros((3, 3)), mask=False)},
    {"faces": np.array([[0, 1, 2]], np.int32)}, {"faces": np.array([[0., 1., 2.]])},
    {"faces": np.array([[0, 1, 2]], np.uint64)}, {"faces": np.zeros((0, 3), np.int64)},
    {"faces": np.array([[0, -1, 2]], np.int64)}, {"faces": np.array([[0, 1, 4]], np.int64)},
    {"faces": np.array([0, 1, 2], np.int64)}, {"faces": np.zeros((1, 4), np.int64)},
    {"faces": np.ma.array([[0, 1, 2]], mask=False)}, {"backend": ""},
    {"canonical_frame": "\n"}, {"linear_unit": ""}, {"gauge_convention": ""},
    {"anchor": None}, {"references": {}},
    {"references": {role: (ref(role),) for role in ("source", "model", "config", "other")}},
    {"references": {role: [ref(role)] for role in ("source", "model", "config")}},
    {"references": {role: () for role in ("source", "model", "config")}},
    {"references": {role: ("not-an-artifact",) for role in ("source", "model", "config")}},
    {"provenance": {"invalid": np.zeros(3)}}, {"provenance": {"nonfinite": float("nan")}},
    {"provenance": {"oversized": "x" * 16384}}, {"provenance": {1: "not-json-key"}},
])
def test_invalid_raw_arrays_or_undeclared_metadata_fail(changes):
    with pytest.raises(ValueError): proposal(**changes)


@pytest.mark.parametrize("changes", [{"frame_index": 1}, {"frame_index": True}, {"frame_index": -1},
    {"frame_index": np.int64(0)}, {"method": ""}, {"image": None}, {"mask": "manual pixels"}])
def test_anchor_requires_explicit_original0_automatic_method_and_refs(changes):
    with pytest.raises(ValueError): anchor(**changes)
