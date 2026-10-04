"""Manufactured meshes and mocked child results only, never a local native run."""
import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wr_certified_solid_controls", ROOT / "infra/certified_solid_controls.py")
module = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
SHA = hashlib.sha256((ROOT / "infra/certified_solid_query.cpp").read_bytes()).hexdigest()


def native_report(control):
    inside = [[False] * len(control.signs) for _ in control.signs]
    for i, p in enumerate(control.parents):
        while p != -1:
            inside[i][p] = True; p = control.parents[p]
    rows = []
    for i, sign in enumerate(control.signs):
        faces = control.faces[control.labels == i]
        ids = np.unique(faces)
        tri = control.vertices[faces] - control.vertices[ids[0]]
        volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
        rows.append(dict(original_component_id=i, original_vertices=len(ids), original_faces=len(faces),
                         witness_original_vertex=int(ids[0]), exact_volume_sign=sign, diagnostic_signed_volume=float(volume)))
    return dict(schema="world_reward.certified_solid_query.v1", status="pass", source_sha256=SHA,
                cgal_version="6.0.1", vertices=len(control.vertices), faces=len(control.faces),
                component_count=len(rows), components=rows, inside=inside,
                represented_coordinates="EPECK exact values of original parsed IEEE754 binary64; no perturbation",
                **{k: True for k in module.TRUE_FLAGS}, **{k: False for k in module.FALSE_FLAGS})


def mock_native(monkeypatch, tmp_path, mutate=None):
    binary = tmp_path / "dummy-binary"
    binary.write_bytes(b"source-contract dummy, not executable")
    bank = module.controls()
    lookup = {module.control_input(c): c for c in bank}
    calls = []
    def run(argv, *, input, capture_output, timeout, check):
        control = lookup[input]; calls.append(control.name)
        assert argv == [str(binary)] and capture_output and not check and 0 < timeout <= 60
        if mutate:
            special = mutate(control, binary)
            if special is not None:
                return special
        if control.expected == "native_reject":
            return SimpleNamespace(returncode=1, stdout=b"", stderr=("certified_solid_query FAIL: " + control.rejection + "\n").encode())
        return SimpleNamespace(returncode=0, stdout=json.dumps(native_report(control)).encode(), stderr=b"")
    monkeypatch.setattr(module.subprocess, "run", run)
    return binary, calls


def test_fresh_outward_box_triangles_have_positive_exact_axis_normals():
    v, f = module.box((-2, -3, -4), (2, 3, 4))
    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, tri.mean(axis=1)) > 0)
    inward = module.box((-2, -3, -4), (2, 3, 4), -1)[1]
    np.testing.assert_array_equal(inward, f[:, ::-1])


def test_six_positive_sources_are_original_dyadic_similarities_with_owned_arrays():
    bank = module.controls()
    assert len(bank) == module.MAX_CALLS == 15 and len({c.name for c in bank}) == 15
    for original, transformed in zip(bank[:6:2], bank[1:6:2]):
        np.testing.assert_array_equal(transformed.vertices, original.vertices[:, [2, 0, 1]] * 8 + [16, -8, 4])
        np.testing.assert_array_equal(transformed.faces, original.faces)
        assert transformed.parents == original.parents and transformed.signs == original.signs
        for array in (original.vertices, original.faces, original.labels, transformed.vertices):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flags.writeable = True


def test_serializer_roundtrip_covers_every_original_vertex_face_and_component():
    for control in module.controls()[:6]:
        tokens = module.control_input(control).decode("ascii").split()
        assert tokens[:4] == ["WR_SOLID_QUERY_V1", str(len(control.vertices)), str(len(control.faces)), str(len(control.signs))]
        nv = len(control.vertices)
        vertices = np.array(tokens[4:4 + 3 * nv], dtype=np.float64).reshape(nv, 3)
        faces = np.array(tokens[4 + 3 * nv:], dtype=np.int64).reshape(-1, 4)
        assert vertices.tobytes() == control.vertices.tobytes()
        np.testing.assert_array_equal(faces[:, :3], control.faces)
        np.testing.assert_array_equal(faces[:, 3], control.labels)


def test_self_crossing_fixture_has_nondegenerate_triangles_nonzero_volume_and_real_crossing():
    control = next(c for c in module.controls() if c.name == "self_crossing")
    tri = control.vertices[control.faces]
    assert np.all(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) > 0)
    assert np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() != 0
    # Edge(4,6) pierces the opposite z=-1 surface, not a shared simplex.
    edge_point = control.vertices[4] + (2 / 3) * (control.vertices[6] - control.vertices[4])
    assert edge_point[2] == -1 and np.all(np.abs(edge_point[:2]) < 1)
    assert control.rejection == "Exact component self-intersection"


def test_mock_full_controls_checks_all_passes_rejections_and_binary_prepost(monkeypatch, tmp_path):
    binary, calls = mock_native(monkeypatch, tmp_path)
    report = module.run_controls(binary, SHA)
    assert report["status"] == "pass" and len(calls) == 15
    assert report["artifacts_before"] == report["artifacts_after"]
    assert report["source_binary_rehashed_after"]
    assert sum(row.get("native_rejection_verified", False) for row in report["records"]) == 8
    assert report["records"][-1]["invalid_orientation_forest_rejected"]
    assert all(row["input_sha256"] and row["stdout_sha256"] for row in report["records"])
    assert not report["production_mesh_validated"] and not report["adoption"]


@pytest.mark.parametrize("field,value", [("source_sha256", "b" * 64), ("cgal_version", "6.1"),
                                         ("vertices", True), ("component_self_intersections_absent", False),
                                         ("forest_adjudicated", True), ("inside", [[False]])])
def test_native_metadata_drift_stops_whole_bank(monkeypatch, tmp_path, field, value):
    def mutate(control, _):
        if control.expected != "material":
            return None
        report = native_report(control); report[field] = value
        return SimpleNamespace(returncode=0, stdout=json.dumps(report).encode(), stderr=b"")
    binary, calls = mock_native(monkeypatch, tmp_path, mutate)
    with pytest.raises(module.ControlError) as error:
        module.run_controls(binary, SHA)
    assert error.value.report["status"] == "fail" and len(calls) == 1
    assert error.value.report["source_binary_rehashed_after"]


@pytest.mark.parametrize("mode", ["timeout", "crash", "wrong_rejection", "mutate_binary"])
def test_timeout_crash_or_wrong_negative_reason_are_not_expected_passes(monkeypatch, tmp_path, mode):
    def mutate(control, binary):
        if mode == "mutate_binary":
            binary.write_bytes(b"changed during run"); return None
        if control.name != "contact":
            return None
        if mode == "timeout":
            raise module.subprocess.TimeoutExpired(str(binary), 60)
        return SimpleNamespace(returncode=-11 if mode == "crash" else 1, stdout=b"", stderr=b"wrong failure")
    binary, calls = mock_native(monkeypatch, tmp_path, mutate)
    with pytest.raises(module.ControlError) as error:
        module.run_controls(binary, SHA)
    assert error.value.report["status"] == "fail" and len(calls) <= 15
    if mode == "timeout":
        assert error.value.report["failure_type"] == "TimeoutExpired"
        assert len(calls) == 7


def test_duplicate_keys_and_nonfinite_native_json_are_rejected():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(ValueError):
            module.strict_json(raw)
