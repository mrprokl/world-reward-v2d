"""Microarrays/provenance only; no complete human manufacture on the laptop."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("author_human_gate_test", ROOT / "infra/author_human_feasibility.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tiny(g):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float64)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    joints = np.zeros((1, 3), np.float64)
    rig = SimpleNamespace(names=("root",), rest_joints=joints, parents=np.array([-1]),
        local_rotations=np.broadcast_to(np.eye(3), (3, 1, 3, 3)).copy())
    reference = SimpleNamespace(rest_vertices=vertices, faces=faces, weights=np.ones((4, 1)), rig=rig)
    return rig, reference


def certificate(g, v, f):
    return dict(schema="world-reward-own-closed-surface-certificate-v1", status="pass",
        tolerance_m=1e-8, max_candidate_pairs=2_000_000, budget_seconds=30.,
        full_original_faces_retained=True, exact_arithmetic_proof=False,
        mesh=dict(vertices=len(v), faces=len(f), components=1, original_face_coverage=len(f),
            topology_closed_oriented=True, embedding_verified=True,
            original_vertices_sha256=g.array_hash(v), original_faces_sha256=g.array_hash(f)),
        embedding=dict(verified=True, forbidden_intersections=[], nested_or_ambiguous_components=[]))


def install_tiny(g, monkeypatch):
    rig, ref = tiny(g)
    monkeypatch.setattr(g.author, "author_rig", lambda: rig)
    monkeypatch.setattr(g.author, "extract_rest_surface", lambda rig, *, deadline: ref)
    monkeypatch.setattr(g, "author_semantics", lambda reference: {"test_micro_fixture": True})
    def deform(reference, name):
        shift = {"rest": 0., "carry": .1, "fingerbend": .2}[name]
        return ref.rest_vertices + shift, ref.faces, rig.rest_joints + shift
    monkeypatch.setattr(g.author, "deform_reference", deform)
    calls = []
    def audit(v, f, **controls):
        calls.append(controls)
        return certificate(g, v, f)
    monkeypatch.setattr(g, "audit_closed_surface", audit)
    return rig, ref, calls


def test_micro_manufacture_exactly_three_states_single_extraction_no_real_mesh(gate, monkeypatch):
    rig, ref, calls = install_tiny(gate, monkeypatch)
    report = {}
    arrays = gate.manufacture(report, deadline=time.monotonic() + 5)
    assert [s["name"] for s in report["states"]] == ["rest", "carry", "fingerbend"]
    assert calls == [dict(tolerance_m=1e-8, max_candidate_pairs=2_000_000, budget_seconds=30.)] * 3
    assert arrays["vertices"].shape == (3, 4, 3)
    assert np.array_equal(arrays["faces"], ref.faces)
    assert not any("quality" in key for key in arrays)


@pytest.mark.parametrize("field", ["status", "schema", "tolerance_m", "max_candidate_pairs",
    "budget_seconds", "full_original_faces_retained", "exact_arithmetic_proof"])
def test_certificate_original_fields_not_relaxed(gate, field):
    _, ref = tiny(gate); c = certificate(gate, ref.rest_vertices, ref.faces)
    c[field] = None
    with pytest.raises(ValueError): gate.require_certificate(c, ref.rest_vertices, ref.faces)


@pytest.mark.parametrize("field", ["vertices", "faces", "components", "original_face_coverage",
    "topology_closed_oriented", "embedding_verified", "original_vertices_sha256", "original_faces_sha256"])
def test_complete_single_component_exact_original_geometry_required(gate, field):
    _, ref = tiny(gate); c = certificate(gate, ref.rest_vertices, ref.faces)
    c["mesh"][field] = None
    with pytest.raises(ValueError): gate.require_certificate(c, ref.rest_vertices, ref.faces)


@pytest.mark.parametrize("field", ["verified", "forbidden_intersections", "nested_or_ambiguous_components"])
def test_every_failed_embedding_stops(gate, field):
    _, ref = tiny(gate); c = certificate(gate, ref.rest_vertices, ref.faces)
    c["embedding"][field] = None
    with pytest.raises(ValueError): gate.require_certificate(c, ref.rest_vertices, ref.faces)


def test_first_failed_state_no_retry_no_next_pose(gate, monkeypatch):
    rig, ref, calls = install_tiny(gate, monkeypatch)
    def failed(v, f, **controls):
        calls.append(controls); c = certificate(gate, v, f); c["status"] = "fail"; return c
    monkeypatch.setattr(gate, "audit_closed_surface", failed)
    report = {}
    with pytest.raises(ValueError): gate.manufacture(report, deadline=time.monotonic() + 5)
    assert len(calls) == len(report["states"]) == 1
    assert report["phase"] == "certify_original_author_rest"


@pytest.mark.parametrize("fault", ["face_change", "rest_change", "weight_change", "rig_change", "rest_replay", "nan_pose"])
def test_fixed_reference_and_exact_replay_fails_before_certificate(gate, monkeypatch, fault):
    rig, ref, calls = install_tiny(gate, monkeypatch)
    def bad(reference, name):
        vertices, faces, joints = ref.rest_vertices.copy(), ref.faces.copy(), rig.rest_joints.copy()
        if fault == "face_change": faces = faces[::-1]
        elif fault == "rest_change": ref.rest_vertices[0, 0] += .01
        elif fault == "weight_change": ref.weights[0, 0] += .01
        elif fault == "rig_change": rig.local_rotations[0, 0, 0, 0] += .01
        elif fault == "rest_replay": vertices[0, 0] += .01
        else: vertices[0, 0] = np.nan
        return vertices, faces, joints
    monkeypatch.setattr(gate.author, "deform_reference", bad)
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 5)
    assert not calls


@pytest.mark.parametrize("fault", ["weights_shape", "weights_nan", "weights_negative", "weights_sum", "rest_nan", "face_float"])
def test_original_reference_invalid_fail_before_replay(gate, monkeypatch, fault):
    rig, ref, calls = install_tiny(gate, monkeypatch)
    if fault == "weights_shape": ref.weights = np.ones((4, 2))
    elif fault == "weights_nan": ref.weights[0, 0] = np.nan
    elif fault == "weights_negative": ref.weights[0, 0] = -1
    elif fault == "weights_sum": ref.weights[0, 0] = .5
    elif fault == "rest_nan": ref.rest_vertices[0, 0] = np.nan
    else: ref.faces = ref.faces.astype(float)
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 5)
    assert not calls


def test_no_resolution_or_pose_sweep(gate, monkeypatch):
    monkeypatch.setattr(gate.author, "RESOLUTION_M", .01)
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 5)


def test_deadline_stops_before_first_certificate(gate, monkeypatch):
    _, _, calls = install_tiny(gate, monkeypatch)
    with pytest.raises(TimeoutError): gate.manufacture({}, deadline=time.monotonic() - 1)
    assert not calls


def test_private_immutable_exclusive_concise_report(gate, tmp_path):
    path = tmp_path / "report.json"
    gate.write_private_report(path, {"status": "fail", "quality_verified": False})
    assert path.stat().st_mode & 0o777 == 0o400
    assert json.loads(path.read_text())["quality_verified"] is False
    with pytest.raises(FileExistsError): gate.write_private_report(path, {})


def test_hash_dtype_and_shape_are_bound(gate):
    a = np.ones((2, 2), np.float64)
    assert len({gate.array_hash(a), gate.array_hash(a.ravel()), gate.array_hash(a.astype(np.float32))}) == 3


@pytest.mark.parametrize("fault", ["writable", "symlink", "empty"])
def test_source_identity_rejects_mutable_alias_or_empty(gate, tmp_path, fault):
    path = tmp_path / "source.py"
    path.write_text("own\n")
    if fault == "writable":
        path.chmod(0o644)
    elif fault == "empty":
        path.write_text(""); path.chmod(0o444)
    else:
        target = tmp_path / "target.py"; target.write_text("own\n"); target.chmod(0o444)
        path.unlink(); path.symlink_to(target)
    with pytest.raises(ValueError): gate.file_identity(path)


def test_exact_source_identity_before_after_readonly(gate, tmp_path):
    path = tmp_path / "source.py"; path.write_bytes(b"own\n"); path.chmod(0o444)
    before = gate.file_identity(path)
    assert before == gate.file_identity(path)
    assert before["bytes"] == 4 and len(before["sha256"]) == 64


@pytest.mark.parametrize("field", ["version", "source_hash"])
def test_meshing_runtime_primary_pins_not_first_seen(gate, monkeypatch, field):
    class Distribution:
        version = "wrong" if field == "version" else "0.26.0"
        def locate_file(self, name): return Path("/fake") / name
    monkeypatch.setattr(gate.importlib.metadata, "distribution", lambda _: Distribution())
    monkeypatch.setattr(gate, "file_identity", lambda *a, **kw: {"bytes": 1, "sha256": "0" * 64})
    with pytest.raises(ValueError): gate.meshing_source_snapshot()


def test_all_ten_digits_and_each_primitive_endpoint_micro_support(gate):
    rig = gate.author.author_rig()
    points = np.asarray([p.end for p in rig.primitives], dtype=np.float64)
    weights = np.zeros((len(points), len(rig.names)))
    for row, p in enumerate(rig.primitives): weights[row, rig.names.index(p.bone)] = 1.
    ref = SimpleNamespace(rig=rig, rest_vertices=points, weights=weights)
    report = gate.author_semantics(ref)
    assert len(report["primitive_endpoint_checks"]) == 49
    assert len(report["ten_digit_checks"]) == 10
    assert report["complete_anatomical_geometry_proof"] is False


@pytest.mark.parametrize("fault", ["missing_surface", "missing_digit_weight"])
def test_author_support_does_not_excuse_missing_digit_geometry(gate, fault):
    rig = gate.author.author_rig()
    points = np.asarray([p.end for p in rig.primitives], dtype=np.float64)
    weights = np.zeros((len(points), len(rig.names)))
    for row, p in enumerate(rig.primitives): weights[row, rig.names.index(p.bone)] = 1.
    if fault == "missing_surface": points[:] = 20.
    else: weights[:, rig.names.index("l_thumb_dip")] = 0.
    with pytest.raises(ValueError): gate.author_semantics(SimpleNamespace(rig=rig, rest_vertices=points, weights=weights))


def test_no_geometry_io_in_pure_manufacture_source(gate):
    import inspect
    text = inspect.getsource(gate.manufacture)
    assert not any(name in text for name in ("np.save", "open(", "torch", "render(", "simplify("))
