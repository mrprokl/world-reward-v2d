"""Tiny pinned-receipt/measurement correction, never a full manufacture locally."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    spec = importlib.util.spec_from_file_location("author_v2_test", ROOT / "infra/author_human_feasibility_v2.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tiny(g):
    r = g.author.author_rig()
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return SimpleNamespace(rig=r, rest_vertices=v, faces=f, weights=np.full((4, len(r.names)), 1 / len(r.names)))


def install_micro(g, monkeypatch):
    ref = tiny(g); identity = g.reference_identity(ref)
    monkeypatch.setattr(g, "REFERENCE_PINS", identity)
    monkeypatch.setattr(g.author, "extract_rest_surface", lambda rig, *, deadline: ref)
    monkeypatch.setattr(g.author, "deform_reference", lambda ref, name: (
        ref.rest_vertices.copy(), ref.faces, ref.rig.rest_joints.copy()))
    calls = []
    def certificate(v, f, **kw): calls.append(("certificate", kw)); return {"test": True}
    monkeypatch.setattr(g, "audit_closed_surface", certificate)
    monkeypatch.setattr(g.original, "require_certificate", lambda *a: None)
    def support(ref, *, deadline): calls.append(("support", {})); return {"test": True}
    monkeypatch.setattr(g, "solid_support", support)
    return ref, calls


def test_same_geometry_and_certificate_precedes_corrected_support(gate, monkeypatch):
    ref, calls = install_micro(gate, monkeypatch); report = {}
    arrays = gate.manufacture(report, deadline=time.monotonic() + 2)
    assert report["original_reference_byte_identical"] is True
    assert [n for n, _ in calls] == ["certificate", "support", "certificate", "certificate"]
    assert all(kw == dict(tolerance_m=1e-8, max_candidate_pairs=2_000_000, budget_seconds=30.)
        for name, kw in calls if name == "certificate")
    assert arrays["vertices"].shape == (3, 4, 3)


@pytest.mark.parametrize("key", ["rest_vertices_sha256", "original_faces_sha256", "fixed_weights_sha256", "rig_hashes", "vertices", "faces"])
def test_geometry_or_rig_divergence_stops_before_any_certificate(gate, monkeypatch, key):
    _, calls = install_micro(gate, monkeypatch)
    gate.REFERENCE_PINS = dict(gate.REFERENCE_PINS); gate.REFERENCE_PINS[key] = None
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 2)
    assert not calls


def test_surface_failure_not_rescued_by_support(gate, monkeypatch):
    _, calls = install_micro(gate, monkeypatch)
    def failed(*args): raise ValueError("embedding failed")
    monkeypatch.setattr(gate.original, "require_certificate", failed)
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 2)
    assert len(calls) == 1 and calls[0][0] == "certificate"


def test_corrected_support_failure_stops_before_extreme_poses(gate, monkeypatch):
    _, calls = install_micro(gate, monkeypatch)
    def failed(*args, **kwargs): raise ValueError("digit gap closed")
    monkeypatch.setattr(gate, "solid_support", failed)
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 2)
    assert len(calls) == 1


def test_deadline_never_becomes_retry(gate, monkeypatch):
    _, calls = install_micro(gate, monkeypatch)
    with pytest.raises(TimeoutError): gate.manufacture({}, deadline=time.monotonic() - 1)
    assert not calls


def test_67_exact_queries_and_exposed_brackets_micro_only(gate, monkeypatch):
    rig = gate.author.author_rig(); vertices = np.zeros((2, 3)); weights = np.ones((2, len(rig.names)))
    ref = SimpleNamespace(rig=rig, rest_vertices=vertices, faces=np.zeros((1, 3), np.int64), weights=weights)
    calls = []
    def query(point, v, f, **kw):
        calls.append(kw); return {"status": "pass", "full_original_faces_retained": True}
    monkeypatch.setattr(gate, "point_surface_query", query)
    report = gate.solid_support(ref, deadline=time.monotonic() + 2)
    assert report["query_count"] == len(calls) == 67
    assert sum(c["expected_state"] == "inside" for c in calls) == 49
    assert sum(c["expected_state"] == "outside" for c in calls) == 18
    assert all(c["tolerance_m"] == 1e-8 for c in calls)
    assert report["anatomical_accuracy_verified"] is False


def test_original_receipt_hash_checked_before_json(gate, tmp_path, monkeypatch):
    p = tmp_path / "report.json"; p.write_text("invalid json"); p.chmod(0o400)
    with pytest.raises(ValueError, match="receipt"): gate.original_report(p)


def test_v1_and_geometry_sources_are_independently_pinned(gate):
    for name, pin in gate.ORIGINAL_SOURCES.items():
        assert gate.original.file_identity(ROOT / name, immutable=False) == pin


def test_existing_false_exposure_counterexample_tinyfield(gate):
    rig = gate.author.author_rig(); neck = next(p for p in rig.primitives if p.name == "neck")
    head = next(p for p in rig.primitives if p.name == "head")
    assert neck.end == head.start and min(head.radii) > max(neck.radii) + 2 * gate.author.RESOLUTION_M
    assert gate.author.evaluate_field(np.array([neck.end]), rig)[0] < 0


def test_no_threshold_resolution_pose_change(gate, monkeypatch):
    monkeypatch.setattr(gate.author, "RESOLUTION_M", .006)
    with pytest.raises(ValueError): gate.manufacture({}, deadline=time.monotonic() + 2)
