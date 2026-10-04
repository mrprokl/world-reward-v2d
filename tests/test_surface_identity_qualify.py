"""Offline manufactured callback contracts, never real native/official qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def q(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    spec = importlib.util.spec_from_file_location("surface_qualify_test", ROOT / "infra/surface_identity_qualify.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_frozen_positive_sources_and_budgets(q):
    from world_reward.surface_identity import SurfaceIdentity
    rows = q.fixtures(np)
    assert tuple(name for name, _ in rows) == q.NAMES
    assert q.BUDGET == 60 and q.OUTER_SECONDS == 63
    assert [(len(v), len(f)) for _, (v, f) in rows] == [(4, 2), (8, 8), (7, 3)]
    for _, (v, f) in rows:
        assert v.dtype == np.float32 and f.dtype == np.int64
        assert np.array_equal(v.astype(np.float64).astype(np.float32), v)
    tube = SurfaceIdentity(*rows[1][1])
    assert len(tube.boundary_loops) == 2 and tube.diagnostics["closed"] is False
    separate = SurfaceIdentity(*rows[2][1])
    assert len(separate.component_keys) == 2
    assert rows[2][1][0][4:, 0].tolist() == [4, 5, 4]


def callbacks(q, tmp_path, fault=None):
    current = {}; counts = dict(native=0, authority=0, budget=0)
    class Mesh:
        def __init__(self, *, vertices, faces, process):
            assert process is False
            self.v, self.f = vertices, faces
        def export(self, path):
            assert not path.exists()
            path.write_bytes(b"manufactured opaque fake GLB")
            current[path] = (self.v.copy(), self.f.copy())
    def load(path):
        counts["native"] += 1
        v, f = current[path]
        if fault == "native_dtype": return v.astype(np.float64), f
        return v.copy(), f.copy()
    def authorities(path, ep):
        counts["authority"] += 1
        assert path.stat().st_mode & 0o777 == 0o444
        v, f = current[path]
        assert ep.object_vertices.dtype == np.float32 and ep.object_faces.dtype == np.int64
        if fault == "source_mutation":
            path.chmod(0o644); path.write_bytes(b"changed source"); path.chmod(0o444)
        return v.astype(np.float64), f.copy(), dict(original_native_FP32_loader_replayed=True,
            native_geometry_byte_exact=fault != "fake_native_proof", full_scene_instances_verified=True,
            model_imports=False, scene_instances=1)
    def budget(path, *, faces, vertices):
        assert faces == vertices == 4096
        counts["budget"] += 1
        v, f = current[Path(path)]
        # Deliberately permute vertices and rows: full multiset/boundary identity,
        # not incidental loader index order, defines preservation.
        order = np.arange(len(v))[::-1]; inverse = np.argsort(order)
        pv = np.vstack((v[order].astype(np.float64), np.repeat(v[order][:1], 4096 - len(v), axis=0)))
        pf = np.vstack((np.roll(inverse[f[::-1]], 1, axis=1), np.zeros((4096 - len(f), 3), np.int64)))
        if fault == "shrink": pv *= .99
        if fault == "flip": pf[0] = pf[0][::-1]
        if fault == "drop": pf[0] = 0
        if fault == "nonzero_padding": pf[-1] = [0, 0, 1]
        if fault == "budget": pv = pv[:-1]
        return pv, pf
    return SimpleNamespace(Trimesh=Mesh), load, authorities, budget, counts


def test_mock_controls_order_exact_shapes_counts_identity_rejections(q, tmp_path):
    tm, load, authority, budget, counts = callbacks(q, tmp_path)
    deadlines = []
    result = q.controls(np, tm, budget, load, authority, tmp_path, lambda: deadlines.append(True))
    assert result["QEM_calls"] == 0 and result["native_loader_calls_including_authority_replays"] == 6
    assert result["counts"] == dict(exports=3, native_loader_attempts=3, native_loader_returns=3,
        authority_attempts=3, authority_returns=3, budget_attempts=3, budget_returns=3)
    assert counts == dict(native=3, authority=3, budget=3)
    assert result["rejected_domain_controls"] == ["invalid_index", "collinear", "nonmanifold_edge"]
    assert len(deadlines) == 6
    assert {p.name for p in tmp_path.iterdir()} == {n + ".glb" for n in q.NAMES}
    assert [r["boundary_loops"] for r in result["records"]] == [1, 2, 2]
    assert all(r["component_boundary_geometry_exact"] and r["simplification_performed"] is False
               for r in result["records"])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["native_dtype", "fake_native_proof", "source_mutation", "shrink", "flip", "drop", "nonzero_padding", "budget"])
def test_mock_failures_stop_no_retry_or_later_fixture(q, tmp_path, fault):
    tm, load, authority, budget, counts = callbacks(q, tmp_path, fault)
    with pytest.raises(ValueError): q.controls(np, tm, budget, load, authority, tmp_path, lambda: None)
    assert counts["native"] == 1 and counts["authority"] <= 1 and counts["budget"] <= 1
    assert {p.name for p in tmp_path.iterdir()} == {q.NAMES[0] + ".glb"}


def test_positive_source_inputs_remain_byte_immutable(q, monkeypatch, tmp_path):
    rows = q.fixtures(np); before = [(v.tobytes(), f.tobytes()) for _, (v, f) in rows]
    monkeypatch.setattr(q, "fixtures", lambda _: rows)
    tm, load, authority, budget, _ = callbacks(q, tmp_path)
    q.controls(np, tm, budget, load, authority, tmp_path, lambda: None)
    assert before == [(v.tobytes(), f.tobytes()) for _, (v, f) in rows]


def test_deadline_stops_before_native_calls(q, tmp_path):
    tm, load, authority, budget, counts = callbacks(q, tmp_path)
    def timeout(): raise TimeoutError("fixed budget")
    with pytest.raises(TimeoutError): q.controls(np, tm, budget, load, authority, tmp_path, timeout)
    assert counts == dict(native=0, authority=0, budget=0) and not list(tmp_path.iterdir())


def test_mounts_exact_code_markers_receipt_six_official_two_native_no_assets(q):
    root = Path("/srv/scenesmith/world-reward"); code = root / "jobs" / ("a" * 40) / q.ENTRY / "code"
    paths = q.mount_paths(root, code)
    assert len(paths) == 10 and paths[0] == code.parent
    assert not any("sample" in str(p) or "weights" in str(p) or "outputs" in p.parts for p in paths)
    assert sum(p.is_relative_to(root / "vendor/v2d_submission_kit") for p in paths) == 6
    assert sum(p.is_relative_to(root / "vendor/video_to_data") for p in paths) == 2


def test_write_is_exclusive_readable_receipt_and_no_numpy_arrays(q, tmp_path):
    path = tmp_path / "report.json"; q.write(path, {"status": "fixture_only"})
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError): q.write(path, {})
    with pytest.raises(TypeError): q.write(tmp_path / "bad.json", {"geometry": np.zeros((3, 3))})
    assert not (tmp_path / "bad.json").exists()


def test_wrapper_fixed_offline_cpu_source_closure_no_model_or_dataset_mounts(q):
    script = ROOT / "infra/run_surface_identity_qualify.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    raw = script.read_text()
    for required in ("63s docker run", "--network none", "--read-only", "--cap-drop ALL", "--user 1000:1000",
        "--cpus 4", "--memory 4g", "noexec,size=128m", "--before", "--status", "inspect_owned", "docker rm -f"):
        assert required in raw
    assert "--gpus" not in raw and "sample" not in raw and "weights/" not in raw
    assert q.IMAGE in raw and "set +x" in raw
    from azure_job import runtime_bundle_paths
    files = {str(p.relative_to(ROOT)): p.read_bytes() for folder in ("infra", "src", "configs")
             for p in (ROOT / folder).rglob("*") if p.is_file() and not p.is_symlink()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    selected = set(runtime_bundle_paths(files, "infra/run_surface_identity_qualify.sh"))
    assert set(q.HELPERS) <= selected
    assert "src/world_reward/raw_shape_proposal.py" in selected


def native_pass(q, proof):
    return dict(stage=q.STAGE, status="pass", phase="complete", producer_revision="a" * 40,
        source_proof=proof, source_runtime_rehashed_after=True, owned_scratch_removed=True,
        GPU_used=False, datasets_read=False, models_read=False, private_values_read=False,
        body_model_constructed=False, reconstruction_accuracy_verified=False, adoption=False,
        elapsed_seconds=1., controls=dict(records=[dict(name=n) for n in q.NAMES],
            counts=dict(exports=3, native_loader_attempts=3, native_loader_returns=3,
                authority_attempts=3, authority_returns=3, budget_attempts=3, budget_returns=3),
            native_loader_calls_including_authority_replays=6, QEM_calls=0,
            negative_GLBS_written=False, under_budget_sources=True,
            all_fixture_sources_frozen_before_measurement=True, all_fixture_sources_rehashed_after=True,
            fixture_manifest=[{"manufactured_test_identity": i} for i in range(6)],
            rejected_domain_controls=["invalid_index", "collinear", "nonmanifold_edge"]))


@pytest.mark.parametrize("fault", [None, "GPU_used", "source_runtime_rehashed_after", "counts", "elapsed", "QEM",
    "negative_GLB", "face_loss", "boundary_change"])
def test_host_seal_requires_all_actual_native_scope_and_counts(q, monkeypatch, tmp_path, fault):
    proof = dict(source="manufactured_proof_only")
    monkeypatch.setattr(q, "host_proof", lambda *_: proof)
    monkeypatch.setattr(q, "output", lambda *_: tmp_path)
    monkeypatch.setattr(q, "container_absence", lambda *_args, **_kw: dict(verified=True, CID_identity="manufactured"))
    row = native_pass(q, proof)
    if fault == "counts": row["controls"]["counts"]["budget_returns"] = 2
    elif fault == "elapsed": row["elapsed_seconds"] = 61.
    elif fault == "QEM": row["controls"]["QEM_calls"] = 1
    elif fault == "negative_GLB": row["controls"]["negative_GLBS_written"] = True
    elif fault == "face_loss": row["controls"]["records"][0]["meaningful_faces_preserved"] = False
    elif fault == "boundary_change": row["controls"]["records"][0]["component_boundary_geometry_exact"] = False
    elif fault: row[fault] = not row[fault]
    row["controls"]["fixture_manifest_sha256"] = hashlib.sha256(json.dumps(
        row["controls"]["fixture_manifest"], sort_keys=True).encode()).hexdigest()
    for record in row["controls"]["records"]:
        record.setdefault("component_boundary_geometry_exact", True)
        record.setdefault("meaningful_faces_preserved", True)
        record["simplification_performed"] = False
    q.write(tmp_path / "native.json", row)
    if fault:
        with pytest.raises(ValueError): q.seal(tmp_path, tmp_path, "a" * 40, json.dumps(proof), 0, cleanup_verified=True)
        assert not (tmp_path / "report.json").exists()
    else:
        assert q.seal(tmp_path, tmp_path, "a" * 40, json.dumps(proof), 0, cleanup_verified=True) == 0
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["status"] == "pass" and report["owned_container_removed"]
        assert tmp_path.stat().st_mode & 0o777 == 0o555
        assert not report["adoption"] and not report["reconstruction_accuracy_verified"]


def test_all_six_source_manifest_frozen_before_any_predicate_or_runtime_call(q, monkeypatch, tmp_path):
    import world_reward.surface_identity as primitive
    progress = {}; real = primitive.SurfaceIdentity
    manifest_seen = []
    def observe(*args, **kwargs):
        assert len(progress["fixture_manifest"]) == 6
        assert progress["all_fixture_sources_frozen_before_measurement"] is True
        manifest_seen.append(progress["fixture_manifest_sha256"])
        if len(manifest_seen) <= 4:
            assert all(not a.flags.writeable for a in args[:2])  # Six original sources frozen, not loader returns.
        return real(*args, **kwargs)
    monkeypatch.setattr(primitive, "SurfaceIdentity", observe)
    tm, load, authority, budget, _ = callbacks(q, tmp_path)
    result = q.controls(np, tm, budget, load, authority, tmp_path, lambda: None, progress)
    assert len(set(manifest_seen)) == 1 and result["all_fixture_sources_rehashed_after"]
    digest = hashlib.sha256(json.dumps(result["fixture_manifest"], sort_keys=True).encode()).hexdigest()
    assert digest == result["fixture_manifest_sha256"]


def test_failure_retains_actual_counts_current_phase_manifest(q, tmp_path):
    progress = {}; tm, load, authority, budget, _ = callbacks(q, tmp_path, "drop")
    with pytest.raises(ValueError): q.controls(np, tm, budget, load, authority, tmp_path, lambda: None, progress)
    assert progress["counts"]["budget_returns"] == 1
    assert progress["current_case"] == q.NAMES[0] and progress["current_phase"] == "official_budget_mesh"
    assert progress["all_fixture_sources_rehashed_after"] and len(progress["fixture_manifest"]) == 6


@pytest.mark.parametrize("mode", ["absent", "daemon", "found", "timeout", "malformed"])
def test_independent_post_container_absence_distinguishes_real_daemon_errors(q, monkeypatch, tmp_path, mode):
    monkeypatch.setattr(q, "output", lambda *_: tmp_path / "out")
    cid = "a" * 64; path = tmp_path / "out.container.cid"; path.write_text(cid);path.chmod(0o400)
    real = q.modules()[0]
    identity = real.identity
    monkeypatch.setattr(q, "modules", lambda: (SimpleNamespace(identity=identity, require=real.require), None))
    original = Path.stat
    def root_stat(p, *args, **kwargs):
        result = original(p, *args, **kwargs)
        # Avoid overriding stat fields used by the real identity function.
        if p == path:
            return SimpleNamespace(**{n:getattr(result,n)for n in dir(result)if n.startswith("st_")}|{"st_uid":0})
        return result
    monkeypatch.setattr(Path, "stat", root_stat)
    def run(args, **kwargs):
        assert kwargs["timeout"] == 5 and args[:3] == ["docker", "inspect", cid]
        if mode == "timeout": raise subprocess.TimeoutExpired(args, 5)
        return SimpleNamespace(returncode=1 if mode != "found" else 0,
            stdout=(cid.encode() if mode == "found" else b""),
            stderr=(f"Error: No such object: {cid}".encode() if mode == "absent" else b"daemon unavailable"))
    monkeypatch.setattr(q.subprocess, "run", run)
    if mode == "absent": assert q.container_absence(tmp_path, "a" * 40)["verified"]
    else:
        with pytest.raises((ValueError, subprocess.TimeoutExpired)): q.container_absence(tmp_path, "a" * 40)
