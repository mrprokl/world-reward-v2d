"""Tiny provenance/replay tests; no libigl process, challenge mesh or GPU."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_mesh_replay_test", infra/"guarded_mesh_diagnose.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def tetra():
    return np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]]), np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])


def history(gate, root, monkeypatch):
    image = "sha256:"+"a"*64; monkeypatch.setenv("WR_IMAGE_ID", image)
    build = {"binary_sha256": "b"*64, "build_report_sha256": "c"*64, "build_info": {"target_faces": 4096}}
    monkeypatch.setattr(gate.guarded, "validate_build", lambda _: build)
    cppsha = gate.sha256(Path(gate.__file__).with_name("mesh_guarded_qem.cpp"))
    monkeypatch.setattr(gate, "HISTORICAL_CPP_SHA", cppsha)
    base = root/"outputs/episode_000000"; objdir = base/"object_grounded"; objdir.mkdir(parents=True)
    glb = objdir/"object.glb"; glb.write_bytes(b"tiny own source fixture, not parsed GLB")
    obj = {"stage": "sam3d_objects_grounded_fixed_frame", "status": "pass", "episode_index": 0,
           "object_sha256": gate.sha256(glb), "input_sha256": "d"*64, "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": []}
    op = objdir/"report.json"; op.write_text(json.dumps(obj))
    cp = root/"validation/guarded_qem_v1/report.json"; cp.parent.mkdir(parents=True)
    control = {"stage": gate.guarded.STAGE, "status": "pass", "script_sha256": gate.HISTORICAL_HELPER_SHA, "build": build,
        "target_faces": 4096, "target_vertices": 4096, "challenge_inputs_used": False, "adoption_performed": False,
        "fixtures": [{"fixture": "new_close_asymmetric_shells", "independent_intersecting_faces": 0, "source_arrays_unchanged": True,
                      "containment": {"true_containment_verified": True}}, {"fixture": "new_disconnected_smooth_asymmetric", "independent_intersecting_faces": 0, "source_arrays_unchanged": True}]}
    cp.write_text(json.dumps(control))
    old = {"stage": "world_reward_cpu_guarded_object_mesh", "status": "fail", "episode_index": 0,
        "producer_revision": gate.HISTORICAL_REVISION, "script_sha256": gate.HISTORICAL_PRODUCER_SHA, "image_id": image,
        "input_track": "track_1", "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [], "adoption_performed": False,
        "error_type": "ValueError", "error": gate.NUMERICAL_REJECTION, "source_intersecting_faces": 0,
        "independent_candidate_intersecting_faces": 0, "target_faces": 4096, "target_vertices": 4096,
        "input_sha256": obj["input_sha256"], "source_hashes": {"object.glb": gate.sha256(glb), "object_report": gate.sha256(op)},
        "control_evidence": {"guarded_gate_sha256": gate.sha256(cp), **build}}
    rp = base/"object_budget_guarded/report.json"; rp.parent.mkdir(); rp.write_text(json.dumps(old))
    return rp, cp, op, glb, old


def test_historical_sha_control_does_not_become_current_gate(gate, tmp_path, monkeypatch):
    history(gate, tmp_path, monkeypatch)
    paths, old, receipts = gate.prerequisites(tmp_path, 0)
    assert old["status"] == "fail" and receipts["historical_helper_sha256"] == gate.HISTORICAL_HELPER_SHA
    assert receipts["historical_helper_sha256"] != gate.sha256(Path(gate.guarded.__file__))
    assert len(paths) == 4


@pytest.mark.parametrize("fault", ["old_revision", "producer_sha", "helper_sha", "cpp_sha", "source_glb", "control_hash", "gate_adopted", "oracle", "episode_bool", "wrong_error", "image", "object_video"])
def test_changed_or_unproved_historical_evidence_fails(gate, tmp_path, monkeypatch, fault):
    rp, cp, op, glb, old = history(gate, tmp_path, monkeypatch)
    if fault == "old_revision": old["producer_revision"] = "e"*40
    elif fault == "producer_sha": old["script_sha256"] = "e"*64
    elif fault == "helper_sha":
        c = json.loads(cp.read_text()); c["script_sha256"] = gate.sha256(Path(gate.guarded.__file__)); cp.write_text(json.dumps(c)); old["control_evidence"]["guarded_gate_sha256"] = gate.sha256(cp)
    elif fault == "cpp_sha": monkeypatch.setattr(gate, "HISTORICAL_CPP_SHA", "e"*64)
    elif fault == "source_glb": glb.write_bytes(b"changed")
    elif fault == "control_hash": cp.write_text(cp.read_text()+"\n")
    elif fault == "gate_adopted": old["adoption_performed"] = True
    elif fault == "oracle": old["oracle_modes"] = ["test"]
    elif fault == "episode_bool": old["episode_index"] = False
    elif fault == "wrong_error": old["error"] = "different failure"
    elif fault == "image": monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"e"*64)
    elif fault == "object_video":
        obj = json.loads(op.read_text()); obj["input_sha256"] = "e"*64; op.write_text(json.dumps(obj)); old["source_hashes"]["object_report"] = gate.sha256(op)
    rp.write_text(json.dumps(old))
    with pytest.raises(ValueError): gate.prerequisites(tmp_path, 0)


@pytest.mark.parametrize("amount,expected", [(0., False), (.009, False), (.011, True)])
def test_rejection_axis_uses_unchanged_strict_boundaries(gate, amount, expected):
    r = gate.rejection_axes({"sampled_bidirectional_chamfer_diagonal_ratio": amount, "net_volume_relative_error": .05,
        "birthface_matched_shells": [{"source_component": 7, "relative_volume_error": .05}, {"source_component": 9, "relative_volume_error": .06}]})
    assert r["sampled_chamfer_diagonal_over_1_percent"] is expected
    assert r["net_volume_over_5_percent"] is False and r["source_components_over_5_percent"] == [9]


def test_same_solver_measurements_saved_even_when_numerical_rejection(gate, tmp_path, monkeypatch):
    rp, cp, op, glb, old = history(gate, tmp_path, monkeypatch)
    v, f = tetra(); source_v, source_f, _ = gate.endpoint.exact_weld(v, f); candidate = source_v*.9, source_f.copy()
    old.update(source_topology=gate.mesh_topology(source_v, source_f), candidate_topology=gate.mesh_topology(*candidate)); rp.write_text(json.dumps(old))
    frozen = {p: p.read_bytes() for p in (rp, cp, op, glb)}
    monkeypatch.setattr(gate.endpoint, "_load_mesh", lambda _: (v.copy(), f.copy()))
    mapping = {"source_vertices": 4, "source_faces": 4, "output_vertices": 4, "output_faces": 4,
               "target_reached": True, "mapping_complete": True, "J": list(range(4)), "I": list(range(4))}
    calls = []; monkeypatch.setattr(gate.guarded, "simplify", lambda source, timeout: (calls.append(timeout) or (candidate, mapping)))
    monkeypatch.setattr(gate, "source_intersections", lambda *_: 0)
    output = tmp_path/"diagnostic/episode000000-guarded-replay"; output.mkdir(parents=True); report = {}
    gate.measure(tmp_path, 0, report, output/"report.json")
    assert calls == [110] and report["status"] == "measurement_complete" and report["unchanged_geometry_gate_status"] == "fail"
    assert report["measurements"]["net_volume_relative_error"] == pytest.approx(1-.9**3)
    assert report["rejection_axes"]["net_volume_over_5_percent"] is True
    assert report["rejection_axes"]["source_components_over_5_percent"] == [0]
    with np.load(output/"candidate_diagnostic.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["vertices"], candidate[0]) and np.array_equal(saved["faces"], candidate[1])
    assert json.loads((output/"mapping_diagnostic.json").read_text()) == mapping
    assert all(p.read_bytes() == value for p, value in frozen.items())


def test_early_failure_freezes_partial_no_heavy_job(gate, tmp_path, monkeypatch):
    output = tmp_path/"diagnostic/episode000000-guarded-replay"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40); monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64)
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux"); original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="Missing bounded"): gate.main(["--episode", "0"])
    p = output/"report.json"; frozen = p.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["historical_rejection_unchanged"] is True
    assert report["historical_candidate_bit_identity_proven"] is False and report["production_gate_pass_claimed"] is False
    with pytest.raises(FileExistsError): gate.main(["--episode", "0"])
    assert p.read_bytes() == frozen


def test_wrapper_minimal_readonly_no_new_production_gate(gate):
    wrapper = Path(gate.__file__).with_name("run_guarded_mesh_diagnose.sh").read_text()
    assert "--gpus" not in wrapper and "123s docker run" in wrapper and "--entrypoint python" in wrapper
    assert 'src=$BASE/object_budget_guarded/report.json,dst=$BASE/object_budget_guarded/report.json,readonly' in wrapper
    assert 'src=$BASE/object_grounded/object.glb,dst=$BASE/object_grounded/object.glb,readonly' in wrapper
    assert "mesh_guarded_qem.cpp" in wrapper and "chown -R" not in wrapper
    assert not any(f"src=$ROOT/{name}" in wrapper for name in ("data", "weights", "outputs,dst", "results,dst", "validation,dst"))
    with pytest.raises(SystemExit): gate.main(["--episode", "1"])
