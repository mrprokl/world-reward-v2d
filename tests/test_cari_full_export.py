"""Tiny draft full-N direct export tests: no Torch, models, data or cloud."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    monkeypatch.syspath_prepend(str(ROOT / "infra"))
    # Source/native geometry tests are deliberately independent of peers'
    # concurrent draft import availability. Actual lineage API is audited later.
    monkeypatch.setitem(sys.modules, "cari_full_refine", SimpleNamespace())
    spec = importlib.util.spec_from_file_location("world_reward_draft_full_export", ROOT / "infra/cari_full_export.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


FACES = np.array([[0, 1, 2]], np.int64)
FACE_SHA = hashlib.sha256(FACES.astype("<i4").tobytes()).hexdigest()


def inputs(gate, count=97):
    spec = gate.public.PublicClipSpec(15, count, "front", 2, 4)
    params = {key: np.zeros((count, dim), np.float32) for key, dim in gate.NATIVE_PARAMETER_DIMS.items()}
    params["mhr_global_rot6d"][:] = [1, 0, 0, 1, 0, 0]
    params["mhr_trans"][:, 2] = np.linspace(1., 2., count, dtype=np.float32)
    params["mhr_scale"][:] = .03; params["mhr_shape"][:] = .04
    params["mhr_hand"][:, 0] = np.arange(count, dtype=np.float32) / 100
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy(); pose[:, 2, 3] = 3
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32)
    return params, pose, vertices, FACES.copy(), gate.public.inferred_camera(spec), spec


def points(params):
    result = np.zeros((len(params["mhr_trans"]), 3, 3), np.float32)
    result[:, :, 2] = params["mhr_trans"][:, 2, None]
    result[:, 1, 0] = .1; result[:, 2, 1] = .1
    return result


def callbacks(count, events, fault=None):
    calls = 0
    chunks = (count + 15) // 16
    def native(params):
        nonlocal calls
        calls += 1; events.append(("native", len(params["mhr_trans"])))
        vertices = points(params)
        result = dict(vertices=vertices, joints=np.repeat(vertices[:, :1], 2, axis=1),
                      keypoints=np.repeat(vertices[:, :1], 4, axis=1), faces=FACES.copy())
        if fault == "native_mutation": params["mhr_hand"][0, 0] += 1
        elif fault == "nonfinite": result["vertices"][0, 0, 0] = np.nan
        elif fault == "geometry_dtype": result["vertices"] = result["vertices"].astype(np.float64)
        elif fault == "topology": result["faces"][0] = [0, 2, 1]
        elif fault in ("replay_vertices", "replay_joints", "replay_keypoints") and calls > chunks:
            result[fault.removeprefix("replay_")][0, 0, 0] += .01
        return result
    def direct(params):
        events.append(("direct", len(params["mhr_trans"])))
        vertices = points(params); dc = np.zeros((len(vertices), 204), np.float32)
        dc[:, 2] = params["mhr_trans"][:, 2]; dc[:, 136:] = .03; pca = dc[:, 136:].copy()
        if fault == "direct_error": vertices[0, 0, 0] += .01
        elif fault == "direct_mutation": params["mhr_hand"][0, 0] += 1
        elif fault == "direct_dtype": dc = dc.astype(np.float64)
        elif fault == "direct_geometry_dtype": vertices = vertices.astype(np.float64)
        elif fault == "pca": pca[0, 0] += .01
        elif fault == "vary_scales": dc[0, 136] += .01; pca = dc[:, 136:].copy()
        return vertices, dc, pca
    def reference(stored, selection):
        events.append(("reference", selection.stop - selection.start))
        vertices = np.zeros((selection.stop - selection.start, 3, 3), np.float64)
        vertices[:, :, 2] = stored["pose"][selection, 2, None].astype(np.float64) * 1000
        vertices[:, 1, 0] = float(np.float32(.1)) * 1000
        vertices[:, 2, 1] = float(np.float32(.1)) * 1000
        if fault == "reference_error": vertices += 3
        elif fault == "reference_nonfinite": vertices[0, 0, 0] = np.nan
        elif fault == "reference_mutation": stored["pose"][0, 0] += 1
        return vertices
    return native, direct, reference


def run(gate, tmp_path, count=97, fault=None, receipt=None, events=None, persist=None):
    params, poses, vertices, faces, K, spec = inputs(gate, count)
    mesh = tmp_path.parent / (tmp_path.name + "-mesh.glb"); mesh.write_bytes(b"own tiny no mesh decode")
    receipt = {name + suffix: 0 for name in gate.CALLS for suffix in ("_attempts", "_returns", "_validated")} if receipt is None else receipt
    events = [] if events is None else events
    frozen = gate.export_geometry(params, poses, vertices, faces, K, spec, tmp_path, mesh,
        *callbacks(count, events, fault), receipt, persist or (lambda: None),
        vertex_count=3, joint_count=2, keypoint_count=4, face_count=1, face_sha=FACE_SHA)
    return frozen, receipt, (params, poses, vertices, faces, K, spec), events


@pytest.mark.parametrize("count,chunks", [(96, [16] * 6), (97, [16] * 6 + [1]), (501, [16] * 31 + [5]), (668, [16] * 41 + [12])])
def test_complete_original_four_routes_no_tail_padding_or_identity_reselection(gate, tmp_path, count, chunks):
    frozen, receipt, values, events = run(gate, tmp_path, count)
    assert set(frozen) == gate.OUTPUTS and receipt["phase"] == "geometry_routes_complete"
    assert receipt["frames"] == count and receipt["chunk_counts"] == chunks
    assert receipt["original_frame_indices"] == list(range(count))
    assert receipt["identity_reselection_performed"] is False
    assert receipt["reference_per_frame_mean_mm"] == [0.] * count
    assert all(receipt[name + suffix] == len(chunks) for name in gate.CALLS for suffix in ("_attempts", "_returns", "_validated"))
    assert [n for role, n in events if role == "native"] == chunks * 2
    assert [n for role, n in events if role == "direct"] == chunks
    assert [n for role, n in events if role == "reference"] == chunks
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as stored:
        assert set(stored.files) == gate.TRAJECTORY_KEYS
        assert np.array_equal(stored["frame_index"], np.arange(count))
        assert stored["pose"].shape == (count, 136) and stored["scales"].shape == (68,)
        assert stored["shape"].tobytes() == values[0]["mhr_shape"][0].tobytes()
        assert stored["object_rotation"].tobytes() == values[1][:, :3, :3].tobytes()
        assert stored["object_translation"].tobytes() == values[1][:, :3, 3].tobytes()
        assert stored["camera_K"].tobytes() == values[4].tobytes()
        assert stored["object_scale"].shape == () and stored["object_scale"] == 1
    assert all(not (tmp_path / name).stat().st_mode & 0o222 for name in frozen)


@pytest.mark.parametrize("fault", ["native_mutation", "nonfinite", "geometry_dtype", "topology", "replay_vertices", "replay_joints",
    "replay_keypoints", "direct_error", "direct_mutation", "direct_dtype", "direct_geometry_dtype", "pca", "vary_scales", "reference_error",
    "reference_nonfinite", "reference_mutation"])
def test_each_route_failure_never_completes(gate, tmp_path, fault):
    receipt = {name + suffix: 0 for name in gate.CALLS for suffix in ("_attempts", "_returns", "_validated")}
    with pytest.raises(ValueError): run(gate, tmp_path, fault=fault, receipt=receipt)
    assert receipt.get("phase") != "geometry_routes_complete"
    assert receipt.get("unchanged_refined_predictions_verified") is not True


def test_nonconstant_identity_fails_before_callbacks(gate, tmp_path):
    params, pose, vertices, faces, K, spec = inputs(gate)
    params["mhr_shape"][1, 0] += 1
    events = []; mesh = tmp_path.parent / "own-mesh.glb"; mesh.write_bytes(b"own")
    with pytest.raises(ValueError):
        gate.export_geometry(params, pose, vertices, faces, K, spec, tmp_path, mesh, *callbacks(97, events), {}, lambda: None)
    assert events == []


def test_immutable_predictions_exist_before_saved_native_and_reference(gate, tmp_path):
    receipt = {name + suffix: 0 for name in gate.CALLS for suffix in ("_attempts", "_returns", "_validated")}
    observations = []
    def persist():
        if receipt.get("phase") in ("frozen_native_replay", "official_reference_replay"):
            assert all((tmp_path / name).is_file() and not (tmp_path / name).stat().st_mode & 0o222 for name in gate.OUTPUTS)
            observations.append(receipt["phase"])
    run(gate, tmp_path, receipt=receipt, persist=persist)
    assert "frozen_native_replay" in observations and "official_reference_replay" in observations


@pytest.mark.parametrize("fault", ["extra", "overwrite", "readonly"])
def test_frozen_output_mutation_after_replay_detected(gate, tmp_path, fault):
    receipt = {name + suffix: 0 for name in gate.CALLS for suffix in ("_attempts", "_returns", "_validated")}
    changed = False
    def persist():
        nonlocal changed
        if receipt.get("phase") == "official_reference_replay" and not changed:
            changed = True
            if fault == "extra": (tmp_path / "old_prediction.pt").write_bytes(b"unexpected")
            elif fault == "overwrite":
                path = tmp_path / "trajectory.npz"; path.chmod(0o644); path.write_bytes(b"changed")
            else: (tmp_path / "object_aligned.glb").chmod(0o644)
    with pytest.raises(ValueError, match="frozen native exports changed"):
        run(gate, tmp_path, receipt=receipt, persist=persist)


def test_camera_generic_spec_exact_not_c96_hardcode(gate):
    params, pose, vertices, faces, K, spec = inputs(gate)
    dc = np.zeros((97, 204), np.float32)
    data = gate.trajectory(dc, params, pose, vertices, faces, K, spec)
    assert np.array_equal(data["camera_K"], K) and K[0, 0] != 1920
    K[0, 0] += .01
    with pytest.raises(ValueError): gate.trajectory(dc, params, pose, vertices, faces, K, spec)


@pytest.mark.parametrize("bad", [True, -1, 30, 15.])
def test_explicit_episode_and_output(gate, bad):
    with pytest.raises(ValueError): gate.output_relative(bad)


def test_cli_no_fit_calibration_or_alternative_inputs(gate):
    assert gate.parser().parse_args(["--episode", "15"]).episode == 15
    for args in ([], ["--episode", "15", "--GT"], ["--episode", "15", "--chunk", "1"], ["--episode", "15", "--episode", "15"]):
        with pytest.raises(SystemExit): gate.parser().parse_args(args)
    assert gate.BUDGET == 600 and gate.CHUNK == 16


def test_wrapper_validates_args_without_cli_and_has_ro_firewall():
    path = ROOT / "infra/run_cari_full_export.sh"
    result = subprocess.run(["rtk", "proxy", "bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for args in ([], ["--episode", "30"], ["--episode", "015"], ["--episode", "15", "--overwrite"]):
        result = subprocess.run(["rtk", "proxy", "bash", str(path), *args], capture_output=True, env={"PATH": os.environ["PATH"]})
        assert result.returncode == 2
    source = path.read_text()
    assert "--network none" in source and "603s docker run" in source
    assert "source_paths(spec)" in source and "validate_pins(spec,pins)" in source
    assert "src=$ROOT/outputs,dst=$ROOT/outputs" not in source and "src=$ROOT/data" not in source


def test_full_actual_report_gates_every_frame_and_complete_route_counts(gate, tmp_path):
    _, receipt, values, _ = run(gate, tmp_path, 501)
    spec = values[-1]
    receipt.update(stage=gate.STAGE, status="pass", phase="complete", episode_index=15,
        clip_spec=dict(episode_index=15, total_frames=501, camera_name="front", height=2, width=4), image_id=gate.IMAGE,
        input_track="track_1", ground_truth_used=False, ground_truth_read=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], network="none", budget_seconds=600,
        learned_inference_calls=0, optimizer_calls=0, converter_LM_calls=0, quality_verified=False, adoption_authorized=False,
        submission_produced=False, submission_eligible=False, final_Parquet_produced=False,
        source_inputs_assets_rehashed=True, source_helpers_rehashed=True, raw_masks_contacts_object_pose_unchanged=True,
        full_original_native_export_verified=True, refined_report_sha256="a" * 64, refined_bundle_sha256="b" * 64)
    gate.validate_export_report(receipt, spec)
    for key, bad in (("frames", 500), ("native_replay_returns", 31), ("quality_verified", True),
                     ("identity_reselection_performed", True), ("native_direct_max_point_mm", .0101),
                     ("reference_per_frame_mean_mm", [0.] * 500 + [2.0001]), ("private_truth_read", True)):
        with pytest.raises(ValueError): gate.validate_export_report(receipt | {key: bad}, spec)


def test_actual_lineage_validation_source_then_refined_count(gate, monkeypatch):
    params, poses, vertices, faces, K, spec = inputs(gate)
    source = dict(schema="cari4d.mhr_wild_inference.v1", frames=[f"{i:06d}" for i in range(97)], gt={},
        metadata=dict(ground_truth_used=False), pr=params | dict(pose_abs=poses, contact_logits=np.zeros((97, 2), np.float32)))
    refined = copy.deepcopy(source); refined["postopt"] = {"native": True}
    calls = []
    monkeypatch.setattr(gate.lineage, "validate_result", lambda old, new, n: calls.append((old, new, n)) or {"verified": True}, raising=False)
    monkeypatch.setattr(gate.lineage, "validate_source_bundle", lambda old, mesh, n: calls.append((old, mesh, n)), raising=False)
    actual_params, actual_poses, meta = gate.validate_refined(source, refined, Path("mesh.glb"), 97)
    assert calls[0] == (source, refined, 97) and calls[1] == (source, Path("mesh.glb"), 97)
    assert meta == {"verified": True} and actual_params["mhr_shape"].tobytes() == params["mhr_shape"].tobytes()
    assert actual_poses.tobytes() == poses.tobytes()


def test_actual_peer_imports_and_lineage_abi_without_stub_or_torch():
    """Fresh interpreter checks real integrated modules, not callback fixtures."""
    source = """
import inspect, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root/'src'), str(root/'infra')]
import cari_full_forward as forward
import cari_full_refine as refine
import cari_full_export as export
assert Path(forward.__file__).resolve() == root/'infra/cari_full_forward.py'
assert Path(refine.__file__).resolve() == root/'infra/cari_full_refine.py'
assert Path(export.__file__).resolve() == root/'infra/cari_full_export.py'
assert export.lineage is refine
assert list(inspect.signature(forward.verify_forward_artifacts).parameters) == ['root','code','spec','pins']
assert list(inspect.signature(refine.verify_refined_artifacts).parameters) == ['root','code','spec','pins']
assert list(inspect.signature(refine.validate_result).parameters) == ['source','result','count']
assert list(inspect.signature(refine.validate_source_bundle).parameters) == ['bundle','mesh','count']
assert list(inspect.signature(refine.validate_refinement_report).parameters) == ['report','spec']
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
"""
    result = subprocess.run(["rtk", "proxy", sys.executable, "-c", source, str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
