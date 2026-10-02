"""Tiny own arrays and strict artifact tests; no real images/models locally."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("identity_diagnostic_test", infra/"identity_rgb_diagnostic.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_translation_exact_decomposition_no_correction(gate):
    truth = np.arange(300).reshape(100, 3).astype(np.float32)/100
    prediction = truth.astype(np.float64)+[.2, -.1, .5]
    frozen = prediction.tobytes(), truth.tobytes()
    d = gate.decomposition(prediction, truth)
    assert d["centroid_error_xyz_cm"] == pytest.approx([20, -10, 50])
    assert d["camera_rms_cm"] == pytest.approx(np.sqrt(3000))
    assert d["camera_pve_cm"] == pytest.approx(d["camera_rms_cm"])
    assert d["centered_rms_cm"] < 1e-12
    assert d["centroid_squared_error_fraction"] == pytest.approx(1)
    assert frozen == (prediction.tobytes(), truth.tobytes())


def test_centered_error_does_not_disappear(gate):
    truth = np.zeros((4, 3)); p = truth+np.array([[1., 0, 0], [-1., 0, 0], [0, 2., 0], [0, -2., 0]])
    d = gate.decomposition(p, truth)
    assert d["centroid_error_norm_cm"] == 0
    assert d["camera_pve_cm"] == pytest.approx(150)
    assert d["camera_rms_cm"] == pytest.approx(np.sqrt(2.5)*100)
    assert d["camera_rms_cm"] == d["centered_rms_cm"]
    assert d["centroid_squared_error_fraction"] == 0


def test_identity_zero_error_fraction_undefined(gate):
    truth = np.eye(3)
    d = gate.decomposition(truth, truth)
    assert d["camera_rms_cm"] == d["camera_pve_cm"] == d["centered_rms_cm"] == 0
    assert d["centroid_squared_error_fraction"] is None
    json.dumps(d, allow_nan=False)


@pytest.mark.parametrize("fault", ["nan", "inf", "integer", "shape", "short", "masked", "overflow"])
def test_invalid_full_correspondences_no_fallback(gate, fault):
    p, q = np.eye(3), np.eye(3)
    if fault == "nan": p[0, 0] = np.nan
    elif fault == "inf": q[0, 0] = np.inf
    elif fault == "integer": p = p.astype(int)
    elif fault == "shape": q = np.zeros((3, 4))
    elif fault == "short": p, q = p[:2], q[:2]
    elif fault == "masked": p = np.ma.array(p, mask=False)
    else: p *= 1e200
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        gate.decomposition(p, q)


@pytest.mark.parametrize("seed", list(range(5)))
def test_rms_identity_all_vertices_float64(gate, seed):
    rng = np.random.default_rng(seed); p, q = rng.normal(size=(18439, 3)), rng.normal(size=(18439, 3))
    d = gate.decomposition(p, q)
    assert d["vertex_count"] == 18439
    assert d["camera_rms_cm"]**2 == pytest.approx(d["centroid_error_norm_cm"]**2+d["centered_rms_cm"]**2, abs=1e-8)
    assert abs(d["rms_squared_identity_residual_cm2"]) < 1e-8


@pytest.fixture
def historical(gate, tmp_path, monkeypatch):
    e = gate.evaluate
    report = {"stage": e.STAGE, "status": "pass", "phase": "complete",
        "prediction_report_sha256": gate.PREDICTION_SHA, "hyperparameters": e.GATES.copy(),
        "adoption_authorized": False, "all_frames_scored": True, "all_cases_retained": True,
        "no_gt_alignment": True, "predictions_frozen_before_private": True,
        "decision": {"synthetic_identity_hypothesis_supported": False, "adoption_authorized": False},
        "frame_metrics": [], "clip_metrics": [], "private_render_report_sha256": "a"*64,
        "common_object_scale_diagnostics": []}
    path = tmp_path/e.BASE/"quality_v2/report.json"; path.parent.mkdir(parents=True)
    def save():
        if path.exists(): path.chmod(0o644)
        path.write_text(json.dumps(report)); path.chmod(0o444)
        monkeypatch.setattr(gate, "ORIGINAL_SHA", gate.sha256(path))
    save()
    return tmp_path, path, report, save


def test_original_rejection_replayed_in_memory_unchanged(gate, historical, monkeypatch):
    root, path, original, _ = historical
    frozen = path.read_bytes(), path.stat().st_mode
    monkeypatch.setattr(gate.evaluate, "run", lambda root, report: report.update(original))
    result, original_path = gate.historical_rejection(root)
    assert result == original and original_path == path
    assert frozen == (path.read_bytes(), path.stat().st_mode)


@pytest.mark.parametrize("numerical_tamper", [False, True])
def test_real_depth_support_tuple_matches_json_lists_without_numeric_relaxation(gate, historical, monkeypatch, numerical_tamper):
    from world_reward.metric_alignment import fit_shared_depth_scale
    root, path, original, save = historical
    z = np.full((8, 8), 2.); masks = np.ones((8, 8), bool)
    support = fit_shared_depth_scale([z]*5, [z*1.5]*5, [masks]*5, list(range(5))).to_dict()
    assert isinstance(support["frames"], tuple)
    original["common_object_scale_diagnostics"] = json.loads(json.dumps([support])); save()
    frozen = path.read_bytes()
    if numerical_tamper: support["shared_scale"] += 1e-12
    def replay(root, report):
        report.update(original); report["common_object_scale_diagnostics"] = [support]
    monkeypatch.setattr(gate.evaluate, "run", replay)
    if numerical_tamper:
        with pytest.raises(ValueError, match="common_object_scale"): gate.historical_rejection(root)
    else:
        assert gate.historical_rejection(root)[0] == original
    assert frozen == path.read_bytes()


@pytest.mark.parametrize("fault", ["adopt", "gate", "prediction", "aligned", "missingframes", "hash", "writable"])
def test_false_historical_contract_blocks_before_replay(gate, historical, monkeypatch, fault):
    root, path, report, save = historical
    if fault == "adopt": report["decision"]["synthetic_identity_hypothesis_supported"] = True
    elif fault == "gate": report["hyperparameters"]["median_human_relative_gain_min"] = 0.
    elif fault == "prediction": report["prediction_report_sha256"] = "b"*64
    elif fault == "aligned": report["no_gt_alignment"] = False
    elif fault == "missingframes": report["all_cases_retained"] = False
    if fault not in ("hash", "writable"): save()
    elif fault == "hash": monkeypatch.setattr(gate, "ORIGINAL_SHA", "b"*64)
    else: path.chmod(0o644)
    calls = []
    monkeypatch.setattr(gate.evaluate, "run", lambda *args: calls.append(args))
    with pytest.raises(ValueError): gate.historical_rejection(root)
    assert not calls


@pytest.mark.parametrize("field", ["decision", "frame_metrics", "clip_metrics", "prediction_report_sha256",
                                  "private_render_report_sha256", "common_object_scale_diagnostics"])
def test_replay_difference_never_overturns_original(gate, historical, monkeypatch, field):
    root, path, original, _ = historical; frozen = path.read_bytes()
    def replay(root, report): report.update(original); report[field] = "tampered"
    monkeypatch.setattr(gate.evaluate, "run", replay)
    with pytest.raises(ValueError, match="replay differs"): gate.historical_rejection(root)
    assert path.read_bytes() == frozen


def test_public_failure_prevents_first_private_read_actual_evaluator(gate, historical, monkeypatch):
    root, _, _, _ = historical
    def reject(_): raise ValueError("public SHA audit rejected")
    monkeypatch.setattr(gate.evaluate, "public_predictions", reject)
    def unexpected(*args): pytest.fail("Private or proxy processing reached on failed public audit")
    monkeypatch.setattr(gate.evaluate, "common_object_proxies", unexpected)
    with pytest.raises(ValueError, match="public SHA"): gate.historical_rejection(root)
    assert not (root/gate.evaluate.BASE/"eval_private").exists()


def test_cpu_readonly_wrapper_no_aligned_prediction_export(gate):
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_identity_rgb_diagnostic.sh").read_text()
    assert "--gpus" not in wrapper and "--network none" in wrapper
    assert 'src=$OUT,dst=$OUT,readonly' in wrapper
    assert 'src=$QUALITY/report.json,dst=$QUALITY/report.json,readonly' in wrapper
    assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in wrapper
    assert "np.save" not in source and "fit_sim" not in source
    assert '"posthoc_diagnostic_only": True' in source and '"new_quality_selection_performed": False' in source
    with pytest.raises(SystemExit): gate.main(["--clip", "0"])


@pytest.fixture
def frozen_geometry(gate, tmp_path, monkeypatch):
    """Real own NPZ/hash/ABI reads; only previously-tested audit entrypoints stubbed."""
    e = gate.evaluate; private = tmp_path/e.BASE/"eval_private"; private.mkdir(parents=True)
    records, pairs, cases, metrics = [], [], [], []
    vertices = np.zeros((18439, 3), np.float32); vertices[:, 2] = 3
    faces = np.zeros((36874, 3), np.int64)
    frozen = []
    for clip in range(3):
        for frame in range(5):
            r = dict(file=f"clip_{clip:02d}_frame_{frame:03d}.png", clip_index=clip, frame_index=frame, sha256="a"*64)
            records.append(r)
            pair = dict(human_faces=faces, raw_vertices_camera_m=vertices+np.array([0, 0, .1*(clip+1)]),
                        shared_vertices_camera_m=vertices+np.array([.01, 0, .1*(clip+1)]))
            pairs.append(pair)
            f = (1160., 1480., 1720.)[clip]
            truth = dict(human_vertices_camera_m=vertices, human_faces=faces,
                object_vertices_camera_m=np.zeros((194, 3), np.float32),
                object_faces=np.tile(np.array([[0, 1, 2]], np.int64), (384, 1)),
                camera_K=np.array([[f, 0., 512.], [0., f, 384.], [0., 0., 1.]]),
                scene_depth_m=np.full((768, 1024), np.nan, np.float32),
                visible_face_indices=np.full((768, 1024), -1, np.int64),
                clip_index=np.array(clip, np.int64), frame_index=np.array(frame, np.int64))
            path = private/(Path(r["file"]).stem+".npz")
            with path.open("xb") as handle: np.savez_compressed(handle, **truth)
            path.chmod(0o444); frozen.append((path, gate.sha256(path)))
            cases.append({k: r[k] for k in ("file", "clip_index", "frame_index")}
                         | dict(rgb_sha256=r["sha256"], truth_sha256=gate.sha256(path)))
            metrics.append({k: r[k] for k in ("clip_index", "frame_index")}
                           | {mode: {"human_pve_cm": gate.decomposition(pair[mode+"_vertices_camera_m"], vertices)["camera_pve_cm"]}
                              for mode in gate.MODES})
    render_path = private/"render-report.json"; render_path.write_text(json.dumps(dict(cases=cases))); render_path.chmod(0o444)
    original_path = tmp_path/e.BASE/"quality_v2/report.json"; original_path.parent.mkdir()
    original_path.write_text("{}"); original_path.chmod(0o444); monkeypatch.setattr(gate, "ORIGINAL_SHA", gate.sha256(original_path))
    original = dict(private_render_report_sha256=gate.sha256(render_path), frame_metrics=metrics,
                    decision=dict(synthetic_identity_hypothesis_supported=False))
    monkeypatch.setattr(gate, "historical_rejection", lambda root: (original, original_path))
    monkeypatch.setattr(e, "public_predictions", lambda root: (records, {}, [], pairs, {}, [], gate.PREDICTION_SHA))
    frozen += [(render_path, gate.sha256(render_path)), (original_path, gate.sha256(original_path))]
    return tmp_path, original, records, pairs, cases, frozen


def test_actual_fifteen_private_npz_abi_and_hashes_unchanged(gate, frozen_geometry):
    root, original, records, pairs, _, frozen = frozen_geometry
    before = [(str(p), p.stat().st_mode, digest) for p, digest in frozen]
    report = {}; gate.diagnose(root, report)
    assert len(report["frame_metrics"]) == 15 and len(report["clip_mean_frame_metrics"]) == 3
    assert report["all_frames_retained"] and report["original_decision"] == original["decision"]
    for clip in report["clip_mean_frame_metrics"]:
        assert clip["raw"]["camera_pve_cm"] == pytest.approx(10*(clip["clip_index"]+1))
        assert clip["raw"]["centered_rms_cm"] < 1e-10
    assert before == [(str(p), p.stat().st_mode, gate.sha256(p)) for p, _ in frozen]
    assert len(list((root/gate.evaluate.BASE/"eval_private").iterdir())) == 16


@pytest.mark.parametrize("fault", ["missing", "order", "PVE", "hash", "truthABI", "writable"])
def test_private_frozen_integrity_coverage_and_no_reselection(gate, frozen_geometry, fault):
    root, original, records, pairs, cases, frozen = frozen_geometry
    if fault == "missing": original["frame_metrics"].pop()
    elif fault == "order": records[-1]["frame_index"] = 3
    elif fault == "PVE": original["frame_metrics"][-1]["shared"]["human_pve_cm"] = 0
    else:
        path = frozen[-3][0]; path.chmod(0o644)
        if fault == "hash": path.write_bytes(b"corrupt")
        elif fault == "truthABI":
            with np.load(path, allow_pickle=False) as archive: values = {k: archive[k].copy() for k in archive.files}
            values["camera_K"][0, 0] = 1280
            with path.open("wb") as handle: np.savez_compressed(handle, **values)
            cases[-1]["truth_sha256"] = gate.sha256(path)
            render_path = frozen[-2][0]; render_path.chmod(0o644)
            render_path.write_text(json.dumps(dict(cases=cases))); render_path.chmod(0o444)
            original["private_render_report_sha256"] = gate.sha256(render_path)
        if fault != "writable": path.chmod(0o444)
    with pytest.raises(ValueError): gate.diagnose(root, {})
