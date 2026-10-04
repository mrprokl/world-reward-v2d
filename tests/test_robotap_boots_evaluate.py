"""Own tiny metric/NPZ/pickle fixtures; no assets, inference, RGB or GT retrieval."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import time
import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("robotap_eval_test", infra / "robotap_boots_evaluate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def metric_fixture():
    q = np.array([[0., 0., 0.]], dtype=np.float64)
    occ = np.array([[False, False, True, False]])
    gt = np.zeros((1, 4, 2), dtype=np.float64)
    pred = gt.copy(); pred[:, 0] = 999  # query excluded
    pred[0, 3, 0] = 1  # exact threshold1 must fail strict<
    return q, occ, gt, np.zeros_like(occ), pred


def test_native_strict_threshold_first_query_excluded_and_false_positives(gate):
    q, occ, gt, po, pred = metric_fixture(); m = gate.tap_metrics(q, occ, gt, po, pred)
    assert m["occlusion_accuracy"] == 2 / 3
    assert m["pts_within_1"] == .5 and m["jaccard_1"] == .25
    assert m["pts_within_2"] == 1 and m["jaccard_2"] == 2 / 3
    assert m["average_jaccard"] == np.mean([.25, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
    po[0, 1] = True
    m = gate.tap_metrics(q, occ, gt, po, pred)
    assert m["pts_within_1"] == .5 and m["jaccard_1"] == 0  # APD ignores predicted visibility


def test_first_query_mask_different_per_point_and_zero_denominators_fail(gate):
    q = np.array([[0., 0., 0.], [2., 0., 0.]])
    occ = np.zeros((2, 4), dtype=bool); gt = np.zeros((2, 4, 2)); pred = gt.copy()
    pred[1, :3] = 999
    m = gate.tap_metrics(q, occ, gt, occ, pred); assert m["average_jaccard"] == 1
    with pytest.raises(ValueError, match="denominator"): gate.tap_metrics(np.array([[3., 0., 0.]]), occ[:1], gt[:1], occ[:1], gt[:1])
    with pytest.raises(ValueError, match="denominator"): gate.tap_metrics(q, np.ones_like(occ), gt, occ, gt)


def test_native_formula_parity_independent_scalar_microreference(gate):
    rng = np.random.default_rng(42); n, t = 3, 7
    q = np.array([[0., 0., 0.], [2., 0., 0.], [1., 0., 0.]])
    occ = rng.random((n, t)) > .6; po = rng.random((n, t)) > .4
    gt = rng.normal(size=(n, t, 2)); pred = gt + rng.normal(size=(n, t, 2)) * 3
    actual = gate.tap_metrics(q, occ, gt, po, pred)
    evaluated = [(i, j) for i in range(n) for j in range(t) if j > int(q[i, 0])]
    positives = sum(not occ[i, j] for i, j in evaluated)
    assert actual["occlusion_accuracy"] == sum(occ[i, j] == po[i, j] for i, j in evaluated) / len(evaluated)
    for threshold in (1, 2, 4, 8, 16):
        tp = fp = correct = 0
        for i, j in evaluated:
            within = sum((pred[i, j] - gt[i, j]) ** 2) < threshold**2
            correct += bool(within and not occ[i, j])
            tp += bool(within and not occ[i, j] and not po[i, j])
            fp += bool(not po[i, j] and (occ[i, j] or not within))
        assert actual[f"pts_within_{threshold}"] == correct / positives
        assert actual[f"jaccard_{threshold}"] == tp / (positives + fp)


def test_preregistered_three_video_decision_no_static_victory_claim(gate):
    def rows(delta=.06, wins=3, oa=.8):
        return [{"boots": {"average_jaccard": .5 + (delta if i < wins else -.01), "occlusion_accuracy": oa},
            "static": {"average_jaccard": .5, "occlusion_accuracy": .8}} for i in range(3)]
    assert gate.decision(rows())["scientific_decision"] == "ACCEPT_2D_DIAGNOSTIC"
    assert gate.decision(rows(.01))["scientific_decision"] == "REJECT"
    assert gate.decision(rows(.1, wins=1))["scientific_decision"] == "REJECT"
    assert gate.decision(rows(oa=.79))["scientific_decision"] == "REJECT"
    assert not gate.decision(rows())["full_hoi_or_cari4d_victory_verified"]


def example():
    return dict(video=np.arange(4 * 3 * 5 * 3, dtype=np.uint8).reshape(4, 3, 5, 3),
        points=np.full((2, 4, 2), .25, dtype=np.float32), occluded=np.zeros((2, 4), dtype=bool))


def save_npz(gate, path, arrays):
    with gate.source.private_writer(path) as stream: np.savez(stream, **arrays)
    path.chmod(0o444)
    return gate.source.identity(path)


def fixture_files(gate, base):
    public, inference = base / "public_v2", base / gate.INFER_NAMESPACE
    for directory in (public / "inputs", inference / "predictions"): directory.mkdir(parents=True)
    rows, selection, pubfiles, predfiles = [], [], {}, {}
    for i, name in enumerate(gate.NAMES):
        ex = example(); video, queries, indices, unavailable = gate.public.initial_queries(ex); n, t = len(indices), len(video)
        pubfiles[name] = save_npz(gate, public / "inputs" / name, dict(video=video, query_points=queries, point_indices=indices))
        row = dict(file=name, source_pickle=gate.public.PICKLES[i], video_key=f"a{i}", frames=t, height=3, width=5, query_count=n,
            point_indices=indices.tolist(), unavailable_original_indices=unavailable, all_original_frames_retained=True, **pubfiles[name])
        rows.append(row); selection.append(dict(pickle_file=gate.public.PICKLES[i], video_key=f"a{i}"))
        t256 = np.broadcast_to(np.array([64., 64.], dtype=np.float32), (n, t, 2)).copy()
        pred = dict(tracks_256=t256, tracks=t256 * np.array([5, 3], dtype=np.float32) / np.float32(256),
            occlusion=np.full((n, t), -10, dtype=np.float32), expected_dist=np.full((n, t), -10, dtype=np.float32), visible=np.ones((n, t), dtype=bool),
            query_points=queries, point_indices=indices, frame_index=np.arange(t, dtype=np.int64),
            static_tracks=np.broadcast_to(queries[:, [2, 1]][:, None], (n, t, 2)).copy(), static_visible=np.ones((n, t), dtype=bool))
        predfiles[name] = save_npz(gate, inference / "predictions" / name, pred)
    manifest = dict(schema="world-reward-robotap-boots-public-v1", initial_query_is_external_oracle=True, future_tracks_or_visibility_public=False,
        frame_crop_or_resize=False, selection=selection, videos=rows, public_namespace=gate.public.PUBLIC_NAMESPACE, serialization_decoder=gate.public.MEDIAPY_DECODER_SOURCE, training_overlap_verified=False, challenge_overlap_verified=False, full_hoi_accuracy_verified=False)
    gate.source.save_bytes(public / "inputs/manifest.json", json.dumps(manifest).encode(), 0o444)
    gate.source.save_bytes(public / "selection.json", json.dumps(selection).encode())
    allpub = {"manifest.json": gate.source.identity(public / "inputs/manifest.json"), **pubfiles}
    report = dict(status="pass", stage="external_robotap_oracle_initial_query_public_adapter", producer_revision="a" * 40,
        source_before={"files": {"infra/robotap_boots_public.py": {"sha256": "b" * 64}}}, public_files=allpub,
        source_after_reverified=True, originals_after_reverified=True, initial_queries_are_external_oracles=True,
        frozen_selection_before_future_label_access=selection, future_labels_available_to_inference=False, inference_performed=False, evaluation_performed=False, gpu_used=False, challenge_inputs_used=False)
    report["source_after"] = copy.deepcopy(report["source_before"])
    report["serialization_decoder"] = gate.public.MEDIAPY_DECODER_SOURCE
    infer = dict(status="pass", stage="public_robotap_native_bootstapir_predictions", producer_revision="c" * 40, script_sha256="d" * 64,
        native_calls_attempted=3, native_calls_completed=3, actual_native_inference=True, all_original_frames_retained=True,
        all_original_selected_queries_retained=True, all_input_assets_sources_rechecked=True, oracle_initial_queries=True,
        private_pickles_read=False, future_tracks_or_visibility_read=False, evaluation_performed=False, challenge_inputs_used=False, adoption_performed=False)
    infer["inference_config"] = dict(pyramid_level=1, resolution=[256,256], query_chunk_size=32, is_training=False, compute_dtype="float32", AMP=False,
        resize="native utils.bilinear align_corners=False", normalize="RGB float32 /255*2-1", frame_offload=False,
        visibility="(1-sigmoid(occlusion))*(1-sigmoid(expected_dist))>0.5")
    infer["videos"] = [dict(file=row["file"], output=predfiles[row["file"]], input=pubfiles[row["file"]], frames=row["frames"], query_count=row["query_count"], point_indices=row["point_indices"]) for row in rows]
    gate.source.save_bytes(public / "report.json", json.dumps(report).encode())
    gate.source.save_bytes(inference / "report.json", json.dumps(infer).encode(), 0o444)
    return dict(schema="world-reward-robotap-boots-evaluation-pins-v1",
        public_report={**gate.source.identity(public / "report.json"), "producer_revision": "a" * 40, "script_sha256": "b" * 64},
        public_manifest=allpub["manifest.json"], public_selection=gate.source.identity(public / "selection.json"), public_files=pubfiles,
        inference_report={**gate.source.identity(inference / "report.json"), "producer_revision": "c" * 40, "script_sha256": "d" * 64}, prediction_files=predfiles)


@pytest.mark.parametrize("fault", ["partial", "extra", "report", "hash", "shape", "nan", "query", "indices", "frameorder", "scale", "static", "visible"])
def test_frozen_prediction_firewall_before_private_gt(gate, tmp_path, monkeypatch, fault):
    base = tmp_path / "base"; pins = fixture_files(gate, base)
    if fault == "partial": pins["prediction_files"].pop(gate.NAMES[-1])
    elif fault == "extra": pins["prediction_files"]["extra.npz"] = pins["prediction_files"][gate.NAMES[0]]
    elif fault == "report": pins["inference_report"]["sha256"] = "0" * 64
    elif fault == "hash": pins["prediction_files"][gate.NAMES[-1]]["sha256"] = "0" * 64
    else:
        path = base / gate.INFER_NAMESPACE / "predictions" / gate.NAMES[-1]; arrays = gate.load_arrays(path)
        if fault == "shape": arrays["tracks"] = arrays["tracks"][:, :-1]
        elif fault == "nan": arrays["tracks"][0, 0, 0] = np.nan
        elif fault == "query": arrays["query_points"][0, 0] = 1
        elif fault == "indices": arrays["point_indices"] = arrays["point_indices"][::-1]
        elif fault == "frameorder": arrays["frame_index"] = arrays["frame_index"][::-1]
        elif fault == "scale": arrays["tracks"] *= 2
        elif fault == "static": arrays["static_tracks"] += 1
        else: arrays["visible"][:] = False
        path.unlink(); pins["prediction_files"][gate.NAMES[-1]] = save_npz(gate, path, arrays)
        report_path = base / gate.INFER_NAMESPACE / "report.json"; report = json.loads(report_path.read_bytes())
        report["videos"][-1]["output"] = pins["prediction_files"][gate.NAMES[-1]]
        report_path.unlink(); gate.source.save_bytes(report_path, json.dumps(report).encode(), 0o444)
        pins["inference_report"].update(gate.source.identity(report_path))
    def forbidden(*_, **__): raise AssertionError("Private truth accessed before frozen predictions")
    monkeypatch.setattr(gate.public, "read_private", forbidden)
    with pytest.raises(ValueError): gate.public_predictions(base, pins)


def test_full_public_prediction_record_only_and_visibility_rounding_interval(gate, tmp_path):
    base = tmp_path / "base"; pins = fixture_files(gate, base); rows = gate.public_predictions(base, pins)
    assert len(rows) == 3 and all(r["row"]["frames"] == 4 for r in rows)
    data = rows[0]["prediction"]
    # native product exactly0.5: either saved FP32 decision is accepted, not replaced.
    data["occlusion"][:] = -np.log(np.float32(3)); data["expected_dist"][:] = -np.log(np.float32(2))
    for visible in (True, False):
        data["visible"][:] = visible
        assert gate.validate_prediction(data, rows[0]["queries"], rows[0]["indices"], rows[0]["row"]) == 8


def test_wrapper_cpu_exact_container_cleanup_and_no_broad_private_mount(gate):
    root = Path(__file__).resolve().parents[1]; wrapper = (root / "infra/run_robotap_boots_evaluate.sh").read_text()
    assert "--network none" in wrapper and "--memory 16g" in wrapper and "--cpus 4" in wrapper
    assert "--cap-drop ALL" in wrapper and "--user 1000:1000" in wrapper and "--read-only" in wrapper
    assert "--interactive --cidfile" in wrapper and 'docker rm -f "$CID"' in wrapper and "250s" in wrapper
    assert 'src=$BASE,dst=$BASE' not in wrapper and 'src=$BASE/eval_private,dst=' not in wrapper
    assert "--gpus" not in wrapper and "import torch" not in (root / "infra/robotap_boots_evaluate.py").read_text()


def test_actual_runtime_bundle_retains_private_adapter_source_closure(gate):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("robotap_evaluate_launcher", root / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
    files = {str(path.relative_to(root)): path.read_bytes() for folder in ("infra", "src", "configs")
        for path in (root / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"] = (root / "pyproject.toml").read_bytes()
    selected = set(launcher.runtime_bundle_paths(files, "infra/run_robotap_boots_evaluate.sh"))
    assert {"infra/robotap_boots_evaluate.py", "infra/robotap_boots_public.py", "infra/robotap_boots_acquire.py", "infra/run_robotap_boots_evaluate.sh"} <= selected


def test_private_original_replay_all_three_indices_no_alignment_or_gt_export(gate, tmp_path):
    base = tmp_path / "base"; pins = fixture_files(gate, base); originals = {}
    for i, name in enumerate(gate.public.PICKLES):
        ex = example(); values = {f"z{i}": example(), f"a{i}": ex}
        path = base / name; gate.source.save_bytes(path, pickle.dumps(values, protocol=4)); originals[name] = gate.source.identity(path)
    receipt = dict(schema="world-reward-robotap-opaque-retention-v1", no_unpickle_or_decode=True, retained_files=originals)
    gate.source.save_bytes(base / "eval_private/retention-receipt.json", json.dumps(receipt).encode())
    report = dict(status="pass", stage=gate.source.STAGE, producer_revision="a" * 40, script_sha256="b" * 64,
        source_and_protocol_after_reverified=True, original_pickles_unmodified=True, pickle_or_rgb_or_gt_decoded=False,
        inference_performed=False, evaluation_performed=False, challenge_inputs_used=False, gpu_used=False,
        disposable_archive_removed_after_successful_retention_receipt=True, retained_files=originals,
        filename_selection={"pickle_member_names": [n.split("pickles/", 1)[1] for n in gate.public.PICKLES]})
    gate.source.save_bytes(base / "report.json", json.dumps(report).encode())
    acquisition = dict(schema="world-reward-robotap-boots-acquisition-pins-v1",
        report={**gate.source.identity(base / "report.json"), "producer_revision": "a" * 40, "script_sha256": "b" * 64},
        retention_receipt=gate.source.identity(base / "eval_private/retention-receipt.json"), pickles=originals,
        protocol={"bytes": gate.source.PROTOCOL_BYTES, "sha256": gate.source.PROTOCOL_SHA256})
    records = gate.public_predictions(base, pins)
    rows = gate.evaluate_private(base, records, acquisition, time.monotonic() + 5)
    assert len(rows) == 3 and all(row["boots"]["average_jaccard"] == row["static"]["average_jaccard"] == 1 for row in rows)
    assert gate.decision(rows)["scientific_decision"] == "REJECT"  # perfectstatic is not motion success
    assert all(set(row) == {"video_index", "boots", "static"} for row in rows)
    assert gate.public.acquire_binding(base, acquisition)
    records[0]["queries"][0, 2] += 1
    with pytest.raises(ValueError, match="replay"): gate.evaluate_private(base, records, acquisition, time.monotonic() + 5)
