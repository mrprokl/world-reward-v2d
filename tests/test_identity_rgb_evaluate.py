"""Data-free paired quality tests: no GT alignment, dropped cases or retuning."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("identity_quality_test", infra / "identity_rgb_evaluate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def clips(human=9., interaction=4.9):
    return [{"raw": {"human_pve_cm": 10., "relative_hand_visible_object_vector_cm": 5., "per_hand_relative_vector_cm": [5., 5.]},
             "shared": {"human_pve_cm": human, "relative_hand_visible_object_vector_cm": interaction, "per_hand_relative_vector_cm": [interaction, interaction]}}
            for _ in range(3)]


def test_predeclared_paired_gain_and_no_adoption(gate):
    d = gate.decision(clips())
    assert d["synthetic_identity_hypothesis_supported"] and d["median_human_relative_gain"] == pytest.approx(.1)
    assert d["adoption_authorized"] is d["full_HOI_verified"] is d["real_domain_verified"] is False


@pytest.mark.parametrize("bad", ["smallgain", "humanregress", "interaction", "zero", "missing", "nan", "negative"])
def test_all_three_clips_and_both_axes_required(gate, bad):
    x = clips()
    if bad == "smallgain": x = clips(9.6)
    elif bad == "humanregress": x[0]["shared"]["human_pve_cm"] = 10.6
    elif bad == "interaction": x[2]["shared"]["per_hand_relative_vector_cm"][0] = 5.0002
    elif bad == "zero": x[0]["raw"]["human_pve_cm"] = 0.
    elif bad == "missing": x.pop()
    elif bad == "nan": x[0]["shared"]["human_pve_cm"] = np.nan
    else: x[0]["shared"]["human_pve_cm"] = -1.
    if bad in ("missing", "nan", "negative"):
        with pytest.raises(ValueError): gate.decision(x)
    else: assert not gate.decision(x)["synthetic_identity_hypothesis_supported"]


def test_primary_camera_error_does_not_align_or_rescale(gate, monkeypatch):
    monkeypatch.setattr(gate, "VERTICES", 100)
    truth = np.zeros((100, 3)); truth[:, 2] = 3.
    left = np.zeros(100, bool); left[:50] = True; right = ~left
    result = gate.paired_metrics(truth+[0, 0, .1], truth+[0, 0, .05], truth,
                                 np.array([[0., 0., 3.]]), np.array([[0., 0., 3.]]), left, right)
    assert result["raw"]["human_pve_cm"] == pytest.approx(10.)
    assert result["shared"]["human_pve_cm"] == pytest.approx(5.)
    assert result["raw"]["relative_hand_visible_object_vector_cm"] == pytest.approx(10.)


def test_frozen_artifact_hash_and_no_overwrite(gate, tmp_path):
    p = tmp_path / "arrays.npz"; p.write_bytes(b"tiny")
    with pytest.raises(ValueError): gate.regular_hash(p)
    p.chmod(0o444); digest = gate.regular_hash(p)
    assert gate.regular_hash(p, digest) == digest
    with pytest.raises(ValueError): gate.regular_hash(p, "0"*64)


def test_common_scale_fitted_only_from_baseline_human(gate, monkeypatch):
    monkeypatch.setattr(gate, "FRAMES", 3)
    rows = []
    for _ in range(9):
        human = np.ones((8, 8), bool); human[-1] = False; obj = ~human
        points = np.zeros((8, 8, 3), np.float32); points[..., 2] = 2.
        rows.append({"raw_depth": np.full((8, 8), 2., np.float32), "rendered_depth": np.full((8, 8), 3., np.float32),
                     "silhouette": human, "human_mask": human, "object_mask": obj,
                     "validity": np.ones((8, 8), bool), "raw_points": points})
    with pytest.raises(ValueError): gate.common_object_proxies(rows)  # Object support must not disappear.
    for r in rows:
        r["human_mask"][4:] = False; r["object_mask"][4:] = True
    p, diagnostics = gate.common_object_proxies(rows)
    assert len(p) == 9 and len(diagnostics) == 3 and all(np.allclose(x[:, 2], 3., atol=1e-14, rtol=0) for x in p)


def test_one_hand_regression_cannot_be_cancelled_by_other_hand(gate):
    x = clips()
    x[0]["shared"]["per_hand_relative_vector_cm"] = [5.001, 2.]
    assert not gate.decision(x)["gates"]["interaction_nonregression"]


def test_zero_interaction_error_is_absolute_nonregression(gate):
    x = clips()
    for c in x:
        for mode in ("raw", "shared"):
            c[mode]["relative_hand_visible_object_vector_cm"] = 0.
            c[mode]["per_hand_relative_vector_cm"] = [0., 0.]
    assert gate.decision(x)["synthetic_identity_hypothesis_supported"]


def test_masked_geometry_rejected(gate):
    with pytest.raises(ValueError): gate.geometry(np.ma.array(np.zeros((gate.VERTICES, 3)), mask=False), gate.VERTICES, "masked")


def test_all_five_human_frames_need_scale_support(gate):
    rows = []
    for i in range(15):
        human = np.zeros((8, 8), bool); human[:4] = True
        if i == 4: human[:] = False
        points = np.zeros((8, 8, 3), np.float32); points[..., 2] = 2.
        rows.append({"raw_depth": np.full((8, 8), 2., np.float32), "rendered_depth": np.full((8, 8), 3., np.float32),
                     "silhouette": human, "human_mask": human, "object_mask": ~human,
                     "validity": np.ones((8, 8), bool), "raw_points": points})
    with pytest.raises(ValueError, match="Every original frame"): gate.common_object_proxies(rows)


@pytest.fixture
def predictions(gate, monkeypatch, tmp_path):
    import identity_rgb_infer as inference
    records = [dict(file=f"clip_{c:02d}_frame_{f:03d}.png", clip_index=c, frame_index=f, sha256="a"*64)
               for c in range(3) for f in range(5)]
    inputs = {"public_manifest_sha256": "a"*64}
    monkeypatch.setattr(inference, "public_inputs", lambda root: (records, inputs))
    monkeypatch.setattr(inference, "validate_raw", lambda data, row: None)
    monkeypatch.setattr(inference, "validate_pair", lambda data, row, raw: None)
    monkeypatch.setattr(inference, "validate_producer_bindings", lambda root, report: True)
    out = tmp_path/gate.BASE/"predictions_v1"; (out/"raw").mkdir(parents=True); (out/"paired").mkdir()
    report = dict(stage=inference.STAGE, status="pass", phase="complete", network="none", ground_truth_used=False,
        private_truth_read=False, challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[],
        raw_frozen_before_shared=True, paired_frozen_before_reference=True, body_calls_completed=15,
        MoGe_calls_completed=15, shared_head_calls_completed=15, official_reference_calls=1, frames=15,
        all_cases_retained=True, sources_assets_rechecked=True, conversion_fidelity_verified=True,
        actual_body_inference=True, actual_MoGe_inference=True, actual_shared_native_forward=True,
        image_id="sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7",
        camera_K=inference.CAMERA_K.tolist(), script_sha256=gate.sha256(Path(inference.__file__)),
        helper_source_sha256=inference.helper_identities(), producer_revision="b"*40,
        hand_region_sha256={"left": hashlib.sha256((np.arange(100) < 50).tobytes()).hexdigest(),
                            "right": hashlib.sha256((np.arange(100) >= 50).tobytes()).hexdigest()},
        official_reference_per_frame_mean_mm=[.001]*15, raw_outputs=[], paired_outputs=[])
    for record in records:
        shape = np.zeros(45, np.float32); shape[0] = record["clip_index"]*.1
        scale = np.zeros(28, np.float32)
        raw = dict(shape_params=shape, scale_params=scale)
        pair = dict(shared_shape_params=shape.copy(), shared_scale_params=scale.copy(), shared_model_controls=np.zeros(204, np.float32),
                    human_faces=np.zeros((2, 3), np.int64), hand_mask_left=np.arange(100) < 50,
                    hand_mask_right=np.arange(100) >= 50, camera_K=inference.CAMERA_K.copy())
        for mode, data, key in (("raw", raw, "raw_outputs"), ("paired", pair, "paired_outputs")):
            path = out/mode/(Path(record["file"]).stem+".npz")
            with path.open("xb") as stream: np.savez_compressed(stream, **data)
            path.chmod(0o444)
            report[key].append(dict(file=record["file"], clip_index=record["clip_index"], frame_index=record["frame_index"],
                rgb_sha256=record["sha256"], artifact=mode+"/"+path.name, sha256=gate.sha256(path), bytes=path.stat().st_size))
    rp = out/"report.json"
    def save():
        if rp.exists(): rp.chmod(0o644)
        rp.write_text(json.dumps(report)); rp.chmod(0o444)
    save()
    return tmp_path, out, report, save


def test_public_consumer_checks_every_frozen_artifact_before_gt(gate, predictions):
    root, _, _, _ = predictions
    result = gate.public_predictions(root)
    assert len(result[2]) == len(result[3]) == 15 and len(result[5]) == 31
    assert not (root/gate.BASE/"eval_private").exists()


@pytest.mark.parametrize("fault", ["shape", "scales", "controls", "hands", "K", "missing", "extra", "source", "reference"])
def test_consumer_rejects_cross_frame_or_frozen_receipt_tamper(gate, predictions, fault):
    root, out, report, save = predictions
    if fault == "missing": report["paired_outputs"].pop()
    elif fault == "extra": (out/"raw/unexpected.npz").write_bytes(b"extra")
    elif fault == "source": report["helper_source_sha256"] = {}
    elif fault == "reference": report["official_reference_per_frame_mean_mm"][-1] = 2.01
    else:
        row = report["paired_outputs"][4]; p = out/row["artifact"]
        with np.load(p, allow_pickle=False) as a: data = {k: a[k].copy() for k in a.files}
        key = {"shape": "shared_shape_params", "scales": "shared_scale_params", "controls": "shared_model_controls",
               "hands": "hand_mask_left", "K": "camera_K"}[fault]
        if fault == "controls": data[key][136] = -.0
        elif fault == "hands": data[key][0] = False
        else: data[key].flat[0] += .01
        p.chmod(0o644)
        with p.open("wb") as s: np.savez_compressed(s, **data)
        p.chmod(0o444); row.update(sha256=gate.sha256(p), bytes=p.stat().st_size)
    save()
    with pytest.raises(ValueError): gate.public_predictions(root)


def test_first_private_boundary_not_reached_on_public_failure(gate, monkeypatch, tmp_path):
    def reject(_): raise ValueError("public audit fails")
    monkeypatch.setattr(gate, "public_predictions", reject)
    report = {"private_truth_used_for_evaluation_only": False}
    with pytest.raises(ValueError, match="public audit"): gate.run(tmp_path, report)
    assert not report["private_truth_used_for_evaluation_only"]


def test_failed_private_read_still_records_evaluation_provenance(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "public_predictions", lambda root: ([], {}, [], [], {}, [], "a"*64))
    monkeypatch.setattr(gate, "common_object_proxies", lambda rows: ([], []))
    report = {"private_truth_used_for_evaluation_only": False}
    with pytest.raises(ValueError, match="artifact required"): gate.run(tmp_path, report)
    assert report["private_truth_used_for_evaluation_only"] and report["predictions_frozen_before_private"]
