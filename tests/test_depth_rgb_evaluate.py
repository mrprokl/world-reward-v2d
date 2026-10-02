"""Tiny all-pixel scoring and real public-consumer contracts; no assets/GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("own_depth_rgb_quality", infra / "depth_rgb_evaluate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def metric(error=.2, count=72):
    return dict(visible_pixels=count, invalid_pixels=0, coverage=1., complete=True,
                mean_absolute_relative_z=float(error), mean_absolute_z_cm=float(error)*100)


def frames(raw=.2, candidate=.18):
    return [dict(clip_index=c, frame_index=f, moge={"object": metric(raw)}, da3={"object": metric(candidate)})
            for c in range(3) for f in range(5)]


def grid():
    truth = np.full((9, 8), 2., np.float32)
    return truth.copy(), np.ones_like(truth, bool), truth, np.ones_like(truth, bool)


def test_uncapped_all_pixel_relative_z_and_cm_not_alignment(gate):
    p, valid, z, support = grid(); p[0, 0] = 202.
    snapshots = [v.tobytes() for v in (p, valid, z, support)]
    result = gate.visible_z_metric(p, valid, z, support)
    assert result["complete"] and result["visible_pixels"] == 72
    assert result["mean_absolute_relative_z"] == pytest.approx(100/72)
    assert result["mean_absolute_z_cm"] == pytest.approx(20000/72)
    assert snapshots == [v.tobytes() for v in (p, valid, z, support)]


@pytest.mark.parametrize("bad", ["mask", "nan", "inf", "zero", "negative", "all"])
def test_invalid_visible_pixels_undefined_not_penalized_or_intersected(gate, bad):
    p, valid, z, support = grid()
    if bad == "mask": valid[0, 0] = False
    elif bad == "all": valid[:] = False
    else: p[0, 0] = {"nan": np.nan, "inf": np.inf, "zero": 0., "negative": -1.}[bad]
    result = gate.visible_z_metric(p, valid, z, support)
    assert not result["complete"] and result["invalid_pixels"] == (72 if bad == "all" else 1)
    assert result["coverage"] == (72-result["invalid_pixels"])/72
    assert result["mean_absolute_relative_z"] is result["mean_absolute_z_cm"] is None


def test_unknown_background_not_object_support_and_zero_baseline(gate):
    p, valid, z, support = grid(); support[0, :4] = False
    z[~support] = np.nan; p[~support] = np.nan; valid[~support] = False
    result = gate.visible_z_metric(p, valid, z, support)
    assert result["complete"] and result["visible_pixels"] == 68
    assert result["mean_absolute_relative_z"] == result["mean_absolute_z_cm"] == 0.


@pytest.mark.parametrize("bad", ["count64", "shape", "pred_dtype", "truth_dtype", "mask_dtype", "truth_nan", "truth_zero", "masked"])
def test_wrong_support_or_hidden_arrays_fail_fatally(gate, bad):
    p, valid, z, support = grid()
    if bad == "count64": support[0] = False
    elif bad == "shape": p = p[:-1]
    elif bad == "pred_dtype": p = p.astype(np.float64)
    elif bad == "truth_dtype": z = z.astype(np.float64)
    elif bad == "mask_dtype": valid = valid.astype(np.uint8)
    elif bad == "truth_nan": z[0, 0] = np.nan
    elif bad == "truth_zero": z[0, 0] = 0.
    else: p = np.ma.array(p, mask=False)
    with pytest.raises(ValueError): gate.visible_z_metric(p, valid, z, support)


def test_predeclared_balanced_gain_without_adoption(gate):
    result = gate.decision(frames())
    assert result["synthetic_object_z_hypothesis_supported"]
    assert result["median_object_relative_gain"] == pytest.approx(.1)
    for key in ("adoption_authorized", "human_accuracy_verified", "calibration_accuracy_verified",
                "full_HOI_verified", "rigid_object_mesh_or_contact_verified", "real_domain_verified"):
        assert result[key] is False


def test_frames_equal_weight_not_visible_pixel_weighted(gate):
    values = frames()
    for c in range(3):
        for f in range(5): values[c*5+f]["moge"]["object"] = metric(.1 if f == 0 else .3, 100000 if f == 0 else 65)
    result = gate.decision(values)
    assert result["clip_metrics"][0]["moge_mean_absolute_relative_z"] == pytest.approx(.26)


@pytest.mark.parametrize("bad", ["smallgain", "regression", "perfect", "missing_support"])
def test_no_perfect_missing_or_regressing_comparison_adopted(gate, bad):
    values = frames()
    if bad == "smallgain": values = frames(candidate=.195)
    elif bad == "regression":
        for row in values[:5]: row["da3"]["object"] = metric(.22)
    elif bad == "perfect":
        for row in values[:5]: row["moge"]["object"] = metric(0.)
    else:
        row = values[14]["moge"]["object"]
        row.update(invalid_pixels=1, coverage=71/72, complete=False, mean_absolute_relative_z=None, mean_absolute_z_cm=None)
    result = gate.decision(values)
    assert not result["synthetic_object_z_hypothesis_supported"] and not result["adoption_authorized"]
    if bad == "missing_support": assert not result["gates"]["all_native_object_support_complete"]


@pytest.mark.parametrize("bad", ["missing", "order", "boolindex", "nan", "negative", "count", "invalid", "coverage", "complete", "hidden_metric"])
def test_decision_strict_order_and_coverage_contract(gate, bad):
    rows = frames(); value = rows[0]["da3"]["object"]
    if bad == "missing": rows.pop()
    elif bad == "order": rows[0], rows[1] = rows[1], rows[0]
    elif bad == "boolindex": rows[0]["clip_index"] = False
    elif bad == "nan": value["mean_absolute_relative_z"] = np.nan
    elif bad == "negative": value["mean_absolute_relative_z"] = -1.
    elif bad == "count": value["visible_pixels"] = 64
    elif bad == "invalid": value["invalid_pixels"] = True
    elif bad == "coverage": value["coverage"] = .99
    elif bad == "complete": value["complete"] = 1
    else: value.update(invalid_pixels=1, coverage=71/72, complete=False)
    with pytest.raises(ValueError): gate.decision(rows)


@pytest.fixture
def public(gate, monkeypatch, tmp_path):
    import depth_rgb_infer as inference
    # Only actual grid dimensions are tiny. The actual public reader, native
    # XYZ validator and producer binding checker execute without replacement.
    monkeypatch.setattr(inference, "WIDTH", 8); monkeypatch.setattr(inference, "HEIGHT", 9)
    directory = tmp_path / gate.BASE / "inputs"; directory.mkdir(parents=True)
    hashes = []
    for c in range(3):
        for f in range(5):
            p = directory / f"clip_{c:02d}_frame_{f:03d}.png"
            p.write_bytes(f"tiny_public_fixture_{c}_{f}".encode()); p.chmod(0o444); hashes.append(gate.sha256(p))
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(inference.protocol.public_manifest(hashes))); manifest.chmod(0o444)
    records, inputs = inference.public_inputs(tmp_path)
    # Native file/version inspections alone are mocked; no assets are local.
    assets = {"model_sha256": "a"*64}; receipt = {"sha256": "b"*64}; asset = {"sha256": "c"*64}
    source = {"source_sha256": "d"*64}; dependency = {"sha256": "e"*64}
    monkeypatch.setattr(inference.da3, "pinned_assets", lambda root: (assets, root / "native_source"))
    monkeypatch.setattr(inference, "da3_dependency", lambda root: dependency)
    monkeypatch.setattr(inference.moge_assets, "model_asset", lambda root: (root / "native_model", receipt, asset))
    monkeypatch.setattr(inference, "moge_source", lambda: source)
    reports, folders = {}, {}
    for backend in ("moge", "da3"):
        folder = tmp_path / gate.BASE / (backend+"_predictions_v1"); folder.mkdir()
        report = dict(stage=inference.STAGE, backend=backend, status="pass", phase="complete", network="none",
            private_truth_read=False, challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False,
            oracle_modes=[], camera_K=inference.CAMERA_K.tolist(), camera_fitted=False, scale_fit=False, identity_fit=False,
            alignment_performed=False, human_inference_performed=False, masks_used=False, pointmap_geometry_filled=False,
            actual_network_calls=15, frames=15, all_cases_retained=True, actual_native_inference=True, sources_assets_rechecked=True,
            image_id=inference.IMAGES[backend], producer_revision="f"*40, script_sha256=gate.sha256(Path(inference.__file__)),
            helper_source_sha256=inference.helper_identities(), outputs=[], **inputs)
        if backend == "da3":
            report.update(assets=assets, dependency=dependency, native_arguments=inference.da3.NATIVE_ARGUMENTS,
                checkpoint_states_loaded=406, checkpoint_load_strict=True, native_sky_correction_unchanged=True,
                native_sky_threshold=.3, native_sky_quantile=.99, confidence_validity_exclusion=False,
                metric_depth_scaling_applied_once=True, native_processed_grid=list(inference.PROCESSED_HW))
        else:
            report.update(acquisition_report=receipt, model_asset=asset, native_source=source,
                focal_geometry_source_sha256=inference.moge_camera.GEOMETRY_SHA,
                native_focal_solver_calls=[{k: r[k] for k in ("file", "clip_index", "frame_index")} |
                    dict(original_solver_returned=True, focal_prior_supplied=True, native_nearest64_valid_pixels=72) for r in records])
        for record in records:
            arrays = inference.camera_arrays(np.full((9, 8), 2., np.float32), record)
            p = folder / (Path(record["file"]).stem+".npz")
            with p.open("xb") as stream: np.savez_compressed(stream, **arrays)
            p.chmod(0o444)
            row = {k: record[k] for k in ("file", "clip_index", "frame_index")} | dict(artifact=p.name,
                sha256=gate.sha256(p), bytes=p.stat().st_size, rgb_sha256=record["sha256"])
            if backend == "da3":
                K = inference.PROCESSED_K.astype(np.float32)
                row.update(processed_camera_K=K.tolist(), metric_depth_factor=inference.metric_factor(K))
            report["outputs"].append(row)
        reports[backend], folders[backend] = report, folder
    def save():
        for backend, report in reports.items():
            p = folders[backend]/"report.json"
            if p.exists(): p.chmod(0o644)
            p.write_text(json.dumps(report)); p.chmod(0o444)
    save()
    return tmp_path, inference, reports, folders, save


def test_real_public_consumer_all_30_arrays_and_receipts_before_private(gate, public):
    root, _, _, _, _ = public
    records, inputs, outputs, frozen, reports = gate.public_predictions(root)
    assert len(records) == 15 and len(frozen) == 32 and set(reports) == {"moge", "da3"}
    assert all(len(v) == 15 for v in outputs.values()) and inputs["public_manifest_sha256"]
    assert not (root/gate.BASE/"eval_private").exists()


@pytest.mark.parametrize("bad", ["hash", "bytes", "writable", "ray", "dtype", "frame", "extra_array", "missing", "extra_file",
    "source", "oracle", "calls", "image", "sky", "factor", "processedK", "native_prior", "native_count", "native_order", "RGB"])
def test_public_tamper_never_reaches_private_boundary(gate, public, monkeypatch, bad):
    root, inference, reports, folders, save = public; report = reports["da3"]; row = report["outputs"][14]
    p = folders["da3"] / row["artifact"]
    if bad == "hash": row["sha256"] = "0"*64
    elif bad == "bytes": row["bytes"] += 1
    elif bad == "writable": p.chmod(0o644)
    elif bad == "missing": report["outputs"].pop()
    elif bad == "extra_file": (folders["moge"] / "unexpected.npz").write_bytes(b"extra")
    elif bad == "source": report["helper_source_sha256"] = {}
    elif bad == "oracle": report["oracle_modes"] = ["private_camera"]
    elif bad == "calls": report["actual_network_calls"] = 14
    elif bad == "image": report["image_id"] = "sha256:"+"0"*64
    elif bad == "sky": report["native_sky_threshold"] = .31
    elif bad == "factor": row["metric_depth_factor"] *= 2
    elif bad == "processedK": row["processed_camera_K"][0][0] += 1
    elif bad.startswith("native_"):
        call = reports["moge"]["native_focal_solver_calls"][14]
        if bad == "native_prior": call["focal_prior_supplied"] = False
        elif bad == "native_count": call["native_nearest64_valid_pixels"] = 1
        else: call["frame_index"] = 0
    elif bad == "RGB":
        image = root/gate.BASE/"inputs"/"clip_02_frame_004.png"; image.chmod(0o644); image.write_bytes(b"changed"); image.chmod(0o444)
    else:
        with np.load(p, allow_pickle=False) as file: data = {k: file[k].copy() for k in file.files}
        if bad == "ray": data["points"][0, 0, 0] += .1
        elif bad == "dtype": data["depth"] = data["depth"].astype(np.float64)
        elif bad == "frame": data["frame_index"] = np.array(0, np.int64)
        else: data["private_truth"] = np.array(1.)
        p.chmod(0o644)
        with p.open("wb") as stream: np.savez_compressed(stream, **data)
        p.chmod(0o444); row.update(sha256=gate.sha256(p), bytes=p.stat().st_size)
    save(); reads = []; original = gate.regular_hash
    def audited(path, *args, **kwargs):
        if "eval_private" in Path(path).parts: reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(gate, "regular_hash", audited)
    state = {"private_truth_used_for_evaluation_only": False}
    with pytest.raises(ValueError): gate.run(root, state)
    assert reads == [] and state["private_truth_used_for_evaluation_only"] is False


def test_private_read_failure_records_firewall_after_complete_public_audit(gate, public):
    root, _, _, _, _ = public; state = {"private_truth_used_for_evaluation_only": False}
    with pytest.raises(ValueError, match="immutable regular"): gate.run(root, state)
    assert state["predictions_frozen_before_private"] is state["private_truth_used_for_evaluation_only"] is True


def test_native_invalid_pixel_remains_for_private_coverage_not_dropped(gate, public):
    root, _, reports, folders, save = public; row = reports["moge"]["outputs"][0]; p = folders["moge"]/row["artifact"]
    with np.load(p, allow_pickle=False) as file: data = {k: file[k].copy() for k in file.files}
    data["validity"][0, 0] = False; data["depth"][0, 0] = np.nan; data["points"][0, 0] = np.nan
    p.chmod(0o644)
    with p.open("wb") as stream: np.savez_compressed(stream, **data)
    p.chmod(0o444); row.update(sha256=gate.sha256(p), bytes=p.stat().st_size); save()
    values = gate.public_predictions(root)[2]["moge"][0]
    result = gate.visible_z_metric(values["depth"], values["validity"], np.full((9, 8), 2., np.float32), np.ones((9, 8), bool))
    assert result["invalid_pixels"] == 1 and result["mean_absolute_relative_z"] is None


@pytest.fixture
def truth():
    import rgb_cohort_protocol as protocol
    ids = np.full((768, 1024), -1, np.int64); ids.flat[:65] = 0; ids.flat[65:130] = 36874
    z = np.full(ids.shape, np.nan, np.float32); z[ids >= 0] = 2.
    data = dict(human_vertices_camera_m=np.broadcast_to(np.zeros((1, 3)), (18439, 3)),
        object_vertices_camera_m=np.broadcast_to(np.zeros((1, 3)), (194, 3)),
        human_faces=np.broadcast_to(np.array([[0, 1, 2]], np.int64), (36874, 3)),
        object_faces=np.broadcast_to(np.array([[0, 1, 2]], np.int64), (384, 3)),
        scene_depth_m=z, visible_face_indices=ids,
        camera_K=np.array([[protocol.D88.focals[0], 0., 512.], [0., protocol.D88.focals[0], 384.], [0., 0., 1.]]),
        clip_index=np.array(0, np.int64), frame_index=np.array(0, np.int64))
    return data, dict(clip_index=0, frame_index=0)


def test_private_truth_uses_only_visible_foreground_and_exact_manufactured_camera(gate, truth):
    data, record = truth; human, obj = gate.validate_truth(data, record)
    assert human.sum() == obj.sum() == 65 and not np.any(human & obj)
    assert np.isnan(data["scene_depth_m"][~(human | obj)]).all()


@pytest.mark.parametrize("bad", ["extra", "wrongK", "frame", "shape", "triangle", "ids", "backgroundZ", "visibleNan", "count64"])
def test_private_truth_schema_camera_and_support_tamper_fails(gate, truth, bad):
    data, record = truth
    if bad == "extra": data["oracle"] = np.array(1.)
    elif bad == "wrongK": data["camera_K"][0, 0] += 1.
    elif bad == "frame": data["frame_index"] = np.array(1, np.int64)
    elif bad == "shape": data["object_vertices_camera_m"] = data["object_vertices_camera_m"][:-1]
    elif bad == "triangle":
        data["object_faces"] = data["object_faces"].copy(); data["object_faces"][0] = [0, 0, 2]
    elif bad == "ids": data["visible_face_indices"].flat[0] = 36874+384
    elif bad == "backgroundZ": data["scene_depth_m"].flat[-1] = 2.
    elif bad == "visibleNan": data["scene_depth_m"].flat[0] = np.nan
    else:
        data["visible_face_indices"].flat[65] = -1; data["scene_depth_m"].flat[65] = np.nan
    with pytest.raises(ValueError): gate.validate_truth(data, record)


def test_offline_wrapper_syntax_and_mount_firewalls():
    infra = Path(__file__).resolve().parents[1]/"infra"
    for name in ("prepare", "assets", "infer", "evaluate", "validation"):
        subprocess.run(["bash", "-n", str(infra/f"run_depth_rgb_{name}.sh")], check=True, capture_output=True)
    source = (infra/"run_depth_rgb_infer.sh").read_text()
    assert "--network none" in source and "183s" in source and "--gpus all" in source
    assert "eval_private" not in source and "automatic_masks" not in source and "weights/mhr" not in source
    assert "src=$BASE/inputs,dst=$BASE/inputs,readonly" in source
    evaluate = (infra/"run_depth_rgb_evaluate.sh").read_text()
    assert "--network none" in evaluate and "123s" in evaluate and "--gpus" not in evaluate
    assert "src=$BASE/eval_private,dst=$BASE/eval_private,readonly" in evaluate
    chain = (infra/"run_depth_rgb_validation.sh").read_text()
    assert chain.index("--backend moge") < chain.index("--backend da3") < chain.index("run_depth_rgb_evaluate.sh")
