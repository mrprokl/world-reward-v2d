"""Tiny own arrays/byte inventories only; no real RGB, GT, models or jobs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat

import numpy as np
import pytest


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def gate(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "src")); monkeypatch.syspath_prepend(str(repo / "infra"))
    module = module_at("wr_anchor_evaluate_test", repo / "infra/tudl_anchor_evaluate.py")
    # Separate synthetic module instances, never mutate old production FRAME_IDS.
    inp = module_at("wr_anchor_inputs_tiny", repo / "infra/tudl_holdout_inputs.py")
    infer = module_at("wr_anchor_infer_tiny", repo / "infra/tudl_anchor_infer.py")
    scoring = module_at("wr_anchor_score_tiny", repo / "infra/tudl_evaluate.py")
    inp.WIDTH, inp.HEIGHT = 8, 6; infer.WIDTH, infer.HEIGHT = 8, 6; scoring.WIDTH, scoring.HEIGHT = 8, 6
    infer.inputs = inp; infer.K = np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]])
    infer.MIN_BORDER_PAIRS = 8
    infer.border_mask = lambda: np.indices((6, 8))[0] == 0
    module.inputs = inp; module.inference = infer; module.scoring = scoring
    return module


def write(path, data, mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): path.chmod(0o600)
    path.write_bytes(data); path.chmod(mode)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_json(path, data, mode=0o444):
    return write(path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode(), mode)


def native_points(depth, K):
    yy, xx = np.indices(depth.shape)
    return np.stack(((xx + .5 - K[0, 2]) * depth / K[0, 0],
        (yy + .5 - K[1, 2]) * depth / K[1, 1], depth), -1).astype(np.float32)


def tiny_arrays(gate, scene, frame):
    inf = gate.inference; depth = np.full((6, 8), 2.2, np.float32)
    baseline = dict(depth=depth, points=native_points(depth, inf.K), validity=np.ones((6, 8), bool), K=inf.K.copy())
    raw = np.full((6, 8), np.float32(2. / float(depth[0, 0])), np.float32); raw[0] = 1.
    coefficient = float(depth[0, 0])
    return inf.candidate_arrays(baseline, raw, coefficient, scene, frame), coefficient


def asset_metadata(gate, root):
    inf = gate.inference; source = inf.moge_assets; da3 = inf.da3.acquisition
    identity = {"sha256": "a" * 64, "bytes": 1}
    files = {"model/v2.py": {"sha256": source.SOURCE_V2_SHA, "bytes": 4},
        "utils/geometry_torch.py": {"sha256": inf.moge_native.GEOMETRY_SHA, "bytes": 5}}
    moge = dict(acquisition_report=identity.copy(), model_asset={"sha256": source.MODEL_SHA, "bytes": source.MODEL_BYTES},
        model_source=dict(revision=source.SOURCE_REVISION,
            direct_url=dict(url="https://github.com/microsoft/MoGe.git", vcs_info=dict(vcs="git", commit_id=source.SOURCE_REVISION)),
            python_files=len(files), python_source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            v2_source_sha256=source.SOURCE_V2_SHA, full_dependency_license_closure_verified=False),
        python_source_files=files, source_directory="/opt/own-test/moge", focal_geometry_source_sha256=inf.moge_native.GEOMETRY_SHA)
    records = [dict(file=da3.SOURCE + "/" + name,
        url=f"https://raw.githubusercontent.com/{da3.SOURCE_REPO}/{da3.SOURCE_REV}/{name}",
        bytes=size, sha256=digest, git_blob_sha1=blob) for name, size, digest, blob in da3.SOURCE_RECORDS]
    assets = dict(acquisition_report=identity.copy(), source_manifest_sha256="3faa74f9b1b22fe076fd37833f1227da94c27256a06e1e8ebc27ce8b2367ede5",
        source_identity={"files": records}, model_sha256=da3.MODEL_RECORDS[-1][2])
    names = {"addict/__init__.py", "addict/addict.py", *["addict-2.4.0.dist-info/" + name
        for name in ("LICENSE", "METADATA", "WHEEL", "top_level.txt", "RECORD")]}
    dependency = dict(sha256=inf.ADDICT_SHA, bytes=3832, path=str(root / inf.ADDICT_PATH),
        import_origin=str(root / inf.ADDICT_PATH) + "/addict/__init__.py", source_sha256={name: "b" * 64 for name in names})
    return dict(MoGe_bindings=moge, DA3_assets=assets, DA3_dependency=dependency)


@pytest.fixture
def frozen(gate, tmp_path):
    root, code = tmp_path, tmp_path / "code"
    base = root / gate.BASE; public, pred = base / "inputs", base / "anchor_predictions_v1"
    public.mkdir(parents=True); pred.mkdir()
    for name in gate.SOURCE_FILES: write(code / name, ("own test source " + name).encode())
    records, data, outputs, ratios, calls = [], [], [], [], []
    for scene, frame_ids in enumerate(gate.inputs.FRAME_IDS, 1):
        for frame in frame_ids:
            name = f"scene_{scene:06d}_frame_{frame:06d}.png"
            identity = write(public / name, ("own tiny RGB identity " + name).encode())
            records.append(dict(scene_id=scene, frame_id=frame, file=name, sha256=identity["sha256"], width=8, height=6))
            arrays, coefficient = tiny_arrays(gate, scene, frame); data.append(arrays)
            ratio = gate.inference.border_ratio(arrays["moge_depth"], arrays["moge_validity"], arrays["da3_depth"])
            ratio.update(scene_id=scene, frame_id=frame); ratios.append(ratio)
            output = dict(file=Path(name).stem + ".npz", scene_id=scene, frame_id=frame, rgb_sha256=identity["sha256"],
                decoded_RGB_sha256="c" * 64, scene_anchor=coefficient, border_proxy=ratio,
                processed_camera_K=gate.inference.da3.PROCESSED_K.astype(np.float32).tolist(),
                metric_depth_factor=gate.inference.da3.metric_factor(gate.inference.da3.PROCESSED_K.astype(np.float32)))
            outputs.append(output)
            calls.append(dict(file=name, scene_id=scene, frame_id=frame, **{"pass": "fixed_prior"},
                focal_prior_supplied=True, original_solver_returned=True, native_nearest64_valid_pixels=40))
    manifest = dict(schema=gate.inputs.SCHEMA, revision=gate.inputs.REVISION, license=gate.inputs.LICENSE,
        selection=gate.inputs.SELECTION, images=records)
    write_json(public / "manifest.json", manifest)
    input_pins = dict(schema=gate.inputs.PINS_SCHEMA, acquisition_report=gate.inputs.ACQUISITION.copy(),
        public_files={path.name: gate.inputs.identity(path) for path in public.iterdir()})
    write_json(code / gate.inference.PIN_FILE, input_pins)
    receipt = {**gate.prediction_contract(), **asset_metadata(gate, root),
        "producer_revision": "d" * 40, "script_sha256": gate.inputs.identity(code / "infra/tudl_anchor_infer.py")["sha256"],
        "source_helpers": {name: gate.inputs.identity(code / name) for name in gate.inference.SOURCE_FILES},
        "input_pin_identity": gate.inputs.identity(code / gate.inference.PIN_FILE), "input_pins": input_pins,
        "input_manifest": gate.inputs.identity(public / "manifest.json"), "public_records": records,
        "scene_anchor_coefficients": [coefficient] * 3,
        "scene_ratio_medians": [[row["ratio_median"] for row in ratios[start:start + 4]] for start in (0, 4, 8)],
        "ratio_diagnostics": ratios, "native_focal_solver_calls": calls, "outputs": outputs}
    def sync():
        for arrays, row in zip(data, outputs):
            path = pred / row["file"]
            if path.exists(): path.chmod(0o600)
            with path.open("wb") as stream: np.savez_compressed(stream, **arrays)
            path.chmod(0o444); row.update(gate.inputs.identity(path))
            row["arrays"] = gate.inference.array_identities(arrays)
            original = {name: arrays["moge_" + name] for name in ("depth", "points", "validity")}; original["K"] = arrays["K"]
            row["original_MoGe_arrays"] = gate.inference.array_identities(original)
            # Fault tests may deliberately make geometry invalid: preserve old
            # checks so the production validator, not fixture repair, rejects.
            if "pointmap_checks" not in row:
                row["pointmap_checks"] = gate.inference.validate_prediction_arrays(arrays, row["scene_id"], row["frame_id"], coefficient)
        receipt_id = write_json(pred / "report.json", receipt, 0o400)
        pins = dict(schema=gate.PINS_SCHEMA,
            report={**receipt_id, "producer_revision": receipt["producer_revision"], "script_sha256": receipt["script_sha256"]},
            outputs={row["file"]: {k: row[k] for k in ("sha256", "bytes")} for row in outputs},
            coefficients=[coefficient] * 3)
        write_json(code / gate.PREDICTION_PINS, pins)
    sync()
    return root, code, data, receipt, sync


def private_fixture(gate, frozen):
    root, code, _, receipt, _ = frozen; private = root / gate.BASE / "eval_private"
    private.mkdir(mode=0o700)
    fixed = ["source/licenses/" + name for name in
        ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json", "attribution-holdout.json")]
    fixed += ["source/base/tudl/" + name for name in ("camera.json", "dataset_info.md", "test_targets_bop19.json")]
    fixed += ["source/" + folder + "/" + name for folder in ("models", "models_eval")
        for name in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")]
    for name in fixed: write(private / name, b"own private byte identity", 0o400)
    for scene, frame_ids in enumerate(gate.inputs.FRAME_IDS, 1):
        directory = private / f"source/test/{scene:06d}"
        K = gate.inference.K.copy(); K[:2, 2] -= .5
        write_json(directory / "scene_camera.json", {str(frame): dict(cam_K=K.ravel().tolist(), depth_scale=1.) for frame in frame_ids}, 0o400)
        write_json(directory / "scene_gt.json", {str(frame): [{"obj_id": scene}, {"obj_id": scene}] for frame in frame_ids}, 0o400)
        write_json(directory / "scene_gt_info.json", {str(frame): [{}, {}] for frame in frame_ids}, 0o400)
        for frame in frame_ids:
            write(directory / f"depth/{frame:06d}.png", b"own synthetic sensor identity", 0o400)
            for instance in (0, 1): write(directory / f"mask_visib/{frame:06d}_{instance:06d}.png", b"own mask identity", 0o400)
    for path in (private / "source", *(private / "source").rglob("*")):
        if path.is_dir(): path.chmod(0o700)
    retained = [{"file": str(path.relative_to(private)), **gate.inputs.identity(path)}
        for path in sorted((private / "source").rglob("*")) if path.is_file()]
    expected = dict(stage="external_tudl_rgb_only_frame_holdout_acquisition", status="pass", phase="complete",
        producer_revision=gate.inputs.ACQUISITION["producer_revision"], script_sha256=gate.inputs.ACQUISITION["script_sha256"],
        dataset_revision=gate.inputs.REVISION, license=gate.inputs.LICENSE, selection=gate.inputs.SELECTION,
        selection_indices=[40, 80, 120, 160], development_selection_indices=[0, 100, 199],
        same_development_scenes_and_objects=True, independent_scenes_or_objects=False,
        temporal_adjacency_or_acceleration_truth_verified=False, device="cpu", gpu_used=False,
        challenge_inputs_used=False, inference_performed=False, ground_truth_used_for_inference=False,
        private_annotations_exported_as_inference_inputs=False, training_overlap_verified=False,
        challenge_overlap_verified=False, accuracy_verified=False, train_or_full_test_downloaded=False,
        bop19_subset_zip_downloaded=True, selection_before_private_annotation_values=True,
        archive_sources_unchanged=True, disposable_archives_removed=True, retained_outputs_unchanged=True,
        source_helpers_unchanged=True, original_frame_coverage_verified=True, development_frame_ids_disjoint=True,
        images_completed=12, public_manifest_sha256=receipt["input_manifest"]["sha256"],
        selected_records=[{k: row[k] for k in ("scene_id", "frame_id", "file")} | {"rgb_sha256": row["sha256"]}
            for row in receipt["public_records"]], retained_files=retained)
    path = private / "acquisition-report.json"; identity = write_json(path, expected, 0o400)
    gate.inputs.ACQUISITION = {**gate.inputs.ACQUISITION, **identity}
    return private, expected, path


def test_full12_frozen_arrays_are_readonly_and_original_helpers_not_monkeypatched(gate, frozen):
    root, code, _, _, _ = frozen; original_ids = gate.scoring.FRAMES
    arrays, records, _, hashes = gate.public_predictions(root, code)
    assert len(arrays) == len(records) == 12 and gate.scoring.FRAMES == original_ids
    assert all(not value.flags.writeable for data in arrays for value in data.values())
    assert hashes["prediction_assets_verified_from_frozen_receipt_only"]
    for data in arrays:
        mapped = gate.score_arrays(data)
        assert mapped["fixed_points"] is data["moge_points"] and mapped["learned_K"] is data["K"]


@pytest.mark.parametrize("fault", ["private_flag", "partial", "extra", "order", "calls", "source", "model", "input",
    "report_hash", "NPZ_hash", "pin_coefficient", "scene_coefficient", "ratio", "geometry", "nan", "drop", "K", "indices", "arrays_extra"])
def test_public_firewall_all_contract_faults_before_any_private_io(gate, frozen, monkeypatch, fault):
    root, code, data, receipt, sync = frozen
    if fault == "private_flag": receipt["private_truth_read"] = True
    elif fault == "partial": receipt["DA3_returns"] = 11
    elif fault == "extra": write(root / gate.BASE / "anchor_predictions_v1/extra.npz", b"extra")
    elif fault == "order": receipt["outputs"] = list(reversed(receipt["outputs"]))
    elif fault == "calls": receipt["native_focal_solver_calls"][-1]["native_nearest64_valid_pixels"] = 1
    elif fault == "source": receipt["source_helpers"]["infra/tudl_anchor_infer.py"]["sha256"] = "e" * 64
    elif fault == "model": receipt["DA3_assets"]["model_sha256"] = "e" * 64
    elif fault == "input": receipt["input_manifest"]["sha256"] = "e" * 64
    elif fault == "scene_coefficient": data[-1]["scene_anchor"] = np.array(3., np.float64)
    elif fault == "ratio": receipt["ratio_diagnostics"] = receipt["ratio_diagnostics"][:-1]
    elif fault == "geometry": data[-1]["anchored_points"] = data[-1]["anchored_points"][:, ::-1].copy()
    elif fault == "nan": data[-1]["anchored_points"][2, 2, 0] = np.nan
    elif fault == "drop": data[-1]["anchored_validity"][2, 2] = False
    elif fault == "K": data[-1]["K"][0, 0] = 11.
    elif fault == "indices": data[-1]["frame_id"] = np.array(1, np.int64)
    elif fault == "arrays_extra": data[-1]["private_gt"] = np.ones(1)
    sync()
    pin_path = code / gate.PREDICTION_PINS
    if fault in ("pin_coefficient", "report_hash", "NPZ_hash"):
        pins = json.loads(pin_path.read_bytes())
        if fault == "pin_coefficient": pins["coefficients"][0] += .1
        elif fault == "report_hash": pins["report"]["sha256"] = "f" * 64
        else: pins["outputs"][next(iter(pins["outputs"]))]["sha256"] = "f" * 64
        write_json(pin_path, pins)
    original = gate.inputs.identity
    def guard(path):
        assert "eval_private" not in str(path), "Private IO crossed before all12 predictions validated"
        return original(path)
    monkeypatch.setattr(gate.inputs, "identity", guard)
    with pytest.raises((ValueError, KeyError)): gate.run(root, {}, code=code)


def test_real_numeric_flow_all12_all_instances_masks_union_no_alignment(gate, frozen, monkeypatch):
    private, _, _ = private_fixture(gate, frozen); root, code, _, _, _ = frozen
    # Synthetic fixtures use their own byte pins, updated independently here.
    pins = json.loads((code / gate.inference.PIN_FILE).read_bytes()); pins["acquisition_report"] = gate.inputs.ACQUISITION.copy()
    write_json(code / gate.inference.PIN_FILE, pins)
    receipt = frozen[3]; receipt["input_pins"] = pins; receipt["input_pin_identity"] = gate.inputs.identity(code / gate.inference.PIN_FILE)
    frozen[4]()
    calls = []
    def read_png(path, *, depth=False):
        calls.append(path)
        if depth: return np.full((6, 8), 2000., np.float64)
        yy, xx = np.indices((6, 8)); left = path.stem.endswith("_000000")
        return (yy > 0) & ((xx < 4) if left else (xx >= 4))
    monkeypatch.setattr(gate.scoring, "read_png", read_png)
    report = {}; gate.run(root, report, code=code)
    assert report["status"] == "pass" and len(report["frames"]) == 12
    assert report["decision"]["real_object_camera_hypothesis_supported"]
    assert all(row["object_instances_scored"] == 2 and row["sensor_valid_object_pixels"] == 40 for row in report["frames"])
    assert len(calls) == 36 and all(private in p.parents for p in calls)
    assert not report["alignment_performed"] and not report["human_quality_verified"]
    assert not report["decision"]["verified_victory_over_CARI4D"]
    assert report["decision"]["median_paired_scene_relative_gain"] > .99999


def score_rows(gate, candidate=(8., 9., 10.)):
    rows = []
    for scene, ids in enumerate(gate.inputs.FRAME_IDS, 1):
        for frame in ids:
            rows.append(dict(scene_id=scene, frame_id=frame,
                fixed=dict(sensor_visible_camera_chamfer_half_cm=10., sensor_valid_object_coverage=1.),
                learned=dict(sensor_visible_camera_chamfer_half_cm=candidate[scene - 1], sensor_valid_object_coverage=1.),
                exact_candidate_validity_matches_baseline=True))
    return rows


def test_decision_scene_means_then_median_reject_regression_coverage_zero_and_order(gate):
    rows = score_rows(gate); result = gate.decision(rows)
    assert result["median_paired_scene_relative_gain"] == .1 and result["real_object_camera_hypothesis_supported"]
    rows = score_rows(gate, (7., 8., 10.6)); assert not gate.decision(rows)["real_object_camera_hypothesis_supported"]
    rows = score_rows(gate)
    for row in rows[:4]: row["fixed"]["sensor_visible_camera_chamfer_half_cm"] = 0.
    assert gate.decision(rows)["per_scene_relative_gain"] is None
    rows = score_rows(gate)
    for mode in ("fixed", "learned"): rows[-1][mode]["sensor_valid_object_coverage"] = .949
    assert not gate.decision(rows)["coverage_gate_pass"]
    for bad in (rows[:-1], list(reversed(rows))):
        with pytest.raises(ValueError): gate.decision(bad)
    rows = score_rows(gate); rows[-1]["learned"]["sensor_visible_camera_chamfer_half_cm"] = float("nan")
    with pytest.raises(ValueError): gate.decision(rows)


def test_no_camera_or_point_alignment_wrong_focal_is_not_erased(gate):
    arrays, _ = tiny_arrays(gate, 1, gate.inputs.FRAME_IDS[0][0]); K = arrays["K"].copy(); K[:2, 2] -= .5
    wrong = arrays["K"].copy(); wrong[0, 0] = wrong[1, 1] = 5.
    arrays["anchored_points"] = native_points(arrays["anchored_depth"], wrong)
    visible = np.indices((6, 8))[0] > 0
    result = gate.scoring.score_frame(gate.score_arrays(arrays), np.full((6, 8), 2.), K, visible)
    assert result["learned"]["sensor_camera_Z_mae_cm"] < 1e-4
    assert result["learned"]["sensor_visible_camera_chamfer_half_cm"] > 10.


@pytest.mark.parametrize("fault", ["extra", "deleted", "tampered", "GTinfo", "writable", "dir", "camera_keys", "instance_missing"])
def test_private_inventory_fails_strict_without_drop(gate, frozen, fault):
    private, receipt, ap = private_fixture(gate, frozen)
    path = private / "source/test/000001/scene_gt_info.json"
    if fault == "extra": write(private / "source/extra", b"extra", 0o400)
    elif fault == "deleted": (private / receipt["retained_files"][0]["file"]).unlink()
    elif fault == "tampered": write(path, b"tamper", 0o400)
    elif fault == "writable": path.chmod(0o600)
    elif fault == "dir": path.parent.chmod(0o755)
    else:
        if fault == "GTinfo": values = json.loads(path.read_bytes()); values[str(gate.inputs.FRAME_IDS[0][0])] = [{}]
        elif fault == "camera_keys": path = path.parent / "scene_camera.json"; values = json.loads(path.read_bytes()); values["0"] = {}
        else: path = path.parent / "scene_gt.json"; values = json.loads(path.read_bytes()); values[str(gate.inputs.FRAME_IDS[0][0])].pop()
        identity = write_json(path, values, 0o400)
        for row in receipt["retained_files"]:
            if row["file"] == str(path.relative_to(private)): row.update(identity)
        identity = write_json(ap, receipt, 0o400); gate.inputs.ACQUISITION.update(identity)
    records = frozen[3]["public_records"]
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.private_inputs(private, records, frozen[3]["input_manifest"]["sha256"])


def test_exact_private_receipt_hash_rejected_before_any_JSON(gate, frozen, monkeypatch):
    private, _, path = private_fixture(gate, frozen)
    write(path, b"not even JSON", 0o400)
    def forbidden(*_): raise AssertionError("Unpinned private bytes parsed")
    monkeypatch.setattr(gate.inputs, "strict_json", forbidden)
    with pytest.raises(ValueError): gate.private_inputs(private, frozen[3]["public_records"], frozen[3]["input_manifest"]["sha256"])


def test_pins_exact_type_inventory_and_coefficient_not_bool(gate, frozen):
    pins = json.loads((frozen[1] / gate.PREDICTION_PINS).read_bytes())
    for mutate in (lambda p: p["coefficients"].__setitem__(0, True),
            lambda p: p["report"].__setitem__("bytes", True), lambda p: p["outputs"].pop(next(iter(p["outputs"]))),
            lambda p: p.__setitem__("extra", {})):
        bad = copy.deepcopy(pins); mutate(bad)
        with pytest.raises(ValueError): gate.validate_prediction_pins(bad)


def test_source_marker_host0644_identity_is_accepted_but_symlink_not(gate, tmp_path):
    marker = tmp_path / "revision"; data = b"a" * 40 + b"\n"; write(marker, data, 0o644)
    assert stat.S_IMODE(marker.stat().st_mode) == 0o644
    assert gate.marker_identity(marker) == {"bytes": 41, "sha256": hashlib.sha256(data).hexdigest()}
    link = tmp_path / "link"; link.symlink_to(marker)
    with pytest.raises(ValueError): gate.marker_identity(link)


def test_literal12_no_old_run_or_monkeyglobal_and_no_assets_acquisition(gate):
    source = Path(gate.__file__).read_text()
    assert "scoring.run(" not in source and "scoring.FRAMES =" not in source and "inputs.FRAME_IDS =" not in source
    assert "moge_bindings(root)" not in source and "pinned_assets(root)" not in source
    assert "signal.alarm(BUDGET)" in source and "path.chmod(0o400)" in source and gate.BUDGET == 180
