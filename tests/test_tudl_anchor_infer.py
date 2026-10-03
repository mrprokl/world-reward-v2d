"""Small synthetic NumPy/ABI fixtures only; no native model or dataset calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src")); monkeypatch.syspath_prepend(str(root / "infra"))
    spec = importlib.util.spec_from_file_location("wr_tudl_anchor_infer_test", root / "infra/tudl_anchor_infer.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def baseline(gate, factor=2., invalid=False):
    yy, xx = np.indices((gate.HEIGHT, gate.WIDTH), dtype=np.float64)
    depth = np.full((gate.HEIGHT, gate.WIDTH), factor, np.float32)
    points = np.stack(((xx + .5 - 320.) / 800. * depth, (yy + .5 - 240.) / 800. * depth, depth), axis=-1).astype(np.float32)
    validity = np.ones(depth.shape, np.bool_)
    if invalid:
        validity[100:110, 100:110] = False; depth[~validity] = np.nan; points[~validity] = np.nan
    return {"depth": depth, "points": points, "validity": validity, "K": gate.K.copy()}


def diagnostics(gate, ratios=(1., 3., 8., 20., 2., 4., 6., 10., .5, 1., 2., 8.)):
    da3 = np.ones((gate.HEIGHT, gate.WIDTH), np.float32); rows = []
    for index, ratio in enumerate(ratios):
        data = baseline(gate, ratio)
        row = gate.border_ratio(data["depth"], data["validity"], da3)
        row.update(scene_id=index // 4 + 1, frame_id=gate.inputs.FRAME_IDS[index // 4][index % 4]); rows.append(row)
    return rows


def test_exact_border_band_no_semantic_claim_and_ratio_no_centre_leak(gate):
    border = gate.border_mask()
    assert border.dtype == np.bool_ and border.shape == (480, 640) and border.sum() == 110592
    assert border[48, 63] and not border[48, 64] and not border[431, 575] and border[432, 575]
    data = baseline(gate, 4.)
    da3 = np.full(data["depth"].shape, 2., np.float32); da3[~border] = 1e-5
    data["depth"][~border] = 9000.
    result = gate.border_ratio(data["depth"], data["validity"], da3)
    assert result == {"border_pixels": 110592, "paired_valid_pixels": 110592, "paired_border_coverage": 1.,
        "minimum_paired_pixels": 1024, "minimum_border_coverage": .95, "ratio_median": 2.}


@pytest.mark.parametrize("fraction", [.949, .951])
def test_predeclared_border95percent_gate_no_fallback(gate, fraction):
    data = baseline(gate); ids = np.flatnonzero(gate.border_mask()); data["validity"].reshape(-1)[ids[int(len(ids) * fraction):]] = False
    da3 = np.ones(data["depth"].shape, np.float32)
    if fraction < .95:
        with pytest.raises(ValueError, match="support"): gate.border_ratio(data["depth"], data["validity"], da3)
    else:
        result = gate.border_ratio(data["depth"], data["validity"], da3)
        assert result["paired_border_coverage"] > .95 and result["ratio_median"] == 2.


@pytest.mark.parametrize("fault", ["moge_validnan", "moge_zero", "da3_nan", "da3_negative", "da3_dtype", "masked", "valid_dtype", "shape"])
def test_depth_and_validity_invalid_native_outputs_fail_without_drop(gate, fault):
    data = baseline(gate); da3 = np.ones(data["depth"].shape, np.float32)
    if fault == "moge_validnan": data["depth"][0, 0] = np.nan
    elif fault == "moge_zero": data["depth"][0, 0] = 0.
    elif fault == "da3_nan": da3[200, 200] = np.nan
    elif fault == "da3_negative": da3[200, 200] = -1.
    elif fault == "da3_dtype": da3 = da3.astype(np.float64)
    elif fault == "masked": da3 = np.ma.array(da3)
    elif fault == "valid_dtype": data["validity"] = data["validity"].astype(np.uint8)
    else: da3 = da3[:-1]
    with pytest.raises(ValueError): gate.border_ratio(data["depth"], data["validity"], da3)


def test_scene_median_of_four_image_medians_not_global_or_per_frame(gate):
    rows = diagnostics(gate)
    assert gate.scene_anchors(rows) == [5.5, 5., 1.5]
    # Unequal support cannot silently weight scene aggregation by pixel count.
    rows[0]["paired_valid_pixels"] -= 1000
    rows[0]["paired_border_coverage"] = rows[0]["paired_valid_pixels"] / rows[0]["border_pixels"]
    assert gate.scene_anchors(rows) == [5.5, 5., 1.5]


@pytest.mark.parametrize("fault", ["truncated", "order", "scene_bool", "development", "ratio_int", "ratio_nan", "ratio_zero", "coverage", "support", "threshold"])
def test_scene_aggregation_needs_all12_exact_fixed_id_supports(gate, fault):
    rows = diagnostics(gate)
    if fault == "truncated": rows.pop()
    elif fault == "order": rows.reverse()
    elif fault == "scene_bool": rows[0]["scene_id"] = True
    elif fault == "development": rows[0]["frame_id"] = 0
    elif fault == "ratio_int": rows[0]["ratio_median"] = 1
    elif fault == "ratio_nan": rows[0]["ratio_median"] = float("nan")
    elif fault == "ratio_zero": rows[0]["ratio_median"] = 0.
    elif fault == "coverage": rows[0]["paired_border_coverage"] = .96
    elif fault == "support": rows[0]["paired_valid_pixels"] = 100
    else: rows[0]["minimum_border_coverage"] = .90
    with pytest.raises(ValueError): gate.scene_anchors(rows)


def test_candidate_baseline_bytes_nativeinvalids_and_validity_exact_preserved(gate):
    original = baseline(gate, invalid=True); before = gate.array_identities(original)
    raw = np.full((480, 640), 1.5, np.float32); raw_before = raw.tobytes()
    result = gate.candidate_arrays(original, raw, 2., 1, 1788)
    assert gate.array_identities(original) == before and raw.tobytes() == raw_before
    assert np.array_equal(result["anchored_validity"], original["validity"])
    assert result["moge_depth"].tobytes() == original["depth"].tobytes()
    assert result["moge_points"].tobytes() == original["points"].tobytes()
    assert np.all(result["anchored_depth"] == 3.) and np.isfinite(result["anchored_points"]).all()
    assert np.isnan(result["moge_depth"][~result["moge_validity"]]).all()
    assert result["scene_anchor"].shape == () and result["scene_anchor"].dtype == np.float64
    checks = gate.validate_prediction_arrays(result, 1, 1788, 2.)
    assert checks["anchored"]["valid_pixels"] == 480 * 640
    assert checks["moge"]["excluded_pixels"] == 100


@pytest.mark.parametrize("fault", ["drop", "expand", "offset", "perframe_scalar", "wrong_ray", "raw_nan", "baseline_ray", "baseline_scale", "extra", "id", "int_coefficient", "scalar_dtype"])
def test_candidate_exact_multiplier_rays_ids_and_no_score_gaming(gate, fault):
    result = gate.candidate_arrays(baseline(gate, invalid=True), np.ones((480, 640), np.float32), 2., 1, 1788)
    if fault == "drop": result["anchored_validity"][0, 0] = False
    elif fault == "expand": result["anchored_validity"][100, 100] = True
    elif fault == "offset": result["anchored_depth"] += .1
    elif fault == "perframe_scalar": result["scene_anchor"] = np.array(3., np.float64)
    elif fault == "wrong_ray": result["anchored_points"][0, 0, 0] += .1
    elif fault == "raw_nan": result["da3_depth"][200, 200] = np.nan
    elif fault == "baseline_ray": result["moge_points"][0, 0, 0] += .1
    elif fault == "baseline_scale": result["moge_depth"][0, 0] += .1
    elif fault == "extra": result["cam_K_GT"] = np.eye(3)
    elif fault == "id": result["frame_id"] = np.array(0, np.int64)
    elif fault == "int_coefficient":
        with pytest.raises(ValueError): gate.validate_prediction_arrays(result, 1, 1788, 2)
        return
    else: result["scene_anchor"] = np.array(2., np.float32)
    with pytest.raises(ValueError): gate.validate_prediction_arrays(result, 1, 1788, 2.)


@pytest.mark.parametrize("coefficient", [True, 1, 0., -1., float("nan"), float("inf"), 1e100, 1e-100])
def test_coefficient_invalid_types_or_overflow_underflow_fails(gate, coefficient):
    with pytest.raises(ValueError): gate.candidate_arrays(baseline(gate), np.ones((480, 640), np.float32), coefficient, 1, 1788)


def test_original_nine_native_validators_are_not_monkeypatched(gate):
    assert gate.moge_native.FRAME_IDS == ((0, 4074, 8227), (3, 4013, 7710), (4, 4028, 7969))
    assert gate.da3.FRAME_IDS == gate.moge_native.FRAME_IDS
    assert gate.inputs.FRAME_IDS == ((1788, 3235, 5138, 6925), (1566, 3137, 4894, 6490), (1512, 3227, 4819, 6647))
    assert gate.STAGE == "public_tudl_holdout_border_anchored_depth_predictions"
    assert gate.IMAGE_ID == "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
    assert gate.BUDGET == 900


def test_da3_frame_calls_original_metric_resize_and_forward_not9validator(gate, monkeypatch):
    class Tensor:
        def __init__(self, data, shape=None): self.data = data; self.shape = shape or data.shape; self.dtype = "float32"
        def __getitem__(self, index): return Tensor(self.data[index])
        def numpy(self): return self.data
        def cpu(self): return self
    fake_torch = SimpleNamespace(float32="float32", cuda=SimpleNamespace(synchronize=lambda: None))
    images = Tensor(np.zeros((1, 3, 392, 518), np.float32))
    cameras = Tensor(np.array([gate.da3.PROCESSED_K], np.float32)); seen = []
    def processor(rgb, **kwargs):
        assert np.array_equal(kwargs.pop("intrinsics"), np.array([gate.K]))
        assert kwargs == dict(extrinsics=None, process_res=518, process_res_method="upper_bound_resize", num_workers=1,
            sequential=True, print_progress=False)
        return images, None, cameras
    monkeypatch.setattr(gate.da3, "native_forward", lambda torch, network, actual: seen.append("native") or object())
    def resize(torch, depth, factor):
        seen.append("resize"); assert depth.shape == (1, 392, 518)
        return Tensor(np.full((1, 1, 480, 640), factor, np.float32))
    monkeypatch.setattr(gate.da3, "resize_metric_depth", resize)
    monkeypatch.setattr(gate.da3, "camera_arrays", lambda *_: pytest.fail("Hardcoded nine-frame DA3 validator forbidden"))
    depth, detail = gate.da3_frame(fake_torch, object(), processor,
        lambda _: SimpleNamespace(depth=np.ones((1, 392, 518), np.float32)), np.zeros((480, 640, 3), np.uint8))
    assert seen == ["native", "resize"] and depth.shape == (480, 640)
    assert detail["metric_depth_factor"] == gate.da3.metric_factor(cameras.data[0])


def test_verify_bindings_rechecks_all_original_sources_and_assets(gate, monkeypatch):
    root, code, revision = Path("/remote"), Path("/code"), "a" * 40
    report = {"source_helpers": {"h": "old"}, "input_pins": {"p": "old"}, "public_records": [],
        "MoGe_bindings": {"m": "old"}, "DA3_assets": {"d": "old"}, "DA3_dependency": {"w": "old"}}
    seen = []
    monkeypatch.setattr(gate, "bound_source", lambda *_: seen.append("source") or report["source_helpers"])
    monkeypatch.setattr(gate, "load_public", lambda *_: ([], {"input_pins": report["input_pins"], "public_records": []}))
    monkeypatch.setattr(gate, "moge_bindings", lambda *_: seen.append("moge") or (Path("/model"), report["MoGe_bindings"]))
    monkeypatch.setattr(gate.da3, "pinned_assets", lambda *_: seen.append("da3") or (report["DA3_assets"], Path("/source")))
    monkeypatch.setattr(gate, "da3_dependency", lambda *_: seen.append("wheel") or report["DA3_dependency"])
    gate.verify_bindings(root, code, revision, report)
    assert seen == ["source", "moge", "da3", "wheel"] and report["source_assets_after_reverified"] is True
    monkeypatch.setattr(gate.da3, "pinned_assets", lambda *_: ({"d": "CHANGED"}, Path("/source")))
    with pytest.raises(ValueError, match="DA3"): gate.verify_bindings(root, code, revision, report)


@pytest.mark.parametrize("failure", ["runtime", "postcheck"])
def test_main_freezes_failure_receipt_and_restores_signal_handlers(gate, tmp_path, monkeypatch, failure):
    monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("lo")]) if str(self) == "/sys/class/net" else original_iterdir(self))
    root = tmp_path; code = tmp_path / "code"; output = root / gate.BASE / "anchor_predictions_v1"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(root)); monkeypatch.setenv("WR_CODE", str(code)); monkeypatch.setenv("WR_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("WR_IMAGE_ID", gate.IMAGE_ID)
    monkeypatch.setattr(gate, "bound_source", lambda *_: {gate.SOURCE_FILES[0]: {"sha256": "a" * 64, "bytes": 1}})
    handlers = {gate.signal.SIGALRM: object(), gate.signal.SIGTERM: object()}; before = handlers.copy(); alarms = []
    def signal(number, handler): previous = handlers[number]; handlers[number] = handler; return previous
    monkeypatch.setattr(gate.signal, "signal", signal); monkeypatch.setattr(gate.signal, "alarm", alarms.append)
    def run(root, code, revision, output, report, persist):
        if failure == "runtime": handlers[gate.signal.SIGALRM]()
        report.update(status="pass", phase="complete")
    monkeypatch.setattr(gate, "run", run)
    def post(*_):
        if failure == "postcheck": raise ValueError("source mutated")
    monkeypatch.setattr(gate, "verify_bindings", post)
    with pytest.raises((ValueError, TimeoutError)): gate.main([])
    assert handlers == before and alarms == [900, 0]
    path = output / "report.json"; report = json.loads(path.read_text())
    assert report["status"] == "fail" and path.stat().st_mode & 0o777 == 0o400
    assert report["private_truth_read"] is False and report["border_is_background_proxy_only"] is True
    assert report["semantic_background_exclusion_performed"] is False
    assert report["independent_scenes_or_objects"] is False and report["training_overlap_verified"] is False


def test_public_loader_pins_are_current_readonly_config_not_first_seen(gate, tmp_path, monkeypatch):
    code = tmp_path / "code"; path = code / gate.PIN_FILE; path.parent.mkdir(parents=True)
    path.write_text('{"explicit":"pins"}'); path.chmod(0o444)
    seen = []
    def reader(directory, pins): seen.append((directory, pins)); return [], {"sha256": "b" * 64, "bytes": 1}
    monkeypatch.setattr(gate.inputs, "public_inputs", reader)
    records, report = gate.load_public(tmp_path, code)
    assert records == [] and seen == [(tmp_path / gate.BASE / "inputs", {"explicit": "pins"})]
    assert report["input_pin_identity"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.chmod(0o644)
    with pytest.raises(ValueError): gate.load_public(tmp_path, code)
