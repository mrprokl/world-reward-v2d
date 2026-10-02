"""Tiny CPU synthetic contracts only: no models, sensor truth, GPU or downloads."""
import importlib.util
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("wr_da3_metric_infer_test", infra/"da3_metric_infer.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def asset_fixture(gate, tmp_path, monkeypatch):
    acquisition = gate.acquisition; files = []; source_records = []; model_records = []
    for base, old_records, target in ((acquisition.SOURCE, acquisition.SOURCE_RECORDS, source_records),
                                     (acquisition.WEIGHTS, acquisition.MODEL_RECORDS, model_records)):
        for name, _, _, blob in old_records:
            path = tmp_path/base/name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Own tiny asset fixture "+name.encode()); size = path.stat().st_size; digest = gate.sha256(path)
            target.append((name, size, digest, blob))
            revision = acquisition.SOURCE_REV if base == acquisition.SOURCE else acquisition.MODEL_REV
            repo = acquisition.SOURCE_REPO if base == acquisition.SOURCE else acquisition.MODEL_REPO
            url = (f"https://raw.githubusercontent.com/{repo}/{revision}/{name}" if base == acquisition.SOURCE else
                   f"https://huggingface.co/{repo}/{'resolve' if blob is None else 'raw'}/{revision}/{name}")
            files.append(dict(file=base+"/"+name, url=url, bytes=size, sha256=digest, git_blob_sha1=blob))
    monkeypatch.setattr(acquisition, "SOURCE_RECORDS", tuple(source_records)); monkeypatch.setattr(acquisition, "MODEL_RECORDS", tuple(model_records))
    entries = [{**r, "file": str(Path(r["file"]).relative_to(acquisition.SOURCE))} for r in files[:len(source_records)]]
    manifest = dict(schema="world-reward-da3-metric-source-v1", source_revision=acquisition.SOURCE_REV,
        python_files=27, python_bytes=192452, source_bytes=225327, files=entries, source_modified=False,
        root_namespace_init_fabricated=False, high_level_api_acquired=False)
    manifest_path = tmp_path/acquisition.SOURCE/"source_manifest.json"; manifest_path.write_text(json.dumps(manifest))
    receipt = dict(stage="pinned_da3_metric_model_and_minimal_source_acquisition", status="pass",
        model_revision=acquisition.MODEL_REV, source_revision=acquisition.SOURCE_REV, license="Apache-2.0",
        challenge_inputs_used=False, inference_performed=False, challenge_overlap_verified=False, source_modified=False,
        files=files, source_manifest_sha256=gate.sha256(manifest_path))
    report_path = tmp_path/acquisition.REPORT; report_path.parent.mkdir(); report_path.write_text(json.dumps(receipt))
    return report_path, receipt, manifest_path, manifest


def test_assets_full_inventory_before_upstream_imports(gate, tmp_path, monkeypatch):
    path, _, _, _ = asset_fixture(gate, tmp_path, monkeypatch); evidence, source = gate.pinned_assets(tmp_path)
    assert evidence["acquisition_report"] == gate.identity(path)
    assert source == tmp_path/gate.acquisition.SOURCE
    assert len(evidence["source_identity"]["files"]) == 32
    assert evidence["model_sha256"] == gate.acquisition.MODEL_RECORDS[-1][2]
    assert not (source/"src/depth_anything_3/__init__.py").exists()


@pytest.mark.parametrize("fault", ["status", "source_revision", "model_revision", "license", "challenge", "overlap", "performed",
    "modified", "boolean_type", "order", "missing_record", "wrong_sha", "wrong_bytes", "url", "file_escape", "asset_bytes", "asset_symlink",
    "report_symlink", "manifest_symlink", "manifest_hash", "manifest_schema", "manifest_api", "manifest_stub", "manifest_record", "extra_source", "extra_model"])
def test_asset_contract_fail_closed(gate, tmp_path, monkeypatch, fault):
    path, receipt, manifest_path, manifest = asset_fixture(gate, tmp_path, monkeypatch)
    if fault in ("status", "source_revision", "model_revision", "license"): receipt[fault] = "wrong"
    elif fault in ("challenge", "overlap", "performed", "modified"):
        receipt[{"challenge": "challenge_inputs_used", "overlap": "challenge_overlap_verified", "performed": "inference_performed", "modified": "source_modified"}[fault]] = True
    elif fault == "boolean_type": receipt["inference_performed"] = 0
    elif fault == "order": receipt["files"][:2] = reversed(receipt["files"][:2])
    elif fault == "missing_record": receipt["files"].pop()
    elif fault == "wrong_sha": receipt["files"][0]["sha256"] = "a"*64
    elif fault == "wrong_bytes": receipt["files"][0]["bytes"] += 1
    elif fault == "url": receipt["files"][0]["url"] += "?secret=value"
    elif fault == "file_escape": receipt["files"][0]["file"] = "../private"
    elif fault == "asset_bytes": (tmp_path/receipt["files"][0]["file"]).write_bytes(b"changed")
    elif fault == "asset_symlink":
        asset = tmp_path/receipt["files"][0]["file"]; original = tmp_path/"outside_asset"; asset.rename(original); asset.symlink_to(original)
    elif fault == "manifest_hash": receipt["source_manifest_sha256"] = "a"*64
    elif fault == "manifest_schema": manifest["schema"] = "wrong"
    elif fault == "manifest_api": manifest["high_level_api_acquired"] = True
    elif fault == "manifest_stub": manifest["root_namespace_init_fabricated"] = True
    elif fault == "manifest_record": manifest["files"][0]["sha256"] = "a"*64
    elif fault == "extra_source": (tmp_path/gate.acquisition.SOURCE/"src/depth_anything_3/api.py").write_text("# Unrecorded code")
    elif fault == "extra_model": (tmp_path/gate.acquisition.WEIGHTS/"extra.bin").write_bytes(b"extra")
    manifest_path.write_text(json.dumps(manifest)); path.write_text(json.dumps(receipt))
    if fault in ("report_symlink", "manifest_symlink"):
        target = path if fault == "report_symlink" else manifest_path; original = tmp_path/"outside"; target.rename(original); target.symlink_to(original)
    with pytest.raises(ValueError): gate.pinned_assets(tmp_path)


def state_fixture(gate, monkeypatch):
    monkeypatch.setattr(gate, "STATE_COUNT", 2)
    expected = dict(weight=np.zeros((2, 3), np.float32), buffer=np.zeros(2, np.float32))
    loaded = {"model."+k: np.full_like(v, 1.25) for k, v in expected.items()}
    return loaded, expected


def test_checkpoint_one_prefix_all_states_no_mutation(gate, monkeypatch):
    loaded, expected = state_fixture(gate, monkeypatch); before = {k: v.copy() for k, v in loaded.items()}
    result = gate.checkpoint_state(loaded, expected)
    assert set(result) == set(expected) and result["weight"] is loaded["model.weight"]
    assert all(np.array_equal(loaded[k], v) for k, v in before.items())
    assert all((v == 0).all() for v in expected.values())


@pytest.mark.parametrize("fault", ["missing", "extra", "no_prefix", "two_prefix", "unknown", "shape", "float64", "int", "nan", "inf", "masked", "expected_count", "expected_dtype"])
def test_checkpoint_never_filter_or_adapt(gate, monkeypatch, fault):
    loaded, expected = state_fixture(gate, monkeypatch)
    if fault == "missing": loaded.pop("model.weight")
    elif fault == "extra": loaded["model.extra"] = np.ones(1, np.float32)
    elif fault in ("no_prefix", "two_prefix", "unknown"):
        loaded[{"no_prefix": "weight", "two_prefix": "model.model.weight", "unknown": "model.unknown"}[fault]] = loaded.pop("model.weight")
    elif fault == "shape": loaded["model.weight"] = loaded["model.weight"].T
    elif fault == "float64": loaded["model.weight"] = loaded["model.weight"].astype(np.float64)
    elif fault == "int": loaded["model.weight"] = loaded["model.weight"].astype(np.int32)
    elif fault in ("nan", "inf"): loaded["model.weight"][0, 0] = float(fault)
    elif fault == "masked": loaded["model.weight"] = np.ma.array(loaded["model.weight"])
    elif fault == "expected_count": expected.pop("buffer")
    else: expected["weight"] = expected["weight"].astype(np.float64)
    with pytest.raises(ValueError): gate.checkpoint_state(loaded, expected)


def test_metric_scale_from_actual_anisotropic_processed_focal_once(gate):
    K = gate.PROCESSED_K.astype(np.float32); before = K.copy()
    factor = gate.metric_factor(K)
    assert factor == (float(K[0, 0])+float(K[1, 1]))/600
    assert factor != 800/300 and factor != 647.5/300
    assert np.array_equal(K, before)
    assert factor == pytest.approx(2.1680555555555556, abs=1e-7)


@pytest.mark.parametrize("fault", ["original_K", "wrong_focal", "wrong_center", "nan", "shape", "int", "masked"])
def test_metric_camera_no_fitting_clipping_or_original_focal(gate, fault):
    K = gate.PROCESSED_K.copy()
    if fault == "original_K": K = gate.FIXED_K
    elif fault == "wrong_focal": K[1, 1] += 1
    elif fault == "wrong_center": K[0, 2] += 1
    elif fault == "nan": K[0, 0] = np.nan
    elif fault == "shape": K = K[0]
    elif fault == "int": K = K.astype(int)
    else: K = np.ma.array(K)
    with pytest.raises(ValueError): gate.metric_factor(K)


def small_camera(gate, monkeypatch):
    monkeypatch.setattr(gate, "WIDTH", 8); monkeypatch.setattr(gate, "HEIGHT", 6)
    monkeypatch.setattr(gate, "FIXED_K", np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]]))
    depth = np.linspace(.3, 2.7, 48, dtype=np.float32).reshape(6, 8)
    return depth, gate.camera_arrays(depth, 1, 0)[0]


def test_original_camera_z_plus_half_pixel_full_valid_no_mutation(gate, monkeypatch):
    depth, arrays = small_camera(gate, monkeypatch); original = depth.copy()
    assert arrays["validity"].all() and arrays["depth"] is depth
    assert arrays["points"][0, 0, 0] == pytest.approx((.5-4)/10*depth[0, 0])
    assert np.array_equal(arrays["points"][..., 2], depth) and np.array_equal(original, depth)
    diagnostics = gate.validate_prediction_arrays(arrays, 1, 0)
    assert diagnostics["valid_pixels"] == 48 and diagnostics["excluded_pixels"] == 0
    assert diagnostics["metric_scale_accuracy_verified"] is False


@pytest.mark.parametrize("fault", ["keys", "scene", "frame", "bool_scene", "scalar_dtype", "scalar_shape", "depth_dtype", "points_dtype", "valid_dtype",
    "drop_confidence", "nan", "zero", "negative", "camera", "ray", "z", "corner_sample", "masked", "depth_shape"])
def test_fixed_full_grid_arrays_fail_closed_no_hidden_alignment(gate, monkeypatch, fault):
    _, arrays = small_camera(gate, monkeypatch)
    if fault == "keys": arrays["private_camera"] = np.eye(3)
    elif fault == "scene": arrays["scene_id"] = np.array(2, np.int64)
    elif fault == "frame": arrays["frame_id"] = np.array(4074, np.int64)
    elif fault == "bool_scene":
        with pytest.raises(ValueError): gate.validate_prediction_arrays(arrays, True, 0)
        return
    elif fault == "scalar_dtype": arrays["frame_id"] = np.array(0, np.int32)
    elif fault == "scalar_shape": arrays["frame_id"] = np.array([0], np.int64)
    elif fault == "depth_dtype": arrays["depth"] = arrays["depth"].astype(np.float64)
    elif fault == "points_dtype": arrays["points"] = arrays["points"].astype(np.float64)
    elif fault == "valid_dtype": arrays["validity"] = arrays["validity"].astype(np.uint8)
    elif fault == "drop_confidence": arrays["validity"][0, 0] = False
    elif fault in ("nan", "zero", "negative"): arrays["depth"][0, 0] = {"nan": np.nan, "zero": 0., "negative": -1.}[fault]
    elif fault == "camera": arrays["K"][0, 0] += 1
    elif fault == "ray": arrays["points"][0, 0, 0] += .1
    elif fault == "z": arrays["points"][0, 0, 2] += .1
    elif fault == "corner_sample": arrays["points"][..., 0] -= arrays["depth"]*.5/10
    elif fault == "masked": arrays["points"] = np.ma.array(arrays["points"])
    else: arrays["depth"] = arrays["depth"][:5]
    with pytest.raises(ValueError): gate.validate_prediction_arrays(arrays, 1, 0)


@pytest.mark.parametrize("value", [np.nan, 0., -1.])
def test_bad_native_depth_abstains_whole_image(gate, monkeypatch, value):
    depth, _ = small_camera(gate, monkeypatch); depth[0, 0] = value
    with pytest.raises(ValueError): gate.camera_arrays(depth, 1, 0)
    with pytest.raises(ValueError): gate.camera_arrays(np.ma.array(np.ones((6, 8), np.float32)), 1, 0)


def test_wrapper_only_public_and_pinned_assets_no_private_parent(gate):
    wrapper = Path(gate.__file__).with_name("run_da3_metric_infer.sh").read_text()
    mounts = [line for line in wrapper.splitlines() if "--mount" in line]
    assert len(mounts) == 6 and sum("readonly" in line for line in mounts) == 5
    assert all("eval_private" not in line and "predictions_v1" not in line and "src=$BASE,dst=" not in line for line in mounts)
    assert "world-reward/da3-metric:0.1" in wrapper and "--network none --memory 32g --cpus 4" in wrapper
    assert 'mkdir "$OUT"; chmod 755 "$OUT"; chown' in wrapper and "chown -R" not in wrapper
    assert "! -e \"$OUT\" && ! -L \"$OUT\"" in wrapper and "--entrypoint python" in wrapper


def test_native_arguments_do_not_feed_prior_to_unavailable_camera_encoder(gate):
    assert gate.NATIVE_ARGUMENTS == dict(extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False,
                                       use_ray_pose=False, ref_view_strategy="saddle_balanced")
    assert gate.BUDGET == 600 and gate.STATE_COUNT == 406
    assert not any(k in gate.NATIVE_ARGUMENTS for k in ("focal", "camera_K", "confidence_threshold"))


class Tensor(np.ndarray):
    def cuda(self): return self


def test_each_native_call_seeded_single_forward_original_kwargs(gate, monkeypatch):
    seeds = []; calls = []; context = []
    monkeypatch.setattr(gate.random, "seed", lambda value: seeds.append(("python", value)))
    monkeypatch.setattr(gate.np.random, "seed", lambda value: seeds.append(("numpy", value)))
    torch = SimpleNamespace(manual_seed=lambda value: seeds.append(("torch", value)),
        cuda=SimpleNamespace(manual_seed_all=lambda value: seeds.append(("cuda", value))),
        no_grad=nullcontext, bfloat16="bf16", autocast=lambda **kw: (context.append(kw) or nullcontext()))
    images = np.zeros((1, 3, 2, 3), np.float32).view(Tensor)
    sentinel = object()
    def net(value, **kwargs):
        calls.append((value.shape, kwargs)); return sentinel
    for _ in range(9): assert gate.native_forward(torch, net, images) is sentinel
    assert seeds == [(name, 0) for _ in range(9) for name in ("python", "numpy", "torch", "cuda")]
    assert calls == [((1, 1, 3, 2, 3), gate.NATIVE_ARGUMENTS)]*9
    assert context == [dict(device_type="cuda", dtype="bf16")]*9


def test_metric_camera_z_scaled_once_before_bilinear_not_reprojection(gate, monkeypatch):
    calls = []; sentinel = object(); depth = np.array([[[1., 2.], [3., 4.]]], np.float32); original = depth.copy()
    def interpolate(value, *, size, mode, align_corners):
        calls.append((value.copy(), size, mode, align_corners)); return sentinel
    torch = SimpleNamespace(from_numpy=lambda value: value.view(Tensor),
        nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)))
    factor = gate.metric_factor(gate.PROCESSED_K.astype(np.float32))
    assert gate.resize_metric_depth(torch, depth, factor) is sentinel
    assert len(calls) == 1 and np.array_equal(calls[0][0], depth[None]*factor)
    assert calls[0][1:] == ((480, 640), "bilinear", False) and np.array_equal(depth, original)
