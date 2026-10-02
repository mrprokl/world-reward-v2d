"""Fresh public producer contracts; no model, pixels, truth, network or GPU."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    infra = Path(__file__).resolve().parents[1]/"infra"; monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec = importlib.util.spec_from_file_location("own_depth_rgb_infer", infra/"depth_rgb_infer.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


@pytest.fixture
def public(module, tmp_path):
    folder = tmp_path/module.BASE/"inputs"; folder.mkdir(parents=True); hashes = []
    for clip in range(3):
        for frame in range(5):
            path = folder/f"clip_{clip:02d}_frame_{frame:03d}.png"; path.write_bytes(f"own_RGB_{clip}_{frame}".encode()); path.chmod(0o444); hashes.append(module.sha256(path))
    manifest = module.protocol.public_manifest(hashes); path = folder/"manifest.json"; path.write_text(json.dumps(manifest)); path.chmod(0o444)
    return tmp_path, folder, manifest


def test_actual_protocol_producer_manifest_consumed_without_private_coefficients(module, public):
    root, folder, _ = public; records, receipt = module.public_inputs(root)
    assert len(records) == 15 and [(r["clip_index"], r["frame_index"]) for r in records] == [(c, f) for c in range(3) for f in range(5)]
    assert receipt["public_manifest_sha256"] == module.sha256(folder/"manifest.json")
    assert set(records[0]) == {"file", "sha256", "width", "height", "path", "clip_index", "frame_index"}
    assert module.CAMERA_K.tolist() == [[1280., 0., 512.], [0., 1280., 384.], [0., 0., 1.]]


@pytest.mark.parametrize("fault", ["private", "K", "schema", "count", "order", "extra", "hash", "writable"])
def test_public_firewall_no_labels_or_masks(module, public, fault):
    root, folder, manifest = public
    if fault == "private": manifest["truth"] = [1.]
    elif fault == "K": manifest["images"][0]["camera_K"] = np.eye(3).tolist()
    elif fault == "schema": manifest["schema"] = "world-reward-identity-rgb-v1"
    elif fault == "count": manifest["images"].pop()
    elif fault == "order": manifest["images"][:2] = reversed(manifest["images"][:2])
    elif fault == "extra": (folder/"mask.png").write_bytes(b"forbidden")
    elif fault == "hash": manifest["images"][0]["sha256"] = "0"*64
    path = folder/"manifest.json"; path.chmod(0o644); path.write_text(json.dumps(manifest)); path.chmod(0o444)
    if fault == "writable": (folder/manifest["images"][0]["file"]).chmod(0o644)
    with pytest.raises(ValueError): module.public_inputs(root)


@pytest.fixture
def arrays(module, monkeypatch):
    monkeypatch.setattr(module, "WIDTH", 8); monkeypatch.setattr(module, "HEIGHT", 6)
    monkeypatch.setattr(module, "CAMERA_K", np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]]))
    record = dict(clip_index=0, frame_index=0); depth = np.linspace(.4, 2., 48, dtype=np.float32).reshape(6, 8)
    return module.camera_arrays(depth, record), record


def test_da3_positive_all_grid_plus_half_pixel_once(module, arrays):
    data, record = arrays; result = module.validate_arrays(data, record)
    assert result["valid_pixels"] == 48 and result["excluded_pixels"] == 0
    assert data["validity"].all() and data["points"][0, 0, 0] == pytest.approx((.5-4)/10*data["depth"][0, 0])
    assert np.array_equal(data["points"][..., 2], data["depth"])


@pytest.mark.parametrize("fault", ["extra", "K", "dtype", "frame", "ray", "Z", "nanvalid", "boolindex"])
def test_prediction_schema_fail_closed(module, arrays, fault):
    data, record = arrays
    if fault == "extra": data["ground_truth_depth"] = data["depth"]
    elif fault == "K": data["camera_K"][0, 0] += 1.
    elif fault == "dtype": data["depth"] = data["depth"].astype(np.float64)
    elif fault == "frame": data["frame_index"] = np.array(1, np.int64)
    elif fault == "ray": data["points"][0, 0, 0] += .1
    elif fault == "Z": data["points"][0, 0, 2] += .1
    elif fault == "nanvalid": data["depth"][0, 0] = np.nan
    else: data["clip_index"] = np.array(False)
    with pytest.raises(ValueError): module.validate_arrays(data, record)


def test_moge_invalid_native_values_preserved_not_excluded_or_filled(module, arrays):
    data, record = arrays; data["validity"][0, 0] = False; data["depth"][0, 0] = np.nan; data["points"][0, 0] = np.nan
    before = data["points"].tobytes(); module.validate_arrays(data, record)
    assert data["points"].tobytes() == before and not data["validity"][0, 0]
    with pytest.raises(ValueError): module.camera_arrays(data["depth"], record)


def test_exact_processed_focal_scaling_not_original_focal(module):
    K = module.PROCESSED_K.astype(np.float32); value = module.metric_factor(K)
    assert value == (float(K[0, 0])+float(K[1, 1]))/600
    assert value == pytest.approx(2.1680555555555556, abs=1e-7) and value != 1280/300
    with pytest.raises(ValueError): module.metric_factor(module.CAMERA_K)


class Tensor(np.ndarray):
    def cuda(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


def test_native_da3_helper_unchanged_single_call_and_exact_metric_resize(module, monkeypatch):
    # Actual old native_forward executes; only CPU/Tensor facade replaces CUDA.
    seeds = []; calls = []; resize = []
    monkeypatch.setattr(module.da3.random, "seed", lambda v: seeds.append(("python", v)))
    monkeypatch.setattr(module.da3.np.random, "seed", lambda v: seeds.append(("numpy", v)))
    depth = np.ones((1, *module.PROCESSED_HW), np.float32)*2
    def interpolate(value, *, size, mode, align_corners):
        resize.append((np.asarray(value).copy(), size, mode, align_corners))
        return np.full((1, 1, module.HEIGHT, module.WIDTH), float(value[0, 0, 0, 0]), np.float32).view(Tensor)
    torch = SimpleNamespace(float32=np.float32, bfloat16="bf16", no_grad=nullcontext, autocast=lambda **_: nullcontext(),
        manual_seed=lambda v: seeds.append(("torch", v)), cuda=SimpleNamespace(manual_seed_all=lambda v: seeds.append(("cuda", v))),
        from_numpy=lambda v: v.view(Tensor), nn=SimpleNamespace(functional=SimpleNamespace(interpolate=interpolate)))
    def processor(images, **kwargs):
        assert set(kwargs) == {"extrinsics", "intrinsics", "process_res", "process_res_method", "num_workers", "sequential", "print_progress"}
        assert kwargs["extrinsics"] is None and kwargs["process_res_method"] == "upper_bound_resize"
        assert kwargs["num_workers"] == 1 and kwargs["sequential"] is True and kwargs["print_progress"] is False
        assert np.array_equal(kwargs["intrinsics"], module.CAMERA_K[None]) and kwargs["process_res"] == 518
        return np.zeros((1, 3, *module.PROCESSED_HW), np.float32).view(Tensor), None, module.PROCESSED_K.astype(np.float32)[None].view(Tensor)
    def network(images, **kwargs):
        calls.append((images.shape, kwargs)); return object()
    result, details = module.da3_frame(torch, network, processor, lambda _: SimpleNamespace(depth=depth), np.zeros((768, 1024, 3), np.uint8), dict(clip_index=0, frame_index=0))
    assert calls == [((1, 1, 3, *module.PROCESSED_HW), module.da3.NATIVE_ARGUMENTS)]
    assert seeds == [(name, 0) for name in ("python", "numpy", "torch", "cuda")]
    assert len(resize) == 1 and resize[0][1:] == ((768, 1024), "bilinear", False)
    assert np.array_equal(resize[0][0], depth[None]*details["metric_depth_factor"])
    assert result["validity"].all() and np.array_equal(depth, np.full_like(depth, 2.))


def test_bindings_use_actual_da3_receipt_shape_not_simple_asset_mock(module, tmp_path, monkeypatch):
    # Reuse the existing complete native producer fixture (all32source+3model),
    # exercising old pinned_assets itself, not patching its result contract.
    spec = importlib.util.spec_from_file_location("own_old_da3_test", Path(__file__).with_name("test_da3_metric_infer.py"))
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    tests.asset_fixture(module.da3, tmp_path, monkeypatch)
    assets, _ = module.da3.pinned_assets(tmp_path)
    dependency = {"path": str(tmp_path/module.ADDICT_PATH), "sha256": module.ADDICT_SHA, "bytes": 3832}
    monkeypatch.setattr(module, "da3_dependency", lambda _: dependency)
    report = dict(backend="da3", image_id=module.IMAGES["da3"], script_sha256=module.sha256(Path(module.__file__)),
        helper_source_sha256=module.helper_identities(), sources_assets_rechecked=True, assets=assets, native_arguments=module.da3.NATIVE_ARGUMENTS, dependency=dependency)
    assert module.validate_bindings(tmp_path, report)
    assert set(assets["acquisition_report"]) == {"sha256", "bytes"}
    report["assets"] = dict(assets, acquisition_report={"path": "invented", **assets["acquisition_report"]})
    with pytest.raises(ValueError): module.validate_bindings(tmp_path, report)


def test_actual_moge_asset_producer_receipt_and_source_contract(module, tmp_path, monkeypatch):
    """Call original model_asset on its real snapshot/Xet/cache contract."""
    assets = module.moge_assets; cache = tmp_path/"weights/cari4d/hf_home/hub"
    blob = cache/"blobs"/assets.XET_SHA[:2]/assets.XET_SHA; blob.parent.mkdir(parents=True); blob.write_bytes(b"own tiny native model fixture")
    digest = module.sha256(blob); monkeypatch.setattr(assets, "MODEL_SHA", digest); monkeypatch.setattr(assets, "MODEL_BYTES", blob.stat().st_size)
    snapshot = cache/f"models--Ruicheng--moge-2-vitl-normal/snapshots/{assets.MODEL_REVISION}/model.pt"
    snapshot.parent.mkdir(parents=True); snapshot.symlink_to(blob)
    receipt = tmp_path/"results/weights-acquisition.json"; receipt.parent.mkdir()
    receipt.write_text(json.dumps({"assets": [{"repo_id": "Ruicheng/moge-2-vitl-normal", "revision": assets.MODEL_REVISION, "cache_dir": str(cache)}]}))
    path, acquired, asset = assets.model_asset(tmp_path)
    assert path == blob and acquired == {"sha256": module.sha256(receipt), "bytes": receipt.stat().st_size}
    assert asset == {"sha256": digest, "bytes": blob.stat().st_size}
    source = {"revision": assets.SOURCE_REVISION, "actual_source": "own native metadata fixture"}
    monkeypatch.setattr(module, "moge_source", lambda: source)
    report = dict(backend="moge", image_id=module.IMAGES["moge"], script_sha256=module.sha256(Path(module.__file__)),
        helper_source_sha256=module.helper_identities(), sources_assets_rechecked=True, acquisition_report=acquired, model_asset=asset, native_source=source)
    assert module.validate_bindings(tmp_path, report)
    report["model_asset"] = dict(asset, path="invented producer field")
    with pytest.raises(ValueError): module.validate_bindings(tmp_path, report)


def test_bindings_unknown_source_fail_before_assets(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module.da3, "pinned_assets", lambda _: pytest.fail("must fail before assets"))
    with pytest.raises(ValueError): module.validate_bindings(tmp_path, {"backend": "da3"})


def test_masked_depth_not_lost_by_asarray(module, arrays):
    data, record = arrays
    with pytest.raises(ValueError): module.camera_arrays(np.ma.array(data["depth"], mask=False), record)


def test_missing_unpinned_dependency_wheel_abstains_before_import(module, tmp_path):
    with pytest.raises(ValueError): module.da3_dependency(tmp_path)
    path = tmp_path/module.ADDICT_PATH; path.parent.mkdir(parents=True); path.write_bytes(b"not the original release")
    with pytest.raises(ValueError): module.da3_dependency(tmp_path)


@pytest.mark.parametrize("args", [[], ["--backend", "body"], ["--backend", "moge", "--focal", "1500"]])
def test_explicit_backend_no_tuning_cli(module, args):
    with pytest.raises(SystemExit): module.main(args)


def test_source_no_old_global_mutation_or_private_read(module):
    source = Path(module.__file__).read_text()
    assert "eval_private" not in source and "human.load_model" not in source and "masks" not in module.ARRAY_KEYS
    assert "da3.WIDTH =" not in source and "moge_camera.WIDTH =" not in source and "protocol.D88" not in source
    assert "da3.native_forward(torch, network, images)" in source and "moge_camera.checked_depth_infer" in source
    assert 'target.chmod(0o444)' in source and 'path.chmod(0o444)' in source
