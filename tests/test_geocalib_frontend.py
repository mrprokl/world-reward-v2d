"""Tiny source/strict-state contracts only: no Torch, GPU or network required."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

INFRA = Path(__file__).resolve().parents[1]/"infra"
spec = importlib.util.spec_from_file_location("geocalib_assets", INFRA/"geocalib_assets.py")
assets = importlib.util.module_from_spec(spec); sys.modules[spec.name] = assets; spec.loader.exec_module(assets)
spec = importlib.util.spec_from_file_location("geocalib_frontend_test", INFRA/"geocalib_frontend.py")
front = importlib.util.module_from_spec(spec); spec.loader.exec_module(front)


class Tensor:
    def __init__(self, data, dtype=None):
        self.data = np.asarray(data, dtype=dtype)
        self.shape = self.data.shape; self.dtype = self.data.dtype

    def all(self): return Tensor(self.data.all())
    def item(self): return self.data.item()
    def numel(self): return self.data.size
    def min(self): return Tensor(self.data.min())
    def max(self): return Tensor(self.data.max())
    def mean(self): return Tensor(self.data.mean())
    def abs(self): return Tensor(np.abs(self.data))
    def __sub__(self, other): return Tensor(self.data-other)


TORCH = SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor), isfinite=lambda x: Tensor(np.isfinite(x.data)),
                        float32=np.dtype("float32"), finfo=np.finfo,
                        linalg=SimpleNamespace(vector_norm=lambda x, dim: Tensor(np.linalg.norm(x.data, axis=dim))))


class Model:
    def __init__(self):
        self.values = {"backbone.weight": Tensor([1.,2.], "float32"),
                       "backbone.bn.running_mean": Tensor([0.,0.], "float32"),
                       "backbone.bn.num_batches_tracked": Tensor(0, "int64")}
        self.loaded = None

    def state_dict(self): return self.values
    def named_parameters(self): return [("backbone.weight", self.values["backbone.weight"])]
    def named_buffers(self): return [(k,v) for k,v in self.values.items() if k != "backbone.weight"]
    def load_state_dict(self, state, strict): self.loaded = (state, strict)


def checkpoint(model): return {"model": copy.deepcopy(model.values), "epoch": 1}


def test_strict_inventory_includes_all_buffers_and_does_not_mutate():
    model = Model(); state = checkpoint(model); keys = list(state["model"]); report = {}
    front.strict_checkpoint(model, state, TORCH, report)
    assert model.loaded[1] is True and list(state["model"]) == keys
    assert report["parameter_keys"] == 1 and report["buffer_keys"] == 2
    assert report["state_numel"] == 5 and report["state_all_finite"]
    assert len(report["state_inventory"]) == len(report["checkpoint_state_inventory"]) == 3


def test_zero_overlap_native_second_component_mapping_only():
    model = Model(); state = {"model": {k.split(".")[0]+".model."+k.split(".",1)[1]: v for k,v in model.values.items()}}
    original = list(state["model"]); report = {}
    front.strict_checkpoint(model, state, TORCH, report)
    assert report["state_key_mapping"] == "remove_second_component_once"
    assert list(state["model"]) == original and set(model.loaded[0]) == set(model.values)


def test_partial_overlap_is_not_repaired():
    model = Model(); state = checkpoint(model); value = state["model"].pop("backbone.bn.running_mean")
    state["model"]["backbone.model.bn.running_mean"] = value; report = {}
    with pytest.raises(ValueError, match="inventory"): front.strict_checkpoint(model, state, TORCH, report)
    assert report["state_key_mapping"] == "none" and model.loaded is None


def test_mapping_collisions_rejected():
    model = Model(); state = {"model": {"backbone.a.weight": Tensor([1.,2.]), "backbone.b.weight": Tensor([1.,2.])}}
    with pytest.raises(ValueError, match="colliding"): front.strict_checkpoint(model, state, TORCH, {})


@pytest.mark.parametrize("bad", [None, {}, {"model": {}}, {"model": {1: Tensor(1)}}, {"model": []}])
def test_invalid_checkpoint(bad):
    with pytest.raises(ValueError): front.strict_checkpoint(Model(), bad, TORCH, {})


@pytest.mark.parametrize("change", ["missing_buffer", "extra_optimizer", "shape", "dtype", "nan", "non_tensor"])
def test_bad_state_never_loads_and_retains_schema(change):
    model = Model(); state = checkpoint(model); report = {}
    if change == "missing_buffer": state["model"].pop("backbone.bn.num_batches_tracked")
    if change == "extra_optimizer": state["model"]["optimizer.state"] = Tensor([0.], "float32")
    if change == "shape": state["model"]["backbone.weight"] = Tensor([1.], "float32")
    if change == "dtype": state["model"]["backbone.weight"] = Tensor([1.,2.], "float64")
    if change == "nan": state["model"]["backbone.weight"] = Tensor([np.nan,2.], "float32")
    if change == "non_tensor": state["model"]["backbone.weight"] = [1.,2.]
    with pytest.raises(ValueError): front.strict_checkpoint(model, state, TORCH, report)
    assert model.loaded is None and report["checkpoint_state_inventory"]


def test_procedural_rgb_new_original_grid_no_labels():
    images = front.procedural_rgb()
    assert images.shape == (2,3,320,416) and images.dtype == np.float32 and images.flags.c_contiguous
    assert np.isfinite(images).all() and 0 <= images.min() < images.max() <= 1
    assert np.array_equal(images, front.procedural_rgb()) and not np.array_equal(images[0], images[1])


def fields():
    up = np.zeros((2,2,320,416), np.float32); up[:,1] = -1
    return {"up_field": Tensor(up), "latitude_field": Tensor(np.zeros((2,1,320,416), np.float32)),
            "up_confidence": Tensor(np.full((2,320,416), .5, np.float32)),
            "latitude_confidence": Tensor(np.full((2,320,416), .7, np.float32))}


def test_exact_native_field_shapes_and_scalars():
    report = front.validate_fields(fields(), TORCH)
    assert report["up_unit_max_error"] == 0 and report["up_confidence"]["min"] == .5
    assert report["latitude_field"]["shape"] == [2,1,320,416]


@pytest.mark.parametrize("bad", ["missing", "extra", "shape", "dtype", "nan", "confidence", "latitude", "unit"])
def test_native_output_failure(bad):
    data = fields()
    if bad == "missing": data.pop("up_field")
    if bad == "extra": data["camera"] = Tensor(1)
    if bad == "shape": data["up_confidence"] = Tensor(np.zeros((2,1,320,416), np.float32))
    if bad == "dtype": data["up_confidence"] = Tensor(data["up_confidence"].data, "float64")
    if bad == "nan": data["up_confidence"].data[0,0,0] = np.nan
    if bad == "confidence": data["up_confidence"].data[0,0,0] = 1.1
    if bad == "latitude": data["latitude_field"].data[0,0,0,0] = 2
    if bad == "unit": data["up_field"].data[0,1,0,0] = -.5
    with pytest.raises(ValueError): front.validate_fields(data, TORCH)


def test_seed_resets_python_numpy_cpu_and_cuda():
    calls = []; torch = SimpleNamespace(manual_seed=lambda x: calls.append(("cpu",x)), cuda=SimpleNamespace(manual_seed_all=lambda x: calls.append(("cuda",x))))
    front.seed_all(torch); a = np.random.rand(); front.seed_all(torch)
    assert np.random.rand() == a and calls == [("cpu",0),("cuda",0)]*2


@pytest.mark.parametrize("url", ["http://github.com/a", "https://evil.example/a", "https://user:pass@github.com/a", "https://github.com:444/a"])
def test_download_host_rejects_untrusted_urls(url):
    with pytest.raises(ValueError): assets.validate_https(url)


def test_publisher_weight_digest_is_explicitly_unknown(monkeypatch):
    data = {"published_at": "2024-09-08T13:05:40Z", "assets": [{"name": "geocalib-pinhole.tar", "size": assets.WEIGHT_BYTES,
            "browser_download_url": assets.WEIGHT_URL, "digest": None}]}
    monkeypatch.setattr(assets, "open_public", lambda _: io.BytesIO(json.dumps(data).encode()))
    result = assets.publisher_metadata()
    assert result["publisher_digest_available"] is result["publisher_digest_verified"] is False
    data["assets"][0]["digest"] = "sha256:"+"f"*64
    with pytest.raises(ValueError): assets.publisher_metadata()


def test_bounded_exclusive_download_and_sha(monkeypatch, tmp_path):
    class Response(io.BytesIO):
        headers = {"Content-Length": "3"}
        def geturl(self): return "https://raw.githubusercontent.com/a"
    monkeypatch.setattr(assets, "open_public", lambda _: Response(b"abc"))
    path = tmp_path/"a"; result = assets.download("https://raw.githubusercontent.com/a", path, 3, hashlib.sha256(b"abc").hexdigest())
    assert result["bytes"] == 3 and path.read_bytes() == b"abc" and path.stat().st_mode&0o777 == 0o444
    with pytest.raises(FileExistsError): assets.download("https://raw.githubusercontent.com/a", path, 3)
    with pytest.raises(ValueError): assets.download("https://raw.githubusercontent.com/a", tmp_path/"b", 3, "0"*64)


@pytest.mark.parametrize("content,size", [(b"abcd",3), (b"ab",3)])
def test_download_bytes_bounds(monkeypatch, tmp_path, content, size):
    class Response(io.BytesIO):
        headers = {}
        def geturl(self): return "https://raw.githubusercontent.com/a"
    monkeypatch.setattr(assets, "open_public", lambda _: Response(content))
    with pytest.raises(ValueError): assets.download("https://raw.githubusercontent.com/a", tmp_path/"a", size)


def test_no_native_slice_repair_or_full_module_execution():
    with pytest.raises(ValueError, match="slice"): assets.extract_classes(b"\n"*17+b"class GeoCalib: pass\n")
    source = (INFRA/"geocalib_frontend.py").read_text()
    assert "from geocalib." not in source and "import geocalib\n" not in source
    assert "weights_only=True" in source and "model.load_state_dict(state, strict=True)" in source


def test_symlink_asset_and_parent_fail(tmp_path):
    real = tmp_path/"real"; real.mkdir(); link = tmp_path/"link"; link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError): assets.safe_path(link/"file")


def test_wrapper_firewall_budget_and_no_models_or_data_mounts():
    source = (INFRA/"run_geocalib_frontend_smoke.sh").read_text()
    assert "--network none" in source and "183s docker run" in source and "--memory 32g --cpus 4" in source
    assert "world-reward/cari4d-source:0.1" in source and front.IMAGE_ID in source
    assert "WR_GEOCALIB_REPORT_RESERVED=1" in source and 'src=$OUT,dst=$OUT' in source
    assert "validation" not in source and "geocalib-pinhole-v1-frontend" in source


def asset_fixture(tmp_path, monkeypatch):
    target = tmp_path/assets.ASSETS; target.mkdir(parents=True)
    receipt_path = tmp_path/assets.REPORT; receipt_path.parent.mkdir()
    records = tuple((name,url,1,hashlib.sha256(b"a").hexdigest()) for name,url,_,_ in assets.SOURCE_RECORDS)
    monkeypatch.setattr(assets, "SOURCE_RECORDS", records); monkeypatch.setattr(assets, "WEIGHT_BYTES", 1)
    monkeypatch.setattr(assets, "CLASS_BYTES", 1); monkeypatch.setattr(assets, "CLASS_SHA", hashlib.sha256(b"a").hexdigest())
    monkeypatch.setattr(assets, "CREDITS", b"a"); monkeypatch.setattr(assets, "extract_classes", lambda _: b"a")
    entries = {}
    for name in [r[0] for r in records]+["classes18_89.py", "CREDITS.txt", "geocalib-pinhole.tar"]:
        (target/name).write_bytes(b"a")
        entries[name] = {"bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}
    entries["geocalib-pinhole.tar"]["url"] = assets.WEIGHT_URL
    receipt = {"stage": "pinned_geocalib_frontend_assets_acquisition", "status": "pass", "phase": "complete",
               "source_revision": assets.SOURCE_REV, "segnext_revision": assets.SEGNEXT_REV,
               "source_subset_verified": True, "publisher_digest_available": False, "publisher_digest_verified": False,
               "camera_solver_acquired": False, "challenge_inputs_used": False, "ground_truth_used": False, "files": entries}
    receipt_path.write_text(json.dumps(receipt))
    return target, receipt_path, receipt


def test_asset_receipt_all_files_verified(tmp_path, monkeypatch):
    target, receipt_path, receipt = asset_fixture(tmp_path, monkeypatch)
    result, identity = front.validate_assets(tmp_path)
    assert result == target and identity["receipt_sha256"] == assets.digest(receipt_path)
    assert identity["files"] == receipt["files"]


@pytest.mark.parametrize("bad", ["sha", "weight_size", "source_missing", "extra", "symlink", "wrong_pin", "bool_truth", "solver", "publisher_digest"])
def test_asset_tamper_fail_before_import(tmp_path, monkeypatch, bad):
    target, path, receipt = asset_fixture(tmp_path, monkeypatch)
    if bad == "sha": (target/"modules.py").write_bytes(b"b")
    if bad == "weight_size": (target/"geocalib-pinhole.tar").write_bytes(b"aa")
    if bad == "source_missing": (target/"LICENSE.SegNeXt").unlink()
    if bad == "extra": (target/"lm_optimizer.py").write_bytes(b"a")
    if bad == "symlink":
        (target/"modules.py").unlink(); (target/"modules.py").symlink_to(target/"geocalib.audit.py")
    if bad == "wrong_pin": receipt["source_revision"] = "0"*40
    if bad == "bool_truth": receipt["source_subset_verified"] = 1
    if bad == "solver": receipt["camera_solver_acquired"] = True
    if bad == "publisher_digest": receipt["publisher_digest_verified"] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): front.validate_assets(tmp_path)
