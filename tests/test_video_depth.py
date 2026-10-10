"""Tiny fake-native contracts; no local model/media download or GPU inference."""
import json
from pathlib import Path
import types

import numpy as np
import pytest

from world_reward import video_depth as adapter


class FakeNetwork:
    metric = True
    encoder = "vits"

    def __init__(self, output=None, returned_fps=None):
        self.output, self.returned_fps = output, returned_fps
        self.calls = []

    def infer_video_depth(self, frames, fps, **kwargs):
        self.calls.append((frames, fps, kwargs))
        output = np.ones(frames.shape[:3], np.float32) if self.output is None else self.output
        return output, fps if self.returned_fps is None else self.returned_fps


def frames():
    return np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)


def test_original_rgb_timeline_native_parameters_and_no_mutation():
    rgb = frames(); before = rgb.copy(); network = FakeNetwork()
    depth = adapter.infer_metric_video(network, rgb, 29.97)
    assert depth.shape == rgb.shape[:3] and depth.dtype == np.float32
    seen, fps, kwargs = network.calls[0]
    assert np.shares_memory(seen, rgb) and not seen.flags.writeable
    assert fps == 29.97 and kwargs == dict(input_size=518, device="cuda", fp32=False)
    assert np.array_equal(rgb, before) and rgb.flags.writeable


def test_boundary_length_does_not_drop_short_video():
    rgb = frames()[:1]; network = FakeNetwork()
    assert adapter.infer_metric_video(network, rgb, 30, fp32=True).shape[0] == 1
    assert network.calls[0][2]["fp32"] is True


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "30"])
def test_bad_fps_fails_before_api(fps):
    network = FakeNetwork()
    with pytest.raises(ValueError): adapter.infer_metric_video(network, frames(), fps)
    assert not network.calls


@pytest.mark.parametrize("rgb", [np.empty((0, 3, 4, 3), np.uint8), np.zeros((3, 4, 3), np.uint8),
    np.zeros((3, 4, 5, 4), np.uint8), np.zeros((3, 4, 5, 3), np.float32)])
def test_bad_rgb_fails_before_api(rgb):
    network = FakeNetwork()
    with pytest.raises(ValueError): adapter.infer_metric_video(network, rgb, 30)
    assert not network.calls


@pytest.mark.parametrize("field,value", [("metric", False), ("metric", None), ("encoder", "vitl")])
def test_relative_or_wrong_model_cannot_sneak_into_metric_path(field, value):
    network = FakeNetwork(); setattr(network, field, value)
    with pytest.raises(ValueError): adapter.infer_metric_video(network, frames(), 30)
    assert not network.calls


@pytest.mark.parametrize("output", [np.ones((2, 4, 5), np.float32), np.ones((3, 4, 5), np.float64),
    np.full((3, 4, 5), np.nan, np.float32), np.full((3, 4, 5), -1, np.float32), np.zeros((3, 4, 5), np.float32)])
def test_incomplete_invalid_or_cast_depth_rejected(output):
    with pytest.raises(ValueError): adapter.infer_metric_video(FakeNetwork(output), frames(), 30)


def test_unsupported_zero_remains_zero_not_invented_evidence():
    output = np.ones((3, 4, 5), np.float32); output[1, 2, 3] = 0
    result = adapter.infer_metric_video(FakeNetwork(output), frames(), 30)
    assert result is output and result[1, 2, 3] == 0


def test_changed_fps_is_not_silent_resampling():
    with pytest.raises(ValueError): adapter.infer_metric_video(FakeNetwork(returned_fps=15), frames(), 30)


def test_offline_native_loader_strict_complete_state(tmp_path, monkeypatch):
    source = tmp_path / "source"; api = source / "video_depth_anything/video_depth.py"
    api.parent.mkdir(parents=True); api.write_bytes(b"native test source")
    checkpoint = tmp_path / "weights.pth"; checkpoint.write_bytes(b"small test fixture")
    monkeypatch.setattr(adapter, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(adapter, "CHECKPOINT_SHA256", adapter._digest(checkpoint))
    monkeypatch.setattr(adapter, "API_SHA256", adapter._digest(api))
    calls = []; state = {"exact": object()}
    class Network:
        def __init__(self, **kwargs): calls.append(("constructor", kwargs))
        def load_state_dict(self, passed, strict): calls.append(("state", passed is state, strict))
        def to(self, device): calls.append(("device", device)); return self
        def eval(self): return self
    def load(path, **kwargs):
        calls.append(("load", path == checkpoint, kwargs)); return state
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True), load=load)
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    module = types.SimpleNamespace(__file__=str(api), VideoDepthAnything=Network)
    monkeypatch.setattr(adapter.importlib, "import_module", lambda _: module)
    network = adapter.load_metric_small(source, checkpoint)
    assert isinstance(network, Network)
    assert calls == [("constructor", dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384], metric=True)),
                     ("load", True, dict(map_location="cpu", weights_only=True)), ("state", True, True), ("device", "cuda")]


def assets_module():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "infra/video_depth_assets.py"
    spec = importlib.util.spec_from_file_location("vda_acquisition_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_frozen_asset_manifest_is_exact_official_small_not_nc_large():
    assets = assets_module(); config = assets.configuration(Path(__file__).resolve().parents[1] / assets.CONFIG)
    assert config["model_bytes"] == adapter.CHECKPOINT_BYTES
    assert config["model_sha256"] == adapter.CHECKPOINT_SHA256
    assert config["model_license"] == "apache-2.0" and config["challenge_overlap_verified"] is False
    assert len(config["source_files"]) == 20
    api = next(row for row in config["source_files"] if row["file"] == "video_depth_anything/video_depth.py")
    assert api["sha256"] == adapter.API_SHA256


@pytest.mark.parametrize("url", ["http://huggingface.co/foo", "https://evil.test/foo", "https://huggingface.co.evil.test/foo",
    "https://user:secret@huggingface.co/foo", "https://xethub.hf.co.evil.test/foo"])
def test_asset_url_allowlist_rejects_untrusted_sources(url):
    assert not assets_module().approved_url(url)


def test_successful_and_failed_acquisition_are_non_overwriting(tmp_path, monkeypatch):
    assets = assets_module(); config = {"source_revision": "a" * 40, "model_revision": "b" * 40,
        "source_repository": "official/source", "model_repository": "official/model", "model_file": "model.pth",
        "source_files": [], "model_license": "apache-2.0", "model_bytes": 3,
        "model_sha256": __import__("hashlib").sha256(b"abc").hexdigest()}
    monkeypatch.setattr(assets, "configuration", lambda _: config)
    monkeypatch.setattr(assets, "publisher_metadata", lambda _: None)
    monkeypatch.setattr(assets, "download", lambda url, path, size: path.write_bytes(b"abc"))
    assets.acquire(tmp_path, tmp_path, "c" * 40)
    receipt = tmp_path / assets.REPORT; before = receipt.read_bytes()
    assert json.loads(before)["status"] == "pass"
    with pytest.raises(ValueError): assets.acquire(tmp_path, tmp_path, "d" * 40)
    assert receipt.read_bytes() == before


def test_failed_assets_removed_only_receipt_remains(tmp_path, monkeypatch):
    assets = assets_module(); config = {"source_revision": "a" * 40, "model_revision": "b" * 40, "model_license": "apache-2.0"}
    monkeypatch.setattr(assets, "configuration", lambda _: config)
    def failed(_): raise ValueError("https://secret-token.invalid should not leak")
    monkeypatch.setattr(assets, "publisher_metadata", failed)
    with pytest.raises(RuntimeError): assets.acquire(tmp_path, tmp_path, "c" * 40)
    assert not (tmp_path / assets.SOURCE_DIR).exists() and not (tmp_path / assets.WEIGHTS_DIR).exists()
    text = (tmp_path / assets.REPORT).read_text()
    assert "secret-token" not in text and json.loads(text)["failed_partial_assets_removed"] is True
