"""Tiny acquisition integrity fixtures; never network, Azure or real assets."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def acquisition(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "test_refinement_assets", Path(__file__).parents[1] / "infra/acquire_cari_refinement_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"tiny synthetic pinned asset"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "ASSETS", {"tiny.npz": (len(payload), digest)})
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, count):
            assert count == len(payload) + 1
            return payload
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url, timeout: calls.append((url, timeout)) or Response())
    return SimpleNamespace(module=module, payload=payload, digest=digest, calls=calls)


def test_exact_asset_hash_size_and_frozen_repeat_no_network(acquisition, tmp_path):
    (tmp_path / "results").mkdir()
    first = acquisition.module.acquire(tmp_path)
    assert len(acquisition.calls) == 1
    assert first["challenge_data_acquired"] is False and first["vendor_source_modified"] is False
    target = tmp_path / "weights/cari4d/refinement/tiny.npz"
    assert target.read_bytes() == acquisition.payload
    assert acquisition.module.acquire(tmp_path) == first and len(acquisition.calls) == 1
    assert json.loads((tmp_path / "results/cari-refinement-assets.json").read_text()) == first


@pytest.mark.parametrize("failure", ["changed", "pointer", "symlink", "existing_partial"])
def test_missing_corrupt_pointer_or_interrupted_asset_never_overwritten(acquisition, tmp_path, failure):
    (tmp_path / "results").mkdir()
    directory = tmp_path / "weights/cari4d/refinement"
    directory.mkdir(parents=True)
    target = directory / "tiny.npz"
    if failure == "symlink": target.symlink_to(tmp_path / "absent")
    elif failure == "existing_partial": (directory / "tiny.npz.part").write_bytes(b"interrupted")
    else: target.write_bytes(b"changed" if failure == "changed" else b"version https://git-lfs.github.com/spec/v1")
    with pytest.raises((ValueError, FileExistsError)): acquisition.module.acquire(tmp_path)
    assert not acquisition.calls
    assert not (tmp_path / "results/cari-refinement-assets.json").exists()


def test_wrong_download_is_removed_and_no_pass_receipt(acquisition, monkeypatch, tmp_path):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(acquisition.module, "verify_asset", lambda *args: (_ for _ in ()).throw(ValueError("wrong hash")))
    with pytest.raises(ValueError): acquisition.module.acquire(tmp_path)
    assert not (tmp_path / "weights/cari4d/refinement/tiny.npz.part").exists()
    assert not (tmp_path / "weights/cari4d/refinement/tiny.npz").exists()
    assert not (tmp_path / "results/cari-refinement-assets.json").exists()


def test_frozen_different_receipt_not_replaced(acquisition, tmp_path):
    (tmp_path / "results").mkdir()
    receipt = tmp_path / "results/cari-refinement-assets.json"
    receipt.write_text('{}')
    with pytest.raises(ValueError, match="frozen"): acquisition.module.acquire(tmp_path)
    assert receipt.read_text() == '{}'


def test_real_pins_are_small_exact_native_lfs_assets():
    source = (Path(__file__).parents[1] / "infra/acquire_cari_refinement_assets.py").read_text()
    assert "126142" in source and "47140" in source and "media.githubusercontent.com/media" in source
    assert "a026fe82599609ee82814b4c18eb0335fa70925740dea5e1c1e1c0930d5edcf7" in source
    assert "65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8" in source
    assert "track_2" not in source and "track_3" not in source
