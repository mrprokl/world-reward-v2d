"""Own tiny NumPy records/pickles only; no network, actual RGB or GT."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"; monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("robotap_public_test", infra / "robotap_boots_public.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def example(q=3, t=4):
    video = np.arange(t * 3 * 5 * 3, dtype=np.uint8).reshape(t, 3, 5, 3)
    points = np.full((q, t, 2), .25, dtype=np.float32)
    occluded = np.zeros((q, t), dtype=np.bool_); occluded[0, 0] = True
    return dict(video=video, points=points, occluded=occluded)


def pins_for(gate, root):
    originals = {}
    for i, name in enumerate(gate.PICKLES):
        values = {f"z{i}": example(), f"a{i}": example()}
        path = root / name; gate.source.save_bytes(path, pickle.dumps(values, protocol=4)); originals[name] = gate.source.identity(path)
    receipt = {"schema": "world-reward-robotap-opaque-retention-v1", "no_unpickle_or_decode": True, "retained_files": originals}
    gate.source.save_bytes(root / "eval_private/retention-receipt.json", json.dumps(receipt).encode())
    report = {"status": "pass", "stage": gate.source.STAGE, "producer_revision": "a" * 40, "script_sha256": "b" * 64,
        "source_and_protocol_after_reverified": True, "original_pickles_unmodified": True, "pickle_or_rgb_or_gt_decoded": False,
        "disposable_archive_removed_after_successful_retention_receipt": True, "retained_files": originals,
        "inference_performed": False, "evaluation_performed": False, "challenge_inputs_used": False, "gpu_used": False,
        "filename_selection": {"pickle_member_names": [n.split("pickles/", 1)[1] for n in gate.PICKLES]}}
    gate.source.save_bytes(root / "report.json", json.dumps(report).encode())
    return {"schema": "world-reward-robotap-boots-acquisition-pins-v1", "report": {**gate.source.identity(root / "report.json"), "producer_revision": "a" * 40, "script_sha256": "b" * 64},
        "retention_receipt": gate.source.identity(root / "eval_private/retention-receipt.json"), "pickles": originals,
        "protocol": {"sha256": gate.source.PROTOCOL_SHA256, "bytes": gate.source.PROTOCOL_BYTES}}


def test_first_key_never_reads_record_values(gate):
    class NoRead(dict):
        def __getitem__(self, key): raise AssertionError("Values accessed before filename selection")
    # Exact dict contract intentionally disallows user-defined mapping classes.
    with pytest.raises(ValueError): gate.first_key(NoRead(a=1))
    class Bomb:
        def __getattribute__(self, _): raise AssertionError("Record accessed before key freeze")
    assert gate.first_key({"z": Bomb(), "a": Bomb()}) == "a"
    for invalid in ({}, {1: example()}, {"é": example()}, {"a\n": example()}):
        with pytest.raises(ValueError): gate.first_key(invalid)


def test_native_query_normalization_order_full_video_and_original_index_cap(gate):
    data = example(q=35); data["occluded"][1] = True
    video, queries, indices, unavailable = gate.initial_queries(data)
    assert video is data["video"] and np.array_equal(video, data["video"])
    assert queries.dtype == np.float64 and indices.dtype == np.int64
    assert queries[0].tolist() == [1., .75, 1.25]  # t,y*H,x*W, NOhalfpx
    assert unavailable == [1] and indices.tolist() == [0, *range(2, 32)]
    assert len(queries) == 31 and video.shape[0] == 4
    data["occluded"][:] = True
    with pytest.raises(ValueError, match="no visible"): gate.initial_queries(data)


@pytest.mark.parametrize("fault", ["video_dtype", "video_shape", "points_dtype", "points_shape", "occ_dtype", "occ_shape", "nanquery", "outsidequery", "extra", "object"])
def test_original_arrays_invalid_no_crop_resize_or_fallback(gate, fault):
    data = example()
    if fault == "video_dtype": data["video"] = data["video"].astype(np.float32)
    elif fault == "video_shape": data["video"] = data["video"][..., :2]
    elif fault == "points_dtype": data["points"] = data["points"].astype(np.float64)
    elif fault == "points_shape": data["points"] = data["points"][:, :-1]
    elif fault == "occ_dtype": data["occluded"] = data["occluded"].astype(np.uint8)
    elif fault == "occ_shape": data["occluded"] = data["occluded"][:, :-1]
    elif fault == "nanquery": data["points"][0, 1, 0] = np.nan
    elif fault == "outsidequery": data["points"][0, 1, 0] = 1.1
    elif fault == "object": data["video"] = data["video"].astype(object)
    else: data["futurelabels"] = np.zeros(1)
    with pytest.raises(ValueError): gate.initial_queries(data)


def test_restricted_unpickler_real_numpy_fixture_and_malicious_global_blocked(gate, tmp_path):
    values = {"own_video": example()}; path = tmp_path / "own.pkl"
    gate.source.save_bytes(path, pickle.dumps(values, protocol=4))
    result = gate.read_private(path, gate.source.identity(path)); assert np.array_equal(result["own_video"]["video"], values["own_video"]["video"])
    bad = tmp_path / "bad.pkl"; gate.source.save_bytes(bad, b"cos\nsystem\n(S'echo must-not-run'\ntR.")
    with pytest.raises(pickle.UnpicklingError, match="Nonallowlisted"): gate.read_private(bad, gate.source.identity(bad))
    tail = tmp_path / "tail.pkl"; gate.source.save_bytes(tail, path.read_bytes() + b"trailing")
    with pytest.raises(ValueError, match="Trailing"): gate.read_private(tail, gate.source.identity(tail))


@pytest.mark.parametrize("fault", ["partial", "extra", "tampered", "report", "receipt", "producer", "boolsize"])
def test_all_five_and_receipt_firewall_before_any_unpickle(gate, tmp_path, monkeypatch, fault):
    root = tmp_path / "original"; root.mkdir(); pins = pins_for(gate, root)
    if fault == "partial": pins["pickles"].pop(gate.PICKLES[-1])
    elif fault == "extra": pins["pickles"]["extra.pkl"] = pins["pickles"][gate.PICKLES[0]]
    elif fault == "tampered":
        path = root / gate.PICKLES[-1]; path.chmod(0o600); path.write_bytes(b"changed"); path.chmod(0o400)
    elif fault == "report": pins["report"]["sha256"] = "0" * 64
    elif fault == "receipt": pins["retention_receipt"]["sha256"] = "0" * 64
    elif fault == "producer": pins["report"]["producer_revision"] = "c" * 40
    else: pins["pickles"][gate.PICKLES[0]]["bytes"] = True
    def forbidden(*_, **__): raise AssertionError("Private unpickle before full provenance")
    monkeypatch.setattr(gate, "read_private", forbidden)
    output = tmp_path / "public"; output.mkdir(); (output / "inputs").mkdir()
    with pytest.raises(ValueError): gate.publish(root, output, pins, time.monotonic() + 5, lambda _: None)
    assert not list((output / "inputs").iterdir())


def test_three_full_videos_initial_queries_only_no_future_payload(gate, tmp_path):
    root = tmp_path / "original"; root.mkdir(); pins = pins_for(gate, root)
    output = tmp_path / "public"; output.mkdir(); (output / "inputs").mkdir(); frozen = []
    manifest = gate.publish(root, output, pins, time.monotonic() + 5, frozen.append)
    assert len(frozen) == 1 and [r["video_key"] for r in frozen[0]] == ["a0", "a1", "a2"]
    assert manifest["initial_query_is_external_oracle"] and not manifest["future_tracks_or_visibility_public"]
    for record in manifest["videos"]:
        with np.load(output / "inputs" / record["file"], allow_pickle=False) as saved:
            assert set(saved.files) == {"video", "query_points", "point_indices"}
            assert saved["video"].shape == (4, 3, 5, 3) and saved["video"].dtype == np.uint8
            assert saved["query_points"][0].tolist() == [1., .75, 1.25]
        assert (output / "inputs" / record["file"]).stat().st_mode & 0o777 == 0o444
    assert gate.acquire_binding(root, pins)


def test_wrapper_sealed_cpu_and_narrow_private_mounts(gate):
    root = Path(__file__).resolve().parents[1]; wrapper = (root / "infra/run_robotap_boots_public.sh").read_text()
    assert "--network none" in wrapper and "--memory 16g" in wrapper and "--cpus 4" in wrapper
    assert "--cap-drop ALL" in wrapper and "--read-only" in wrapper and "--security-opt no-new-privileges" in wrapper
    assert "--entrypoint /usr/bin/env" in wrapper and "-i PATH=/opt/conda/bin" in wrapper
    assert "190s" in wrapper and "--kill-after=10s" in wrapper
    assert 'src=$BASE,dst=$BASE' not in wrapper and 'src=$BASE/eval_private,dst=' not in wrapper
    assert "docker run --rm --interactive --cidfile" in wrapper and "--gpus" not in wrapper
    assert 'CIDFILE="$OUT/.container.cid"' in wrapper and 'docker rm -f "$CID"' in wrapper
    assert 'world-reward.job=run_robotap_boots_public' in wrapper and 'world-reward.revision=$REV' in wrapper
    assert "import torch" not in (root / "infra/robotap_boots_public.py").read_text()


def test_public_helper_full_prebinding_once_no_redundant_posthash(gate, tmp_path, monkeypatch):
    root = tmp_path / "original"; root.mkdir(); pins = pins_for(gate, root)
    output = tmp_path / "public"; output.mkdir(); (output / "inputs").mkdir(); calls = []
    original = gate.acquire_binding
    def counted(root, pins): calls.append(True); return original(root, pins)
    monkeypatch.setattr(gate, "acquire_binding", counted)
    gate.publish(root, output, pins, time.monotonic() + 5, lambda _: None)
    assert calls == [True]  # main.finally, not this helper, performs full post-check.


def test_no_speculative_jpeg_or_unpublished_normalization(gate):
    data = example(); data["video"] = [b"not an actual JPEG"] * 4
    with pytest.raises(ValueError, match="full RGB"): gate.initial_queries(data)
    assert gate.TAP_SOURCE["sha256"] == "90cd01e53e23f6d489d3a6cd840cfd93fed4c1a6f4a0dfd373933cc164f0e570"


def test_actual_public_runtime_bundle_has_all_embedded_python_helpers(gate):
    import azure_job
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): path.read_bytes()
             for folder in ("infra", "configs", "src")
             for path in (root / folder).rglob("*")
             if path.is_file() and path.suffix not in (".pyc",)}
    selected = set(azure_job.runtime_bundle_paths(files, "infra/run_robotap_boots_public.sh"))
    assert set(gate.FILES) <= selected
    assert not any("infer" in path or "evaluate" in path for path in selected if path.startswith("infra/"))


def test_exact_mediapy_protocol4_compatibility_only_and_plain_storage_preserved(gate, tmp_path, monkeypatch):
    import types
    module = types.ModuleType("mediapy")
    # Author-only faithful primary class; no third-party package imported.
    exec("class _VideoArray(np.ndarray):\n def __new__(cls,a,metadata=None):\n  obj=np.asarray(a).view(cls);obj.metadata=metadata;return obj\n def __array_finalize__(self,obj):\n  if obj is not None:self.metadata=getattr(obj,'metadata',None)\n", {"np": np, "__name__": "mediapy"}, module.__dict__)
    monkeypatch.setitem(sys.modules, "mediapy", module)
    original = example(); wrapped = module._VideoArray(original["video"], metadata={"fps": 30})
    original["video"] = wrapped
    raw = pickle.dumps({"own": original}, protocol=4)
    assert b"fps" not in raw  # primary class inherits ndarray pickle, no metadata state override
    path = tmp_path / "own_mediapy.pkl"; gate.source.save_bytes(path, raw)
    decoded = gate.read_private(path, gate.source.identity(path))["own"]
    assert type(decoded["video"]) is gate.MediaPyVideoArray
    before = decoded["video"]
    plain, queries, indices, _ = gate.initial_queries(decoded)
    assert type(plain) is np.ndarray and np.shares_memory(plain, before)
    assert (plain.dtype, plain.shape, plain.strides, plain.tobytes()) == (wrapped.dtype, wrapped.shape, wrapped.strides, wrapped.tobytes())
    assert queries[0].tolist() == [1., .75, 1.25] and indices.tolist() == [0, 1, 2]
    class Unrelated(np.ndarray): pass
    bad = example(); bad["video"] = bad["video"].view(Unrelated)
    with pytest.raises(ValueError): gate.initial_queries(bad)
    for name in ("VideoMetadata", "_OtherArray", "read_video", "decompress_video"):
        with pytest.raises(pickle.UnpicklingError): gate.RestrictedUnpickler(__import__('io').BytesIO()).find_class("mediapy", name)
    assert gate.OUTPUT.endswith("/public_v2")


def test_original_failure_identity_before_json_and_no_relabel(gate, tmp_path, monkeypatch):
    root = tmp_path / "original"; (root / "public_v1").mkdir(parents=True)
    gate.source.save_bytes(root / "public_v1/report.json", b"not JSON")
    def forbidden(*_): raise AssertionError("Original FAIL parsed before exact byte pin")
    monkeypatch.setattr(gate, "strict_json", forbidden)
    with pytest.raises(ValueError, match="pinned"): gate.original_failure(root)


def test_plain_mediapy_strided_bytes_and_chunk_digest_no_full_rgb_copy(gate):
    own = np.arange(12 * 9, dtype=np.uint8).reshape(12, 9)[::2, ::2].view(gate.MediaPyVideoArray)
    plain = gate.plain_video(own)
    assert plain.strides == own.strides and np.shares_memory(plain, own)
    assert gate.array_digest(plain) == hashlib.sha256(own.tobytes(order="C")).hexdigest()
