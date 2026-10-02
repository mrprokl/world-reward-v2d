"""Offline, data-free regressions for the Body smoke's input/source/Hub guards."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def smoke():
    path = Path(__file__).resolve().parents[1] / "infra/body_smoke.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_body_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(relative, payload):
    return {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_input(root, relative, payload=b"tiny-not-media"):
    path = root / "data" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, record(relative, payload)


def test_manifest_file_verifies_single_record_bytes_hash_and_path(smoke, tmp_path):
    path, entry = write_input(tmp_path, "track_1/meta/tiny.json", b"{}")
    actual, digest = smoke._manifest_file(tmp_path, {"files": [entry]}, entry["path"])
    assert actual == path
    assert digest == hashlib.sha256(b"{}").hexdigest()


@pytest.mark.parametrize("count", [0, 2])
def test_manifest_requires_exactly_one_path_record(smoke, tmp_path, count):
    _, entry = write_input(tmp_path, "track_1/meta/tiny.json")
    with pytest.raises(ValueError, match="exactly one"):
        smoke._manifest_file(tmp_path, {"files": [entry] * count}, entry["path"])


@pytest.mark.parametrize("sha", ["", "0" * 63, "A" * 64, "g" * 64, None])
def test_manifest_rejects_invalid_hash(smoke, tmp_path, sha):
    _, entry = write_input(tmp_path, "track_1/meta/tiny.json")
    entry["sha256"] = sha
    with pytest.raises((ValueError, TypeError)):
        smoke._manifest_file(tmp_path, {"files": [entry]}, entry["path"])


@pytest.mark.parametrize("tamper", ["hash", "bytes", "file_same_size", "file_size"])
def test_manifest_rejects_content_or_record_tampering(smoke, tmp_path, tamper):
    path, entry = write_input(tmp_path, "track_1/meta/tiny.json", b"abcd")
    if tamper == "hash":
        entry["sha256"] = "0" * 64
    elif tamper == "bytes":
        entry["bytes"] += 1
    elif tamper == "file_same_size":
        path.write_bytes(b"wxyz")
    else:
        path.write_bytes(b"different-length")
    with pytest.raises(ValueError, match="integrity failed"):
        smoke._manifest_file(tmp_path, {"files": [entry]}, entry["path"])


def test_manifest_path_tamper_does_not_match_another_record(smoke, tmp_path):
    _, entry = write_input(tmp_path, "track_1/meta/tiny.json")
    with pytest.raises(ValueError, match="exactly one"):
        smoke._manifest_file(tmp_path, {"files": [entry]}, "track_1/meta/other.json")


def test_manifest_rejects_leaf_symlink_even_to_file_inside_data(smoke, tmp_path):
    target, _ = write_input(tmp_path, "track_1/meta/target.json")
    relative = "track_1/meta/link.json"
    (tmp_path / "data" / relative).symlink_to(target)
    with pytest.raises(ValueError, match="escapes"):
        smoke._manifest_file(tmp_path, {"files": [record(relative, target.read_bytes())]}, relative)


def test_manifest_rejects_symlinked_parent_escaping_data(smoke, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tiny.json").write_bytes(b"{}")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "linked").symlink_to(outside, target_is_directory=True)
    relative = "linked/tiny.json"
    with pytest.raises(ValueError, match="escapes"):
        smoke._manifest_file(tmp_path, {"files": [record(relative, b"{}")]}, relative)


@pytest.mark.parametrize("relative", ["../outside.json", "../../outside.json"])
def test_manifest_rejects_parent_traversal_before_file_access(smoke, tmp_path, relative):
    (tmp_path / "data").mkdir()
    with pytest.raises(ValueError, match="escapes"):
        smoke._manifest_file(tmp_path, {"files": [record(relative, b"{}")]}, relative)


def test_manifest_rejects_absolute_outside_path(smoke, tmp_path):
    (tmp_path / "data").mkdir()
    relative = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escapes"):
        smoke._manifest_file(tmp_path, {"files": [record(relative, b"{}")]}, relative)


@pytest.fixture
def audited_inputs(smoke, tmp_path):
    # Bytes are intentionally not valid MP4/PNG: the guard must not decode media.
    total = 5
    episodes_relative = "track_1/meta/episodes.jsonl"
    episodes_payload = (json.dumps({"episode_index": smoke.EPISODE, "length": total}) + "\n").encode()
    episodes, episode_entry = write_input(tmp_path, episodes_relative, episodes_payload)
    relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{smoke.EPISODE:06d}.mp4"
    video, video_entry = write_input(tmp_path, relative, b"not-an-mp4")
    manifest = {
        "track": "track_1", "repo_id": "nvidia/video_to_data_challenge",
        "revision": smoke.DATASET_REVISION, "files": [episode_entry, video_entry],
    }
    results = tmp_path / "results"
    results.mkdir()
    manifest_path = results / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    masks = tmp_path / f"outputs/episode_{smoke.EPISODE:06d}/automatic_masks"
    human_masks = masks / "masks/0"
    human_masks.mkdir(parents=True)
    for index in range(total):
        (human_masks / f"{index:06d}.png").write_bytes(b"not-a-png")
    report = {
        "stage": "automatic_masks", "status": "pass", "episode_index": smoke.EPISODE,
        "frames": total, "input_track": "track_1", "input_sha256": video_entry["sha256"],
        "ground_truth_used": False, "hand_labeled_test": False, "oracle_modes": [],
    }
    report_path = masks / "report.json"
    report_path.write_text(json.dumps(report))
    prompts = {"prompts": [
        {"object_id": object_id, "frame_index": 2, "points": None,
         "point_labels": None, "mask_path": None,
         "box": {"x0": 1, "y0": 1, "x1": 3, "y1": 3}}
        for object_id in (0, 1)
    ]}
    prompts_path = masks / "prompts.json"
    prompts_path.write_text(json.dumps(prompts))
    return SimpleNamespace(
        root=tmp_path, total=total, video=video, video_entry=video_entry,
        manifest=manifest, manifest_path=manifest_path, episodes=episodes,
        report=report, report_path=report_path, prompts=prompts,
        prompts_path=prompts_path, human_masks=human_masks,
    )


def test_validate_inputs_uses_verified_paths_hashes_and_original_full_frame_indices(smoke, audited_inputs, monkeypatch):
    paths = []
    original = smoke._manifest_file

    def tracked(root, manifest, relative):
        paths.append(relative)
        return original(root, manifest, relative)

    monkeypatch.setattr(smoke, "_manifest_file", tracked)
    result = smoke._validate_inputs(audited_inputs.root)
    assert paths == ["track_1/meta/episodes.jsonl", audited_inputs.video_entry["path"]]
    assert result["video"] == audited_inputs.video
    assert result["video_sha256"] == audited_inputs.video_entry["sha256"]
    assert result["total_frames"] == 5
    assert result["indices"] == [0, 2, 4]
    assert result["human_masks"] == audited_inputs.human_masks
    assert result["mask_report_sha256"] == hashlib.sha256(audited_inputs.report_path.read_bytes()).hexdigest()
    assert result["prompts_sha256"] == hashlib.sha256(audited_inputs.prompts_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("key,value", [
    ("track", "track_2"), ("repo_id", "other/repo"), ("revision", "main"),
])
def test_validate_inputs_manifest_identity_fails_before_file_lookup(smoke, audited_inputs, monkeypatch, key, value):
    audited_inputs.manifest[key] = value
    audited_inputs.manifest_path.write_text(json.dumps(audited_inputs.manifest))
    monkeypatch.setattr(smoke, "_manifest_file", lambda *_: pytest.fail("Must reject identity before input lookup"))
    with pytest.raises(ValueError, match="pinned official Track 1"):
        smoke._validate_inputs(audited_inputs.root)


def test_validate_inputs_provenance_binds_to_verified_video_hash(smoke, audited_inputs, monkeypatch):
    original = smoke._manifest_file

    def changed_hash(root, manifest, relative):
        path, digest = original(root, manifest, relative)
        return path, "f" * 64 if relative.endswith(".mp4") else digest

    monkeypatch.setattr(smoke, "_manifest_file", changed_hash)
    with pytest.raises(ValueError, match="input_sha256"):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("key,value", [
    ("stage", "manual_masks"), ("status", "fail"), ("episode_index", 14),
    ("frames", 4), ("input_track", "track_2"), ("input_sha256", "0" * 64),
    ("ground_truth_used", True), ("ground_truth_used", 0),
    ("ground_truth_used", None), ("hand_labeled_test", True),
    ("hand_labeled_test", 0), ("hand_labeled_test", None),
    ("oracle_modes", ["first_frame_gt"]), ("oracle_modes", None),
])
def test_validate_inputs_strict_automatic_mask_provenance(smoke, audited_inputs, key, value):
    audited_inputs.report[key] = value
    audited_inputs.report_path.write_text(json.dumps(audited_inputs.report))
    with pytest.raises(ValueError, match=key):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("key", ["ground_truth_used", "hand_labeled_test", "frames", "oracle_modes"])
def test_validate_inputs_provenance_fields_cannot_be_omitted(smoke, audited_inputs, key):
    del audited_inputs.report[key]
    audited_inputs.report_path.write_text(json.dumps(audited_inputs.report))
    with pytest.raises(ValueError, match=key):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("length", [True, 2, 5.0, "5"])
def test_validate_inputs_official_episode_count_must_be_full_integer(smoke, audited_inputs, length):
    payload = (json.dumps({"episode_index": smoke.EPISODE, "length": length}) + "\n").encode()
    audited_inputs.episodes.write_bytes(payload)
    audited_inputs.manifest["files"][0] = record("track_1/meta/episodes.jsonl", payload)
    audited_inputs.manifest_path.write_text(json.dumps(audited_inputs.manifest))
    with pytest.raises(ValueError, match="frame count"):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_validate_inputs_episode_record_exactly_once(smoke, audited_inputs, kind):
    lines = [] if kind == "missing" else [{"episode_index": smoke.EPISODE, "length": 5}] * 2
    payload = "".join(json.dumps(line) + "\n" for line in lines).encode()
    audited_inputs.episodes.write_bytes(payload)
    audited_inputs.manifest["files"][0] = record("track_1/meta/episodes.jsonl", payload)
    audited_inputs.manifest_path.write_text(json.dumps(audited_inputs.manifest))
    with pytest.raises(ValueError, match="frame count"):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("key,value", [
    ("points", []), ("points", [{"x": 1, "y": 1}]),
    ("point_labels", [1]), ("mask_path", "manual.png"),
])
def test_validate_inputs_rejects_point_or_mask_prompts(smoke, audited_inputs, key, value):
    audited_inputs.prompts["prompts"][0][key] = value
    audited_inputs.prompts_path.write_text(json.dumps(audited_inputs.prompts))
    with pytest.raises(ValueError, match="No manual"):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("ids", [(0,), (0, 0), (1, 2), (0, 1, 2)])
def test_validate_inputs_requires_exact_person_object_prompt_ids(smoke, audited_inputs, ids):
    audited_inputs.prompts["prompts"] = [{"object_id": index} for index in ids]
    audited_inputs.prompts_path.write_text(json.dumps(audited_inputs.prompts))
    with pytest.raises(ValueError, match="SAM2 initializer"):
        smoke._validate_inputs(audited_inputs.root)


@pytest.mark.parametrize("kind", ["missing", "extra", "renamed"])
def test_validate_inputs_masks_cover_every_original_index_no_reindex(smoke, audited_inputs, kind):
    if kind == "missing":
        (audited_inputs.human_masks / "000002.png").unlink()
    elif kind == "extra":
        (audited_inputs.human_masks / "000005.png").write_bytes(b"not-a-png")
    else:
        (audited_inputs.human_masks / "000002.png").rename(audited_inputs.human_masks / "2.png")
    with pytest.raises(ValueError, match="every original frame"):
        smoke._validate_inputs(audited_inputs.root)


@pytest.fixture
def fake_torch():
    requests = []
    sentinel = object()

    def original(*args, **kwargs):
        requests.append((args, kwargs))
        return sentinel

    return SimpleNamespace(hub=SimpleNamespace(load=original)), original, requests, sentinel


def test_local_dinov3_loader_checks_pin_rewrites_and_returns_restorable_original(smoke, fake_torch, monkeypatch, tmp_path):
    torch, original, requests, sentinel = fake_torch
    pin_calls = []
    monkeypatch.setattr(smoke, "_pinned_checkout", lambda *args: pin_calls.append(args))
    repository = tmp_path / "dinov3"
    restored, calls = smoke._install_local_dinov3_loader(torch, repository)
    assert pin_calls == [(repository, smoke.DINOV3_REVISION)]
    assert restored is original
    try:
        result = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", source="github", pretrained=False,
                                weights=None, custom=7)
        assert result is sentinel
        assert requests == [((str(repository), "dinov3_vitb16"),
                             {"source": "local", "pretrained": False, "weights": None, "custom": 7})]
        assert calls == ["dinov3_vitb16"]
    finally:
        torch.hub.load = restored
    assert torch.hub.load is original


@pytest.mark.parametrize("repo,kwargs", [
    ("other/repo", {"source": "github", "pretrained": False}),
    ("facebookresearch/dinov3:main", {"source": "github", "pretrained": False}),
    ("facebookresearch/dinov3", {"source": "local", "pretrained": False}),
    ("facebookresearch/dinov3", {"pretrained": False}),
    ("facebookresearch/dinov3", {"source": "github", "pretrained": True}),
    ("facebookresearch/dinov3", {"source": "github"}),
    ("facebookresearch/dinov3", {"source": "github", "pretrained": 0}),
])
def test_local_dinov3_loader_rejects_unexpected_or_download_requests(smoke, fake_torch, monkeypatch, tmp_path, repo, kwargs):
    torch, original, requests, _ = fake_torch
    monkeypatch.setattr(smoke, "_pinned_checkout", lambda *_: None)
    restored, calls = smoke._install_local_dinov3_loader(torch, tmp_path / "dinov3")
    try:
        with pytest.raises(RuntimeError):
            torch.hub.load(repo, "dinov3_vitb16", **kwargs)
        assert requests == []
        assert calls == []
    finally:
        torch.hub.load = restored
    assert torch.hub.load is original


def test_local_dinov3_loader_does_not_patch_hub_if_source_pin_fails(smoke, fake_torch, monkeypatch, tmp_path):
    torch, original, requests, _ = fake_torch

    def fail_pin(*_):
        raise RuntimeError("Missing or unpinned source")

    monkeypatch.setattr(smoke, "_pinned_checkout", fail_pin)
    with pytest.raises(RuntimeError, match="unpinned"):
        smoke._install_local_dinov3_loader(torch, tmp_path / "dinov3")
    assert torch.hub.load is original
    assert requests == []


def test_source_identity_empty_inventory_rejects_missing_six_data_source_files(smoke, monkeypatch, tmp_path):
    vendor = tmp_path / "vendor/video_to_data"
    expected = vendor / smoke.BODY_PACKAGE
    (expected / "data").mkdir(parents=True)
    pin_calls = []
    monkeypatch.setattr(smoke, "_pinned_checkout", lambda *args: pin_calls.append(args))
    with pytest.raises(RuntimeError, match="source inventory is incomplete"):
        smoke._source_identity(tmp_path)
    assert pin_calls == [(vendor, smoke.UPSTREAM_REVISION)]
    assert list((expected / "data").rglob("*.py")) == []


def test_source_identity_partial_inventory_rejects_without_installed_tree_or_git(smoke, monkeypatch, tmp_path):
    expected = tmp_path / "vendor/video_to_data" / smoke.BODY_PACKAGE
    (expected / "data/transforms").mkdir(parents=True)
    (expected / "data/__init__.py").write_text("# tiny Python source\n")
    (expected / "data/transforms/__init__.py").write_text("# tiny Python source\n")
    monkeypatch.setattr(smoke, "_pinned_checkout", lambda *_: None)
    with pytest.raises(RuntimeError, match="source inventory is incomplete"):
        smoke._source_identity(tmp_path)
