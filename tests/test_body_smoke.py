"""Offline, data-free regressions for the Body smoke's input/source/Hub guards."""

from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


class StubTensor:
    """Only the tensor operations required by the checkpoint guard; no Torch."""

    def __init__(self, values, *, requires_grad=False, device="cpu"):
        self.values = np.asarray(values).copy()
        self.requires_grad = requires_grad
        self.device = device

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    def detach(self):
        return StubTensor(self.values, requires_grad=False, device=self.device)

    def clone(self):
        return StubTensor(self.values, requires_grad=self.requires_grad, device=self.device)

    def cpu(self):
        return StubTensor(self.values, requires_grad=self.requires_grad, device="cpu")


class StubModule:
    def __init__(self, parameters, buffers):
        self.parameters = dict(parameters)
        self.buffers = dict(buffers)
        self.state = OrderedDict({**self.parameters, **self.buffers})
        self.parameter_enumeration_args = []
        self.buffer_enumeration_args = []

    def state_dict(self):
        return self.state.copy()

    def named_parameters(self, **kwargs):
        self.parameter_enumeration_args.append(kwargs)
        return self.parameters.items()

    def named_buffers(self, **kwargs):
        self.buffer_enumeration_args.append(kwargs)
        return self.buffers.items()


@pytest.fixture
def asset_checkpoint():
    parameters = {
        "backbone.weight": StubTensor([1.0, 2.0], requires_grad=True),
        "backbone.frozen_weight": StubTensor([3.0], requires_grad=False),
    }
    buffers = {
        "head_pose.mhr.character_torch.rest_vertices": StubTensor([[1.0, 2.0, 3.0]]),
        "head_pose_hand.mhr.character_torch.skinning_weights": StubTensor([0.4, 0.6]),
        "head_pose.faces": StubTensor([[0, 1, 2]]),
        "head_pose.scale_params": StubTensor([0.0]),
        "head_pose.keypoint_mapping": StubTensor([0, 2]),
    }
    module = StubModule(parameters, buffers)
    state = OrderedDict((name, tensor.clone()) for name, tensor in module.state.items())
    state._metadata = OrderedDict({"": {"version": 1}, "head_pose": {"version": 2}})
    torch = SimpleNamespace(
        isfinite=lambda tensor: np.isfinite(tensor.values),
        equal=lambda a, b: np.array_equal(a.values, b.values),
    )
    calls = []

    def loader(target, merged, *, strict):
        calls.append((target, merged, strict))
        assert strict is True
        assert set(merged) == set(target.state)
        target.state = OrderedDict((name, tensor.clone()) for name, tensor in merged.items())

    return SimpleNamespace(module=module, state=state, torch=torch, loader=loader, calls=calls)


def load_assets(smoke, fixture):
    return smoke._load_checkpoint_with_asset_buffers(
        fixture.module, fixture.state, fixture.loader, fixture.torch
    )


def test_asset_checkpoint_calls_original_loader_strict_preserves_metadata_and_input_state(smoke, asset_checkpoint):
    fixture = asset_checkpoint
    before = {name: tensor.values.copy() for name, tensor in fixture.state.items()}
    original_keys = list(fixture.state)
    metadata = fixture.state._metadata
    result = load_assets(smoke, fixture)
    assert len(fixture.calls) == 1
    target, merged, strict = fixture.calls[0]
    assert target is fixture.module and strict is True
    assert merged is not fixture.state
    assert merged._metadata is metadata
    assert list(fixture.state) == original_keys
    assert fixture.state._metadata is metadata
    assert all(np.array_equal(fixture.state[name].values, value) for name, value in before.items())
    assert fixture.module.parameter_enumeration_args == [{"remove_duplicate": False}]
    assert fixture.module.buffer_enumeration_args == [{"remove_duplicate": False}]
    assert result == {
        "mode": "strict_network_and_head_state_with_explicit_asset_buffer_retention",
        "retained_mhr_asset_buffer_names": [], "parameter_tensors_loaded": 2,
        "unexpected_keys": [],
    }


def test_asset_checkpoint_retains_only_two_exact_character_torch_buffer_namespaces(smoke, asset_checkpoint):
    fixture = asset_checkpoint
    names = [name for name in fixture.state if ".character_torch." in name]
    for name in names:
        del fixture.state[name]
    keys_before = set(fixture.state)
    result = load_assets(smoke, fixture)
    assert result["retained_mhr_asset_buffer_names"] == sorted(names)
    assert set(fixture.state) == keys_before  # merged dictionary, not checkpoint mutation
    merged = fixture.calls[0][1]
    for name in names:
        assert fixture.torch.equal(merged[name], fixture.module.buffers[name])
        assert merged[name] is not fixture.module.buffers[name]
        assert merged[name].requires_grad is False
        assert not np.shares_memory(merged[name].values, fixture.module.buffers[name].values)


@pytest.mark.parametrize("name", ["backbone.weight", "backbone.frozen_weight"])
def test_every_parameter_including_frozen_is_required(smoke, asset_checkpoint, name):
    del asset_checkpoint.state[name]
    with pytest.raises(RuntimeError, match="Required model state missing"):
        load_assets(smoke, asset_checkpoint)
    assert asset_checkpoint.calls == []


@pytest.mark.parametrize("name", ["backbone.weight", "backbone.frozen_weight"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_every_parameter_including_frozen_must_be_finite(smoke, asset_checkpoint, name, nonfinite):
    asset_checkpoint.state[name] = StubTensor([nonfinite])
    with pytest.raises(RuntimeError, match="parameter missing or nonfinite"):
        load_assets(smoke, asset_checkpoint)
    assert asset_checkpoint.calls == []


@pytest.mark.parametrize("prefix", ["head_pose.mhr.character_torch.", "head_pose_hand.mhr.character_torch."])
def test_parameter_cannot_use_asset_buffer_exception_even_when_frozen_and_in_both_enumerations(smoke, asset_checkpoint, prefix):
    fixture = asset_checkpoint
    name = prefix + "frozen_learned_parameter"
    value = StubTensor([7.0], requires_grad=False)
    fixture.module.parameters[name] = value
    fixture.module.buffers[name] = value
    fixture.module.state[name] = value
    with pytest.raises(RuntimeError, match="Required model state missing"):
        load_assets(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("prefix", ["head_pose.mhr.character_torch.", "head_pose_hand.mhr.character_torch."])
def test_requires_grad_buffer_cannot_be_retained_when_missing(smoke, asset_checkpoint, prefix):
    fixture = asset_checkpoint
    name = prefix + "gradient_buffer"
    value = StubTensor([1.0], requires_grad=True)
    fixture.module.buffers[name] = value
    fixture.module.state[name] = value
    with pytest.raises(RuntimeError, match="Required model state missing"):
        load_assets(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("name", [
    "head_pose.faces", "head_pose.scale_params", "head_pose.keypoint_mapping",
    "head_pose.mhr.character_torch_extra.buffer", "head_pose.mhr.character_torch",
    "head_pose_hand.mhr.character_torch_extra.buffer",
    "head_pose_other.mhr.character_torch.buffer",
])
def test_nonwhitelisted_head_topology_scale_keypoint_or_lookalike_buffer_must_be_in_checkpoint(smoke, asset_checkpoint, name):
    fixture = asset_checkpoint
    if name not in fixture.module.state:
        value = StubTensor([1.0])
        fixture.module.state[name] = value
        fixture.module.buffers[name] = value
    fixture.state.pop(name, None)
    with pytest.raises(RuntimeError, match="Required model state missing"):
        load_assets(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("name", [
    "head_pose.mhr.character_torch.rest_vertices",
    "head_pose_hand.mhr.character_torch.skinning_weights",
])
def test_checkpoint_asset_buffer_must_exactly_equal_explicit_asset(smoke, asset_checkpoint, name):
    asset_checkpoint.state[name].values.flat[0] += 0.00001
    with pytest.raises(RuntimeError, match="contradicts explicit immutable MHR asset"):
        load_assets(smoke, asset_checkpoint)
    assert asset_checkpoint.calls == []


@pytest.mark.parametrize("name", [
    "backbone.unexpected_weight", "head_pose.mhr.character_torch.unexpected_buffer",
])
def test_unexpected_checkpoint_keys_fail_even_in_allowed_namespace(smoke, asset_checkpoint, name):
    asset_checkpoint.state[name] = StubTensor([1.0])
    with pytest.raises(RuntimeError, match="Unexpected checkpoint keys"):
        load_assets(smoke, asset_checkpoint)
    assert asset_checkpoint.calls == []


@pytest.mark.parametrize("mutation", ["inplace", "replace"])
def test_loader_cannot_mutate_immutable_asset_buffer_after_load(smoke, asset_checkpoint, mutation):
    fixture = asset_checkpoint
    original = fixture.loader
    name = "head_pose.mhr.character_torch.rest_vertices"

    def mutating_loader(module, merged, *, strict):
        original(module, merged, strict=strict)
        if mutation == "inplace":
            module.state[name].values.flat[0] += 1
        else:
            module.state[name] = StubTensor([[9.0, 9.0, 9.0]])

    fixture.loader = mutating_loader
    with pytest.raises(RuntimeError, match="buffers changed during checkpoint loading"):
        load_assets(smoke, fixture)
    assert fixture.calls[0][2] is True


def test_retained_buffer_snapshot_is_independent_of_loader_inplace_mutation(smoke, asset_checkpoint):
    fixture = asset_checkpoint
    name = "head_pose_hand.mhr.character_torch.skinning_weights"
    del fixture.state[name]

    def bad_loader(module, merged, *, strict):
        assert strict is True
        # Mutate the supplied retained tensor and the module alike. The expected
        # reference must not alias loader input, or this corruption goes unseen.
        merged[name].values.flat[0] += 1
        module.state = OrderedDict(merged)

    fixture.loader = bad_loader
    with pytest.raises(RuntimeError, match="buffers changed during checkpoint loading"):
        load_assets(smoke, fixture)


def test_checkpoint_without_metadata_loads_without_inventing_metadata(smoke, asset_checkpoint):
    fixture = asset_checkpoint
    fixture.state = dict(fixture.state)
    result = load_assets(smoke, fixture)
    assert result["parameter_tensors_loaded"] == 2
    assert not hasattr(fixture.calls[0][1], "_metadata")


@pytest.fixture
def explicit_asset_checkpoint(asset_checkpoint):
    fixture = asset_checkpoint
    fixture.explicit = OrderedDict({
        "character_torch.rest_vertices": StubTensor([[1.0, 2.0, 3.0]]),
        "network.weight": StubTensor([[0.1, 0.2], [0.3, 0.4]]),
        "network.bias": StubTensor([0.5, 0.6]),
    })
    for prefix in ("head_pose.mhr.", "head_pose_hand.mhr."):
        # Replace the two model-local inventories with exact independent copies.
        for container in (fixture.module.state, fixture.module.parameters, fixture.module.buffers, fixture.state):
            for name in list(container):
                if name.startswith(prefix):
                    del container[name]
        for relative, independent in fixture.explicit.items():
            name = prefix + relative
            constructed = StubTensor(independent.values, device="cuda:0")
            fixture.module.state[name] = constructed
            if relative.startswith("network."):
                fixture.module.parameters[name] = constructed  # frozen asset parameters
            else:
                fixture.module.buffers[name] = constructed
            fixture.state[name] = independent.clone()  # checkpoints conventionally on CPU
    for prefix in ("head_pose.", "head_pose_hand."):
        name = prefix + "hand_pose_comps_ori"
        value = StubTensor(np.eye(54), device="cuda:0")
        fixture.module.state[name] = value
        fixture.module.buffers[name] = value
        fixture.state[name] = value.cpu()
    name = "backbone.encoder.mask_token"
    value = StubTensor(np.zeros((1, 4)), requires_grad=False, device="cuda:0")
    fixture.module.state[name] = value
    fixture.module.parameters[name] = value
    fixture.state[name] = value.cpu()
    fixture.module.backbone = SimpleNamespace(encoder=SimpleNamespace(embed_dim=4))

    def same_device_equal(a, b):
        if a.device != b.device:
            raise RuntimeError("Stub equal requires canonical same-device comparison")
        return np.array_equal(a.values, b.values)

    fixture.torch.equal = same_device_equal
    fixture.torch.eye = lambda size, *, device, dtype: StubTensor(np.eye(size, dtype=dtype), device=device)
    fixture.torch.count_nonzero = lambda tensor: SimpleNamespace(item=lambda: int(np.count_nonzero(tensor.values)))
    return fixture


def load_explicit(smoke, fixture):
    return smoke._load_checkpoint_with_asset_buffers(
        fixture.module, fixture.state, fixture.loader, fixture.torch,
        explicit_asset_state=fixture.explicit,
    )


def test_explicit_complete_two_mhr_inventories_cpu_asset_gpu_model_verified_then_strict(smoke, explicit_asset_checkpoint):
    fixture = explicit_asset_checkpoint
    result = load_explicit(smoke, fixture)
    assert len(fixture.calls) == 1 and fixture.calls[0][2] is True
    assert result["retained_mhr_asset_buffer_names"] == []
    assert result["parameter_tensors_loaded"] == 7
    assert fixture.calls[0][1]._metadata is fixture.state._metadata


def test_explicit_verified_frozen_asset_parameters_and_unused_states_can_be_retained(smoke, explicit_asset_checkpoint):
    fixture = explicit_asset_checkpoint
    names = [name for name in fixture.state if name.startswith(("head_pose.mhr.", "head_pose_hand.mhr."))]
    names += ["head_pose.hand_pose_comps_ori", "head_pose_hand.hand_pose_comps_ori", "backbone.encoder.mask_token"]
    for name in names:
        del fixture.state[name]
    before = set(fixture.state)
    result = load_explicit(smoke, fixture)
    assert result["retained_mhr_asset_buffer_names"] == sorted(names)
    assert set(fixture.state) == before
    for name in names:
        retained = fixture.calls[0][1][name]
        assert retained.requires_grad is False
        assert not np.shares_memory(retained.values, fixture.module.state[name].values)
    assert fixture.calls[0][1]._metadata is fixture.state._metadata


@pytest.mark.parametrize("prefix", ["head_pose.mhr.", "head_pose_hand.mhr."])
def test_independent_asset_verification_includes_frozen_network_parameters(smoke, explicit_asset_checkpoint, prefix):
    fixture = explicit_asset_checkpoint
    name = prefix + "network.weight"
    fixture.module.state[name].values.flat[0] += 0.001
    with pytest.raises(RuntimeError, match="differs from independent explicit asset"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("kind", ["missing_constructed", "extra_constructed", "missing_independent", "extra_independent"])
def test_exact_full_mhr_asset_inventory_required_not_partial_character_buffers(smoke, explicit_asset_checkpoint, kind):
    fixture = explicit_asset_checkpoint
    if kind == "missing_constructed":
        del fixture.module.state["head_pose_hand.mhr.network.bias"]
    elif kind == "extra_constructed":
        fixture.module.state["head_pose.mhr.extra"] = StubTensor([1.0])
    elif kind == "missing_independent":
        del fixture.explicit["network.bias"]
    else:
        fixture.explicit["extra"] = StubTensor([1.0])
    with pytest.raises(RuntimeError, match="explicit asset inventory"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("prefix", ["head_pose.", "head_pose_hand."])
@pytest.mark.parametrize("kind", ["shape", "nonidentity", "requires_grad", "nonfinite"])
def test_original_hand_pca_requires_exact_nonlearned_identity_54(smoke, explicit_asset_checkpoint, prefix, kind):
    fixture = explicit_asset_checkpoint
    name = prefix + "hand_pose_comps_ori"
    values = np.eye(53) if kind == "shape" else np.eye(54)
    if kind == "nonidentity":
        values[0, 1] = 1e-8
    elif kind == "nonfinite":
        values[0, 0] = np.nan
    fixture.module.state[name] = StubTensor(values, requires_grad=kind == "requires_grad", device="cuda:0")
    with pytest.raises(RuntimeError, match="deterministic identity"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("values", [
    np.zeros((4,)), np.zeros((2, 4)), np.zeros((1, 3)),
    np.full((1, 4), np.nan), np.full((1, 4), np.inf), np.ones((1, 4)),
])
def test_dino_mask_token_must_have_exact_shape_finite_zero_initialization(smoke, explicit_asset_checkpoint, values):
    fixture = explicit_asset_checkpoint
    fixture.module.state["backbone.encoder.mask_token"] = StubTensor(values, device="cuda:0")
    with pytest.raises(RuntimeError, match="pinned zero initialization"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("name", [
    "head_pose.mhr.network.weight", "head_pose_hand.mhr.network.bias",
    "head_pose.hand_pose_comps_ori", "head_pose_hand.hand_pose_comps_ori",
    "backbone.encoder.mask_token",
])
def test_checkpoint_cannot_contradict_independent_frozen_asset_or_deterministic_unused_state(smoke, explicit_asset_checkpoint, name):
    fixture = explicit_asset_checkpoint
    fixture.state[name].values.flat[0] += 0.001
    with pytest.raises(RuntimeError, match="contradicts explicit immutable MHR asset"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("name", ["backbone.weight", "backbone.frozen_weight"])
def test_explicit_asset_branch_still_requires_all_nonasset_network_parameters(smoke, explicit_asset_checkpoint, name):
    fixture = explicit_asset_checkpoint
    del fixture.state[name]
    with pytest.raises(RuntimeError, match="Required model state missing"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


@pytest.mark.parametrize("name", ["backbone.weight", "backbone.frozen_weight"])
def test_explicit_asset_branch_still_rejects_nonfinite_network_parameters(smoke, explicit_asset_checkpoint, name):
    fixture = explicit_asset_checkpoint
    fixture.state[name] = StubTensor([np.nan])
    with pytest.raises(RuntimeError, match="parameter missing or nonfinite"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


def test_lookalike_third_mhr_prefix_is_not_independent_asset_allowlist(smoke, explicit_asset_checkpoint):
    fixture = explicit_asset_checkpoint
    name = "head_pose_other.mhr.network.weight"
    value = StubTensor([1.0], requires_grad=False)
    fixture.module.state[name] = value
    fixture.module.parameters[name] = value
    with pytest.raises(RuntimeError, match="Required model state missing"):
        load_explicit(smoke, fixture)
    assert fixture.calls == []


def test_explicit_retained_parameter_snapshot_cannot_mutate_during_loading(smoke, explicit_asset_checkpoint):
    fixture = explicit_asset_checkpoint
    name = "head_pose.mhr.network.weight"
    del fixture.state[name]

    def bad_loader(module, merged, *, strict):
        assert strict is True
        merged[name].values.flat[0] += 1
        module.state = OrderedDict(merged)

    fixture.loader = bad_loader
    with pytest.raises(RuntimeError, match="buffers changed during checkpoint loading"):
        load_explicit(smoke, fixture)


def episode_inputs(smoke, audited_inputs, episode_index):
    """Add a distinct tiny episode while preserving the default15 fixtures."""
    other = episode_index
    episodes = [json.loads(line) for line in audited_inputs.episodes.read_text().splitlines()]
    if other != smoke.EPISODE:
        episodes.append({"episode_index": other, "length": audited_inputs.total})
    payload = "".join(json.dumps(item) + "\n" for item in episodes).encode()
    audited_inputs.episodes.write_bytes(payload)
    audited_inputs.manifest["files"][0] = record("track_1/meta/episodes.jsonl", payload)
    relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{other:06d}.mp4"
    if other != smoke.EPISODE:
        video, entry = write_input(audited_inputs.root, relative, f"not-media-episode-{other}".encode())
        audited_inputs.manifest["files"].append(entry)
    else:
        video, entry = audited_inputs.video, audited_inputs.video_entry
    audited_inputs.manifest_path.write_text(json.dumps(audited_inputs.manifest))
    masks = audited_inputs.root / f"outputs/episode_{other:06d}/automatic_masks"
    human = masks / "masks/0"
    human.mkdir(parents=True, exist_ok=True)
    for index in range(audited_inputs.total):
        (human / f"{index:06d}.png").write_bytes(b"not-a-png")
    report = dict(audited_inputs.report, episode_index=other, input_sha256=entry["sha256"])
    (masks / "report.json").write_text(json.dumps(report))
    (masks / "prompts.json").write_text(json.dumps(audited_inputs.prompts))
    return SimpleNamespace(video=video, entry=entry, masks=masks, human=human, report=report)


@pytest.mark.parametrize("episode", range(30))
def test_validate_inputs_all_track1_episodes_keep_distinct_paths_provenance(smoke, audited_inputs, episode):
    chosen = episode_inputs(smoke, audited_inputs, episode)
    original_default_video = audited_inputs.video.read_bytes()
    result = smoke._validate_inputs(audited_inputs.root, episode_index=episode)
    assert result["episode_index"] == episode
    assert result["dataset_revision"] == smoke.DATASET_REVISION
    assert result["video"] == chosen.video
    assert result["video_sha256"] == chosen.entry["sha256"]
    assert result["human_masks"] == chosen.human
    assert result["indices"] == [0, 2, 4]
    assert audited_inputs.video.read_bytes() == original_default_video


@pytest.mark.parametrize("episode", [-1, 30, True, False, 15., "15", None, np.int64(15)])
def test_validate_inputs_episode_index_strict_before_any_file_lookup(smoke, tmp_path, monkeypatch, episode):
    monkeypatch.setattr(smoke, "_manifest_file", lambda *_: pytest.fail("Invalid episode must fail before any file lookup"))
    with pytest.raises(ValueError, match="integer Track 1 episode index"):
        smoke._validate_inputs(tmp_path, episode_index=episode)


@pytest.mark.parametrize("field,value", [("episode_index", 15), ("episode_index", True), ("episode_index", 1.), ("input_sha256", "0" * 64)])
def test_selected_episode_rejects_wrong_mask_provenance(smoke, audited_inputs, field, value):
    chosen = episode_inputs(smoke, audited_inputs, 1)
    chosen.report[field] = value
    (chosen.masks / "report.json").write_text(json.dumps(chosen.report))
    with pytest.raises(ValueError, match="provenance"):
        smoke._validate_inputs(audited_inputs.root, episode_index=1)


@pytest.mark.parametrize("episode_value", [True, 1., "1"])
def test_official_episode_identity_requires_integer_not_boolean_or_alias(smoke, audited_inputs, episode_value):
    chosen = episode_inputs(smoke, audited_inputs, 1)
    entries = [{"episode_index": smoke.EPISODE, "length": 5}, {"episode_index": episode_value, "length": 5}]
    payload = "".join(json.dumps(item) + "\n" for item in entries).encode()
    audited_inputs.episodes.write_bytes(payload)
    audited_inputs.manifest["files"][0] = record("track_1/meta/episodes.jsonl", payload)
    audited_inputs.manifest_path.write_text(json.dumps(audited_inputs.manifest))
    with pytest.raises(ValueError, match="episode 1 frame count"):
        smoke._validate_inputs(audited_inputs.root, episode_index=1)


def test_body_argument_parser_default15_unchanged_and_all30_supported(smoke):
    default = smoke._argument_parser().parse_args([])
    assert default.episode == 15 and default.full_video is False
    assert default.root == Path("/srv/scenesmith/world-reward")
    for index in range(30):
        parsed = smoke._argument_parser().parse_args(["--episode", str(index), "--full-video"])
        assert parsed.episode == index and parsed.full_video is True


@pytest.mark.parametrize("episode", ["-1", "30", "1.0", "true"])
def test_body_argument_parser_rejects_invalid_episode(smoke, episode):
    with pytest.raises(SystemExit):
        smoke._argument_parser().parse_args(["--episode", episode])


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_body_main_routes_selected_episode_to_input_guard_before_model_loading(smoke, monkeypatch, episode):
    class StopBeforeModel(Exception): pass
    calls = []
    def validate(root, *, episode_index):
        calls.append((root, episode_index))
        return {"total_frames": 5, "indices": [0, 2, 4]}
    def stop(_root): raise StopBeforeModel()
    monkeypatch.setattr(smoke, "_validate_inputs", validate)
    monkeypatch.setattr(smoke, "_source_identity", stop)
    monkeypatch.setattr(smoke.platform, "system", lambda: "Linux")
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter([Path("/sys/class/net/lo")]) if str(path) == "/sys/class/net" else original_iterdir(path))
    monkeypatch.setattr(smoke.sys, "argv", ["body_smoke.py", "--root", "/srv/tiny-episode-test", "--episode", str(episode), "--full-video"])
    monkeypatch.setattr(smoke.sys, "dont_write_bytecode", smoke.sys.dont_write_bytecode)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MOMENTUM_ENABLED", "WANDB_MODE"):
        monkeypatch.setenv(name, "fixture")
    with pytest.raises(StopBeforeModel):
        smoke.main()
    assert calls == [(Path("/srv/tiny-episode-test"), episode)]
