"""Tiny metadata/video-byte provenance regressions; no real media or model."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def masks(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("body_smoke", root / "infra/body_smoke.py")
    body = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(body)
    monkeypatch.setitem(sys.modules, "body_smoke", body)
    spec = importlib.util.spec_from_file_location("world_reward_test_automatic_masks", root / "infra/automatic_masks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_file(root, relative, payload):
    path = root / "data" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def inputs(masks, tmp_path):
    episodes = [{"episode_index": index, "length": 5 + index} for index in range(30)]
    metadata = [{"episode_index": index, "object_prompt": f" object-{index}. "} for index in range(30)]
    records = []
    for relative, values in (("track_1/meta/episodes.jsonl", episodes), ("track_1/meta/episodes_metadata.jsonl", metadata)):
        records.append(write_file(tmp_path, relative, "".join(json.dumps(value) + "\n" for value in values).encode()))
    for index in range(30):
        relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{index:06d}.mp4"
        records.append(write_file(tmp_path, relative, f"not-an-mp4-{index}".encode()))
    manifest = {"track": "track_1", "repo_id": "nvidia/video_to_data_challenge", "revision": masks.DATASET_REVISION, "files": records}
    (tmp_path / "results").mkdir()
    (tmp_path / "results/input-manifest.json").write_text(json.dumps(manifest))
    return dict(root=tmp_path, manifest=manifest, episodes=episodes, metadata=metadata)


def save_manifest(inputs):
    (inputs["root"] / "results/input-manifest.json").write_text(json.dumps(inputs["manifest"]))


def change_metadata(inputs, which):
    index = 0 if which == "episodes" else 1
    relative = inputs["manifest"]["files"][index]["path"]
    payload = "".join(json.dumps(value) + "\n" for value in inputs[which]).encode()
    inputs["manifest"]["files"][index] = write_file(inputs["root"], relative, payload)
    save_manifest(inputs)


@pytest.mark.parametrize("episode", range(30))
def test_all30_episodes_require_exact_metadata_and_original_video_hash(masks, inputs, episode):
    paths = []
    original = masks._manifest_file
    def track(*arguments): paths.append(arguments[2]); return original(*arguments)
    masks._manifest_file = track
    result = masks._validate_inputs(inputs["root"], episode)
    video_relative = f"track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4"
    assert paths == ["track_1/meta/episodes.jsonl", "track_1/meta/episodes_metadata.jsonl", video_relative]
    assert result["episode_index"] == episode and result["total_frames"] == 5 + episode
    assert result["object_prompt"] == f"object-{episode}."
    assert result["video"] == inputs["root"] / "data" / video_relative
    assert result["video_sha256"] == hashlib.sha256(f"not-an-mp4-{episode}".encode()).hexdigest()
    assert result["dataset_revision"] == masks.DATASET_REVISION
    assert result["metadata_sha256"] == {Path(record["path"]).name: record["sha256"] for record in inputs["manifest"]["files"][:2]}


@pytest.mark.parametrize("episode", [-1, 30, True, False, 15., "15", None])
def test_invalid_episode_rejected_before_manifest_io(masks, tmp_path, episode):
    with pytest.raises(ValueError, match="integer Track 1 episode"):
        masks._validate_inputs(tmp_path, episode)


@pytest.mark.parametrize("key,value", [("track", "track_2"), ("track", "track_3"), ("repo_id", "other/repo"), ("revision", "main")])
def test_pinned_manifest_identity_required_before_file_reads(masks, inputs, monkeypatch, key, value):
    inputs["manifest"][key] = value
    save_manifest(inputs)
    monkeypatch.setattr(masks, "_manifest_file", lambda *_: pytest.fail("Manifest identity must fail before reads"))
    with pytest.raises(ValueError, match="pinned official Track 1"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("index", [0, 1, 17])
def test_all_three_required_files_content_hash_checked(masks, inputs, index):
    relative = inputs["manifest"]["files"][index]["path"]
    path = inputs["root"] / "data" / relative
    data = path.read_bytes()
    path.write_bytes(b"X" * len(data))
    with pytest.raises(ValueError, match="integrity"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("index", [0, 1, 17])
def test_all_three_required_files_bytes_checked(masks, inputs, index):
    inputs["manifest"]["files"][index]["bytes"] += 1
    save_manifest(inputs)
    with pytest.raises(ValueError, match="integrity"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("index", [0, 1, 17])
@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_required_manifest_records_exactly_once(masks, inputs, index, mode):
    if mode == "duplicate": inputs["manifest"]["files"].append(dict(inputs["manifest"]["files"][index]))
    else: del inputs["manifest"]["files"][index]
    save_manifest(inputs)
    with pytest.raises(ValueError, match="exactly one record"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("which", ["episodes", "metadata"])
@pytest.mark.parametrize("mode", ["duplicate", "missing"])
def test_chosen_episode_record_exactly_once(masks, inputs, which, mode):
    if mode == "duplicate": inputs[which].append(dict(inputs[which][15]))
    else: del inputs[which][15]
    change_metadata(inputs, which)
    with pytest.raises(ValueError, match="exactly one record for episode 15"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("which", ["episodes", "metadata"])
@pytest.mark.parametrize("bad", [True, 15., "15", -1, 30, None])
def test_metadata_identity_strict_integer_in_range(masks, inputs, which, bad):
    inputs[which][15]["episode_index"] = bad
    change_metadata(inputs, which)
    with pytest.raises(ValueError, match="integer Track 1 episode identities"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("bad", [True, False, 2, -1, 20., "20", None])
def test_full_original_frame_count_strict_integer_at_least3(masks, inputs, bad):
    inputs["episodes"][15]["length"] = bad
    change_metadata(inputs, "episodes")
    with pytest.raises(ValueError, match="frame count"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None, True, 3, ["object"]])
def test_public_object_prompt_must_be_nonempty_string_no_guess(masks, inputs, bad):
    inputs["metadata"][15]["object_prompt"] = bad
    change_metadata(inputs, "metadata")
    with pytest.raises(ValueError, match="nonempty string"):
        masks._validate_inputs(inputs["root"], 15)


def test_missing_prompt_fails_without_fallback(masks, inputs):
    del inputs["metadata"][15]["object_prompt"]
    change_metadata(inputs, "metadata")
    with pytest.raises(ValueError, match="object_prompt"):
        masks._validate_inputs(inputs["root"], 15)


@pytest.mark.parametrize("index", [0, 1, 17])
def test_symlink_input_not_accepted_even_with_correct_hash(masks, inputs, tmp_path, index):
    relative = inputs["manifest"]["files"][index]["path"]
    path = inputs["root"] / "data" / relative
    target = tmp_path / f"outside-{index}"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="escapes"):
        masks._validate_inputs(inputs["root"], 15)


def test_video_path_never_taken_from_metadata(masks, inputs):
    inputs["metadata"][15]["video_path"] = "../track_2/forbidden.mp4"
    inputs["episodes"][15]["video_path"] = "../track_3/forbidden.mp4"
    change_metadata(inputs, "metadata")
    change_metadata(inputs, "episodes")
    result = masks._validate_inputs(inputs["root"], 15)
    assert result["video"].name == "episode_000015.mp4"
    assert "track_1" in result["video"].parts


@pytest.mark.parametrize("interface_names", [[], ["lo", "eth0"], ["eth0"]])
def test_main_requires_network_none_before_metadata_or_models(masks, monkeypatch, interface_names):
    monkeypatch.setattr(masks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda *_: iter(Path("/sys/class/net") / name for name in interface_names))
    monkeypatch.setattr(masks, "_validate_inputs", lambda *_args, **_kwargs: pytest.fail("Must fail isolation first"))
    with pytest.raises(RuntimeError, match="docker --network none"):
        masks.main()


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_main_routes_chosen_episode_to_input_guard_before_outputs_models(masks, monkeypatch, episode):
    class StopAtGuard(Exception): pass
    calls = []
    def guard(root, episode_index): calls.append((root, episode_index)); raise StopAtGuard()
    monkeypatch.setattr(masks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda *_: iter([Path("/sys/class/net/lo")]))
    monkeypatch.setattr(masks, "_validate_inputs", guard)
    monkeypatch.setattr(sys, "argv", ["automatic_masks.py", "--root", "/srv/frozen-test", "--episode", str(episode)])
    with pytest.raises(StopAtGuard): masks.main()
    assert calls == [(Path("/srv/frozen-test"), episode)]
def test_detector_batch_cannot_shadow_validated_input_provenance():
    """Input identity must survive detector loops until the final report."""
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1] / "infra/automatic_masks.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignments = [node for node in ast.walk(main) if isinstance(node, ast.Name)
                   and node.id == "inputs" and isinstance(node.ctx, ast.Store)]
    assert len(assignments) == 1
    gate = next(node for node in main.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "inputs" for target in node.targets))
    assert isinstance(gate.value, ast.Call) and gate.value.func.id == "_validate_inputs"
