"""Episode/full-video planning only; no geometry assets, labels, or CUDA."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def pose(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_object_pose_smoke", infra / "object_pose_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_sparse_mode_and_all_full_episode_choices(pose):
    parsed = pose._argument_parser().parse_args([])
    assert parsed.episode == 15 and parsed.full_video is False
    for episode in range(30):
        parsed = pose._argument_parser().parse_args(["--episode", str(episode), "--full-video"])
        assert parsed.episode == episode and parsed.full_video is True


@pytest.mark.parametrize("episode", ["-1", "30", "15.0", "false", "../track_2", ""])
def test_invalid_episode_fails_before_heavy_imports(pose, episode):
    with pytest.raises(SystemExit): pose._argument_parser().parse_args(["--episode", episode])


def test_selected_episode_reaches_input_gate_before_trimesh_cuda(pose, monkeypatch):
    monkeypatch.setattr(pose.platform, "system", lambda: "Linux")
    monkeypatch.setattr(pose.Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", ["object_pose_smoke.py", "--episode", "0", "--full-video"])
    received = []
    def input_gate(root, *, episode_index):
        received.append(episode_index)
        raise RuntimeError("synthetic stop before geometry imports")
    monkeypatch.setattr(pose, "_validate_inputs", input_gate)
    with pytest.raises(RuntimeError, match="synthetic stop"): pose.main()
    assert received == [0]


def test_dynamic_full_sparse_indices_and_selected_episode_provenance(pose):
    source = Path(pose.__file__).read_text()
    assert 'outputs/episode_000015' not in source
    assert 'indices = list(range(inputs["total_frames"]))' in source
    assert 'indices = inputs["indices"]' in source
    assert '"episode_index": args.episode' in source
    assert 'full_body.get("episode_index") != args.episode' in source


@pytest.mark.parametrize("argument", ["--pointmap-directory", "--root", "--manual-mask", "--shape-fit", "--no-gt"])
def test_pose_has_no_new_research_algorithm_or_arbitrary_input_modes(pose, argument):
    with pytest.raises(SystemExit): pose._argument_parser().parse_args([argument])
