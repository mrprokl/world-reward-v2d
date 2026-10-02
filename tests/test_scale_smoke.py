"""Argument/episode routing only, no image data or CUDA imports."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def scale(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_scale_smoke", infra / "scale_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_episode_and_all_track1_episode_choices(scale):
    assert scale._argument_parser().parse_args([]).episode == 15
    for episode in range(30):
        assert scale._argument_parser().parse_args(["--episode", str(episode)]).episode == episode


@pytest.mark.parametrize("episode", ["-1", "30", "15.0", "true", "../track_2", ""])
def test_invalid_episode_fails_before_inference(scale, episode):
    with pytest.raises(SystemExit): scale._argument_parser().parse_args(["--episode", episode])


def test_selected_episode_reaches_input_gate_before_heavy_imports(scale, monkeypatch):
    monkeypatch.setattr(scale.platform, "system", lambda: "Linux")
    monkeypatch.setattr(scale.Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", ["scale_smoke.py", "--episode", "7"])
    received = []
    def input_gate(root, *, episode_index):
        received.append(episode_index)
        raise RuntimeError("synthetic stop before image/render imports")
    monkeypatch.setattr(scale, "_validate_inputs", input_gate)
    with pytest.raises(RuntimeError, match="synthetic stop"): scale.main()
    assert received == [7]


def test_dynamic_indices_not_fixed_501_frame_assumption(scale):
    source = Path(scale.__file__).read_text()
    assert '[0, 250, 500]' not in source
    assert 'outputs/episode_000015' not in source
    assert 'indices != inputs["indices"]' in source
    assert '"episode_index": args.episode' in source


@pytest.mark.parametrize("argument", ["--full-video", "--pointmap-directory", "--root", "--no-gt"])
def test_scale_exposes_no_new_algorithm_or_arbitrary_input_modes(scale, argument):
    with pytest.raises(SystemExit): scale._argument_parser().parse_args([argument])
