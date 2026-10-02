"""Pure CLI routing for automatic frame-zero object generation."""

import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def objects(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_object_smoke", infra / "object_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_modes_and_all_episode_choices(objects):
    parsed = objects._argument_parser().parse_args([])
    assert parsed.episode == 15 and parsed.aligned_pointmap is False
    assert parsed.root == Path("/srv/scenesmith/world-reward")
    for episode in range(30):
        parsed = objects._argument_parser().parse_args(["--episode", str(episode), "--aligned-pointmap"])
        assert parsed.episode == episode and parsed.aligned_pointmap is True


@pytest.mark.parametrize("episode", ["-1", "30", "15.0", "true", "../track_3", ""])
def test_invalid_episode_fails_before_inference(objects, episode):
    with pytest.raises(SystemExit): objects._argument_parser().parse_args(["--episode", episode])


def test_existing_root_argument_preserved(objects):
    assert objects._argument_parser().parse_args(["--root", "/srv/test"]).root == Path("/srv/test")


def test_selected_episode_reaches_input_gate_before_heavy_imports(objects, monkeypatch):
    monkeypatch.setattr(objects.platform, "system", lambda: "Linux")
    monkeypatch.setattr(objects.Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", ["object_smoke.py", "--episode", "29", "--aligned-pointmap"])
    received = []
    def input_gate(root, *, episode_index):
        received.append(episode_index)
        raise RuntimeError("synthetic stop before assets/models")
    monkeypatch.setattr(objects, "_validate_inputs", input_gate)
    with pytest.raises(RuntimeError, match="synthetic stop"): objects.main()
    assert received == [29]


def test_paths_and_provenance_are_selected_episode_not_hardcoded(objects):
    source = Path(objects.__file__).read_text()
    assert 'outputs/episode_000015' not in source
    assert '"episode_index": 15' not in source
    assert '"episode_index": args.episode' in source
    assert 'base / "automatic_masks/masks/1/000000.png"' in source
    assert 'base / "scale_smoke/report.json"' in source


@pytest.mark.parametrize("argument", ["--pointmap-directory", "--full-video", "--manual-mask", "--no-gt"])
def test_no_arbitrary_grounding_or_new_algorithm_flags(objects, argument):
    with pytest.raises(SystemExit): objects._argument_parser().parse_args([argument])
