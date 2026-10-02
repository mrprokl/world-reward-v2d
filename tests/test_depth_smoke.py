"""Lightweight episode/CLI routing without importing real GPU/media packages."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def depth(monkeypatch):
    body = SimpleNamespace(EPISODE=15, TRACK1_EPISODE_COUNT=30, _validate_inputs=lambda *_args, **_kwargs: None)
    monkeypatch.setitem(sys.modules, "body_smoke", body)
    path = Path(__file__).resolve().parents[1] / "infra/depth_smoke.py"
    spec = importlib.util.spec_from_file_location("world_reward_test_depth_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_depth_default_episode15_unchanged_and_all30_supported(depth):
    default = depth._argument_parser().parse_args([])
    assert default.episode == 15 and default.full_video is False
    for index in range(30):
        parsed = depth._argument_parser().parse_args(["--episode", str(index), "--full-video"])
        assert parsed.episode == index and parsed.full_video is True


@pytest.mark.parametrize("episode", ["-1", "30", "1.0", "true"])
def test_depth_parser_rejects_invalid_episodes_without_model_import(depth, episode):
    with pytest.raises(SystemExit):
        depth._argument_parser().parse_args(["--episode", episode])


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_depth_passes_selected_episode_to_pinned_input_guard_before_heavy_import(depth, monkeypatch, episode):
    class StopAtInputGuard(Exception): pass
    calls = []
    def validate(root, *, episode_index):
        calls.append((root, episode_index))
        raise StopAtInputGuard()
    monkeypatch.setattr(depth, "_validate_inputs", validate)
    monkeypatch.setattr(depth.platform, "system", lambda: "Linux")
    monkeypatch.setattr(Path, "iterdir", lambda *_: iter([Path("/sys/class/net/lo")]))
    monkeypatch.setenv("WR_ROOT", "/srv/tiny-episode-test")
    monkeypatch.setattr(sys, "argv", ["depth_smoke.py", "--episode", str(episode)])
    with pytest.raises(StopAtInputGuard):
        depth.main()
    assert calls == [(Path("/srv/tiny-episode-test"), episode)]


def test_automatic_masks_launcher_reuses_runtime_immutable_sources_network_none():
    script = (Path(__file__).resolve().parents[1] / "infra/run_automatic_masks.sh").read_text()
    assert 'CODE="${WR_CODE:?Require immutable committed source}"' in script
    assert "docker run --rm --gpus all --network none" in script
    assert "--network host" not in script and "docker build" not in script
    assert 'src=$CODE,dst=$CODE,readonly' in script
    for directory in ("vendor", "data", "weights", "results"):
        assert f'src=$ROOT/{directory},dst=$ROOT/{directory},readonly' in script
    assert "HF_HUB_OFFLINE=1" in script and "TRANSFORMERS_OFFLINE=1" in script
    assert 'world-reward/grounding:0.1 python "$CODE/infra/automatic_masks.py" --root "$ROOT" "$@"' in script
    assert "--episode 15" not in script
