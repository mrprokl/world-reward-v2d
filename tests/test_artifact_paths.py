from pathlib import Path
import pytest
from world_reward.artifact_paths import episode_output, output_prefix


def test_legacy_default_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("WR_OUTPUT_PREFIX", raising=False)
    assert output_prefix() == "outputs"
    assert episode_output(tmp_path, 8) == tmp_path / "outputs/episode_000008"


def test_isolated_namespace(monkeypatch, tmp_path):
    prefix = "experiments/qwen4d-v1-" + "a" * 40 + "/outputs"
    monkeypatch.setenv("WR_OUTPUT_PREFIX", prefix)
    assert episode_output(tmp_path, 9) == tmp_path / prefix / "episode_000009"


@pytest.mark.parametrize("value", ["", "/outputs", "../outputs", "outputs/a", "experiments/a/outputs", "experiments/a-" + "a" * 40 + "/../outputs"])
def test_invalid_prefix(monkeypatch, value):
    monkeypatch.setenv("WR_OUTPUT_PREFIX", value)
    with pytest.raises(ValueError):
        output_prefix()


@pytest.mark.parametrize("episode", [True, False, -1, 30, "8", 8.0])
def test_invalid_episode(monkeypatch, tmp_path, episode):
    monkeypatch.delenv("WR_OUTPUT_PREFIX", raising=False)
    with pytest.raises(ValueError):
        episode_output(tmp_path, episode)


def test_symlink_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("WR_OUTPUT_PREFIX", raising=False)
    actual = tmp_path / "actual"
    actual.mkdir()
    (tmp_path / "outputs").symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError):
        episode_output(tmp_path, 8)


def test_relative_root_rejected(monkeypatch):
    monkeypatch.delenv("WR_OUTPUT_PREFIX", raising=False)
    with pytest.raises(ValueError):
        episode_output(Path("relative"), 8)
