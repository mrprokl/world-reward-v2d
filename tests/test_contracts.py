import numpy as np
import pytest

from world_reward.contracts import Reconstruction, require_video_only_provenance
from world_reward.data import DEFAULT_CONFIG, allowed_path, download


def reconstruction(**changes):
    values = dict(pose=np.zeros((4,136)), scales=np.zeros(68), shape=np.zeros(45),
                  object_rotation=np.tile(np.eye(3),(4,1,1)), object_translation=np.zeros((4,3)), object_scale=1.0)
    values.update(changes)
    return Reconstruction(**values)


def test_full_clip_contract():
    reconstruction().validate(4)
    with pytest.raises(ValueError):
        reconstruction().validate(3)


@pytest.mark.parametrize("changes", [{"object_scale":0}, {"object_scale":float("inf")},
    {"pose":np.full((4,136),np.nan)}, {"scales":np.zeros((4,68))},
    {"object_rotation":np.tile(np.diag([-1,1,1]),(4,1,1))},
    {"object_rotation":np.tile(2*np.eye(3),(4,1,1))}, {"object_translation":np.zeros((3,3))}])
def test_invalid_reconstruction_fails_closed(changes):
    with pytest.raises(ValueError):
        reconstruction(**changes).validate(4)


@pytest.mark.parametrize("path", ["track_1/meta/info.json", "track_1/meta/episodes_metadata.jsonl",
    "track_1/data/chunk-000/episode_000000.parquet",
    "track_1/videos/chunk-000/observation.images.exo_camera/episode_000029.mp4"])
def test_only_declared_track1_inputs(path):
    assert allowed_path(path)


@pytest.mark.parametrize("path", ["track_2/data/episode_000000.parquet", "track_3/mesh.glb",
    "track_1/../track_2/mesh.glb", "/track_1/meta/info.json", "track_1/cameras.json",
    "track_1/poses.npy", "track_1/object.glb", "track_1\\meta\\info.json"])
def test_leakage_and_unexpected_assets_blocked(path):
    assert not allowed_path(path)


def test_explicit_nonoracle_provenance_required():
    require_video_only_provenance({"ground_truth_used":False,"input_track":"track_1","oracle_modes":[]})
    for p in ({}, {"ground_truth_used":True}, {"ground_truth_used":False,"input_track":"track_2"},
              {"ground_truth_used":False,"input_track":"track_1","oracle_modes":["first_rotation"]}):
        with pytest.raises(ValueError):
            require_video_only_provenance(p)


def test_config_found_in_source_checkout():
    assert DEFAULT_CONFIG.is_file()


def test_local_download_blocked_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr("world_reward.data.platform.system", lambda:"Darwin")
    with pytest.raises(RuntimeError, match="remote Linux"):
        download(DEFAULT_CONFIG,tmp_path/'data',tmp_path/'manifest.json')
    assert not (tmp_path/'data').exists()
