"""Tiny manufactured arrays only; no challenge data, model or local rendering."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import full4d_video as video


def fixture():
    vertices = np.array([[0., 0., 3.], [.1, 0., 3.], [0., .2, 3.]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    target = np.repeat(vertices[None], 3, axis=0)
    trajectory = dict(object_vertices=vertices.copy(), object_faces=faces.copy(),
        object_rotation=np.repeat(np.eye(3)[None], 3, axis=0),
        object_translation=np.array([[0., 0., 0.], [.1, 0., .1], [.2, 0., .2]]),
        object_scale=np.asarray(1., np.float32),
        camera_K=np.array([[1920., 0, 768.], [0, 1920., 576.], [0, 0, 1.]]),
        frame_index=np.arange(3, dtype=np.int64))
    return target, faces, trajectory


def test_full_t_safe_geometry_preserved():
    target, faces, trajectory = fixture()
    original = target.copy()
    before = {k: v.copy() for k, v in trajectory.items()}
    video.checked_geometry(target, faces, trajectory, np.arange(3), 3)
    np.testing.assert_array_equal(target, original)
    for key, value in before.items():
        np.testing.assert_array_equal(trajectory[key], value)


@pytest.mark.parametrize('mutation', ['tail', 'scale', 'rotation', 'behind', 'nan', 'camera'])
def test_bad_export_fails_without_repair(mutation):
    target, faces, trajectory = fixture()
    if mutation == 'tail': trajectory['frame_index'][-1] = 99
    if mutation == 'scale': trajectory['object_scale'] = np.asarray(.1)
    if mutation == 'rotation': trajectory['object_rotation'][1, 0, 0] = -1
    if mutation == 'behind': target[1, 1, 2] = -1
    if mutation == 'nan': target[1, 1, 1] = np.nan
    if mutation == 'camera': trajectory['camera_K'][0, 1] = 1
    with pytest.raises(ValueError):
        video.checked_geometry(target, faces, trajectory, np.arange(3), 3)


def test_native_rigid_object_poses_are_not_recentred_or_scaled():
    _, _, trajectory = fixture()
    original = trajectory['object_vertices'].copy()
    for i in range(3):
        np.testing.assert_array_equal(video.object_at_frame(trajectory, i),
            original + trajectory['object_translation'][i])
    np.testing.assert_array_equal(trajectory['object_vertices'], original)


def test_near_plane_fails_no_surface_deletion():
    _, _, trajectory = fixture()
    trajectory['object_translation'][1, 2] = -4
    with pytest.raises(ValueError, match='no clipping'):
        video.object_at_frame(trajectory, 1)


def test_one_fixed_camera_display_resize_only():
    _, _, trajectory = fixture()
    camera = trajectory['camera_K'].copy()
    actual = video.display_intrinsics(camera, 1536, 1152)
    np.testing.assert_array_equal(actual, [[400., 0., 160.], [0., 400., 120.], [0., 0., 1.]])
    np.testing.assert_array_equal(camera, trajectory['camera_K'])
    with pytest.raises(ValueError): video.display_intrinsics(camera, 100, 100)


def test_floor_is_one_whole_clip_background_assumption_without_moving_mesh():
    target, _, trajectory = fixture()
    target[:, 2, 1] = [1., 2., 100.]
    before = target.copy()
    assert video.fixed_floor_height(target) == 2.
    np.testing.assert_array_equal(target, before)
    camera = video.display_intrinsics(trajectory['camera_K'], 1536, 1152)
    first = video.floor_background(camera, 2.)
    second = video.floor_background(camera, 2.)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (video.HEIGHT, video.WIDTH, 3) and first.dtype == np.uint8


def test_new_namespace_cannot_consume_baseline_or_wrong_revision(tmp_path):
    revision = 'a'*40
    experiment = tmp_path/'experiments'/f'full4d-v1-{revision}'
    prefix = f'experiments/full4d-v1-{revision}/outputs'
    pin_root = experiment/'pins'
    base, pins, actual = video.experiment_paths(tmp_path, revision, prefix, pin_root, 17)
    assert actual == experiment
    assert base == experiment/'outputs/episode_000017/cari_shared_export_v1'
    assert pins == pin_root/'cari_clip_000017_shared_export_pins.json'
    for bad in ('outputs', prefix.replace(revision, 'b'*40), '../outputs'):
        with pytest.raises(ValueError):
            video.experiment_paths(tmp_path, revision, bad, pin_root, 17)
    with pytest.raises(ValueError):
        video.experiment_paths(tmp_path, revision, prefix, pin_root, True)
    with pytest.raises(ValueError):
        video.experiment_paths(tmp_path, revision, prefix, tmp_path/'configs', 17)


def test_fixed_encoder_keeps30fps_and_all_frames_without_selection_filters():
    command = video.ffmpeg_command('/usr/bin/ffmpeg', Path('/output/test.mp4'), 634)
    assert command[command.index('-r')+1] == '30'
    assert command[command.index('-crf')+1] == '30'
    assert command[command.index('-preset')+1] == 'fast'
    assert command[command.index('-s')+1] == '640x280'
    assert not set(command) & {'-vf', '-ss', '-t', '-frames:v', '-skip_frame'}
    assert '-n' in command


def test_encoded_video_requires_full_tail_fixed_rate_and_viewport():
    stream = dict(codec_type='video', codec_name='h264', width=640, height=280,
        r_frame_rate='30/1', avg_frame_rate='30/1', nb_read_frames='634')
    video.verify_encoded(dict(streams=[stream]), 634)
    for key, value in [('nb_read_frames', '633'), ('avg_frame_rate', '15/1'),
            ('width', 320), ('codec_type', 'audio')]:
        bad = dict(stream); bad[key] = value
        with pytest.raises(ValueError):
            video.verify_encoded(dict(streams=[bad]), 634)


def test_encoder_size_ceiling_depends_only_on_original_duration():
    short = video.ffmpeg_command('ffmpeg', Path('/output.mp4'), 300)
    long = video.ffmpeg_command('ffmpeg', Path('/output.mp4'), 1500)
    assert int(short[short.index('-maxrate')+1]) > int(long[long.index('-maxrate')+1])
    with pytest.raises(ValueError): video.ffmpeg_command('ffmpeg', Path('/output.mp4'), True)
