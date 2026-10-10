"""Tiny full-T wiring/coordinate contracts; no local RGB/media/model jobs."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import video_depth_pose_real as runner
from test_sequence_contact_patch_real import tiny_bank, protocol
from test_sequence_pose_contact import config


def test_frozen_pose_config_is_integrated_with_actual_external_approval():
    from pathlib import Path
    cfg = runner.settings(Path(__file__).parents[1])
    assert cfg['depth_sigma_diameter'] == runner.real.saved.DEPTH_CFG.depth_sigma_diameter
    assert cfg['external_validation_approval']['report_pin']['bytes'] == 4504
    assert cfg['gates']['compare_against'] == ['original']


class Tensor:
    def __init__(self, value): self.value = value
    def cpu(self): return self
    def numpy(self): return self.value


def test_renderer_camera_is_only_pixel_basis_conversion():
    original = np.array([[800., 0., 319.5], [0., 800., 239.5], [0., 0., 1.]])
    matrix = runner.renderer_camera(original)
    assert matrix[0, 2] == 320 and matrix[1, 2] == 240
    np.testing.assert_array_equal(matrix[:, :2], original[:, :2])
    np.testing.assert_array_equal(original[:, 2], [319.5, 239.5, 1])


def test_one_human_scale_uses_three_fixed_original_frames_equal_weight_and_no_object():
    depth = np.ones((415, 4, 5), np.float32) * 2
    human = np.ones((415, 3, 3), np.float64); human[:, :, 2] = 4
    bank = {'K': np.eye(3)}; sources = {'human': human, 'human_faces': np.array([[0, 1, 2]])}
    calls = []
    def raster(vertices, faces, k, w, h):
        calls.append((vertices.copy(), k.copy(), w, h))
        return Tensor(np.ones((4, 5), bool)), Tensor(np.ones((4, 5), np.float32) * 4)
    def masks(frame):
        observed = np.ones((4, 5), bool); obj = np.zeros_like(observed); obj[0, 0] = True
        return observed, obj
    # Tiny fixtures exercise exact scientific primitive, not an alternate fit.
    monkeypatch = pytest.MonkeyPatch()
    original = runner.fit_shared_depth_scale
    def wrapped(z, h, valid, indices):
        assert indices == (0, 207, 414)
        assert all(not v[0, 0] for v in valid)
        return original(z, h, valid, indices, min_correspondences_per_frame=8)
    monkeypatch.setattr(runner, 'fit_shared_depth_scale', wrapped)
    try: aligned = runner.shared_human_scale(depth, bank, sources, masks, raster)
    finally: monkeypatch.undo()
    assert aligned.shared_scale == 2 and aligned.supported_frames == 3
    assert [r.frame_index for r in aligned.frames] == [0, 207, 414]
    assert len(calls) == 3 and all((w, h) == (640, 480) for _, _, w, h in calls)


def test_depth_sampling_keeps_unknown_tracks_and_requires_four_object_neighbors():
    z = np.ones((3, 4, 5), np.float32) * 2
    xy = np.array([[[1.25, 1.25], [3.5, 2.5]], [[1.25, 1.25], [np.nan, np.nan]],
                   [[1.25, 1.25], [3.5, 2.5]]])
    visible = np.array([[True, True], [True, False], [True, True]])
    bank = dict(frame_index=np.arange(3), xy=xy, visible=visible)
    def masks(frame):
        person = np.zeros((4, 5), bool); obj = np.ones_like(person)
        if frame == 1: person[2, 2] = True
        return person, obj
    measured, support = runner.sampled_depth(z, bank, masks, 1.5)
    assert np.isnan(measured[1]).all() and not support[1].any()
    np.testing.assert_array_equal(support[[0, 2]], np.ones((2, 2), bool))
    np.testing.assert_allclose(measured[[0, 2]], 3.)
    np.testing.assert_array_equal(bank['visible'], visible)


def test_two_pose_variants_have_identical_depth_rgb_priors_and_budget(tmp_path, monkeypatch):
    bank = tiny_bank(); pool, ids = runner.real.bounded_pool(bank, protocol())
    depth = np.full(bank['visible'].shape, 2., np.float64); support = bank['visible'].copy()
    depth[~support] = np.nan; calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(rotations=args[4], translations=args[5], diagnostics={'converged': True})
    monkeypatch.setattr(runner, 'refine_sequence', fake)
    fit_cfg = replace(config(), max_nfev=300); cfg = protocol(); cfg['variants'] = list(runner.VARIANTS)
    for name in runner.VARIANTS:
        row = runner.fit_worker((name, bank, depth, support, pool, ids, cfg, fit_cfg, str(tmp_path), 'a' * 40))
        assert row['status'] == 'complete'
        with np.load(tmp_path / row['file'], allow_pickle=False) as output:
            np.testing.assert_array_equal(output['object_vertices'], bank['vertices'])
            np.testing.assert_array_equal(output['frame_index'], bank['frame_index'])
            np.testing.assert_array_equal(output['tracks_depth_m'], depth)
            np.testing.assert_array_equal(output['depth_visible'], support)
    assert len(calls) == 2
    assert all(args[10].max_nfev == 300 for args, _ in calls)
    for args, kw in calls:
        assert args[4] is bank['rotations'] and args[5] is bank['translations']
        assert kw['tracks_depth_m'] is depth and kw['depth_visible'] is support
        assert kw['depth_config'] is runner.real.saved.DEPTH_CFG
        assert 'depth_covariance_config' not in kw
    assert 'contact_evidence' not in calls[0][1]
    assert calls[1][1]['contact_evidence'] is pool
    assert calls[1][1]['contact_patch_config'].max_candidates == 8
    assert calls[1][1]['contact_distance_batch_size'] == 32


def test_worker_failure_seals_typed_receipt_without_claiming_fit(tmp_path, monkeypatch):
    bank = tiny_bank(); pool, ids = runner.real.bounded_pool(bank, protocol())
    def fail(*_a, **_kw): raise ValueError('fixed execution failure')
    monkeypatch.setattr(runner, 'refine_sequence', fail)
    row = runner.fit_worker((runner.VARIANTS[0], bank, np.ones(bank['visible'].shape), bank['visible'],
        pool, ids, protocol(), config(), str(tmp_path), 'a' * 40))
    assert row['status'] == 'fail' and row['error_type'] == 'ValueError'
    assert not list(tmp_path.glob('*.npz')) and (tmp_path / row['receipt_file']).is_file()


def test_quality_requires_both_converged_and_no_rgb_contact_or_motion_regression():
    metrics = {k: 1. for k in ('RGB_mean_px', 'selected_anatomical_triangle_mean_m',
        'same_J1_surface_gap_p95_m', 'acceleration_proxy_m_s2_p95',
        'angular_acceleration_proxy_rad_s2_p95', 'RGB_motion_increment_error_mean_px')}
    bank = {'original': dict(metrics), **{name: dict(metrics) for name in runner.VARIANTS}}
    converged = dict.fromkeys(runner.VARIANTS, True); cfg = protocol()
    assert runner.quality_decision(cfg, bank, converged, runner.VARIANTS[0])['passed']
    bank[runner.VARIANTS[0]]['RGB_motion_increment_error_mean_px'] *= 1.1
    assert not runner.quality_decision(cfg, bank, converged, runner.VARIANTS[0])['passed']
    converged[runner.VARIANTS[1]] = False
    assert not runner.quality_decision(cfg, bank, converged, runner.VARIANTS[1])['passed']


class Capture:
    def __init__(self, images): self.images = images; self.index = 0; self.released = False
    def get(self, prop):
        return {1: self.index, 5: 30., 7: len(self.images)}[prop]
    def read(self):
        if self.index >= len(self.images): return False, None
        result = self.images[self.index]; self.index += 1; return True, result
    def release(self): self.released = True


def test_decoder_full_original_indices_and_fixed_rgb_grid(monkeypatch):
    images = [np.full((4, 5, 3), i, np.uint8) for i in range(3)]; capture = Capture(images)
    resize_calls = []
    def resize(image, size, interpolation):
        resize_calls.append((size, interpolation)); return np.resize(image, (480, 640, 3))
    cv2 = SimpleNamespace(VideoCapture=lambda _: capture, CAP_PROP_FRAME_COUNT=7, CAP_PROP_POS_FRAMES=1,
        INTER_AREA=3, COLOR_BGR2RGB=4, resize=resize, cvtColor=lambda x, _: x[..., ::-1].copy())
    monkeypatch.setattr(runner.real.old, 'actual_fps', lambda *a: 30.)
    frames, hashes = runner.decode_video(cv2, '/fixture-only', (4, 5), frame_count=3)
    assert frames.shape == (3, 480, 640, 3) and len(set(hashes)) == 3
    assert resize_calls == [((640, 480), 3)] * 3 and capture.released


def test_no_sample_level_ground_truth_camera_or_manual_prompt_in_runner_source():
    from pathlib import Path
    text = (Path(__file__).parents[1] / 'infra/video_depth_pose_real.py').read_text()
    assert "'per_frame" not in text and 'ground_truth_used=False' in text
    assert 'ProcessPoolExecutor(max_workers=2' in text
    assert 'del depths' in text and 'depth_observations.npz' in text
    assert 'target.npy' in text and 'body_full' not in text
