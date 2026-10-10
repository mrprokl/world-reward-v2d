"""Execute the actual pose_initializer with tiny mock raster/image evidence."""
import copy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'infra'))
import form_hoi_external_predict as predictor


def run_mocked(monkeypatch, *, unique_fit):
    """All96 frames/all25 slots, no decoding/GT/inference/rendering locally."""
    import world_reward.exact_mesh_dedup as dedup
    import world_reward.rigid_alignment as alignment
    captured = []; batches = []; plans = []
    trimesh = ModuleType('trimesh')
    trimesh.Trimesh = lambda v, f, process: SimpleNamespace(centroid=np.zeros(3))
    trimesh.sample = SimpleNamespace(sample_surface=lambda mesh, n, seed: (np.ones((40, 3)), None))
    monkeypatch.setitem(sys.modules, 'trimesh', trimesh)
    class FakeNPZ:
        def __enter__(self): return {'depth': np.ones((8, 8)), 'mask': np.ones((8, 8), bool)}
        def __exit__(self, *_): pass
    monkeypatch.setattr(predictor.np, 'load', lambda *a, **kw: FakeNPZ())
    monkeypatch.setattr(predictor, 'mask', lambda *a: np.ones((8, 8), bool))
    points = np.zeros((8, 8, 3)); points[..., 2] = 2
    monkeypatch.setattr(predictor, 'pointmap', lambda *a: points)
    class FakeRotation:
        @staticmethod
        def create_group(name):
            assert name == 'O'
            # All24 are legal proper rotations, with some repeated deliberately
            # to exercise reuse while preserving ALL original hypothesis slots.
            return SimpleNamespace(as_matrix=lambda: np.repeat(np.eye(3)[None], 24, axis=0))
    monkeypatch.setattr(sys.modules['scipy.spatial.transform'], 'Rotation', FakeRotation)
    fit_count = [0]
    def fit(sampled, observed, r, t):
        dx = (fit_count[0] % 25) * .001 if unique_fit == 'vary' else .1 if unique_fit else 0.
        fit_count[0] += 1
        ft = t + [dx, 0., 0.]
        return SimpleNamespace(rotation=r.copy(), translation=ft, final_residual=.4, initial_residual=.5)
    monkeypatch.setattr(alignment, 'align_observed_points', fit)
    renderer = ModuleType('camera_render')
    class Rendered:
        def __init__(self, geometry): self.geometry = geometry
        def cpu(self): return self
        def numpy(self): return self.geometry
    def render(batch, faces, K, width, height):
        assert width == height == 8 and 1 <= len(batch) <= 4
        batches.append(batch.copy())
        return [Rendered(x) for x in batch], None
    renderer.raster_camera_mesh_batch = render
    renderer.silhouette_iou = lambda geometry, mask: float(geometry[0, 0] + .5)
    monkeypatch.setitem(sys.modules, 'camera_render', renderer)
    selection = ModuleType('object_pose_smoke'); selection._finite_pose_candidate = lambda row: True
    def select(records, indices, hypotheses, centroid):
        assert indices == list(range(96)) and hypotheses == 25
        captured.extend(copy.deepcopy(records))
        return np.repeat(np.eye(3)[None], 96, axis=0), np.zeros((96, 3)), np.ones(96, bool), {}
    selection._select_latent_pose_path = select
    monkeypatch.setitem(sys.modules, 'object_pose_smoke', selection)
    original = dedup.deduplicate_meshes
    def plan(meshes):
        result = original(meshes); plans.append(result); return result
    monkeypatch.setattr(dedup, 'deduplicate_meshes', plan)
    vertices = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]])
    result = predictor.pose_initializer(vertices, np.array([[0, 1, 2]]), 1., np.eye(3),
        np.array([0., 0., 2.]), dict(width=8, height=8), Path('/fake'), np.eye(3))
    return result, captured, batches, plans


@pytest.mark.parametrize('unique_fit', [False, True])
def test_actual_initializer_restores_all25_initial_fitted_pairs_all96(monkeypatch, unique_fit):
    result, records, batches, plans = run_mocked(monkeypatch, unique_fit=unique_fit)
    assert len(records) == len(plans) == 96
    assert all(len(r['candidates']) == 25 and r['pose_observed'] is True for r in records)
    for frame in records:
        for slot, row in enumerate(frame['candidates']):
            assert row['hypothesis_index'] == slot
            assert row['initial_silhouette_iou'] == .5
            assert row['fitted_silhouette_iou'] == (.6 if unique_fit else .5)
            assert row['selected_silhouette_iou'] == (.6 if unique_fit else .5)
            assert row['translation'][0] == (.1 if unique_fit else 0.)
    unique = 2 if unique_fit else 1
    assert sum(len(b) for b in batches) == 96*unique  # Not 96*50, without dropping a slot.
    stats = result[3]['exact_raster_reuse']
    assert stats['original_meshes'] == 96*50 and stats['unique_meshes'] == 96*unique
    assert stats['reused_meshes'] == 96*(50-unique) and stats['observed_frames'] == 96
    assert result[0].shape == (96, 3, 3) and result[1].shape == (96, 3)
    assert result[0].dtype == result[1].dtype == np.float32


def test_required_source_closure_declares_exact_dedup_helper():
    assert 'src/world_reward/exact_mesh_dedup.py' in predictor.HELPERS


def test_distinct_fitted_scores_preserve_pairing_and_odd25_tail(monkeypatch):
    result, records, batches, plans = run_mocked(monkeypatch, unique_fit='vary')
    for frame in records:
        for slot, row in enumerate(frame['candidates']):
            assert row['initial_silhouette_iou'] == .5
            assert row['fitted_silhouette_iou'] == .5 + slot*.001
            assert row['selected_silhouette_iou'] == .5 + slot*.001
            assert row['translation'][0] == slot*.001
    assert all(len(p.unique) == 25 for p in plans)
    assert sum(len(b) for b in batches) == 96*25
    assert result[3]['exact_raster_reuse']['original_meshes'] == 96*50
