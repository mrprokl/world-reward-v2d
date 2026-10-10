"""Integration ABI/gate tests only, no local CUDA/model/data/render transfer."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / 'infra/camera_render.py'
spec = importlib.util.spec_from_file_location('camera_render_capacity_test', PATH)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class Leaf:
    def __init__(self, array): self.array = array
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


def native_mock(monkeypatch):
    calls = []
    v = np.array([[[0, 0, 2], [.1, 0, 2], [0, .1, 2]]], np.float32)
    projected = types.SimpleNamespace(verts_padded=lambda: Leaf(v))
    mesh = types.SimpleNamespace(faces_list=lambda: [Leaf(np.array([[0, 1, 2]], np.int64))])
    class Settings:
        def __init__(self, **kw): self.options = kw
    class Rasterizer:
        def __init__(self, **kw): calls.append(('construct', kw))
        def transform(self, m):
            assert m is mesh
            calls.append(('transform', m)); return projected
        def __call__(self, m):
            assert m is mesh
            calls.append(('full', m))
            return types.SimpleNamespace(pix_to_face='ref_mask', zbuf='ref_z')
    def rasterize(m, **kw):
        assert m is projected
        calls.append(('rasterize', kw)); return 'opt_mask', 'opt_z', None, None
    p3d = types.ModuleType('pytorch3d')
    sub = types.ModuleType('pytorch3d.renderer')
    sub.MeshRasterizer, sub.RasterizationSettings = Rasterizer, Settings
    msub = types.ModuleType('pytorch3d.renderer.mesh'); msub.rasterize_meshes = rasterize
    for key, value in [('pytorch3d', p3d), ('pytorch3d.renderer', sub), ('pytorch3d.renderer.mesh', msub)]:
        monkeypatch.setitem(sys.modules, key, value)
    return mesh, calls


def test_explicit_optimization_native_transform_once_unchanged_raster_abi(monkeypatch):
    mesh, calls = native_mock(monkeypatch); info = {}
    result = renderer._raster_native(mesh, types.SimpleNamespace(get_znear=lambda: None), (96, 128), 1,
                                     full_face_capacity=False, capacity_diagnostics=info)
    assert result == ('opt_mask', 'opt_z')
    assert [c[0] for c in calls] == ['construct', 'transform', 'rasterize']
    kwargs = calls[-1][1]
    assert kwargs == dict(image_size=(96, 128), blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, clip_barycentric_coords=False, cull_backfaces=False,
        cull_to_frustum=False, z_clip_value=None, bin_size=16, max_faces_per_bin=1)
    assert info['native_NDC_transform'] is True


@pytest.mark.parametrize('mode', [True, None])
def test_reference_and_ungated_default_preserve_original_native_path(monkeypatch, mode):
    monkeypatch.delenv('WR_RASTER_CAPACITY_GATE', raising=False)
    mesh, calls = native_mock(monkeypatch)
    assert renderer._raster_native(mesh, types.SimpleNamespace(get_znear=lambda: None), (96, 128), 42,
        full_face_capacity=mode) == ('ref_mask', 'ref_z')
    assert [c[0] for c in calls] == ['construct', 'full']
    assert calls[0][1]['raster_settings'].options['max_faces_per_bin'] == 42


def receipt(tmp_path):
    path = tmp_path / 'gate.json'
    r = dict(schema='world_reward.raster_capacity_gate.v1', status='pass',
        source_pins=renderer._raster_source_pins(), image_id='sha256:fake',
        expected_pytorch3d_revision=renderer.PYTORCH3D_REVISION,
        gates=dict(all_exact_masks_and_depth_tolerance=True, external_batch_sizes=[1, 4]))
    return path, r


def write_receipt(path, record):
    path.write_text(json.dumps(record)); path.chmod(0o444)


def test_sealed_real_gate_only_enables_same_source_image(monkeypatch, tmp_path):
    path, record = receipt(tmp_path); write_receipt(path, record)
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE', str(path.resolve()))
    monkeypatch.setenv('WR_IMAGE_ID', 'sha256:fake')
    pin = renderer._checked_pin(path)
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE_BYTES', str(pin['bytes']))
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE_SHA256', pin['sha256'])
    assert renderer._optimized_capacity(None) is True
    assert renderer._optimized_capacity(True) is False


@pytest.mark.parametrize('changed', ['status', 'image_id', 'source_pins', 'expected_pytorch3d_revision', 'gates'])
def test_invalid_gate_cannot_silently_adopt(monkeypatch, tmp_path, changed):
    path, record = receipt(tmp_path); record[changed] = 'different'
    write_receipt(path, record); monkeypatch.setenv('WR_RASTER_CAPACITY_GATE', str(path.resolve()))
    monkeypatch.setenv('WR_IMAGE_ID', 'sha256:fake')
    pin = renderer._checked_pin(path)
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE_BYTES', str(pin['bytes']))
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE_SHA256', pin['sha256'])
    with pytest.raises((ValueError, AttributeError)):
        renderer._optimized_capacity(None)


def test_writable_or_symlink_gate_rejected(monkeypatch, tmp_path):
    path, record = receipt(tmp_path); path.write_text(json.dumps(record))
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE', str(path.resolve()))
    with pytest.raises(ValueError): renderer._optimized_capacity(None)
    path.chmod(0o444); link = tmp_path / 'alias.json'; link.symlink_to(path)
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE', str(link))
    with pytest.raises(ValueError): renderer._optimized_capacity(None)


def test_non_boolean_explicit_control_rejected():
    with pytest.raises(ValueError): renderer._optimized_capacity(0)


def test_missing_or_changed_explicit_receipt_pin_rejected(monkeypatch, tmp_path):
    path, record = receipt(tmp_path); write_receipt(path, record)
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE', str(path.resolve()))
    monkeypatch.setenv('WR_IMAGE_ID', 'sha256:fake')
    monkeypatch.delenv('WR_RASTER_CAPACITY_GATE_BYTES', raising=False)
    monkeypatch.delenv('WR_RASTER_CAPACITY_GATE_SHA256', raising=False)
    with pytest.raises(ValueError): renderer._optimized_capacity(None)
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE_BYTES', str(path.stat().st_size))
    monkeypatch.setenv('WR_RASTER_CAPACITY_GATE_SHA256', '0' * 64)
    with pytest.raises(ValueError): renderer._optimized_capacity(None)


def test_implicit_native_near_plane_clipping_cannot_change_optimized_faces(monkeypatch):
    mesh, calls = native_mock(monkeypatch)
    with pytest.raises(ValueError):
        renderer._raster_native(mesh, types.SimpleNamespace(get_znear=lambda: .1), (96, 128), 1,
                               full_face_capacity=False)
    assert not any(c[0] == 'rasterize' for c in calls)


def test_external_gate_rejects_unpinned_or_replaced_public_stage_before_mesh_import(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'trimesh', types.ModuleType('trimesh'))
    args = types.SimpleNamespace(root=tmp_path,
        external_object_report=tmp_path/'wrong/report.json', external_object_report_bytes=9165,
        external_object_report_sha256='0'*64)
    with pytest.raises(ValueError): renderer._load_external_capacity_geometry(args)


def test_procedural_gate_has_full_grid_four_rigid_poses_with_occlusion_and_all_faces():
    # No CUDA/render: validate only declared procedural geometry, not accuracy.
    pytest.importorskip('trimesh')
    vertices, faces, K = renderer._procedural_capacity_cases()
    assert vertices.shape[0] == 4 and len(faces) > 1000
    assert np.all(vertices[..., 2] > 1e-4)
    np.testing.assert_array_equal(K, [[1920., 0, 768.], [0, 1920., 576.], [0, 0, 1]])
    assert faces.max() < vertices.shape[1]


def test_scalar_batch_signatures_expose_explicit_reference_without_breaking_callers():
    import inspect
    for f in (renderer.raster_camera_mesh, renderer.raster_camera_mesh_batch):
        assert inspect.signature(f).parameters['full_face_capacity'].default is None


def test_gate_wrapper_preserves_offline_gpu_owned_durable_paths():
    wrapper = PATH.with_name('run_raster_capacity_gate.sh').read_text()
    assert '--network none --read-only' in wrapper and '--gpus all' in wrapper
    assert 'exec 8<' in wrapper and 'flock -n 8' in wrapper
    assert '--cidfile "$CID"' in wrapper and 'world_reward.raster_capacity.owner' in wrapper
    assert '--capacity-gate' in wrapper and '--signal=TERM --kill-after=10s 1203s' in wrapper
    assert 'eval_private' not in wrapper and 'track_1' not in wrapper and 'az ' not in wrapper
    assert 'readonly' in wrapper and '743576' not in wrapper  # Count frozen in Python gate.
