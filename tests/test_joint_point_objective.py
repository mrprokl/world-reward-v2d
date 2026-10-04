"""Own tiny contracts; optional REAL Torch gradients, never native model calls."""
from dataclasses import replace
import ast
import hashlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest

from world_reward import joint_point_objective as op


def evidence(frames=4, support=None):
    vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [0, .5, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.int64)
    bary = np.array([[.5, .25, .25], [.25, .5, .25]], np.float64)
    points = (vertices[faces[[0, 0]]].astype(float)*bary[:, :, None]).sum(1)
    k = np.array([[24., 0, 12.], [0, 24., 12.], [0, 0, 1.]])
    xy = (points[:, :2]/2*[24, 24]+[12, 12])*[256/24, 256/24]
    queries = np.c_[np.zeros(2), xy[:, 1]*24/256, xy[:, 0]*24/256]
    return op.FixedTriangleTracks(vertices, faces, np.zeros(2, np.int64), bary, k, (24, 24),
        np.arange(frames, dtype=np.int64), np.arange(11, 11+frames, dtype=np.int64),
        np.arange(2, dtype=np.int64), queries, np.broadcast_to(xy, (frames, 2, 2)).copy(),
        np.ones((frames, 2), bool) if support is None else support,
        tuple(f'frame_{i:06d}' for i in range(frames)),
        ('manufactured-only-source-not-a-qualified-producer',), op.FRAME_CONVENTION)


def config(weight=1.): return op.PointObjectiveConfig(4., weight, 'manufactured independent calibration reference only')


def test_config_has_no_silent_scale_weight_or_calibration_defaults():
    with pytest.raises(TypeError): op.PointObjectiveConfig()
    with pytest.raises(TypeError): op.PointObjectiveConfig(4., 1.)
    assert config().residual_scale_256_px == 4


@pytest.mark.parametrize('field,value', [('residual_scale_256_px', 0), ('residual_scale_256_px', True),
    ('residual_scale_256_px', np.nan), ('weight', -1), ('weight', False), ('weight', np.inf),
    ('calibration_reference', ''), ('calibration_reference', 'private\nsecret')])
def test_invalid_explicit_configuration_fails(field, value):
    with pytest.raises(ValueError): replace(config(), **{field: value})


def test_evidence_copies_readonly_bytes_and_preserves_every_slot_and_source_id():
    original = evidence(); copied = replace(original)
    assert copied.mesh_sha256 == original.mesh_sha256 and copied.evidence_sha256 == original.evidence_sha256
    assert copied.frame_index.tolist() == [0, 1, 2, 3] and copied.source_frame_ids.tolist() == [11, 12, 13, 14]
    for name, a in vars(copied).items():
        if isinstance(a, np.ndarray):
            assert not a.flags.writeable and not np.shares_memory(a, getattr(original, name))
            with pytest.raises(ValueError): a.flags.writeable = True


@pytest.mark.parametrize('fault', ['short', 'index', 'bool_ids', 'source_order', 'query_order', 'query_time',
    'query_outside', 'query_duplicate', 'track_nan', 'track_outside', 'invisible_outside', 'support_dtype',
    'mesh_dtype', 'faces_dtype', 'negative_face', 'bad_face', 'degenerate', 'duplicate_point', 'negative_bary',
    'bad_sum', 'skew_K', 'camera_bottom', 'grid', 'frame_convention', 'references', 'frame_names'])
def test_malformed_full_t_attachment_or_evidence_never_drops_or_repairs(fault):
    e = evidence(); values = {}
    if fault == 'short': values['frame_index'] = np.arange(2, dtype=np.int64)
    elif fault == 'index': values['frame_index'] = np.array([0, 1, 3, 4], np.int64)
    elif fault == 'bool_ids': values['source_frame_ids'] = np.ones(4, bool)
    elif fault == 'source_order': values['source_frame_ids'] = e.source_frame_ids[::-1]
    elif fault == 'query_order': values['query_ids'] = e.query_ids[::-1]
    elif fault.startswith('query_'):
        a = e.query_points.copy()
        if fault == 'query_time': a[0, 0] = 1
        elif fault == 'query_outside': a[0, 1] = 24
        else: a[1] = a[0]
        values['query_points'] = a
    elif fault in ('track_nan', 'track_outside', 'invisible_outside'):
        a = e.tracks_256.copy(); a[2, 0, 0] = np.nan if fault == 'track_nan' else 256
        values['tracks_256'] = a
        if fault == 'invisible_outside': values['native_visible'] = np.zeros((4, 2), bool)
    elif fault == 'support_dtype': values['native_visible'] = e.native_visible.astype(np.float32)
    elif fault == 'mesh_dtype': values['vertices'] = e.vertices.astype(np.float64)
    elif fault == 'faces_dtype': values['faces'] = e.faces.astype(np.int32)
    elif fault == 'negative_face': values['face_indices'] = np.array([-1, 0], np.int64)
    elif fault == 'bad_face': values['faces'] = np.array([[0, 1, 3]], np.int64)
    elif fault == 'degenerate': values['vertices'] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32)
    elif fault in ('duplicate_point', 'negative_bary', 'bad_sum'):
        a = e.barycentric.copy()
        if fault == 'duplicate_point': a[1] = a[0]
        elif fault == 'negative_bary': a[0] = [-.25, .5, .75]
        else: a[0] *= .5
        values['barycentric'] = a
    elif fault in ('skew_K', 'camera_bottom'):
        a = e.K.copy(); a[0, 1] = 1 if fault == 'skew_K' else 0
        if fault == 'camera_bottom': a[2, 0] = .1
        values['K'] = a
    elif fault == 'grid': values['image_size'] = (True, 24)
    elif fault == 'frame_convention': values['frame_convention'] = 'independent-world-gauge'
    elif fault == 'frame_names': values['native_frame_names'] = ('wrong',)*4
    else: values['source_references'] = ()
    with pytest.raises(ValueError): replace(e, **values)


def test_all_occluded_frames_remain_explicit_without_becoming_support():
    support = np.array([[1, 1], [1, 0], [0, 0], [0, 1]], bool)
    e = evidence(support=support)
    assert e.support_counts().tolist() == [1, 1] and not e.native_visible[2].any()
    unsupported = evidence(support=np.zeros((4, 2), bool))
    assert unsupported.support_counts().tolist() == [0, 0]  # Evidence can represent absence honestly.


def test_distinct_f64_barycentrics_cannot_weight_duplicate_native_f32_attachments():
    e = evidence(); bary = e.barycentric.copy(); bary[1] = bary[0]+[1e-10, -1e-10, 0]
    with pytest.raises(ValueError): replace(e, barycentric=bary)


def test_source_and_mesh_identities_include_byte_order_slots_gauge_and_reference():
    e = evidence(); changed = e.vertices.copy(); changed[0, 0] += .125
    assert replace(e, vertices=changed).mesh_sha256 != e.mesh_sha256
    assert replace(e, source_frame_ids=e.source_frame_ids+20).evidence_sha256 != e.evidence_sha256
    assert replace(e, source_references=('different-source',)).evidence_sha256 != e.evidence_sha256
    assert replace(e, barycentric=e.barycentric[::-1], query_points=e.query_points[::-1]).evidence_sha256 != e.evidence_sha256


def torch_inputs(e, *, requires_grad=False):
    torch = pytest.importorskip('torch', reason='No local Torch installed; native/autograd qualification remains pending')
    r = torch.eye(3, dtype=torch.float64).repeat(len(e.frame_index), 1, 1)
    t = torch.tensor(np.tile([0., 0., 2.], (len(e.frame_index), 1)), requires_grad=requires_grad)
    index = torch.arange(len(e.frame_index), dtype=torch.int64)
    return torch, r, t, index


def test_real_torch_analytic_translation_gradient_and_gradcheck():
    e = evidence(); torch, r, t, index = torch_inputs(e, requires_grad=True)
    t = (t+torch.tensor([.03, -.02, .1], dtype=t.dtype)).detach().requires_grad_(True)
    value = op.point_reprojection_loss(r, t, e, config(), index)
    value.backward()
    assert torch.isfinite(t.grad).all() and torch.count_nonzero(t.grad[1:]) > 0 and torch.count_nonzero(t.grad[0]) == 0
    assert torch.autograd.gradcheck(lambda x: op.point_reprojection_loss(r, x, e, config(), index), (t,), eps=1e-6)


def test_real_torch_track_equal_smooth_reference_with_occlusion():
    e = evidence(support=np.array([[1, 1], [1, 1], [1, 0], [0, 0]], bool)); torch, r, t, index = torch_inputs(e)
    tracks = e.tracks_256.copy(); tracks[1, 0, 0] += 4.; tracks[2, 0, 0] += 8.; tracks[1, 1, 1] += 12.
    e = replace(e, tracks_256=tracks)
    expected = ((np.sqrt(2)-1+np.sqrt(5)-1)/2+(np.sqrt(10)-1))/2
    assert op.point_reprojection_loss(r, t, e, config(), index).item() == pytest.approx(expected, abs=1e-12)


def test_real_torch_forced_first_observation_excluded_and_long_support_not_extra_weight():
    e = evidence(); torch, r, t, index = torch_inputs(e)
    changed = e.tracks_256.copy(); changed[0] += 3.
    assert op.point_reprojection_loss(r, t, replace(e, tracks_256=changed), config(), index).item() == 0
    assert op.point_reprojection_loss(r, t, e, config(0), index).item() == 0


@pytest.mark.parametrize('fault', ['nonpositive_invisible', 'nonproper', 'nonfinite', 'short', 'reordered', 'unsupported_track'])
def test_real_torch_invalid_geometry_fulltimeline_support_fails_without_drop(fault):
    e = evidence(); torch, r, t, index = torch_inputs(e)
    if fault == 'nonpositive_invisible':
        t[2, 2] = -1; e = replace(e, native_visible=np.array([[1, 1], [1, 1], [0, 0], [1, 1]], bool))
    elif fault == 'nonproper': r[2, 0, 0] = 2
    elif fault == 'nonfinite': t[2, 0] = float('nan')
    elif fault == 'short': r, t = r[:-1], t[:-1]
    elif fault == 'reordered': index = index.flip(0)
    else: e = replace(e, native_visible=np.array([[1, 1], [1, 0], [1, 0], [1, 0]], bool))
    with pytest.raises(ValueError): op.point_reprojection_loss(r, t, e, config(), index)


def test_real_torch_unit_similarity_translation_gradient_is_same_pixel_loss():
    e = evidence(); torch, r, t, index = torch_inputs(e)
    t[:, 0] = .02
    base = op.point_reprojection_loss(r, t, e, config(), index)
    scaled = replace(e, vertices=(e.vertices*4).astype(np.float32))
    assert op.point_reprojection_loss(r, t*4, scaled, config(), index).item() == pytest.approx(base.item(), abs=1e-14)


class FakeTensor:
    """Only namespace/identity test, NOT an autograd or numerical substitute."""
    def __init__(self, a): self.a = np.array(a, copy=True); self.device = 'cpu'; self.dtype = self.a.dtype; self.requires_grad = False
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


FAKE_SOURCE = '''from __future__ import annotations
def _torch():
    return T
class MHRParityPostOptimizer:
    def __init__(self, bundle, object_vertices, object_faces, cfg=None, *, mhr_layer):
        self.cfg = cfg
        self.frame_indices = np.arange(len(bundle['pr']['pose_abs']), dtype=np.int64)
        self.object_vertices = Tensor(object_vertices)
        self.object_faces = Tensor(object_faces)
        self.object_axis = Tensor(np.zeros((len(self.frame_indices), 3)))
        self.object_rotation_initial = Tensor(bundle['pr']['pose_abs'][:, :3, :3])
    def _object_rotation_for_indices(self, indices):
        return self.object_rotation_initial
    def _object_state(self, indices, *, include_surface=True):
        return self.object_rotation_initial, self.translation, None, None
    def loss(self, indices, step, *, include_diagnostics=True):
        return BASE, METRICS
    def run(self):
        total, metrics = self.loss(T.arange(len(self.frame_indices)), 0)
        return {'postopt': {'total': total, 'metrics': metrics}}
'''


def native_fixture(tmp_path, monkeypatch):
    path = tmp_path/'tiny_native.py'; path.write_text(FAKE_SOURCE)
    module = ModuleType('manufactured_native_only'); module.__file__ = str(path)
    module.np = np; module.Tensor = FakeTensor; module.BASE = object(); module.METRICS = {'native-only': object()}
    module.T = SimpleNamespace(int64=np.dtype(np.int64), is_tensor=lambda x:isinstance(x, FakeTensor),
        arange=lambda n, device=None:FakeTensor(np.arange(n, dtype=np.int64)), equal=lambda x,y:np.array_equal(x.a, y.a))
    exec(compile(FAKE_SOURCE, str(path), 'exec', dont_inherit=True), vars(module))
    monkeypatch.setattr(op, 'NATIVE_SOURCE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    return module


def native_args(e):
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (len(e.frame_index), 4, 4)).copy()
    cfg = SimpleNamespace(num_steps=300, batch_size=0, frame_start=0, frame_limit=0,
        freeze_object_rotation=True, freeze_body_internal_translations=True)
    return {'frames': list(e.native_frame_names), 'pr': {'pose_abs': poses}}, cfg


def test_explicit_native_subclass_zero_weight_returns_exact_native_loss_objects(tmp_path, monkeypatch):
    e = evidence(); native = native_fixture(tmp_path, monkeypatch); bundle, cfg = native_args(e)
    original = native.MHRParityPostOptimizer
    cls = op.native_point_optimizer_class(native, e, config(0))
    model = cls(bundle, e.vertices, e.faces, cfg, mhr_layer=object())
    total, metrics = model.loss(native.T.arange(4), 0)
    assert total is native.BASE and metrics is native.METRICS and native.MHRParityPostOptimizer is original
    result = op.run_joint_point_refinement(native, bundle, e.vertices, e.faces, cfg,
        mhr_layer=object(), evidence=e, point_config=config(0))
    assert result['postopt']['total'] is native.BASE and result['postopt']['metrics'] is native.METRICS
    assert result['postopt']['point_objective']['quality_verified'] is False


@pytest.mark.parametrize('fault', ['source_sha', 'method_patch', 'foreign_globals', 'torch_patch', 'class_module'])
def test_native_source_live_method_and_namespace_binding_fails_closed(tmp_path, monkeypatch, fault):
    native = native_fixture(tmp_path, monkeypatch)
    if fault == 'source_sha': Path(native.__file__).write_text(FAKE_SOURCE+'\n# changed')
    elif fault == 'method_patch': native.MHRParityPostOptimizer.loss = lambda self, *a, **k: 0
    elif fault == 'foreign_globals':
        f = native.MHRParityPostOptimizer.loss
        import types
        native.MHRParityPostOptimizer.loss = types.FunctionType(f.__code__, dict(vars(native)))
    elif fault == 'torch_patch': native._torch = lambda: None
    else: native.MHRParityPostOptimizer.__module__ = 'foreign'
    with pytest.raises(ValueError): op.native_point_optimizer_class(native, evidence(), config(0))


@pytest.mark.parametrize('fault', ['rotation_unlocked', 'subset', 'window_batch', 'mesh', 'unsupported_track', 'indices', 'frame_names'])
def test_extension_refuses_extra_parameters_subset_mesh_or_unsupported_fit(tmp_path, monkeypatch, fault):
    e = evidence(); native = native_fixture(tmp_path, monkeypatch); bundle, cfg = native_args(e)
    if fault == 'rotation_unlocked': cfg.freeze_object_rotation = False
    elif fault == 'subset': cfg.frame_start = 1
    elif fault == 'window_batch': cfg.batch_size = 2
    elif fault == 'unsupported_track': e = evidence(support=np.zeros((4, 2), bool))
    elif fault == 'frame_names': bundle['frames'] = list(reversed(bundle['frames']))
    cls = op.native_point_optimizer_class(native, e, config(1 if fault == 'unsupported_track' else 0))
    vertices = e.vertices.copy()
    if fault == 'mesh': vertices[0, 0] += .125
    if fault == 'indices':
        model = cls(bundle, vertices, e.faces, cfg, mhr_layer=object())
        with pytest.raises(ValueError): model.loss(FakeTensor([0, 2, 1, 3]), 0)
    else:
        with pytest.raises(ValueError): cls(bundle, vertices, e.faces, cfg, mhr_layer=object())


def test_native_file_and_evidence_rehashed_before_and_after_run(tmp_path, monkeypatch):
    e = evidence(); native = native_fixture(tmp_path, monkeypatch); bundle, cfg = native_args(e)
    cls = op.native_point_optimizer_class(native, e, config(0)); model = cls(bundle, e.vertices, e.faces, cfg, mhr_layer=object())
    Path(native.__file__).write_text(FAKE_SOURCE+'\n# changed')
    with pytest.raises(ValueError): model.run()


def test_positive_extension_adds_to_same_native_loss_not_disconnected_fit(tmp_path, monkeypatch):
    e = evidence(); native = native_fixture(tmp_path, monkeypatch); native.BASE = 1.5
    bundle, cfg = native_args(e)
    cls = op.native_point_optimizer_class(native, e, config()); model = cls(bundle, e.vertices, e.faces, cfg, mhr_layer=object())
    model.translation = FakeTensor(np.tile([0., 0., 2.], (4, 1)))
    calls = []
    def manufactured_loss(r, t, ev, pcfg, indices):
        calls.append((r, t, ev, pcfg, indices)); return .25
    monkeypatch.setattr(op, 'point_reprojection_loss', manufactured_loss)
    total, metrics = model.loss(native.T.arange(4), 12, include_diagnostics=False)
    assert total == 1.75 and metrics['loss_total'] == 1.75 and metrics['loss_point_reprojection'] == .25
    assert len(calls) == 1 and calls[0][0] is model.object_rotation_initial and calls[0][1] is model.translation
    assert calls[0][2] is e and native.METRICS.keys() == {'native-only'}


def test_native_configuration_cannot_change_after_constructor(tmp_path, monkeypatch):
    e = evidence(); native = native_fixture(tmp_path, monkeypatch); bundle, cfg = native_args(e)
    cls = op.native_point_optimizer_class(native, e, config(0)); model = cls(bundle, e.vertices, e.faces, cfg, mhr_layer=object())
    cfg.num_steps = 200
    with pytest.raises(ValueError): model.loss(native.T.arange(4), 0)


def test_module_has_no_torch_import_until_loss_no_io_of_model_or_labels():
    tree = ast.parse(inspect.getsource(op))
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
        ('torch' in [a.name for a in n.names] or getattr(n, 'module', '') == 'torch') for n in tree.body)
    source = inspect.getsource(op)
    assert 'super().loss(' in source and 'super().run()' in source
    assert 'torch.load' not in source and 'np.load' not in source and 'setattr(native' not in source
    assert op.NATIVE_SOURCE_SHA256 == '84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b'
