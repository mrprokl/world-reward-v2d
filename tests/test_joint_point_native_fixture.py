"""Actual pinned native code + REAL Torch, with explicitly manufactured state.

Azure caller provides ONLY its authenticated original source path. No constructor,
body model, renderer, contact kernel, asset or external RGB/label is exercised.
Whitelisted AST scopes use their FULL-source original CodeTypes, without running
native imports. This is API/autograd qualification, not native HOI.
"""
import ast
import builtins
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import sys
import types

import numpy as np
import pytest

from world_reward import joint_point_objective as op

SCOPES = ('_torch', '_as_tensor', '_to_numpy', '_zero_like_tensor',
    '_public_acceleration_loss', 'pose_matrix', '_layer_output_vertices',
    '_layer_output_joints', '_layer_output_keypoints', '_layer_output_coco17',
    '_should_run_postopt_diagnostics', 'torch_cat', '_contact_activation_summary', 'MHRParityPostOptConfig',
    'MHRParityPostOptimizer')


def original_scopes(raw, path):
    """Compile whole source, execute ONLY explicitly allowed original code objects.

    Compiling a reduced AST changes CPython 3.11 import-binding optimizations in
    class methods; using original CodeTypes keeps the production bytecode gate.
    """
    assert len(raw) == 92824 and hashlib.sha256(raw).hexdigest() == op.NATIVE_SOURCE_SHA256
    tree = ast.parse(raw)
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in SCOPES}
    assert set(nodes) == set(SCOPES) and not any(n.decorator_list for n in nodes.values() if isinstance(n, ast.FunctionDef))
    codes = {c.co_name: c for c in compile(raw, str(path), 'exec', dont_inherit=True).co_consts if isinstance(c, types.CodeType)}
    module = types.ModuleType('manufactured_actual_native_scopes')
    module.__file__ = str(path)
    # Constants/state below are manufactured, not fabricated production metadata.
    vars(module).update(np=np, copy=copy, json=json, asdict=asdict, dataclass=dataclass,
        Path=Path, Mapping=Mapping, Any=Any, MHRLayerOutput=type('UnusedModelOutput', (), {}),
        MHR_POSTOPT_FULL_CLIP_BATCH_SIZE=0, MHR_POSTOPT_FULL_CLIP_BATCH_SAMPLING='full_clip_v1',
        MHR_POSTOPT_BATCH_SAMPLING='shuffled_contiguous_window_epoch_v1',
        MHR_COLLISION_PROXY_MODE_4000='manufactured_no_proxy', MHR_COLLISION_PROXY_MODES=('manufactured_no_proxy',),
        DEFAULT_MHR_COLLISION_PROXY_ASSET=Path('/never-read-manufactured-proxy'),
        DEFAULT_HAND_SURFACE_SPEC=Path('/never-read-manufactured-hand'),
        MHR_DEFAULT_CONTACT_ACTIVATION_DISTANCE_M=.05, MHR_DEFAULT_PENETRATION_WEIGHT=2.,
        MHR_BODY_ROTATION_CONTROL_COUNT=2, BODY_CONT_INTERNAL_TRANSLATION_SLICE=slice(2, 3),
        MHR_FOOT_DIAGNOSTIC_VERTICES_PER_SIDE=2, MHR_PARITY_POSTOPT_CHECKPOINT_SCHEMA='manufactured-no-checkpoint',
        OBJECT_MESH_SCENE_LOADER_REVISION='source-unused-manufactured-state',
        MHR_CONTACT_ACTIVATION_REVISION='network-active-and-initial-surface-distance-lt-v1',
        POSTOPT_CROP_CONTRACT='manufactured-no-crop-or-render', POSTOPT_RENDER_SIZE=256)
    previous = sys.modules.get(module.__name__); sys.modules[module.__name__] = module
    try:
        for name in SCOPES:
            node = nodes[name]
            assert not codes[name].co_freevars
            if isinstance(node, ast.FunctionDef):
                function = types.FunctionType(codes[name], vars(module), name,
                    tuple(ast.literal_eval(d) for d in node.args.defaults) or None)
                function.__kwdefaults__ = {a.arg: ast.literal_eval(d) for a, d in
                    zip(node.args.kwonlyargs, node.args.kw_defaults) if d is not None} or None
                setattr(module, name, function)
            else:
                assert not node.bases and not node.keywords
                assert [ast.dump(d) for d in node.decorator_list] == (["Name(id='dataclass', ctx=Load())"]
                    if name == 'MHRParityPostOptConfig' else [])
                cls = builtins.__build_class__(types.FunctionType(codes[name], vars(module)), name)
                setattr(module, name, dataclass(cls) if name == 'MHRParityPostOptConfig' else cls)
    finally:
        if previous is None: sys.modules.pop(module.__name__, None)
        else: sys.modules[module.__name__] = previous
    return module


def load_actual_native_scope():
    value = os.environ.get('WR_NATIVE_OPTIMIZER_SOURCE_PATH')
    if value is None: pytest.skip('Actual pinned source not supplied; no local native runtime claim')
    torch = pytest.importorskip('torch', reason='Real Torch required; no fake numerical substitute')
    path = Path(value)
    assert path.is_absolute() and '..' not in path.parts and str(path) == str(path.resolve()) and not path.is_symlink()
    raw = path.read_bytes()
    module = original_scopes(raw, path)
    # The production factory checks these live class methods against FULL source.
    op._native_binding(module)
    return torch, module, raw


def manufactured_evidence():
    v = np.array([[-.25, -.25, 0], [.25, -.25, 0], [0, .25, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int64)
    b = np.array([[.5, .25, .25], [.25, .5, .25]], np.float64)
    p = (v[f[[0, 0]]]*b[:, :, None]).sum(1)
    xy = (p[:, :2]/2*64+[32., 32.])*4
    return op.FixedTriangleTracks(v, f, np.array([0, 0], np.int64), b,
        np.array([[64., 0, 32.], [0, 64., 32.], [0, 0, 1.]]), (64, 64),
        np.arange(4, dtype=np.int64), np.arange(7, 11, dtype=np.int64), np.arange(2, dtype=np.int64),
        np.c_[np.zeros(2), xy[:, 1]/4, xy[:, 0]/4], np.broadcast_to(xy, (4, 2, 2)).copy(),
        np.ones((4, 2), bool), tuple(f'manufactured_frame_{i}' for i in range(4)),
        ('manufactured-no-model-no-external-calibration',), op.FRAME_CONVENTION)


def manufactured_instance(torch, native, cls, evidence):
    # Deliberately bypass __init__: it needs real PyTorch3D/body/spec assets.
    state = cls.__new__(cls)
    state.cfg = native.MHRParityPostOptConfig(device='cpu', w_contact=0., w_silhouette=0.,
        w_penetration=0., w_human_pose_prior=1., w_temporal=0., w_object_translation_prior=0.,
        num_steps=300, batch_size=0, report_every=100, checkpoint_path=None).checked()
    state._point_native_config = asdict(state.cfg)
    state.device, state.dtype = torch.device('cpu'), torch.float32
    state.frame_indices = np.arange(4, dtype=np.int64)
    state.object_vertices = torch.tensor(evidence.vertices.copy())
    state.object_faces = torch.tensor(evidence.faces.copy())
    state.object_surface = state.object_vertices.clone()
    state.penetration_surface = state.object_vertices.clone()
    state.object_rotation_initial = torch.eye(3).repeat(4, 1, 1)
    state.object_axis = torch.zeros((4, 3))
    state.object_translation = torch.tensor(np.tile([.03, -.02, 2.], (4, 1)), dtype=torch.float32, requires_grad=True)
    state.object_translation_initial = state.object_translation.detach().clone()
    state.body_pose_initial = torch.zeros((4, 3))
    state.body_pose_rotation_initial = state.body_pose_initial[:, :2].clone()
    state.body_pose = state.body_pose_rotation_initial.clone().requires_grad_(True)
    state.params_fixed = {'mhr_body_pose_cont': state.body_pose_initial}
    state.hand_surface_vertex_indices = torch.tensor([[0], [1]], dtype=torch.int64)
    state.network_contact_mask = torch.zeros((4, 2))
    state.contact_mask = torch.zeros((4, 2))
    state.initial_contact_distances_m = None
    state.initial_contact_proximity_mask = torch.zeros((4, 2), dtype=torch.bool)

    class ManufacturedMHR:
        def mhr_forward(self, params):
            controls = params['mhr_body_pose_cont']
            # Connected manufactured joints/vertices, not a body/hand model.
            delta = torch.nn.functional.pad(controls[:, :2], (0, 1))[:, None]
            vertices = torch.tensor([[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.], [0., 0., 1.1]])[None]+delta
            return dict(vertices=vertices, joints=vertices, keypoints=vertices,
                coco17=vertices[:, :1].expand(-1, 17, -1))

    state.mhr_layer = ManufacturedMHR()
    native.MHRParityPostOptimizer._initialize_frozen_object_geometry_cache(state)
    native.MHRParityPostOptimizer._initialize_pose_diagnostics(state)
    state.optimizer = torch.optim.Adam(state._optimizer_parameter_groups(), lr=state.cfg.lr)
    state.scheduler = torch.optim.lr_scheduler.LambdaLR(state.optimizer,
        lambda step: step/state.cfg.warmup_steps if step < state.cfg.warmup_steps else
        max(0., .5*(1+np.cos(np.pi*(step-state.cfg.warmup_steps)/(state.cfg.schedule_steps-state.cfg.warmup_steps)))))
    state.rng = np.random.default_rng(0)
    state.batch_start_order = np.empty(0, dtype=np.int64); state.batch_start_cursor = 0
    state.start_step = 0; state.history = []
    pose = native.pose_matrix(state.object_rotation_initial, state.object_translation.detach()).numpy()
    state.bundle = {'frames': list(evidence.native_frame_names), 'pr': {'pose_abs': pose,
        'mhr_body_pose_cont': state.body_pose_initial.numpy().copy()}}
    state.pr = state.bundle['pr']
    return state


@pytest.fixture
def actual_native():
    torch, native, original_raw = load_actual_native_scope()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield torch, native, original_raw
    finally:
        torch.set_num_threads(previous_threads)


def test_actual_native_scopes_super_loss_object_state_and_original_301_updates(actual_native):
    torch, native, original_raw = actual_native
    e = manufactured_evidence()
    zero = op.PointObjectiveConfig(4., 0., 'manufactured data-free NOT a validation-calibrated production config')
    positive = op.PointObjectiveConfig(4., 1., 'manufactured data-free NOT a validation-calibrated production config')
    baseline = manufactured_instance(torch, native, native.MHRParityPostOptimizer, e)
    zero_instance = manufactured_instance(torch, native, op.native_point_optimizer_class(native, e, zero), e)
    guided = manufactured_instance(torch, native, op.native_point_optimizer_class(native, e, positive), e)
    indices = torch.arange(4)
    a, am = baseline.loss(indices, 0, include_diagnostics=True)
    z, zm = zero_instance.loss(indices, 0, include_diagnostics=True)
    assert torch.equal(a, z) and set(am) == set(zm)
    assert all(torch.equal(am[k], zm[k]) for k in am)
    before = op.point_reprojection_loss(guided.object_rotation_initial, guided.object_translation, e, positive, indices)
    total, metrics = guided.loss(indices, 0, include_diagnostics=True)
    assert torch.equal(total, metrics['loss_total']) and 'loss_point_reprojection' in metrics
    total.backward()
    assert guided.object_translation.grad[1:, :2].abs().sum() > 0
    assert torch.isfinite(guided.object_translation.grad).all()
    guided.optimizer.zero_grad(set_to_none=True)
    baseline_result = baseline.run(); zero_result = zero_instance.run(); guided_result = guided.run()
    assert torch.equal(baseline.object_translation, zero_instance.object_translation)
    assert torch.equal(baseline.body_pose, zero_instance.body_pose)
    assert torch.equal(guided.body_pose, guided.body_pose_rotation_initial)
    assert torch.equal(guided._body_pose_for_indices(indices)[:, 2:], guided.body_pose_initial[:, 2:])
    assert len(guided.optimizer.state[guided.object_translation]) > 0
    assert int(guided.optimizer.state[guided.object_translation]['step']) == 301
    assert [r['iter'] for r in guided.history] == [0., 100., 200., 300.]
    assert all(r['batch_start'] == 0. and r['batch_size'] == 4. for r in guided.history)
    after = op.point_reprojection_loss(guided.object_rotation_initial, guided.object_translation, e, positive, indices)
    assert torch.isfinite(after) and after < before
    assert torch.equal(guided.object_rotation_initial, baseline.object_rotation_initial)
    assert np.array_equal(baseline_result['pr']['pose_abs'], zero_result['pr']['pose_abs'])
    assert np.array_equal(guided_result['pr']['pose_abs'][0], baseline_result['pr']['pose_abs'][0])
    assert guided_result['postopt']['fixed_parameters'][0] == 'object_rotation'
    assert guided_result['postopt']['frame_indices'] == list(range(4))
    assert guided_result['postopt']['point_objective']['quality_verified'] is False
    assert Path(native.__file__).read_bytes() == original_raw
    assert guided_result['postopt']['optimized_parameters'] == ['object_translation', 'mhr_body_pose_cont_rotation_controls']
