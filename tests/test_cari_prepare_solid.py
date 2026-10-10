"""Procedural, data-free solid preparation contracts; no native prep or media."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'infra'))
    spec = importlib.util.spec_from_file_location('test_solid_prepare', REPO / 'infra/cari_prepare.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tetra():
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], np.float64)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int64)
    return v, f


def packed():
    v, f = tetra()
    return np.concatenate((v, np.repeat(v[:1], 4092, axis=0))), np.concatenate((f, np.zeros((4092, 3), np.int64)))


def test_parser_legacy_default_and_explicit_solid(prepare):
    assert prepare._argument_parser().parse_args([]).mesh_source == 'default'
    for episode in range(30):
        p = prepare._argument_parser().parse_args(['--episode', str(episode), '--mesh-source', 'solid'])
        assert p.episode == episode and p.mesh_source == 'solid'
    for mode in ('volume', 'conditioned', 'oracle', '../solid'):
        with pytest.raises(SystemExit): prepare._argument_parser().parse_args(['--mesh-source', mode])


def test_zero_padding_only_no_processing_or_mutation(prepare):
    v, f = packed(); old = v.tobytes(), f.tobytes()
    active, (cv, cf), top = prepare._solid_compact(v, f, np)
    assert active.tolist() == [0, 1, 2, 3]
    assert np.array_equal(cv, tetra()[0]) and np.array_equal(cf, tetra()[1])
    assert top['components'][0]['euler'] == 2 and top['components'][0]['volume_sign'] == 1
    assert old == (v.tobytes(), f.tobytes())


@pytest.mark.parametrize('fault', ['dtype', 'index', 'repeated', 'orphan', 'collapsed_f32', 'quantization_bound'])
def test_no_tolerance_repair_or_hidden_padding(prepare, fault):
    v, f = packed()
    if fault == 'dtype': v = v.astype(np.float32)
    elif fault == 'index': f[0, 0] = -1
    elif fault == 'repeated': f[4] = [0, 0, 1]
    elif fault == 'orphan': v[-1] = [.5, .5, .5]
    elif fault == 'collapsed_f32':
        v[1] = [1., 1., 1.]; v[2] = [1. + 2 ** -26, 1., 1.]; v[3] = [1., 1. + 2 ** -26, 1.]
    elif fault == 'quantization_bound': v[1, 0] = 1000.000023
    with pytest.raises(ValueError): prepare._solid_compact(v, f, np)


def test_disconnected_cavity_orientations_counted_without_drop(prepare):
    v, f = tetra(); outer = v * 8; inner = v + [.5, .5, .5]
    vertices = np.concatenate((outer, inner, np.repeat(outer[:1], 4088, axis=0)))
    faces = np.concatenate((f, f[:, ::-1] + 4, np.zeros((4088, 3), np.int64)))
    _, _, topology = prepare._solid_compact(vertices, faces, np)
    assert sorted(c['volume_sign'] for c in topology['components']) == [-1, 1]
    assert topology['faces'] == 8


@pytest.fixture
def source_case(prepare, tmp_path, monkeypatch):
    import solid_geometry_loader as loader
    code = tmp_path / 'code'; root = tmp_path / 'root'; episode = 9
    base = root / 'outputs/episode_000009'; pin_path = code / 'configs/solid_mesh_000009_pins.json'
    pin_path.parent.mkdir(parents=True); pin_path.write_text('{"synthetic_only":true}'); pin_path.chmod(0o444)
    canonical = base / 'object_budget_solid_fake/object_fixed_canonical.glb'
    canonical.parent.mkdir(parents=True); canonical.write_bytes(b'opaque synthetic canonical bytes'); canonical.chmod(0o444)
    pose_dir = base / 'object_pose_full_solid'; pose_dir.mkdir()
    copied = pose_dir / 'object_fixed_canonical.glb'; copied.write_bytes(canonical.read_bytes()); copied.chmod(0o444)
    obj, alignment, transform = base / 'object_grounded/report.json', base / 'scale_smoke/report.json', base / 'object_grounded/transform.json'
    for path in (obj, alignment, transform): path.parent.mkdir(parents=True, exist_ok=True)
    obj.write_text('{"opaque":"source"}'); alignment.write_text('{"opaque":"gauge"}')
    transform.write_text('{"scale":[0.375,0.375,0.375]}')
    v, f = packed(); active = np.arange(4, dtype=np.int64)
    pose = pose_dir / 'geometry_and_poses.npz'
    arrays = dict(vertices=v, faces=f, frame_index=np.arange(3, dtype=np.int64),
                  rotation=np.repeat(np.eye(3)[None], 3, axis=0), translation=np.zeros((3, 3)), object_scale=np.array(1.))
    np.savez(pose, **arrays); pose.chmod(0o444)
    receipt = dict(cpu_report_sha256='1' * 64, native_report_sha256='2' * 64,
                   cpu_producer_revision='3' * 40, cpu_script_sha256='4' * 64)
    report = dict(mesh_source='solid', execution_verified=True, original_frame_coverage_verified=True, fixed_shape=True,
                  object_report_sha256=prepare.sha256(obj), geometry_and_poses_sha256=prepare.sha256(pose),
                  fixed_canonical_mesh_sha256=prepare.sha256(canonical),
                  topology_budget={**receipt, 'committed_pins_sha256': prepare.sha256(pin_path)})
    report_path = pose_dir / 'report.json'; report_path.write_text(json.dumps(report)); report_path.chmod(0o444)
    pose_dir.chmod(0o555)
    for name in {*loader.SOURCE_HELPERS, 'infra/cari_prepare.py', 'infra/solid_geometry_loader.py',
                 'infra/mesh_precision_diagnostic.py', 'src/world_reward/mesh_geometry.py',
                 'infra/cari_wrapper_common.sh', 'infra/run_cari_prepare.sh', 'src/world_reward/artifact_paths.py'}:
        path = code / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fake pure source'); path.chmod(0o444)
    monkeypatch.setattr(prepare, '__file__', str(code / 'infra/cari_prepare.py'))
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs)); return v, f, active, {}, canonical, receipt
    monkeypatch.setattr(loader, 'load', load)
    return SimpleNamespace(root=root, code=code, episode=episode, inputs=dict(video_sha256='5' * 64, total_frames=3),
                           pose=pose, pin=pin_path, report=report, arrays=arrays, v=v, f=f, calls=calls,
                           canonical=canonical, copied=copied, transform=transform, report_path=report_path)


def invoke(prepare, case):
    return prepare._solid_preflight(case.root, case.episode, case.inputs, case.report, case.pose, np)


def test_independent_pins_exact_geometry_and_gauge_before_output(prepare, source_case):
    c = source_case; result = invoke(prepare, c)
    assert np.array_equal(result[0], c.v) and np.array_equal(result[1], c.f)
    assert c.calls[0][0][0:3] == (c.root, 9, c.inputs['video_sha256'])
    assert c.calls[0][0][-1] == .375 and c.calls[0][1]['pins'] == {'synthetic_only': True}
    assert not (c.root / 'outputs/episode_000009/cari_inputs').exists()
    prepare._solid_recheck(result[-1])


@pytest.mark.parametrize('fault', ['missing_pins', 'wrong_pin', 'wrong_route', 'wrong_canonical', 'copy_changed',
                                 'wrong_producer', 'shape_changed', 'wrong_scale', 'frame_gap', 'bool_frames', 'extra_array'])
def test_solid_provenance_or_array_fault_stops_before_output(prepare, source_case, fault):
    c = source_case
    if fault == 'missing_pins': c.pin.unlink()
    elif fault == 'wrong_pin': c.report['topology_budget']['committed_pins_sha256'] = 'f' * 64
    elif fault == 'wrong_route': c.report['mesh_source'] = 'default'
    elif fault == 'wrong_canonical': c.report['fixed_canonical_mesh_sha256'] = 'f' * 64
    elif fault == 'copy_changed': c.copied.chmod(0o644); c.copied.write_bytes(b'changed'); c.copied.chmod(0o444)
    elif fault == 'wrong_producer': c.report['topology_budget']['cpu_producer_revision'] = 'f' * 40
    else:
        data = copy.deepcopy(c.arrays)
        if fault == 'shape_changed': data['vertices'][1, 0] = 2.
        elif fault == 'wrong_scale': data['object_scale'] = np.array(.375)
        elif fault == 'frame_gap': data['frame_index'] = np.array([0, 1, 3], np.int64)
        elif fault == 'bool_frames': data['frame_index'] = np.ones(3, bool)
        elif fault == 'extra_array': data['hidden'] = np.array(1)
        c.pose.chmod(0o644); np.savez(c.pose, **data); c.pose.chmod(0o444)
        c.report['geometry_and_poses_sha256'] = prepare.sha256(c.pose)
    c.report_path.chmod(0o644); c.report_path.write_text(json.dumps(c.report)); c.report_path.chmod(0o444)
    with pytest.raises((ValueError, FileNotFoundError)): invoke(prepare, c)
    assert not (c.root / 'outputs/episode_000009/cari_inputs').exists()


@pytest.mark.parametrize('target', ['pose', 'pin', 'canonical', 'helper'])
def test_prepost_source_freeze_detects_mutation(prepare, source_case, target):
    c = source_case; ledger = invoke(prepare, c)[-1]
    path = c.code / 'infra/cari_prepare.py' if target == 'helper' else getattr(c, target)
    path.chmod(0o644); path.write_bytes(b'changed after preflight'); path.chmod(0o444)
    with pytest.raises(ValueError, match='changed'): prepare._solid_recheck(ledger)


@pytest.mark.parametrize('fault', ['unsealed_parent', 'unsealed_report', 'extra_file', 'in_memory_report', 'mutable_helper'])
def test_unsealed_or_incomplete_pose_is_not_an_authoritative_pass(prepare, source_case, fault):
    c = source_case
    if fault == 'unsealed_parent': c.pose.parent.chmod(0o755)
    elif fault == 'unsealed_report': c.report_path.chmod(0o644)
    elif fault == 'extra_file':
        c.pose.parent.chmod(0o755); (c.pose.parent / 'unknown').write_bytes(b'no resume'); c.pose.parent.chmod(0o555)
    elif fault == 'in_memory_report': c.report['fixed_shape'] = False
    elif fault == 'mutable_helper': (c.code / 'infra/cari_prepare.py').chmod(0o644)
    with pytest.raises(ValueError): invoke(prepare, c)


def test_exact_local_orientation_is_not_only_a_signed_volume_proxy(prepare):
    v, f = tetra(); stored = v.copy(); stored[[1, 2]] = stored[[2, 1]]
    with pytest.raises(ValueError, match='local triangle'):
        prepare._solid_float32_orientation(v, stored, f)


def homogeneous(points, matrix):
    return np.dot(matrix, np.column_stack((points, np.ones(len(points)))).T).T[:, :3]


def projected(matrix, max_deviance=1e-5):
    result = matrix.copy()
    error = np.abs(matrix[:3, :3] @ matrix[:3, :3].T - np.eye(3)).max()
    if 1e-13 < error < max_deviance:
        u, _, vt = np.linalg.svd(matrix[:3, :3]); result[:3, :3] = u @ vt
    return result


@pytest.fixture
def serialized(monkeypatch):
    import mesh_precision_diagnostic as raw
    import object_budget_endpoint as endpoint
    v, f = tetra(); A = np.eye(4); A[:3, 3] = [.25, -.5, .125]
    class Scene:
        def __init__(self, matrix):
            self.geometry = {'mesh': object()}
            self.graph = SimpleNamespace(base_frame='world', nodes_geometry=['mesh'], repair_rigid=1e-5)
            self.edges = [['world', 'mesh', {'matrix': matrix.copy()}]]
            self.effective = projected(matrix)
            self.graph.to_edgelist = lambda: self.edges
        def __getitem__(self, key): return self.effective, 'mesh'
    scene = Scene(A)
    # A special method belongs on the graph class, not its instance.
    class Graph(SimpleNamespace):
        def __getitem__(self, key): return scene.effective, 'mesh'
    scene.graph = Graph(**vars(scene.graph))
    case = SimpleNamespace(v=v, f=f, A=A, scene=scene,
        records=[{'node_transform_identity': False}], local=[(v.astype(np.float32), f.copy())],
        world=homogeneous(v, A)[f], loaded=(homogeneous(v, A), f.copy()),
        native=(homogeneous(v, A).astype(np.float32), f.copy()))
    case.trimesh = SimpleNamespace(Scene=Scene, load=lambda *args, **kwargs: scene,
        transform_points=homogeneous, transformations=SimpleNamespace(fix_rigid=projected))
    case.native_load = lambda _: case.native
    def set_matrix(matrix):
        case.A = matrix; scene.edges[0][2]['matrix'] = matrix.copy(); scene.effective = projected(matrix)
        case.world = homogeneous(v, matrix)[f]
        case.loaded = homogeneous(v, scene.effective), f.copy()
        case.native = case.loaded[0].astype(np.float32), f.copy()
    case.set_matrix = set_matrix
    monkeypatch.setattr(raw, 'raw_glb', lambda _: (case.local, case.world, case.records))
    monkeypatch.setattr(endpoint, '_load_mesh', lambda _: case.loaded)
    return case


def serialized_call(prepare, case, tmp_path, transform=True):
    from exact_mesh_geometry import exact_mesh_topology
    return prepare._solid_serialized_mesh(tmp_path / 'opaque.glb', (case.v, case.f),
        exact_mesh_topology(case.v, case.f), case.trimesh, np, case.native_load, case.A if transform else None)


def test_actual_glb_accessor_and_scene_transform_contract_without_loader_processing(prepare, serialized, tmp_path):
    case = serialized; proof, native = serialized_call(prepare, case, tmp_path)
    assert proof['faces'] == 4 and proof['all_meaningful_triangles_preserved']
    assert proof['native_rigidprojection_used'] is False
    assert proof['local_coordinates_or_topology_repaired'] is False and 'geometry_repaired' not in proof
    assert np.array_equal(native[0], case.native[0]) and not np.shares_memory(native[0], case.native[0])
    with pytest.raises(ValueError, match='hidden scene'): serialized_call(prepare, case, tmp_path, False)
    case.loaded = case.loaded[0], case.f[:, ::-1]
    with pytest.raises(ValueError, match='represented rigid'): serialized_call(prepare, case, tmp_path)


def test_no_meaningful_face_loss_or_coordinate_guess(prepare, serialized, tmp_path):
    serialized.local = [(serialized.v.astype(np.float32), serialized.f[:-1])]
    with pytest.raises(ValueError, match='POSITION'): serialized_call(prepare, serialized, tmp_path)


def test_scene_matrix_is_retained_not_assumed_baked_float32(prepare, serialized, tmp_path):
    case = serialized; A = np.eye(4); A[:3, 3] = [float(np.float32(.1)), 0., 0.]; case.set_matrix(A)
    assert not np.array_equal(case.loaded[0].astype(np.float32).astype(np.float64), case.loaded[0])
    proof, native = serialized_call(prepare, case, tmp_path)
    assert proof['native_float32_loader_replayed'] is True
    assert native[0].dtype == np.float32 and np.array_equal(native[0], case.loaded[0].astype(np.float32))
    case.loaded = case.loaded[0].astype(np.float32).astype(np.float64), case.f
    with pytest.raises(ValueError, match='represented rigid'): serialized_call(prepare, case, tmp_path)


def test_float32_rotation_native_projection_is_explicit_not_an_exact_raw_world_claim(prepare, serialized, tmp_path):
    case = serialized; A = np.eye(4)
    c, s = float(np.float32(.6)), float(np.float32(.8))
    A[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]; A[:3, 3] = [float(np.float32(.1)), .25, -.125]
    case.set_matrix(A)
    assert not np.array_equal(case.world, case.loaded[0][case.f])
    proof, native = serialized_call(prepare, case, tmp_path)
    assert proof['native_rigidprojection_used'] is True
    assert np.array_equal(proof['raw_scene_transform'], A)
    assert np.array_equal(proof['effective_native_transform'], projected(A))
    assert proof['oriented_triangles_sha256'] != proof['effective_oriented_triangles_sha256']
    rotations = np.repeat(np.eye(3)[None], 3, axis=0); translations = np.zeros((3, 3))
    poses = np.repeat(np.eye(4)[None], 3, axis=0) @ np.linalg.inv(A)
    assert prepare._solid_camera_roundtrip((case.v, case.f), rotations, translations, poses.astype(np.float32), native, np) <= 1e-5


@pytest.mark.parametrize('fault', ['raw', 'graph', 'policy', 'extra_geometry', 'extra_instance', 'native',
    'native_dtype', 'native_faces', 'world', 'permuted_lineage', 'orphan_parent'])
def test_distinct_raw_effective_and_native_stages_fail_closed(prepare, serialized, tmp_path, fault):
    c = serialized
    if fault == 'raw': c.scene.edges[0][2]['matrix'][0, 3] += .125
    elif fault == 'graph': c.scene.effective[0, 3] += .125
    elif fault == 'policy': c.scene.graph.repair_rigid = None
    elif fault == 'extra_geometry': c.scene.geometry['orphan'] = object()
    elif fault == 'extra_instance': c.scene.graph.nodes_geometry.append('another')
    elif fault == 'native': c.native[0][0, 0] += np.float32(.125)
    elif fault == 'native_dtype': c.native = c.native[0].astype(np.float64), c.native[1]
    elif fault == 'native_faces': c.native = c.native[0], c.f[:, ::-1]
    elif fault == 'world': c.world[0, 0, 0] += .125
    elif fault == 'permuted_lineage':
        perm = np.array([1, 0, 2, 3]); c.local = [(c.v[perm].astype(np.float32), perm[c.f])]
    elif fault == 'orphan_parent': c.scene.edges.clear()
    with pytest.raises(ValueError): serialized_call(prepare, c, tmp_path)


def test_metric_identity_replays_actual_native_fp32_loader(prepare, serialized, tmp_path):
    case = serialized; case.set_matrix(np.eye(4)); case.records[0]['node_transform_identity'] = True
    proof, native = serialized_call(prepare, case, tmp_path, False)
    assert proof['native_rigidprojection_used'] is False
    assert np.array_equal(native[0], case.v.astype(np.float32))


def test_saved_float32_poses_actual_camera_roundtrip_uses_same_inherited_bound(prepare):
    v, f = tetra(); A = np.eye(4); A[:3, 3] = [.25, -.5, .125]
    rotations = np.repeat(np.eye(3)[None], 3, axis=0); translations = np.zeros((3, 3))
    poses = np.repeat(np.eye(4)[None], 3, axis=0); aligned = poses @ np.linalg.inv(A)
    native = homogeneous(v, A).astype(np.float32), f
    assert prepare._solid_camera_roundtrip((v, f), rotations, translations, aligned.astype(np.float32), native, np) == 0.
    translations[-1, 0] = 1000.000023; poses[-1, 0, 3] = translations[-1, 0]; aligned = poses @ np.linalg.inv(A)
    with pytest.raises(ValueError, match='inherited 1e-5'):
        prepare._solid_camera_roundtrip((v, f), rotations, translations, aligned.astype(np.float32), native, np)
    with pytest.raises(ValueError, match='Actual saved'):
        prepare._solid_camera_roundtrip((v, f), rotations, translations, aligned, native, np)


def test_camera_roundtrip_never_substitutes_raw_A_positions_for_native_FP32_positions(prepare):
    v, f = tetra(); rotations = np.repeat(np.eye(3)[None], 3, axis=0); translations = np.zeros((3, 3))
    saved = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    native = v.astype(np.float32), f.copy(); native[0][1, 0] += np.float32(.001)
    with pytest.raises(ValueError, match='inherited 1e-5'):
        prepare._solid_camera_roundtrip((v, f), rotations, translations, saved, native, np)


@pytest.fixture
def native_sources(prepare, tmp_path, monkeypatch):
    import importlib
    from solid_geometry_loader import identity
    # The native package is outside the code namespace in production. Keep the
    # same relationship even when pytest's basetemp lives inside this checkout.
    monkeypatch.setattr(prepare, '__file__', str(tmp_path / 'code/infra/cari_prepare.py'))
    package = tmp_path / 'trimesh'; native = tmp_path / 'native'; modules = {}; pins = {}
    for name in prepare.SOLID_TRIMESH_SOURCES:
        path = package / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# manufactured source only\n')
        pins[name] = identity(path, readonly=False)
        module = 'trimesh.' + name.removesuffix('/__init__.py').removesuffix('.py').replace('/', '.')
        modules[module] = SimpleNamespace(__file__=str(path))
    transform_namespace = {}
    exec(compile('def fix_rigid(matrix, max_deviance=1e-5):\n return matrix\n',
                 str(package / 'transformations.py'), 'exec'), transform_namespace)
    texts = {'lib_mhr/contact.py': '''from __future__ import annotations
import torch  # must never execute this module import
def load_object_mesh(path: str | Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    return trimesh.util.concatenate(scene.dump(concatenate=False))
''', 'learning/training/mhr_opt_refineout.py': '''from __future__ import annotations
import torch  # must never execute this module import
def _load_object_vertices(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    mesh = load_object_mesh(path)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)
'''}
    native_pins = {}
    for name, text in texts.items():
        path = native / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
        native_pins[name] = identity(path, readonly=False)
    v, f = tetra(); mesh = SimpleNamespace(vertices=v, faces=f); calls = []
    def load(*args, **kwargs):
        calls.append(kwargs); return SimpleNamespace(dump=lambda **kw: [mesh])
    fake = SimpleNamespace(__version__='5.1.0', __file__=str(package / '__init__.py'), load=load,
        transformations=SimpleNamespace(fix_rigid=transform_namespace['fix_rigid']),
        util=SimpleNamespace(concatenate=lambda meshes: meshes[0]))
    monkeypatch.setattr(prepare, 'SOLID_TRIMESH_SOURCES', pins)
    monkeypatch.setattr(prepare, 'SOLID_NATIVE_LOAD_SOURCES', native_pins)
    original_import = importlib.import_module
    monkeypatch.setattr(importlib, 'import_module', lambda name, *args: modules[name] if name in modules else original_import(name, *args))
    return SimpleNamespace(package=package, native=native, modules=modules, fake=fake, calls=calls, mesh=mesh)


def test_original_two_function_execution_has_no_optimizer_or_model_import(prepare, native_sources):
    c = native_sources; before = set(sys.modules)
    load, ledger = prepare._solid_native_sources(c.native, c.fake, np)
    v, f = load(Path('/opaque/synthetic.glb'))
    assert v.dtype == np.float32 and f.dtype == np.int64
    assert np.array_equal(v, c.mesh.vertices) and np.array_equal(f, c.mesh.faces)
    assert c.calls == [{'force': 'scene', 'process': False}]
    assert len(ledger) == len(prepare.SOLID_TRIMESH_SOURCES) + 2
    assert load.__code__.co_filename == str(c.native / 'learning/training/mhr_opt_refineout.py')
    assert not {'torch', 'learning', 'lib_mhr'} & (set(sys.modules) - before)
    prepare._solid_recheck(ledger)


@pytest.mark.parametrize('fault', ['version', 'installed_source', 'native_source', 'import_origin', 'function_origin', 'symlink'])
def test_installed_and_original_loader_sources_are_authenticated_before_execution(prepare, native_sources, fault):
    c = native_sources
    if fault == 'version': c.fake.__version__ = '5.1.1'
    elif fault == 'installed_source': (c.package / 'base.py').write_text('# changed\n')
    elif fault == 'native_source': (c.native / 'lib_mhr/contact.py').write_text('# changed\n')
    elif fault == 'import_origin': c.modules['trimesh.base'].__file__ = '/wrong/source/base.py'
    elif fault == 'function_origin': c.fake.transformations.fix_rigid = projected
    elif fault == 'symlink':
        p = c.package / 'base.py'; content = p.read_bytes(); p.unlink()
        other = c.package / 'other.py'; other.write_bytes(content); p.symlink_to(other)
    with pytest.raises(ValueError): prepare._solid_native_sources(c.native, c.fake, np)
    assert not c.calls


def test_authenticated_native_source_posthash_detects_later_mutation(prepare, native_sources):
    c = native_sources; _, ledger = prepare._solid_native_sources(c.native, c.fake, np)
    (c.package / 'scene/transforms.py').write_text('# altered after authentication\n')
    with pytest.raises(ValueError, match='changed'): prepare._solid_recheck(ledger)


def test_default_numeric_statements_and_report_literal_preserved(prepare):
    # Compare to the actual capacity/batched-depth producer immediately before
    # latent admission; older single-frame depth scheduling is separately tested.
    historical = ast.parse(subprocess.check_output(['git', 'show', 'a51482f63911fa05188ceb3e74d5f5ddc416c8a1:infra/cari_prepare.py'], text=True))
    current = ast.parse(Path(prepare.__file__).read_text())
    old_main = next(n for n in historical.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    new_main = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    statements = [n for n in old_main.body if isinstance(n, (ast.With, ast.Assign)) and
                  (isinstance(n, ast.With) or any(isinstance(t, ast.Name) and t.id in ('metric_mesh', 'active', 'result', 'aligned_poses') for t in n.targets))]
    new_nodes = {ast.dump(n) for n in ast.walk(new_main)}
    for node in statements: assert ast.dump(node) in new_nodes
    assert 'process=False' in Path(prepare.__file__).read_text()


@pytest.fixture
def fake_shell(tmp_path):
    binary = tmp_path / 'bin'; binary.mkdir(); log = tmp_path / 'args'; ctl = tmp_path / 'ctl'
    commands = {'docker': 'printf "%s\\0" "$@" >> "$FAKE_LOG"', 'id': 'echo 123',
                'systemctl': 'echo "$*" >> "$FAKE_CTL"; if [[ "$3" == --property=LoadState ]]; then echo loaded; else echo inactive; fi'}
    for name, script in commands.items():
        p = binary / name; p.write_text('#!/usr/bin/env bash\nset -eu\n' + script + '\n'); p.chmod(0o755)
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], WR_ROOT=str(tmp_path / 'runtime'),
               WR_CODE=str(REPO), WR_CODE_REVISION='a' * 40, FAKE_LOG=str(log), FAKE_CTL=str(ctl))
    def run(*args): return subprocess.run(['bash', str(REPO / 'infra/run_cari_prepare.sh'), *args], env=env, capture_output=True, text=True, timeout=5)
    def report(stage, episode=9):
        p = Path(env['WR_ROOT']) / f'outputs/episode_{episode:06d}/{stage}/report.json'
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
    return run, report, log, ctl, env


def test_wrapper_routes_only_explicit_solid_full_report(fake_shell):
    run, report, log, ctl, _ = fake_shell
    report('object_pose_full'); failed = run('--episode', '9', '--mesh-source', 'solid')
    assert failed.returncode != 0 and not log.exists()
    report('object_pose_full_solid'); result = run('--episode', '9', '--mesh-source', 'solid')
    assert result.returncode == 0, result.stderr
    assert log.read_bytes().decode().rstrip('\0').split('\0')[-4:] == ['--episode', '9', '--mesh-source', 'solid']
    assert not ctl.exists()


def test_wrapper_solid15_does_not_wait_historical_default_unit(fake_shell):
    run, report, log, ctl, _ = fake_shell; report('object_pose_full_solid', 15)
    result = run('--mesh-source', 'solid')
    assert result.returncode == 0, result.stderr
    assert not ctl.exists() and '--mesh-source\0solid' in log.read_bytes().decode()


@pytest.mark.parametrize('args', [('--mesh-source',), ('--mesh-source', 'volume'), ('--mesh-source', 'solid', '--mesh-source', 'default')])
def test_wrapper_invalid_mode_fails_before_commands(fake_shell, args):
    run, _, log, ctl, _ = fake_shell
    assert run(*args).returncode == 2 and not log.exists() and not ctl.exists()


def test_wrapper_default_docker_arguments_unchanged(fake_shell):
    run, report, log, _, _ = fake_shell; report('object_pose_full', 15)
    result = run(); assert result.returncode == 0, result.stderr
    assert log.read_bytes().decode().rstrip('\0').split('\0')[-2:] == ['--episode', '15']
    assert '--mesh-source' not in log.read_bytes().decode()


@pytest.mark.parametrize('mode', ['forward', 'converter', 'adapter', 'refine'])
def test_shared_parser_mesh_source_is_prepare_only(mode):
    result = subprocess.run(['bash', '-c', 'source "$1"; wr_parse_cari_arguments "$2" --mesh-source solid',
                             'test', str(REPO / 'infra/cari_wrapper_common.sh'), mode], capture_output=True, timeout=5)
    assert result.returncode == 2


def test_real_bundle_resolver_contains_inert_loader_raw_reader_and_exact_sources():
    spec = importlib.util.spec_from_file_location('solid_prepare_bundle', REPO / 'infra/azure_job.py')
    job = importlib.util.module_from_spec(spec); spec.loader.exec_module(job)
    names = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', '-z'], text=True).split('\0')
    files = {name: (REPO / name).read_bytes() for name in names if name and (REPO / name).is_file()}
    paths = job.runtime_bundle_paths(files, 'infra/run_cari_prepare.sh')
    assert {'infra/solid_geometry_loader.py', 'infra/mesh_precision_diagnostic.py', 'infra/exact_mesh_geometry.py',
            'src/world_reward/exact_triangle_predicates.py', 'configs/solid_mesh_000009_pins.json',
            'infra/cari_wrapper_common.sh', 'infra/run_cari_prepare.sh'} <= set(paths)
