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
                 'infra/cari_wrapper_common.sh', 'infra/run_cari_prepare.sh'}:
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


def test_actual_glb_accessor_and_scene_transform_contract_without_loader_processing(prepare, tmp_path, monkeypatch):
    import mesh_precision_diagnostic as raw
    import object_budget_endpoint as endpoint
    from exact_mesh_geometry import exact_mesh_topology
    v, f = tetra(); A = np.eye(4); A[:3, 3] = [.25, -.5, .125]
    transform = lambda points, matrix: points @ matrix[:3, :3].T + matrix[:3, 3]
    world = transform(v, A)
    monkeypatch.setattr(raw, 'raw_glb', lambda _: ([(v.astype(np.float32), f)], world[f], [{'node_transform_identity': False}]))
    monkeypatch.setattr(endpoint, '_load_mesh', lambda _: (world, f))
    proof = prepare._solid_serialized_mesh(tmp_path / 'opaque.glb', (v, f), exact_mesh_topology(v, f),
                                            SimpleNamespace(transform_points=transform), np, A)
    assert proof['faces'] == 4 and proof['all_meaningful_triangles_preserved']
    with pytest.raises(ValueError, match='hidden scene'): prepare._solid_serialized_mesh(
        tmp_path / 'opaque.glb', (v, f), exact_mesh_topology(v, f), SimpleNamespace(transform_points=transform), np)
    monkeypatch.setattr(endpoint, '_load_mesh', lambda _: (world, f[:, ::-1]))
    with pytest.raises(ValueError, match='represented rigid'): prepare._solid_serialized_mesh(
        tmp_path / 'opaque.glb', (v, f), exact_mesh_topology(v, f), SimpleNamespace(transform_points=transform), np, A)


def test_no_meaningful_face_loss_or_coordinate_guess(prepare, tmp_path, monkeypatch):
    import mesh_precision_diagnostic as raw
    import object_budget_endpoint as endpoint
    from exact_mesh_geometry import exact_mesh_topology
    v, f = tetra(); transform = lambda points, matrix: points @ matrix[:3, :3].T + matrix[:3, 3]
    monkeypatch.setattr(raw, 'raw_glb', lambda _: ([(v.astype(np.float32), f[:-1])], v[f[:-1]], [{'node_transform_identity': True}]))
    monkeypatch.setattr(endpoint, '_load_mesh', lambda _: (v, f))
    with pytest.raises(ValueError, match='POSITION'): prepare._solid_serialized_mesh(
        tmp_path / 'opaque.glb', (v, f), exact_mesh_topology(v, f), SimpleNamespace(transform_points=transform), np)


def test_scene_matrix_is_retained_not_assumed_baked_float32(prepare, tmp_path, monkeypatch):
    import mesh_precision_diagnostic as raw
    import object_budget_endpoint as endpoint
    from exact_mesh_geometry import exact_mesh_topology
    v, f = tetra(); A = np.eye(4); A[:3, 3] = [float(np.float32(.1)), 0., 0.]
    transform = lambda points, matrix: points @ matrix[:3, :3].T + matrix[:3, 3]
    world = transform(v, A)
    assert not np.array_equal(world.astype(np.float32).astype(np.float64), world)
    monkeypatch.setattr(raw, 'raw_glb', lambda _: ([(v.astype(np.float32), f)], world[f], [{'node_transform_identity': False}]))
    monkeypatch.setattr(endpoint, '_load_mesh', lambda _: (world, f))
    prepare._solid_serialized_mesh(tmp_path / 'opaque.glb', (v, f), exact_mesh_topology(v, f),
                                   SimpleNamespace(transform_points=transform), np, A)
    monkeypatch.setattr(endpoint, '_load_mesh', lambda _: (world.astype(np.float32).astype(np.float64), f))
    with pytest.raises(ValueError, match='represented rigid'):
        prepare._solid_serialized_mesh(tmp_path / 'opaque.glb', (v, f), exact_mesh_topology(v, f),
                                       SimpleNamespace(transform_points=transform), np, A)


def test_saved_float32_poses_actual_camera_roundtrip_uses_same_inherited_bound(prepare):
    v, f = tetra(); A = np.eye(4); A[:3, 3] = [.25, -.5, .125]
    rotations = np.repeat(np.eye(3)[None], 3, axis=0); translations = np.zeros((3, 3))
    poses = np.repeat(np.eye(4)[None], 3, axis=0); aligned = poses @ np.linalg.inv(A)
    factory = SimpleNamespace(transform_points=lambda points, matrix: points @ matrix[:3, :3].T + matrix[:3, 3])
    assert prepare._solid_camera_roundtrip((v, f), rotations, translations, aligned, A, factory, np) == 0.
    translations[-1, 0] = 1000.000023; poses[-1, 0, 3] = translations[-1, 0]; aligned = poses @ np.linalg.inv(A)
    with pytest.raises(ValueError, match='inherited 1e-5'):
        prepare._solid_camera_roundtrip((v, f), rotations, translations, aligned, A, factory, np)


def test_default_numeric_statements_and_report_literal_preserved(prepare):
    historical = ast.parse(subprocess.check_output(['git', 'show', '1a51c715a46c376d82650c595d008e83bf250554:infra/cari_prepare.py'], text=True))
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
