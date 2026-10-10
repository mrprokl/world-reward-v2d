"""Tiny real consumer API/serialization tests; no models, media or Azure calls."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cari_prepare as prepare
import object_pose_smoke as pose
import surface_geometry_loader as loader


def latent_case(total=3):
    vertices = np.array([[0., 0., 2.], [.2, 0., 2.], [0., .2, 2.], [0., 0., 2.]])
    faces = np.array([[0, 1, 2]], np.int64)
    v = np.concatenate((vertices, np.repeat(vertices[:1], 4092, axis=0)))
    f = np.concatenate((faces, np.zeros((4095, 3), np.int64)))
    rows = []
    for i in range(total):
        c = dict(hypothesis_index=0, rotation=np.eye(3).tolist(), translation=[i*.1, 0., 0.],
            initial_silhouette_iou=.8, fitted_silhouette_iou=.8,
            selected_silhouette_iou=.8, selected_depth_residual_m=.01)
        rows.append(dict(frame_index=i, candidates=[c], selected=c, pose_observed=True,
            observation_status='automatic_mask_and_inferred_depth') if i in (0, total-1) else
            pose._missing_pose_report(i, 0, 'a'*64, 'empty_automatic_mask'))
    r, t, flags, temporal = pose._select_latent_pose_path(rows, list(range(total)), 2, vertices.mean(0))
    report = dict(allow_unobserved_poses=True, latent_pose_initializer=True, latent_poses_measured=False,
        pose_observed=flags.tolist(), observed_pose_frames=2, latent_pose_frames=total-2,
        temporal_selection=temporal, frames=rows, ground_truth_used=False, hand_labeled_test=False,
        oracle_modes=[])
    data = dict(vertices=v, faces=f, rotation=r, translation=t, frame_index=np.arange(total, dtype=np.int64),
        object_scale=np.array(1.), pose_observed=flags)
    budget = dict(source_domain='surface', metric_scale_baked_once=.5,
        canonical_vertices_count=4, canonical_faces_count=1)
    return data, report, budget


def write_npz(path, data):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **data)
    path.chmod(0o444)


def test_optin_surface_only_defaults_fail_before_input_io(monkeypatch):
    assert prepare._argument_parser().parse_args([]).allow_unobserved_poses is False
    args = prepare._argument_parser().parse_args(['--mesh-source', 'surface', '--allow-unobserved-poses'])
    assert args.allow_unobserved_poses
    monkeypatch.setattr(prepare.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(prepare.Path, 'iterdir', lambda _: iter([Path('lo')]))
    monkeypatch.setattr(prepare.sys, 'argv', ['cari_prepare.py', '--allow-unobserved-poses'])
    monkeypatch.setattr(prepare, '_validate_inputs', lambda *_a, **_k: pytest.fail('Invalid flag must fail before IO'))
    with pytest.raises(ValueError, match='surface-only'):
        prepare.main()


def test_actual_latent_npz_admission_preserves_geometry_poses_all_frames_and_no_interpolation_prediction(tmp_path):
    data, report, budget = latent_case(); path = tmp_path/'geometry_and_poses.npz'; write_npz(path, data)
    out = prepare._latent_surface_geometry_and_poses(path, data['vertices'], data['faces'], report, 3, budget, np)
    for actual, key in [(out[0], 'vertices'), (out[1], 'faces'), (out[3], 'rotation'), (out[4], 'translation')]:
        assert actual.dtype == data[key].dtype and actual.tobytes() == data[key].tobytes()
        assert not actual.flags.writeable
    assert len(out[5][0]) == 4  # Canonical orphan retained, not confused with padding.
    flags = prepare._latent_observations(report, data['pose_observed'], 3, np)
    assert flags.tolist() == [True, False, True] and not flags.flags.writeable
    metadata = prepare._latent_pose_metadata(flags, np)
    assert metadata['requires_native_refinement'] and not metadata['final_prediction']
    assert metadata['pose_observed_frame_index'] == [0, 1, 2] and not metadata['latent_poses_measured']
    with pytest.raises(ValueError, match='Exact full pose payload'):
        loader.preflight_geometry_and_poses(path, expected_v=data['vertices'], expected_f=data['faces'],
            topology_budget=budget)  # Default historical reader is deliberately unchanged.


@pytest.mark.parametrize('bad', ['drop_flag', 'extra', 'f32', 'geometry', 'indices', 'scale', 'nan', 'rotation',
    'flag_integer', 'leading', 'trailing', 'report_flags', 'frame', 'fabricated', 'oracle', 'temporal', 'sentinel'])
def test_latent_admission_rejects_corruption_and_fabricated_observations(tmp_path, bad):
    data, report, budget = latent_case()
    if bad == 'drop_flag': del data['pose_observed']
    elif bad == 'extra': data['oracle_pose'] = np.zeros(1)
    elif bad == 'f32': data['translation'] = data['translation'].astype(np.float32)
    elif bad == 'geometry': data['vertices'] = data['vertices'].copy(); data['vertices'][0, 0] += .01
    elif bad == 'indices': data['frame_index'][1] = 4
    elif bad == 'scale': data['object_scale'] = np.array(.9)
    elif bad == 'nan': data['translation'] = data['translation'].copy(); data['translation'][1, 0] = np.nan
    elif bad == 'rotation': data['rotation'] = data['rotation'].copy(); data['rotation'][1, 0, 0] = 2.
    elif bad == 'flag_integer': data['pose_observed'] = data['pose_observed'].astype(np.int64)
    elif bad in ('leading', 'trailing'):
        data['pose_observed'] = data['pose_observed'].copy(); data['pose_observed'][0 if bad == 'leading' else -1] = False
    elif bad == 'report_flags': report['pose_observed'][1] = True
    elif bad == 'frame': report['frames'][1]['frame_index'] = 4
    elif bad == 'fabricated': report['frames'][1]['selected'] = report['frames'][0]['selected']
    elif bad == 'oracle': report['ground_truth_used'] = True
    elif bad == 'temporal': report['temporal_selection']['edge_extrapolation'] = True
    else: report['temporal_selection']['candidate_indices'][1] = 0
    path = tmp_path/'geometry_and_poses.npz'; write_npz(path, data)
    reference, _, _ = latent_case()
    with pytest.raises(ValueError):
        prepare._latent_surface_geometry_and_poses(path, reference['vertices'], reference['faces'], report, 3, budget, np)


def test_native_solver_initializer_serialization_retains_all_flags_without_changing_top_level_ABI(tmp_path):
    import joblib
    import cari_clip_inputs as native_consumer
    data, report, _ = latent_case(96)
    flags = prepare._latent_observations(report, data['pose_observed'], 96, np)
    names = [f'{i:06d}' for i in range(96)]; aligned = np.repeat(np.eye(4)[None], 96, axis=0)
    aligned[:, :3, 3] = data['translation']; matrix = np.eye(4); sha = 'a'*64
    # Execute the exact production branch, not a serialization imitation.
    tree = ast.parse(Path(prepare.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    branch = next(n for n in main.body if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'args.allow_unobserved_poses'
        and any(isinstance(c, ast.Name) and c.id == 'pose_metadata' for c in ast.walk(n)))
    path = tmp_path/'own_object_poses.pkl'
    env = dict(args=SimpleNamespace(allow_unobserved_poses=True), joblib=joblib, np=np, names=names,
        aligned_poses=aligned, A=matrix, pose_observed=flags, pose_path=tmp_path/'source.npz',
        object_poses_path=path, sha256=lambda _:sha, _latent_pose_metadata=prepare._latent_pose_metadata)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), '<actual native initializer serialization>', 'exec'), env)
    saved = joblib.load(path)
    assert set(saved) == {'frames', 'obj_pose_world', 'metadata'}
    assert prepare._surface_saved_poses(saved, names, aligned, sha, matrix, np, pose_observed=flags) is saved['obj_pose_world']
    assert saved['metadata']['pose_observed'] == [True] + [False]*94 + [True]
    spec = native_consumer.PublicClipSpec(26, 96, 'front_stereo_camera_left', 1152, 1536)
    native_consumer.validate_poses(saved, dict(source_object_mesh_to_aligned_transform=matrix), spec)
    with pytest.raises(ValueError): prepare._surface_saved_poses(saved, names, aligned, sha, matrix, np)
    saved['metadata']['pose_observed'][1] = True
    with pytest.raises(ValueError):
        prepare._surface_saved_poses(saved, names, aligned, sha, matrix, np, pose_observed=flags)


@pytest.mark.parametrize('admit', [False, True])
def test_actual_surface_preflight_has_explicit_nonrenamed_namespace_and_pins(tmp_path, monkeypatch, admit):
    data, report, budget = latent_case(); root = tmp_path/'root'; code = tmp_path/'code'
    monkeypatch.setattr(prepare, '__file__', str(code/'infra/cari_prepare.py'))
    monkeypatch.setattr(loader, 'SOURCE_HELPERS', ())
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode()); path.chmod(0o444)
    for n in ('cari_prepare.py', 'cari_wrapper_common.sh', 'run_cari_prepare.sh', 'surface_pose_report_capacity.py'):
        write(code/'infra'/n, b'tiny isolated helper stub')
    write(code/'src/world_reward/artifact_paths.py', b'tiny isolated helper stub')
    pin = code/'configs/surface_mesh_000026_pins.json'; write(pin, {'scope': 'tiny test'})
    base = root/'outputs/episode_000026'
    write(base/'object_grounded/report.json', {}); write(base/'scale_smoke/report.json', {})
    write(base/'object_grounded/transform.json', {'scale': [.5, .5, .5]})
    canonical = base/'object_budget_surface/object_fixed_canonical.glb'; write(canonical, b'no actual geometry decoder')
    parent = base/'object_pose_full_surface_latent'; parent.mkdir(); path = parent/'geometry_and_poses.npz'
    write_npz(path, data); write(parent/'object_fixed_canonical.glb', canonical.read_bytes())
    budget.update(committed_pins_sha256=loader.identity(pin)['sha256'], producer_report_sha256='b'*64,
        cpu_native_report_sha256='c'*64, geometry_operations_applied=False)
    report.update(mesh_source='surface', execution_verified=True, original_frame_coverage_verified=True,
        fixed_shape=True, object_report_sha256=prepare.sha256(base/'object_grounded/report.json'),
        geometry_and_poses_sha256=prepare.sha256(path), fixed_canonical_mesh_sha256=prepare.sha256(canonical),
        topology_budget=budget)
    write(parent/'report.json', report); parent.chmod(0o555)
    monkeypatch.setattr(loader, 'load', lambda *_a, **_k:(data['vertices'], data['faces'], np.array([0], np.int64),
        {}, canonical, budget))
    args = (root, 26, {'video_sha256': 'a'*64, 'total_frames': 3}, report, path, np)
    if admit:
        result = prepare._surface_preflight(*args, allow_unobserved_poses=True)
        assert result[3].tobytes() == data['rotation'].tobytes()
        assert result[4].tobytes() == data['translation'].tobytes()
    else:
        with pytest.raises(ValueError, match='Canonical surface full-trajectory path'):
            prepare._surface_preflight(*args)
