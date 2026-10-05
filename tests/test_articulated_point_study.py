"""Tiny policy/ABI/provenance checks; not a native scientific qualification."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'infra'))
import articulated_point_study as s
import articulated_point_native as n
import articulated_point_tracks as tracks
import mediapipe_cpu_runtime_verify as rt
from world_reward.articulated_point_cohort import SCENES


def test_frozen_protocol_and_cutoff_contract():
    raw = (ROOT/s.PROTOCOL).read_bytes()
    assert s.PROTOCOL_PIN == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    p = json.loads(raw)
    assert p['budgets_seconds'] == s.BUDGETS
    assert p['cohort']['scenes'] == [row[0] for row in SCENES]
    assert p['calibration']['fits_on_development'] == 0
    assert p['reserved']['fits'] == 8 and p['reserved']['updates_each'] == 301
    assert p['reserved']['bit_identical_claim'] is False
    assert p['claims']['challenge_inputs_used'] is p['claims']['adoption'] is False


def test_lazy_host_import_has_no_model_or_data_side_effect():
    code = f"import runpy,sys;runpy.run_path({str(ROOT/'infra/articulated_point_study.py')!r},run_name='audit');assert 'torch' not in sys.modules;assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, '-I', '-B', '-S', '-c', code], check=True, capture_output=True)


def test_complete_transitive_source_closure():
    import azure_job
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes() for folder in ('infra', 'src', 'configs')
        for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['pyproject.toml'] = (ROOT/'pyproject.toml').read_bytes()
    selected = azure_job.runtime_bundle_paths(files, 'infra/run_articulated_point_study.sh')
    assert set(s.HELPERS) <= set(selected)
    assert set(n.SOURCE_HELPERS) <= set(selected)
    assert set(tracks.SOURCE_HELPERS) <= set(selected)


def test_scientific_policy_hash_ignores_only_stage_seals():
    c = {'articulated_study': json.loads((ROOT/s.PROTOCOL).read_text())}
    before = n.policy_sha256(c)
    c['articulated_study'].update(stage='tracks', manufacture={'path': 'owned', 'report': {}}, tracks={})
    assert n.policy_sha256(c) == before
    c['articulated_study']['calibration']['fits_on_development'] = 1
    assert n.policy_sha256(c) != before


def test_reserved_metrics_are_deferred_until_both_native_results_are_frozen():
    tree = ast.parse((ROOT/'infra/articulated_point_native.py').read_text())
    fit = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_fit')
    reserved = next(node for node in fit.body if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple) and isinstance(node.target.elts[0], ast.Name)
        and node.target.elts[0].id == 'j')
    arm_loop = next(node for node in reserved.body if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == 'arm')
    metric_calls = [node for node in ast.walk(arm_loop) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id in ('_metrics', '_truth')]
    assert metric_calls == []
    append = [node for node in ast.walk(arm_loop) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'append'
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pending_evaluation']
    assert len(append) == 1
    calls = [node for node in ast.walk(reserved) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id in ('_metrics', '_truth')]
    assert len(calls) == 2 and all(node.lineno > arm_loop.end_lineno for node in calls)


def valid(stage):
    proof = {'source_binding': {'fixture': 'unchanged'}}
    r = dict(stage='articulated_point_'+stage+'_native_v1', status='pass', phase='complete',
        frames=144, source_binding=proof['source_binding'], protocol_identity=s.PROTOCOL_PIN,
        source_inputs_assets_rehashed_after=True, challenge_inputs_used=False, adoption=False)
    if stage == 'manufacture':
        r.update(native_decode_calls=6, native_decoded_frames=144, reconstitution_decode_calls=6,
            reconstitution_decoded_frames=144, primitive_render_calls=12, primitive_render_frames=288,
            tracker_executed=False, native_metadata_rehashed_after=True, scenes=[{}]*6, public_views=[{}]*6)
    elif stage == 'tracks':
        r.update(model_loads=1, native_calls_attempted=6, native_calls_returned=6, native_calls_completed=6,
            private_truth_read=False, calibration_performed=False, runtime_rehashed_after=True,
            manufacture_rehashed_after=True, outputs_rehashed_after=True, scenes=[{}]*6,
            outputs={str(i): {} for i in range(12)})
    else:
        r.update(constructor_attempts=10, constructor_returns=10, optimizer_run_attempts=8,
            optimizer_run_returns=8, actual_native_updates=2408, descriptive_four_pairs_only=True,
            calibrated_before_reserved_reads=True, both_results_frozen_before_private_evaluation=True, statistical_population_gain_verified=False,
            positive_weight_executed=True, reserved_results=[dict(status='pass', both_results_frozen_before_private_evaluation=True,
                results={'A_original': {}, 'B_point': {}})]*4)
    return r, proof


@pytest.mark.parametrize('stage', s.STAGES)
def test_complete_stage_census(stage):
    r, p = valid(stage); s.validate_stage(rt, r, stage, p)
    for key in ('frames', 'challenge_inputs_used', 'source_inputs_assets_rehashed_after', 'status'):
        bad = copy.deepcopy(r); bad[key] = 'wrong'
        with pytest.raises(ValueError): s.validate_stage(rt, bad, stage, p)
    if stage == 'fit':
        bad = copy.deepcopy(r)
        bad['reserved_results'][0]['both_results_frozen_before_private_evaluation'] = False
        with pytest.raises(ValueError): s.validate_stage(rt, bad, stage, p)


@pytest.mark.parametrize('stage', s.STAGES)
def test_integer_counts_cannot_be_bool(stage):
    r, p = valid(stage)
    key = {'manufacture': 'native_decode_calls', 'tracks': 'model_loads', 'fit': 'actual_native_updates'}[stage]
    r[key] = True
    with pytest.raises(ValueError): s.validate_stage(rt, r, stage, p)


def test_material_projection_full_frames_without_crop_or_alignment():
    points = np.array([[.1, -.2, .3], [-.2, .4, .1]], np.float64)
    pose = np.broadcast_to(np.eye(4), (24, 4, 4)).copy()
    pose[:, :3, 3] = np.column_stack((np.linspace(0, .1, 24), np.zeros(24), np.full(24, 4)))
    K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]])
    value = n._projection(np, dict(canonical_points=points, K=K), dict(pose=pose))
    xyz = points[None]+pose[:, None, :3, 3]
    expected = np.stack(((800*xyz[..., 0]/xyz[..., 2]+320)*.4,
                         (800*xyz[..., 1]/xyz[..., 2]+240)*(256/480)), -1)
    assert np.allclose(value, expected, rtol=0, atol=1e-13) and value.shape == (24, 2, 2)
    pose[:, 2, 3] = -1
    with pytest.raises(ValueError): n._projection(np, dict(canonical_points=points, K=K), dict(pose=pose))


def test_tracker_mount_path_never_whole_manufacture(tmp_path, monkeypatch):
    code = tmp_path/'jobs'/'code'; code.mkdir(parents=True)
    directory = tmp_path/'manufacture'; directory.mkdir()
    (directory/'report.json').write_text('{}')
    files = {f'scene_{i:02d}_{suffix}' for i in range(6) for suffix in ('public.npz', 'private.npz', 'source.pth', 'object.obj')}
    for name in files: (directory/name).write_bytes(b'fixture')
    monkeypatch.setattr(n, 'authenticate_manufacture', lambda *_: (directory, dict(outputs={k: {} for k in files})))
    c = {'articulated_study': {}}
    proof = dict(boots_files={}, frozen={})
    paths = s.stage_mounts(rt, code, c, proof, 'tracks')
    assert directory not in paths
    assert set(paths) == {code.parent, directory/'report.json', *(directory/n for n in files if n.endswith('_public.npz'))}


def test_old_native_defaults_unchanged():
    import inspect
    assert inspect.signature(s.base.prerequisites).parameters['entry'].default == s.base.ENTRY
    assert inspect.signature(s.base.cleanup).parameters['image'].default == s.base.IMAGE
    assert s.base.profile('component_preflight')[0] == s.base.STUDY_PIN


def test_protocol_or_stage_change_does_not_create_old_cohort_retry():
    for name in ('infra/articulated_point_study.py', 'infra/articulated_point_native.py', 'infra/articulated_point_tracks.py'):
        ast.parse((ROOT/name).read_text())
    assert n.STAGES['manufacture'] != 'joint_point_component_preflight_v1'
    assert not ({row[0] for row in SCENES} & {'dev_left', 'dev_right'})
    assert "frames=[f'{t:06d}'" in (ROOT/'infra/articulated_point_native.py').read_text()
    assert "names = tuple(f'{t:06d}'" in (ROOT/'infra/articulated_point_tracks.py').read_text()
