"""Explicit v4 source/profile and actual Joblib controls, never challenge data."""
import copy
from dataclasses import asdict
import json

import joblib
import numpy as np
import pytest

from test_cari_clip_inputs import gate, surface_fixture, write_json
import object_pose_smoke as pose
import cari_prepare as prepare


def repin(gate, root, spec):
    paths = gate.relative_paths(spec); deps = gate.dependency_paths(spec, object_source='surface_latent')
    row = json.loads((root/paths['input_report']).read_text())
    row['input_report_sha256'] = {role: gate.identity(root/name)['sha256'] for role, name in deps.items()}
    row['file_sha256']['object_poses'] = gate.identity(root/paths['object_poses'])['sha256']
    write_json(root/paths['input_report'], row)
    return dict(schema='world-reward-cari-clip-input-pins-v4', object_source='surface_latent',
        allow_unobserved_poses=True, clip_spec=asdict(spec),
        input_report=gate.identity(root/paths['input_report']) | {k: row[k] for k in ('producer_revision', 'script_sha256')},
        source_files={name: gate.identity(root/name) for name in gate.source_paths(spec, object_source='surface_latent')})


def case(gate, root, total=96):
    spec = gate.PublicClipSpec(26, total, 'front', 2, 4); surface_fixture(gate, root, spec)
    old = root/gate.dependency_paths(spec, object_source='surface')['object']
    new = root/gate.dependency_paths(spec, object_source='surface_latent')['object']
    new.parent.mkdir(); new.write_bytes(old.read_bytes())  # Keep historical surface receipt unchanged.
    original = json.loads(new.read_text()); rows = []
    for i in range(total):
        c = dict(hypothesis_index=0, rotation=np.eye(3).tolist(), translation=[i*.01, 0., 2.],
            initial_silhouette_iou=.8, fitted_silhouette_iou=.8, selected_silhouette_iou=.8, selected_depth_residual_m=.01)
        rows.append(dict(frame_index=i, candidates=[c], selected=c, pose_observed=True,
            observation_status='automatic_mask_and_inferred_depth') if i in (0, total-1) else
            pose._missing_pose_report(i, 0, 'a'*64, 'empty_automatic_mask'))
    _, _, flags, temporal = pose._select_latent_pose_path(rows, list(range(total)), 2, np.zeros(3))
    original.update(allow_unobserved_poses=True, latent_pose_initializer=True, latent_poses_measured=False,
        observed_pose_frames=2, latent_pose_frames=total-2, pose_observed=flags.tolist(),
        frames=rows, temporal_selection=temporal)
    write_json(new, original)
    paths = gate.relative_paths(spec); report_path = root/paths['input_report']; report = json.loads(report_path.read_text())
    report.update(allow_unobserved_poses=True, object_pose_observations=prepare._latent_pose_metadata(flags, np),
        object_pose_source=dict(report=str(new.relative_to(root)),
            geometry_and_poses=str(new.with_name('geometry_and_poses.npz').relative_to(root)),
            geometry_and_poses_sha256=original['geometry_and_poses_sha256']))
    write_json(report_path, report)
    payload_path = root/paths['object_poses']; payload = joblib.load(payload_path)
    payload['metadata'].update(prepare._latent_pose_metadata(flags, np)); joblib.dump(payload, payload_path)
    return spec, repin(gate, root, spec), old


@pytest.mark.parametrize('total', [96, 97])
def test_actual_v4_consumer_preserves_native_all_t_poses_and_flags(gate, tmp_path, monkeypatch, total):
    spec, pins, historical = case(gate, tmp_path, total); before = historical.read_bytes()
    monkeypatch.setattr(np, 'load', lambda *_a, **_k: pytest.fail('Input validator must not decode geometry or sensors'))
    assert gate.source_profile(pins) == 'surface_latent'
    gate.validate_pins(spec, pins)
    out = gate.verify_public_inputs(tmp_path, spec, pins)
    assert len(out['source_files']) == 15 and historical.read_bytes() == before
    assert out['poses']['metadata']['pose_observed'] == [True] + [False]*(total-2) + [True]
    assert out['poses']['metadata']['requires_native_refinement'] and not out['poses']['metadata']['final_prediction']
    assert out['poses']['obj_pose_world'].shape == (total, 4, 4)
    assert out['poses']['obj_pose_world'].dtype == np.float32
    assert np.all(np.diff(out['poses']['obj_pose_world'][:, 0, 3]) > 0)  # No static replacement.
    expected = gate.source_paths(spec, object_source='surface_latent')
    assert 'outputs/episode_000026/object_pose_full_surface_latent/report.json' in expected
    assert 'outputs/episode_000026/object_pose_full_surface/report.json' not in expected


@pytest.mark.parametrize('bad', ['v3', 'profile', 'flag_false', 'flag_integer', 'missingflag', 'missing_record',
    'leading', 'trailing', 'frames', 'fabricated', 'temporal', 'prepared_flag', 'prepared_final',
    'pickle_flag', 'pickle_final', 'posepath', 'oracle', 'mixed', 'fractionalflags', 'sentinel',
    'fake_status', 'prepared_index', 'pickle_index'])
def test_v4_rejects_relabeling_missing_evidence_or_legacy_namespace(gate, tmp_path, bad):
    spec, pins, _ = case(gate, tmp_path)
    paths = gate.relative_paths(spec); deps = gate.dependency_paths(spec, object_source='surface_latent')
    prepared_path = tmp_path/paths['input_report']; original_path = tmp_path/deps['object']
    if bad in {'v3', 'profile', 'flag_false', 'flag_integer', 'missingflag', 'missing_record', 'mixed'}:
        if bad == 'v3': pins['schema'] = 'world-reward-cari-clip-input-pins-v3'
        elif bad == 'profile': pins['object_source'] = 'surface'
        elif bad == 'flag_false': pins['allow_unobserved_poses'] = False
        elif bad == 'flag_integer': pins['allow_unobserved_poses'] = 1
        elif bad == 'missingflag': del pins['allow_unobserved_poses']
        elif bad == 'missing_record': pins['source_files'].pop(deps['object'])
        else: pins['source_files'][gate.dependency_paths(spec, object_source='surface')['object']] = pins['source_files'].pop(deps['object'])
    else:
        prepared = json.loads(prepared_path.read_text()); original = json.loads(original_path.read_text())
        if bad in ('leading', 'trailing'): original['pose_observed'][0 if bad == 'leading' else -1] = False
        elif bad == 'frames': original['frames'][1]['frame_index'] = 5
        elif bad == 'fabricated': original['frames'][1]['selected'] = original['frames'][0]['selected']
        elif bad == 'temporal': original['temporal_selection']['edge_extrapolation'] = True
        elif bad == 'sentinel': original['temporal_selection']['candidate_indices'][1] = 0
        elif bad == 'fake_status': original['frames'][1]['observation_status'] = 'measured_static_pose'
        elif bad == 'prepared_flag': prepared['object_pose_observations']['pose_observed'][1] = True
        elif bad == 'prepared_final': prepared['object_pose_observations']['final_prediction'] = True
        elif bad == 'prepared_index': prepared['object_pose_observations']['pose_observed_frame_index'][1] = True
        elif bad == 'fractionalflags': prepared['object_pose_observations']['pose_observed'][1] = 0
        elif bad in ('pickle_flag', 'pickle_final', 'pickle_index'):
            path = tmp_path/paths['object_poses']; payload = joblib.load(path)
            if bad == 'pickle_flag': payload['metadata']['pose_observed'][1] = True
            elif bad == 'pickle_final': payload['metadata']['final_prediction'] = True
            else: payload['metadata']['pose_observed_frame_index'][1] = True
            joblib.dump(payload, path)
        elif bad == 'posepath': prepared['object_pose_source']['geometry_and_poses'] = 'outputs/episode_000026/object_pose_full_surface/geometry_and_poses.npz'
        elif bad == 'oracle': original['oracle_modes'] = ['GT']
        write_json(original_path, original); write_json(prepared_path, prepared); pins = repin(gate, tmp_path, spec)
    with pytest.raises(ValueError): gate.verify_public_inputs(tmp_path, spec, pins)


def test_v4_source_admission_is_explicit_not_default_gate_relaxation(gate, tmp_path):
    spec, pins, _ = case(gate, tmp_path)
    for version, profile in [('v1', 'default'), ('v2', 'solid'), ('v3', 'surface')]:
        old = copy.deepcopy(pins); del old['allow_unobserved_poses']
        old['schema'] = 'world-reward-cari-clip-input-pins-'+version
        if version == 'v1': del old['object_source']
        else: old['object_source'] = profile
        with pytest.raises(ValueError): gate.validate_pins(spec, old)
