"""Small evidence/numeric/delegation checks; never local native/model inference."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))
from world_reward import native_joint_refinement as joint


def evidence(n=6, q=33):
    points = np.c_[np.linspace(-.1, .1, q), np.linspace(.1, -.1, q), np.zeros(q)]
    K = np.array([[800., 0, 320], [0, 800, 240], [0, 0, 1]])
    xy = points[:, :2] / 2 * [800, 800] + [320, 240]
    return joint.JointImageEvidence(points, np.broadcast_to(xy, (n, q, 2)).copy(),
        np.ones((n, q), bool), np.zeros((n, 133, 2)), np.ones((n, 133), np.float32),
        K, np.arange(n, dtype=np.int64))


def config():
    return joint.NativeJointConfig(5., 1., 1., 10., .1, 1e-4, 'published_native_proposal_tiny_DEV_only')


def test_all_33_points_and_full_occlusions_remain_literal_immutable():
    e = evidence(); xy = e.object_xy.copy(); supported = e.object_supported.copy()
    supported[2] = False; xy[2] = np.nan
    updated = replace(e, object_xy=xy, object_supported=supported)
    assert len(updated.points) == 33 and len(updated.frame_index) == 6
    assert not updated.object_supported[2].any() and np.isnan(updated.object_xy[2]).all()
    for value in vars(updated).values():
        if not isinstance(value, np.ndarray): continue
        assert not value.flags.writeable
        with pytest.raises(ValueError): value.flags.writeable = True
    assert updated.points.dtype == e.points.dtype


@pytest.mark.parametrize('fault', ['timeline', 'unsupported_fill', 'supported_nan', 'skew', 'score_nan', 'bool_index'])
def test_no_chronology_support_camera_or_native_score_repair(fault):
    e = evidence(); values = {}
    if fault == 'timeline': values['frame_index'] = np.arange(6, dtype=np.int64) + 1
    elif fault == 'bool_index': values['frame_index'] = np.ones(6, bool)
    elif fault == 'unsupported_fill':
        m = e.object_supported.copy(); m[1, 0] = False; values['object_supported'] = m
    elif fault == 'supported_nan':
        xy = e.object_xy.copy(); xy[1, 0] = np.nan; values['object_xy'] = xy
    elif fault == 'skew':
        K = e.K.copy(); K[0, 1] = 1; values['K'] = K
    else:
        s = e.human_scores.copy(); s[1, 0] = np.nan; values['human_scores'] = s
    with pytest.raises(ValueError): replace(e, **values)


def test_numeric_DEV_correct_geometry_beats_separated_pair_without_static_motion():
    e = evidence(); xyz = np.broadcast_to(e.points, (6, 33, 3)).copy(); xyz[..., 2] += 2
    xyz[:, :, 0] += np.arange(6)[:, None] * .02
    xy = xyz[..., :2] / xyz[..., 2:] * np.diag(e.K)[:2] + e.K[:2, 2]
    before = xyz.copy(); before[:, :, 2] += .2
    a = joint.reprojection_statistics(before, e.K, xy, e.object_supported, 5., omit_initializer=True)
    b = joint.reprojection_statistics(xyz, e.K, xy, e.object_supported, 5., omit_initializer=True)
    assert a['loss'] > 0 and b['loss'] == 0 and b['observations'] == 5 * 33
    # This checks a factor, NOT native fitting / external HOI quality.
    assert np.ptp(xyz[:, 0, 0]) > 0


def test_numeric_factor_unknowns_never_generate_nan_loss_or_hide_slots():
    e = evidence(); xyz = np.broadcast_to(e.points, (6, 33, 3)).copy(); xyz[..., 2] += 2
    support = e.object_supported.copy(); support[:, 1] = False
    xy = e.object_xy.copy(); xy[:, 1] = np.nan
    row = joint.reprojection_statistics(xyz, e.K, xy, support, 5., omit_initializer=True)
    assert row['loss'] == 0 and row['unsupported_points'] == 1 and row['observations'] == 5 * 32
    xyz[-1, -1, 2] = 0
    with pytest.raises(ValueError): joint.reprojection_statistics(xyz, e.K, xy, support, 5.)


@pytest.mark.parametrize('field,value', [('image_sigma_px', 0), ('human_translation_lr', 0),
    ('object_image_weight', -1), ('human_image_weight', True), ('human_translation_prior_weight', np.inf),
    ('development_reference', '')])
def test_coefficients_have_no_silent_defaults_or_episode_tuning(field, value):
    with pytest.raises(ValueError): replace(config(), **{field: value})


def test_zero_extension_exactly_delegates_original_objects_and_parameter_groups(monkeypatch):
    loss = (object(), {'loss_total': object()}); decoded = object(); result = {'pr': object()}; groups = [object()]
    class Base:
        def __init__(self, bundle, vertices, faces, cfg, *, mhr_layer): self.frame_indices = np.arange(6)
        def loss(self, indices, step, *, include_diagnostics=True): return loss
        def _decode_body(self, indices): return decoded
        def result(self): return result
        def _optimizer_parameter_groups(self): return groups
    monkeypatch.setattr(joint, '_native_binding', lambda native: (Path('not_native'), Base, lambda: None))
    cfg = SimpleNamespace(num_steps=300, batch_size=0, frame_start=0, frame_limit=0,
        freeze_object_rotation=True, freeze_body_internal_translations=True, checkpoint_path=None)
    bundle = dict(gt={}, metadata={'ground_truth_used': False}, frames=[f'{i:06d}' for i in range(6)])
    cls = joint.native_joint_optimizer_class(SimpleNamespace(), evidence(), config(), enabled=False)
    instance = cls(bundle, None, None, cfg, mhr_layer=None)
    assert instance.loss(None, 181) is loss and instance._decode_body(None) is decoded
    assert instance.result() is result and instance._optimizer_parameter_groups() is groups
    assert not hasattr(instance, 'human_translation')
    cfg.freeze_object_rotation = False
    with pytest.raises(ValueError): cls(bundle, None, None, cfg, mhr_layer=None)


def test_joint_translation_is_separate_and_no_oracle_file_io_in_numeric_module():
    import inspect, ast
    tree = ast.parse(inspect.getsource(joint))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    forbidden = {'open', 'load', 'read_bytes', 'read_text', 'VideoCapture', 'infer_metric_video'}
    assert not any(isinstance(c.func, ast.Name) and c.func.id in forbidden for c in calls)
    text = inspect.getsource(joint)
    assert "self.human_translation" in text and "params['mhr_trans']" in text
    assert "groups.append" in text and "super().loss" in text
    assert 'freeze_object_rotation=True' in text and 'checkpoint_path=None' in text


def test_textureless_fallback_is_explicit_and_never_replaces_33_tracks():
    e = evidence(); values = dict(points=np.empty((0, 3)), object_xy=np.empty((6, 0, 2)),
        object_supported=np.empty((6, 0), bool))
    with pytest.raises(ValueError): replace(e, **values)
    fallback = replace(e, **values, allow_empty_object_evidence=True)
    assert fallback.object_xy.shape == (6, 0, 2) and len(fallback.object_counts()) == 0
    assert e.object_xy.shape == (6, 33, 2)


def test_absent_automatic_actor_keeps_full_T_no_body_fill():
    e = evidence(); xy=e.human_xy.copy(); scores=e.human_scores.copy()
    xy[2] = np.nan; scores[2] = 0
    e = replace(e, human_xy=xy, human_scores=scores)
    assert np.isnan(e.human_xy[2]).all() and not (e.human_scores[2] > 0).any()
    scores[2, 3] = 1
    with pytest.raises(ValueError): replace(e, human_scores=scores)


def test_frozen_runner_protocol_and_factor_DEV_are_not_heldout_claims():
    import native_joint_real as runner
    cfg, proposal = runner.settings(Path(__file__).resolve().parents[1])
    assert cfg['cohort']['episodes'] == [9, 1, 14, 7]
    assert cfg['fallback_native_silhouette_contact_no_material_tracks_episodes'] == [14]
    assert not cfg['production_adopted'] and cfg['budget_seconds'] == 1800
    assert runner.controlled_factor_DEV()['passed']
    assert not runner.controlled_factor_DEV()['native_optimizer_tested']
    assert proposal.human_translation_prior_weight == 10


def test_predeclared_QA_rejects_worse_contact_even_with_good_RGB():
    import native_joint_real as runner
    cfg, _ = runner.settings(Path(__file__).resolve().parents[1])
    a = dict(RGB_mean_px=3., RGB_observations=20, RGB_motion_increment_error_mean_px=2.,
        reserved_human_RGB_mean_px=2., same_anatomical_contact_mean_m=.01,
        same_anatomical_contact_p95_m=.02, object_acceleration_p95_m_s2=2.,
        object_angular_acceleration_p95_rad_s2=2., human_centroid_acceleration_p95_m_s2=2.)
    b = dict(a, RGB_mean_px=1., same_anatomical_contact_mean_m=.011)
    result = runner.quality_decision(cfg, a, b)
    assert not result['passed'] and not result['gates']['same_anatomical_contact_mean_m_nonworse']
    assert not result['production_adopted'] and not result['heldout_4D_accuracy_verified']
    a.update(RGB_observations=0, RGB_mean_px=None, RGB_motion_increment_error_mean_px=None)
    b = dict(a)
    result = runner.quality_decision(cfg, a, b)
    assert result['passed'] and 'RGB_mean_px_nonworse' not in result['gates']


def test_prior_only_witness_selection_is_same_anatomical_ID_not_moved_world_point(monkeypatch):
    import native_joint_real as runner
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    bank = dict(vertices=v, faces=np.array([[0,1,2]]), translations=np.zeros((3,3)),
        rotations=np.broadcast_to(np.eye(3),(3,3,3)).copy())
    human = np.broadcast_to(np.array([[0.,0,0],[.5,.5,0],[2,2,0]]),(3,3,3)).copy()
    src = dict(human=human, QA_hand_ids=np.array([[0,1],[1,2]]))
    activation=np.array([[True,False],[False,True],[False,False]])
    ids=runner.frozen_anatomical_witness_ids(bank, src, activation)
    assert ids.tolist()==[[0,-1],[-1,1],[-1,-1]]


def test_wrapper_is_offline_exact_image_GPU_singlelease_and_frozen_timeout():
    import subprocess
    path=Path(__file__).resolve().parents[1]/'infra/run_native_joint_real.sh'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert '--network none --read-only' in text and 'flock -n 8' in text
    assert '1803s docker run' in text and '--cidfile' in text and 'docker rm -f' in text
    assert 'easydict' not in text and 'VDA' not in text


def test_same_producer_has_distinct_episode_CIDs_and_never_overwrites_outputs():
    import inspect, native_joint_real as runner
    path=Path(__file__).resolve().parents[1]/'infra/run_native_joint_real.sh'
    text=path.read_text()
    assert 'OUT="$BASE/episode_$PADDED"' in text
    assert 'NAME="wr-native-joint-$REV-$PADDED"' in text
    assert "out.mkdir(mode=0o755)" in text and "done=p/'host-exit.json'" in text
    assert "reserved_output(out, revision, episode)" in inspect.getsource(runner.run)
    assert "{'.container.cid'}" in inspect.getsource(runner.reserved_output)
