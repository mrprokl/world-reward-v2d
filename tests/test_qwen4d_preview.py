"""Tiny synthetic contracts only: no challenge media, model or GPU execution."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import qwen4d_preview as preview


def test_original_body_sampling_not_old_preview_rounding():
    assert preview.frame_indices(634) == [0, 317, 633]
    assert preview.frame_indices(415) == [0, 207, 414]
    assert preview.frame_indices(399) == [0, 199, 398]
    for bad in (True, 3., 2, 0):
        with pytest.raises(ValueError):
            preview.frame_indices(bad)


def test_namespace_never_adopts_historical_outputs_or_other_revision(tmp_path):
    revision = 'a'*40
    prefix = f'experiments/qwen4d-v1-{revision}/outputs'
    assert preview.candidate_base(tmp_path, prefix, revision, 8) == tmp_path/prefix/'episode_000008'
    for bad in ('outputs', '/outputs', '../outputs', prefix+'/..', prefix.replace('a'*40, 'b'*40)):
        with pytest.raises(ValueError):
            preview.candidate_base(tmp_path, bad, revision, 8)
    with pytest.raises(ValueError):
        preview.candidate_base(tmp_path, prefix, revision, True)


def test_native_wxyz_scale_once_then_rotate_and_translate_without_mutation():
    vertices = np.array([[1., 0, 1], [0, 1, 1], [0, 0, 2]])
    original = vertices.copy()
    transform = dict(rotation=[np.sqrt(.5), 0, 0, np.sqrt(.5)], scale=[2., 3., 4.], translation=[5., 6., 10.])
    actual = preview.object_camera_vertices(vertices, transform)
    np.testing.assert_allclose(actual, [[5., 8., 14.], [2., 6., 14.], [5., 6., 18.]], atol=1e-12)
    np.testing.assert_array_equal(vertices, original)


@pytest.mark.parametrize('field,value', [('rotation', [0, 0, 0, 0]), ('rotation', [np.nan, 0, 0, 1]),
    ('scale', [1, 0, 1]), ('translation', [0, 0, -10]), ('translation', [0, 0])])
def test_bad_transform_fails_without_clipping_or_rescue(field, value):
    transform = dict(rotation=[1, 0, 0, 0], scale=[1, 1, 1], translation=[0, 0, 5])
    transform[field] = value
    with pytest.raises(ValueError):
        preview.object_camera_vertices(np.array([[0., 0, 1], [1, 0, 1], [0, 1, 1]]), transform)


def camera_inputs():
    K = np.array([[1920., 0, 768.], [0, 1920., 576.], [0, 0, 1.]])
    indices = [0, 317, 633]
    evidence = [dict(frame_index=i, estimated_body_K=K.tolist()) for i in indices]
    intrinsics = dict(fx=1920., fy=1920., cx=768., cy=576., width=1536, height=1152)
    return K, indices, evidence, intrinsics


def test_common_camera_is_exact_not_replaced_or_close_aligned():
    K, indices, evidence, intrinsics = camera_inputs()
    np.testing.assert_array_equal(preview.shared_camera(np.full(3, 1920.), evidence, intrinsics,
        indices, 1536, 1152), K)
    intrinsics['fx'] += 1e-8
    with pytest.raises(ValueError, match='Shared camera differs'):
        preview.shared_camera(np.full(3, 1920.), evidence, intrinsics, indices, 1536, 1152)
    intrinsics['fx'] = 1920.
    evidence[1]['estimated_body_K'][1][2] += 1e-8
    with pytest.raises(ValueError, match='Shared camera differs'):
        preview.shared_camera(np.full(3, 1920.), evidence, intrinsics, indices, 1536, 1152)


def test_camera_order_and_resolution_cannot_be_repaired():
    _, indices, evidence, intrinsics = camera_inputs()
    with pytest.raises(ValueError):
        preview.shared_camera(np.full(3, 1920.), evidence[::-1], intrinsics, indices, 1536, 1152)
    intrinsics['height'] = 240
    with pytest.raises(ValueError):
        preview.shared_camera(np.full(3, 1920.), evidence, intrinsics, indices, 1536, 1152)


def test_provenance_rejects_truth_or_manual_labels_including_false_like_types():
    report = dict(stage='body', status='pass', episode_index=8, input_track='track_1',
        input_sha256='a'*64, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    preview.report_contract(report, 'body', 8, 'a'*64)
    for key, value in (('ground_truth_used', True), ('ground_truth_used', 0),
            ('hand_labeled_test', True), ('oracle_modes', ['gt']), ('episode_index', 8.)):
        bad = dict(report); bad[key] = value
        with pytest.raises(ValueError):
            preview.report_contract(bad, 'body', 8, 'a'*64)


def test_strict_json_rejects_duplicate_and_nonfinite_fields():
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        with pytest.raises(ValueError):
            preview.strict_json(raw)


def test_source_binding_is_verified_before_and_after_read(tmp_path):
    path = tmp_path/'report.json'; path.write_text(json.dumps({'status': 'pass'}))
    sources = preview.Sources()
    pin = preview.identity(path)
    assert sources.json(path, pin) == {'status': 'pass'}
    sources.verify()
    path.write_text(json.dumps({'status': 'fail'}))
    with pytest.raises(ValueError):
        sources.verify()
    with pytest.raises(ValueError):
        preview.Sources().bind(path, pin)


def test_source_binding_rejects_symlinks_and_hardlinks(tmp_path):
    import os
    path = tmp_path/'file'; path.write_bytes(b'tiny')
    link = tmp_path/'link'; link.symlink_to(path)
    with pytest.raises(ValueError):
        preview.identity(link)
    hard = tmp_path/'hard'; os.link(path, hard)
    with pytest.raises(ValueError):
        preview.identity(hard)


def test_absent_candidate_object_is_not_drawn_as_a_static_trajectory():
    rgb = np.full((2, 2, 3), 100, np.uint8)
    human = np.array([[2., np.inf], [np.inf, np.inf]])
    absent = np.full((2, 2), np.inf)
    actual = preview.overlay(rgb, human, absent)
    np.testing.assert_array_equal(actual[1], rgb[1])
    assert not np.array_equal(actual[0, 0], rgb[0, 0])


def test_mask_overlay_has_no_depth_order_in_overlap_and_does_not_modify_inputs():
    rgb = np.full((1, 3, 3), 100, np.uint8)
    human = np.array([[True, False, True]])
    obj = np.array([[False, True, True]])
    result = preview.mask_overlay(rgb, human, obj)
    np.testing.assert_array_equal(result[0, 2],
        np.rint(48 + .52*(np.asarray(preview.HUMAN_RGB)+np.asarray(preview.OBJECT_RGB))/2))
    np.testing.assert_array_equal(rgb, 100)


def failed_experiment():
    return dict(schema='world_reward.qwen4d_initializers.v1', status='complete_diagnostic_not_quality_pass',
        phase='complete', producer_revision=preview.FAILED_PRODUCER, baseline_modified=False,
        model_math_changed=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        full_4D_run=False, quality_verified=False, source_rehashed_after=True,
        episodes=[dict(episode=ep, status='fail', phase='object_grounded', error_type='ValueError',
            stages=[dict(stage=stage, report={'bytes': 123, 'sha256': 'a'*64})
                for stage in ('body_smoke', 'depth_smoke', 'scale_smoke')]) for ep in (8, 9, 26)])


def test_failure_review_is_explicit_and_original_failures_are_not_promoted():
    import inspect
    assert inspect.signature(preview.run).parameters['body_only_failure_review'].default is False
    value = failed_experiment()
    assert len(preview.failure_review_contract(value, preview.FAILED_PRODUCER)) == 3
    value['episodes'][0]['status'] = 'pass'
    with pytest.raises(ValueError):
        preview.failure_review_contract(value, preview.FAILED_PRODUCER)


@pytest.mark.parametrize('field,value', [('status', 'pass'), ('phase', 'running'),
    ('ground_truth_used', 0), ('full_4D_run', True), ('source_rehashed_after', False)])
def test_failure_review_rejects_status_reclassification_or_unverified_sources(field, value):
    report = failed_experiment(); report[field] = value
    with pytest.raises(ValueError):
        preview.failure_review_contract(report, preview.FAILED_PRODUCER)


def test_failure_review_requires_the_complete_exact_three_clip_boundary():
    for change in ('episode_order', 'failure_phase', 'error', 'stage_order', 'revision'):
        report = failed_experiment()
        if change == 'episode_order': report['episodes'].reverse()
        if change == 'failure_phase': report['episodes'][0]['phase'] = 'preview'
        if change == 'error': report['episodes'][0]['error_type'] = 'RuntimeError'
        if change == 'stage_order': report['episodes'][0]['stages'].reverse()
        if change == 'revision': report['producer_revision'] = 'b'*40
        with pytest.raises(ValueError):
            preview.failure_review_contract(report, preview.FAILED_PRODUCER)
