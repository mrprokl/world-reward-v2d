"""Tiny automatic census tests, no video/model or ownership accuracy claims."""
import copy
import numpy as np
import pytest

from person_pose_bank_probe import saved_person_banks, validate_paths


def diagnostic():
    rows = []
    for i, frame in enumerate(np.unique(np.linspace(0, 414, 16, dtype=int)).tolist()):
        row = dict(frame=frame, person_candidates=2, object_candidates=1)
        if i < 3:
            boxes = [dict(box=[1., 1., 8., 9.], score=.9), dict(box=[12., 1., 18., 9.], score=.8)]
            row['detector_observations'] = [dict(query='person.', boxes=boxes, retained=copy.deepcopy(boxes)),
                                             dict(query='object.', boxes=[], retained=[])]
        rows.append(row)
    return dict(episode=9, frames=415, confidence=.3, nms_iou=.7,
                detector_revision='12bdfa3120f3e7ec7b434d90674b3396eccf88eb',
                input_track='track_1', ground_truth_used=False, hand_labeled_test=False, observations=rows)


def test_all_original_person_slots_and_actual_seed_indices_not_first_three_frames():
    d = diagnostic()
    result = saved_person_banks(d, 24, 16, 415, 9)
    assert [x['frame_index'] for x in result] == [0, 27, 55]
    assert [len(x['person_ids']) for x in result] == [2, 2, 2]
    assert len(set(x for r in result for x in r['person_ids'])) == 6
    assert result[0]['boxes'].dtype == np.float64


def test_genuine_empty_census_retained_not_full_image_person():
    d = diagnostic()
    d['observations'][0]['person_candidates'] = 0
    d['observations'][0]['detector_observations'][0].update(boxes=[], retained=[])
    result = saved_person_banks(d, 24, 16, 415, 9)
    assert result[0]['person_ids'] == () and result[0]['boxes'].shape == (0, 4)
    assert result[0]['detector_scores'].shape == (0,)


@pytest.mark.parametrize('change', ['drop', 'retained_order', 'wrong_query', 'extra', 'indices', 'gt', 'threshold'])
def test_no_nms_target_filtering_interpolation_or_protocol_repair(change):
    d = diagnostic()
    if change == 'drop': d['observations'][0]['detector_observations'][0]['retained'].pop()
    if change == 'retained_order': d['observations'][0]['detector_observations'][0]['retained'].reverse()
    if change == 'wrong_query': d['observations'][0]['detector_observations'][0]['query'] = 'selected person.'
    if change == 'extra': d['observations'][3]['detector_observations'] = []
    if change == 'indices': d['observations'][1]['frame'] = 1
    if change == 'gt': d['ground_truth_used'] = True
    if change == 'threshold': d['confidence'] = .1
    with pytest.raises(ValueError): saved_person_banks(d, 24, 16, 415, 9)


def config():
    pin = dict(bytes=3, sha256='a'*64)
    rows = []
    for e in (9, 26):
        base = f'outputs/episode_{e:06d}/automatic_masks'
        row = dict(episode=e, total_frames=415, automatic_source_sha256='b'*64,
                   video=f'data/track_1/videos/chunk-000/episode_{e:06d}.mp4',
                   report=base+'/report.json', diagnostic=base+'/seed-diagnostics.json')
        row['files'] = {row[k]: pin.copy() for k in ('video', 'report', 'diagnostic')}
        rows.append(row)
    return dict(episodes=rows, metadata_files={x: pin.copy() for x in
                ('results/input-manifest.json', 'data/track_1/meta/episodes.jsonl')})


@pytest.mark.parametrize('change', ['gt', 'absolute', 'extra', 'episode', 'pin_bool', 'pin_invalid', 'video'])
def test_exact_mount_scope_before_model_or_read(change):
    p = config()
    validate_paths(p)
    if change == 'gt': p['metadata_files']['data/track_2/calibration.json'] = dict(bytes=3, sha256='a'*64)
    if change == 'absolute': p['episodes'][0]['video'] = '/etc/passwd'
    if change == 'extra': p['episodes'][0]['files']['.secrets/token'] = dict(bytes=3, sha256='a'*64)
    if change == 'episode': p['episodes'][0]['episode'] = True
    if change == 'pin_bool': next(iter(p['metadata_files'].values()))['bytes'] = True
    if change == 'pin_invalid': next(iter(p['metadata_files'].values()))['sha256'] = 'bad'
    if change == 'video': p['episodes'][0]['files'].pop(p['episodes'][0]['video'])
    with pytest.raises(ValueError): validate_paths(p)
