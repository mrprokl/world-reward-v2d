"""Manufactured action-major poison controls; no dataset or HTTP."""
import hashlib
import io
import json

import pytest

import vcoco_role_stream as projection


def fixture():
    return [dict(role_object_id=[11, 12, 13, 21, 22, 23, 0, 24, 0], label=[1, 0, 1],
        image_id=[501, 502, 503], action_name='new_action', role_name=['agent', 'obj', 'instr'],
        ann_id=[11, 12, 13], extras=dict(marker='OLD_EXTRA_POISON')),
        dict(action_name='other_action', role_name=['agent'], ann_id=[13], image_id=[503],
             label=[[1]], role_object_id=[13])]


def project(raw, ids=frozenset((501, 503)), **kwargs):
    return projection.project_vcoco_actions(lambda: io.BytesIO(raw), ids, **kwargs)


@pytest.mark.parametrize('chunk', [1, 2, 7, 65536])
def test_role_major_selected_rows_and_original_provenance(chunk):
    raw = json.dumps(fixture()).encode()
    actions, report = project(raw, chunk_bytes=chunk)
    assert actions[0]['ann_id'] == [11, 13]
    assert actions[0]['label'] == [1, 1]
    assert actions[0]['role_object_id'] == [11, 13, 21, 23, 0, 0]
    assert actions[0]['projection_source'] == dict(action_slot=0, row_slots=[0, 2], row_count=3)
    assert actions[1]['label'] == [1] and actions[1]['role_object_id'] == [13]
    assert report['source_row_count'] == 4 and report['selected_row_count'] == 3
    assert report['source_identity'] == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert not report['excluded_semantic_values_decoded'] and not report['source_authenticated']


def test_excluded_poison_semantics_never_decoded_on_either_pass(monkeypatch):
    raw = b'[{"role_object_id":[11,12,21,1e999],"label":[1,1e999],"ann_id":[11,12],"image_id":[501,502],"role_name":["agent","obj"],"action_name":"new_action","extra":{"marker":"OLD_EXTRA_POISON","v":1e999}}]'
    decoded = []
    original = projection.js.strict_decode

    def recording(value):
        decoded.append(value)
        return original(value)

    monkeypatch.setattr(projection.js, 'strict_decode', recording)
    actions, _ = project(raw, frozenset((501,)), chunk_bytes=3)
    assert actions[0]['label'] == [1] and actions[0]['role_object_id'] == [11, 21]
    assert not any(b'1e999' in value or b'OLD_EXTRA_POISON' in value for value in decoded)
    with pytest.raises(ValueError, match='Nonfinite'):
        project(raw, frozenset((502,)))


def test_empty_eligible_preserves_action_schema_without_semantics():
    raw = json.dumps(fixture()).encode()
    actions, report = project(raw, frozenset())
    assert len(actions) == 2 and all(not a['ann_id'] and not a['role_object_id'] for a in actions)
    assert report['selected_row_count'] == 0
    assert project(b'[]')[0] == []


@pytest.mark.parametrize('fault', ['trailing', 'duplicate_key', 'role_nested', 'short_role',
    'extra_role', 'short_label', 'float_label', 'bool_role', 'wrong_label', 'missing_field',
    'mixed_ids', 'row_mismatch', 'wrong_agent_name', 'duplicate_roles', 'duplicate_action', 'invalid_skipped'])
def test_malformed_structure_selected_semantics_or_skipped_syntax_fail(fault):
    value = fixture()
    if fault == 'role_nested': value[0]['role_object_id'][0] = [11]
    elif fault == 'short_role': value[0]['role_object_id'].pop()
    elif fault == 'extra_role': value[0]['role_object_id'].append(0)
    elif fault == 'short_label': value[0]['label'].pop()
    elif fault == 'float_label': value[0]['label'][0] = 1.0
    elif fault == 'bool_role': value[0]['role_object_id'][0] = True
    elif fault == 'wrong_label': value[0]['label'][0] = -1
    elif fault == 'missing_field': del value[0]['ann_id']
    elif fault == 'mixed_ids': value[0]['ann_id'][0] = [11]
    elif fault == 'row_mismatch': value[0]['image_id'].pop()
    elif fault == 'wrong_agent_name': value[0]['role_name'][0] = 'person'
    elif fault == 'duplicate_roles': value[0]['role_name'][2] = 'obj'
    elif fault == 'duplicate_action': value[1]['action_name'] = value[0]['action_name']
    raw = json.dumps(value).encode()
    if fault == 'trailing': raw += b'tail'
    elif fault == 'duplicate_key': raw = raw.replace(b'"ann_id": [11, 12, 13]', b'"ann_id": [11, 12, 13], "ann_id": [11, 12, 13]')
    elif fault == 'invalid_skipped': raw = raw.replace(b'"OLD_EXTRA_POISON"', b'NaN')
    with pytest.raises((ValueError, UnicodeError)):
        project(raw, chunk_bytes=2)


def test_complete_bytes_change_between_passes_rejected_even_in_excluded_extras():
    raw = json.dumps(fixture()).encode()
    sources = iter((raw, raw.replace(b'OLD_EXTRA_POISON', b'OTHER_POISON')))
    with pytest.raises(ValueError, match='Complete source bytes changed'):
        projection.project_vcoco_actions(lambda: io.BytesIO(next(sources)), frozenset((501,)))


def test_structural_change_between_passes_rejected_before_output():
    raw = json.dumps(fixture()).encode()
    sources = iter((raw, raw.replace(b'501', b'509')))
    with pytest.raises(ValueError, match='Structural source changed'):
        projection.project_vcoco_actions(lambda: io.BytesIO(next(sources)), frozenset((501,)))


def test_limits_callbacks_and_bounded_reads():
    raw = json.dumps(fixture()).encode()
    reads, checks = [], []

    class Source(io.BytesIO):
        def read(self, count):
            assert 0 < count <= 7
            reads.append(count)
            return super().read(count)

    projection.project_vcoco_actions(lambda: Source(raw), frozenset((501,)),
                                     chunk_bytes=7, check=lambda: checks.append(True))
    assert reads and checks
    with pytest.raises(ValueError, match='stream byte'):
        project(raw, max_bytes=20)
    with pytest.raises(ValueError, match='row/token'):
        project(raw, field_bytes=4)


@pytest.mark.parametrize('ids', [[501], {True}, {501.0}, {-1}, {0}])
def test_invalid_eligible_ids_fail_before_source_open(ids):
    def forbidden():
        pytest.fail('Invalid selector opened source')

    with pytest.raises(ValueError):
        projection.project_vcoco_actions(forbidden, ids)


def test_structural_flat_and_nx1_vectors_supported():
    value = fixture()
    for action in value:
        for key in ('ann_id', 'image_id'):
            action[key] = [[i] for i in action[key]]
    assert project(json.dumps(value).encode())[0][0]['ann_id'] == [11, 13]


@pytest.mark.parametrize('key', ['max_bytes', 'field_bytes', 'chunk_bytes'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5, None])
def test_invalid_limits_fail_before_source_open(key, value):
    def forbidden():
        pytest.fail('Invalid limits opened source')

    with pytest.raises(ValueError):
        projection.project_vcoco_actions(forbidden, frozenset(), **{key: value})


def test_eligible_snapshot_cannot_change_between_passes():
    raw = json.dumps(fixture()).encode()
    eligible = {501}

    def callback():
        eligible.add(502)

    actions, _ = projection.project_vcoco_actions(lambda: io.BytesIO(raw), eligible, check=callback)
    assert actions[0]['image_id'] == [501]


@pytest.mark.parametrize('chunk', [1, 2, 7, 65536])
def test_filtered_coco_annotations_skip_historical_geometry_before_image_id(chunk, monkeypatch):
    raw = b'{"images":[{"id":501}],"annotations":[{"id":11,"bbox":[1e999],"marker":"OLD_POISON","image_id":502},{"bbox":[1,2,3,4],"image_id":501,"id":12}],"other":{"v":1e999}}'
    values = []
    original = projection.js.strict_decode

    def recording(value):
        values.append(value)
        return original(value)

    monkeypatch.setattr(projection.js, 'strict_decode', recording)
    rows = list(projection.iter_filtered_coco_annotations(io.BytesIO(raw), {501}, chunk_bytes=chunk))
    assert rows == [(501, dict(bbox=[1, 2, 3, 4], image_id=501, id=12))]
    assert not any(b'OLD_POISON' in value or b'1e999' in value for value in values)


@pytest.mark.parametrize('raw', [b'{}', b'{"annotations":{}}', b'{"annotations":[]}tail',
    b'{"annotations":[{"image_id":true}]}', b'{"annotations":[{"x":1}]}',
    b'{"annotations":[{"image_id":502,"x":NaN}]}', b'{"annotations":[],"annotations":[]}'])
def test_filtered_coco_complete_syntax_and_structural_ids_fail(raw):
    with pytest.raises(ValueError):
        list(projection.iter_filtered_coco_annotations(io.BytesIO(raw), {501}, chunk_bytes=2))
