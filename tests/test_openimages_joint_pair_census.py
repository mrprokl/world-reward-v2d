"""Tiny manufactured metadata controls, never dataset/model accuracy."""
from copy import deepcopy

import pytest

from openimages_joint_pair_census import image_census

HUMAN = {'/m/man', '/m/person'}
PARTS = {'/m/hand'}


def bbox(cls, xy):
    return dict(LabelName=cls, XMin=str(xy[0]), YMin=str(xy[1]), XMax=str(xy[2]), YMax=str(xy[3]),
                IsGroupOf='0', IsDepiction='0', IsInside='0')


def relation(a, b):
    row = dict(LabelName1=a['LabelName'], LabelName2=b['LabelName'], RelationshipLabel='holds')
    for suffix, src in (('1', a), ('2', b)):
        row.update({k+suffix: src[k] for k in ('XMin', 'YMin', 'XMax', 'YMax')})
    return row


def sample():
    rows = [bbox('/m/man', (0, 0, .3, 1)), bbox('/m/man', (.6, 0, 1, 1)),
            bbox('/m/cup', (.1, .1, .2, .2)), bbox('/m/book', (.7, .1, .8, .2))]
    return rows, [relation(rows[0], rows[2])]


def test_new_metadata_gate_no_candidate_filter_or_mutation():
    boxes, relations = sample(); before = deepcopy((boxes, relations))
    result = image_census(boxes, relations, HUMAN, PARTS)
    assert result['metadata_eligible'] and result['resolved_positive_pairs'] == 1
    assert (boxes, relations) == before


def test_hierarchy_duplicate_is_not_a_second_person():
    boxes, relations = sample(); boxes[1] = bbox('/m/person', (0, 0, .3, 1))
    result = image_census(boxes, relations, HUMAN, PARTS)
    assert result['countable_person_boxes'] == 2
    assert not result['two_separated_person_boxes'] and not result['metadata_eligible']


@pytest.mark.parametrize('flag', ['IsGroupOf', 'IsDepiction', 'IsInside'])
@pytest.mark.parametrize('value', ['-1', '1'])
def test_unknown_group_depiction_inside_never_counted(flag, value):
    boxes, relations = sample(); boxes[0][flag] = value
    result = image_census(boxes, relations, HUMAN, PARTS)
    assert not result['metadata_eligible'] and result['unscorable_positive_pairs'] == 1


def test_exact_endpoint_does_not_override_ambiguous_overlap():
    boxes, relations = sample(); boxes.append(deepcopy(boxes[2]))
    result = image_census(boxes, relations, HUMAN, PARTS)
    assert result['positive_pairs'] == 1 and result['resolved_positive_pairs'] == 0
    assert result['unscorable_positive_pairs'] == 1


def test_positive_duplicate_only_is_deduplicated_not_distinct_objects():
    boxes, relations = sample(); relations *= 2
    result = image_census(boxes, relations, HUMAN, PARTS)
    assert result['positive_pairs'] == result['resolved_positive_pairs'] == 1
    boxes[3]['LabelName'] = '/m/hand'
    assert not image_census(boxes, relations, HUMAN, PARTS)['two_separated_object_boxes']


def test_relation_classes_not_hierarchy_aliased():
    boxes, relations = sample(); relations[0]['LabelName1'] = '/m/person'
    result = image_census(boxes, relations, HUMAN, PARTS)
    assert result['resolved_positive_pairs'] == 0 and not result['metadata_eligible']


def test_invalid_coordinate_fails_not_repaired():
    boxes, relations = sample(); boxes[0]['XMin'] = 'nan'
    with pytest.raises(ValueError):
        image_census(boxes, relations, HUMAN, PARTS)
