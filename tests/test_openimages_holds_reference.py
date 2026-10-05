"""Manufactured original-row controls; no dataset or empirical qualification."""
import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from world_reward.openimages_holds_reference import (
    PERSON_CLASSES, parse_holds_reference, parse_holds_vocabulary,
)

MAN, PERSON, WOMAN = "/m/04yx4", "/m/01g317", "/m/03bt1vf"
CUP, BOTTLE = "/m/02p5f1q", "/m/01xs3r"
ID = "0123456789abcdef"


def bbox(cls, box=(0, 0, 1, 1), group="0"):
    return dict(ImageID=ID, LabelName=cls, IsGroupOf=group,
        **dict(zip(("XMin", "YMin", "XMax", "YMax"), map(str, box))))


def holds(a=MAN, b=CUP, pa=(0, 0, 1, 1), pb=(.2, .3, .4, .5)):
    r = dict(ImageID=ID, RelationshipLabel="holds", LabelName1=a, LabelName2=b)
    for suffix, box in (("1", pa), ("2", pb)):
        r.update(dict(zip((k + suffix for k in ("XMin", "YMin", "XMax", "YMax")), map(str, box))))
    return r


def basic():
    return [bbox(MAN), bbox(CUP, (.2, .3, .4, .5))], [holds()]


def test_exact_normalized_regions_and_all_original_inventory_order():
    boxes, rows = basic()
    boxes += [bbox(WOMAN, (.6, 0, 1, 1), "1"), bbox(BOTTLE, (.6, .6, .8, .8), "1")]
    r = parse_holds_reference(boxes, rows)
    assert r.persons.class_ids == (MAN, WOMAN) and r.objects.class_ids == (CUP, BOTTLE)
    assert r.persons.instance_ids == ("bbox:00000000", "bbox:00000002")
    np.testing.assert_array_equal(r.persons.group_of, [False, True])
    np.testing.assert_array_equal(r.objects.group_of, [False, True])
    np.testing.assert_array_equal(r.holds, [[0, 0]])
    np.testing.assert_array_equal(r.positive_person_indices, [0])
    np.testing.assert_array_equal(r.positive_person_boxes_xyxy, [[0, 0, 1, 1]])
    assert r.total_positive_person_slots == 1 and r.unbound_positive_person_count == 0
    assert r.binding_diagnostics["exact_bindings"] == 2
    assert r.binding_diagnostics["iou_bindings"] == 0


def test_rounding_uses_unique_same_class_iou_without_snapping():
    boxes, rows = basic(); rows[0]["XMin2"] = ".20000003"
    before = copy.deepcopy(rows)
    r = parse_holds_reference(boxes, rows)
    np.testing.assert_array_equal(r.holds, [[0, 0]])
    assert r.binding_diagnostics["exact_bindings"] == 1
    assert r.binding_diagnostics["iou_bindings"] == 1
    np.testing.assert_array_equal(r.objects.boxes_xyxy, [[.2, .3, .4, .5]])
    assert rows == before


def test_iou_threshold_is_inclusive_half():
    boxes = [bbox(MAN), bbox(CUP, (0, 0, 1, .5))]
    r = parse_holds_reference(boxes, [holds(pb=(0, 0, 1, 1))])
    np.testing.assert_array_equal(r.holds, [[0, 0]])
    r = parse_holds_reference(boxes, [holds(pb=(0, 0, 1, 1), pa=(0, 0, .499999, 1))])
    assert not len(r.holds) and r.positive_person_indices[0] == -1


@pytest.mark.parametrize("endpoint", ["person", "object"])
@pytest.mark.parametrize("extra_group", ["0", "1", "-1"])
def test_exact_match_never_bypasses_ambiguous_or_group_overlap(endpoint, extra_group):
    boxes, rows = basic()
    boxes.append(bbox(MAN, (.01, 0, 1, 1), extra_group) if endpoint == "person"
                 else bbox(CUP, (.21, .3, .4, .5), extra_group))
    r = parse_holds_reference(boxes, rows)
    assert not len(r.holds) and r.unbound_positive_person_count == 1
    assert r.binding_diagnostics["ambiguous_endpoints"] == 1
    assert r.total_positive_person_slots == 1


@pytest.mark.parametrize("endpoint", ["person", "object"])
@pytest.mark.parametrize("group", ["1", "-1"])
def test_unique_group_positive_retains_unscorable_person_slot(endpoint, group):
    boxes, rows = basic(); boxes[int(endpoint == "object")]["IsGroupOf"] = group
    r = parse_holds_reference(boxes, rows)
    assert not len(r.holds) and r.total_positive_person_slots == 1
    assert r.unbound_positive_person_count == 1
    np.testing.assert_array_equal(r.unscorable_positive_person_slots, [0])
    assert r.binding_diagnostics["group_endpoints"] == 1
    assert r.binding_diagnostics["unknown_group_boxes"] == int(group == "-1")


def test_man_person_not_aliases_even_identical_geometry():
    boxes, rows = basic(); boxes[0]["LabelName"] = PERSON
    r = parse_holds_reference(boxes, rows)
    assert r.persons.class_ids == (PERSON,) and r.positive_person_classes == (MAN,)
    assert not len(r.holds) and r.positive_person_indices[0] == -1
    assert r.binding_diagnostics["no_match_endpoints"] == 1


def test_other_class_overlap_is_not_a_same_class_match():
    boxes, rows = basic(); boxes.append(bbox(BOTTLE, (.2, .3, .4, .5)))
    r = parse_holds_reference(boxes, rows)
    np.testing.assert_array_equal(r.holds, [[0, 0]])
    assert len(r.objects.instance_ids) == 2


def test_all_missing_positive_endpoints_retained_no_reference_boxes_added():
    r = parse_holds_reference([], [holds(), holds(a=WOMAN, pa=(0, 0, .4, .4))])
    assert r.persons.boxes_xyxy.shape == r.objects.boxes_xyxy.shape == (0, 4)
    assert r.holds.shape == (0, 2) and r.total_positive_person_slots == 2
    np.testing.assert_array_equal(r.positive_person_indices, [-1, -1])
    np.testing.assert_array_equal(r.unscorable_positive_person_slots, [0, 1])
    assert r.binding_diagnostics["endpoint_count"] == 3


def test_multiobject_duplicate_pair_and_partially_unbound_positive_person():
    boxes, rows = basic(); rows += [holds(), holds(b=BOTTLE, pb=(.6, .6, .8, .8))]
    r = parse_holds_reference(boxes, rows)
    np.testing.assert_array_equal(r.holds, [[0, 0]])
    assert r.total_positive_person_slots == 1 and r.unbound_positive_person_count == 1
    assert r.positive_person_indices[0] == 0 and r.positive_person_unscorable[0]
    assert r.binding_diagnostics["duplicate_positive_relations"] == 1
    assert r.binding_diagnostics["positive_relations"] == 2
    boxes.append(bbox(BOTTLE, (.6, .6, .8, .8)))
    r = parse_holds_reference(boxes, rows)
    np.testing.assert_array_equal(r.holds, [[0, 0], [0, 1]])
    assert r.unbound_positive_person_count == 0


def test_distinct_exact_endpoints_binding_same_instance_have_unique_holds():
    boxes, rows = basic(); other = holds(); other["XMin2"] = ".20000001"; rows.append(other)
    r = parse_holds_reference(boxes, rows)
    np.testing.assert_array_equal(r.holds, [[0, 0]])
    assert r.binding_diagnostics["positive_relations"] == 2
    assert r.binding_diagnostics["endpoint_count"] == 3
    assert r.binding_diagnostics["exact_bindings"] == 2 and r.binding_diagnostics["iou_bindings"] == 1


def test_duplicate_box_rows_preserved_as_ambiguous_not_dropped():
    boxes, rows = basic(); boxes.append(copy.deepcopy(boxes[1]))
    r = parse_holds_reference(boxes, rows)
    assert len(r.objects.instance_ids) == 2 and not len(r.holds)
    assert r.binding_diagnostics["ambiguous_endpoints"] == 1


def test_empty_and_noneligible_relations_have_no_fake_positive():
    r = parse_holds_reference([], [])
    assert r.holds.shape == (0, 2) and r.total_positive_person_slots == 0
    ignored = [dict(RelationshipLabel="on"), holds(a=CUP, b=BOTTLE), holds(b=WOMAN)]
    r = parse_holds_reference([], ignored)
    assert r.total_positive_person_slots == 0 and r.binding_diagnostics["ignored_relations"] == 3


@pytest.mark.parametrize("field,value", [("XMin", "nan"), ("YMin", "-0.01"),
    ("XMax", "1.01"), ("YMax", "0"), ("XMin", True), ("XMax", None),
    ("IsGroupOf", "2"), ("IsGroupOf", "0.0"), ("IsGroupOf", True), ("LabelName", "Person")])
def test_malformed_box_rows_fail_closed(field, value):
    boxes, rows = basic(); boxes[0][field] = value
    with pytest.raises(ValueError): parse_holds_reference(boxes, rows)


@pytest.mark.parametrize("field,value", [("XMin1", "inf"), ("YMax2", ".3"),
    ("XMin2", "-.01"), ("LabelName1", "Man"), ("RelationshipLabel", None)])
def test_malformed_positive_relation_rows_fail_closed(field, value):
    boxes, rows = basic(); rows[0][field] = value
    with pytest.raises(ValueError): parse_holds_reference(boxes, rows)


def test_missing_fields_and_mixed_original_image_ids_rejected():
    boxes, rows = basic(); del boxes[0]["IsGroupOf"]
    with pytest.raises(ValueError): parse_holds_reference(boxes, rows)
    boxes, rows = basic(); rows[0]["ImageID"] = "f" * 16
    with pytest.raises(ValueError): parse_holds_reference(boxes, rows)
    with pytest.raises(ValueError): parse_holds_reference([], [dict(RelationLabel="holds")])


def test_inputs_preserved_and_all_returned_arrays_byte_immutable():
    boxes, rows = basic(); before = copy.deepcopy((boxes, rows))
    r = parse_holds_reference(boxes, rows)
    boxes[0]["XMax"] = ".9"; rows[0]["XMax1"] = ".9"
    np.testing.assert_array_equal(r.persons.boxes_xyxy, [[0, 0, 1, 1]])
    boxes, rows = before; assert parse_holds_reference(boxes, rows).holds.shape == (1, 2)
    for a in [r.persons.boxes_xyxy, r.persons.group_of, r.objects.boxes_xyxy,
              r.holds, r.positive_person_boxes_xyxy, r.positive_person_indices,
              r.positive_person_unscorable, r.unscorable_positive_person_slots]:
        assert not a.flags.writeable
        with pytest.raises(ValueError): a.setflags(write=True)
    with pytest.raises(TypeError): r.binding_diagnostics["exact_bindings"] = 0
    with pytest.raises(FrozenInstanceError): r.total_positive_person_slots = 0


def test_vocabulary_exact_header_rows_sorted_deduplicated_and_no_hierarchy():
    rows = [("LabelName1", "LabelName2", "RelationshipLabel"), (MAN, CUP, "holds"),
            (WOMAN, BOTTLE, "holds"), (MAN, CUP, "holds"), (PERSON, CUP, "on"),
            (CUP, BOTTLE, "holds"), (MAN, WOMAN, "holds"), ("ignored", "ignored", "is")]
    assert parse_holds_vocabulary(rows) == tuple(sorted(((MAN, CUP), (WOMAN, BOTTLE))))
    assert (PERSON, CUP) not in parse_holds_vocabulary(rows)
    assert parse_holds_vocabulary([dict(LabelName1=MAN, LabelName2=CUP, RelationshipLabel="holds")]) == ((MAN, CUP),)
    assert len(PERSON_CLASSES) == 5 and parse_holds_vocabulary([]) == ()


@pytest.mark.parametrize("rows", [[("header", "bad")], [("Man", CUP, "holds")],
    [dict(LabelName1=MAN, LabelName2=CUP, RelationLabel="holds")], [(MAN, CUP, "")]])
def test_bad_vocabulary_schema_rejected(rows):
    with pytest.raises(ValueError): parse_holds_vocabulary(rows)
