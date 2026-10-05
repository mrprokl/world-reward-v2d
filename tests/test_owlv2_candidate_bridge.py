"""Tiny full-grid manufactured controls only; no RGB/model/GPU/quality."""
from dataclasses import replace

import numpy as np
import pytest

from world_reward.owlv2_candidate_bridge import bridge_owlv2_candidates
from world_reward.owlv2_object_observations import (
    Owlv2ObjectObservations, SCHEMA, GRID, PATCH_COUNT,
)


def bank():
    boxes = np.tile(np.array([.5, .5, 1., 1.], np.float32), (PATCH_COUNT, 1))
    boxes[0] = 0.
    corners = np.concatenate((boxes[:, :2]-boxes[:, 2:]/2,
                              boxes[:, :2]+boxes[:, 2:]/2), axis=1)*np.float32(20)
    return Owlv2ObjectObservations(17, (10, 20), GRID,
        np.arange(PATCH_COUNT, dtype=np.int64), boxes,
        np.arange(PATCH_COUNT, dtype=np.float32)-3000, corners)


def test_lossless_native_logits_ids_all_rows_zero_and_padded_duplicates():
    b = bank(); g = bridge_owlv2_candidates(b)
    assert g.original_frame_index == 17 and g.image_size == (10, 20)
    assert g.object_ids == b.object_ids and len(g.object_ids) == 3600
    assert g.raw_scores.dtype == np.float32
    assert np.array_equal(g.raw_scores, b.objectness_logits)
    assert np.array_equal(g.boxes_original_xyxy, b.boxes_original_xyxy)
    assert g.raw_scores[0] == -3000 and g.raw_scores[-1] == 599
    assert not g.box_positive_area[0] and g.box_positive_area[1:].all()
    assert not g.box_in_original_image.any()  # preserved padding, not clipped
    assert len(np.unique(g.boxes_original_xyxy, axis=0)) == 2
    assert g.provenance_references[0][0] == SCHEMA
    assert len(g.provenance_references[0][1]) == 64


def test_bridge_independent_immutable_output_no_alias_with_mutable_native_owner():
    b = bank(); g = bridge_owlv2_candidates(b)
    for source, target in ((b.objectness_logits, g.raw_scores),
                           (b.boxes_original_xyxy, g.boxes_original_xyxy)):
        assert not np.shares_memory(source, target) and not target.flags.writeable
        with pytest.raises(ValueError): target.flags.writeable = True
    b.objectness_logits.flags.writeable = True; b.objectness_logits[:] = 0
    assert g.raw_scores[0] == -3000


def test_native_fingerprint_includes_scores_geometry_frame_and_grid():
    b = bank(); first = bridge_owlv2_candidates(b).provenance_references
    for other in (replace(b, original_frame_index=18),
                  replace(b, objectness_logits=b.objectness_logits+np.float32(1))):
        assert bridge_owlv2_candidates(other).provenance_references != first


@pytest.mark.parametrize('field', ['objectness_logits', 'boxes_original_xyxy', 'patch_ids'])
def test_forcibly_mutated_invalid_native_field_rejected_without_repair(field):
    b = bank(); a = getattr(b, field); a.flags.writeable = True
    if field == 'patch_ids': a[0] = 99
    else: a.flat[0] = np.nan
    with pytest.raises(ValueError): bridge_owlv2_candidates(b)


@pytest.mark.parametrize('bad', [None, {}, [], 'native-bank'])
def test_not_a_native_bank_rejected(bad):
    with pytest.raises(ValueError): bridge_owlv2_candidates(bad)
