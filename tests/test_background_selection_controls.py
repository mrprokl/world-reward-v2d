"""Fresh manufactured counterexamples, not an ownership validation dataset.

Legacy selection is deliberately NOT repaired or retuned here. A passing test
can demonstrate that a sharply separated, stable wrong pair passes its gates.
The complete evidence bank preserves alternatives without certifying an owner.
No challenge frame, label, model or closed research record enters these tests.
"""
from dataclasses import replace

import numpy as np

from world_reward.actor_selection import select_interacting_actor
from world_reward.interaction_candidate_evidence import (
    GenericObjectObservations, build_interaction_candidate_evidence,
)
from world_reward.person_pose_observations import PersonPoseObservations
from world_reward.prompt_selection import BoxDetection, FrameDetections


SMALL_PERSON = (10., 20., 50., 100.)
LARGE_PERSON = (180., 0., 300., 120.)
SMALL_TARGET = (25., 65., 35., 75.)
BACKGROUND_TARGET = (235., 45., 245., 55.)


def _detections(background_score=.99):
    return tuple(FrameDetections(
        t, 320, 120,
        (BoxDetection(SMALL_PERSON, .70), BoxDetection(LARGE_PERSON, .95)),
        (BoxDetection(SMALL_TARGET, .55),
         BoxDetection(BACKGROUND_TARGET, background_score)),
    ) for t in (0, 11, 22))


def _observations():
    # Declared automatic output fixture: left small-person wrist/root sits on
    # SMALL_TARGET; the large-person hands sit elsewhere. This is not a model's
    # measured anatomical accuracy or human annotation of any test record.
    xy = np.full((2, 133, 2), 0., np.float64)
    xy[0, [9, 91]], xy[0, [10, 112]] = (30., 70.), (45., 95.)
    xy[1, [9, 91]], xy[1, [10, 112]] = (185., 115.), (295., 115.)
    people = PersonPoseObservations(
        0, (120, 320), ('small-person', 'large-person'),
        np.asarray((SMALL_PERSON, LARGE_PERSON), np.float64),
        np.asarray((.70, .95)), xy, np.ones((2, 133), np.float32),
    )
    objects = GenericObjectObservations(
        0, (120, 320), ('small-target', 'background-target'),
        np.asarray((SMALL_TARGET, BACKGROUND_TARGET)), np.asarray((.55, .99)),
    )
    return people, objects


def test_legacy_confident_background_pair_passes_despite_declared_other_interaction():
    original = _detections()
    selected = select_interacting_actor(original)
    # Known counterexample, NOT desired production behavior. Never present
    # gate success or this regression test as correct ownership.
    assert all(frame.persons == (BoxDetection(LARGE_PERSON, .95),)
               for frame in selected.frames)
    assert selected.scores[0].coverage == 1
    assert selected.scores[0].contaminated is False
    assert all(len(frame.objects) == 2 for frame in original)


def test_legacy_owner_changes_when_only_object_confidence_changes():
    high = select_interacting_actor(_detections(.99))
    low = select_interacting_actor(_detections(.35))
    # Geometry, timeline and person scores are identical. Detector confidence
    # alone can change owner; it cannot establish the actual interaction.
    assert high.frames[0].persons[0].box == LARGE_PERSON
    assert low.frames[0].persons[0].box == SMALL_PERSON


def test_complete_evidence_preserves_background_and_both_sides_without_selection():
    people, objects = _observations()
    evidence = build_interaction_candidate_evidence(people, objects)
    a = evidence.arrays
    assert evidence.features.shape == (2 * 2 * 2, 10)
    np.testing.assert_array_equal(a['person_slots'], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(a['side_indices'], [0, 0, 1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(a['generic_object_slots'], [0, 1, 0, 1] * 2)
    assert evidence.features[0, 0] == 0  # supplied small-person LEFT wrist
    assert evidence.features[2, 0] > 0   # RIGHT is a distinct anatomical slot
    assert evidence.features[5, 0] > 0   # large-person LEFT / background object
    assert evidence.features[0, 9] < evidence.features[1, 9]
    assert evidence.selection_performed is False
    assert evidence.anatomical_ownership_verified is False
    assert evidence.hoi_evidence is None  # no HOI => all objects still present


def test_missing_anatomy_keeps_targets_without_inventing_off_or_observations():
    people, objects = _observations()
    scores = people.raw_scores.copy()
    scores[0, [9, 91]] = 0
    missing = replace(people, raw_scores=scores)
    evidence = build_interaction_candidate_evidence(missing, objects)
    assert evidence.features.shape == (8, 10)
    assert not evidence.feature_supported[:2, :4].any()
    assert np.isnan(evidence.features[:2, :4]).all()
    assert evidence.scope['OFF'] != evidence.scope['UNKNOWN']
    assert evidence.scope['absence_predictions_generated'] is False
    np.testing.assert_array_equal(missing.keypoints_original_xy, people.keypoints_original_xy)
    np.testing.assert_array_equal(evidence.objects.boxes_original_xyxy, objects.boxes_original_xyxy)
