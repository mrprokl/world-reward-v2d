from dataclasses import asdict
import json

import pytest

from world_reward.actor_selection import select_interacting_actor
from world_reward.prompt_selection import BoxDetection, FrameDetections, select_seed_prompts


ACTOR = (5, 10, 45, 90)
DISTRACTOR = (140, 10, 180, 90)


def box(coordinates, score=0.8):
    return BoxDetection(coordinates, score)


def target(x=30, y=50, score=0.9):
    return box((x - 5, y - 5, x + 5, y + 5), score)


def frame(index, persons=None, objects=None):
    return FrameDetections(
        index, 200, 100,
        (box(ACTOR, 0.75), box(DISTRACTOR, 0.99)) if persons is None else persons,
        (target(),) if objects is None else objects,
    )


def test_interaction_evidence_beats_higher_confidence_distractor_and_preserves_seed_schema():
    inputs = [frame(20), frame(0), frame(10)]
    actor = select_interacting_actor(inputs)
    assert [f.frame_index for f in actor.frames] == [0, 10, 20]
    assert all(f.persons == (box(ACTOR, 0.75),) for f in actor.frames)
    assert all(f.objects is inputs[[f.frame_index for f in inputs].index(f.frame_index)].objects
               for f in actor.frames)
    assert actor.scores[0].outside_distance == 0
    assert actor.scores[0].coverage == 1
    json.dumps([asdict(score) for score in actor.scores], allow_nan=False)
    seed = select_seed_prompts(actor.frames, object_prompt="exact public target")
    assert seed.frame_index == 0
    assert seed.person.box == tuple(float(x) for x in ACTOR)


def test_person_input_order_does_not_change_track_identity_or_result():
    normal = [frame(i) for i in range(3)]
    reversed_boxes = [frame(i, persons=tuple(reversed(f.persons))) for i, f in enumerate(normal)]
    a = select_interacting_actor(normal)
    b = select_interacting_actor(list(reversed(reversed_boxes)))
    assert a == b


def test_equally_close_distinct_people_fail_affinity_not_confidence_tiebreak():
    a = box((10, 10, 50, 90), 0.99)
    b = box((70, 10, 110, 90), 0.75)
    with pytest.raises(ValueError, match="Ambiguous actor affinity"):
        select_interacting_actor([frame(i, (a, b), (target(60),)) for i in range(3)])


def test_zero_outside_distance_ties_use_center_affinity_with_clear_margin():
    tight = box((10, 20, 50, 80), 0.8)
    enclosing = box((0, 0, 180, 100), 0.99)
    actor = select_interacting_actor([frame(i, (tight, enclosing)) for i in range(3)])
    assert all(f.persons == (tight,) for f in actor.frames)
    assert actor.scores[0].outside_distance == actor.scores[1].outside_distance == 0
    assert actor.scores[1].center_distance - actor.scores[0].center_distance > 0.05


def test_moving_actor_object_beats_stationary_distractor():
    frames = []
    for i in range(5):
        shift = i * 5
        actor = box((5 + shift, 10, 45 + shift, 90), 0.75)
        frames.append(frame(i * 12, (actor, box(DISTRACTOR, 0.99)), (target(30 + shift),)))
    result = select_interacting_actor(frames)
    assert result.scores[0].observations == 5
    assert all(f.persons[0].box[0] == 5 + i * 5 for i, f in enumerate(result.frames))


def test_missing_person_observation_keeps_identity_but_never_fabricates_box():
    frames = [frame(i) for i in range(5)]
    frames[2] = frame(2, persons=(box(DISTRACTOR, 0.99),))
    result = select_interacting_actor(frames)
    assert result.scores[0].observations == 4
    assert result.scores[0].coverage == 0.8
    assert result.frames[2].persons == ()
    assert result.frames[3].persons == (box(ACTOR, 0.75),)


def test_symmetric_crossing_association_rejects_instead_of_identity_flip_or_restart():
    left, right = box((10, 10, 50, 90)), box((70, 10, 110, 90))
    middle_a, middle_b = box((35, 10, 75, 90)), box((45, 10, 85, 90))
    with pytest.raises(ValueError, match="Ambiguous person-track association"):
        select_interacting_actor([
            frame(0, (left, right)), frame(1, (middle_a, middle_b)),
            frame(2, (right, left)),
        ], association_margin=0.5)


def test_exact_crossing_tie_rejects_at_default_gate():
    a, b = box((10, 10, 60, 90)), box((40, 10, 90, 90))
    # Two distinct detections have equal association overlap to both old tracks.
    c, d = box((25, 10, 75, 90)), box((25, 11, 75, 89))
    with pytest.raises(ValueError, match="Ambiguous person-track association"):
        select_interacting_actor([frame(0, (a, b)), frame(1, (c, d)), frame(2)])


def test_two_supported_person_observations_not_enough():
    with pytest.raises(ValueError, match="insufficient observations"):
        select_interacting_actor([frame(0), frame(1), frame(2, persons=())])


def test_less_than_half_object_frames_person_support_fails():
    frames = [frame(i, persons=(box(ACTOR),) if i < 3 else ()) for i in range(7)]
    with pytest.raises(ValueError, match="coverage"):
        select_interacting_actor(frames)


def test_exact_three_and_half_coverage_accepted_without_missing_boxes_filled():
    frames = [frame(i, persons=(box(ACTOR),) if i < 3 else ()) for i in range(6)]
    result = select_interacting_actor(frames)
    assert result.scores[0].observations == 3
    assert result.scores[0].coverage == 0.5
    assert all(f.persons == () for f in result.frames[3:])


def test_unsupported_closer_track_not_discarded_to_manufacture_distractor_winner():
    frames = [frame(i, persons=(box(ACTOR), box(DISTRACTOR)) if i < 2
                    else (box(DISTRACTOR),)) for i in range(5)]
    with pytest.raises(ValueError, match="insufficient observations"):
        select_interacting_actor(frames)


def test_expired_identity_fragments_not_joined_to_pass_support_gate():
    frames = [frame(i, persons=(box(ACTOR),) if i in (0, 1, 4, 5) else ()) for i in range(6)]
    with pytest.raises(ValueError, match="insufficient observations"):
        select_interacting_actor(frames)


def test_missing_ambiguous_or_invalid_object_frames_are_not_affinity_evidence():
    frames = [frame(i) for i in range(5)]
    frames[1] = frame(1, objects=(target(30, score=0.9), target(170, score=0.88)))
    frames[3] = frame(3, objects=(box((-1, 0, 10, 10), 0.99),))
    result = select_interacting_actor(frames)
    assert result.scores[0].observations == 3
    assert result.scores[0].coverage == 1
    assert len(result.frames[1].objects) == 2  # original seed gate still rejects it


def test_all_ambiguous_object_frames_fail():
    with pytest.raises(ValueError, match="Insufficient unambiguous object"):
        select_interacting_actor([frame(i, objects=(target(30), target(170))) for i in range(4)])


def test_median_affinity_not_single_noisy_frame_distractor():
    frames = [frame(i, objects=(target(170) if i == 2 else target(30),)) for i in range(5)]
    result = select_interacting_actor(frames)
    assert all(f.persons[0].box == tuple(float(x) for x in ACTOR) for f in result.frames)


@pytest.mark.parametrize("kwargs", [
    {"confidence_threshold": -0.1}, {"confidence_threshold": float("nan")},
    {"object_ambiguity_margin": True}, {"association_iou_threshold": 1.1},
    {"association_margin": float("inf")}, {"affinity_margin": -1},
    {"min_observations": 2}, {"min_observations": True}, {"min_coverage": 0.49},
    {"min_coverage": 1.1}, {"max_missing_observations": -1},
])
def test_invalid_or_understrength_gate_settings_fail(kwargs):
    with pytest.raises(ValueError):
        select_interacting_actor([frame(i) for i in range(3)], **kwargs)


@pytest.mark.parametrize("frames", [[], None, [frame(0), frame(0)], [frame(-1)],
    [FrameDetections(0, 200, 0)], [frame(0), FrameDetections(1, 201, 100)], [{}]])
def test_bad_frames_fail_explicitly(frames):
    with pytest.raises(ValueError):
        select_interacting_actor(frames)
