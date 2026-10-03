"""Procedural, non-challenge evidence for fixed-window actor initialization.

Boxes are generated here, not copied from videos or human-labeled detections.
These tests exercise the existing selector with unchanged default gates; they
validate bookkeeping/safety, not detector accuracy or reconstructed trajectories.
"""

import pytest

from world_reward.actor_selection import select_interacting_actor
from world_reward.prompt_selection import BoxDetection, FrameDetections, select_seed_prompts


WIDTH, HEIGHT = 1000, 120
ACTOR = BoxDetection((20, 10, 60, 110), 0.75)
BYSTANDER = BoxDetection((850, 10, 890, 110), 0.99)


def _target(center=40):
    return BoxDetection((center - 5, 55, center + 5, 65), 0.9)


def _frame(index, persons, objects=None):
    return FrameDetections(index, WIDTH, HEIGHT, tuple(persons),
                           (_target(),) if objects is None else tuple(objects))


def _stable_window(missing=()):
    missing = set(missing)
    return [_frame(step * 40, (BYSTANDER,) if step in missing else (ACTOR, BYSTANDER))
            for step in range(16)]


@pytest.mark.parametrize("missing", range(3))
def test_fixed_three_with_one_missing_abstains_without_selecting_bystander(missing):
    with pytest.raises(ValueError, match="Closest actor track has insufficient observations"):
        select_interacting_actor(_stable_window((missing,))[:3])


@pytest.mark.parametrize("missing", range(16))
def test_fixed_sixteen_uses_fifteen_real_actor_observations_without_imputation(missing):
    frames = _stable_window((missing,))
    result = select_interacting_actor(frames)
    assert result.scores[0].observations == 15
    assert result.scores[0].coverage == 15 / 16
    assert result.scores[0].contaminated is False
    assert result.scores[1].observations == 16  # competitor was not removed
    assert result.scores[1].outside_distance > result.scores[0].outside_distance
    assert [frame.frame_index for frame in result.frames] == [frame.frame_index for frame in frames]
    for step, selected in enumerate(result.frames):
        assert selected.persons == (() if step == missing else (ACTOR,))
        assert selected.objects is frames[step].objects
    seed = select_seed_prompts(result.frames, object_prompt="procedural target", total_frames=601)
    original = next(frame for frame in frames if frame.frame_index == seed.frame_index)
    assert seed.person in original.persons
    assert seed.object in original.objects
    assert seed.person == ACTOR
    assert seed.frame_index == (40 if missing == 0 else 0)


def test_three_complete_observations_remain_sufficient_at_unchanged_gates():
    result = select_interacting_actor(_stable_window()[:3])
    assert result.scores[0].observations == 3
    assert result.scores[0].coverage == 1


def test_fixed_window_result_is_invariant_to_frame_and_person_input_order():
    frames = _stable_window((1,))
    reversed_frames = [_frame(frame.frame_index, reversed(frame.persons), frame.objects)
                       for frame in reversed(frames)]
    assert select_interacting_actor(frames) == select_interacting_actor(reversed_frames)


def _known_motion(indices):
    # One pixel/frame is the same known trajectory for both observation grids.
    return [_frame(index, (BoxDetection((20 + index, 10, 60 + index, 110), 0.75), BYSTANDER),
                   (_target(40 + index),)) for index in indices]


def test_sparse_nonoverlap_fragments_are_not_joined_to_manufacture_support():
    with pytest.raises(ValueError, match="Closest actor track has insufficient observations"):
        select_interacting_actor(_known_motion(range(0, 451, 50)))


def test_denser_actual_frames_bridge_the_same_motion_without_changing_association_gate():
    dense = _known_motion(range(0, 451, 10))
    sparse = {frame.frame_index: frame for frame in _known_motion(range(0, 451, 50))}
    assert all(frame == sparse[frame.frame_index] for frame in dense if frame.frame_index in sparse)
    result = select_interacting_actor(dense)
    assert result.scores[0].observations == 46
    assert result.scores[0].coverage == 1
    assert result.scores[1].observations == 46
    assert all(selected.frame_index == original.frame_index and
               selected.persons == (original.persons[0],) and
               selected.objects is original.objects
               for selected, original in zip(result.frames, dense))


@pytest.mark.parametrize("visible", [(0, 1), (0, 1, 2), (0, 1, 4, 5)])
def test_closest_unsupported_lowcoverage_or_expired_fragments_do_not_select_bystander(visible):
    frames = _stable_window(set(range(16)) - set(visible))
    with pytest.raises(ValueError, match="Closest actor track has insufficient observations or coverage"):
        select_interacting_actor(frames)


def _crossing_window(*, object_center=200, gap=False):
    left = BoxDetection((170, 10, 230, 110), 0.95)
    right = BoxDetection((210, 10, 270, 110), 0.94)
    middle_a = BoxDetection((190, 10, 250, 110), 0.95)
    middle_b = BoxDetection((190, 11, 250, 109), 0.94)
    frames = []
    for step in range(16):
        background = (left, right) if step != 1 else (middle_a, middle_b)
        if gap and step in (2, 3):
            background = ()
        frames.append(_frame(step * 40, (ACTOR, *background), (_target(object_center),)))
    return frames


def test_full_window_does_not_rehabilitate_an_ambiguous_closest_crossing():
    with pytest.raises(ValueError, match="Ambiguous person-track association in selected actor") as error:
        select_interacting_actor(_crossing_window())
    assert "40" in str(error.value)  # original observed frame, not dense step number
    assert "contaminated=True" in str(error.value)


@pytest.mark.parametrize("gap", [False, True])
def test_independent_actor_survives_but_all_crossing_competitors_remain_contaminated(gap):
    result = select_interacting_actor(_crossing_window(object_center=40, gap=gap))
    assert result.scores[0].observations == 16
    assert result.scores[0].contaminated is False
    assert all(frame.persons == (ACTOR,) for frame in result.frames)
    assert len(result.scores) >= 3
    assert all(score.contaminated for score in result.scores[1:])
    # Even a long unambiguous suffix or expiration/reappearance does not reset it.
    if gap:
        assert len(result.scores) > 3
        assert any(160 in score.ambiguous_frame_indices for score in result.scores[1:])
    else:
        assert all(600 in score.ambiguous_frame_indices for score in result.scores[1:])


def test_two_equally_close_persistent_actors_remain_ambiguous_with_more_observations():
    a = BoxDetection((20, 10, 60, 110), 0.99)
    b = BoxDetection((80, 10, 120, 110), 0.75)
    frames = [_frame(step * 40, (a, b), (_target(70),)) for step in range(16)]
    with pytest.raises(ValueError, match="Ambiguous actor affinity"):
        select_interacting_actor(frames)
