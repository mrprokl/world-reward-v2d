"""Tiny generated scenes: accuracy assertions use only test-created ground truth."""

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward.pose_selection import NoFeasiblePathError, select_pose_path


def yaw(degrees):
    angles = np.asarray(degrees)
    if angles.ndim:
        angles = angles[..., None]
    return Rotation.from_euler("z", angles, degrees=True).as_matrix()


def angular_errors(predicted, target):
    relative = predicted @ target.swapaxes(-1, -2)
    return Rotation.from_matrix(relative).magnitude()


def fixture(frames=4, candidates=2):
    return dict(
        rotations=np.tile(np.eye(3), (frames, candidates, 1, 1)),
        translations=np.zeros((frames, candidates, 3)),
        image_costs=np.zeros((frames, candidates)),
    )


def test_occluded_ambiguous_unaries_do_not_create_greedy_jump():
    scene = fixture(frames=7)
    truth = np.column_stack((np.arange(7) * 0.1, np.zeros((7, 2))))
    scene["translations"][:, 0] = truth
    scene["translations"][:, 1] = truth + [0.8, 0, 0]
    scene["image_costs"][:] = [[0, 3], [0, 3], [4, 0], [4, 0], [4, 0], [0, 3], [0, 3]]
    greedy = scene["translations"][np.arange(7), scene["image_costs"].argmin(axis=1)]
    path = select_pose_path(
        **scene,
        observation_confidence=np.array([1, 1, 0, 0, 0, 1, 1]),
        translation_weight=10,
    )
    np.testing.assert_array_equal(path.candidate_indices, np.zeros(7, dtype=int))
    np.testing.assert_allclose(path.translations, truth)
    assert np.linalg.norm(greedy - truth, axis=1).mean() > 0.3
    assert np.linalg.norm(np.diff(path.translations, axis=0), axis=1).max() > 0


def test_future_image_evidence_resolves_an_ambiguous_initial_pose():
    scene = fixture(frames=4)
    scene["translations"][:, 1, 0] = 1
    # Greedy starts on the wrong branch; a full path sees the later evidence.
    scene["image_costs"][:] = [[0.2, 0], [0.2, 0], [0, 5], [0, 5]]
    path = select_pose_path(**scene, translation_weight=4)
    np.testing.assert_array_equal(path.candidate_indices, [0, 0, 0, 0])
    assert scene["image_costs"].argmin(axis=1).tolist() == [1, 1, 0, 0]


def test_distinguishable_false_180_flip_is_rejected_without_declared_symmetry():
    scene = fixture(frames=7)
    truth = yaw(np.arange(7) * 10)
    scene["rotations"][:, 0] = truth
    scene["rotations"][:, 1] = truth @ yaw(180)
    scene["image_costs"][:] = [[0, 5], [0, 5], [0.2, 0], [0.2, 0], [0.2, 0], [0, 5], [0, 5]]
    greedy = scene["rotations"][np.arange(7), scene["image_costs"].argmin(axis=1)]
    path = select_pose_path(**scene, rotation_weight=2)
    np.testing.assert_array_equal(path.candidate_indices, np.zeros(7, dtype=int))
    assert angular_errors(path.rotations, truth).max() < 1e-12
    assert angular_errors(greedy, truth).max() > 3


def test_strong_image_evidence_preserves_real_fast_translation_and_rotation():
    scene = fixture(frames=5)
    truth_translation = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
    truth_rotation = yaw([0, 0, 170, 170, 170])
    scene["translations"][:, 0] = truth_translation
    scene["rotations"][:, 0] = truth_rotation
    scene["image_costs"][:, 1] = 10
    path = select_pose_path(**scene, rotation_weight=0.5)
    np.testing.assert_array_equal(path.candidate_indices, [0, 0, 0, 0, 0])
    np.testing.assert_allclose(path.translations, truth_translation)
    assert angular_errors(path.rotations, truth_rotation).max() < 1e-12
    assert np.linalg.norm(np.diff(path.translations, axis=0), axis=1).max() == 1
    assert angular_errors(path.rotations[1:], path.rotations[:-1]).max() > 2.9


def test_discrete_c2_symmetry_selects_valid_representatives_not_rotation_averages():
    scene = fixture(frames=4, candidates=1)
    raw = yaw([0, 180, 3, 186])
    scene["rotations"][:, 0] = raw
    group = yaw([0, 180])
    path = select_pose_path(**scene, symmetries=group)
    expected = raw @ group[path.symmetry_indices]
    np.testing.assert_allclose(path.rotations, expected)
    np.testing.assert_allclose(np.linalg.det(path.rotations), 1)
    assert angular_errors(path.rotations[1:], path.rotations[:-1]).max() < 0.11
    # The arbitrary global equivalent gauge is intentionally not called GT.
    plain = select_pose_path(**scene)
    assert plain.transition_cost > path.transition_cost + 20


def test_viterbi_matches_exhaustive_global_optimum():
    rng = np.random.default_rng(41)
    scene = fixture(frames=4, candidates=3)
    scene["rotations"] = yaw(rng.uniform(-180, 180, size=12)).reshape(4, 3, 3, 3)
    scene["translations"] = rng.normal(size=(4, 3, 3))
    scene["image_costs"] = rng.normal(size=(4, 3))
    times = np.array([0, 0.5, 1.5, 2])
    scores = []
    for indices in itertools.product(range(3), repeat=4):
        selected = np.array(indices)
        translations = scene["translations"][np.arange(4), selected]
        rotations = scene["rotations"][np.arange(4), selected]
        distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        angles = angular_errors(rotations[1:], rotations[:-1])
        scores.append(
            scene["image_costs"][np.arange(4), selected].sum()
            + np.sum((0.7 * distances**2 + 0.3 * angles**2) / np.diff(times))
        )
    path = select_pose_path(**scene, frame_times=times, translation_weight=0.7, rotation_weight=0.3)
    assert path.total_cost == pytest.approx(min(scores))
    assert path.total_cost == pytest.approx(path.unary_cost + path.transition_cost)


def test_symmetry_lifted_viterbi_matches_exhaustive_state_paths():
    scene = fixture(frames=3)
    scene["rotations"] = yaw([10, 100, 170, 20, 35, 210]).reshape(3, 2, 3, 3)
    scene["translations"][:, 1, 0] = 0.1
    scene["image_costs"][:] = [[0, 0.5], [0.5, 0], [0.1, 0.3]]
    group = yaw([0, 180])
    scores = []
    for states in itertools.product(range(4), repeat=3):
        indices, gauges = np.array(states) // 2, np.array(states) % 2
        rotations = scene["rotations"][np.arange(3), indices] @ group[gauges]
        translations = scene["translations"][np.arange(3), indices]
        distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        angles = angular_errors(rotations[1:], rotations[:-1])
        scores.append(scene["image_costs"][np.arange(3), indices].sum() + np.sum(distances**2 + angles**2))
    path = select_pose_path(**scene, symmetries=group)
    assert path.total_cost == pytest.approx(min(scores))


def test_speed_bound_uses_actual_time_and_fails_if_it_rules_out_all_paths():
    scene = fixture(frames=3, candidates=1)
    scene["translations"][:, 0, 0] = [0, 0.1, 1.1]
    path = select_pose_path(**scene, frame_times=np.array([0, 1, 11]), max_translation_speed=0.11)
    np.testing.assert_allclose(path.translations, scene["translations"][:, 0])
    with pytest.raises(NoFeasiblePathError, match="frame 2"):
        select_pose_path(**scene, max_translation_speed=0.11)


def test_rotation_speed_bound_rejects_an_infeasible_flip():
    scene = fixture(frames=2, candidates=1)
    scene["rotations"][:, 0] = yaw([0, 180])
    with pytest.raises(NoFeasiblePathError, match="frame 1"):
        select_pose_path(**scene, max_rotation_speed=0.5)


def test_identical_nonidentity_rotations_obey_an_exact_zero_speed_bound():
    scene = fixture(frames=4, candidates=1)
    scene["rotations"][:, 0] = yaw(37)
    path = select_pose_path(**scene, max_rotation_speed=0)
    assert path.transition_cost == pytest.approx(0, abs=1e-20)


def test_extreme_numeric_magnitudes_fail_explicitly_instead_of_returning_nan():
    scene = fixture(frames=2, candidates=1)
    scene["translations"][1, 0, 0] = 1e308
    with pytest.raises(ValueError, match="transition costs overflowed"):
        select_pose_path(**scene)
    scene = fixture(frames=2, candidates=1)
    scene["image_costs"][:] = 1e308
    with pytest.raises(ValueError, match="path costs overflowed"):
        select_pose_path(**scene)
    scene = fixture(frames=2, candidates=1)
    with pytest.raises(ValueError, match="frame-time intervals overflowed"):
        select_pose_path(**scene, frame_times=np.array([-1e308, 1e308]))
    scene["translations"][1, 0, 0] = 2
    with pytest.raises(ValueError, match="transition speeds overflowed"):
        select_pose_path(
            **scene,
            frame_times=np.array([0, 1e-308]),
            translation_weight=0,
            max_translation_speed=1,
        )


def test_explicit_candidate_mask_and_single_frame_tie_are_deterministic():
    scene = fixture(frames=1, candidates=3)
    path = select_pose_path(**scene, valid_candidates=np.array([[False, True, True]]))
    assert path.candidate_indices.tolist() == [1]
    assert path.symmetry_indices.tolist() == [0]
    assert path.total_cost == 0


def test_missing_interior_frame_candidates_cannot_be_silently_dropped():
    scene = fixture()
    valid = np.ones((4, 2), dtype=bool)
    valid[2] = False
    with pytest.raises(NoFeasiblePathError, match="frame 2 has no valid candidate"):
        select_pose_path(**scene, valid_candidates=valid)


def test_input_arrays_are_not_changed_or_aliased():
    scene = fixture()
    originals = {name: value.copy() for name, value in scene.items()}
    path = select_pose_path(**scene)
    path.rotations[0] = yaw(90)
    path.translations[0] = 5
    for name, value in scene.items():
        np.testing.assert_array_equal(value, originals[name])


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"rotations": np.zeros((4, 2, 3))}, "rotations must have shape"),
        ({"rotations": np.empty((0, 2, 3, 3))}, "T and K"),
        ({"rotations": np.empty((4, 0, 3, 3))}, "T and K"),
        ({"rotations": np.tile(np.diag([-1, 1, 1]), (4, 2, 1, 1))}, "proper SO"),
        ({"rotations": np.tile(2 * np.eye(3), (4, 2, 1, 1))}, "proper SO"),
        ({"rotations": np.full((4, 2, 3, 3), np.nan)}, "finite"),
        ({"translations": np.zeros((4, 3))}, "translations must have shape"),
        ({"translations": np.full((4, 2, 3), np.inf)}, "finite"),
        ({"translations": np.ones((4, 2, 3), dtype=complex)}, "real numbers"),
        ({"image_costs": np.zeros((4, 3))}, "image_costs must have shape"),
        ({"image_costs": np.full((4, 2), -np.inf)}, "finite"),
        ({"observation_confidence": np.ones((4, 2))}, "observation_confidence"),
        ({"observation_confidence": np.array([1, 1, -0.1, 1])}, "observation_confidence"),
        ({"observation_confidence": np.array([1, 1, 1.1, 1])}, "observation_confidence"),
        ({"frame_times": np.array([0, 1, 1, 3])}, "strictly increase"),
        ({"frame_times": np.array([3, 2, 1, 0])}, "strictly increase"),
        ({"frame_times": np.zeros(3)}, "frame_times"),
        ({"valid_candidates": np.ones((4, 2))}, "boolean"),
        ({"valid_candidates": np.ones((4, 3), dtype=bool)}, "boolean"),
        ({"valid_candidates": np.zeros((4, 2), dtype=bool)}, "no valid candidate"),
        ({"translation_weight": -1}, "nonnegative scalar"),
        ({"rotation_weight": np.nan}, "nonnegative scalar"),
        ({"translation_weight": [1, 2]}, "nonnegative scalar"),
        ({"rotation_weight": True}, "nonnegative scalar"),
        ({"max_translation_speed": -1}, "nonnegative scalar"),
        ({"max_rotation_speed": np.inf}, "nonnegative scalar"),
        ({"symmetries": np.empty((0, 3, 3))}, "nonempty shape"),
        ({"symmetries": yaw([180])}, "identity"),
        ({"symmetries": yaw([0, 0])}, "duplicates"),
        ({"symmetries": yaw([0, 90])}, "closed discrete"),
        ({"symmetries": np.array([np.eye(3), np.diag([-1, 1, 1])])}, "proper SO"),
    ],
)
def test_invalid_inputs_fail_fast(changes, message):
    scene = fixture()
    scene.update(changes)
    with pytest.raises(ValueError, match=message):
        select_pose_path(**scene)
