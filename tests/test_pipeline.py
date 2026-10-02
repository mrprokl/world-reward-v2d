"""Tiny declarative planning tests: no actual videos, models, GPUs or remote IO."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from world_reward import pipeline
from world_reward.pipeline import plan_episode


NAMES = (
    "automatic_masks", "body_smoke", "depth_smoke", "scale_smoke", "object_grounded",
    "body_full", "depth_full", "object_pose_full", "cari_adapter", "cari_inputs",
    "cari_forward", "cari_conversion",
)


@pytest.mark.parametrize("episode", range(30))
def test_all_track1_episodes_have_exact_frozen_paths_commands_and_stage_order(episode):
    plan = plan_episode(episode)
    assert isinstance(plan, tuple)
    assert tuple(stage.name for stage in plan) == NAMES
    seen = set()
    for stage in plan:
        assert set(stage.dependencies) <= seen
        seen.add(stage.name)
        assert stage.action == "run"
        assert stage.argv[1:3] == ("--episode", str(episode))
        assert stage.output_directory.startswith(f"outputs/episode_{episode:06d}/")
        assert stage.report_path == stage.output_directory + "/report.json"
        assert "track_2" not in repr(stage) and "track_3" not in repr(stage)
        assert all("\n" not in argument and ";" not in argument for argument in stage.argv)
    assert len({stage.report_path for stage in plan}) == len(plan)
    assert sum(not stage.requires_gpu for stage in plan) == 1
    assert next(stage for stage in plan if stage.name == "cari_inputs").requires_gpu is False


@pytest.mark.parametrize("episode", [-1, 30, 100, True, False, None, "15", 15.0, [], {}])
def test_episode_validation_has_no_default_or_ambiguous_integer(episode):
    with pytest.raises(ValueError, match="0..29"):
        plan_episode(episode)


def test_exact_sparse_full_object_and_nested_adapter_contracts():
    stages = {stage.name: stage for stage in plan_episode(0)}
    assert stages["body_smoke"].report_stage == "sam3d_body_three_frame_smoke"
    assert stages["body_full"].report_stage == "sam3d_body_full_video_initializer"
    assert stages["depth_smoke"].report_stage == "monocular_moge2_three_frame"
    assert stages["depth_full"].report_stage == "monocular_moge2_full_video"
    assert stages["scale_smoke"].dependencies == ("automatic_masks", "body_smoke", "depth_smoke")
    assert stages["scale_smoke"].prerequisite_reports == ("results/camera-render.json",)
    assert "--aligned-pointmap" in stages["object_grounded"].argv
    assert stages["object_grounded"].report_stage == "sam3d_objects_grounded_fixed_frame"
    assert stages["object_pose_full"].report_stage == "fixed_scale_full_object_pose_initializer"
    assert stages["object_pose_full"].dependencies == (
        "automatic_masks", "scale_smoke", "object_grounded", "body_full", "depth_full")
    assert stages["cari_adapter"].output_directory == "outputs/episode_000000/body_full/cari_adapter"
    assert stages["cari_adapter"].report_stage == "native_cari_body_adapter_full_video"
    for name in ("body_full", "depth_full", "object_pose_full"):
        assert "--full-video" in stages[name].argv
    for name in ("body_smoke", "depth_smoke"):
        assert "--full-video" not in stages[name].argv
    for name in ("body_smoke", "body_full"):
        assert stages[name].argv[-2:] == ("--inference-type", "body")
    assert all("--kernel-only" not in stage.argv and "--diagnose-only" not in stage.argv for stage in stages.values())
    assert all("run_body_converter.sh" not in stage.argv[0] for stage in stages.values())


def test_native_exact_producers_are_not_body_only_or_checkpoint_smoke():
    stages = {stage.name: stage for stage in plan_episode(15)}
    assert stages["cari_inputs"].report_stage == "world_reward_native_cari_inputs"
    assert stages["cari_forward"].report_stage == "world_reward_native_cari_full_forward"
    assert stages["cari_conversion"].report_stage == "world_reward_native_cari_official_conversion"
    assert stages["cari_inputs"].dependencies == (
        "automatic_masks", "body_full", "depth_full", "scale_smoke", "object_pose_full", "cari_adapter")
    assert stages["cari_forward"].dependencies == ("body_full", "cari_inputs")
    assert stages["cari_conversion"].dependencies == (
        "body_full", "cari_adapter", "object_pose_full", "cari_inputs", "cari_forward")


def test_states_are_not_retained_or_mutable_and_only_verified_completion_reuses():
    states = {"automatic_masks": "complete"}
    plan = plan_episode(0, states)
    states["automatic_masks"] = "failed"
    assert plan[0].action == "reuse"
    assert all(stage.action == "run" for stage in plan[1:])
    with pytest.raises(FrozenInstanceError):
        plan[0].action = "run"
    assert all(stage.action == "run" for stage in plan_episode(0, {}))
    assert all(stage.action == "run" for stage in plan_episode(0, {name: "absent" for name in NAMES}))
    assert all(stage.action == "reuse" for stage in plan_episode(0, {name: "complete" for name in NAMES}))


@pytest.mark.parametrize("state", ["failed", "incomplete"])
@pytest.mark.parametrize("name", NAMES)
def test_any_frozen_failure_or_incomplete_output_aborts_entire_plan(name, state):
    with pytest.raises(ValueError, match=f"Frozen {state} output for {name}"):
        plan_episode(0, {name: state})


@pytest.mark.parametrize("state", [None, True, 1, "pass", "running", "", [], {}])
def test_a_report_status_or_other_value_is_not_a_verified_artifact_state(state):
    with pytest.raises(ValueError, match="Invalid artifact state"):
        plan_episode(0, {"automatic_masks": state})


@pytest.mark.parametrize("states", [[], (), "complete", 1, True])
def test_state_collection_must_be_a_mapping(states):
    with pytest.raises(ValueError, match="mapping"):
        plan_episode(0, states)


@pytest.mark.parametrize("name", ["body_converter", "cari_kernel", "../automatic_masks", "x;rm -rf /", "track_2"])
def test_no_arbitrary_stage_or_shell_control_enters_command_plan(name):
    with pytest.raises(ValueError, match="Unknown producing stage"):
        plan_episode(0, {name: "complete"})


@pytest.mark.parametrize("name", NAMES[1:])
def test_reusing_a_derivative_cannot_rerun_missing_frozen_dependencies(name):
    with pytest.raises(ValueError, match="requires complete frozen dependencies"):
        plan_episode(0, {name: "complete"})


def test_independent_full_body_can_reuse_without_sparse_scale_branch():
    plan = plan_episode(0, {"automatic_masks": "complete", "body_full": "complete"})
    assert {stage.name for stage in plan if stage.action == "reuse"} == {"automatic_masks", "body_full"}


def test_duplicate_or_nontopological_internal_definitions_fail_fast(monkeypatch):
    original = pipeline._STAGES
    monkeypatch.setattr(pipeline, "_STAGES", (*original, original[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        plan_episode(0)
    monkeypatch.setattr(pipeline, "_STAGES", (replace(original[0], dependencies=("cari_forward",)), *original[1:]))
    with pytest.raises(ValueError, match="non-topological"):
        plan_episode(0)
    monkeypatch.setattr(pipeline, "_STAGES", (original[0], replace(original[1], dependencies=("automatic_masks", "automatic_masks")), *original[2:]))
    with pytest.raises(ValueError, match="non-topological"):
        plan_episode(0)


def test_fixed_wrapper_scripts_exist_without_importing_or_running_them():
    repository = Path(__file__).resolve().parents[1]
    for stage in plan_episode(0):
        assert (repository / stage.argv[0]).is_file()


def test_planner_never_reads_output_files_or_imports_heavy_libraries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pure planning must not inspect artifacts")
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert len(plan_episode(29)) == 12
