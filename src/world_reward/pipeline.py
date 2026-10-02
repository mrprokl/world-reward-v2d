"""Pure planner for the current automatic Track 1/native CARI route.

This is deliberately not an executor or an artifact verifier. ``complete`` means
the caller has independently verified the exact report stage, episode/input
identity, explicit video-only provenance, dependency and artifact hashes, and
stage-specific numerical/full-frame gates. Merely finding ``report.json`` is
never sufficient. Existing output directories without a verified complete
report must be classified ``incomplete`` (or ``failed``), not ``absent``.

All input/source/model/license and global camera-gate preflights remain external.
The plan grants neither submission eligibility nor challenge accuracy. Commands
are fixed argument tuples for remote use only; no shell, filesystem, network or
heavy-library work occurs here. A future scheduler must bind native wrappers'
``--wait-for`` to its exact producer units: episode 15 wrappers otherwise retain
historical implicit waits. The tuples below are not a launch recipe on their own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


ArtifactState = Literal["absent", "complete", "incomplete", "failed"]


@dataclass(frozen=True)
class PipelineStage:
    """One immutable, topologically ordered producing stage relative to WR_ROOT."""

    name: str
    report_stage: str
    argv: tuple[str, ...]
    output_directory: str
    dependencies: tuple[str, ...]
    action: Literal["run", "reuse"]
    requires_gpu: bool
    prerequisite_reports: tuple[str, ...] = ()

    @property
    def report_path(self) -> str:
        return f"{self.output_directory}/report.json"


@dataclass(frozen=True)
class _Stage:
    name: str
    report_stage: str
    script: str
    dependencies: tuple[str, ...]
    flags: tuple[str, ...] = ()
    requires_gpu: bool = True
    directory: str | None = None
    prerequisite_reports: tuple[str, ...] = ()


# This is the one supported route, not a user-extensible shell/DAG framework.
# Sparse initializers cannot be substituted with full reports: scale_smoke reads
# the original three-frame body/depth artifacts. Official Body-only conversion
# is unnecessary because the final native converter produces shared identity.
_STAGES = (
    _Stage("automatic_masks", "automatic_masks", "run_automatic_masks.sh", ()),
    _Stage("body_smoke", "sam3d_body_three_frame_smoke", "run_body_smoke.sh",
           ("automatic_masks",), ("--inference-type", "body")),
    _Stage("depth_smoke", "monocular_moge2_three_frame", "run_depth_smoke.sh",
           ("automatic_masks",)),
    _Stage("scale_smoke", "predicted_human_anchored_moge2_pointmaps", "run_scale_smoke.sh",
           ("automatic_masks", "body_smoke", "depth_smoke"),
           prerequisite_reports=("results/camera-render.json",)),
    _Stage("object_grounded", "sam3d_objects_grounded_fixed_frame", "run_object_smoke.sh",
           ("automatic_masks", "scale_smoke"), ("--aligned-pointmap",)),
    _Stage("body_full", "sam3d_body_full_video_initializer", "run_body_smoke.sh",
           ("automatic_masks",), ("--full-video", "--inference-type", "body")),
    _Stage("depth_full", "monocular_moge2_full_video", "run_depth_smoke.sh",
           ("automatic_masks",), ("--full-video",)),
    _Stage("object_pose_full", "fixed_scale_full_object_pose_initializer", "run_object_pose_smoke.sh",
           ("automatic_masks", "scale_smoke", "object_grounded", "body_full", "depth_full"),
           ("--full-video",)),
    _Stage("cari_adapter", "native_cari_body_adapter_full_video", "run_cari_body_adapter_smoke.sh",
           ("body_full",), directory="body_full/cari_adapter"),
    _Stage("cari_inputs", "world_reward_native_cari_inputs", "run_cari_prepare.sh",
           ("automatic_masks", "body_full", "depth_full", "scale_smoke", "object_pose_full", "cari_adapter"),
           requires_gpu=False),
    _Stage("cari_forward", "world_reward_native_cari_full_forward", "run_cari_forward.sh",
           ("body_full", "cari_inputs")),
    _Stage("cari_conversion", "world_reward_native_cari_official_conversion", "run_cari_converter.sh",
           ("body_full", "cari_adapter", "object_pose_full", "cari_inputs", "cari_forward")),
)


def plan_episode(
    episode_index: int, states: Mapping[str, ArtifactState] | None = None,
) -> tuple[PipelineStage, ...]:
    """Plan all stages for one episode, reusing only externally verified states.

    Missing state keys mean ``absent``; callers must inventory remote outputs
    before relying on this default. Failed/incomplete artifacts abort the whole
    plan rather than implying retries or overwrites. A frozen complete stage
    requires every direct dependency to be complete too: producing a new input
    underneath an already reused derivative would invalidate its provenance.
    """
    if type(episode_index) is not int or not 0 <= episode_index < 30:
        raise ValueError("Require an integer Track 1 episode index in 0..29")
    if states is not None and not isinstance(states, Mapping):
        raise ValueError("Artifact states must be an externally validated mapping")
    supplied = dict(states) if states is not None else {}
    names = {stage.name for stage in _STAGES}
    if len(names) != len(_STAGES):
        raise ValueError("Duplicate producing stage names")
    if supplied.keys() - names:
        raise ValueError("Unknown producing stage state")
    for name, state in supplied.items():
        if type(state) is not str or state not in ("absent", "complete", "incomplete", "failed"):
            raise ValueError(f"Invalid artifact state for {name}")
        if state in ("incomplete", "failed"):
            raise ValueError(f"Frozen {state} output for {name}; do not overwrite or retry implicitly")

    base = f"outputs/episode_{episode_index:06d}"
    plan, seen = [], set()
    for stage in _STAGES:
        if len(set(stage.dependencies)) != len(stage.dependencies) or not set(stage.dependencies) <= seen:
            raise ValueError(f"Invalid or non-topological dependencies for {stage.name}")
        reuse = supplied.get(stage.name, "absent") == "complete"
        if reuse and any(supplied.get(name, "absent") != "complete" for name in stage.dependencies):
            raise ValueError(f"Complete {stage.name} requires complete frozen dependencies")
        plan.append(PipelineStage(
            name=stage.name, report_stage=stage.report_stage,
            argv=(f"infra/{stage.script}", "--episode", str(episode_index), *stage.flags),
            output_directory=f"{base}/{stage.directory or stage.name}",
            dependencies=stage.dependencies, action="reuse" if reuse else "run",
            requires_gpu=stage.requires_gpu, prerequisite_reports=stage.prerequisite_reports,
        ))
        seen.add(stage.name)
    return tuple(plan)
