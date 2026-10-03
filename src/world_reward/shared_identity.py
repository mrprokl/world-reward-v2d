"""Pure full-timeline native identity contracts, before geometry or caching.

The fixed policy copies shape45 and PCA-scale28 from original frame zero to
every original frame. Other native parameter blocks are byte-preserved. This
module neither loads nor decodes a model, selects identity using quality,
reuses cached joints/keypoints, or establishes reconstruction accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from numbers import Integral
from types import MappingProxyType

import numpy as np

from world_reward.timeline import NATIVE_WINDOW


NATIVE_PARAMETER_DIMS = MappingProxyType({
    "mhr_global_rot6d": 6,
    "mhr_trans": 3,
    "mhr_body_pose_cont": 260,
    "mhr_hand": 108,
    "mhr_shape": 45,
    "mhr_scale": 28,
    "mhr_face": 72,
})
IDENTITY_BLOCKS = ("mhr_shape", "mhr_scale")
HISTORICAL_GEOMETRY_CHECKS = (
    "native_roundtrip_max_error_m", "translation_once_max_error_m",
    "projection_max_error_px", "mhr_geometry_forward_verified",
)


def _frame_count(total_frames: int) -> int:
    if (isinstance(total_frames, (bool, np.bool_)) or not isinstance(total_frames, Integral)
            or total_frames < NATIVE_WINDOW):
        raise ValueError("total_frames must be an integer >=96; no padding or fabricated frames")
    return int(total_frames)


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes(order="C") == right.tobytes(order="C")


def validate_native_parameters(
    parameters: Mapping[str, np.ndarray], total_frames: int, *, require_shared_identity: bool = False,
) -> None:
    """Validate exactly seven full native FP32 blocks without conversion.

    Root6D uses interleaved matrix columns, not continuous-body layout. Reject
    degeneracy rather than replacing an invalid rotation. Translation depth
    must be positive and expression identically zero. The optional identity
    check compares row bytes, including signed zeros. Inputs remain untouched;
    this establishes parameter ABI only, not geometry, provenance or quality.
    """
    count = _frame_count(total_frames)
    if (not isinstance(parameters, Mapping) or set(parameters) != set(NATIVE_PARAMETER_DIMS)
            or type(require_shared_identity) is not bool):
        raise ValueError("Exact seven native parameter blocks and explicit bool policy required")
    for name, dimension in NATIVE_PARAMETER_DIMS.items():
        value = parameters[name]
        if (type(value) is not np.ndarray or value.dtype != np.dtype("float32")
                or value.shape != (count, dimension) or not np.isfinite(value).all()):
            raise ValueError(f"{name} requires an unmasked finite native float32 [{count},{dimension}] array")
    basis = parameters["mhr_global_rot6d"].astype(np.float64).reshape(count, 3, 2)
    first = basis[:, :, 0]
    norms = np.linalg.norm(first, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError("Original native root6D first column is degenerate")
    first = first / norms[:, None]
    second = basis[:, :, 1] - np.sum(first * basis[:, :, 1], axis=1)[:, None] * first
    if np.any(np.linalg.norm(second, axis=1) <= 1e-8):
        raise ValueError("Original native root6D columns are degenerate")
    if np.any(parameters["mhr_trans"][:, 2] <= 0):
        raise ValueError("Every original native translation must have positive camera depth")
    if np.count_nonzero(parameters["mhr_face"]):
        raise ValueError("Every original expression must be zero")
    if require_shared_identity:
        for name in IDENTITY_BLOCKS:
            first_bytes = parameters[name][0].tobytes()
            if any(row.tobytes() != first_bytes for row in parameters[name]):
                raise ValueError("Full native shape/PCA identity must be byte-constant")


def share_first_frame_identity(
    parameters: Mapping[str, np.ndarray], total_frames: int,
) -> dict[str, np.ndarray]:
    """Return owned copies with first-original-frame shape/PCA on all N.

    No original is changed or aliased, including when source blocks alias one
    another. Outputs are C-contiguous and have no shared storage with any
    source or other output. Nonidentity blocks retain their original C-order
    bytes. Cached geometry is deliberately not accepted or returned; callers
    must decode joints/keypoints again before constructing native crops,
    neutral-height normalization, renders or caches.
    """
    validate_native_parameters(parameters, total_frames)
    count = int(total_frames)
    original = {name: (value.shape, value.dtype.str, value.tobytes(order="C")) for name, value in parameters.items()}
    shared = {
        name: np.repeat(parameters[name][:1], count, axis=0)
        if name in IDENTITY_BLOCKS else parameters[name].copy(order="C")
        for name in NATIVE_PARAMETER_DIMS
    }
    validate_native_parameters(shared, count, require_shared_identity=True)
    for name, value in shared.items():
        if not value.flags.owndata or not value.flags.c_contiguous:
            raise ValueError("Each shared native block must own its contiguous storage")
        if any(np.shares_memory(value, source) for source in parameters.values()):
            raise ValueError("A shared native block aliases the original initializer")
        if name in IDENTITY_BLOCKS:
            if value[0].tobytes() != parameters[name][0].tobytes():
                raise ValueError("Original frame-zero identity bytes changed")
        elif not _same_bytes(value, parameters[name]):
            raise ValueError("A nonidentity native block changed")
    for index, name in enumerate(NATIVE_PARAMETER_DIMS):
        if any(np.shares_memory(shared[name], shared[other]) for other in tuple(NATIVE_PARAMETER_DIMS)[index + 1:]):
            raise ValueError("Shared native output blocks must have independent storage")
    if {name: (value.shape, value.dtype.str, value.tobytes(order="C")) for name, value in parameters.items()} != original:
        raise ValueError("Original native initializer changed during identity copying")
    return shared


def shared_initializer_metadata(metadata: dict, total_frames: int) -> dict:
    """Copy provenance and explicitly invalidate original geometry checks.

    The caller must apply this policy before its first shared-identity decode.
    Declared ordering is not evidence that an external runtime obeyed it.
    Original decoder/projection checks are retained only as historical checks,
    never re-labelled as checks of the new initializer. No cached J/KP reuse
    is authorized. Explicit original no-oracle provenance is required rather
    than fabricated; unknown source/camera/license fields are deep-copied.
    """
    count = _frame_count(total_frames)
    if (type(metadata) is not dict or metadata.get("ground_truth_used") is not False
            or metadata.get("hand_labeled_test") is not False or metadata.get("oracle_modes") != []
            or any(metadata.get(name, False) is not False for name in ("ground_truth_read", "private_truth_read"))):
        raise ValueError("Explicit original no-GT/no-hand-label/no-oracle metadata required")
    if "historical_original_initializer_checks" in metadata:
        raise ValueError("Already relabelled identity metadata cannot silently overwrite its history")
    result = copy.deepcopy(metadata)
    historical = {name: result.pop(name) for name in HISTORICAL_GEOMETRY_CHECKS if name in result}
    result.update(
        historical_original_initializer_checks=historical,
        identity_policy="native_original_frame_zero_before_geometry_crops_renders_caches_no_quality_selection",
        identity_selection_frame_index=0,
        identity_choice_phase="before_native_geometry_crop_render_cache",
        identity_quality_selection_performed=False,
        human_identity_clip_constant=True,
        source_frames=count,
        original_frame_indices=list(range(count)),
        original_geometry_recovered=False,
        mhr_geometry_forward_verified=False,
        projection_reverified_after_identity_change=False,
        joint_keypoint_redecode_required=True,
        source_joint_keypoint_reuse_permitted=False,
        quality_verified=False,
        adoption_authorized=False,
        submission_eligible=False,
        challenge_performance_verified=False,
    )
    return result
