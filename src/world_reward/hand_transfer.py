"""Finger-only proposals from already decoded, asset-verified MHR controls.

SAM's native ``mhr_model_params[204]`` concatenates pose136 and scales68;
the finger columns have already passed the head's mean/component expansion.
Do not substitute hand108, zeroed raw266, or stale global rotations. The caller
must separately verify source assets, model-control roundtrip, original video
identity and permission to use these predictions. This numerical helper does
not certify provenance, licensing, visibility, accuracy or submission status.

Copy only verified finger indices into a final pose136 trajectory. Root,
translation, body and wrists remain bit-identical; identity/scales are not even
arguments. Sparse source records produce sparse proposals without filling,
interpolating, repeating or shortening the final trajectory. A proposal is
never automatically adopted; image/geometry falsification belongs downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandTransferProposal:
    pose: np.ndarray
    report: dict


def _float_array(value, name: str, columns: int) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot hide invalid entries behind a mask")
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1:] != (columns,) or not len(array)
            or array.dtype.kind != "f" or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite floating-point nonempty [N,{columns}] array")
    return array


def _indices(value, name: str, count: int) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot hide indices behind a mask")
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer [{count}] array")
    return array


def transfer_finger_controls(
    final_pose,
    source_controls,
    source_frame_indices,
    hand_indices_left,
    hand_indices_right,
    *,
    handoff_mask=None,
) -> HandTransferProposal:
    """Copy decoded fingers only, returning a new dtype-preserving pose proposal.

    ``final_pose[T,136]`` is indexed by original frame number. The source
    ``[S,204]`` rows correspond exactly to sorted unique integer
    ``source_frame_indices[S]`` in ``[0,T)``. No fallback/interpolation is made
    for frames without a source row. Both supplied hand index lists must have
    27 unique entries, be disjoint, and together equal columns 68 through 121.
    Their order need not be sorted: source204 and final136 use the same absolute
    indices, rather than assigning vector positions to an invented hand layout.

    Optional ``handoff_mask[S,2]`` is an explicit boolean selection in left/right
    order. False means keep the final fingers, not zero them; this is merely a
    caller-supplied selection, not a claimed visibility estimate. Default selects
    both hands in every source row. Copy to the final floating dtype; reject
    nonfinite casts rather than silently saturating. Inputs are never modified.
    """
    final = _float_array(final_pose, "final_pose", 136)
    source = _float_array(source_controls, "source_controls", 204)
    frames = _indices(source_frame_indices, "source_frame_indices", len(source))
    if (np.any(frames < 0) or np.any(frames >= len(final))
            or np.any(frames[1:] <= frames[:-1])):
        raise ValueError("Source frame indices must be sorted unique original indices within the final trajectory")
    left = _indices(hand_indices_left, "hand_indices_left", 27)
    right = _indices(hand_indices_right, "hand_indices_right", 27)
    combined = np.concatenate([left, right])
    if (np.any(combined < 68) or np.any(combined >= 122)
            or not np.array_equal(np.sort(combined), np.arange(68, 122))):
        raise ValueError("Verified left/right hand indices must be disjoint and partition exactly columns 68:122")
    if handoff_mask is None:
        selected = np.ones((len(source), 2), dtype=np.bool_)
    else:
        if np.ma.isMaskedArray(handoff_mask):
            raise ValueError("handoff_mask cannot hide selections behind a mask")
        selected = np.asarray(handoff_mask)
        if selected.shape != (len(source), 2) or selected.dtype != np.bool_:
            raise ValueError("handoff_mask must be an explicit boolean [S,2] array in left/right order")
    candidate = final.copy()
    transfers = []
    for side, columns in enumerate((left, right)):
        rows = np.flatnonzero(selected[:, side])
        if len(rows):
            with np.errstate(over="ignore", invalid="ignore"):
                values = source[np.ix_(rows, columns)].astype(final.dtype)
            if not np.isfinite(values).all():
                raise ValueError("Selected source finger controls cannot be represented as finite final-pose dtype")
            candidate[np.ix_(frames[rows], columns)] = values
        transfers.append(frames[rows].tolist())
    # Exact comparisons, including signed zero and dtype, not numerical allclose.
    fixed = np.r_[np.arange(68), np.arange(122, 136)]
    if candidate[:, fixed].tobytes() != final[:, fixed].tobytes():
        raise RuntimeError("Finger proposal changed fixed root/body/wrist controls")
    report = {
        "schema": "world-reward-finger-only-control-transfer-v1",
        "scope": "numerical_proposal_not_automatic_adoption_or_accuracy_verification",
        "final_frames": len(final), "source_frames": len(source),
        "source_original_frame_indices": frames.tolist(),
        "hand_order": ["left_hand", "right_hand"],
        "hand_indices_left": left.tolist(), "hand_indices_right": right.tolist(),
        "selected_original_frames_by_hand": transfers,
        "selected_left_hand_frames": len(transfers[0]), "selected_right_hand_frames": len(transfers[1]),
        "finger_column_count": 54, "fixed_column_count": 82,
        "final_dtype": str(final.dtype), "source_dtype": str(source.dtype),
        "nonhand_controls_bit_identical": True, "full_original_frame_coverage_preserved": True,
        "source_missing_frames_interpolated": False, "source_scale_columns_copied": False,
        "body_or_wrist_controls_copied": False, "identity_fitted": False,
        "input_arrays_modified": False, "adoption_performed": False,
    }
    return HandTransferProposal(candidate, report)
