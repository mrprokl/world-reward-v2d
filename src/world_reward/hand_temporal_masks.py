"""Streaming IMAGE versus automatic-anchor video masks; no physical identity.

Callbacks configure/authenticate native SAM2 outside this pure operator. Masks
are availability proposals, not visibility, contact or calibrated confidence.
Late-born hands may be missed; no reseed, nearest match or static fill is made.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_reward.automatic_candidate_bank import _sam2_outputs
from world_reward.hand_observations import HandInstances, _copy, _size


@dataclass(frozen=True, eq=False)
class TemporalMaskFrame:
    branch: str  # A=image; B=video. IDs below are proposal slots, never actors.
    position: int
    original_frame_id: int
    proposal_ids: tuple[int, ...]
    raw_boxes: np.ndarray
    native_boxes: np.ndarray
    box_usable: np.ndarray
    masks: np.ndarray
    supported: np.ndarray  # Empty native masks can still have support=True.
    raw_scores: np.ndarray | None  # IMAGE predicted IoUs only; never selection.
    evidence: str


@dataclass(frozen=True, eq=False)
class TemporalMaskSummary:
    frame_ids: np.ndarray
    image_size: tuple[int, int]
    anchor_position: int | None
    anchor_frame_id: int | None
    anchor_raw_boxes: np.ndarray
    anchor_native_boxes: np.ndarray
    anchor_box_usable: np.ndarray
    seeded_proposal_ids: tuple[int, ...]
    native_forward_frames: int
    native_reverse_frames: int


def _boxes(hands, height, width):
    if type(hands) is not HandInstances:
        raise ValueError("Original typed 21-landmark hand evidence required")
    xy = _copy(hands.original_xy.values, "original XY", kind="f")
    available = _copy(hands.original_xy.supported, "original support", dtype=np.bool_)
    finite = np.isfinite(xy).all(axis=(1, 2))
    raw = np.full((len(xy), 4), np.nan, np.float64)
    raw[finite, :2] = xy[finite].min(axis=1)
    raw[finite, 2:] = xy[finite].max(axis=1)
    # Same ALL21/clip/FP32 convention as hand_mask_proposals, not a new fit.
    native = np.clip(raw, [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
    usable = finite & available.all(axis=1) & (native[:, 0] < native[:, 2]) & (native[:, 1] < native[:, 3])
    return tuple(_copy(v, "owned box evidence") for v in (raw, native, usable))


def stream_temporal_hand_masks(frames, *, frame_ids, image_size, image_predictor,
                               init_state, seed_box, propagate, emit, release_state=None):
    """A once per original RGB; B two independent states at first usable box frame.

    ``frames`` yields (original_id, uint8[H,W,3], HandInstances), exactly in
    declared int64[T] strictly increasing order. No RGB stack is retained.
    ``init_state()`` returns a NEW native state over that same full timeline.
    ``seed_box(state, frame_idx=position, obj_id=original_anchor_slot,
    box=float32[4], normalize_coords=True)`` receives every usable anchor box,
    identical in both states, including those whose IMAGE mask was empty.
    ``propagate(state, start_frame_idx=position, reverse=bool)`` yields exact
    native (position, ids, float[N,1,H,W] logits). Both native anchor outputs
    are validated; ownership is forward at/after anchor and reverse before.

    emit receives owned readonly records. B records are emitted forward then
    reverse (not chronological); positions/original IDs are explicit. Errors
    propagate, never retry or become an empty mask. Optional release_state frees
    each owned state in finally; e.g. native dict.clear allows sequential GPU
    states without retaining tensors. Caller seals all outputs only on success.
    No anchor means explicit N=0 B abstention on ALL frames, no video callback.
    """
    ids = _copy(frame_ids, "original frame IDs", dtype=np.int64)
    height, width = _size(image_size)
    if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Complete strictly increasing original int64 frame IDs required")
    if (not all(callable(v) for v in (init_state, seed_box, propagate, emit))
            or (release_state is not None and not callable(release_state))):
        raise ValueError("Explicit native callbacks and stream sink required")
    anchor = None; anchor_boxes = None; iterator = iter(frames)

    def record(branch, position, boxes, masks, support, scores, evidence):
        raw, native, usable = boxes
        emit(TemporalMaskFrame(branch, position, int(ids[position]), tuple(range(len(raw))),
            *(_copy(v, "owned streamed evidence") for v in (raw, native, usable, masks, support)),
            None if scores is None else _copy(scores, "raw native IMAGE scores"), evidence))

    for position, original_id in enumerate(ids):
        try:
            item = next(iterator)
        except StopIteration as error:
            raise ValueError("Original timeline is missing a frame") from error
        if (type(item) is not tuple or len(item) != 3 or type(item[0]) is not int
                or item[0] != original_id or type(item[1]) is not np.ndarray
                or item[1].dtype != np.uint8 or item[1].shape != (height, width, 3)):
            raise ValueError("Unchanged original IDs/RGB grid required in declared order")
        boxes = _boxes(item[2], height, width); raw, native, usable = boxes
        masks = np.zeros((len(raw), height, width), np.bool_)
        support = np.zeros(len(raw), np.bool_); scores = np.full(len(raw), np.nan, np.float64)
        active = np.flatnonzero(usable)
        if len(active):
            if not all(callable(getattr(image_predictor, n, None)) for n in ("set_image", "predict")):
                raise ValueError("Native IMAGE set_image/predict required")
            image_predictor.set_image(_copy(item[1], "readonly original RGB", dtype=np.uint8))
            values, score = _sam2_outputs(image_predictor.predict(
                box=_copy(native[active], "native IMAGE boxes"), multimask_output=False), len(active), height, width)
            masks[active] = values; support[active] = True; scores[active] = score
            if anchor is None:
                anchor = position; anchor_boxes = boxes
        record("A", position, boxes, masks, support, scores, "image_prompt")
        del item, masks
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("Original timeline contains an extra frame")

    if anchor is None:
        anchor_boxes = (_copy(np.empty((0, 4), np.float64), "empty raw boxes"),
                        _copy(np.empty((0, 4), np.float32), "empty native boxes"),
                        _copy(np.empty(0, np.bool_), "empty box support"))
        for position in range(len(ids)):
            record("B", position, anchor_boxes, np.empty((0, height, width), np.bool_),
                   np.empty(0, np.bool_), None, "no_anchor_abstention")
        seeded = (); counts = (0, 0)
    else:
        raw, native, usable = anchor_boxes
        seeded = tuple(map(int, np.flatnonzero(usable))); prior = None; counts = []
        for reverse in (False, True):
            state = init_state()
            if state is None or state is prior:
                raise ValueError("Forward/reverse must have distinct fresh native states")
            try:
                for proposal_id in seeded:
                    seed_box(state, frame_idx=anchor, obj_id=proposal_id,
                             box=_copy(native[proposal_id], "native anchor box"), normalize_coords=True)
                expected = tuple(range(anchor, -1, -1)) if reverse and anchor > 0 else \
                           (() if reverse else tuple(range(anchor, len(ids))))
                outputs = iter(propagate(state, start_frame_idx=anchor, reverse=reverse))
                for position in expected:
                    try:
                        value = next(outputs)
                    except StopIteration as error:
                        raise ValueError("Native propagation dropped an original frame") from error
                    if type(value) is not tuple or len(value) != 3 or type(value[0]) is not int or value[0] != position:
                        raise ValueError("Native frame position/order differs")
                    object_ids = value[1]
                    if (type(object_ids) not in (list, tuple) or any(type(i) is not int for i in object_ids)
                            or len(object_ids) != len(seeded) or set(object_ids) != set(seeded)):
                        raise ValueError("Native IDs must preserve every anchor slot exactly once")
                    logits = _copy(value[2], "native video logits", shape=(len(seeded), 1, height, width), kind="f")
                    if not np.isfinite(logits).all():
                        raise ValueError("Native video logits must be finite; no failed-frame filtering")
                    if not reverse or position < anchor:
                        masks = np.zeros((len(raw), height, width), np.bool_)
                        for native_index, proposal_id in enumerate(object_ids):
                            masks[proposal_id] = logits[native_index, 0] > 0.0
                        record("B", position, anchor_boxes, masks, usable, None,
                               "anchor_prompt" if position == anchor else "video_inferred")
                try:
                    next(outputs)
                except StopIteration:
                    pass
                else:
                    raise ValueError("Native propagation duplicated/added a frame")
                counts.append(len(expected))
            finally:
                if release_state is not None:
                    release_state(state)
            prior = state
    return TemporalMaskSummary(ids, image_size, anchor, None if anchor is None else int(ids[anchor]),
                               *anchor_boxes, seeded, *counts)
