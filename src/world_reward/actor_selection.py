"""Select the interacting actor from sparse automatic boxes, not human labels.

Per-class NMS must precede this step. Person identity uses only gated sparse-frame
IoU; ambiguous associations fail closed instead of silently changing identity.
Affinity uses actual observed boxes on unambiguous target-object frames. It is an
image-evidence engineering gate, not a guarantee of identity under arbitrary
occlusion, camera cuts or crossings between sampled frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from statistics import median

import numpy as np
from scipy.optimize import linear_sum_assignment

from world_reward.prompt_selection import (
    BoxDetection, FrameDetections, non_maximum_suppression,
)


@dataclass(frozen=True)
class ActorTrackScore:
    track_id: int
    outside_distance: float
    center_distance: float
    observations: int
    coverage: float


@dataclass(frozen=True)
class ActorSelection:
    frames: tuple[FrameDetections, ...]
    track_id: int
    scores: tuple[ActorTrackScore, ...]


@dataclass
class _Track:
    track_id: int
    last_step: int
    last_box: BoxDetection
    observations: dict[int, BoxDetection] = field(default_factory=dict)


def _unit_interval(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return result


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _iou(a: BoxDetection, b: BoxDetection) -> float:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union


def _associate(
    active: list[_Track], detections: tuple[BoxDetection, ...],
    iou_threshold: float, margin: float,
) -> dict[int, int]:
    """Maximum-weight one-to-one assignment with explicit unmatched dummies."""
    if not active or not detections:
        return {}
    count_tracks, count_boxes = len(active), len(detections)
    size = count_tracks + count_boxes
    weights = np.zeros((size, size), dtype=np.float64)
    weights[:count_tracks, :count_boxes] = -1.0
    for i, track in enumerate(active):
        for j, detection in enumerate(detections):
            overlap = _iou(track.last_box, detection)
            if overlap > iou_threshold:
                weights[i, j] = overlap
    rows, columns = linear_sum_assignment(weights, maximize=True)
    optimum = float(weights[rows, columns].sum())
    matches = {
        int(i): int(j) for i, j in zip(rows, columns)
        if i < count_tracks and j < count_boxes and weights[i, j] > 0
    }
    # A near-equivalent assignment with an actual identity edge removed is a
    # crossing/identity ambiguity. Abort globally: restarting those detections as
    # fresh eligible tracks would launder ambiguous identity through fragmentation.
    for i, j in matches.items():
        alternative = weights.copy()
        alternative[i, j] = -1.0
        alt_rows, alt_columns = linear_sum_assignment(alternative, maximize=True)
        difference = optimum - float(alternative[alt_rows, alt_columns].sum())
        if difference <= margin or math.isclose(difference, margin, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Ambiguous person-track association; no identity flip or restart allowed")
    return matches


def _affinity(person: BoxDetection, obj: BoxDetection) -> tuple[float, float]:
    x0, y0, x1, y1 = person.box
    ox0, oy0, ox1, oy1 = obj.box
    ox, oy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
    diagonal = math.hypot(x1 - x0, y1 - y0)
    outside = math.hypot(max(x0 - ox, 0.0, ox - x1), max(y0 - oy, 0.0, oy - y1))
    center = math.hypot(ox - (x0 + x1) / 2, oy - (y0 + y1) / 2)
    return outside / diagonal, center / diagonal


def select_interacting_actor(
    frames: Sequence[FrameDetections], *, confidence_threshold: float = 0.3,
    object_ambiguity_margin: float = 0.05, association_iou_threshold: float = 0.1,
    association_margin: float = 0.05, max_missing_observations: int = 1,
    min_observations: int = 3, min_coverage: float = 0.5,
    affinity_margin: float = 0.05,
) -> ActorSelection:
    """Track people and require a clearly closer, sufficiently observed actor.

    Primary score is median object-center distance *outside* the person's box,
    divided by person-box diagonal (zero inside). For exact primary ties, including
    two enclosing boxes, median normalized center distance is secondary. A winner
    must beat the runner-up by more than ``affinity_margin`` on that ranking axis,
    and have >= ``min_observations`` plus >= ``min_coverage`` of unambiguous object
    frames. Sparse observations are never interpolated or joined after expiry.
    ``max_missing_observations`` counts missing supplied sparse-frame observations,
    not elapsed video frames. Ambiguous background crossings also fail this simple
    implementation rather than guessing which identity remained independent.
    """
    threshold = _unit_interval(confidence_threshold, "confidence_threshold")
    object_margin = _unit_interval(object_ambiguity_margin, "object_ambiguity_margin")
    iou_threshold = _unit_interval(association_iou_threshold, "association_iou_threshold")
    match_margin = _unit_interval(association_margin, "association_margin")
    coverage_threshold = _unit_interval(min_coverage, "min_coverage")
    affinity_threshold = _unit_interval(affinity_margin, "affinity_margin")
    maximum_missing = _integer(max_missing_observations, "max_missing_observations", 0)
    minimum_observations = _integer(min_observations, "min_observations", 3)
    if coverage_threshold < 0.5:
        raise ValueError("min_coverage must be >= 0.5")
    if not isinstance(frames, Sequence) or not frames:
        raise ValueError("Nonempty sequence of FrameDetections is required")

    indexed: dict[int, FrameDetections] = {}
    dimensions = None
    for frame in frames:
        if not isinstance(frame, FrameDetections):
            raise ValueError("frames must be FrameDetections instances")
        index = _integer(frame.frame_index, "frame_index", 0)
        width = _integer(frame.width, "width", 1)
        height = _integer(frame.height, "height", 1)
        if index in indexed:
            raise ValueError(f"duplicate frame_index {index}")
        if dimensions is not None and dimensions != (width, height):
            raise ValueError("Sparse frames must share one pixel coordinate system")
        dimensions = width, height
        indexed[index] = frame

    ordered_frames = sorted(indexed.items())
    tracks: list[_Track] = []
    objects: dict[int, BoxDetection] = {}
    for step, (index, frame) in enumerate(ordered_frames):
        # IoU=1 only validates/deduplicates; actual per-class NMS is caller-side.
        persons = non_maximum_suppression(frame.persons, frame.width, frame.height, threshold, 1)
        candidates = non_maximum_suppression(frame.objects, frame.width, frame.height, threshold, 1)
        if candidates:
            unambiguous = len(candidates) == 1
            if len(candidates) > 1:
                gap = candidates[0].score - candidates[1].score
                unambiguous = gap > object_margin and not math.isclose(
                    gap, object_margin, rel_tol=0, abs_tol=1e-12
                )
            if unambiguous:
                objects[index] = candidates[0]

        active = [track for track in tracks if step - track.last_step <= maximum_missing + 1]
        matches = _associate(active, persons, iou_threshold, match_margin)
        matched_detections = set(matches.values())
        for track_index, detection_index in matches.items():
            track = active[track_index]
            track.last_step = step
            track.last_box = persons[detection_index]
            track.observations[index] = persons[detection_index]
        for detection_index, person in enumerate(persons):
            if detection_index not in matched_detections:
                tracks.append(_Track(len(tracks), step, person, {index: person}))

    if len(objects) < minimum_observations:
        raise ValueError("Insufficient unambiguous object frames for actor selection")
    scores: list[ActorTrackScore] = []
    for track in tracks:
        values = [
            _affinity(person, objects[index])
            for index, person in track.observations.items() if index in objects
        ]
        if values:
            outside, center = zip(*values)
            scores.append(ActorTrackScore(
                track.track_id, float(median(outside)), float(median(center)),
                len(values), len(values) / len(objects),
            ))
    if not scores:
        raise ValueError("No person track observed on valid object frames")
    scores.sort(key=lambda score: (score.outside_distance, score.center_distance, score.track_id))
    winner = scores[0]
    # Unsupported close tracks still compete. Never discard them to manufacture
    # a comfortable margin for a farther, well-observed distractor.
    if winner.observations < minimum_observations or winner.coverage < coverage_threshold:
        raise ValueError("Closest actor track has insufficient observations or coverage")
    if len(scores) > 1:
        runner = scores[1]
        if math.isclose(winner.outside_distance, runner.outside_distance, rel_tol=0, abs_tol=1e-12):
            gap = runner.center_distance - winner.center_distance
        else:
            gap = runner.outside_distance - winner.outside_distance
        if gap <= affinity_threshold or math.isclose(gap, affinity_threshold, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Ambiguous actor affinity; winner lacks clear margin over runner-up")

    observations = tracks[winner.track_id].observations
    filtered = tuple(
        FrameDetections(
            index, int(frame.width), int(frame.height),
            (observations[index],) if index in observations else (), frame.objects,
        )
        for index, frame in ordered_frames
    )
    return ActorSelection(filtered, winner.track_id, tuple(scores))
