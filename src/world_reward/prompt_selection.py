"""Deterministic, automatic detector-box seeds for the official SAM2 JSON schema.

Inputs are independent person/target detections, in full-frame *pixel xyxy*
coordinates. The caller must obtain ``object_prompt`` from allowed public metadata
and run that exact target query; this module never invents a label or repairs a
box. Missing, invalid and ambiguous frame candidates are rejected, not hand-fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from collections.abc import Sequence


@dataclass(frozen=True)
class BoxDetection:
    box: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class FrameDetections:
    frame_index: int
    width: int
    height: int
    persons: Sequence[BoxDetection] = ()
    objects: Sequence[BoxDetection] = ()


@dataclass(frozen=True)
class SeedSelection:
    frame_index: int
    person: BoxDetection
    object: BoxDetection
    object_prompt: str
    rejected_frames: tuple[tuple[int, str], ...] = ()

    @property
    def confidence(self) -> float:
        """Conservative pair confidence: its less-confident member."""
        return min(self.person.score, self.object.score)

    def to_sam2_json(self) -> dict:
        """Match pinned ``Sam2Prompts.to_dict``; no human points/masks included."""
        return {
            "prompts": [
                {
                    "frame_index": self.frame_index,
                    "object_id": object_id,
                    "points": None,
                    "point_labels": None,
                    "box": dict(zip(("x0", "y0", "x1", "y1"), detection.box)),
                    "mask_path": None,
                }
                for object_id, detection in ((0, self.person), (1, self.object))
            ]
        }


def _real(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validated_detection(
    detection: BoxDetection, width: int, height: int, threshold: float
) -> BoxDetection | None:
    if not isinstance(detection, BoxDetection):
        raise ValueError("detections must be BoxDetection instances")
    score = _real(detection.score)
    if score is None or not threshold <= score <= 1:
        return None
    try:
        if len(detection.box) != 4:
            return None
        box = tuple(_real(coordinate) for coordinate in detection.box)
    except TypeError:
        return None
    if any(coordinate is None for coordinate in box):
        return None
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    return BoxDetection(box=box, score=score)


def _ordered_valid_detections(
    detections: Sequence[BoxDetection], width: int, height: int, threshold: float,
) -> list[BoxDetection]:
    if not isinstance(detections, Sequence):
        raise ValueError("detections must be a sequence (empty when missing)")
    valid: dict[tuple[float, float, float, float], BoxDetection] = {}
    for detection in detections:
        checked = _validated_detection(detection, width, height, threshold)
        if checked is not None:
            previous = valid.get(checked.box)
            if previous is None or checked.score > previous.score:
                valid[checked.box] = checked
    return sorted(valid.values(), key=lambda item: (-item.score, item.box))


def _box_iou(a: BoxDetection, b: BoxDetection) -> float:
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    # A common scale leaves IoU unchanged and avoids overflowing area products.
    scale = max(ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0)
    intersection = (intersection_width / scale) * (intersection_height / scale)
    area_a = ((ax1 - ax0) / scale) * ((ay1 - ay0) / scale)
    area_b = ((bx1 - bx0) / scale) * ((by1 - by0) / scale)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(
    detections: Sequence[BoxDetection], width: int, height: int,
    confidence_threshold: float, iou_threshold: float = 0.7,
) -> tuple[BoxDetection, ...]:
    """Greedy same-class NMS; run independently for person and object queries.

    Drop invalid/low-confidence boxes, deduplicate exact boxes, then retain the
    highest-confidence original box and suppress candidates with IoU *strictly*
    greater than the threshold. Equal scores sort by xyxy coordinates, making
    results independent of input order. No averaging, clamping or box invention.
    Threshold 0 suppresses any positive overlap; threshold 1 keeps all distinct
    boxes. Containment alone is not duplication: a giant enclosing box has low
    IoU with a small instance and remains available to the ambiguity gate.
    """
    width = _integer(width, "width", minimum=1)
    height = _integer(height, "height", minimum=1)
    confidence = _real(confidence_threshold)
    threshold = _real(iou_threshold)
    if confidence is None or not 0 <= confidence <= 1:
        raise ValueError("confidence_threshold must be finite in [0, 1]")
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("iou_threshold must be finite in [0, 1]")
    ordered = _ordered_valid_detections(detections, width, height, confidence)
    kept: list[BoxDetection] = []
    for candidate in ordered:
        if not any(_box_iou(candidate, previous) > threshold for previous in kept):
            kept.append(candidate)
    return tuple(kept)


def _select_detection(
    detections: Sequence[BoxDetection], width: int, height: int,
    threshold: float, ambiguity_margin: float,
) -> tuple[BoxDetection | None, str | None]:
    # NMS is an explicit caller-side preprocessing step; selecting raw near-tied
    # boxes still fails closed. Exact duplicates remain a single hypothesis.
    ordered = _ordered_valid_detections(detections, width, height, threshold)
    if not ordered:
        return None, "no valid detection above confidence threshold"
    if len(ordered) > 1:
        difference = ordered[0].score - ordered[1].score
        if difference <= ambiguity_margin or math.isclose(
            difference, ambiguity_margin, rel_tol=0, abs_tol=1e-12
        ):
            return None, "ambiguous distinct detections within confidence margin"
    return ordered[0], None


def select_seed_prompts(
    frames: Sequence[FrameDetections], *, object_prompt: str,
    confidence_threshold: float = 0.3, ambiguity_margin: float = 0.05,
    total_frames: int | None = None,
) -> SeedSelection:
    """Choose an unambiguous same-frame pair, or fail instead of inventing boxes.

    Rank valid pairs by minimum member confidence, then confidence sum; exact
    frame-ranking ties choose the earliest original frame index, independent of
    input order. Distinct near-tied detections *within* a class reject that frame.
    This is confidence selection, not a claim that the frame is fully unoccluded.
    The exact public-metadata query is retained verbatim for provenance.
    """
    if not isinstance(object_prompt, str) or not object_prompt.strip():
        raise ValueError("a nonempty exact public-metadata object_prompt is required")
    threshold = _real(confidence_threshold)
    margin = _real(ambiguity_margin)
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be finite in [0, 1]")
    if margin is None or not 0 <= margin <= 1:
        raise ValueError("ambiguity_margin must be finite in [0, 1]")
    if total_frames is not None:
        total_frames = _integer(total_frames, "total_frames", minimum=1)

    if not isinstance(frames, Sequence):
        raise ValueError("frames must be a sequence")
    seen: set[int] = set()
    rejected: list[tuple[int, str]] = []
    pairs: list[SeedSelection] = []
    for frame in frames:
        if not isinstance(frame, FrameDetections):
            raise ValueError("frames must be FrameDetections instances")
        index = _integer(frame.frame_index, "frame_index", minimum=0)
        if index in seen:
            raise ValueError(f"duplicate frame_index {index}")
        if total_frames is not None and index >= total_frames:
            raise ValueError(f"frame_index {index} is outside total_frames")
        seen.add(index)
        width = _integer(frame.width, "width", minimum=1)
        height = _integer(frame.height, "height", minimum=1)
        person, person_error = _select_detection(
            frame.persons, width, height, threshold, margin
        )
        obj, object_error = _select_detection(
            frame.objects, width, height, threshold, margin
        )
        errors = [f"{kind}: {error}" for kind, error in (
            ("person", person_error), ("object", object_error)
        ) if error]
        if errors:
            rejected.append((index, "; ".join(errors)))
            continue
        pairs.append(SeedSelection(index, person, obj, object_prompt))

    if not pairs:
        reasons = "; ".join(f"frame {index}: {reason}" for index, reason in sorted(rejected))
        raise ValueError("No unambiguous valid person/object seed pair" + (
            f" ({reasons})" if reasons else ": no frames supplied"
        ))
    best = min(pairs, key=lambda pair: (
        -pair.confidence, -(pair.person.score + pair.object.score), pair.frame_index
    ))
    return SeedSelection(
        best.frame_index, best.person, best.object, object_prompt,
        tuple(sorted(rejected)),
    )
