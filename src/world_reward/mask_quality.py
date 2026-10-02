"""Ground-truth-free geometry/temporal diagnostics for automatic binary masks.

These are image-space proxies, not mask accuracy or actor/contact correctness.
Empty masks, overlap, fragmentation, large motion and occlusion are reported, not
rejected as prediction errors. Only malformed inputs fail. Mask arrays may be
streamed frame by frame; the report never retains masks or component label maps.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from numbers import Integral, Real

import numpy as np
from scipy import ndimage

from world_reward.prompt_selection import BoxDetection, FrameDetections


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _mask_iterator(masks: Iterable[np.ndarray], name: str):
    if isinstance(masks, np.ndarray) and masks.ndim != 3:
        raise ValueError(f"{name} must have shape [T, H, W]")
    try:
        return iter(masks)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of [H, W] boolean masks") from exc


def _validate_mask(mask: object, name: str, frame_index: int) -> np.ndarray:
    if np.ma.isMaskedArray(mask):
        raise ValueError(f"{name} at frame {frame_index}: masked arrays are not binary masks")
    try:
        value = np.asarray(mask)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} at frame {frame_index}: invalid mask array") from exc
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} at frame {frame_index}: expected nonempty [H, W]")
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{name} at frame {frame_index}: mask contains nonfinite values")
    if value.dtype != np.dtype(bool):
        raise ValueError(f"{name} at frame {frame_index}: boolean mask required (no implicit threshold)")
    return value


def _geometry(mask: np.ndarray) -> dict:
    height, width = mask.shape
    y, x = np.nonzero(mask)
    area = int(x.size)
    result = {
        "area_pixels": area,
        "area_fraction": area / (height * width),
        "bbox_xyxy": None,
        "centroid_xy": None,
        "component_count": 0,
        "largest_component_area_pixels": 0,
        "largest_component_fraction": None,
        "principal_components": [],
    }
    if not area:
        return result
    result["bbox_xyxy"] = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
    result["centroid_xy"] = [float(x.mean()) + 0.5, float(y.mean()) + 0.5]
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel())[1:]
    # Top three components keep noisy masks from producing enormous reports.
    # Equal-area ties use row-major component label order, not a visual choice.
    principal_labels = sorted(range(1, count + 1), key=lambda label: (-int(areas[label - 1]), label))[:3]
    slices = ndimage.find_objects(labels)
    centroids = ndimage.center_of_mass(mask, labels, principal_labels)
    components = []
    for label, center in zip(principal_labels, centroids):
        sy, sx = slices[label - 1]
        component_area = int(areas[label - 1])
        components.append({
            "area_pixels": component_area,
            "fraction_of_mask": component_area / area,
            "bbox_xyxy": [int(sx.start), int(sy.start), int(sx.stop), int(sy.stop)],
            "centroid_xy": [float(center[1]) + 0.5, float(center[0]) + 0.5],
        })
    result["component_count"] = int(count)
    result["largest_component_area_pixels"] = int(areas.max())
    result["largest_component_fraction"] = int(areas.max()) / area
    result["principal_components"] = components
    return result


def _overlap(person: np.ndarray, obj: np.ndarray, pa: int, oa: int) -> dict:
    intersection = int(np.count_nonzero(person & obj))
    union = pa + oa - intersection
    return {
        "intersection_pixels": intersection,
        "fraction_of_person": intersection / pa if pa else None,
        "fraction_of_object": intersection / oa if oa else None,
        "mask_iou": intersection / union if union else None,
    }


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return intersection / union


def _checked_detections(detections: Sequence[BoxDetection], width: int, height: int) -> list[BoxDetection]:
    if not isinstance(detections, Sequence):
        raise ValueError("Independent detections must be a sequence")
    checked = []
    for detection in detections:
        if not isinstance(detection, BoxDetection):
            raise ValueError("Independent detections must be BoxDetection instances")
        try:
            coordinates = tuple(detection.box)
        except TypeError as exc:
            raise ValueError("Independent detection requires finite pixel xyxy box") from exc
        if len(coordinates) != 4 or any(
            isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(float(x))
            for x in coordinates
        ):
            raise ValueError("Independent detection requires finite pixel xyxy box")
        x0, y0, x1, y1 = tuple(float(x) for x in coordinates)
        score = detection.score
        if (isinstance(score, bool) or not isinstance(score, Real)
                or not math.isfinite(float(score)) or not 0 <= float(score) <= 1):
            raise ValueError("Independent detection score must be finite in [0, 1]")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("Independent detection box outside image bounds or nonpositive area")
        checked.append(BoxDetection((x0, y0, x1, y1), float(score)))
    return sorted(checked, key=lambda detection: (-detection.score, detection.box))


def _agreement(bbox: list[int] | None, detections: list[BoxDetection]) -> dict:
    comparisons = [
        {"box_xyxy": list(detection.box), "score": detection.score,
         "bbox_iou": _box_iou(bbox, detection.box) if bbox is not None else None}
        for detection in detections
    ]
    return {
        "detections": comparisons,
        "max_bbox_iou": max(item["bbox_iou"] for item in comparisons)
        if bbox is not None and comparisons else None,
    }


def _change(previous: dict, current: dict, diagonal: float, gap: int) -> dict:
    before, after = previous["area_pixels"], current["area_pixels"]
    area_delta = after - before
    c0, c1 = previous["centroid_xy"], current["centroid_xy"]
    distance = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) if c0 is not None and c1 is not None else None
    return {
        "area_delta_pixels": area_delta,
        "absolute_area_change_over_max_area": abs(area_delta) / max(before, after)
        if max(before, after) else 0.0,
        "centroid_distance_pixels": distance,
        "centroid_distance_over_image_diagonal": distance / diagonal if distance is not None else None,
        "centroid_distance_over_image_diagonal_per_frame": distance / (diagonal * gap)
        if distance is not None else None,
        "empty_transition": (before == 0) != (after == 0),
    }


def analyze_mask_sequence(
    person_masks: Iterable[np.ndarray], object_masks: Iterable[np.ndarray],
    frame_indices: Sequence[int], *, detections: Sequence[FrameDetections] | None = None,
    total_frames: int | None = None,
) -> dict:
    """Report strict paired binary-mask geometry in original video-frame space.

    ``person_masks`` and ``object_masks`` are boolean [T,H,W] arrays or iterables of
    boolean [H,W] arrays, with one mask per strictly increasing original index.
    Sparse indices are allowed; temporal records explicitly retain their gaps.
    Bboxes are half-open pixel xyxy; centroids use pixel centers (x+0.5,y+0.5).
    Components use 8-connectivity, reporting only the largest three.
    Optional independent detections must use those same indices/dimensions; their
    box agreement is a proxy, not GT, and may itself be wrong. Missing detections
    are represented as absent, not extrapolated. No quality cutoffs are applied.
    """
    if not isinstance(frame_indices, Sequence) and not isinstance(frame_indices, np.ndarray):
        raise ValueError("frame_indices must be a sequence of original integer indices")
    indices = [_integer(index, "frame_index", 0) for index in frame_indices]
    if not indices or any(after <= before for before, after in zip(indices, indices[1:])):
        raise ValueError("frame_indices must be nonempty and strictly increasing (no duplicates)")
    if total_frames is not None:
        total_frames = _integer(total_frames, "total_frames", 1)
        if indices[-1] >= total_frames:
            raise ValueError("Original frame index is outside total_frames")
    detection_frames: dict[int, FrameDetections] = {}
    if detections is not None:
        if not isinstance(detections, Sequence):
            raise ValueError("detections must be a sequence of FrameDetections")
        for frame in detections:
            if not isinstance(frame, FrameDetections):
                raise ValueError("detections must contain FrameDetections instances")
            index = _integer(frame.frame_index, "detection frame_index", 0)
            if index not in indices or index in detection_frames:
                raise ValueError("Detection frame indices must be unique original mask indices")
            detection_frames[index] = frame

    person_iter = _mask_iterator(person_masks, "person_masks")
    object_iter = _mask_iterator(object_masks, "object_masks")
    records, temporal = [], []
    dimensions = None
    previous = None
    try:
        for index, person_input, object_input in zip(indices, person_iter, object_iter, strict=True):
            person = _validate_mask(person_input, "person_masks", index)
            obj = _validate_mask(object_input, "object_masks", index)
            if person.shape != obj.shape or (dimensions is not None and person.shape != dimensions):
                raise ValueError(f"Mask shapes must share fixed [H,W] at original frame {index}")
            dimensions = person.shape
            height, width = dimensions
            person_geometry, object_geometry = _geometry(person), _geometry(obj)
            record = {
                "frame_index": index,
                "person": person_geometry,
                "object": object_geometry,
                "overlap": _overlap(person, obj, person_geometry["area_pixels"], object_geometry["area_pixels"]),
                "detection_agreement": None,
            }
            if index in detection_frames:
                frame = detection_frames[index]
                if (_integer(frame.width, "detection width", 1), _integer(frame.height, "detection height", 1)) != (width, height):
                    raise ValueError("Independent detections must share mask image dimensions")
                record["detection_agreement"] = {
                    "person": _agreement(person_geometry["bbox_xyxy"], _checked_detections(frame.persons, width, height)),
                    "object": _agreement(object_geometry["bbox_xyxy"], _checked_detections(frame.objects, width, height)),
                }
            if previous is not None:
                gap = index - previous["frame_index"]
                diagonal = math.hypot(width, height)
                temporal.append({
                    "from_frame_index": previous["frame_index"], "to_frame_index": index,
                    "frame_gap": gap, "contiguous": gap == 1,
                    "person": _change(previous["person"], person_geometry, diagonal, gap),
                    "object": _change(previous["object"], object_geometry, diagonal, gap),
                })
            records.append(record)
            previous = record
    except ValueError as exc:
        if str(exc).startswith("zip() argument"):
            raise ValueError("Mask counts must match original frame_indices length") from exc
        raise
    height, width = dimensions
    return {
        "schema": "world-reward-mask-diagnostics-v1",
        "interpretation": "Ground-truth-free geometry/temporal proxies; not accuracy or correctness estimates",
        "width": width, "height": height, "frame_count": len(indices),
        "original_frame_indices": indices, "component_connectivity": 8,
        "frames": records, "temporal_pairs": temporal,
        "summary": {
            "person_empty_frames": sum(record["person"]["area_pixels"] == 0 for record in records),
            "object_empty_frames": sum(record["object"]["area_pixels"] == 0 for record in records),
            "overlap_frames": sum(record["overlap"]["intersection_pixels"] > 0 for record in records),
            "frames_with_independent_detections": len(detection_frames),
        },
    }
