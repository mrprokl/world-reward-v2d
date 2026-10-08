"""Pure contracts for automatic box-conditioned instance video tracking.

SAM3.1's semantic box API is an exemplar query, not an instance ID. Its native
interactive prompt encoder explicitly accepts labels 2/3 for opposite box
corners. This codec uses that instance route; it does not create click labels.
"""
from __future__ import annotations

import math


def corner_prompt(box, width: int, height: int):
    """Original exclusive XYXY pixels -> native relative corner tokens 2/3."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError("Positive original integer image grid required")
    if type(box) not in (list, tuple) or len(box) != 4:
        raise ValueError("Four original XYXY bounds required")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
        raise ValueError("Finite, nonboolean original box coordinates required")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Box must be nonempty on the original image; never clamp")
    return [[x0 / width, y0 / height], [x1 / width, y1 / height]], [2, 3]


def seed_rows(rows, episode: int, indices, width: int, height: int, rgb_hashes):
    """No best-result picking: consume all complete saved prefix pair boxes."""
    selected = [r for r in rows if r.get("ep") == episode]
    if sorted(r.get("index") for r in selected) != sorted(indices):
        raise ValueError("Exact frozen saved prefix frame cohort required")
    by_index = {r["index"]: r for r in selected}
    result = []
    for index in indices:
        row = by_index[index]
        if (row.get("width"), row.get("height"), row.get("rgb_sha256")) != (
            width, height, rgb_hashes[index]
        ):
            raise ValueError("Saved model evidence differs from original RGB/grid")
        if row.get("status") != "pair_returned":
            raise ValueError("Missing seed pair remains an explicit failure")
        for object_id, role in enumerate(("person", "object")):
            box = row.get(role + "_bbox")
            points, labels = corner_prompt(box, width, height)
            result.append(dict(frame_index=index, object_id=object_id, role=role,
                               box=box, points=points, point_labels=labels,
                               rgb_sha256=row["rgb_sha256"],
                               request_sha256=row["request_sha256"]))
    return result


def native_masks(ids, logits, object_logits, height: int, width: int):
    """Native sign decisions, with absence kept explicitly (no static filling)."""
    import numpy as np
    if type(ids) is not list or ids != [0, 1]:
        raise ValueError("Exactly two fixed native physical identities required")
    logits = np.asarray(logits)
    scores = np.asarray(object_logits)
    if (logits.shape != (2, 1, height, width) or scores.shape not in ((2,), (2, 1))
            or not np.isfinite(logits).all() or not np.isfinite(scores).all()):
        raise ValueError("Finite original-grid native mask/presence logits required")
    presence = scores.reshape(2) > 0
    masks = (logits[:, 0] > 0) & presence[:, None, None]
    return masks, presence


def temporal_summary(areas, adjacent_ious):
    """Descriptive video-only diagnostics, never an accuracy metric or filter."""
    import numpy as np
    if len(areas) < 1 or any(type(n) is not int or n < 0 for n in areas):
        raise ValueError("Explicit nonnegative original-grid areas required")
    if len(adjacent_ious) != len(areas) - 1:
        raise ValueError("Full original frame adjacency required")
    measured = [x for x in adjacent_ious if x is not None]
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in measured):
        raise ValueError("Valid literal-mask IoUs required")
    gaps, current = [], 0
    for n in areas:
        if n == 0:
            current += 1
        elif current:
            gaps.append(current); current = 0
    if current:
        gaps.append(current)
    return dict(frames=len(areas), visible_frames=sum(n > 0 for n in areas),
                empty_frames=sum(n == 0 for n in areas), longest_empty_run=max(gaps, default=0),
                median_visible_pixels=float(np.median([n for n in areas if n > 0])) if any(areas) else 0.0,
                adjacent_mask_iou_median=float(np.median(measured)) if measured else None,
                quality_verified=False, interpolation=False)
