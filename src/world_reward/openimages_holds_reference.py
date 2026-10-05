"""Pure Open Images holds parsing for the prospective 128-image protocol.

No source/data I/O, hierarchy aliases, new reference boxes or geometry repair.
Coordinates are original normalized FP64. Every box (including group/unknown
group boxes) remains in the reference inventory. Relation endpoints deduplicate
by exact class/coordinates, then require ONE same-class IoU>=.5 box; an exact
match does not bypass an overlapping second/group match. Unscorable positive
person slots remain explicit retrieval misses, never disappeared annotations.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
import math
import re

import numpy as np

from .hoi_object_ranking_evaluation import ReferenceRegions

PERSON_CLASSES = frozenset(("/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"))


def _owned(value, dtype, shape=None):
    a = np.asarray(value, dtype=dtype)
    if shape is not None:
        a = a.reshape(shape)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _class(value):
    if type(value) is not str or not re.fullmatch(r"/[mg]/[A-Za-z0-9_]{1,128}", value):
        raise ValueError("Original class MID required; no class aliases")
    return value


def _box(row, suffix=""):
    try:
        raw = [row[k + suffix] for k in ("XMin", "YMin", "XMax", "YMax")]
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (str, int, float, np.integer, np.floating)) for x in raw):
            raise ValueError("Numeric original normalized coordinates required")
        b = tuple(float(x) for x in raw)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Complete original coordinate fields required") from exc
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in b) or b[2] <= b[0] or b[3] <= b[1]:
        raise ValueError("Finite positive-area original boxes inside [0,1] required")
    return b


def _group(value):
    if type(value) is str and value in ("-1", "0", "1"):
        return int(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value in (-1, 0, 1):
        return int(value)
    raise ValueError("Original IsGroupOf must be -1, 0 or 1; unknown censors")


def parse_holds_vocabulary(rows):
    """Read official triplet row dictionaries or CSV tuples/header, no aliases.

    Returns unique sorted exact (person MID, nonperson MID) holds class pairs.
    Other relations/subject classes are not added to the evaluation vocabulary.
    """
    pairs = set()
    for i, row in enumerate(rows):
        if isinstance(row, Mapping):
            try:
                a, b, relation = (row[k] for k in ("LabelName1", "LabelName2", "RelationshipLabel"))
            except KeyError as exc:
                raise ValueError("Official triplet fields required") from exc
        elif isinstance(row, (tuple, list)) and len(row) == 3:
            a, b, relation = row
            if i == 0 and tuple(row) == ("LabelName1", "LabelName2", "RelationshipLabel"):
                continue
        else:
            raise ValueError("Official three-column triplet row required")
        if type(relation) is not str or not relation:
            raise ValueError("Original relationship label required")
        if relation == "holds":
            a, b = _class(a), _class(b)
            if a in PERSON_CLASSES and b not in PERSON_CLASSES:
                pairs.add((a, b))
    return tuple(sorted(pairs))


@dataclass(frozen=True, eq=False)
class OpenImagesHoldsReference:
    """Unbound count is unscorable person slots, including unbound held objects.

    Distinct unbound person/object endpoint counts live in binding_diagnostics.
    The consumer must honor positive_person_unscorable for the retrieval
    denominator; resolved holds alone are not the complete positive inventory.
    """
    persons: ReferenceRegions
    objects: ReferenceRegions
    holds: np.ndarray                     # unique resolved (person,object) indices
    positive_person_classes: tuple[str, ...]
    positive_person_boxes_xyxy: np.ndarray # all exact-deduplicated positive endpoints
    positive_person_indices: np.ndarray   # -1 if endpoint cannot bind
    positive_person_unscorable: np.ndarray # includes any unbound positive object
    unscorable_positive_person_slots: np.ndarray
    total_positive_person_slots: int
    unbound_positive_person_count: int
    binding_diagnostics: Mapping


def parse_holds_reference(bbox_rows, relation_rows):
    """Parse one image's original CSV row dictionaries without I/O or metrics.

    Unknown IsGroupOf=-1 censors just like group=1 and is counted separately.
    Duplicate box rows remain distinct/ambiguous; only repeated positive
    relation endpoints/pairs deduplicate. A person with ANY unresolved positive
    endpoint stays flagged unscorable even if another relation resolves.
    """
    inventories = [[], []]
    unknown_groups = 0
    image_id = None
    def same_image(row):
        nonlocal image_id
        if "ImageID" in row:
            value = row["ImageID"]
            if type(value) is not str or not re.fullmatch(r"[0-9a-f]{16}", value):
                raise ValueError("Original image ID required")
            if image_id is not None and image_id != value:
                raise ValueError("One original image only")
            image_id = value
    for i, row in enumerate(bbox_rows):
        if not isinstance(row, Mapping):
            raise ValueError("Original bounding-box row dictionary required")
        same_image(row)
        try:
            cls, group = _class(row["LabelName"]), _group(row["IsGroupOf"])
        except KeyError as exc:
            raise ValueError("Original bounding-box class/group fields required") from exc
        unknown_groups += group == -1
        inventories[int(cls not in PERSON_CLASSES)].append((f"bbox:{i:08d}", cls, _box(row), group != 0))
    regions = [ReferenceRegions(tuple(x[0] for x in r), tuple(x[1] for x in r),
        _owned([x[2] for x in r], np.float64, (-1, 4)), _owned([x[3] for x in r], np.bool_)) for r in inventories]
    positive, duplicate, ignored = [], 0, 0
    seen = set()
    for row in relation_rows:
        if not isinstance(row, Mapping) or type(row.get("RelationshipLabel")) is not str:
            raise ValueError("Original RelationshipLabel row dictionary required")
        same_image(row)
        if row["RelationshipLabel"] != "holds":
            ignored += 1; continue
        try:
            a, b = _class(row["LabelName1"]), _class(row["LabelName2"])
        except KeyError as exc:
            raise ValueError("Original relation endpoint classes required") from exc
        pa, pb = (a, _box(row, "1")), (b, _box(row, "2"))
        if a not in PERSON_CLASSES or b in PERSON_CLASSES:
            ignored += 1; continue
        if (pa, pb) in seen:
            duplicate += 1; continue
        seen.add((pa, pb)); positive.append((pa, pb))
    diagnostics = dict(exact_bindings=0, iou_bindings=0, no_match_endpoints=0,
        ambiguous_endpoints=0, group_endpoints=0, unknown_group_boxes=unknown_groups,
        duplicate_positive_relations=duplicate, ignored_relations=ignored)
    cache = {}
    def bind(endpoint, region):
        if endpoint in cache:
            return cache[endpoint]
        cls, box = endpoint; b = np.asarray(box, np.float64)
        candidates = np.flatnonzero(np.array([c == cls for c in region.class_ids], bool))
        ob = region.boxes_xyxy[candidates]
        width = np.maximum(0., np.minimum(b[2:], ob[:, 2:]) - np.maximum(b[:2], ob[:, :2]))
        intersection = width[:, 0] * width[:, 1]
        union = np.prod(b[2:] - b[:2]) + np.prod(ob[:, 2:] - ob[:, :2], axis=1) - intersection
        matched = candidates[np.divide(intersection, union, out=np.zeros(len(union)), where=union > 0) >= .5]
        if len(matched) != 1:
            diagnostics["no_match_endpoints" if not len(matched) else "ambiguous_endpoints"] += 1
            result = -1
        elif region.group_of[matched[0]]:
            diagnostics["group_endpoints"] += 1; result = -1
        else:
            result = int(matched[0])
            diagnostics["exact_bindings" if np.array_equal(b, region.boxes_xyxy[result]) else "iou_bindings"] += 1
        cache[endpoint] = result
        return result
    person_slots, indices, unscorable, holds = {}, [], [], set()
    unbound_people, unbound_objects, unbound_relations = set(), set(), 0
    for pa, pb in positive:
        pi, oi = bind(pa, regions[0]), bind(pb, regions[1])
        if pa not in person_slots:
            person_slots[pa] = len(person_slots); indices.append(pi); unscorable.append(False)
        slot = person_slots[pa]
        if pi < 0 or oi < 0:
            unscorable[slot] = True
            unbound_relations += 1
        if pi < 0:
            unbound_people.add(pa)
        if oi < 0:
            unbound_objects.add(pb)
        if pi >= 0 and oi >= 0:
            holds.add((pi, oi))
    endpoints = tuple(person_slots)
    diagnostics.update(positive_relations=len(positive), endpoint_count=len(cache), resolved_positive_pairs=len(holds),
        unbound_person_endpoints=len(unbound_people), unbound_object_endpoints=len(unbound_objects),
        unbound_positive_relations=unbound_relations)
    flags = _owned(unscorable, np.bool_)
    return OpenImagesHoldsReference(*regions, _owned(sorted(holds), np.int64, (-1, 2)),
        tuple(x[0] for x in endpoints), _owned([x[1] for x in endpoints], np.float64, (-1, 4)),
        _owned(indices, np.int64), flags, _owned(np.flatnonzero(flags), np.int64), len(endpoints),
        int(flags.sum()), MappingProxyType(diagnostics))
