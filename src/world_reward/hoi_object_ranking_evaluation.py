"""External person-region held-object attribution, NOT hand/contact truth.

Reference geometry stays exclusively in this evaluator. Both score arms use
the same native proposals and evaluator-only eligible hand slots. Missing and
unknown candidates remain in retrieval; conditional concordance cannot hide
its matching denominator. No dataset parser, model, quality decision or I/O.
"""
from dataclasses import dataclass
import math

import numpy as np

from .hoi_detr_observations import HOIDetrObservations
from .hoi_object_ranking import score_hoi_objects


def _owned(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _keys(values, name, unique=False):
    if (type(values) is not tuple or any(type(v) is not str or not v.strip()
            or len(v) > 256 or any(ord(c) < 32 for c in v) for v in values)
            or unique and len(set(values)) != len(values)):
        raise ValueError(name + ": bounded string tuple required")
    return values


@dataclass(frozen=True, eq=False)
class ReferenceRegions:
    instance_ids: tuple[str, ...]
    class_ids: tuple[str, ...]
    boxes_xyxy: np.ndarray
    group_of: np.ndarray

    def __post_init__(self):
        _keys(self.instance_ids, "instance_ids", unique=True)
        _keys(self.class_ids, "class_ids")
        n = len(self.instance_ids); b, g = np.asarray(self.boxes_xyxy), np.asarray(self.group_of)
        if (len(self.class_ids) != n or np.ma.isMaskedArray(self.boxes_xyxy)
                or b.dtype != np.float64 or b.shape != (n, 4) or not np.isfinite(b).all()
                or np.any(b[:, 2:] <= b[:, :2]) or np.ma.isMaskedArray(self.group_of)
                or g.dtype != np.bool_ or g.shape != (n,)):
            raise ValueError("Finite positive-area FP64 reference boxes and boolean group flags required")
        object.__setattr__(self, "boxes_xyxy", _owned(b))
        object.__setattr__(self, "group_of", _owned(g))


@dataclass(frozen=True, eq=False)
class ObjectRankingEvaluation:
    original_frame_index: int
    image_size: tuple[int, int]
    person_ids: tuple[str, ...]
    object_ids: tuple[str, ...]
    hand_person_indices: np.ndarray          # [native slots], -1 = unknown/not hand
    object_reference_indices: np.ndarray     # [native slots], geometry-only, -1 unknown
    person_object_reference_indices: np.ndarray  # [persons,role1 slots], vocabulary-censored
    score_banks: tuple                       # unchanged complete native object slots per person
    positive_pairs: np.ndarray              # [persons,reference objects], ALL manual positives
    reliable_negative_pairs: np.ndarray
    positive_person: np.ndarray
    reference_unscorable: np.ndarray
    eligible_hand_count: np.ndarray
    positive_count: np.ndarray
    negative_count: np.ndarray
    matched_positive_count: np.ndarray
    matched_negative_count: np.ndarray
    comparison_count: np.ndarray
    known_candidate_count: np.ndarray        # aliases aggregated once per reference instance
    unknown_candidate_count: np.ndarray      # every unmapped native slot retained
    concordance_a: np.ndarray                # [persons], NaN if no comparisons
    concordance_b: np.ndarray
    retrieval_a: np.ndarray                  # [persons], no support = miss zero
    retrieval_b: np.ndarray
    informative_person_count: int
    positive_person_count: int
    image_concordance_a: float | None
    image_concordance_b: float | None
    image_retrieval_a: float | None
    image_retrieval_b: float | None


def _pixels(regions, grid, coordinates):
    b = regions.boxes_xyxy
    if coordinates == "normalized":
        if np.any((b < 0) | (b > 1)):
            raise ValueError("Normalized reference boxes must be inside [0,1], never clipped")
        h, w = grid; b = b * np.array([w, h, w, h], np.float64)
    elif coordinates != "pixels":
        raise ValueError("Explicit pixels or normalized reference coordinates required")
    with np.errstate(over="ignore", invalid="ignore"):
        area = np.prod(b[:, 2:] - b[:, :2], axis=1)
    if not np.isfinite(b).all() or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Finite represented reference pixel areas required")
    return b, area


def _best_a(scores):
    return max(tuple(map(float, row)) for row in scores)


def _retrieval(scores, references, positive):
    if not scores:
        return 0.
    best = max(scores)
    winning = [ref for score, ref in zip(scores, references) if score == best]
    return sum(ref >= 0 and positive[ref] for ref in winning) / len(winning)


def evaluate_hoi_object_ranking(observations: HOIDetrObservations, persons: ReferenceRegions,
                                objects: ReferenceRegions, holds, holds_vocabulary,
                                *, reference_coordinates="pixels"):
    """Half-tie concordance and full-bank retrieval; no acceptance decision.

    holds is a unique int64[relations,2] vector of (person index,object index).
    The exact tuple-of-class-pairs vocabulary supports ONLY absent-pair negatives.
    Hand centres require unique closed-box containment without group overlap;
    object boxes require exactly one IoU>=.5 match, also without group overlap.
    Off-vocabulary/group/ambiguous candidates stay unknown retrieval competitors.
    Known reference aliases count once (best A tuple/max B); unknown native slots
    remain separate competitors, NOT physical instances or proven negatives.
    """
    if type(observations) is not HOIDetrObservations or any(type(x) is not ReferenceRegions for x in (persons, objects)):
        raise ValueError("Native observations and explicit external ReferenceRegions required")
    score_hoi_objects(observations)  # Validate native full bank BEFORE reference association.
    p, q = len(persons.instance_ids), len(objects.instance_ids)
    links = np.asarray(holds)
    if (np.ma.isMaskedArray(holds) or links.dtype != np.int64 or links.ndim != 2 or links.shape[1:] != (2,)
            or np.any(links < 0) or np.any(links[:, 0] >= p) or np.any(links[:, 1] >= q)
            or len(np.unique(links, axis=0)) != len(links)):
        raise ValueError("Unique original person/object int64 relation indices required")
    if type(holds_vocabulary) is not tuple:
        raise ValueError("Exact holds vocabulary class-pair tuple required")
    for pair in holds_vocabulary:
        _keys(pair, "holds vocabulary pair")
        if len(pair) != 2:
            raise ValueError("Exact person/object class pairs required")
    if len(set(holds_vocabulary)) != len(holds_vocabulary):
        raise ValueError("Duplicate holds vocabulary pair")
    vocabulary = set(holds_vocabulary)
    pb, _ = _pixels(persons, observations.image_size, reference_coordinates)
    ob, oa = _pixels(objects, observations.image_size, reference_coordinates)
    roles = observations.class_ids; n = len(roles)
    hands, slots = np.flatnonzero(roles == 0), np.flatnonzero(roles == 1)
    boxes = observations.boxes_original_xyxy.astype(np.float64)
    hp, om = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for slot in hands:
        centre = (boxes[slot, :2] + boxes[slot, 2:]) / 2
        matched = np.flatnonzero(np.all((centre >= pb[:, :2]) & (centre <= pb[:, 2:]), axis=1))
        if len(matched) == 1 and not persons.group_of[matched[0]]:
            hp[slot] = matched[0]
    for slot in slots:
        width = np.maximum(0., np.minimum(boxes[slot, 2:], ob[:, 2:]) - np.maximum(boxes[slot, :2], ob[:, :2]))
        intersection = width[:, 0] * width[:, 1]
        area = np.prod(boxes[slot, 2:] - boxes[slot, :2])
        union = area + oa - intersection
        iou = np.divide(intersection, union, out=np.zeros(q, np.float64), where=union > 0)
        matched = np.flatnonzero(iou >= .5)
        if len(matched) == 1 and not objects.group_of[matched[0]]:
            om[slot] = matched[0]
    positive = np.zeros((p, q), bool); positive[links[:, 0], links[:, 1]] = True
    scope = np.array([[(a, b) in vocabulary for b in objects.class_ids] for a in persons.class_ids], bool).reshape(p, q)
    negative = scope & ~positive & ~persons.group_of[:, None] & ~objects.group_of[None, :]
    mappings = np.full((p, len(slots)), -1, np.int64); banks = []
    counts = np.zeros((7, p), np.int64)  # hand, matched pos/neg, comparisons, known/unknown, unscorable
    ca, cb = np.full(p, np.nan), np.full(p, np.nan)
    ra, rb = np.zeros(p), np.zeros(p)
    for person in range(p):
        eligible = hands[hp[hands] == person]
        bank = score_hoi_objects(observations, eligible_hand_slots=eligible); banks.append(bank)
        counts[0, person] = len(eligible)
        counts[6, person] = bool(persons.group_of[person] or np.any(positive[person] & (objects.group_of | ~scope[person])))
        for j, slot in enumerate(slots):
            ref = om[slot]
            if ref >= 0 and scope[person, ref] and not persons.group_of[person]:
                mappings[person, j] = ref
        known = sorted(set(int(x) for x in mappings[person] if x >= 0))
        unknown = np.flatnonzero(mappings[person] < 0)
        counts[4:6, person] = len(known), len(unknown)
        if not len(eligible):
            continue
        sa, sb, refs = [], [], []
        for ref in known:
            aliases = np.flatnonzero(mappings[person] == ref)
            sa.append(_best_a(bank.scores_a[aliases])); sb.append(float(bank.scores_b[aliases].max())); refs.append(ref)
        for j in unknown:
            sa.append(tuple(map(float, bank.scores_a[j]))); sb.append(float(bank.scores_b[j])); refs.append(-1)
        ra[person] = _retrieval(sa, refs, positive[person]); rb[person] = _retrieval(sb, refs, positive[person])
        pi = [i for i, ref in enumerate(refs) if ref >= 0 and positive[person, ref]]
        ni = [i for i, ref in enumerate(refs) if ref >= 0 and negative[person, ref]]
        counts[1:4, person] = len(pi), len(ni), len(pi) * len(ni)
        if pi and ni:
            ca[person] = sum((sa[i] > sa[j]) + .5 * (sa[i] == sa[j]) for i in pi for j in ni) / (len(pi) * len(ni))
            cb[person] = sum((sb[i] > sb[j]) + .5 * (sb[i] == sb[j]) for i in pi for j in ni) / (len(pi) * len(ni))
    pp = positive.any(axis=1); informative = pp & (counts[3] > 0)
    mean = lambda a, mask: float(a[mask].mean()) if mask.any() else None
    arrays = (hp, om, mappings, positive, negative, pp, counts[6].astype(bool), counts[0],
              positive.sum(axis=1, dtype=np.int64), negative.sum(axis=1, dtype=np.int64),
              counts[1], counts[2], counts[3], counts[4], counts[5], ca, cb, ra, rb)
    frozen = tuple(map(_owned, arrays))
    return ObjectRankingEvaluation(observations.original_frame_index, observations.image_size,
        persons.instance_ids, objects.instance_ids, *frozen[:3], tuple(banks), *frozen[3:],
        int(informative.sum()), int(pp.sum()), mean(ca, informative), mean(cb, informative), mean(ra, pp), mean(rb, pp))


@dataclass(frozen=True)
class ExactImageTest:
    test: str
    image_count: int
    informative_count: int
    positive_count: int
    negative_count: int
    zero_count: int
    mean_delta: float
    p_one_sided: float | None
    sufficient_images: bool


def _deltas(deltas):
    a = np.asarray(deltas)
    if np.ma.isMaskedArray(deltas) or a.dtype != np.float64 or a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Complete finite nonempty FP64 image-paired delta vector required")
    return a


def _dyadic_values(a):
    ratios = [float(x).as_integer_ratio() for x in a]
    den = max(d for _, d in ratios)
    return [v * (den // d) for v, d in ratios], den


def exact_image_signflip(deltas):
    """Original exact sign-flip test, all zeros retained, at most 20 images.

    Integer dyadic sums avoid accidental float ties; no resampling. Fewer than
    six images returns unscorable p=None, not an acceptance/rejection decision.
    """
    a = _deltas(deltas); n = len(a)
    if n > 20:
        raise ValueError("Exact exhaustive sign-flip resource limit is 20 images")
    values, den = _dyadic_values(a); target = sum(values)
    sums = [0]
    for v in values:
        sums = [s + v for s in sums] + [s - v for s in sums]
    p = sum(s >= target for s in sums) / len(sums) if n >= 6 else None
    return ExactImageTest("exact_signflip", n, n, int((a > 0).sum()), int((a < 0).sum()),
                          int((a == 0).sum()), target / (den * n), p, n >= 6)


def exact_image_sign_test(deltas):
    """Distinct one-sided paired binomial sign test; zero ties excluded.

    Eligibility requires at least six informative images supplied by the caller,
    including exact-zero deltas. Ties abstain only from the directional binomial
    vote; they remain in the image mean/count. No substitution for sign-flip.
    """
    a = _deltas(deltas); wins, losses = int((a > 0).sum()), int((a < 0).sum())
    n = wins + losses
    p = sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n) if len(a) >= 6 else None
    values, den = _dyadic_values(a)
    return ExactImageTest("exact_binomial_sign", len(a), len(a), wins, losses, int((a == 0).sum()), sum(values) / (den * len(a)), p, len(a) >= 6)
