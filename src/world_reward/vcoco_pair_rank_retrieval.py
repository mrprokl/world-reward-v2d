"""Positive-only role-pair masks and expected tie-neutral Recall@1/3/5.

No predictor, labels-as-negatives, contact/owner truth or source authentication.
The unchanged evaluator validates complete native banks and all same-image
reference context. Exact-coordinate aliases receive one proposal rank slot;
different geometries matching one native positive pair receive one hit total.
"""
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType

import numpy as np

from . import vcoco_pair_retrieval as original

BUDGETS = (1, 3, 5)


def _snapshot(arrays):
    return tuple((a.shape, a.dtype.str, hashlib.sha256(a.tobytes()).digest()) for a in arrays)


def _hit_probability(n, m, draws):
    """1-C(n-m,b)/C(n,b), using at most five stable bounded factors."""
    original._require(type(n) is type(m) is type(draws) is int
                      and 0 <= m <= n and 0 <= draws <= min(n, 5), 'Valid cutoff tie counts required')
    if not m or not draws:
        return 0.
    if n-m < draws:
        return 1.
    missing, hit = 1., 0.
    for j in range(draws):
        chance = m/(n-j)
        hit += missing*chance
        missing *= 1.-chance
    return min(1., max(0., hit))


@dataclass(frozen=True, eq=False)
class VcocoPairRankRetrieval:
    status: str
    validation: original.VcocoPairRetrieval
    budgets: tuple
    native_positive_pairs: tuple
    positive_mask: np.ndarray          # ALL canonical P/O groups, even unsupported positives
    scores: np.ndarray                 # ALL groups; unsupported remains NaN
    supported: np.ndarray
    person_boxes: np.ndarray
    object_boxes: np.ndarray
    person_to_group: np.ndarray
    object_to_group: np.ndarray
    person_members: tuple
    object_members: tuple
    positive_hit_probabilities: np.ndarray  # budgets x unique localized native positives
    expected_positive_hits: tuple      # None if no localized reference pair
    recalls: tuple                     # Recall@1/3/5; None if no localized pair
    any_positive_at1: float | None     # EXACT original top-tie positive fraction
    scope: object


def evaluate_vcoco_pair_rank_retrieval(person_boxes, object_boxes, scores, support,
                                     reference, instances, image_id, image_size,
                                     person_ids, object_ids):
    """Keep every input slot; return a positive-only mask and expected recalls.

    Groups use lexicographic exact-box order, matching PairRouteBank/np.unique.
    At the cutoff tie, select uniformly without replacement; no ID ordering is
    invented. Higher-score groups are all selected. A native GT pair already
    retrieved above the cutoff counts once; unknown proposals still occupy rank
    slots. Missing detections/unsupported positives remain in the denominator.
    This expectation is not a particular deterministic ranked prediction.
    """
    validated = original.evaluate_vcoco_pair_retrieval(
        person_boxes, object_boxes, scores, support, reference, instances,
        image_id, image_size, person_ids, object_ids)
    borrowed = (person_boxes, object_boxes, scores, support)
    before = _snapshot(borrowed)
    reference_before = original._source_fingerprint(instances, reference)
    p, pi, pg = np.unique(person_boxes, axis=0, return_index=True, return_inverse=True)
    o, oi, og = np.unique(object_boxes, axis=0, return_index=True, return_inverse=True)
    group_scores = np.array(scores[np.ix_(pi, oi)], dtype=np.float64)
    group_support = np.array(support[np.ix_(pi, oi)], dtype=bool)
    pm = validated.person_reference_annotation_ids[pi]
    om = validated.object_reference_annotation_ids[oi]
    positives = tuple(sorted((r.agent_annotation_id, r.object_annotation_id)
                             for r in reference.localized_positive_pairs if r.image_id == image_id))
    original._require(len(positives) == validated.localized_positive_pair_count,
                      'Exact validated positive pair population required')
    labels = np.full(group_scores.shape, -1, np.int64)
    for index, (person, obj) in enumerate(positives):
        matched = (pm == person)[:, None] & (om == obj)[None, :]
        original._require(not np.any(labels[matched] >= 0), 'A proposal maps to at most one native pair')
        labels[matched] = index
    positive_mask = labels >= 0
    values, native_pairs = group_scores[group_support], labels[group_support]
    probabilities = np.zeros((len(BUDGETS), len(positives)), np.float64)
    for row, budget in enumerate(BUDGETS):
        selected = min(budget, len(values))
        if not selected:
            continue
        cutoff = np.partition(values, len(values)-selected)[len(values)-selected]
        above, tied = values > cutoff, values == cutoff
        known_above = native_pairs[above]
        probabilities[row, known_above[known_above >= 0]] = 1.
        counts = np.bincount(native_pairs[tied & (native_pairs >= 0)], minlength=len(positives))
        tie_size, draws = int(tied.sum()), selected-int(above.sum())
        for index in np.flatnonzero((counts > 0) & (probabilities[row] == 0.)):
            probabilities[row, index] = _hit_probability(tie_size, int(counts[index]), draws)
    if len(values):
        highest = values == values.max()
        original._require(int(highest.sum()) == validated.top_unique_pair_count
                          and int((native_pairs[highest] >= 0).sum()) == validated.top_known_positive_count,
                          'Unchanged original top-tie association required')
    expected = tuple(math.fsum(row) for row in probabilities) if positives else (None,)*len(BUDGETS)
    recalls = tuple(v/len(positives) for v in expected) if positives else (None,)*len(BUDGETS)
    original._require(before == _snapshot(borrowed)
                      and reference_before == original._source_fingerprint(instances, reference),
                      'Borrowed full bank/reference mutated during rank evaluation')
    sealed = original._sealed
    return VcocoPairRankRetrieval(
        'scorable_localized_positive' if positives else 'no_localized_positive', validated, BUDGETS, positives,
        sealed(positive_mask), sealed(group_scores), sealed(group_support), sealed(p), sealed(o),
        sealed(pg.astype(np.int64)), sealed(og.astype(np.int64)),
        tuple(sealed(np.flatnonzero(pg == i).astype(np.int64)) for i in range(len(p))),
        tuple(sealed(np.flatnonzero(og == i).astype(np.int64)) for i in range(len(o))),
        sealed(probabilities), expected, recalls, validated.image_positive_retrieval if positives else None,
        MappingProxyType(dict(official_AP=False, continuous_pixel_iou=.5, expected_uniform_cutoff_ties=True,
            rank_view_only=True, unknown_pairs_verified_negative=False, contact_verified=False,
            ownership_verified=False, anatomical_side_verified=False, source_authenticated=False,
            alias_identity='exact_numerical_coordinates_not_physical')))
