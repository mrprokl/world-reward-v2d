"""Same-bank mask/point association evidence, not actor, target or contact labels.

Automatic proposals are observations, never certified physical identities. Query
ownership is a caller-supplied persistent seed hypothesis. Native full-T point
visibility is kept, including backward tracking before a query's seed time.
No models, learned weights, acceptance thresholds, temporal DP or I/O live here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment


def _owned(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _key(value, name):
    if (type(value) is not str or not value.strip() or len(value) > 256
            or any(ord(char) < 32 for char in value)):
        raise ValueError(f"{name}: bounded nonempty string required")
    return value


def _keys(values, name):
    if type(values) is not tuple:
        raise ValueError(f"{name}: tuple of unique keys required")
    for value in values:
        _key(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f"{name}: duplicate key")
    return values


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name}: nonnegative original frame index required")
    return int(value)


def _array(value, name, shape, dtypes):
    array = np.asarray(value)
    if (np.ma.isMaskedArray(value) or array.shape != shape
            or array.dtype not in tuple(np.dtype(dtype) for dtype in dtypes)):
        raise ValueError(f"{name}: original declared shape/dtype required")
    return array


@dataclass(frozen=True, eq=False)
class MaskAnchor:
    """All original-grid binary masks at one fixed original frame.

    Keys/masks sort together for detection-order invariance. Empty masks and
    zero candidates are retained; masks must still declare the positive H,W
    image grid. Duplicate keys are errors; overlapping masks remain independent
    proposals, not disjoint labels. Stored arrays own immutable bytes.
    """
    frame_index: int
    mask_ids: tuple[str, ...]
    masks: np.ndarray

    def __post_init__(self):
        index = _index(self.frame_index, "anchor frame")
        keys = _keys(self.mask_ids, "mask_ids")
        masks = np.asarray(self.masks)
        if (np.ma.isMaskedArray(self.masks) or masks.dtype != np.bool_
                or masks.ndim != 3 or masks.shape[0] != len(keys)
                or min(masks.shape[1:]) < 1):
            raise ValueError("Original boolean masks[candidates,H,W] required")
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        object.__setattr__(self, "frame_index", index)
        object.__setattr__(self, "mask_ids", tuple(keys[i] for i in order))
        object.__setattr__(self, "masks", _owned(masks[order]))


@dataclass(frozen=True)
class HypothesisSeed:
    """Persistent proposal key bound to one automatic anchor-mask seed.

    The seed time initializes its point queries; it is NOT a physical appearance
    time. This small primitive does not merge later proposals of the same object.
    """
    hypothesis_id: str
    frame_index: int
    mask_id: str

    def __post_init__(self):
        _key(self.hypothesis_id, "hypothesis_id")
        _key(self.mask_id, "seed mask_id")
        object.__setattr__(self, "frame_index", _index(self.frame_index, "seed frame"))


@dataclass(frozen=True, eq=False)
class PointMaskAnchorEvidence:
    """All hypotheses x candidates, with finite fractions and explicit support.

    Counts use original queries, without deduplication, refill or reassignment.
    native_visible_count includes visible off-image points in the denominator.
    Off-image points cannot be tested for mask membership and are reported as
    unknown; they are NOT silently dropped to improve a score. Entirely hidden,
    off-image or zero-query hypotheses have no supported candidate score.
    Empty candidate masks are retained with unsupported zero scores.

    fraction_inside = inside_count / native_visible_count, zero if no visible
    queries. This is point retention, NOT mask precision, likelihood or calibrated
    probability: sparse/duplicated queries cannot measure an entire silhouette.
    """
    frame_index: int
    hypothesis_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    query_count: np.ndarray
    native_visible_count: np.ndarray
    on_image_count: np.ndarray
    off_image_count: np.ndarray
    hidden_count: np.ndarray
    inside_count: np.ndarray
    mask_area: np.ndarray
    fraction_inside: np.ndarray
    supported: np.ndarray


@dataclass(frozen=True, eq=False)
class PointMaskEvidence:
    frame_index: np.ndarray
    seeds: tuple[HypothesisSeed, ...]
    query_ownership: tuple[str, ...]
    query_birth_frame_indices: np.ndarray
    query_points_xy: np.ndarray
    anchors: tuple[PointMaskAnchorEvidence, ...]


def _seed_bank(frame_index, anchors, seeds):
    indices = np.asarray(frame_index)
    if (np.ma.isMaskedArray(frame_index) or indices.dtype != np.int64
            or indices.ndim != 1 or not len(indices)
            or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64))):
        raise ValueError("Complete original int64 arange(T) required")
    if (type(anchors) is not tuple or not anchors
            or any(type(anchor) is not MaskAnchor for anchor in anchors)
            or any(a.frame_index >= b.frame_index for a, b in zip(anchors, anchors[1:]))
            or any(anchor.frame_index >= len(indices) for anchor in anchors)):
        raise ValueError("Strictly increasing original MaskAnchor tuple required")
    height, width = anchors[0].masks.shape[1:]
    if any(anchor.masks.shape[1:] != (height, width) for anchor in anchors):
        raise ValueError("All masks must share the same original image grid")
    if type(seeds) is not tuple or any(type(seed) is not HypothesisSeed for seed in seeds):
        raise ValueError("Tuple of HypothesisSeed required, including zero-query hypotheses")
    _keys(tuple(seed.hypothesis_id for seed in seeds), "hypothesis_ids")
    if len({(seed.frame_index, seed.mask_id) for seed in seeds}) != len(seeds):
        raise ValueError("Duplicate automatic seed source under different hypothesis keys")
    seeds = tuple(sorted(seeds, key=lambda seed: seed.hypothesis_id))
    by_frame = {anchor.frame_index: anchor for anchor in anchors}
    for seed in seeds:
        if seed.frame_index not in by_frame or seed.mask_id not in by_frame[seed.frame_index].mask_ids:
            raise ValueError("Every hypothesis must bind an actual automatic anchor mask")
    return indices, seeds, by_frame


def point_mask_evidence(frame_index, anchors, seeds, tracks, visibility,
                        query_ownership, query_birth_frame_indices, query_points_xy):
    """Count native visible full-T correspondences inside every automatic mask.

    frame_index is exact int64 arange(T). anchors is a strictly increasing tuple
    of MaskAnchor on that timeline; all share the SAME original image grid.
    tracks are original float32/64[Q,T,2] XY, visibility original bool[Q,T].
    Visible coordinates must be finite; hidden coordinates may be nonfinite and
    never enter counts. Query births are int64[Q] and ownership tuple[Q] refers
    to supplied HypothesisSeed keys. No time/visibility/coordinate is modified.

    Explicit original query_points_xy[Q,2], not predicted tracks at birth, proves
    every query was seeded INSIDE its bound automatic mask. Pixel membership is
    floor(XY), with no offset correction, resizing or clipping. Backward native
    observations before the seed frame are legitimate and remain supported.
    Multiple masks may own separate queries at the same pixel; no precision or
    independent-sample claims are made. Zero-query hypotheses remain in the bank.
    """
    indices, seeds, by_frame = _seed_bank(frame_index, anchors, seeds)
    height, width = anchors[0].masks.shape[1:]
    by_hypothesis = {seed.hypothesis_id: seed for seed in seeds}
    if type(query_ownership) is not tuple:
        raise ValueError("Original per-query ownership tuple required")
    for key in query_ownership:
        if _key(key, "query owner") not in by_hypothesis:
            raise ValueError("Query ownership must refer to a supplied seed hypothesis")
    count = len(query_ownership)
    points = _array(tracks, "tracks", (count, len(indices), 2), ("float32", "float64"))
    visible = _array(visibility, "visibility", (count, len(indices)), ("bool",))
    birth = _array(query_birth_frame_indices, "query births", (count,), ("int64",))
    initial = _array(query_points_xy, "query_points_xy", (count, 2), ("float32", "float64"))
    if (np.any(birth < 0) or np.any(birth >= len(indices))
            or not np.isfinite(points[visible]).all() or not np.isfinite(initial).all()):
        raise ValueError("Finite native visible tracks/queries and original query births required")
    if np.any((initial[:, 0] < 0) | (initial[:, 0] >= width)
              | (initial[:, 1] < 0) | (initial[:, 1] >= height)):
        raise ValueError("Original queries must be on the seed image grid, without clipping")
    initial_pixel = np.floor(initial).astype(np.int64)
    for q, key in enumerate(query_ownership):
        seed = by_hypothesis[key]
        if birth[q] != seed.frame_index:
            raise ValueError("Query birth must match its source-bound automatic seed frame")
        anchor = by_frame[seed.frame_index]
        mask = anchor.masks[anchor.mask_ids.index(seed.mask_id)]
        if not mask[initial_pixel[q, 1], initial_pixel[q, 0]]:
            raise ValueError("Original query lies outside its automatic ownership seed mask")
    owner_indices = tuple(np.array([q for q, key in enumerate(query_ownership)
                                    if key == seed.hypothesis_id], dtype=np.int64) for seed in seeds)
    totals = np.array([len(query) for query in owner_indices], dtype=np.int64)
    results = []
    for anchor in anchors:
        t = anchor.frame_index
        native_count = np.zeros(len(seeds), np.int64)
        on_count = np.zeros(len(seeds), np.int64)
        inside = np.zeros((len(seeds), len(anchor.mask_ids)), np.int64)
        for h, queries in enumerate(owner_indices):
            xy = points[queries[visible[queries, t]], t]
            native_count[h] = len(xy)
            on = ((xy[:, 0] >= 0) & (xy[:, 0] < width)
                  & (xy[:, 1] >= 0) & (xy[:, 1] < height))
            pixel = np.floor(xy[on]).astype(np.int64)
            on_count[h] = len(pixel)
            inside[h] = anchor.masks[:, pixel[:, 1], pixel[:, 0]].sum(axis=1, dtype=np.int64)
        area = anchor.masks.sum(axis=(1, 2), dtype=np.int64)
        fractions = np.divide(inside, native_count[:, None],
                              out=np.zeros(inside.shape, np.float64), where=native_count[:, None] > 0)
        support = (on_count[:, None] > 0) & (area[None, :] > 0)
        arrays = (totals, native_count, on_count, native_count - on_count,
                  totals - native_count, inside, area, fractions, support)
        results.append(PointMaskAnchorEvidence(t, tuple(seed.hypothesis_id for seed in seeds),
                                               anchor.mask_ids, *map(_owned, arrays)))
    return PointMaskEvidence(_owned(indices), seeds, query_ownership, _owned(birth),
                             _owned(initial), tuple(results))


@dataclass(frozen=True, eq=False)
class MaskIoUEvidence:
    reference_frame_index: int
    frame_index: int
    hypothesis_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    intersection: np.ndarray
    union: np.ndarray
    fraction_iou: np.ndarray
    supported: np.ndarray


def _mask_iou_counts(reference_masks, candidate_masks):
    counts = np.zeros((len(reference_masks), len(candidate_masks)), np.int64)
    for i, mask in enumerate(reference_masks):
        counts[i] = (candidate_masks & mask).sum(axis=(1, 2), dtype=np.int64)
    a = reference_masks.sum(axis=(1, 2), dtype=np.int64)
    b = candidate_masks.sum(axis=(1, 2), dtype=np.int64)
    union = a[:, None] + b[None, :] - counts
    scores = np.divide(counts, union, out=np.zeros(counts.shape, np.float64), where=union > 0)
    support = (a[:, None] > 0) & (b[None, :] > 0)
    return tuple(map(_owned, (counts, union, scores, support)))


def mask_iou_evidence(reference: MaskAnchor, candidate: MaskAnchor):
    """A-arm binary IoU only, with caller-supplied reference masks/row keys.

    Both banks stay complete; empty masks have unsupported zero scores, including
    empty/empty (not perfect IoU). This does not choose or update a previous
    identity, supply masks at missing frames, or greedily propagate a path.
    """
    if (type(reference) is not MaskAnchor or type(candidate) is not MaskAnchor
            or reference.masks.shape[1:] != candidate.masks.shape[1:]):
        raise ValueError("Two original-grid MaskAnchor banks required")
    return MaskIoUEvidence(reference.frame_index, candidate.frame_index, reference.mask_ids,
                           candidate.mask_ids, *_mask_iou_counts(reference.masks, candidate.masks))


@dataclass(frozen=True, eq=False)
class SeedMaskIoUAnchorEvidence:
    frame_index: int
    hypothesis_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    reference_frame_indices: np.ndarray
    intersection: np.ndarray
    union: np.ndarray
    fraction_iou: np.ndarray
    supported: np.ndarray


def seed_mask_iou_evidence(frame_index, anchors, seeds):
    """A-arm initial-mask IoU on EXACTLY B's bank, including mixed seed times.

    Each hypothesis retains its source-bound seed mask as reference; later/earlier
    anchor comparison is offline and does not infer appearance times. This small
    baseline is seed-mask IoU, NOT an implemented previous-anchor temporal tracker.
    Row references and full bank are explicit; no point evidence, greedy updates,
    target selection or arbitrary tied reference enter the A arm.
    """
    _, seeds, by_frame = _seed_bank(frame_index, anchors, seeds)
    reference = np.empty((len(seeds), *anchors[0].masks.shape[1:]), bool)
    for h, seed in enumerate(seeds):
        source = by_frame[seed.frame_index]
        reference[h] = source.masks[source.mask_ids.index(seed.mask_id)]
    birth = _owned(np.array([seed.frame_index for seed in seeds], np.int64))
    keys = tuple(seed.hypothesis_id for seed in seeds)
    return tuple(SeedMaskIoUAnchorEvidence(anchor.frame_index, keys, anchor.mask_ids, birth,
                  *_mask_iou_counts(reference, anchor.masks)) for anchor in anchors)


@dataclass(frozen=True, eq=False)
class PartialAssignment:
    """Native global partial matching plus EVERY competitive edge/null route.

    optimal_routes gives edges that occur in at least one globally optimal
    assignment within explicit caller tie_tolerance. They are alternatives, NOT
    simultaneously compatible matches. Forced-route regrets are raw score
    differences, not probabilities/confidence; unsupported edges have +infinity.
    Dummy-row permutations are ignored. An ambiguous row has no chosen candidate;
    unique matched/unmatched rows are mathematical decisions, not semantic truth.
    """
    hypothesis_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    scores: np.ndarray
    supported: np.ndarray
    unmatched_score: float
    tie_tolerance: float
    optimum_score: float
    edge_regrets: np.ndarray
    unmatched_regrets: np.ndarray
    candidate_unmatched_regrets: np.ndarray
    optimal_routes: np.ndarray
    optimal_unmatched: np.ndarray
    optimal_candidate_unmatched: np.ndarray
    row_status: tuple[str, ...]
    unique_candidate_ids: tuple[str | None, ...]
    unique: bool
    solver_calls: int


def _unit(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not 0 <= value <= 1):
        raise ValueError(f"{name}: explicit finite real [0,1] required")
    return float(value)


def partial_mask_assignment(hypothesis_ids, candidate_ids, scores, supported,
                            *, unmatched_score, tie_tolerance):
    """Maximize supplied mask scores using scipy's Hungarian solver, no DP.

    A/B must supply the SAME explicit per-hypothesis unmatched_score and numeric
    tie_tolerance; no defaults/calibration/learned weights are invented here.
    Each unmatched hypothesis earns unmatched_score; unused candidates earn 0.
    Thus a single match competes against one null, not twice the null threshold.
    Unsupported edges are forbidden but all hypotheses/candidates remain present.

    Canonically sort keys and arrays for order invariance. Force each supported
    edge and each hypothesis/candidate null in turn to expose all globally optimal
    routes, including equality with null; no arbitrary tied identity is returned.
    Cost is 1+supported_edges+H+C native O((H+C)^3) solves. This is a bounded-bank
    mathematical primitive, not tracking, temporal recovery or empirical gain.
    """
    rows, cols = _keys(hypothesis_ids, "hypothesis_ids"), _keys(candidate_ids, "candidate_ids")
    n, m = len(rows), len(cols)
    values = _array(scores, "scores", (n, m), ("float32", "float64"))
    valid = _array(supported, "supported", (n, m), ("bool",))
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Finite raw mask scores in [0,1] required, including unsupported slots")
    null = _unit(unmatched_score, "unmatched_score")
    if (isinstance(tie_tolerance, (bool, np.bool_)) or not isinstance(tie_tolerance, Real)
            or not math.isfinite(float(tie_tolerance)) or tie_tolerance < 0):
        raise ValueError("Explicit finite nonnegative numeric tie_tolerance required")
    tolerance = float(tie_tolerance)
    ro = np.array(sorted(range(n), key=lambda i: rows[i]), np.int64)
    co = np.array(sorted(range(m), key=lambda j: cols[j]), np.int64)
    rows, cols = tuple(rows[i] for i in ro), tuple(cols[j] for j in co)
    values = np.array(values[np.ix_(ro, co)], dtype=np.float64, copy=True)
    valid = np.array(valid[np.ix_(ro, co)], copy=True)
    weights = np.full((n + m, n + m), -np.inf)
    weights[:n, :m] = np.where(valid, values, -np.inf)
    weights[np.arange(n), m + np.arange(n)] = null
    weights[n + np.arange(m), np.arange(m)] = 0.
    weights[n:, m:] = 0.
    calls = 0
    def maximum(matrix):
        nonlocal calls
        calls += 1
        if not len(matrix):
            return 0.
        rr, cc = linear_sum_assignment(matrix, maximize=True)
        return float(math.fsum(matrix[rr, cc]))
    optimum = maximum(weights)
    edge_best = np.full((n, m), -np.inf)
    for i, j in np.argwhere(valid):
        forced = weights.copy()
        forced[i] = -np.inf
        forced[i, j] = values[i, j]
        edge_best[i, j] = maximum(forced)
    null_best = np.zeros(n, np.float64)
    for i in range(n):
        forced = weights.copy()
        forced[i] = -np.inf
        forced[i, m + i] = null
        null_best[i] = maximum(forced)
    candidate_null_best = np.zeros(m, np.float64)
    for j in range(m):
        forced = weights.copy()
        forced[:n, j] = -np.inf
        candidate_null_best[j] = maximum(forced)
    regrets, null_regrets, candidate_regrets = (optimum - edge_best, optimum - null_best,
                                                optimum - candidate_null_best)
    routes = valid & (np.abs(regrets) <= tolerance)
    null_routes = np.abs(null_regrets) <= tolerance
    candidate_null_routes = np.abs(candidate_regrets) <= tolerance
    statuses, choices = [], []
    for i in range(n):
        count = int(routes[i].sum()) + int(null_routes[i])
        if not count:
            raise ValueError("Numeric comparison found no optimum route; inspect solver/tie_tolerance")
        statuses.append("AMBIGUOUS" if count > 1 else "UNIQUE_UNMATCHED" if null_routes[i] else "UNIQUE_MATCH")
        choices.append(cols[int(np.flatnonzero(routes[i])[0])] if count == 1 and not null_routes[i] else None)
    arrays = (values, valid, regrets, null_regrets, candidate_regrets,
              routes, null_routes, candidate_null_routes)
    vv, ss, er, nr, cr, oo, on, oc = map(_owned, arrays)
    return PartialAssignment(rows, cols, vv, ss, null, tolerance, optimum, er, nr, cr,
                             oo, on, oc, tuple(statuses), tuple(choices),
                             all(status != "AMBIGUOUS" for status in statuses), calls)
