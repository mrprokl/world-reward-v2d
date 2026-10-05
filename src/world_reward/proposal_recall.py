"""Visible-instance proposal recall only; never ownership or real-image accuracy.

Fixed IoU thresholds, every native proposal (including duplicates), no selector.
The caller authenticates source/prediction/truth before passing these arrays.
Zero-visible truth remains in the ledger, with undefined IoU and no credit.
"""
from dataclasses import dataclass
from types import MappingProxyType
import math

import numpy as np

THRESHOLDS = (.25, .5, .75)
_POPCOUNT = np.array([i.bit_count() for i in range(256)], np.uint8)


def _require(ok, message):
    if not ok: raise ValueError(message)


def _array(value, dtype, ndim, name):
    _require(type(value) is np.ndarray and value.dtype == dtype and value.ndim == ndim,
             f"Original unmasked {name} dtype/header required")
    return value


def _readonly(value):
    out = np.array(value, copy=True); out.setflags(write=False); return out


def _matching(edges):
    """Exact maximum-cardinality augmenting paths; not max-IoU or a selector."""
    g, n = edges.shape; matched = np.full(g, -1, np.int64); owners = np.full(n, -1, np.int64)
    for start in range(g):
        queue = [start]; seen_g = {start}; seen_n = set(); parent = {}; found = -1
        for row in queue:
            for col in np.flatnonzero(edges[row]):
                col = int(col)
                if col in seen_n: continue
                seen_n.add(col); parent[col] = row
                if owners[col] == -1: found = col; break
                next_row = int(owners[col])
                if next_row not in seen_g: seen_g.add(next_row); queue.append(next_row)
            if found != -1: break
        while found != -1:
            row = parent[found]; previous = int(matched[row])
            matched[row] = found; owners[found] = row; found = previous
    return matched


@dataclass(frozen=True)
class VisibleProposalRecall:
    image_size: tuple[int, int]
    entity_ids: tuple[str, ...]
    entity_kinds: tuple[str, ...]
    visible_areas: np.ndarray
    eligible: np.ndarray
    iou: np.ndarray
    best_iou: np.ndarray
    matching_slots: np.ndarray
    native_areas: np.ndarray
    generate_seconds: float | None
    metrics: object


def evaluate_visible_proposals(packed_masks, image_size, truth_masks, entity_kinds, *,
                               native_areas, entity_ids=None, generate_seconds=None):
    """Packed little-endian native N masks versus private G visible bool masks.

    Native int64 [H,W] header/areas must match actual bits, with zero tail bits.
    Truth masks must be mutually exclusive visible first-surface instances.
    Matching slots are diagnostic, not predictions; raw native indices persist.
    Per-kind capacity is matched independently, so human/object capacities are
    not additive; matching_slots contains the single joint all-kind assignment.
    Zero visible area cannot establish occlusion rather than cropping/off-grid.
    """
    header = _array(image_size, np.dtype('int64'), 1, 'image_size')
    _require(header.shape == (2,) and np.all(header > 0), 'Positive original H/W required')
    h, w = map(int, header); pixels = h*w
    packed = _array(packed_masks, np.dtype('uint8'), 2, 'packed_masks')
    truth = _array(truth_masks, np.dtype('bool'), 3, 'truth_masks')
    areas = _array(native_areas, np.dtype('int64'), 1, 'native_areas')
    n, g = len(packed), len(truth)
    _require(packed.shape == (n, (pixels+7)//8) and truth.shape == (g,h,w)
             and areas.shape == (n,), 'Exact full original-grid bank geometry required')
    if pixels % 8:
        _require(not np.any(packed[:, -1] & np.uint8(255 ^ ((1 << (pixels % 8))-1))),
                 'Nonzero packed padding bits forbidden')
    actual = _POPCOUNT[packed].sum(1, dtype=np.int64)
    _require(np.array_equal(actual, areas), 'Native areas differ from complete actual mask bits')
    _require(type(entity_kinds) is tuple and len(entity_kinds) == g
             and all(k in ('human','object') for k in entity_kinds), 'Exact private instance kinds required')
    if entity_ids is None: entity_ids = tuple(f'entity_{i:06d}' for i in range(g))
    _require(type(entity_ids) is tuple and len(entity_ids) == g and len(set(entity_ids)) == g
             and all(type(s) is str and s for s in entity_ids), 'Distinct stable private ledger IDs required')
    _require(not np.any(truth.sum(0, dtype=np.int64) > 1), 'Visible first-surface truth instances overlap')
    _require(generate_seconds is None or (type(generate_seconds) in (int,float)
             and math.isfinite(generate_seconds) and generate_seconds >= 0), 'Finite native cost required')
    visible = truth.sum((1,2), dtype=np.int64); eligible = visible > 0
    iou = np.full((g,n), np.nan, np.float64); best = np.full(g, np.nan, np.float64)
    for slot in np.flatnonzero(eligible):
        bits = np.packbits(truth[slot].ravel(), bitorder='little')
        overlap = _POPCOUNT[np.bitwise_and(packed, bits)].sum(1, dtype=np.int64)
        iou[slot] = overlap / (areas + visible[slot] - overlap)
        best[slot] = iou[slot].max() if n else 0.
    matching = np.full((3,g), -1, np.int64); recalls = {}
    kinds = np.array(entity_kinds)
    for index, threshold in enumerate(THRESHOLDS):
        edges = (iou >= threshold) & eligible[:,None]; matching[index] = _matching(edges)
        groups = {}
        for kind, chosen in (('all',eligible),('human',eligible & (kinds=='human')),('object',eligible & (kinds=='object'))):
            count = int(chosen.sum()); recovered = int(np.count_nonzero(best[chosen] >= threshold))
            unique = int(np.count_nonzero(_matching(edges[chosen]) >= 0))
            groups[kind] = MappingProxyType(dict(eligible=count, recovered=recovered, one_to_one_recovered=unique,
                recall=recovered/count if count else None, one_to_one_recall=unique/count if count else None))
        recalls[str(threshold)] = MappingProxyType(groups)
    metrics = MappingProxyType(dict(thresholds=THRESHOLDS, native_masks=n, truth_instances=g,
        eligible_instances=int(eligible.sum()), zero_visible_instances=int((~eligible).sum()),
        visible_foreground_pixels=int(visible.sum()), zero_visible_cause='unknown_from_visible_masks',
        kind_matching_scope='independent_within_kind_nonadditive',
        recalls=MappingProxyType(recalls), scope='visible_synthetic_proposal_recall_only',
        ownership_verified=False, accuracy_verified=False, adoption=False))
    return VisibleProposalRecall((h,w),entity_ids,entity_kinds,*map(_readonly,
        (visible,eligible,iou,best,matching,areas)), None if generate_seconds is None else float(generate_seconds),metrics)


def validate_raster_truth(truth_masks, visible_entity_ids, visible_face_indices, face_entity_ids):
    """Optional strict authored-raster consistency; no media/source authentication."""
    truth = _array(truth_masks,np.dtype('bool'),3,'truth_masks'); g,h,w = truth.shape
    labels = _array(visible_entity_ids,np.dtype('int32'),2,'visible_entity_ids')
    faces = _array(visible_face_indices,np.dtype('int64'),2,'visible_face_indices')
    ids = _array(face_entity_ids,np.dtype('int32'),1,'face_entity_ids')
    _require(labels.shape == faces.shape == (h,w) and len(ids)>0 and h>0 and w>0
        and np.all((ids>=0)&(ids<g)) and np.all((faces>=-1)&(faces<len(ids))), 'Original raster inventory invalid')
    expected = np.full((h,w),-1,np.int32); seen = faces>=0; expected[seen] = ids[faces[seen]]
    _require(np.array_equal(labels,expected)
        and all(np.array_equal(truth[i],expected==i) for i in range(g)), 'Truth masks/face labels/geometry disagree')
    return True


def aggregate_visible_recall(results):
    """Image-macro and instance-micro diagnostics; empty categories are unsupported."""
    _require(type(results) in (tuple,list) and len(results)>0
        and all(type(r) is VisibleProposalRecall for r in results), 'Complete evaluated image bank required')
    summary = {}
    for threshold in THRESHOLDS:
        groups = {}
        for kind in ('all','human','object'):
            rows = [r.metrics['recalls'][str(threshold)][kind] for r in results]
            count = sum(x['eligible'] for x in rows); supported = [x for x in rows if x['eligible']]
            groups[kind] = dict(eligible_instances=count, supported_images=len(supported),
                macro_image_recall=sum(x['recall'] for x in supported)/len(supported) if supported else None,
                micro_instance_recall=sum(x['recovered'] for x in rows)/count if count else None,
                macro_image_one_to_one_recall=sum(x['one_to_one_recall'] for x in supported)/len(supported) if supported else None,
                micro_instance_one_to_one_recall=sum(x['one_to_one_recovered'] for x in rows)/count if count else None)
        summary[str(threshold)] = groups
    costs = [r.generate_seconds for r in results if r.generate_seconds is not None]
    return dict(images=len(results), native_masks=sum(len(r.native_areas) for r in results),
        native_masks_per_image=[len(r.native_areas) for r in results],
        zero_visible_instances=sum(int((~r.eligible).sum()) for r in results),
        visible_pixels_per_image=[r.visible_areas.tolist() for r in results],
        cost_images=len(costs), native_generate_seconds=sum(costs) if costs else None,
        recalls=summary, scope='visible_synthetic_proposal_recall_only',
        occlusion_fraction_available=False, ownership_verified=False, accuracy_verified=False, adoption=False)
