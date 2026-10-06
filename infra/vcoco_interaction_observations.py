"""Lossless saved P/pose/HOI/OWL join; no I/O, model, scoring or ownership.

The caller authenticates files, sealed receipts, complete populations and source
provenance. Array fingerprints below bind numerical transport, not those facts.
NumPy and the existing numerical contracts are imported only when invoked.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import re
from types import MappingProxyType

ENDPOINT_FIELDS = (
    'person_raw_boxes', 'person_raw_scores', 'person_raw_labels',
    'person_retained_boxes', 'person_retained_scores', 'person_retained_raw_slots',
    'person_retained_ids', 'person_model_pred_boxes', 'person_model_logits',
    'person_model_input_ids', 'person_model_attention_mask', 'owl_patch_ids',
    'owl_boxes_padded_normalized_cxcywh', 'owl_objectness_logits',
    'owl_boxes_original_xyxy', 'image_size', 'original_frame_index')
POSE_FIELDS = ('person_ids', 'boxes_original_xyxy', 'detector_scores',
    'keypoints_original_xy', 'raw_scores', 'native_valid', 'in_original_image',
    'image_size', 'original_frame_index', 'original_slot', 'acquired_ordinal')
HOI_FIELDS = ('query_logits', 'query_boxes_cxcywh', 'query_tokens',
    'native_nms_detections', 'native_nms_keep', 'retained_nms_positions',
    'query_ids', 'class_ids', 'boxes_original_xyxy', 'raw_scores', 'decayed_scores',
    'hand_object_pairs', 'hand_object_logits', 'object_target_pairs',
    'object_target_logits', 'image_size', 'original_frame_index',
    'original_slot', 'acquired_ordinal')


def _require(value, message):
    if not value:
        raise ValueError(message)


def _identity(a):
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def _snapshot(np, row, arrays, fields):
    _require(type(row) is dict and isinstance(arrays, Mapping)
             and set(arrays) == set(fields) and type(row.get('arrays')) is dict
             and set(row['arrays']) == set(fields), 'Exact complete saved array fields required')
    result = {}
    for name in fields:
        a, pin = arrays[name], row['arrays'][name]
        _require(type(a) is np.ndarray and not a.dtype.hasobject
                 and type(pin) is dict and set(pin) == {'shape', 'dtype', 'sha256'}
                 and type(pin['shape']) is list and all(type(x) is int and x >= 0 for x in pin['shape'])
                 and type(pin['dtype']) is str and type(pin['sha256']) is str
                 and re.fullmatch('[0-9a-f]{64}', pin['sha256']) is not None
                 and _identity(a) == pin, 'Actual saved array identity differs')
        result[name] = np.frombuffer(a.tobytes(order='C'), dtype=a.dtype).reshape(a.shape)
    return MappingProxyType(result)


def _grid(np, row, a, *, slots):
    size = row.get('image_size')
    _require(type(row.get('image_id')) is str and re.fullmatch('[0-9a-f]{32}', row['image_id']) is not None
             and type(size) is list and len(size) == 2 and all(type(x) is int and x > 0 for x in size)
             and type(row.get('original_slot')) is int and 0 <= row['original_slot'] < 16
             and type(row.get('acquired_ordinal')) is int and 0 <= row['acquired_ordinal'] < 16
             and type(row.get('original_frame_index')) is int and row['original_frame_index'] == 0
             and row.get('file') == f"image_{row['original_slot']:06d}.npz", 'Original opaque image/grid/slots required')
    _require(a['image_size'].dtype == np.int64 and a['image_size'].shape == (2,)
             and a['image_size'].tolist() == size, 'Original stored image grid differs')
    for name in ('original_frame_index', 'original_slot', 'acquired_ordinal') if slots else ('original_frame_index',):
        _require(a[name].dtype == np.int64 and a[name].shape == ()
                 and int(a[name]) == row[name], 'Original scalar transport differs')
    return row['image_id'], row['original_slot'], row['acquired_ordinal'], tuple(size)


@dataclass(frozen=True, eq=False)
class InteractionObservations:
    image_id: str
    original_slot: int
    acquired_ordinal: int
    person: object
    hoi: object
    owl: object
    evidence: object
    endpoint_arrays: Mapping
    pose_arrays: Mapping
    hoi_arrays: Mapping


def reconstruct_interaction(endpoint_row, endpoint17, pose_row, pose11, hoi_row, hoi19):
    """Join one genuine saved image, retaining all P*2*3600 and K*3600 routes.

    Missing pose files are not replaced with synthesized observations. Genuine
    P=0 or K=0 is valid; NaN feature support is produced by existing builders.
    No RGB decode, old six-array/episode ABI, alias fusion or parameter is used.
    """
    import numpy as np
    from world_reward.person_pose_observations import PersonPoseObservations
    from world_reward.hoi_detr_observations import HOIDetrObservations
    from world_reward.owlv2_object_observations import Owlv2ObjectObservations
    from world_reward.owlv2_candidate_bridge import bridge_owlv2_candidates
    from world_reward.interaction_candidate_evidence import build_interaction_candidate_evidence

    e = _snapshot(np, endpoint_row, endpoint17, ENDPOINT_FIELDS)
    p = _snapshot(np, pose_row, pose11, POSE_FIELDS)
    h = _snapshot(np, hoi_row, hoi19, HOI_FIELDS)
    grid = _grid(np, endpoint_row, e, slots=False)
    _require(grid == _grid(np, pose_row, p, slots=True) == _grid(np, hoi_row, h, slots=True),
             'All three banks must share exact original image/grid/slots')
    image_id, slot, ordinal, size = grid
    m = endpoint_row.get('person_postprocessor_rows')
    _require(type(m) is int and 0 <= m <= 900 and endpoint_row.get('person_native_queries') == 900
             and endpoint_row.get('owl_patches') == hoi_row.get('owl_patches') == 3600,
             'Complete native 900/3600 census required')
    shapes = {'person_raw_boxes': (m, 4), 'person_raw_scores': (m,),
              'person_model_pred_boxes': (1, 900, 4), 'person_model_logits': (1, 900, 256)}
    _require(all(e[name].shape == shape and e[name].dtype == np.float32 for name, shape in shapes.items())
             and e['person_raw_labels'].shape == (m,) and e['person_raw_labels'].dtype.kind == 'U', 'Native GDI shapes/dtypes required')
    tokens, mask, logits = e['person_model_input_ids'], e['person_model_attention_mask'], e['person_model_logits']
    _require(tokens.dtype == np.int64 and tokens.ndim == 2 and tokens.shape[0] == 1
             and 0 < tokens.shape[1] <= 256 and (tokens >= 0).all() and mask.shape == tokens.shape
             and mask.dtype.kind in 'biu' and np.isin(mask, [0, 1]).all() and mask.any(), 'Native text/token grid required')
    active = np.zeros((1, 256), bool); active[:, :tokens.shape[1]] = mask.astype(bool)
    _require(np.isfinite(logits[:, :, active[0]]).all() and np.isneginf(logits[:, :, ~active[0]]).all()
             and all(np.isfinite(e[name]).all() for name in shapes if name != 'person_model_logits'),
             'Original native finite values and masked negative-infinity logits required')
    ids = e['person_retained_ids']
    n = len(ids)
    _require(ids.ndim == 1 and ids.dtype.kind == 'U'
             and ids.tolist() == endpoint_row.get('person_ids') == pose_row.get('person_ids')
             == hoi_row.get('source_person_ids')
             and ids.tolist() == [f'image:{image_id}/person/retained:{i:06d}' for i in range(n)]
             and type(endpoint_row.get('person_retained_rows')) is int and endpoint_row['person_retained_rows'] == n
             and type(pose_row.get('persons')) is int and pose_row['persons'] == n,
             'Complete original person IDs/order/count required')
    slots = e['person_retained_raw_slots']
    _require(slots.dtype == np.int64 and slots.shape == (n,) and n <= m
             and ((slots >= 0) & (slots < m)).all(), 'Original retained raw slot indices required')
    _require(type(endpoint_row.get('identity')) is dict
             and set(endpoint_row['identity']) == {'bytes', 'sha256'}
             and type(endpoint_row['identity']['bytes']) is int and endpoint_row['identity']['bytes'] > 0
             and re.fullmatch('[0-9a-f]{64}', str(endpoint_row['identity']['sha256'])) is not None
             and hoi_row.get('endpoint_bank_identity') == endpoint_row['identity'], 'Same saved endpoint file reference required')
    for name, source in (('person_ids', 'person_retained_ids'), ('boxes_original_xyxy', 'person_retained_boxes'),
                         ('detector_scores', 'person_retained_scores')):
        _require(p[name].dtype == e[source].dtype and p[name].shape == e[source].shape
                 and p[name].tobytes() == e[source].tobytes(), 'Pose must use unchanged original person census')
    _require(e['person_retained_boxes'].dtype == e['person_retained_scores'].dtype == np.float64
             and p['keypoints_original_xy'].dtype == np.float64 and p['raw_scores'].dtype == np.float32,
             'Actual original person/pose dtypes required')
    person = PersonPoseObservations(0, size, tuple(ids.tolist()), p['boxes_original_xyxy'],
        p['detector_scores'], p['keypoints_original_xy'], p['raw_scores'])
    for name in ('native_valid', 'in_original_image'):
        _require(p[name].dtype == np.bool_ and p[name].shape == (n, 133)
                 and p[name].tobytes() == getattr(person, name).tobytes(), 'Stored native validity/grid flag differs')
    hoi = HOIDetrObservations(0, size, **{name: h[name] for name in HOI_FIELDS[:15]})
    for name, count in (('native_detections', len(hoi.query_ids)), ('hand_object_pairs', len(hoi.hand_object_pairs)),
                         ('object_target_pairs', len(hoi.object_target_pairs))):
        _require(type(hoi_row.get(name)) is int and hoi_row[name] == count, 'Complete HOI receipt counts required')
    owl = Owlv2ObjectObservations(0, size, (60, 60), e['owl_patch_ids'],
        e['owl_boxes_padded_normalized_cxcywh'], e['owl_objectness_logits'], e['owl_boxes_original_xyxy'])
    evidence = build_interaction_candidate_evidence(person, bridge_owlv2_candidates(owl), hoi)
    for row, original, fields in ((endpoint_row, endpoint17, ENDPOINT_FIELDS),
                                  (pose_row, pose11, POSE_FIELDS), (hoi_row, hoi19, HOI_FIELDS)):
        _require(all(_identity(original[name]) == row['arrays'][name] for name in fields), 'Source arrays mutated during join')
    return InteractionObservations(image_id, slot, ordinal, person, hoi, owl, evidence, e, p, h)
