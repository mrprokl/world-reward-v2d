"""Pure V-COCO v3 reference join, not a selector, official AP or rights proof.

Primary s-gupta/v-coco 489cc4db74f2f10ab4b134f67da3874afbf245ab:
vsrl_utils.py 5985B SHA256 5b8ae544d79dde56cdd8ee55c47271eed01ba1f0161c665977d5e7a8c32955ac
(38–49): flat role-major IDs reshape role_count×N then transpose.
vsrl_eval.py 17878B SHA256 eb6e765503bcc27fb73628476b641328e6f8bb26e4839f3f4645dc13afe72567
(142–168): label1 positive, positive agent role equals ann_id, role0 missing.
Caller authenticates original COCO2014 IDs/bytes, licensing, freshness, split
and JSON duplicate keys BEFORE this decoded-object API. Required fields are
validated; extra action/instance fields are preserved, not interpreted. No
actual annotation values were read to develop the procedural tests. Missing
non-agent role ID0 is UNKNOWN, never OFF.
General positive roles remain even when not localized nonperson pair targets.
Raw XYWH is retained: no clipping/+1/repair/official IoU is performed here.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType


def _require(condition, message):
    if not condition: raise ValueError(message)


def _integer(value, *, positive=False):
    _require(type(value) is int and value >= (1 if positive else 0), 'Original JSON integer ID/flag required')
    return value


def _text(value):
    _require(type(value) is str and 0 < len(value) <= 256 and value.strip() == value
        and all(ord(c) >= 32 and ord(c) != 127 for c in value), 'Bounded original name required')
    return value


def _number(value):
    _require(type(value) in (int,float) and math.isfinite(value), 'Finite original geometry required')
    return value


def _freeze(value, depth=0):
    _require(depth <= 64, 'Bounded metadata structure required')
    if type(value) is dict:
        _require(all(type(k) is str for k in value), 'JSON string keys required')
        return MappingProxyType({k:_freeze(v,depth+1) for k,v in value.items()})
    if type(value) is list: return tuple(_freeze(v,depth+1) for v in value)
    _require(value is None or type(value) in (str,bool,int,float), 'Plain decoded JSON metadata required')
    if type(value) in (int,float): _require(math.isfinite(value), 'Finite metadata required')
    return value


def _vector(value):
    _require(type(value) is list, 'Flat or explicit Nx1 original vector required')
    if value and all(type(v) is list and len(v) == 1 for v in value): value = [v[0] for v in value]
    _require(all(type(v) is int for v in value), 'Integer vector; no float/bool IDs or mixed shapes')
    return tuple(value)


@dataclass(frozen=True)
class CocoInstanceReference:
    annotation_id: int
    image_id: int
    category_id: int
    iscrowd: int
    bbox_xywh: tuple
    area: int | float
    raw_box_positive: bool
    raw_box_inside_image: bool
    positive_area: bool
    extra_fields: object


@dataclass(frozen=True)
class VcocoRoleRow:
    action_slot: int
    row_slot: int
    action_name: str
    image_id: int
    agent_annotation_id: int
    label: int
    role_names: tuple
    role_object_ids: tuple
    agent: CocoInstanceReference
    positive_role_endpoints: tuple  # Nonpositive rows: IDs retained, endpoints unconsulted None.
    nonperson_pair_eligible: tuple
    pair_unscorable_reasons: tuple


@dataclass(frozen=True)
class LocalizedPositivePair:
    image_id: int
    agent_annotation_id: int
    object_annotation_id: int
    row_role_references: tuple     # Every action/role source row; one physical pair weight.


@dataclass(frozen=True)
class VcocoRoleReference:
    rows: tuple
    localized_positive_pairs: tuple
    action_metadata: tuple
    source_fingerprint: str       # Content digest only, not authentication.
    scope: object


def parse_vcoco_role_reference(actions, instances, images):
    """Join supplied records; no file/network/model/selection or annotation repair.

    Actions need six native fields; vectors may be flat or explicit Nx1, while
    role_object_id MUST be the original flat role-major serialization. Unknown
    shapes fail qualification, not a scientific quality judgment. Positive
    nonzero endpoints must resolve same-image; zero stays missing. Nonpositive
    role IDs are retained without inventing endpoint/negative semantics.
    Optional projection_source retains caller-supplied original action/row slots;
    its indices are structural provenance, not authenticated source evidence.
    """
    _require(all(type(x) is list for x in (actions,instances,images)), 'Decoded original record lists required')
    original = json.dumps((actions,instances,images),sort_keys=True,allow_nan=False,separators=(',',':'))
    image_index = {}
    for image in images:
        _require(type(image) is dict and {'id','width','height'} <= image.keys(), 'COCO image grid fields required')
        iid = _integer(image['id'],positive=True)
        _require(iid not in image_index, 'Ambiguous duplicate original image ID')
        image_index[iid] = (_integer(image['width'],positive=True),_integer(image['height'],positive=True))
    index = {}
    for record in instances:
        fields = {'id','image_id','category_id','iscrowd','bbox','area'}
        _require(type(record) is dict and fields <= record.keys(), 'Original COCO instance fields required')
        aid = _integer(record['id'],positive=True); iid = _integer(record['image_id'],positive=True)
        _require(aid not in index and iid in image_index, 'Ambiguous instance ID or absent original image')
        category = _integer(record['category_id'],positive=True); crowd = _integer(record['iscrowd'])
        _require(crowd in (0,1) and type(record['bbox']) is list and len(record['bbox']) == 4, 'Original crowd/XYWH ABI required')
        box = tuple(_number(v) for v in record['bbox']); area = _number(record['area'])
        x,y,w,h = box; width,height = image_index[iid]
        sums = (x+w,y+h); _require(all(math.isfinite(v) for v in sums), 'Raw endpoint arithmetic overflow')
        positive = w > 0 and h > 0
        index[aid] = CocoInstanceReference(aid,iid,category,crowd,box,area,positive,
            bool(positive and x >= 0 and y >= 0 and sums[0] <= width and sums[1] <= height),area > 0,
            _freeze({k:v for k,v in record.items() if k not in fields}))
    rows, metadata, pairs = [], [], {}; seen_actions, source_slots = set(), set()
    required = {'action_name','role_name','ann_id','label','image_id','role_object_id'}
    for action_slot, action in enumerate(actions):
        _require(type(action) is dict and required <= action.keys(), 'Six native action fields required')
        name = _text(action['action_name']); _require(name not in seen_actions, 'Ambiguous duplicate action name')
        seen_actions.add(name); roles = action['role_name']
        _require(type(roles) is list and len(roles) > 0, 'Original ordered roles required')
        roles = tuple(_text(v) for v in roles)
        _require(roles[0] == 'agent' and len(set(roles)) == len(roles), 'Unique roles with agent first required')
        ann, labels, image_ids = (_vector(action[k]) for k in ('ann_id','label','image_id'))
        n = len(ann); flat = action['role_object_id']
        _require(len(labels) == len(image_ids) == n and type(flat) is list and len(flat) == n*len(roles), 'Exact role-major row alignment required')
        _require(all(type(v) is int and v >= 0 for v in flat) and all(v in (0,1) for v in labels), 'Raw role IDs/nonnegative and labels0/1 required')
        source_slot, row_slots = action_slot, tuple(range(n))
        if 'projection_source' in action:
            projection = action['projection_source']
            _require(type(projection) is dict and set(projection) == {'action_slot','row_slots','row_count'},
                'Explicit original projection provenance required')
            source_slot = _integer(projection['action_slot']); row_count = _integer(projection['row_count'])
            selected = projection['row_slots']
            _require(type(selected) is list and len(selected) == n and all(type(v) is int and 0 <= v < row_count for v in selected)
                and all(a < b for a,b in zip(selected,selected[1:])), 'Unique sorted original row slots required')
            row_slots = tuple(selected)
        _require(source_slot not in source_slots, 'Ambiguous original action slot')
        source_slots.add(source_slot)
        metadata.append(_freeze({k:v for k,v in action.items() if k not in required})); positives = set()
        for input_row,(aid,label,iid) in enumerate(zip(ann,labels,image_ids)):
            row_slot = row_slots[input_row]
            _integer(aid,positive=True);_integer(iid,positive=True)
            _require(aid in index and index[aid].image_id == iid and index[aid].category_id == 1,
                'Original agent must resolve to same-image COCO person')
            agent = index[aid]; ids = tuple(flat[role*n+input_row] for role in range(len(roles)))
            endpoints, eligible, reasons = [], [], []
            if label == 1:
                _require(aid not in positives and ids[0] == aid, 'Duplicate positive action/person or conflicting agent role')
                positives.add(aid)
            for role,rid in enumerate(ids):
                endpoint = None; why = []
                if not label: why.append('nonpositive_role_unconsulted_not_pair_negative')
                elif rid == 0: why.append('missing_role_unknown_not_off')
                else:
                    _require(rid in index and index[rid].image_id == iid, 'Dangling or cross-image positive endpoint')
                    endpoint = index[rid]
                    if role == 0: why.append('agent_role_not_object_pair')
                    if agent.iscrowd: why.append('crowd_agent')
                    if not agent.raw_box_positive or not agent.positive_area: why.append('invalid_raw_agent_geometry')
                    if rid == aid and role != 0: why.append('self_role')
                    if endpoint.category_id == 1 and role != 0: why.append('person_role_not_nonperson_target')
                    if endpoint.iscrowd and role != 0: why.append('crowd_target')
                    if not endpoint.raw_box_positive or not endpoint.positive_area: why.append('invalid_raw_endpoint_geometry')
                ok = label == 1 and role > 0 and endpoint is not None and not why
                endpoints.append(endpoint);eligible.append(ok);reasons.append(tuple(why))
                if ok: pairs.setdefault((iid,aid,rid),[]).append((source_slot,row_slot,role,name,roles[role]))
            rows.append(VcocoRoleRow(source_slot,row_slot,name,iid,aid,label,roles,ids,agent,
                tuple(endpoints),tuple(eligible),tuple(reasons)))
    _require(original == json.dumps((actions,instances,images),sort_keys=True,allow_nan=False,separators=(',',':')),
        'Caller source changed while parsing')
    pair_rows = tuple(LocalizedPositivePair(*key,tuple(refs)) for key,refs in pairs.items())
    scope = MappingProxyType(dict(role_zero='unknown_missing_not_off',nonpositive_pairs_verified_negative=False,
        pair_weight='one_per_unique_image_agent_object_ID',roles_exhaustive_contact=False,anatomical_side_labeled=False,
        temporal_identity_labeled=False,task_target_labeled=False,official_AP_computed=False,
        raw_geometry_clipped=False,source_authenticated=False,rights_verified=False,
        nonpositive_role_endpoints_consulted=False,image_extra_metadata_consulted=False))
    return VcocoRoleReference(tuple(rows),pair_rows,tuple(metadata),hashlib.sha256(original.encode()).hexdigest(),scope)
