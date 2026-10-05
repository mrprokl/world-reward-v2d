"""Two-pass action-major projection; excluded labels/role IDs are never decoded.

This is a tiny-tested metadata primitive, not an acquisition, census or selector.
The caller authenticates source bytes, original COCO2014 IDs, rights/freshness
and the eligible image set. Every byte is lexically checked on both passes.
Only structural names/agent/image IDs are consulted across excluded records.
Reuse the immutable JSON lexer without changing historical producer code.
"""
from contextlib import closing
import hashlib

import metadata_json_stream as js

STRUCTURAL = frozenset(('action_name', 'role_name', 'ann_id', 'image_id'))
SEMANTIC = frozenset(('label', 'role_object_id'))


def _require(value, message):
    if not value:
        raise ValueError(message)


class _HashedReader:
    def __init__(self, source):
        self.source, self.digest, self.bytes = source, hashlib.sha256(), 0

    def read(self, count):
        raw = self.source.read(count)
        _require(type(raw) is bytes, 'Binary source required')
        self.digest.update(raw)
        self.bytes += len(raw)
        return raw


def _decoded(stream):
    raw = bytearray()
    stream.captures.append((raw, stream.row_bytes))
    try:
        stream.value()
    finally:
        stream.captures.pop()
    return js.strict_decode(bytes(raw))


def _vector(value):
    _require(type(value) is list, 'Structural native ID vector required')
    if value and all(type(v) is list and len(v) == 1 for v in value):
        value = [v[0] for v in value]
    _require(all(type(v) is int and v > 0 for v in value), 'Positive native IDs, no bool/float/mixed shapes')
    return tuple(value)


def _structure(value):
    _require(set(value) == STRUCTURAL, 'Four structural action fields required')
    name, roles = value['action_name'], value['role_name']
    _require(type(name) is str and bool(name) and type(roles) is list and roles
             and roles[0] == 'agent' and all(type(r) is str and bool(r) for r in roles)
             and len(set(roles)) == len(roles), 'Named action and unique agent-first roles required')
    ann, images = _vector(value['ann_id']), _vector(value['image_id'])
    _require(len(ann) == len(images), 'Native agent/image row counts differ')
    return name, tuple(roles), ann, images


def _semantic_array(stream, *, field, structure, keep):
    """Count every original scalar; decode only eligible entries, role-major."""
    n, roles = len(structure[2]), len(structure[1])
    selected, count = [], 0
    stream.space()
    stream.expect(91)
    stream.space()
    if stream.peek() != 93:
        while True:
            stream.space()
            if field == 'role_object_id':
                _require(stream.peek() not in (91, 123), 'Flat role-major ID array required')
            row = count if field == 'label' else count % n if n else -1
            if row in keep:
                value = _decoded(stream)
                if field == 'label' and type(value) is list and len(value) == 1:
                    value = value[0]
                _require(type(value) is int and (value in (0, 1) if field == 'label' else value >= 0),
                         'Native consulted label0/1 or nonnegative role ID required')
                selected.append(value)
            else:
                stream.value()
            count += 1
            _require(count <= n * (1 if field == 'label' else roles), 'Original semantic vector too long')
            stream.space()
            if stream.peek() == 93:
                break
            stream.expect(44)
    stream.expect(93)
    _require(count == n * (1 if field == 'label' else roles), 'Original semantic row count differs')
    return selected


def _actions(stream, previous, eligible):
    stream.space()
    stream.expect(91)
    stream.space()
    structures, projected, names = [], [], set()
    if stream.peek() == 93:
        stream.take()
        stream.finish()
        _require(previous is None or not previous, 'Source action count changed')
        return structures, projected
    while True:
        slot = len(structures)
        if previous is not None:
            _require(slot < len(previous), 'Source action count changed')
            expected = previous[slot]
            keep = {i for i, image in enumerate(expected[3]) if image in eligible}
        else:
            expected, keep = None, set()
        stream.expect(123)
        fields, observed, semantic = set(), {}, {}
        stream.space()
        if stream.peek() != 125:
            while True:
                key = stream.string(decode=True)
                _require(key not in fields, 'Duplicate action key')
                fields.add(key)
                stream.space()
                stream.expect(58)
                if key in STRUCTURAL:
                    observed[key] = _decoded(stream)
                elif key in SEMANTIC and previous is not None:
                    semantic[key] = _semantic_array(stream, field=key, structure=expected, keep=keep)
                else:
                    stream.value()  # Includes all labels/role IDs in pass one and unknown extras.
                stream.space()
                if stream.peek() == 125:
                    break
                stream.expect(44)
                stream.space()
        stream.expect(125)
        _require(STRUCTURAL | SEMANTIC <= fields, 'Six native action fields required')
        structure = _structure(observed)
        _require(structure[0] not in names, 'Duplicate action name')
        names.add(structure[0])
        if expected is not None:
            _require(structure == expected, 'Structural source changed between passes')
            row_slots = sorted(keep)
            projected.append(dict(action_name=structure[0], role_name=list(structure[1]),
                ann_id=[structure[2][i] for i in row_slots], image_id=[structure[3][i] for i in row_slots],
                label=semantic['label'], role_object_id=semantic['role_object_id'],
                projection_source=dict(action_slot=slot, row_slots=row_slots, row_count=len(structure[2]))))
        structures.append(structure)
        stream.space()
        if stream.peek() == 93:
            break
        stream.expect(44)
        stream.space()
    stream.expect(93)
    stream.finish()
    _require(previous is None or len(previous) == len(structures), 'Source action count changed')
    return structures, projected


def project_vcoco_actions(open_source, eligible_image_ids, *, check=lambda: None,
                          max_bytes=16 << 20, field_bytes=2 << 20, chunk_bytes=65536,
                          expected_image_ids=None):
    """Open the SAME immutable source twice; return selected native action rows.

    Returned role IDs retain role-major order over selected rows; original row
    indices/counts are explicit projection metadata. This does not verify split,
    annotation→image/category joins, source authentication or excluded semantics.
    """
    _require(type(eligible_image_ids) in (set, frozenset)
             and all(type(i) is int and i > 0 for i in eligible_image_ids), 'Explicit positive eligible image IDs required')
    _require(all(type(v) is int and v > 0 for v in (max_bytes, field_bytes, chunk_bytes)),
             'Positive integer byte limits required')
    eligible_image_ids = frozenset(eligible_image_ids)
    _require(expected_image_ids is None or (type(expected_image_ids) in (set, frozenset)
             and all(type(i) is int and i > 0 for i in expected_image_ids)),
             'Explicit positive expected image IDs required')
    expected_image_ids = None if expected_image_ids is None else frozenset(expected_image_ids)
    structures, actions, identity = None, None, None
    for pass_index in range(2):
        with closing(open_source()) as source:
            reader = _HashedReader(source)
            stream = js._Stream(reader, field_bytes, max_bytes, check, chunk_bytes)
            observed, projected = _actions(stream, structures, eligible_image_ids)
            pin = dict(bytes=reader.bytes, sha256=reader.digest.hexdigest())
        if pass_index == 0:
            _require(expected_image_ids is None or {i for s in observed for i in s[3]} == expected_image_ids,
                     'Structural role-file image IDs differ from official split')
            structures, identity = observed, pin
        else:
            _require(pin == identity, 'Complete source bytes changed between passes')
            actions = projected
    return actions, dict(source_identity=identity, passes=2, action_count=len(structures),
        source_row_count=sum(len(s[2]) for s in structures),
        selected_row_count=sum(len(a['ann_id']) for a in actions),
        excluded_semantic_values_decoded=False, source_authenticated=False,
        rights_verified=False, census_performed=False, rgb_read=False)


def iter_filtered_coco_annotations(source, eligible_image_ids, *, check=lambda: None,
                                   max_bytes=512 << 20, row_bytes=1 << 20, chunk_bytes=65536):
    """Stream original annotations; only fresh eligible image rows are decoded.

    IDs remain native; no clipping, category filtering, join or source proof.
    Caller MUST exhaust this generator for complete lexical/UTF8/EOF/ZIP CRC.
    """
    _require(type(eligible_image_ids) in (set, frozenset)
             and all(type(i) is int and i > 0 for i in eligible_image_ids),
             'Explicit positive eligible image IDs required')
    eligible_image_ids = frozenset(eligible_image_ids)
    stream = js._Stream(source, row_bytes, max_bytes, check, chunk_bytes)
    stream.space()
    stream.expect(123)
    stream.space()
    seen = set()
    if stream.peek() != 125:
        while True:
            key = stream.string(decode=True)
            _require(key not in seen, 'Duplicate COCO root key')
            seen.add(key)
            stream.space()
            stream.expect(58)
            if key == 'annotations':
                for iid, raw in stream.rows('image_id'):
                    _require(type(iid) is int and iid > 0, 'Original annotation image ID required')
                    if iid in eligible_image_ids:
                        yield iid, js.strict_decode(raw)
            else:
                stream.value()
            stream.space()
            if stream.peek() == 125:
                break
            stream.expect(44)
            stream.space()
    stream.expect(125)
    stream.finish()
    _require('annotations' in seen, 'Original annotations root absent')
