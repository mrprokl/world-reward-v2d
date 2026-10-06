"""Explicit-context full48 endpoint adapter; no jobs, private inputs or models.

The caller authenticates acquisition, runtime/assets and its whole source, loads
the unchanged original models once, and owns bounded cleanup/failure receipts.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import rgb_endpoint_bank as original
from world_reward import rgb_bank_inputs as public

rt = original.rt
SCHEMA = 'world_reward.vcoco_fit_cal_endpoint_observations.v1'
COUNT = 48
PROFILE = dict(images=48, model_loads=2, person_query='person.', confidence=.3,
               text_threshold=.25, nms_iou=.7, owl_patches=3600, person_native_queries=900)
KEYS = frozenset(('person_raw_boxes', 'person_raw_scores', 'person_raw_labels',
    'person_retained_boxes', 'person_retained_scores', 'person_retained_raw_slots',
    'person_retained_ids', 'person_model_pred_boxes', 'person_model_logits',
    'person_model_input_ids', 'person_model_attention_mask', 'owl_patch_ids',
    'owl_boxes_padded_normalized_cxcywh', 'owl_objectness_logits',
    'owl_boxes_original_xyxy', 'image_size', 'original_frame_index'))
REUSE = {'infra/rgb_endpoint_bank.py': dict(bytes=34983, sha256='7a4641eb7da72cd4610750b19758ee0b7afe0c7ac2865ce8e1f6ddb6715d3081'),
    'src/world_reward/rgb_bank_inputs.py': dict(bytes=4240, sha256='f6f00d4c2f51be665112ac697753ae755fb914caeaf6362472ae76c38b7ffa99'),
    'src/world_reward/owlv2_object_observations.py': dict(bytes=8748, sha256='8587489a860b44a74bc0149c67960128b6b12aa1e4d51bc07e0e52103da5b888')}
HELPERS = tuple(dict.fromkeys(('infra/vcoco_fit_cal_endpoint_observations.py', *REUSE, *original.HELPERS)))


@dataclass(frozen=True)
class EndpointContext:
    inputs: Path
    output: Path

    def __post_init__(self):
        for name in ('inputs', 'output'):
            object.__setattr__(self, name, rt.canonical(getattr(self, name)))
        rt.require(self.inputs != self.output and self.inputs not in self.output.parents
                   and self.output not in self.inputs.parents, 'Separate canonical input/output namespaces')


def source_identity(code, source):
    """Only reusable source/origins, not a substitute for whole-caller proof."""
    code = rt.canonical(code)
    for module, name in ((original, 'infra/rgb_endpoint_bank.py'),
            (public, 'src/world_reward/rgb_bank_inputs.py'), (rt, 'infra/mediapipe_cpu_runtime_verify.py'),
            (original.owl, 'infra/owlv2_native_qualify.py'), (original.gdi, 'infra/openimages_joint_pair_gdi.py')):
        rt.require(Path(module.__file__).resolve() == code/name, 'Actual qualified helper origin differs')
    rt.require(Path(__file__).resolve() == code/HELPERS[0]
        and all(source['helpers'][name] == wanted and rt.identity(code/name, 2 << 20) == wanted
                for name, wanted in REUSE.items()), 'Qualified numerical helper bytes differ')
    return {name: rt.identity(code/name, 2 << 20) for name in REUSE}


def public_inputs(context, pin, *, native_mounts=False):
    inputs = public.read_inputs(context.inputs, pin, COUNT, identity=rt.identity, pinned=rt.pinned,
                               maximum_slots=COUNT, readonly_directory=not native_mounts)
    rt.require([public.original_slot(row, COUNT) for row in inputs['images']] == list(range(COUNT)),
               'Every fixed48 original slot required; no smaller population')
    return inputs


def check(deadline):
    rt.require(type(deadline) in (int, float) and math.isfinite(deadline), 'Finite caller deadline required')
    if time.monotonic() >= deadline:
        raise TimeoutError('Inclusive complete endpoint deadline')


def array_identities(arrays):
    return {name: dict(shape=list(a.shape), dtype=a.dtype.str,
                      sha256=hashlib.sha256(a.tobytes()).hexdigest()) for name, a in arrays.items()}


def validate_arrays(arrays, row):
    """All17 original native arrays; no clipping/threshold/output repair."""
    import numpy as np
    from world_reward.owlv2_object_observations import Owlv2ObjectObservations
    rt.require(type(arrays) is dict and set(arrays) == KEYS
        and all(type(a) is np.ndarray and not a.dtype.hasobject for a in arrays.values()), 'Exact17 native arrays required')
    size, frame = arrays['image_size'], arrays['original_frame_index']
    rt.require(size.dtype == np.int64 and size.shape == (2,) and size.tolist() == [row['height'], row['width']]
        and frame.dtype == np.int64 and frame.shape == () and int(frame) == 0, 'Original grid/still frame0 required')
    original.gdi.validate_native_text_logits(arrays['person_model_pred_boxes'], arrays['person_model_logits'],
        arrays['person_model_input_ids'], arrays['person_model_attention_mask'], 256)
    labels = arrays['person_raw_labels']
    rt.require(labels.dtype.kind == 'U' and labels.ndim == 1, 'Every raw person text label required')
    replay = original.gdi.retained_bank(arrays['person_raw_boxes'], arrays['person_raw_scores'], labels.tolist(),
                                      row['image_id'], 'person', row['width'], row['height'])
    rt.require(all(arrays['person_'+name].dtype == value.dtype and arrays['person_'+name].shape == value.shape
        and arrays['person_'+name].tobytes() == value.tobytes() for name, value in replay.items()),
        'Exact original post-NMS rows/slots/IDs required')
    Owlv2ObjectObservations(0, (row['height'], row['width']), (60, 60), arrays['owl_patch_ids'],
        arrays['owl_boxes_padded_normalized_cxcywh'], arrays['owl_objectness_logits'], arrays['owl_boxes_original_xyxy'])


def observe(context, inputs, models, deadline, *, records=None):
    """Consume one original load_models result; preserve partial records on failure.

    No model constructor here. The original loaded keepalive remains caller-owned.
    All48 file hashes authenticate before the first inference and rehash in finally.
    """
    records = [] if records is None else records
    rt.require(type(records) is list and not records and type(models) is tuple and len(models) == 4,
               'Fresh records and exact original loaded-model tuple required')
    rt.require(type(inputs) is dict and set(inputs) == {'schema', 'images'} and inputs['schema'] == public.SCHEMA
        and type(inputs['images']) is list and len(inputs['images']) == COUNT
        and [public.original_slot(r, COUNT) for r in inputs['images']] == list(range(COUNT))
        and all(set(r) == public.KEYS for r in inputs['images']), 'Complete blinded48 projection required')
    detect, model, operations, keepalive = models
    pins = {row['file']: {k: row[k] for k in ('bytes', 'sha256')} for row in inputs['images']}
    def verify():
        for name, pin in pins.items():
            check(deadline); rt.require(rt.identity(context.inputs/name, 16 << 20) == pin, 'Original JPEG bytes changed')
    verify(); frozen = original.encode(inputs)
    try:
        for ordinal, row in enumerate(inputs['images']):
            check(deadline); rgb = public.decode_rgb(context.inputs, row, identity=rt.identity)
            arrays, metadata = original.bank_arrays(rgb, row, ordinal, detect, model, operations)
            validate_arrays(arrays, row); before = array_identities(arrays); operations.tensor_ops.cuda.synchronize()
            metadata.update(original_slot=ordinal, acquired_ordinal=ordinal)
            metadata.update(original.save_bank(context.output, ordinal, arrays))
            rt.require(before == array_identities(arrays) == metadata['arrays'], 'Native save changed arrays')
            records.append(metadata); check(deadline)
    finally:
        rt.require(original.encode(inputs) == frozen, 'Public input metadata mutated'); verify()
    return records


def validate_records(context, inputs, records, deadline):
    """Reopen/hash every saved17-array bank, never another model forward."""
    import numpy as np
    rt.require(type(inputs) is dict and set(inputs) == {'schema', 'images'} and inputs['schema'] == public.SCHEMA
        and type(inputs['images']) is list and len(inputs['images']) == COUNT
        and type(records) is list and len(records) == COUNT, 'Complete48 native inputs/banks required')
    for ordinal, (row, record) in enumerate(zip(inputs['images'], records)):
        check(deadline); path = context.output/f'image_{ordinal:06d}.npz'
        rt.require(record['image_id'] == row['image_id'] and record['bank_index'] == record['original_slot']
            == record['acquired_ordinal'] == ordinal and record['original_frame_index'] == 0
            and record['input_file'] == row['file'] and record['input_identity'] == {k: row[k] for k in ('bytes', 'sha256')}
            and record['image_size'] == [row['height'], row['width']] and record['file'] == path.name
            and rt.identity(path, 16 << 20) == record['identity'], 'Original complete48 output binding required')
        with np.load(path, allow_pickle=False) as saved:
            rt.require(set(saved.files) == KEYS, 'Original17 saved members required')
            arrays = {n: saved[n] for n in saved.files}
        validate_arrays(arrays, row)
        rt.require(array_identities(arrays) == record['arrays'] and record['person_native_queries'] == 900
            and record['owl_patches'] == 3600 and record['person_postprocessor_rows'] == len(arrays['person_raw_labels'])
            and record['person_retained_rows'] == len(arrays['person_retained_ids'])
            and record['person_ids'] == arrays['person_retained_ids'].tolist(), 'Full native census/array identities differ')
