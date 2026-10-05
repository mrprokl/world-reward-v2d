"""Saved-only endpoint capacity; two separate CPU phases, never model inference.

All complete acquired prediction arrays authenticate before reference access.
DEV failure never mounts/parses RESERVED labels. Source and native qualification
proofs remain distinct from the endpoint scientific decision.
"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1]/'src')]
import mediapipe_cpu_runtime_verify as rt
import rgb_endpoint_bank as bank

ROOT = rt.ROOT
DATA = Path('/srv/world-reward-data/coco_endpoint_v2')
OUTPUT = ROOT/'results/coco-endpoint-evaluation-v2'
ENTRY = 'run_coco_endpoint_evaluate'
CONFIG = 'configs/coco_endpoint_evaluation_v2.json'
NATIVE_FILES = ('infra/coco_endpoint_evaluate.py', CONFIG, *bank.NATIVE_FILES,
    'src/world_reward/endpoint_proposal_recall.py', 'src/world_reward/proposal_bbox_recall.py',
    'src/world_reward/proposal_recall.py')
HELPERS = (*NATIVE_FILES, 'infra/run_coco_endpoint_evaluate.sh', *bank.HELPERS,
    'infra/coco_endpoint_prepare.py', 'infra/run_coco_endpoint_prepare.sh', 'configs/coco_endpoint_v2.json',
    'infra/coco_proposal_prepare.py', 'infra/run_coco_proposal_prepare.sh', 'configs/coco_proposal_v1.json',
    'infra/openimages_joint_pair_acquire.py')
PIN_NAMES = ('census', 'cohort', 'acquisition', 'public', 'native', 'host')
ARRAY_NAMES = {'person_'+n for n in ('raw_boxes', 'raw_scores', 'raw_labels', 'retained_boxes',
    'retained_scores', 'retained_raw_slots', 'retained_ids', 'model_pred_boxes', 'model_logits',
    'model_input_ids', 'model_attention_mask')} | {'owl_'+n for n in ('patch_ids',
    'boxes_padded_normalized_cxcywh', 'objectness_logits', 'boxes_original_xyxy')} | {'image_size', 'original_frame_index'}


def encode(v): return bank.encode(v)


def write(path, v):
    raw = encode(v); rt.require(len(raw) <= 2 << 20, 'Bounded scalar evaluator receipt')
    rt.write(path, raw); return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check(deadline):
    if not math.isfinite(deadline) or time.monotonic() >= deadline: raise TimeoutError('Inclusive180s saved evaluator deadline')


def configuration(code, source):
    p = rt.pinned(code/CONFIG, source['helpers'][CONFIG], 16 << 10)
    fixed = dict(schema='world_reward.coco_endpoint_evaluation.v2', slots=64, development_slots=32,
        reserved_slots=32, minimum_acquired_development_images_before_reference_access=24,
        primary_iou=.5, object_budgets=[32, 128, 3600], primary_object_budget=128,
        person_gate=.7, object_gate=.7, missing_slot_recall=0., budget_seconds=180,
        memory_bytes=6 << 30, image_id=bank.IMAGE, quality_verified=False, ownership_verified=False, adoption=False)
    rt.require(set(p) == set(fixed) | {'helper_pins'} and all(type(p.get(k)) is type(v) and p[k] == v for k, v in fixed.items()),
        'Frozen approved separate-endpoint metric policy required')
    rt.require(set(p['helper_pins']) == {'infra/mediapipe_cpu_runtime_verify.py',
        'src/world_reward/endpoint_proposal_recall.py', 'src/world_reward/proposal_bbox_recall.py',
        'src/world_reward/proposal_recall.py', 'src/world_reward/owlv2_object_observations.py'}, 'Complete fixed metric helper pins required')
    for name, wanted in p['helper_pins'].items(): rt.require(source['helpers'][name] == wanted, 'Released exact metric/helper changed')
    return p


def source_binding(code, revision, helpers, entry):
    rt.require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Actual producer revision required')
    return rt.source(ROOT, code, revision, entry, helpers)


def receipt_paths():
    return dict(census=DATA/'metadata/report.json', cohort=DATA/'metadata/cohort.json',
        acquisition=DATA/'eval_private/manifest.json', public=DATA/'inputs/manifest.json',
        native=bank.OUTPUT/'native.json', host=bank.OUTPUT/'host.json')


def structure(values, pins):
    """No reference bytes/values here; validate all64 acquired/missing slots."""
    cohort, census, acquisition, public = (values[n] for n in ('cohort', 'census', 'acquisition', 'public'))
    rows = cohort['records']; categories = cohort['categories']
    rt.require(cohort['schema'] == 'world_reward.coco_endpoint_cohort.v2' and cohort['hash_namespace'] == 'world_reward.coco_endpoint_v2/'
        and cohort['freeze_before_rgb'] is cohort['no_replacements'] is True
        and len(rows) == 64 and [r['slot'] for r in rows] == list(range(64))
        and [r['split'] for r in rows] == ['DEV']*32+['RESERVED']*32
        and len({r['image_id'] for r in rows}) == len({r['photo_id'] for r in rows}) == 64,
        'Complete untouched new64 cohort required')
    rt.require(census['schema'] == acquisition['schema'] == 'world_reward.coco_endpoint_prepare.v2'
        and census['phase'] == 'census' and acquisition['phase'] == 'acquire'
        and census['status'] == acquisition['status'] == 'pass'
        and census['cohort_identity'] == acquisition['cohort_identity'] == pins['cohort']
        and acquisition['census_report_identity'] == pins['census']
        and acquisition['public_inputs_identity'] == pins['public'] and acquisition['selected_slots'] == rows
        and all(v['source_and_inputs_rehashed_after'] is v['outputs_sealed'] is True for v in (census, acquisition))
        and census['freeze_before_rgb'] is acquisition['freeze_before_rgb'] is True
        and acquisition['reference_values_read_for_acquisition'] is False and acquisition['replacement_count'] == acquisition['retry_count'] == 0
        and census['closed_individual_reference_values_read'] is acquisition['closed_individual_reference_values_read'] is False,
        'Actual census freeze and complete no-replacement acquisition required')
    rt.require(len({c['id'] for c in categories}) == len(categories)
        and len([c for c in categories if c['name'] == 'person']) == 1, 'Original category catalog required')
    status = acquisition['records']; mappings = acquisition['public_mappings']; acquired = [r for r in status if r['status'] == 'acquired']
    rt.require(len(status) == 64 and [r['slot'] for r in status] == list(range(64))
        and len(mappings) == len(acquired) == len(public['images'])
        and public['schema'] == 'world_reward.rgb_proposal_inputs.v1' and set(public) == {'schema', 'images'}, 'Complete acquired and unavailable slot census required')
    for r, original in zip(status, rows):
        rt.require(r['image_id'] == original['image_id'] and r['split'] == original['split']
            and r['status'] in ('acquired', 'unavailable'), 'Missing original slot changed')
    for r, m, image in zip(acquired, mappings, public['images']):
        original = rows[r['slot']]; opaque = hashlib.sha256(('world_reward.coco_endpoint_v2/'+f"{r['image_id']:012d}").encode()).hexdigest()[:32]
        rt.require(m == dict(slot=r['slot'], public_image_id=opaque, public_file=f"image_{r['slot']:06d}.jpg", image_pin=r['image_pin'])
            and image == dict(image_id=opaque, file=m['public_file'], **r['image_pin'], width=original['image']['width'], height=original['image']['height'])
            and (r['jpeg_header']['width'], r['jpeg_header']['height']) == (image['width'], image['height']), 'Independent opaque original RGB mapping differs')
    counts = dict(slots=64, acquired=len(acquired), missing=64-len(acquired),
        acquired_DEV=sum(r['split'] == 'DEV' for r in acquired), acquired_RESERVED=sum(r['split'] == 'RESERVED' for r in acquired))
    rt.require(acquisition['counts'] == counts and counts['acquired_DEV'] >= 24, 'Prospective minimum24 DEV before all reference access required')
    for row in rows:
        rt.require(row['reference_file'] == f"reference_{row['slot']:06d}.json" and row['image']['id'] == row['image_id']
            and row['image']['license'] == 4 and row['image']['width'] > 0 and row['image']['height'] > 0, 'Native new reference/grid identity required')
    return {m['slot']: m for m in mappings}


def authenticate(code, pins, revisions):
    """Host complete source and old runtime proofs; old reference values unread."""
    import coco_endpoint_prepare as prep
    paths = receipt_paths(); values = {n: rt.pinned(path, pins[n], 4 << 20) for n, path in paths.items()}
    sources = {}
    for name in ('census', 'acquisition'):
        revision = revisions[name]; old = ROOT/'jobs'/revision/prep.ENTRY/'code'
        sources[name] = source_binding(old, revision, prep.HELPERS, prep.ENTRY)
        rt.require(values[name]['producer_revision'] == revision and values[name]['source_binding'] == sources[name]
            and values[name]['configuration_identity'] == sources[name]['helpers'][prep.CONFIG], 'Authentic original metadata source required')
    rt.require(values['cohort']['source_binding'] == sources['census'] and values['cohort']['producer_revision'] == revisions['census'], 'Cohort/census source mismatch')
    mapping = structure(values, pins)
    producer = revisions['bank']; old = ROOT/'jobs'/producer/bank.ENTRY/'code'
    sources['bank'] = source_binding(old, producer, bank.HELPERS, bank.ENTRY)
    for name in bank.NATIVE_FILES:
        rt.require(rt.identity(code/name, 2 << 20, empty=True) == sources['bank']['helpers'][name],
            'Current reused bank implementation/config must equal original producer, not current provenance alone')
    p = bank.configuration(code, sources['bank']); prior = bank.qualifications(code, p, live=True)
    native, host = values['native'], values['host']
    rt.require(host['schema'] == p['schema'] and host['stage'] == 'rgb_endpoint_bank_host' and host['status'] == 'pass'
        and host['producer_revision'] == producer and host['source_binding'] == sources['bank']
        and host['original_qualification'] == prior and host['public_inputs_identity'] == pins['public']
        and host['native_report_identity'] == pins['native'] and host['native_exit_status'] == 0
        and host['owned_cleanup_verified'] is host['outputs_sealed'] is host['source_inputs_runtime_assets_rehashed_after'] is True
        and host['native_images'] == native['images'], 'Actual sealed complete bank host PASS required')
    proof_pin = native['proof_identity']; proof = rt.pinned(bank.OUTPUT/'proof.json', proof_pin, 2 << 20)
    rt.require(proof['source'] == sources['bank'] and proof['native_files'] == {n: sources['bank']['helpers'][n] for n in bank.NATIVE_FILES}
        and proof['inputs_identity'] == pins['public'] and proof['images'] == len(values['public']['images'])
        and proof['owl_runtime'] == prior['owl']['native_runtime'] and native['runtime_identity'] == proof['owl_runtime'], 'Original independent native proof differs')
    bank.validate_report(native, p, producer, proof_pin, values['public'])
    frozen = {str(path): pins[n] for n, path in paths.items()}; frozen[str(bank.OUTPUT/'proof.json')] = proof_pin
    # Original census already bound these byte identities; hash-only, never JSON
    # decode historical references. They never enter the CPU evaluation mount.
    for name, wanted in values['census']['frozen_inputs'].items():
        rt.require(rt.identity(Path(name), 1 << 30, readonly=False) == wanted, 'Original census exclusion/cache bytes changed')
    for image in values['public']['images']:
        path = DATA/'inputs'/image['file']; wanted = {k: image[k] for k in ('bytes', 'sha256')}
        rt.require(rt.identity(path, 16 << 20) == wanted, 'Acquired original RGB byte identity differs'); frozen[str(path)] = wanted
    for row in native['images']:
        path = bank.OUTPUT/row['file']; rt.require(rt.identity(path, 16 << 20) == row['identity'], 'All native bank bytes must seal before truth'); frozen[str(path)] = row['identity']
    rt.require({x.name for x in bank.OUTPUT.iterdir()} == {'host.json', 'native.json', 'proof.json', '.container.cid',
        *[r['file'] for r in native['images']]}, 'Original complete no-extra bank output inventory')
    for row in values['cohort']['records']:
        path = DATA/'metadata'/row['reference_file']; rt.require(rt.identity(path, 4 << 20) == row['reference_identity'], 'New reference seal differs')
    return dict(values=values, pins=pins, revisions=revisions, sources=sources, mapping=mapping, frozen=frozen, original_qualification=prior)


def validate_npz(path, row):
    """Whole native arrays, no missing-row repair or predictor invocation."""
    import numpy as np
    import zipfile
    from world_reward.owlv2_object_observations import Owlv2ObjectObservations
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        rt.require(len(entries) == len(ARRAY_NAMES) and {x.filename for x in entries} == {n+'.npy' for n in ARRAY_NAMES}
            and all(not x.flag_bits & 1 and x.compress_type == 0 and 0 < x.file_size <= 16 << 20 for x in entries)
            and sum(x.file_size for x in entries) <= 16 << 20, 'Original bounded stored NPZ members only')
    with np.load(path, allow_pickle=False) as z:
        rt.require(set(z.files) == ARRAY_NAMES, 'Exact complete original endpoint NPZ fields required')
        a = {n: z[n] for n in z.files}
    rt.require(set(row['arrays']) == ARRAY_NAMES and all(a[n].dtype.str == r['dtype'] and list(a[n].shape) == r['shape']
        and hashlib.sha256(a[n].tobytes()).hexdigest() == r['sha256'] for n, r in row['arrays'].items()), 'Every native array identity required')
    rt.require(a['image_size'].dtype == np.int64 and a['image_size'].shape == (2,) and a['image_size'].tolist() == row['image_size']
        and a['original_frame_index'].dtype == np.int64 and a['original_frame_index'].shape == () and int(a['original_frame_index']) == 0,
        'Original still-image grid/frame required')
    objects = Owlv2ObjectObservations(0, tuple(row['image_size']), (60, 60), a['owl_patch_ids'],
        a['owl_boxes_padded_normalized_cxcywh'], a['owl_objectness_logits'], a['owl_boxes_original_xyxy'])
    bank.gdi.validate_native_text_logits(a['person_model_pred_boxes'], a['person_model_logits'],
        a['person_model_input_ids'], a['person_model_attention_mask'], 256)
    labels = a['person_raw_labels']; rt.require(labels.ndim == 1 and labels.dtype.kind == 'U', 'Complete original text labels required')
    replay = bank.gdi.retained_bank(a['person_raw_boxes'], a['person_raw_scores'], labels.tolist(),
        row['image_id'], 'person', row['image_size'][1], row['image_size'][0])
    for name, expected in replay.items():
        actual = a['person_'+name]; rt.require(actual.dtype == expected.dtype and actual.shape == expected.shape
            and actual.tobytes() == expected.tobytes(), 'Every original post-NMS row/slot/ID must replay exactly')
    rt.require(len(labels) == row['person_postprocessor_rows'] and len(replay['retained_ids']) == row['person_retained_rows']
        and replay['retained_ids'].tolist() == row['person_ids'], 'Original complete person census mismatch')
    return a['person_retained_boxes'], objects


def references(value, record, categories):
    import numpy as np
    rt.require(type(value) is dict and set(value) == {'annotations'} and type(value['annotations']) is list, 'Only original per-image reference projection')
    ids, boxes, kinds = [], [], []
    for r in value['annotations']:
        rt.require(type(r['id']) is int and r['id'] not in ids and r['image_id'] == record['image_id']
            and type(r['iscrowd']) is int and r['iscrowd'] == 0 and type(r['area']) in (int, float)
            and math.isfinite(r['area']) and r['area'] > 0 and r['category_id'] in categories
            and type(r['bbox']) is list and len(r['bbox']) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in r['bbox'])
            and r['bbox'][2] > 0 and r['bbox'][3] > 0, 'Complete original noncrowd positive continuous XYWH required')
        ids.append(r['id']); boxes.append(r['bbox']); kinds.append('human' if categories[r['category_id']] == 'person' else 'object')
    rt.require(kinds.count('human') >= 2 and kinds.count('object') >= 2, 'Original declared multiperson/multiobject population')
    return np.asarray(boxes, np.float64), tuple(kinds), tuple(str(i) for i in ids)


def metric_decision(phase, metrics):
    primary = metrics['primary']; p, o = primary['person_recall'], primary['object_recall']
    rt.require(metrics['slots'] == 32 and primary['iou'] == .5 and primary['object_budget'] == 128
        and primary['person_gate'] == primary['object_gate'] == .7
        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in (p, o))
        and primary['both_endpoint_gates_pass'] is (p >= .7 and o >= .7)
        and metrics['persons']['recalls']['0.5']['macro_fixed_slot_one_to_one_recall'] == p
        and metrics['objects']['128']['recalls']['0.5']['macro_fixed_slot_one_to_one_recall'] == o,
        'Exact declared fixed32 endpoint gate and metric coherence required')
    return ('DEV_CAPACITY_' if phase == 'dev' else 'RESERVED_CAPACITY_') + ('PASS' if primary['both_endpoint_gates_pass'] else 'FAIL')


def validate_dev(host, native, proof, binding, source):
    revision = host['producer_revision']
    rt.require(host['schema'] == 'world_reward.coco_endpoint_evaluation.v2' and host['stage'] == 'endpoint_evaluation_host'
        and host['phase'] == 'dev' and host['status'] == 'pass' and host['outputs_sealed'] is host['owned_cleanup_verified'] is True
        and host['source_binding'] == source and host['native_exit_status'] == 0 and native['producer_revision'] == revision
        and host['producer_revisions'] == binding['revisions'] and host['decision'] == 'DEV_CAPACITY_PASS'
        and proof['source'] == source and proof['phase'] == 'dev' and proof['approved_dev'] is None
        and proof['binding']['pins'] == binding['pins']
        and proof['binding']['sources'] == binding['sources'] and proof['binding']['revisions'] == binding['revisions']
        and host['source_inputs_predictions_references_rehashed_after'] is True
        and native['schema'] == host['schema'] and native['stage'] == 'saved_cpu_endpoint_recall'
        and native['status'] == 'pass' and native['phase'] == 'dev' and native['decision'] == metric_decision('dev', native['metrics']) == 'DEV_CAPACITY_PASS'
        and native['inputs'] == binding['pins'] and host['inputs'] == binding['pins']
        and native['all_native_banks_frozen_before_truth'] is native['source_inputs_predictions_references_rehashed_after'] is True
        and native['reference_files_decoded'] == len([r for r in binding['values']['acquisition']['records'] if r['split'] == 'DEV' and r['status'] == 'acquired'])
        and native['models_loaded'] == 0 and native['gpu_used'] is native['predictions_modified'] is False
        and all(row[k] is False for row in (host, native) for k in ('ownership_verified', 'quality_verified', 'adoption')),
        'Independent sealed DEV both-gate PASS mandatory before RESERVED mount')


def approved_dev(pin, binding):
    rt.require(stat.S_IMODE(rt.canonical(OUTPUT/'dev').lstat().st_mode) == 0o500
        and {p.name for p in (OUTPUT/'dev').iterdir()} == {'host.json', 'native.json', 'proof.json', '.container.cid'}, 'Original sealed DEV inventory required')
    host = rt.pinned(OUTPUT/'dev/host.json', pin, 2 << 20)
    native = rt.pinned(OUTPUT/'dev/native.json', host['native_report_identity'], 2 << 20)
    revision = host['producer_revision']; source = source_binding(ROOT/'jobs'/revision/ENTRY/'code', revision, HELPERS, ENTRY)
    proof = rt.pinned(OUTPUT/'dev/proof.json', native['proof_identity'], 4 << 20)
    validate_dev(host, native, proof, binding, source)
    return dict(host=pin, native=host['native_report_identity'], proof=native['proof_identity'], source=source)


def evaluate(phase, binding, banks, load_reference):
    from world_reward.endpoint_proposal_recall import evaluate_endpoint_proposals, aggregate_endpoint_fixed_slots
    values = binding['values']; rows = values['cohort']['records']; mapping = binding['mapping']
    chosen = rows[:32] if phase == 'dev' else rows[32:]
    categories = {c['id']: c['name'] for c in values['cohort']['categories']}; results = []; per_image = []
    # Caller has qualified every acquired public prediction, including the other
    # split, before this callback is allowed to read a single reference value.
    for row in chosen:
        if row['slot'] not in mapping: results.append(None); per_image.append(dict(slot=row['slot'], missing=True)); continue
        iid = mapping[row['slot']]['public_image_id']; boxes, objects = banks[iid]
        truth, kinds, ids = references(load_reference(row), row, categories)
        result = evaluate_endpoint_proposals(boxes, objects, truth, kinds, entity_ids=ids); results.append(result)
        per_image.append(dict(slot=row['slot'], missing=False, person_instances=kinds.count('human'), object_instances=kinds.count('object'),
            person_recall=result.person.metrics['recalls']['0.5']['all']['one_to_one_recall'],
            object_recall_128=result.objects[128].metrics['recalls']['0.5']['all']['one_to_one_recall']))
    metrics = aggregate_endpoint_fixed_slots(results)
    return dict(metrics=metrics, per_image=per_image, decision=metric_decision(phase, metrics))


def cpu(code, revision, phase, proof_pin, deadline):
    started = time.monotonic(); out = OUTPUT/phase; proof = rt.pinned(out/'proof.json', proof_pin, 4 << 20)
    report = dict(schema='world_reward.coco_endpoint_evaluation.v2', stage='saved_cpu_endpoint_recall', status='fail', phase=phase,
        producer_revision=revision, proof_identity=proof_pin, inputs=proof['binding']['pins'], models_loaded=0,
        gpu_used=False, predictions_modified=False, ownership_verified=False, quality_verified=False, adoption=False,
        all_native_banks_frozen_before_truth=False, reference_files_decoded=0)
    try:
        rt.require({x.name for x in Path('/sys/class/net').iterdir()} == {'lo'}
            and proof['phase'] == phase and proof['source']['producer_revision'] == revision
            and set(proof['native_files']) == set(NATIVE_FILES)
            and {str(x.relative_to(code)) for x in code.rglob('*') if x.is_file()} == set(NATIVE_FILES),
            'Offline CPU-only exact approved source/phase inventory')
        configuration(code, proof['source'])
        rt.require(proof['native_files'] == {n: proof['source']['helpers'][n] for n in set(NATIVE_FILES)}, 'Mounted source projection differs from whole host closure')
        for name, wanted in proof['native_files'].items(): rt.require(rt.identity(code/name, 2 << 20, empty=True) == wanted, 'Mounted immutable evaluator/helper changed')
        binding = proof['binding']; phase_refs = {f"reference_{i:06d}.json" for i in (range(32) if phase == 'dev' else range(32, 64))}
        rt.require({p.name for p in (DATA/'metadata').iterdir()} == {'cohort.json', 'report.json'} | phase_refs,
            'Only approved phase32 reference leaves can be mounted')
        binding['mapping'] = structure(binding['values'], binding['pins'])  # JSON object keys are strings; reconstruct authoritative integer slots.
        for name, path in receipt_paths().items():
            rt.require(rt.pinned(path, binding['pins'][name], 4 << 20) == binding['values'][name], 'Actual mounted receipt differs from authenticated host projection')
        bank.validate_report(binding['values']['native'], bank.configuration(code, binding['sources']['bank']),
            binding['revisions']['bank'], binding['values']['native']['proof_identity'], binding['values']['public'])
        for path, wanted in proof['frozen'].items():
            if Path(path).name not in phase_refs:
                rt.require(rt.identity(Path(path), 1 << 30) == wanted, 'Every original receipt/prediction seal required')
        if phase == 'reserved':
            dev = proof['approved_dev']; host = rt.pinned(OUTPUT/'dev/host.json', dev['host'], 2 << 20)
            native = rt.pinned(OUTPUT/'dev/native.json', dev['native'], 2 << 20)
            original = rt.pinned(OUTPUT/'dev/proof.json', dev['proof'], 4 << 20)
            rt.require(host['native_report_identity'] == dev['native'] and native['proof_identity'] == dev['proof'], 'DEV linked byte seals required')
            validate_dev(host, native, original, binding, dev['source'])
        else: rt.require(proof['approved_dev'] is None, 'DEV cannot consume a later-phase authorization')
        banks = {}
        for row in binding['values']['native']['images']:
            check(deadline); path = bank.OUTPUT/row['file']
            rt.require(rt.identity(path, 16 << 20) == row['identity'], 'Every complete bank must freeze before private read')
            banks[row['image_id']] = validate_npz(path, row)
        rt.require(set(banks) == {r['image_id'] for r in binding['values']['public']['images']}, 'No acquired bank missing')
        report['all_native_banks_frozen_before_truth'] = True
        for path, wanted in proof['frozen'].items():
            if Path(path).name in phase_refs: rt.require(rt.identity(Path(path), 4 << 20) == wanted, 'Approved reference seals required after all banks')
        def load(row):
            check(deadline); rt.require(row['reference_file'] in phase_refs, 'Other phase reference access forbidden')
            report['reference_files_decoded'] += 1
            return rt.pinned(DATA/'metadata'/row['reference_file'], row['reference_identity'], 4 << 20)
        report.update(evaluate(phase, binding, banks, load))
        for path, wanted in proof['frozen'].items(): rt.require(rt.identity(Path(path), 1 << 30) == wanted, 'Saved prediction/reference/source changed')
        for name, wanted in proof['native_files'].items(): rt.require(rt.identity(code/name, 2 << 20, empty=True) == wanted, 'Mounted evaluator/helper source changed')
        check(deadline); report.update(status='pass', source_inputs_predictions_references_rehashed_after=True)
    except BaseException as exc: report['error_type'] = bank.error(exc)
    report['elapsed_seconds'] = time.monotonic()-started; write(out/'native.json', report)
    return report


def dispatch(code, revision, phase, pins, revisions, dev_pin=None):
    started = time.monotonic(); deadline = started+180
    source = rt.source(ROOT, code, revision, ENTRY, HELPERS); policy = configuration(code, source)
    binding = authenticate(code, pins, revisions)
    dev = approved_dev(dev_pin, binding) if phase == 'reserved' else None
    rt.require(phase == 'reserved' or dev_pin is None, 'DEV cannot consume reserved authorization')
    out = OUTPUT/phase; rt.require(not out.exists(), 'Fresh once-only independent phase output required')
    if not OUTPUT.exists(): OUTPUT.mkdir(mode=0o700)
    out.mkdir(mode=0o700); owner = out.lstat(); name = 'world-reward-endpoint-eval-'+phase+'-'+revision[:12]
    host = dict(schema=policy['schema'], stage='endpoint_evaluation_host', status='fail', phase=phase,
        producer_revision=revision, source_binding=source, inputs=pins, producer_revisions=revisions,
        owned_cleanup_verified=False, outputs_sealed=False, quality_verified=False, ownership_verified=False, adoption=False)
    refs = binding['values']['cohort']['records'][:32] if phase == 'dev' else binding['values']['cohort']['records'][32:]
    proof_pin = None
    try:
        rt.require(bank.command(['docker', 'image', 'inspect', bank.IMAGE, '--format', '{{.Id}}'], deadline) == bank.IMAGE,
            'Actual original qualified CPU image required')
        rt.require(not bank.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Occupied owned CPU namespace')
        frozen = dict(binding['frozen'])
        for row in refs: frozen[str(DATA/'metadata'/row['reference_file'])] = row['reference_identity']
        if dev is not None:
            for leaf in ('host', 'native', 'proof'): frozen[str(OUTPUT/'dev'/(leaf+'.json'))] = dev[leaf]
        # No model/RGB bytes are read in child; they were authenticated on host.
        frozen = {n: pin for n, pin in frozen.items() if not n.endswith('.jpg')}
        files = {n: source['helpers'][n] for n in set(NATIVE_FILES)}
        proof = dict(binding=binding, source=source, native_files=files, frozen=frozen, phase=phase, approved_dev=dev)
        proof_pin = write(out/'proof.json', proof)
        mounts = {code/n for n in files} | {Path(n) for n in frozen}
        cmd = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(out/'.container.cid'),
            '--label', 'world-reward.job='+ENTRY, '--label', 'world-reward.revision='+revision, '--network', 'none',
            '--user', '0:0', '--memory', '6g', '--cpus', '4', '--pids-limit', '128', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m', '--entrypoint', '/usr/bin/env']
        for path in sorted(mounts):
            rt.canonical(path); rt.require(path.is_file() and ',' not in str(path), 'Only explicit readonly source/prediction/ref leaves')
            cmd += ['--mount', f'type=bind,src={path},dst={path},readonly']
        cmd += ['--mount', f'type=bind,src={out},dst={out}', bank.IMAGE, '-i', 'PATH=/opt/conda/bin:/usr/bin:/bin',
            'HOME=/tmp', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'WR_CODE='+str(code),
            'WR_CODE_REVISION='+revision, 'WR_ENDPOINT_EVAL_DEADLINE='+format(deadline-15, '.17g'), '/opt/conda/bin/python', '-I', '-B',
            str(code/'infra/coco_endpoint_evaluate.py'), '--native', '--phase', phase, '--proof-bytes', str(proof_pin['bytes']), '--proof-sha256', proof_pin['sha256']]
        result = subprocess.run(cmd, env=dict(PATH='/usr/bin:/bin', HOME='/nonexistent', DOCKER_HOST='unix://'+str(ROOT/'docker.sock')),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=max(.001, deadline-time.monotonic()-15), check=False)
        host['native_exit_status'] = result.returncode; host['native_report_identity'] = rt.identity(out/'native.json', 2 << 20)
        report = rt.pinned(out/'native.json', host['native_report_identity'], 2 << 20)
        rt.require(result.returncode == 0 and report['schema'] == policy['schema'] and report['stage'] == 'saved_cpu_endpoint_recall'
            and report['status'] == 'pass' and report['phase'] == phase
            and report['producer_revision'] == revision and report['proof_identity'] == proof_pin and report['inputs'] == pins
            and report['all_native_banks_frozen_before_truth'] is report['source_inputs_predictions_references_rehashed_after'] is True,
            'Actual complete saved CPU evaluation required')
        rt.require(report['metrics']['slots'] == 32 and report['models_loaded'] == 0 and report['gpu_used'] is False
            and report['ownership_verified'] is report['quality_verified'] is report['adoption'] is False,
            'Complete exact32 saved-only numerical scope required')
        rt.require(report['decision'] == metric_decision(phase, report['metrics']), 'Saved scientific decision differs from frozen metrics')
        host['decision'] = report['decision']
    except BaseException as exc: host['error_type'] = bank.error(exc)
    finally:
        try:
            current = out.lstat(); rt.require((current.st_dev, current.st_ino, current.st_uid) == (owner.st_dev, owner.st_ino, owner.st_uid), 'Owned output replaced')
            # Same owned label/CID discipline; CPU has the same immutable image.
            cleanup_cpu(out/'.container.cid', name, revision, time.monotonic()+10); host['owned_cleanup_verified'] = True
        except BaseException as exc: host.update(status='fail', cleanup_error_type=bank.error(exc))
        try:
            rt.require(authenticate(code, pins, revisions) == binding and rt.source(ROOT, code, revision, ENTRY, HELPERS) == source,
                'Full original source/runtime/inputs/predictions changed')
            if proof_pin is not None: rt.require(rt.identity(out/'proof.json', 4 << 20) == proof_pin, 'CPU proof changed')
            if 'native_report_identity' in host: rt.require(rt.identity(out/'native.json', 2 << 20) == host['native_report_identity'], 'CPU report changed')
            for row in refs: rt.require(rt.identity(DATA/'metadata'/row['reference_file'], 4 << 20) == row['reference_identity'], 'Phase references changed')
            if phase == 'reserved': rt.require(approved_dev(dev_pin, binding) == dev, 'Original DEV authorization changed')
            check(deadline); host['source_inputs_predictions_references_rehashed_after'] = True
            if 'error_type' not in host and host['owned_cleanup_verified'] and host.get('native_exit_status') == 0: host['status'] = 'pass'
        except BaseException as exc: host.update(status='fail', post_error_type=bank.error(exc))
        current = rt.canonical(out).lstat()
        rt.require((current.st_dev, current.st_ino, current.st_uid) == (owner.st_dev, owner.st_ino, owner.st_uid), 'Never seal a replaced output namespace')
        with (out/'host.json').open('xb') as stream:
            os.fchmod(stream.fileno(), 0o400)
            for leaf in out.iterdir():
                s = rt.canonical(leaf).lstat(); rt.require(leaf.name in ('host.json', 'native.json', 'proof.json', '.container.cid')
                    and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == owner.st_uid, 'Only owned evaluator receipt leaves')
                leaf.chmod(0o400)
            out.chmod(0o500); host['outputs_sealed'] = True; host['elapsed_seconds'] = time.monotonic()-started
            if time.monotonic() >= deadline: host.update(status='fail', post_error_type='TimeoutError')
            stream.write(encode(host)); stream.flush(); os.fsync(stream.fileno())
            if time.monotonic() >= deadline and host['status'] == 'pass':
                host.update(status='fail', post_error_type='TimeoutError'); stream.seek(0); stream.write(encode(host)); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
    return host


def cleanup_cpu(path, name, revision, deadline):
    rt.identity(path, 65, readonly=False); raw = path.read_bytes()
    rt.require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Actual owned CPU CID required'); cid = raw.decode().strip()
    found = bank.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'id='+cid], deadline)
    rt.require(found in ('', cid), 'Ambiguous owned CPU container')
    if found:
        actual = bank.command(['docker', 'inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'], deadline)
        rt.require(actual == bank.IMAGE+'|/'+name+'|'+ENTRY+'|'+revision, 'Never remove foreign CPU container')
        bank.command(['docker', 'rm', '-f', cid], deadline)
    rt.require(not bank.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', 'name=^/'+name+'$'], deadline), 'Owned CPU name survives')
    path.chmod(0o400)


def arguments(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--native', action='store_true')
    parser.add_argument('--phase', choices=('dev', 'reserved'), required=True)
    for name in PIN_NAMES+('dev', 'proof'):
        parser.add_argument('--'+name+'-bytes', type=int); parser.add_argument('--'+name+'-sha256')
    for name in ('census', 'acquisition', 'bank'): parser.add_argument('--'+name+'-revision')
    names = [x.split('=')[0] for x in argv if x.startswith('--')]
    rt.require(len(names) == len(set(names)), 'Duplicate arguments forbidden')
    a = parser.parse_args(argv)
    present = set()
    for name in PIN_NAMES+('dev', 'proof'):
        size, digest = getattr(a, name+'_bytes'), getattr(a, name+'_sha256')
        rt.require((size is None) == (digest is None), 'Independent byte/SHA pins must be supplied together')
        if size is not None:
            rt.require(0 < size <= 4 << 20 and re.fullmatch('[0-9a-f]{64}', digest), 'Bounded independent literal pin required')
            present.add(name)
    revisions = [getattr(a, n+'_revision') for n in ('census', 'acquisition', 'bank')]
    if a.native: rt.require(present == {'proof'} and revisions == [None]*3, 'Native accepts only pinned host proof and explicit phase')
    else:
        rt.require(present == set(PIN_NAMES) | ({'dev'} if a.phase == 'reserved' else set())
            and all(type(r) is str and re.fullmatch('[0-9a-f]{40}', r) for r in revisions), 'Host requires all independent producers/receipts and RESERVED DEV pin only')
    return a


def main():
    a = arguments(sys.argv[1:]); code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    rt.require(sys.platform == 'linux' and os.geteuid() == 0 and code == ROOT/'jobs'/revision/ENTRY/'code'
        and Path(__file__).resolve() == code/'infra/coco_endpoint_evaluate.py', 'Actual immutable Azure evaluator source')
    def pin(name): return dict(bytes=getattr(a, name+'_bytes'), sha256=getattr(a, name+'_sha256'))
    if a.native:
        deadline = float(os.environ['WR_ENDPOINT_EVAL_DEADLINE']); check(deadline)
        def expired(*_): raise TimeoutError('Inclusive CPU evaluator deadline')
        old = {s: signal.signal(s, expired) for s in (signal.SIGTERM, signal.SIGALRM)}
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        try: result = cpu(code, revision, a.phase, pin('proof'), deadline)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for s, handler in old.items(): signal.signal(s, handler)
    else:
        rt.require(os.uname().nodename == 'world-reward-ncc-h100-02' and a.proof_bytes is a.proof_sha256 is None, 'VM02 host independent pins required')
        result = dispatch(code, revision, a.phase, {n: pin(n) for n in PIN_NAMES},
            {n: getattr(a, n+'_revision') for n in ('census', 'acquisition', 'bank')}, pin('dev') if a.phase == 'reserved' else None)
    print(encode(dict(status=result['status'], phase=a.phase, decision=result.get('decision'), quality_verified=False)).decode().strip())
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(encode(dict(status='fail', error_type=bank.error(exc))).decode().strip()); raise SystemExit(1) from None
