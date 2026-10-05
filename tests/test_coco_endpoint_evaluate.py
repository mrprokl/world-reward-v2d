"""Tiny fake saved banks/reference projections, never actual photos or models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import subprocess
import sys

import numpy as np
import pytest

import coco_endpoint_evaluate as p


def seal(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw); path.chmod(0o400)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def config(): return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def metrics(passed):
    v = 1. if passed else 0.
    return dict(slots=32, primary=dict(iou=.5, object_budget=128, person_gate=.7, object_gate=.7,
        person_recall=v, object_recall=1., both_endpoint_gates_pass=passed),
        persons={'recalls': {'0.5': {'macro_fixed_slot_one_to_one_recall': v}}},
        objects={'128': {'recalls': {'0.5': {'macro_fixed_slot_one_to_one_recall': 1.}}}})


def binding_fixture(acquired=range(64)):
    pins = {n: dict(bytes=1, sha256=n[0]*64) for n in p.PIN_NAMES}
    rows, status, maps, images = [], [], [], []
    for i in range(64):
        iid = i+1000; image = dict(id=iid, license=4, width=20, height=16)
        row = dict(slot=i, split='DEV' if i < 32 else 'RESERVED', image_id=iid, photo_id=str(i), image=image,
            reference_file=f'reference_{i:06d}.json', reference_identity=dict(bytes=1, sha256='f'*64))
        rows.append(row); s = dict(slot=i, split=row['split'], image_id=iid, status='unavailable')
        if i in acquired:
            pin = dict(bytes=10, sha256=f'{i:064x}'); opaque = hashlib.sha256(('world_reward.coco_endpoint_v2/'+f'{iid:012d}').encode()).hexdigest()[:32]
            s.update(status='acquired', image_pin=pin, jpeg_header=dict(width=20, height=16))
            maps.append(dict(slot=i, public_image_id=opaque, public_file=f'image_{i:06d}.jpg', image_pin=pin))
            images.append(dict(image_id=opaque, file=f'image_{i:06d}.jpg', **pin, width=20, height=16))
        status.append(s)
    cohort = dict(schema='world_reward.coco_endpoint_cohort.v2', hash_namespace='world_reward.coco_endpoint_v2/',
        freeze_before_rgb=True, no_replacements=True, records=rows, categories=[dict(id=1, name='person'), dict(id=2, name='tool')])
    common = dict(schema='world_reward.coco_endpoint_prepare.v2', status='pass', source_and_inputs_rehashed_after=True,
        closed_individual_reference_values_read=False,
        outputs_sealed=True, freeze_before_rgb=True, cohort_identity=pins['cohort'])
    census = dict(common, phase='census')
    acquisition = dict(common, phase='acquire', census_report_identity=pins['census'], public_inputs_identity=pins['public'],
        selected_slots=rows, reference_values_read_for_acquisition=False, replacement_count=0, retry_count=0,
        records=status, public_mappings=maps, counts=dict(slots=64, acquired=len(images), missing=64-len(images),
            acquired_DEV=sum(i < 32 for i in acquired), acquired_RESERVED=sum(i >= 32 for i in acquired)))
    values = dict(cohort=cohort, census=census, acquisition=acquisition,
        public=dict(schema='world_reward.rgb_proposal_inputs.v1', images=images), native=dict(images=[]))
    result = dict(values=values, pins=pins, mapping={m['slot']: m for m in maps})
    return result


def truth(row):
    return dict(annotations=[dict(id=row['slot']*10+i, image_id=row['image_id'], iscrowd=0, area=4,
        category_id=1 if i < 2 else 2, bbox=[i*3, 0, 2, 2]) for i in range(4)])


def observation():
    # Native normalized boxes encode two original pixel targets exactly in a20 square.
    b = np.zeros((3600, 4), dtype=np.float32)
    b[0] = [.35, .05, .1, .1]; b[1] = [.5, .05, .1, .1]
    corners = np.concatenate((b[:, :2]-b[:, 2:]/np.float32(2), b[:, :2]+b[:, 2:]/np.float32(2)), axis=1)*np.float32(20)
    return p.Owlv2ObjectObservations(0, (16, 20), (60, 60), np.arange(3600, dtype=np.int64),
        b, np.arange(3600, 0, -1, dtype=np.float32), corners)


def arrays_and_row(tmp_path):
    arrays = dict(person_raw_boxes=np.asarray([[0, 0, 2, 2], [3, 0, 5, 2]], dtype=np.float32),
        person_raw_scores=np.asarray([.8, .9], dtype=np.float32), person_raw_labels=np.asarray(['person', 'person']),
        person_model_pred_boxes=np.full((1, 900, 4), .5, dtype=np.float32),
        person_model_input_ids=np.asarray([[101, 102]], dtype=np.int64), person_model_attention_mask=np.ones((1, 2), dtype=np.int64))
    logits = np.full((1, 900, 256), -np.inf, dtype=np.float32); logits[:, :, :2] = 0; arrays['person_model_logits'] = logits
    replay = p.bank.gdi.retained_bank(arrays['person_raw_boxes'], arrays['person_raw_scores'], ['person', 'person'], 'a'*32, 'person', 20, 16)
    arrays.update({'person_'+k: v for k, v in replay.items()})
    o = observation(); arrays.update(owl_patch_ids=o.patch_ids, owl_objectness_logits=o.objectness_logits,
        owl_boxes_padded_normalized_cxcywh=o.boxes_padded_normalized_cxcywh, owl_boxes_original_xyxy=o.boxes_original_xyxy,
        image_size=np.asarray([16, 20], np.int64), original_frame_index=np.asarray(0, np.int64))
    saved = p.bank.save_bank(tmp_path, 0, arrays)
    row = dict(saved, image_id='a'*32, image_size=[16, 20], person_ids=replay['retained_ids'].tolist(),
        person_retained_rows=2, person_postprocessor_rows=2)
    return tmp_path/saved['file'], arrays, row


def test_real_saved_npz_all17_arrays_native_nms_and_owl_inverse(tmp_path):
    path, arrays, row = arrays_and_row(tmp_path)
    persons, objects = p.validate_npz(path, row)
    assert len(p.ARRAY_NAMES) == 17 and np.array_equal(persons, arrays['person_retained_boxes'])
    assert objects.patch_ids.shape == (3600,) and objects.original_frame_index == 0
    assert np.isneginf(arrays['person_model_logits'][:, :, 2:]).all()


@pytest.mark.parametrize('fault', ['missing', 'extra', 'patch', 'grid', 'frame', 'dtype', 'masked', 'retained', 'rows'])
def test_malformed_bank_rejected_before_reference_even_with_updated_outer_pin(tmp_path, fault):
    path, arrays, row = arrays_and_row(tmp_path)
    if fault == 'missing': del arrays['owl_patch_ids']
    elif fault == 'extra': arrays['selected_owner'] = np.asarray(1)
    elif fault == 'patch': arrays['owl_patch_ids'] = arrays['owl_patch_ids'][::-1]
    elif fault == 'grid': arrays['image_size'] = np.asarray([16, 21], np.int64)
    elif fault == 'frame': arrays['original_frame_index'] = np.asarray(1, np.int64)
    elif fault == 'dtype': arrays['owl_objectness_logits'] = arrays['owl_objectness_logits'].astype(np.float64)
    elif fault == 'masked': arrays['person_model_logits'][:, :, 2] = 0
    elif fault == 'retained': arrays['person_retained_raw_slots'] = arrays['person_retained_raw_slots'][::-1]
    elif fault == 'rows': row['person_postprocessor_rows'] = 3
    path.chmod(0o600); path.unlink(); saved = p.bank.save_bank(tmp_path, 0, arrays); row.update(saved)
    with pytest.raises(ValueError): p.validate_npz(path, row)


def test_all64_structure_missing_slots_and_minimumdev():
    binding = binding_fixture(range(24))
    mapping = p.structure(binding['values'], binding['pins'])
    assert len(mapping) == 24 and 23 in mapping and 24 not in mapping
    short = binding_fixture(range(23))
    with pytest.raises(ValueError): p.structure(short['values'], short['pins'])


@pytest.mark.parametrize('fault', ['partial', 'mapping', 'missing', 'cohort', 'source', 'public', 'category'])
def test_complete_structure_rejects_partial_mapping_or_unfrozen_receipts(fault):
    binding = binding_fixture(); v = binding['values']
    if fault == 'partial': v['acquisition']['records'].pop()
    elif fault == 'mapping': v['acquisition']['public_mappings'][0]['public_image_id'] = '0'*32
    elif fault == 'missing': v['acquisition']['records'][0]['status'] = 'unavailable'
    elif fault == 'cohort': v['cohort']['records'][0]['split'] = 'RESERVED'
    elif fault == 'source': v['census']['outputs_sealed'] = False
    elif fault == 'public': v['public']['images'].reverse()
    elif fault == 'category': v['cohort']['categories'].append(dict(id=3, name='person'))
    with pytest.raises(ValueError): p.structure(v, binding['pins'])


def test_dev_gate_allfixed32_missingzero_no_reserved_callback():
    binding = binding_fixture(range(24)); banks = {m['public_image_id']: (np.asarray([[0, 0, 2, 2], [3, 0, 5, 2]], np.float64), observation()) for m in binding['mapping'].values()}
    read = []
    def load(row): read.append(row['slot']); return truth(row)
    result = p.evaluate('dev', binding, banks, load)
    assert read == list(range(24)) and result['metrics']['slots'] == 32 and result['metrics']['missing_slots'] == 8
    assert result['metrics']['primary']['person_recall'] == .75
    assert result['metrics']['primary']['object_recall'] == .75 and result['decision'] == 'DEV_CAPACITY_PASS'
    assert result['metrics']['objects']['3600']['recalls']['0.5']['macro_fixed_slot_one_to_one_recall'] == .75


def test_dev_failure_empty_person_preserves_denominator_and_no_reserved_read():
    binding = binding_fixture(); banks = {m['public_image_id']: (np.empty((0, 4)), observation()) for m in binding['mapping'].values()}; read = []
    result = p.evaluate('dev', binding, banks, lambda row: (read.append(row['slot']) or truth(row)))
    assert result['decision'] == 'DEV_CAPACITY_FAIL' and read == list(range(32))
    assert result['metrics']['primary']['person_recall'] == 0 and result['metrics']['primary']['object_recall'] == 1


def test_duplicate_reference_boxes_distinctannotationids_not_dropped():
    row = binding_fixture()['values']['cohort']['records'][0]; value = truth(row)
    value['annotations'][1]['bbox'] = value['annotations'][0]['bbox'].copy()
    boxes, kinds, ids = p.references(value, row, {1: 'person', 2: 'tool'})
    assert len(boxes) == len(set(ids)) == 4 and np.array_equal(boxes[0], boxes[1])
    value['annotations'][1]['id'] = value['annotations'][0]['id']
    with pytest.raises(ValueError): p.references(value, row, {1: 'person', 2: 'tool'})


def test_whole_prediction_validation_precedes_any_truth(tmp_path, monkeypatch):
    binding = binding_fixture(); code = tmp_path/'code'; code.mkdir(); out = tmp_path/'results/dev'; out.mkdir(parents=True)
    monkeypatch.setattr(p, 'OUTPUT', out.parent); monkeypatch.setattr(p.bank, 'OUTPUT', tmp_path/'banks')
    p.bank.OUTPUT.mkdir(); metadata = tmp_path/'metadata'; metadata.mkdir()
    monkeypatch.setattr(p, 'DATA', tmp_path)
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, 'iterdir', lambda path: iter([Path('lo')]) if path == Path('/sys/class/net') else original_iterdir(path))
    for name in ['cohort.json', 'report.json']+[f'reference_{i:06d}.json' for i in range(32)]: seal(metadata/name, b'{}')
    rows = [dict(image_id=f'{i:032x}', file=f'image_{i:06d}.npz', identity=seal(p.bank.OUTPUT/f'image_{i:06d}.npz', b'bank')) for i in range(64)]
    binding['values']['native']['images'] = rows
    proof = dict(binding=binding, phase='dev', source={'producer_revision': 'a'*40, 'helpers': {}}, native_files={}, frozen={}, approved_dev=None)
    calls = []
    monkeypatch.setattr(p, 'NATIVE_FILES', ()); monkeypatch.setattr(p, 'structure', lambda *a: binding['mapping'])
    monkeypatch.setattr(p, 'configuration', lambda *a: config())
    binding.update(sources={'bank': {}}, revisions={'bank': 'a'*40})
    binding['values']['native']['proof_identity'] = {'bytes': 1, 'sha256': 'a'*64}
    pin = seal(out/'proof.json', p.encode(proof))
    monkeypatch.setattr(p, 'receipt_paths', lambda: {})
    monkeypatch.setattr(p.bank, 'configuration', lambda *a: {})
    monkeypatch.setattr(p.bank, 'validate_report', lambda *a: None)
    def validate(path, row):
        calls.append(row['file'])
        if len(calls) == 64: raise ValueError('last bank incomplete')
        return np.empty((0, 4)), observation()
    monkeypatch.setattr(p, 'validate_npz', validate)
    monkeypatch.setattr(p, 'evaluate', lambda *a: (_ for _ in ()).throw(AssertionError('must not read truth')))
    result = p.cpu(code, 'a'*40, 'dev', pin, time.monotonic()+10)
    assert result['status'] == 'fail' and len(calls) == 64 and result['reference_files_decoded'] == 0


def test_config_metric_source_pins_match_and_no_changed_scoring():
    cfg = config(); root = Path(__file__).resolve().parents[1]
    assert cfg['person_gate'] == cfg['object_gate'] == .7 and cfg['object_budgets'] == [32, 128, 3600]
    for name, wanted in cfg['helper_pins'].items():
        raw = (root/name).read_bytes(); assert wanted == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert cfg['minimum_acquired_development_images_before_reference_access'] == 24 and cfg['budget_seconds'] == 180


def dispatch_fixture(tmp_path, monkeypatch, *, native_fail=False):
    root = tmp_path/'root'; code = root/'code'; code.mkdir(parents=True)
    output = root/'results/evaluation'; output.parent.mkdir(); metadata = tmp_path/'metadata'; metadata.mkdir()
    monkeypatch.setattr(p, 'ROOT', root); monkeypatch.setattr(p, 'OUTPUT', output); monkeypatch.setattr(p, 'DATA', tmp_path)
    binding = binding_fixture(); binding.update(sources={}, revisions={'census': 'a'*40, 'acquisition': 'a'*40, 'bank': 'a'*40}, frozen={})
    for i in range(64):
        row = binding['values']['cohort']['records'][i]
        row['reference_identity'] = seal(metadata/row['reference_file'], p.encode(truth(row)))
    source = {'producer_revision': 'a'*40, 'helpers': {}}
    for name in set(p.NATIVE_FILES): source['helpers'][name] = seal(code/name, b'code')
    monkeypatch.setattr(p.rt, 'source', lambda *a: source); monkeypatch.setattr(p, 'configuration', lambda *a: config())
    monkeypatch.setattr(p, 'authenticate', lambda *a: deepcopy(binding)); commands = []
    monkeypatch.setattr(p, 'cleanup_cpu', lambda *a: commands.append(['cleanup']))
    def command(args, deadline):
        commands.append(args); return p.bank.IMAGE if args[1:3] == ['image', 'inspect'] else ''
    monkeypatch.setattr(p.bank, 'command', command)
    def run(cmd, **kwargs):
        commands.append(cmd); out = output/'dev'; (out/'.container.cid').write_bytes(b'a'*64)
        pin = p.rt.identity(out/'proof.json', 4 << 20)
        report = dict(schema=config()['schema'], stage='saved_cpu_endpoint_recall', status='fail' if native_fail else 'pass',
            phase='dev', producer_revision='a'*40, proof_identity=pin, inputs=binding['pins'],
            all_native_banks_frozen_before_truth=True, source_inputs_predictions_references_rehashed_after=True,
            metrics=metrics(False), models_loaded=0, gpu_used=False, ownership_verified=False, quality_verified=False,
            adoption=False, decision='DEV_CAPACITY_FAIL')
        seal(out/'native.json', p.encode(report)); return SimpleNamespace(returncode=1 if native_fail else 0)
    monkeypatch.setattr(p.subprocess, 'run', run)
    return code, binding, output, commands


def test_actual_mock_host_devfailure_is_scientific_not_technical_and_no_reserved_mount(tmp_path, monkeypatch):
    code, binding, output, commands = dispatch_fixture(tmp_path, monkeypatch)
    result = p.dispatch(code, 'a'*40, 'dev', binding['pins'], binding['revisions'])
    assert result['status'] == 'pass' and result['decision'] == 'DEV_CAPACITY_FAIL'
    assert result['owned_cleanup_verified'] and result['outputs_sealed']
    cmd = next(c for c in commands if c[:2] == ['docker', 'run'])
    mounts = [cmd[i+1] for i, v in enumerate(cmd) if v == '--mount']
    reference = [m for m in mounts if '/reference_' in m]
    assert len(reference) == 32 and all(f'reference_{i:06d}.json' in '\n'.join(reference) for i in range(32))
    assert not any('reference_000032.json' in m or m.endswith('.jpg,readonly') for m in mounts)
    assert '--gpus' not in cmd and cmd[cmd.index('--memory')+1] == '6g' and '-I' in cmd and '-B' in cmd
    assert (output/'dev').stat().st_mode & 0o777 == 0o500


@pytest.mark.parametrize('fault', ['native', 'posthash', 'late', 'report', 'proof'])
def test_mock_host_cleanup_and_fail_no_falsepass(tmp_path, monkeypatch, fault):
    code, binding, output, commands = dispatch_fixture(tmp_path, monkeypatch, native_fail=fault == 'native')
    run = p.subprocess.run; clock = [0.]
    if fault == 'late': monkeypatch.setattr(p.time, 'monotonic', lambda: clock[0])
    def changed(*args, **kwargs):
        result = run(*args, **kwargs)
        if fault == 'posthash': monkeypatch.setattr(p, 'authenticate', lambda *a: {})
        elif fault == 'late': clock[0] = 181.
        elif fault in ('report', 'proof'):
            path = output/'dev'/('native.json' if fault == 'report' else 'proof.json')
            path.chmod(0o600); seal(path, b'changed')
        return result
    monkeypatch.setattr(p.subprocess, 'run', changed)
    result = p.dispatch(code, 'a'*40, 'dev', binding['pins'], binding['revisions'])
    assert result['status'] == 'fail' and result['owned_cleanup_verified'] and ['cleanup'] in commands
    assert json.loads((output/'dev/host.json').read_bytes())['status'] == 'fail'


def test_reserved_invalid_dev_authorization_fails_before_out_or_cpu(tmp_path, monkeypatch):
    code, binding, output, commands = dispatch_fixture(tmp_path, monkeypatch)
    def reject(*a): raise ValueError('DEV rejected')
    monkeypatch.setattr(p, 'approved_dev', reject)
    with pytest.raises(ValueError): p.dispatch(code, 'a'*40, 'reserved', binding['pins'], binding['revisions'], dict(bytes=1, sha256='f'*64))
    assert not output.exists() and not commands


def test_reserved_cpu_invocation_mounts_only_its32_refs_and_three_dev_seals(tmp_path, monkeypatch):
    code, binding, output, commands = dispatch_fixture(tmp_path, monkeypatch)
    dev = {}
    for n in ('host', 'native', 'proof'): dev[n] = seal(output/'dev'/(n+'.json'), b'{}')
    dev['source'] = {}; monkeypatch.setattr(p, 'approved_dev', lambda *a: dev)
    def run(cmd, **kwargs):
        commands.append(cmd); out = output/'reserved'; (out/'.container.cid').write_bytes(b'a'*64)
        pin = p.rt.identity(out/'proof.json', 4 << 20)
        report = dict(schema=config()['schema'], stage='saved_cpu_endpoint_recall', status='pass', phase='reserved',
            producer_revision='a'*40, proof_identity=pin, inputs=binding['pins'], all_native_banks_frozen_before_truth=True,
            source_inputs_predictions_references_rehashed_after=True, metrics=metrics(True), models_loaded=0, gpu_used=False,
            ownership_verified=False, quality_verified=False, adoption=False, decision='RESERVED_CAPACITY_PASS')
        seal(out/'native.json', p.encode(report)); return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess, 'run', run)
    result = p.dispatch(code, 'a'*40, 'reserved', binding['pins'], binding['revisions'], dev['host'])
    assert result['status'] == 'pass' and result['decision'] == 'RESERVED_CAPACITY_PASS'
    cmd = next(c for c in commands if c[:2] == ['docker', 'run']); mounts = [cmd[i+1] for i, v in enumerate(cmd) if v == '--mount']
    refs = [m for m in mounts if '/reference_' in m]
    assert len(refs) == 32 and all(f'reference_{i:06d}.json' in '\n'.join(refs) for i in range(32, 64))
    assert not any('reference_000000.json' in m for m in mounts)
    assert all(any(str(output/'dev'/(n+'.json')) in m and m.endswith(',readonly') for m in mounts) for n in ('host', 'native', 'proof'))
    assert cmd[cmd.index('--name')+1] == 'world-reward-endpoint-eval-reserved-'+'a'*12


def test_wrong_cpu_image_prevents_any_container_or_private_reference_mount(tmp_path, monkeypatch):
    code, binding, output, commands = dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(p.bank, 'command', lambda *a: 'wrong-image')
    result = p.dispatch(code, 'a'*40, 'dev', binding['pins'], binding['revisions'])
    assert result['status'] == 'fail' and not any(c[:2] == ['docker', 'run'] for c in commands)


def test_cleanup_failure_still_rehashes_all_source_inputs_no_falsepass(tmp_path, monkeypatch):
    code, binding, output, commands = dispatch_fixture(tmp_path, monkeypatch)
    count = []; original = p.authenticate
    monkeypatch.setattr(p, 'authenticate', lambda *a: (count.append('hash') or original(*a)))
    def failed(*a): raise ValueError('owned cleanup rejected')
    monkeypatch.setattr(p, 'cleanup_cpu', failed)
    result = p.dispatch(code, 'a'*40, 'dev', binding['pins'], binding['revisions'])
    assert len(count) == 2 and result['status'] == 'fail' and not result['owned_cleanup_verified']
    assert result['source_inputs_predictions_references_rehashed_after'] and result['cleanup_error_type'] == 'ValueError'


@pytest.mark.parametrize('stdout', ['', 'a'*64, 'foreign'])
def test_real_cpu_cleanup_never_removes_foreign_container(tmp_path, monkeypatch, stdout):
    path = tmp_path/'cid'; path.write_bytes(b'a'*64); commands = []
    def command(args, *_):
        commands.append(args)
        if args[:2] == ['docker', 'inspect']: return 'foreign-image|/wrong|wrong|wrong'
        if 'id='+'a'*64 in args: return stdout
        return ''
    monkeypatch.setattr(p.bank, 'command', command)
    if stdout:
        with pytest.raises(ValueError): p.cleanup_cpu(path, 'name', 'a'*40, time.monotonic()+10)
        assert not any(a[:2] == ['docker', 'rm'] for a in commands)
    else:
        p.cleanup_cpu(path, 'name', 'a'*40, time.monotonic()+10)
        assert path.stat().st_mode & 0o777 == 0o400


def dev_fixture(tmp_path, monkeypatch):
    output = tmp_path/'evaluation'; folder = output/'dev'; folder.mkdir(parents=True)
    monkeypatch.setattr(p, 'OUTPUT', output)
    binding = binding_fixture(); binding.update(sources={'bank': {}}, revisions={n: 'a'*40 for n in ('census', 'acquisition', 'bank')})
    source = dict(producer_revision='a'*40, helpers={})
    proof = dict(phase='dev', source=source, approved_dev=None, binding=binding)
    proof_pin = seal(folder/'proof.json', p.encode(proof))
    native = dict(schema=config()['schema'], stage='saved_cpu_endpoint_recall', status='pass', phase='dev',
        producer_revision='a'*40, inputs=binding['pins'], decision='DEV_CAPACITY_PASS', metrics=metrics(True),
        all_native_banks_frozen_before_truth=True, source_inputs_predictions_references_rehashed_after=True,
        models_loaded=0, gpu_used=False, predictions_modified=False, ownership_verified=False, quality_verified=False,
        adoption=False, reference_files_decoded=32, proof_identity=proof_pin)
    native_pin = seal(folder/'native.json', p.encode(native))
    host = dict(schema=config()['schema'], stage='endpoint_evaluation_host', status='pass', phase='dev',
        producer_revision='a'*40, producer_revisions=binding['revisions'], source_binding=source, inputs=binding['pins'],
        outputs_sealed=True, owned_cleanup_verified=True, source_inputs_predictions_references_rehashed_after=True,
        native_exit_status=0, native_report_identity=native_pin, decision='DEV_CAPACITY_PASS',
        ownership_verified=False, quality_verified=False, adoption=False)
    pin = seal(folder/'host.json', p.encode(host)); seal(folder/'.container.cid', b'a'*64); folder.chmod(0o500)
    monkeypatch.setattr(p, 'source_binding', lambda *a: source)
    return binding, host, native, proof, source, pin


def test_approved_dev_requires_actual_three_byte_seals_and_same_source(tmp_path, monkeypatch):
    binding, host, native, proof, source, pin = dev_fixture(tmp_path, monkeypatch)
    got = p.approved_dev(pin, binding)
    assert got == dict(host=pin, native=host['native_report_identity'], proof=native['proof_identity'], source=source)
    path = p.OUTPUT/'dev/native.json'; path.chmod(0o600); path.write_bytes(b'changed'); path.chmod(0o400)
    with pytest.raises(ValueError): p.approved_dev(pin, binding)


@pytest.mark.parametrize('fault', ['hostfail', 'nativefail', 'gate', 'metric', 'partial', 'source', 'pins', 'model', 'quality', 'count'])
def test_reserved_contradictory_or_incomplete_dev_never_authorizes(tmp_path, monkeypatch, fault):
    binding, host, native, proof, source, _ = dev_fixture(tmp_path, monkeypatch)
    if fault == 'hostfail': host['status'] = 'fail'
    elif fault == 'nativefail': native['status'] = 'fail'
    elif fault == 'gate': native['decision'] = 'DEV_CAPACITY_FAIL'
    elif fault == 'metric': native['metrics']['primary']['person_recall'] = .69
    elif fault == 'partial': native['metrics']['slots'] = 31
    elif fault == 'source': proof['source'] = {}
    elif fault == 'pins': proof['binding'] = deepcopy(binding); proof['binding']['pins']['public']['sha256'] = 'a'*64
    elif fault == 'model': native['models_loaded'] = 1
    elif fault == 'quality': native['quality_verified'] = True
    elif fault == 'count': native['reference_files_decoded'] = 31
    with pytest.raises((ValueError, KeyError)): p.validate_dev(host, native, proof, binding, source)


def host_arguments(phase='dev'):
    result = ['--phase', phase]
    for n in p.PIN_NAMES+ (('dev',) if phase == 'reserved' else ()):
        result += ['--'+n+'-bytes', '1', '--'+n+'-sha256', 'a'*64]
    for n in ('census', 'acquisition', 'bank'): result += ['--'+n+'-revision', 'a'*40]
    return result


def test_cli_independent_allornone_phase_pins_and_native_boundary():
    assert p.arguments(host_arguments()).phase == 'dev'
    assert p.arguments(host_arguments('reserved')).dev_bytes == 1
    native = ['--native', '--phase', 'dev', '--proof-bytes', '1', '--proof-sha256', 'a'*64]
    assert p.arguments(native).native
    for args in (host_arguments()+['--dev-bytes', '1', '--dev-sha256', 'a'*64],
                 host_arguments('reserved')[:-6], native+['--bank-revision', 'a'*40],
                 ['--phase', 'reserved'], host_arguments()+['--public-bytes=2'], native[:-2]):
        with pytest.raises((ValueError, SystemExit)): p.arguments(args)


def test_isolated_native_leaf_import_no_models_or_metadata_import(tmp_path):
    root = Path(__file__).resolve().parents[1]; code = tmp_path/'code'
    for name in set(p.NATIVE_FILES): seal(code/name, (root/name).read_bytes())
    script = ("import importlib.util,sys; spec=importlib.util.spec_from_file_location('endpoint_cpu',sys.argv[1]); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'coco_endpoint_prepare' not in sys.modules; "
        "assert len(m.ARRAY_NAMES)==17; print('isolated-native-import-pass')")
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(code/'infra/coco_endpoint_evaluate.py')],
        capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'isolated-native-import-pass'
    assert not list(code.rglob('*.pyc'))
