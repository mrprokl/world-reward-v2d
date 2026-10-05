"""Tiny manufactured cohort/provenance controls, no RGB/model/network/GPU."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    s = importlib.util.spec_from_file_location('wr_hoi_cohort_tests', REPO/'infra/hoi_detr_model_qualify.py')
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


@pytest.fixture
def p(gate): return json.loads((REPO/gate.COHORT_PROTOCOL).read_bytes())


def test_protocol_pin_scope_and_original_numeric_policy(gate, p):
    raw = (REPO/gate.COHORT_PROTOCOL).read_bytes()
    assert gate.COHORT_PROTOCOL_PIN == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    assert gate.selected_protocol_pin(p) == gate.COHORT_PROTOCOL_PIN
    old = json.loads((REPO/gate.IMAGE_PROTOCOL).read_bytes())
    for k in ('runtime', 'acquisition', 'fairscale', 'native_config', 'compatibility_patch', 'checkpoint_buffers', 'seed', 'amp', 'tf32', 'import_wheels', 'qualified_model'):
        assert p[k] == old[k]
    assert p['external_cohort']['slots'] == 15 and len(p['external_cohort']['inputs']) == 8
    assert all(p[k] is False for k in gate.FLAGS)
    assert 'external_RGB' not in p and 'procedural_RGB' not in p


def fixture(gate, p):
    c = p['external_cohort']; folder = Path(c['root']); rows = []
    values = {}; images = {}
    by_slot = {r['cohort_slot']: r for r in c['inputs']}
    for slot in range(c['slots']):
        if slot not in by_slot:
            rows.append(dict(image_id='f'*15+format(slot, 'x'), status='unavailable', reference_geometry_read=False)); continue
        r = by_slot[slot]; iid = r['image_id']; rp = dict(bytes=1, sha256='b'*64)
        meta = dict(Rotation='0.0', OriginalLandingURL='https://www.flickr.com/photos/test/123', Author='Creator')
        row = dict(image_id=iid, status='acquired', reference_geometry_read=False, image_pin=r['identity'], rights_pin=rp,
                   publisher_original_metadata=meta, publisher_md5_matched=True, rotation='0.0', creator_grant_verified=True)
        rows.append(row)
        values[str(folder/iid/'rights.json')] = dict(image_id=iid, license='CC-BY-2.0', individual_creator_declaration_verified=True,
            creator_ld_json=dict(license='https://creativecommons.org/licenses/by/2.0/', acquireLicensePage=meta['OriginalLandingURL'], author=dict(name='Creator')))
        images[r['path']] = r['identity']
    values[str(folder/'manifest.json')] = dict(excluded_QA_image=c['excluded_QA_image'], reference_geometry_read=False,
        challenge_inputs_used=False, local_heavy_transfer=False, records=rows)
    prior = p['qualified_model']; h = dict(status='pass', producer_revision=prior['producer_revision'], native_report_identity=prior['native'],
        phase='complete', actual_model_qualified=True, source_binding={'old': 'source'}, **{k: False for k in gate.FLAGS})
    h.update({k: True for k in ('source_rehashed_after', 'inputs_rehashed_after', 'image_unchanged', 'owned_containers_removed', 'owned_overlay_removed')})
    values[str(gate.ROOT/prior['path'])] = h
    values[str((gate.ROOT/prior['path']).parent/'native.json')] = dict(status='pass', native_forward_calls=1,
        strict_checkpoint=dict(keys=1796, strict=True, weights_only=True), **{k: False for k in gate.FLAGS})
    return values, images


@pytest.mark.parametrize('fault', [None, 'qa', 'reference', 'rights', 'rotation', 'replace', 'missing_status', 'oldcleanup', 'strict'])
def test_host_cohort_fail_closed_before_inference(gate, p, monkeypatch, fault):
    p = copy.deepcopy(p); values, images = fixture(gate, p)
    manifest = values[p['external_cohort']['root']+'/manifest.json']; acquired = next(r for r in manifest['records'] if r['status'] == 'acquired')
    if fault == 'qa': acquired['image_id'] = p['external_cohort']['excluded_QA_image']
    if fault == 'reference': manifest['reference_geometry_read'] = True
    if fault == 'rights': values[p['external_cohort']['root']+'/'+acquired['image_id']+'/rights.json']['creator_ld_json']['license'] = 'CC-BY-NC'
    if fault == 'rotation': acquired['rotation'] = ''
    if fault == 'replace': p['external_cohort']['inputs'][0]['cohort_slot'] = 2
    if fault == 'missing_status': next(r for r in manifest['records'] if r['status'] != 'acquired')['status'] = 'model_error_as_empty'
    if fault == 'oldcleanup': values[str(gate.ROOT/p['qualified_model']['path'])]['owned_containers_removed'] = False
    if fault == 'strict': values[str((gate.ROOT/p['qualified_model']['path']).parent/'native.json')]['strict_checkpoint']['strict'] = False
    calls = []
    monkeypatch.setattr(gate, 'exact', lambda rt, path, pin, limit: calls.append(str(path)) if images[str(path)] == pin else pytest.fail('Wrong exact RGB pin'))
    monkeypatch.setattr(gate, 'old_source', lambda *args: calls.append('old_source'))
    rt = SimpleNamespace(pinned=lambda path, *args: values[str(path)])
    if fault:
        with pytest.raises(ValueError): gate.authenticate_external_cohort(rt, p)
    else:
        result = gate.authenticate_external_cohort(rt, p)
        assert len(result) == 19 and calls[-1] == 'old_source' and len(calls) == 9


def test_gpu_exact_jpegs_only_no_rights_metadata_or_reference_mount(gate, p, tmp_path):
    proof = dict(runtime=dict(image=dict(Id='sha256:'+'f'*64)), pin=dict(bytes=1, sha256='a'*64))
    for phase in ('overlay', 'model'):
        _, _, mounts, args = gate.container_plan(tmp_path/'code', tmp_path/'out', 'a'*40, p, proof, phase, 1800)
        external = [a for a, b, ro in mounts if str(a).startswith(p['external_cohort']['root'])]
        assert external == ([Path(r['path']) for r in p['external_cohort']['inputs']] if phase == 'model' else [])
        assert all(a.suffix == '.jpg' for a in external)
        assert '--external-cohort' in args and '--external-rgb' not in args
        assert all(ro for a, b, ro in mounts if a in external)


def test_batch_load_once_empty_outputs_valid_and_scope_flag_propagates(gate):
    t = ast.parse((REPO/gate.HELPERS[0]).read_bytes()); f = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_model')
    loop = next(n for n in f.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'c')
    text = ast.unparse(loop)
    assert 'infer_hoi_detr_frame(model, rgb' in text and 'np.savez_compressed' in text
    assert 'build_detector' not in text and 'torch.load' not in text
    assert 'if external:' in text and 'minimum_native_hand_object_pairs' in text
    assert 'continue' not in text and 'except' not in text
    main = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    assert 'external_cohort=args.external_cohort' in ast.unparse(main)


def test_large_scope_complete_qualified_inputs_and_bounded_sidecar(gate):
    raw = (REPO/gate.COHORT128_PROTOCOL).read_bytes(); p = json.loads(raw)
    assert gate.COHORT128_PROTOCOL_PIN == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    c = p['external_cohort']
    assert (c['slots'], c['acquired'], c['missing']) == (128, 61, 67)
    assert len(c['inputs']) == len({r['image_id'] for r in c['inputs']}) == 61
    assert len({r['cohort_slot'] for r in c['inputs']}) == 61
    assert all(0 <= r['cohort_slot'] < 128 and r['original_frame_index'] == 0 for r in c['inputs'])
    assert gate.selected_protocol_pin(p) == gate.COHORT128_PROTOCOL_PIN
    original = json.loads((REPO/gate.COHORT_PROTOCOL).read_bytes())
    for k in ('runtime', 'acquisition', 'native_config', 'checkpoint_buffers', 'amp', 'tf32', 'seed', 'budget_seconds'):
        assert p[k] == original[k]
    source = (REPO/gate.HELPERS[0]).read_text()
    assert "out/'observations_manifest.json'" in source and 'saved_outputs' in source
    proof = dict(runtime=dict(image=dict(Id='sha256:'+'f'*64)), pin=dict(bytes=1, sha256='a'*64))
    _, _, mounts, args = gate.container_plan(Path('/tmp/code'), Path('/tmp/out'), 'a'*40, p, proof, 'model', 1800)
    assert '--external-cohort128' in args and '--external-cohort' not in args
    assert len([m for m in mounts if str(m[0]).startswith(c['root'])]) == 61


def test_large_inspect_projection_keeps_all_validation_metadata(gate):
    source = (REPO/gate.HELPERS[0]).read_text()
    tree = ast.parse(source); validator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_container')
    keys = {n.slice.value for n in ast.walk(validator) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
            and n.value.id == 'value' and isinstance(n.slice, ast.Constant)}
    assert keys == {'Image', 'Name', 'Config', 'HostConfig', 'Mounts'}
    calls = []
    def command(args, deadline):
        calls.append(args)
        return json.dumps({k: {} for k in keys}) if args[-1] == '{{json .}}' else '{}'
    runtime = SimpleNamespace(command=command)
    value = gate.container_inspect(runtime, 'name', 1800, large_cohort=True)
    assert set(value) == keys
    assert len(calls) == 5
    assert set(c[-1] for c in calls) == {'{{json .'+k+'}}' for k in keys}
    assert '{{json .Args}}' not in [c[-1] for c in calls]
    gate.container_inspect(runtime, 'name', 1800)
    assert calls[-1][-1] == '{{json .}}'


def qualified_fixture(gate):
    """Manufactured full128 metadata only: 73 acquired ->61 +12 header misses."""
    p = json.loads((REPO/gate.COHORT128_PROTOCOL).read_bytes())
    values, images = fixture(gate, p); c = p['external_cohort']
    parent = values[c['root']+'/manifest.json']
    parent.update(schema='manufactured_parent.v1', source_proof={'verified': True, 'bytes': 1})
    acquired = next(r for r in parent['records'] if r['status'] == 'acquired')
    missing = [r for r in parent['records'] if r['status'] != 'acquired']
    for slot, row in enumerate(missing):
        row['image_id'] = f'{0xFFFFFFFF00000000+slot:016x}'
        row['reason'] = 'original_access_failure'
    for row in missing[:12]:
        iid = row['image_id']; row.clear(); row.update(copy.deepcopy(acquired)); row['image_id'] = iid
        row['reason'] = 'original_acquisition_note'
    qualified = copy.deepcopy(parent)
    qualified.update(schema='world_reward.openimages_fresh128_input_qualification.v1',
        parent_acquisition_pin=c['manifest'], original_RGB_root=c['root'], header_only_CPU=True)
    censored = {r['image_id'] for r in missing[:12]}
    for row in qualified['records']:
        if row['status'] != 'acquired': continue
        width = 4097 if row['image_id'] in censored else 4096
        row['original_JPEG_header'] = dict(format='JPEG', width=width, height=4096, pixels=width*4096)
        if row['image_id'] in censored:
            row.update(status='unscorable_header', reason='predeclared_original_JPEG_16Mi_pixel_limit_no_replacement')
    values[c['qualified_manifest']['path']] = qualified
    return p, values, images, parent, qualified


def test_qualified_full128_header_only_preserves_inputs_and_config(gate, monkeypatch):
    p, values, images, parent, qualified = qualified_fixture(gate)
    snapshots = copy.deepcopy((p, parent, qualified)); c = p['external_cohort']; calls = []
    counts = gate.validate_external_cohort_qualification(parent, qualified, c['manifest'], c['root'])
    assert counts == dict(parent_acquired=73, qualified_acquired=61, header_unscorable=12, original_missing=55)
    monkeypatch.setattr(gate, 'exact', lambda rt, path, pin, limit: calls.append(str(path)) if images[str(path)] == pin else pytest.fail('Wrong exact RGB pin'))
    monkeypatch.setattr(gate, 'old_source', lambda *args: calls.append('old_source'))
    frozen = gate.authenticate_external_cohort(SimpleNamespace(pinned=lambda path, *args: values[str(path)]), p)
    assert len(frozen) == 126 and calls == [r['path'] for r in c['inputs']]+['old_source']
    assert (p, parent, qualified) == snapshots


@pytest.mark.parametrize('fault', [
    'parent_pin', 'root', 'schema', 'cpu', 'reference', 'top_extra', 'top_mutation',
    'reorder', 'duplicate', 'replace', 'parent_header', 'missing_header', 'header_extra',
    'width_bool', 'height_float', 'pixels_bool', 'pixels_mismatch', 'width_zero', 'height_negative',
    'format_int', 'animation_int', 'acquired_too_large', 'acquired_not_jpeg', 'acquired_animated',
    'false_censor', 'bad_reason', 'parent_pin_field', 'parent_metadata', 'parent_bool_as_int',
    'record_extra', 'missing_mutated', 'missing_header_added', 'missing_promoted', 'source_reason_changed',
])
def test_qualified_header_record_tampering_rejected_before_rgb(gate, monkeypatch, fault):
    p, values, _, parent, qualified = qualified_fixture(gate); c = p['external_cohort']
    acquired = next(r for r in qualified['records'] if r['status'] == 'acquired')
    censored = next(r for r in qualified['records'] if r['status'] == 'unscorable_header')
    missing = next(r for r in qualified['records'] if r['status'] == 'unavailable')
    header = acquired['original_JPEG_header']
    if fault == 'parent_pin': qualified['parent_acquisition_pin'] = {'bytes': 1, 'sha256': '0'*64}
    if fault == 'root': qualified['original_RGB_root'] += '_other'
    if fault == 'schema': qualified['schema'] = 'model_selected.v1'
    if fault == 'cpu': qualified['header_only_CPU'] = 1
    if fault == 'reference': qualified['reference_geometry_read'] = True
    if fault == 'top_extra': qualified['model_quality'] = 1
    if fault == 'top_mutation': qualified['source_proof']['verified'] = 1
    if fault == 'reorder': qualified['records'] = qualified['records'][::-1]
    if fault == 'duplicate': parent['records'][1] = copy.deepcopy(parent['records'][0]); qualified['records'][1] = copy.deepcopy(qualified['records'][0])
    if fault == 'replace': acquired['image_id'] = 'e'*16
    if fault == 'parent_header': parent['records'][0]['original_JPEG_header'] = dict(header)
    if fault == 'missing_header': acquired.pop('original_JPEG_header')
    if fault == 'header_extra': header['model_score'] = .9
    if fault == 'width_bool': header.update(width=True, pixels=header['height'])
    if fault == 'height_float': header['height'] = float(header['height'])
    if fault == 'pixels_bool': header.update(width=1, height=1, pixels=True)
    if fault == 'pixels_mismatch': header['pixels'] -= 1
    if fault == 'width_zero': header.update(width=0, pixels=0)
    if fault == 'height_negative': header.update(height=-1, pixels=-header['width'])
    if fault == 'format_int': header['format'] = 1
    if fault == 'animation_int': header['is_animated'] = 0
    if fault == 'acquired_too_large': header.update(width=4097, pixels=4097*header['height'])
    if fault == 'acquired_not_jpeg': header['format'] = 'PNG'
    if fault == 'acquired_animated': header['is_animated'] = True
    if fault == 'false_censor': censored['original_JPEG_header'].update(width=4096, pixels=4096**2)
    if fault == 'bad_reason': censored['reason'] = 'no_detected_interaction'
    if fault == 'parent_pin_field': acquired['image_pin'] = {'bytes': 1, 'sha256': '0'*64}
    if fault == 'parent_metadata': acquired['publisher_original_metadata']['Author'] = 'Other'
    if fault == 'parent_bool_as_int': acquired['publisher_md5_matched'] = 1
    if fault == 'record_extra': acquired['ranking_result'] = 1
    if fault == 'missing_mutated': missing['reason'] = 'model_error'
    if fault == 'missing_header_added': missing['original_JPEG_header'] = dict(header)
    if fault == 'missing_promoted': missing['status'] = 'acquired'
    if fault == 'source_reason_changed': censored['creator_grant_verified'] = False
    calls = []
    monkeypatch.setattr(gate, 'exact', lambda *args: calls.append('RGB'))
    monkeypatch.setattr(gate, 'old_source', lambda *args: calls.append('model'))
    with pytest.raises(ValueError):
        gate.authenticate_external_cohort(SimpleNamespace(pinned=lambda path, *args: values[str(path)]), p)
    assert calls == []


@pytest.mark.parametrize('format_,animated', [('PNG', None), (None, None), ('JPEG', True), ('JPEG', False)])
def test_header_reason_uses_only_available_format_animation_area(gate, format_, animated):
    p, _, _, parent, qualified = qualified_fixture(gate); c = p['external_cohort']
    row = next(r for r in qualified['records'] if r['status'] == 'unscorable_header')
    header = row['original_JPEG_header']; header.update(format=format_, width=2, height=3, pixels=6)
    if animated is not None: header['is_animated'] = animated
    if format_ == 'JPEG' and animated is False:
        with pytest.raises(ValueError): gate.validate_external_cohort_qualification(parent, qualified, c['manifest'], c['root'])
    else:
        assert gate.validate_external_cohort_qualification(parent, qualified, c['manifest'], c['root'])['header_unscorable'] == 12


@pytest.mark.parametrize('maxpixels', [True, 0, -1, float(16 << 20)])
def test_header_area_policy_rejects_noninteger_or_nonpositive_bound(gate, maxpixels):
    p, _, _, parent, qualified = qualified_fixture(gate)
    with pytest.raises(ValueError):
        gate.validate_external_cohort_qualification(parent, qualified, p['external_cohort']['manifest'], p['external_cohort']['root'], maxpixels)


def test_header_qualification_keeps_old15_missing_policy_unchanged(gate, p, monkeypatch):
    values, _ = fixture(gate, p)
    row = next(r for r in values[p['external_cohort']['root']+'/manifest.json']['records'] if r['status'] != 'acquired')
    row.update(status='unscorable_header', reason='predeclared_original_JPEG_16Mi_pixel_limit_no_replacement')
    monkeypatch.setattr(gate, 'exact', lambda *args: None)
    with pytest.raises(ValueError):
        gate.authenticate_external_cohort(SimpleNamespace(pinned=lambda path, *args: values[str(path)]), p)
