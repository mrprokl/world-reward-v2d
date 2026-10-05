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


def test_large_scope_complete73_inputs_and_bounded_sidecar(gate):
    raw = (REPO/gate.COHORT128_PROTOCOL).read_bytes(); p = json.loads(raw)
    assert gate.COHORT128_PROTOCOL_PIN == dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    c = p['external_cohort']
    assert (c['slots'], c['acquired'], c['missing']) == (128, 73, 55)
    assert len(c['inputs']) == len({r['image_id'] for r in c['inputs']}) == 73
    assert len({r['cohort_slot'] for r in c['inputs']}) == 73
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
    assert len([m for m in mounts if str(m[0]).startswith(c['root'])]) == 73
