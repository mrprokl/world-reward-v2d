"""Procedural tiny saved banks only; no Azure, RGB, models or role references."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

import vcoco_full_interaction_join as p
from test_vcoco_interaction_observations import fixture, identities
from test_vcoco_full_interaction_core import at_slot


def save(path, arrays):
    with path.open('xb') as f: np.savez(f, **arrays)
    path.chmod(0o400)
    return p.rt.identity(path)


def metadata_proof():
    rows, files = [], {}
    for i in range(48):
        meta = dict(image_id=f'{i:032x}', original_slot=i, acquired_ordinal=i, original_frame_index=0,
            image_size=[8, 12], file=f'image_{i:06d}.npz', identity=dict(bytes=1, sha256='a'*64))
        paths = [f'/unopened/{name}/image_{i:06d}.npz' for name in ('endpoint', 'pose', 'hoi')]
        e = dict(deepcopy(meta), person_ids=[], person_retained_rows=0)
        v = dict(deepcopy(meta), person_ids=[], persons=0, endpoint_bank_identity=meta['identity'])
        h = dict(deepcopy(meta), source_person_ids=[], hand_object_pairs=0, endpoint_bank_identity=meta['identity'])
        files.update({n: meta['identity'] for n in paths}); rows.append(dict(endpoint=e, pose=v, hoi=h, paths=paths))
    return dict(schema=p.SCHEMA, producer_revision='b'*40, image_id=p.IMAGE, source={}, records=rows, files=files)


def test_full144_all47_prevalidated_before_first_unselected_join(monkeypatch):
    proof = metadata_proof(); calls = []
    monkeypatch.setattr(p, 'load_bank', lambda path, row, fields: calls.append(('load', len(fields))) or {})
    def build(*args, population):
        assert sum(x[0] == 'load' for x in calls) >= 144 and population == 48
        calls.append(('join',)); return args[0]['original_slot']
    monkeypatch.setattr(p.numerical, 'reconstruct_interaction', build)
    monkeypatch.setattr(p, 'save_evidence', lambda value, deadline: dict(slot=value, identity=dict(bytes=1)))
    result = p.join_all(proof, time.monotonic()+10)
    assert [r['slot'] for r in result] == list(range(48))
    assert [r[1] for r in calls[:144]] == [17, 11, 19]*48
    assert sum(r[0] == 'join' for r in calls) == 48


def test_last_raw_bank_failure_prevents_every_join(monkeypatch):
    proof = metadata_proof(); calls = []
    def load(path, row, fields):
        calls.append(path)
        if str(path) == proof['records'][-1]['paths'][-1]: raise ValueError('tiny last raw failure')
        return {}
    monkeypatch.setattr(p, 'load_bank', load)
    monkeypatch.setattr(p.numerical, 'reconstruct_interaction', lambda *a, **kw: pytest.fail('premature construct'))
    with pytest.raises(ValueError): p.join_all(proof, time.monotonic()+10)
    assert len(calls) == 144


@pytest.mark.parametrize('fault', ['missing', 'alias', 'slot', 'ordinal', 'grid', 'ids', 'endpoint_pin', 'filepin'])
def test_complete_population_not_guessed_or_compressed(fault):
    proof = metadata_proof()
    if fault == 'missing': proof['records'].pop()
    elif fault == 'alias': proof['records'][-1]['paths'][1] = proof['records'][0]['paths'][1]
    elif fault == 'slot': proof['records'][-1]['endpoint']['original_slot'] = 0
    elif fault == 'ordinal': proof['records'][-1]['pose']['acquired_ordinal'] = 0
    elif fault == 'grid': proof['records'][-1]['hoi']['image_size'] = [1, 1]
    elif fault == 'ids': proof['records'][-1]['hoi']['source_person_ids'] = ['other']
    elif fault == 'endpoint_pin': proof['records'][-1]['pose']['endpoint_bank_identity'] = dict(bytes=2, sha256='b'*64)
    else: proof['files'][proof['records'][0]['paths'][0]] = dict(bytes=2, sha256='a'*64)
    with pytest.raises(ValueError): p.validate_population(proof)


@pytest.mark.parametrize('fault', ['extra', 'duplicate', 'pickle', 'shape', 'dtype', 'hash', 'symlink', 'hardlink', 'writable', 'truncated'])
def test_npz_transport_exact_plain_and_fail_closed(tmp_path, fault):
    args, _ = fixture(0, False); arrays = args[3]; path = tmp_path/'tiny.npz'
    row = dict(args[2], identity=save(path, arrays))
    if fault in ('extra', 'duplicate', 'pickle', 'truncated'):
        path.chmod(0o600)
        if fault in ('extra', 'duplicate'):
            with zipfile.ZipFile(path, 'a') as z:
                with pytest.warns(UserWarning) if fault == 'duplicate' else __import__('contextlib').nullcontext():
                    z.writestr('extra.npy' if fault == 'extra' else 'image_size.npy', b'x')
        elif fault == 'pickle':
            arrays = dict(arrays, person_ids=np.array([object()], object)); row['arrays'] = identities(arrays)
            with path.open('wb') as f: np.savez(f, **arrays)
        else: path.write_bytes(path.read_bytes()[:-32])
        path.chmod(0o400); row['identity'] = p.rt.identity(path)
    elif fault == 'shape': row['arrays']['image_size']['shape'] = [99999999]
    elif fault == 'dtype': row['arrays']['image_size']['dtype'] = '<f8'
    elif fault == 'hash': row['arrays']['image_size']['sha256'] = 'b'*64
    elif fault == 'symlink': alias = tmp_path/'alias'; alias.symlink_to(path); path = alias
    elif fault == 'hardlink': os.link(path, tmp_path/'alias')
    else: path.chmod(0o600)
    with pytest.raises((ValueError, zipfile.BadZipFile, EOFError)):
        p.load_bank(path, row, p.numerical.POSE_FIELDS)


@pytest.mark.parametrize('people,pairs', [(0, False), (0, True), (2, False), (2, True)])
def test_save_reopen_full_unselected_evidence_bytes_and_nan(tmp_path, monkeypatch, people, pairs):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    args, _ = fixture(people, pairs); args = at_slot(args, 47)
    value = p.numerical.reconstruct_interaction(*args, population=48)
    arrays = p.evidence_arrays(value); before = identities(arrays)
    row = p.save_evidence(value, time.monotonic()+10)
    actual = p.load_bank(tmp_path/row['file'], row, tuple(arrays), maximum=p.MAX_EVIDENCE)
    assert len(actual) == 46 and identities(actual) == before == identities(arrays)
    assert all(actual[n].tobytes() == a.tobytes() for n, a in arrays.items())
    assert row['person_side_object_rows'] == people*2*3600
    assert row['person_side_pair_rows'] == people*2*(4 if pairs else 0)
    assert row['pair_object_rows'] == (4 if pairs else 0)*3600
    assert value.evidence.selection_performed is False and not value.evidence.scope['absence_predictions_generated']
    assert tmp_path.joinpath(row['file']).stat().st_mode & 0o777 == 0o400


def test_save_partial_failure_removes_only_owned_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path)
    args, _ = fixture(0, False); value = p.numerical.reconstruct_interaction(*at_slot(args, 0), population=48)
    def broken(stream, **arrays): stream.write(b'partial'); raise OSError('tiny write failure')
    monkeypatch.setattr(np, 'savez', broken)
    with pytest.raises(OSError): p.save_evidence(value, time.monotonic()+10)
    assert not (tmp_path/'image_000000.npz').exists()


def test_partial_complete_rows_preserved_on_later_failure(monkeypatch):
    proof = metadata_proof(); completed = []
    monkeypatch.setattr(p, 'load_bank', lambda *a: {})
    monkeypatch.setattr(p.numerical, 'reconstruct_interaction', lambda row, *a, **kw: row['original_slot'])
    def save_row(slot, deadline):
        if slot == 1: raise ValueError('tiny later failure')
        return dict(slot=slot, identity=dict(bytes=1))
    monkeypatch.setattr(p, 'save_evidence', save_row)
    with pytest.raises(ValueError): p.join_all(proof, time.monotonic()+10, records_out=completed)
    assert completed == [dict(slot=0, identity=dict(bytes=1))]


def test_import_and_native_whitelist_no_models_private_receipts_or_rgb(tmp_path):
    code = f'''import sys
sys.path[:0]=[{str(Path('infra').resolve())!r},{str(Path('src').resolve())!r}]
class Deny:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('numpy','torch','onnxruntime','vcoco_full_pose_replica','vcoco_full_hoi_run','vcoco_full_public_replica'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
import vcoco_full_interaction_join as p
assert len(p.NATIVE_FILES)==10
'''
    subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True)
    assert not any(word in ' '.join(p.NATIVE_FILES) for word in ('config', 'acquire', 'model', 'replica', 'runtime_verify.py /'))
    tree = ast.parse(Path(p.__file__).read_text())
    assert not any(isinstance(n, ast.Import) and any(a.name in ('numpy', 'torch') for a in n.names) for n in tree.body)


def test_native_publication_deadline_demotes_original_fd(tmp_path, monkeypatch):
    out = tmp_path/'out'; out.mkdir(mode=0o700); p.rt.write(out/'proof.json', b'{}\n')
    monkeypatch.setattr(p, 'OUTPUT', out); monkeypatch.setattr(p, 'check', lambda _: (_ for _ in ()).throw(TimeoutError()))
    report = dict(status='pass', images=[])
    p.publish_native(report, 1, out.lstat())
    actual = p.rt.strict((out/'native.json').read_bytes())
    assert actual['status'] == 'fail' and actual['publication_error_type'] == 'TimeoutError'
    assert (out/'native.json').stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize('foreign', [False, True])
def test_cleanup_saved_cid_independent_of_name_never_foreignremove(tmp_path, monkeypatch, foreign):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path); cid = 'a'*64; (tmp_path/'.container.cid').write_text(cid)
    calls = []; live = [True]; name = 'new-own'; rev = 'b'*40
    def command(argv, deadline):
        calls.append(argv)
        if argv[:2] == ['docker', 'ps']: return cid if live[0] and 'id='+cid in argv else ''
        if argv[:2] == ['docker', 'inspect']: return p.IMAGE+'|/'+('renamed' if foreign else name)+'|'+p.ENTRY+'|'+rev
        if argv[:2] == ['docker', 'rm']: live[0] = False; return ''
        raise AssertionError(argv)
    monkeypatch.setattr(p, 'command', command)
    if foreign:
        with pytest.raises(ValueError): p.cleanup(name, rev, 10)
        assert not any(a[1] == 'rm' for a in calls)
    else: p.cleanup(name, rev, 10); assert not live[0]


def cli():
    return ['--pose-replica-revision', 'b'*40, '--pose-receipt-bytes', '1', '--pose-receipt-sha256', 'a'*64,
        *[x for label in ('host', 'native', 'proof') for x in ('--pose-'+label+'-bytes', '1', '--pose-'+label+'-sha256', 'c'*64)]]


def test_actual_pose_and_replica_pins_mandatory_no_overrides():
    a = p.arguments(cli()); assert len(a.pose_pins) == 3
    for bad in ([], cli()+['--pose-receipt-bytes', '2'], cli()+['--code', '/bad'], cli()[:-1]+['wrong']):
        with pytest.raises(ValueError): p.arguments(bad)
    with pytest.raises(SystemExit): p.arguments(cli()+['--image', 'invented'])


def test_source_original_hoi_helpers_unchanged_and_exec_modes(tmp_path):
    import io
    import tarfile
    import vcoco_full_hoi_run as h
    archive = subprocess.check_output(['rtk', 'proxy', 'git', 'archive', '--format=tar', p.HOI_REV])
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        files = {r.name: (r, t.extractfile(r).read()) for r in t if r.isfile()}
    assert all(Path(n).read_bytes() == files[n][1] for n in h.HELPERS)
    import azure_job
    selected = azure_job.runtime_bundle_paths({n: row[1] for n, row in files.items()}, 'infra/run_vcoco_full_hoi_observations.sh')
    assert p.HOI_EXECUTABLES == frozenset(n for n in selected if files[n][0].mode & 0o111)


def native_fixture(proof, revision, pin):
    rows = []
    for i, original in enumerate(proof['records']):
        people, pairs = original['pose']['persons'], original['hoi']['hand_object_pairs']
        metadata = p.expected_evidence_metadata(original)
        rows.append(dict(image_id=original['endpoint']['image_id'], original_slot=i, acquired_ordinal=i,
            persons=people, objects=3600, native_pairs=pairs, person_side_object_rows=people*7200,
            person_side_pair_rows=people*2*pairs, pair_object_rows=pairs*3600,
            supported_counts=[0, 0, 0], nan_counts=[people*7200*10, people*2*pairs*15, pairs*3600*2],
            raw_bank_fingerprint='a'*64, evidence_fingerprint='b'*64, file=f'image_{i:06d}.npz',
            identity=dict(bytes=1, sha256='c'*64), person_ids=original['endpoint']['person_ids'],
            object_id_fingerprint=dict(shape=[3600], dtype='<U17', sha256='d'*64),
            arrays={n: dict(shape=shape, dtype=dtype, sha256='d'*64) for n, (shape, dtype) in metadata.items()}))
    return dict(schema=p.SCHEMA, stage='native_full48_interaction_join', status='pass', phase='complete',
        producer_revision=revision, image_id=p.IMAGE, proof_identity=pin, source_inputs_rehashed_after=True,
        all144_banks_prevalidated=True, input_arrays_per_image=47, output_arrays_per_image=46,
        models_loaded=0, images=rows, **{n: False for n in p.FLAGS})


@pytest.mark.parametrize('fault', ['none', 'truncated', 'prevalidated', 'inputcount', 'outputcount', 'model', 'source', 'rgb', 'objects', 'arraydtype', 'arraymissing', 'arrayhash', 'ids'])
def test_native_receipt_complete_fields_not_generic_claims(monkeypatch, fault):
    proof = metadata_proof(); pin = dict(bytes=1, sha256='a'*64); rev = 'b'*40
    native = native_fixture(proof, rev, pin)
    monkeypatch.setattr(p.rt, 'identity', lambda *a, **kw: dict(bytes=1, sha256='c'*64))
    if fault == 'truncated': native['images'].pop()
    elif fault == 'prevalidated': native['all144_banks_prevalidated'] = False
    elif fault == 'inputcount': native['input_arrays_per_image'] = 46
    elif fault == 'outputcount': native['output_arrays_per_image'] = 47
    elif fault == 'model': native['models_loaded'] = 1
    elif fault == 'source': native['source_inputs_rehashed_after'] = False
    elif fault == 'rgb': native['RGB_decoded'] = True
    elif fault == 'objects': native['images'][-1]['objects'] = 3599
    elif fault == 'arraydtype': native['images'][0]['arrays']['base_features']['dtype'] = '<f4'
    elif fault == 'arraymissing': native['images'][0]['arrays'].pop('bridge_supported')
    elif fault == 'arrayhash': native['images'][0]['arrays']['base_supported']['sha256'] = 'bad'
    elif fault == 'ids': native['images'][0]['object_id_fingerprint']['sha256'] = 'foreign'
    if fault == 'none': p.validate_native(native, proof, rev, pin)
    else:
        with pytest.raises(ValueError): p.validate_native(native, proof, rev, pin)


def test_output_metadata_matches_genuine_constructor_exact_shapes_dtypes():
    for people, pairs in ((0, False), (0, True), (2, False), (2, True)):
        args, _ = fixture(people, pairs); args = at_slot(args, 47)
        value = p.numerical.reconstruct_interaction(*args, population=48)
        record = dict(endpoint=args[0], pose=args[2], hoi=args[4])
        expected = p.expected_evidence_metadata(record)
        assert expected == {n: (list(a.shape), a.dtype.str) for n, a in p.evidence_arrays(value).items()}


@pytest.mark.parametrize('failure', ['none', 'native', 'total_bound', 'evidence_bound', 'foreign_leaf', 'foreign_mode'])
def test_host_narrow_mounts_full144_failure_sourcepost_and_publication(tmp_path, monkeypatch, failure):
    import sealed_callback_publication as publication
    root = tmp_path/'root'; root.mkdir(); code = root/'code'; code.mkdir(); out = root/'output'; revision = 'b'*40
    proof = metadata_proof(); before = dict(current_source=dict(binding={}))
    for n in p.NATIVE_FILES:
        leaf = code/n; leaf.parent.mkdir(parents=True, exist_ok=True); p.rt.write(leaf, b'pass\n')
    for name in ('revision', 'source-sha256'): p.rt.write(root/name, b'pin\n')
    for i, r in enumerate(proof['records']):
        paths = []
        for j, key in enumerate(('endpoint', 'pose', 'hoi')):
            path = root/f'bank_{i:02d}_{j}.npz'; p.rt.write(path, b'x'); paths.append(str(path)); proof['files'][str(path)] = p.rt.identity(path)
            r[key]['identity'] = proof['files'][str(path)]
        r['paths'] = paths
    proof['files'] = {path: p.rt.identity(path) for r in proof['records'] for path in r['paths']}
    monkeypatch.setattr(p, 'OUTPUT', out); monkeypatch.setattr(p.sys, 'platform', 'linux')
    monkeypatch.setattr(p.os, 'geteuid', lambda: 0); monkeypatch.setattr(p.os, 'uname', lambda: SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    calls = []
    def authenticate(*a): calls.append('authenticate'); return before, proof
    monkeypatch.setattr(p, 'authenticate', authenticate); monkeypatch.setattr(p, 'image', lambda _: {'Id': p.IMAGE})
    monkeypatch.setattr(p, 'command', lambda *a: ''); monkeypatch.setattr(p, 'cleanup', lambda *a: calls.append('cleanup'))
    monkeypatch.setattr(p, 'validate_native', lambda *a: None)
    def native_run(argv, **kwargs):
        mounts = [argv[i+1] for i, v in enumerate(argv) if v == '--mount']; readonly = [m for m in mounts if m.endswith(',readonly')]
        assert len(readonly) == len(p.NATIVE_FILES)+2+144 and not any('.jpg' in x or 'report.json' in x or 'public-pose' in x for x in readonly)
        assert '--gpus' not in argv and argv[argv.index('--memory')+1] == '8g' and argv[argv.index('--network')+1] == 'none'
        assert kwargs['env']['DOCKER_HOST'] == 'unix://'+str(p.ROOT/'docker.sock')
        p.rt.write(out/'native.json', p.encode(dict(images=[], status='fail' if failure == 'native' else 'pass')))
        if failure == 'foreign_leaf': p.rt.write(out/'foreign.json', b'untouched')
        if failure == 'foreign_mode': (out/'native.json').chmod(0o600)
        if failure == 'evidence_bound': p.rt.write(out/'image_000000.npz', b'xx')
        return SimpleNamespace(returncode=1 if failure == 'native' else 0)
    monkeypatch.setattr(p.subprocess, 'run', native_run)
    if failure == 'total_bound':
        monkeypatch.setattr(p, 'MAX_TOTAL', 1); monkeypatch.setattr(p, 'MAX_CONTROL_TOTAL', 1)
    elif failure == 'evidence_bound': monkeypatch.setattr(p, 'MAX_TOTAL', 1)
    if failure in ('foreign_leaf', 'foreign_mode'):
        with pytest.raises(ValueError):
            p.dispatch_work(code, revision, 'c'*40, dict(bytes=1, sha256='d'*64), {}, time.monotonic(), time.monotonic()+20)
        assert not (out/'report.json').exists()
        if failure == 'foreign_leaf': assert (out/'foreign.json').read_bytes() == b'untouched'
        else: assert stat.S_IMODE((out/'native.json').stat().st_mode) == 0o600
        assert stat.S_IMODE(out.stat().st_mode) == 0o700
        return
    result = p.dispatch_work(code, revision, 'c'*40, dict(bytes=1, sha256='d'*64), {}, time.monotonic(), time.monotonic()+20)
    assert result['status'] == ('pass' if failure == 'none' else 'fail')
    assert calls == ['authenticate', 'cleanup', 'authenticate'] and result['source_inputs_image_rehashed_after']
    assert result['output_capacity_gate_passed'] is (failure not in ('total_bound', 'evidence_bound'))
    if failure in ('total_bound', 'evidence_bound'):
        assert result['failure_stage'] == 'output_capacity' and result['output_capacity_error_type'] == 'ValueError'
        assert {n: p.rt.identity(out/n, p.MAX_EVIDENCE) for n in result['saved_outputs']} == result['saved_outputs']
    assert result['outputs_sealed'] and out.stat().st_mode & 0o777 == 0o500
    assert p.rt.strict((out/'report.json').read_bytes())['status'] == result['status']
