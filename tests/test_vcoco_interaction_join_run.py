"""Authored saved-byte controls only; no native models, RGB, references or Azure."""
import ast
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

import vcoco_interaction_join_run as p
from test_vcoco_interaction_observations import fixture, identities


def save(path, arrays):
    with path.open('xb') as stream: np.savez(stream, **arrays)
    path.chmod(0o400)
    return p.rt.identity(path)


def array_fixture(tmp_path):
    args, _ = fixture(n=0, k=False)
    pin = save(tmp_path/'tiny.npz', args[3]); row = dict(args[2], identity=pin)
    return tmp_path/'tiny.npz', row, args[3]


def test_real_array_transport_all11_bytes_and_no_mutation(tmp_path):
    path, row, expected = array_fixture(tmp_path)
    result = p.load_bank(path, row, p.numerical.POSE_FIELDS)
    assert identities(result) == identities(expected)
    assert p.rt.identity(path) == row['identity']


@pytest.mark.parametrize('fault', ['extra', 'duplicate', 'pickle', 'arraypin', 'npy_shape', 'npy_dtype', 'truncated', 'symlink', 'hardlink', 'writable'])
def test_npz_fail_closed_before_factory(tmp_path, fault):
    path, row, arrays = array_fixture(tmp_path)
    if fault in ('extra', 'duplicate', 'pickle', 'npy_shape', 'npy_dtype', 'truncated'):
        path.chmod(0o600)
        if fault in ('extra', 'duplicate'):
            with zipfile.ZipFile(path, 'a') as z:
                with pytest.warns(UserWarning) if fault == 'duplicate' else __import__('contextlib').nullcontext():
                    z.writestr('extra.npy' if fault == 'extra' else 'image_size.npy', b'bad')
        elif fault == 'pickle':
            arrays = dict(arrays, person_ids=np.array([object()], dtype=object))
            with path.open('wb') as f: np.savez(f, **arrays)
            row['arrays'] = identities(arrays)
        elif fault in ('npy_shape', 'npy_dtype'):
            row = deepcopy(row)
            row['arrays']['image_size']['shape' if fault == 'npy_shape' else 'dtype'] = [99999999] if fault == 'npy_shape' else '<f8'
        else:
            path.write_bytes(path.read_bytes()[:-40])
        path.chmod(0o400); row['identity'] = p.rt.identity(path)
    elif fault == 'arraypin': row['arrays']['image_size']['sha256'] = 'f'*64
    elif fault == 'symlink':
        alias = tmp_path/'alias.npz'; alias.symlink_to(path); path = alias
    elif fault == 'hardlink': os.link(path, tmp_path/'alias.npz')
    else: path.chmod(0o600)
    with pytest.raises((ValueError, zipfile.BadZipFile, EOFError)): p.load_bank(path, row, p.numerical.POSE_FIELDS)


def metadata_proof():
    rows = []; files = {}
    for i in range(16):
        meta = dict(image_id=f'{i:032x}', original_slot=i, acquired_ordinal=i, original_frame_index=0,
            image_size=[8, 12], file=f'image_{i:06d}.npz', identity=dict(bytes=1, sha256='a'*64))
        paths = [f'/unopened/{name}_{i}.npz' for name in ('endpoint', 'pose', 'hoi')]
        files.update({path: meta['identity'] for path in paths})
        rows.append(dict(endpoint=deepcopy(meta), pose=dict(deepcopy(meta), persons=0), hoi=dict(deepcopy(meta), hand_object_pairs=0), paths=paths))
    return dict(schema=p.SCHEMA, image_id=p.IMAGE, records=rows, files=files)


def test_all16_all47_raw_banks_prevalidated_before_first_join(monkeypatch):
    proof = metadata_proof(); calls = []
    def load(path, row, fields): calls.append(('load', str(path), len(fields))); return dict()
    def reconstruct(*args):
        assert sum(x[0] == 'load' for x in calls) >= 48
        calls.append(('join',)); return args[0]['original_slot']
    monkeypatch.setattr(p, 'load_bank', load); monkeypatch.setattr(p.numerical, 'reconstruct_interaction', reconstruct)
    monkeypatch.setattr(p, 'summary', lambda x: x)
    assert p.join_all(proof, __import__('time').monotonic()+10) == list(range(16))
    assert [x[2] for x in calls[:48]] == [17, 11, 19]*16
    assert sum(x[0] == 'join' for x in calls) == 16


def test_last_raw_bank_failure_means_zero_joins(monkeypatch):
    proof = metadata_proof(); calls = []
    def load(path, row, fields):
        calls.append(str(path))
        if str(path) == proof['records'][-1]['paths'][-1]: raise ValueError('authored last-bank failure')
        return {}
    monkeypatch.setattr(p, 'load_bank', load)
    monkeypatch.setattr(p.numerical, 'reconstruct_interaction', lambda *a: pytest.fail('factory ran before full prevalidation'))
    with pytest.raises(ValueError): p.join_all(proof, __import__('time').monotonic()+10)
    assert len(calls) == 48


@pytest.mark.parametrize('fault', ['missing', 'duplicate_path', 'duplicate_image', 'wrong_slot', 'hoi_ordinal', 'grid', 'frame', 'filepin'])
def test_fullpopulation_mismatch_never_joins(monkeypatch, fault):
    proof = metadata_proof()
    if fault == 'missing': proof['records'].pop()
    elif fault == 'duplicate_path': proof['records'][1]['paths'][0] = proof['records'][0]['paths'][0]
    elif fault == 'duplicate_image': proof['records'][1]['endpoint']['image_id'] = proof['records'][0]['endpoint']['image_id']
    elif fault == 'wrong_slot': proof['records'][-1]['endpoint']['original_slot'] = 0
    elif fault == 'hoi_ordinal': proof['records'][-1]['hoi']['acquired_ordinal'] = 0
    elif fault == 'grid': proof['records'][-1]['hoi']['image_size'] = [2, 3]
    elif fault == 'frame': proof['records'][-1]['pose']['original_frame_index'] = 1
    else: proof['files'][proof['records'][0]['paths'][0]] = dict(bytes=2, sha256='a'*64)
    monkeypatch.setattr(p, 'load_bank', lambda *a: None)
    monkeypatch.setattr(p.numerical, 'reconstruct_interaction', lambda *a: pytest.fail('factory unexpectedly called'))
    with pytest.raises(ValueError): p.join_all(proof, __import__('time').monotonic()+10)


@pytest.mark.parametrize('n,k', [(0, False), (0, True), (2, False), (2, True)])
def test_real_pure_join_summary_complete_cartesian_and_unknowns(n, k):
    args, _ = fixture(n=n, k=k)
    value = p.numerical.reconstruct_interaction(*args); row = p.summary(value)
    pairs = 4 if k else 0
    assert (row['person_side_object_rows'], row['person_side_pair_rows'], row['pair_object_rows']) == (n*2*3600, n*2*pairs, pairs*3600)
    assert len(row['raw_bank_fingerprint']) == len(row['evidence_fingerprint']) == 64
    assert value.evidence.selection_performed is False
    assert value.evidence.scope['absence_predictions_generated'] is False


def test_stdlib_only_import_and_exact_native_dependency_whitelist(tmp_path):
    code = f'''import sys
sys.path[:0]=[{str(Path('infra').resolve())!r},{str(Path('src').resolve())!r}]
class Deny:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('numpy','torch','onnxruntime','vcoco_person_pose_observations','vcoco_hoi_saved_replica','vcoco_replica_completion'):
   raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
import vcoco_interaction_join_run as p
assert len(p.NATIVE_FILES)==10
'''
    subprocess.run([sys.executable, '-I', '-B', '-c', code], check=True)
    assert not any(x in ' '.join(p.NATIVE_FILES) for x in ('config', 'dwpose_smoke', 'hoi_saved_replica', 'model', 'acquire', 'replica_completion'))
    src = ast.parse(Path(p.__file__).read_text())
    imports = {n.name for x in src.body if isinstance(x, ast.Import) for n in x.names}
    assert imports & {'numpy', 'torch', 'vcoco_hoi_saved_replica', 'vcoco_person_pose_observations'} == set()


def test_native_publication_deadline_demotes_same_owned_inode(tmp_path, monkeypatch):
    out = tmp_path/'out'; out.mkdir(mode=0o700); p.rt.write(out/'proof.json', b'{}\n')
    monkeypatch.setattr(p, 'OUTPUT', out)
    report = dict(status='pass')
    monkeypatch.setattr(p, 'check', lambda _: (_ for _ in ()).throw(TimeoutError()))
    p.publish_native(report, 1, out.lstat())
    assert p.rt.strict((out/'native.json').read_bytes())['status'] == 'fail'
    assert (out/'native.json').stat().st_mode & 0o777 == 0o400
    assert p.rt.strict((out/'native.json').read_bytes())['publication_error_type'] == 'TimeoutError'


def test_foreign_native_parent_does_not_publish(tmp_path, monkeypatch):
    out = tmp_path/'out'; out.mkdir(mode=0o700); owner = out.lstat(); out.rename(tmp_path/'old'); out.mkdir(mode=0o700)
    monkeypatch.setattr(p, 'OUTPUT', out)
    with pytest.raises(ValueError): p.publish_native(dict(status='pass'), 1, owner)
    assert not (out/'native.json').exists()


@pytest.mark.parametrize('foreign', [True, False])
def test_exact_cid_cleanup_never_removes_foreign_or_renamed_container(tmp_path, monkeypatch, foreign):
    monkeypatch.setattr(p, 'OUTPUT', tmp_path); cid = 'a'*64; (tmp_path/'.container.cid').write_text(cid); calls = []
    name = 'new-owned'; revision = 'b'*40; present = [True]
    def command(argv, deadline):
        calls.append(argv)
        if argv[:2] == ['docker', 'ps']:
            return cid if present[0] and any(x == 'id='+cid for x in argv) else ''
        if argv[:2] == ['docker', 'inspect']:
            return p.IMAGE+'|/'+('foreign' if foreign else name)+'|'+p.ENTRY+'|'+revision
        if argv[:2] == ['docker', 'rm']: present[0] = False; return ''
        raise AssertionError(argv)
    monkeypatch.setattr(p, 'command', command)
    if foreign:
        with pytest.raises(ValueError): p.cleanup(name, revision, 10)
        assert not any(x[1] == 'rm' for x in calls)
    else:
        p.cleanup(name, revision, 10)
        assert calls[-2][-1] == 'id='+cid and calls[-1][-1] == 'name=^/'+name+'$'


def test_actual_cli_requires_independent_receiver_pin_no_profile_override():
    argv = ['--hoi-replica-revision', 'b'*40, '--hoi-receipt-bytes', '123', '--hoi-receipt-sha256', 'a'*64]
    assert p.arguments(argv).pin == dict(bytes=123, sha256='a'*64)
    for bad in ([], argv+['--code', '/bad'], argv[:-1]+['x'*64]):
        with pytest.raises(ValueError): p.arguments(bad)
    with pytest.raises(SystemExit): p.arguments(argv+['--image', 'fake'])


def fake_native(proof, revision, pin):
    rows = []
    for i, x in enumerate(proof['records']):
        people, pairs = x['pose'].get('persons', 0), x['hoi'].get('hand_object_pairs', 0)
        rows.append(dict(image_id=x['endpoint']['image_id'], original_slot=i, acquired_ordinal=i,
            persons=people, objects=3600, native_pairs=pairs, person_side_object_rows=people*7200,
            person_side_pair_rows=people*2*pairs, pair_object_rows=pairs*3600,
            supported_counts=[0, 0, 0], nan_counts=[0, 0, 0],
            raw_bank_fingerprint='a'*64, evidence_fingerprint='b'*64))
    return dict(schema=p.SCHEMA, stage='native_vcoco_interaction_join', status='pass', phase='complete',
        producer_revision=revision, image_id=p.IMAGE, proof_identity=pin, images=rows, source_inputs_rehashed_after=True,
        arrays_per_image=47, models_loaded=0, **{n: False for n in ('GPU_used', 'reference_metadata_read', 'RGB_decoded',
            'FIT_performed', 'selection_performed', 'ownership_verified', 'quality_verified', 'adoption')})


@pytest.mark.parametrize('fault', ['arrays', 'truncated', 'objects', 'sourcepost', 'model', 'rgb', 'slot', 'fingerprint'])
def test_native_receipt_no_falsepass_from_generic_flags(fault):
    proof = metadata_proof(); pin = dict(bytes=23, sha256='a'*64); rev = 'b'*40
    native = fake_native(proof, rev, pin); p.validate_native(native, proof, rev, pin)
    if fault == 'arrays': native['arrays_per_image'] = 46
    elif fault == 'truncated': native['images'].pop()
    elif fault == 'objects': native['images'][0]['objects'] = 3599
    elif fault == 'sourcepost': native['source_inputs_rehashed_after'] = False
    elif fault == 'model': native['models_loaded'] = 1
    elif fault == 'rgb': native['RGB_decoded'] = True
    elif fault == 'slot': native['images'][-1]['original_slot'] = 0
    else: native['images'][0]['evidence_fingerprint'] = 'not-digest'
    with pytest.raises(ValueError): p.validate_native(native, proof, rev, pin)


def test_host_lifecycle_narrow_mounts_and_failure_postchecks(tmp_path, monkeypatch):
    import sealed_callback_publication as publication
    root = tmp_path/'root'; root.mkdir(); code = root/'code'; code.mkdir(); (code/'infra').mkdir(); out = root/'result'
    rev = 'b'*40; proof = metadata_proof(); before = dict(current_source=dict(binding={}))
    # Tiny authored files stand in for all authenticated source/NPZ mounts.
    for name in p.NATIVE_FILES:
        path = code/name; path.parent.mkdir(parents=True, exist_ok=True); p.rt.write(path, b'pass\n')
    for name in ('revision', 'source-sha256'): p.rt.write(code.parent/name, b'fixture\n')
    files = {}
    for i, record in enumerate(proof['records']):
        for j, key in enumerate(('endpoint', 'pose', 'hoi')):
            path = root/f'bank_{i}_{j}.npz'; p.rt.write(path, b'opaque\n')
            pin = p.rt.identity(path); record['paths'][j] = str(path); record[key]['identity'] = pin; files[str(path)] = pin
    proof['files'] = files
    monkeypatch.setattr(p, 'ROOT', root); monkeypatch.setattr(p, 'OUTPUT', out)
    monkeypatch.setattr(p.sys, 'platform', 'linux'); monkeypatch.setattr(p.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(p.os, 'uname', lambda: SimpleNamespace(nodename='scenesmith-ncc-h100-01'))
    calls = []
    def authenticate(*args): calls.append('auth'); return deepcopy(before), deepcopy(proof)
    monkeypatch.setattr(p, 'authenticate', authenticate); monkeypatch.setattr(p, 'image', lambda _: dict(Id=p.IMAGE))
    monkeypatch.setattr(p, 'command', lambda *args: '')
    def run(argv, **kw):
        assert kw['env']['DOCKER_HOST'] == 'unix://'+str(root/'docker.sock')
        assert '--gpus' not in argv and argv[argv.index('--network')+1] == 'none'
        assert argv[argv.index('--memory')+1] == '6g' and argv[argv.index('--cpus')+1] == '4'
        assert 'CUDA_VISIBLE_DEVICES=' in argv
        mounts = [argv[i+1] for i, item in enumerate(argv) if item == '--mount']
        assert len(mounts) == len(p.NATIVE_FILES)+2+48+1
        assert not any(x in ' '.join(mounts) for x in ('rgb', 'completion', 'checkpoint', 'model'))
        path = out/'proof.json'; pin = p.rt.identity(path)
        p.rt.write(out/'.container.cid', ('a'*64).encode())
        p.rt.write(out/'native.json', p.encode(fake_native(proof, rev, pin)))
        return SimpleNamespace(returncode=1)  # Even complete-looking receipt does not override actual process failure.
    monkeypatch.setattr(p.subprocess, 'run', run)
    # Use real lifecycle/sealed publisher, with no actual Linux dirfsync dependency on macOS.
    monkeypatch.setattr(p, 'sync', lambda _: None)
    result = p.dispatch_work(code, rev, 'c'*40, dict(bytes=1, sha256='d'*64), __import__('time').monotonic(), __import__('time').monotonic()+20)
    assert result['status'] == 'fail' and result['native_exit_status'] == 1 and result['source_inputs_assets_image_rehashed_after'] is True
    assert calls == ['auth', 'auth'] and result['owned_cleanup_verified'] is True
    assert out.stat().st_mode & 0o777 == 0o500
    assert all(x.stat().st_mode & 0o777 == 0o400 for x in out.iterdir())
    assert p.rt.strict((out/'report.json').read_bytes())['status'] == 'fail'


def test_original_df7_executable_contract_exact_git_runtime_archive():
    import azure_job
    import json
    import lzma
    import tarfile
    raw = subprocess.check_output(['rtk', 'proxy', 'git', 'archive', p.POSE_REV])
    archive, _ = azure_job.runtime_archive(raw, 'infra/run_vcoco_person_pose_observations.sh')
    assert hashlib.sha256(lzma.compress(archive, preset=6)).hexdigest() == p.POSE_XZ
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        files = {x.name: (x.mode & 0o555, source.extractfile(x).read()) for x in source if x.isfile()}
    assert len(files) == 347
    assert {name for name, (mode, _) in files.items() if mode & 0o111} == p.EXECUTABLES
    assert all(mode == (0o555 if name in p.EXECUTABLES else 0o444) for name, (mode, _) in files.items())
    entries = {name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, (_, raw) in files.items()}
    directories = {'.'} | {str(parent) for name in files for parent in Path(name).parents if str(parent) != '.'}
    entries.update({name: dict(directory=True) for name in directories})
    assert len(entries) == 352
    assert hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest() == p.POSE_CLOSURE
