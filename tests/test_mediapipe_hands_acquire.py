"""Manufactured tiny ZIP/metadata fixtures; never live network, models, or media."""
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import time
import urllib.request
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('mediapipe_acquisition_test', ROOT/'infra/mediapipe_hands_acquire.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def zipped(rows):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        for name, body in rows: archive.writestr(name, body)
    return out.getvalue()


class Response(io.BytesIO):
    def __init__(self, data, url, *, mime='application/octet-stream', status=200, length=None, encoding=None):
        super().__init__(data); self.url = url; self.status = status; self.reads = []
        self.headers = email.message.Message(); self.headers['Content-Type'] = mime
        self.headers['Content-Length'] = str(len(data) if length is None else length)
        if encoding: self.headers['Content-Encoding'] = encoding

    def geturl(self): return self.url

    def read(self, count=-1):
        assert 0 <= count <= 1 << 20; self.reads.append(count)
        return super().read(count)


class Opener:
    def __init__(self, responses): self.responses = responses; self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout)); return self.responses[request.full_url]


def setup(gate, tmp_path, monkeypatch):
    root = tmp_path/'root'; revision = 'a'*40; code = root/'jobs'/revision/gate.JOB/'code'
    (code/'infra').mkdir(parents=True); (code/'src/world_reward').mkdir(parents=True)
    for name in ('mediapipe_hands_acquire.py', 'run_mediapipe_hands_acquire.sh'):
        (code/'infra'/name).write_bytes((ROOT/'infra'/name).read_bytes())
    (code/'src/world_reward/__init__.py').write_bytes(b'')
    (code.parent/'revision').write_text(revision+'\n'); (code.parent/'source-sha256').write_text('b'*64+'\n')
    for p in sorted(code.rglob('*'), reverse=True): p.chmod(0o555 if p.is_dir() else 0o444)
    code.chmod(0o555)
    for name in ('revision', 'source-sha256'): (code.parent/name).chmod(0o444)
    monkeypatch.setattr(gate, '__file__', str(code/'infra/mediapipe_hands_acquire.py'))
    for folder in (gate.EVIDENCE, gate.WEIGHTS, gate.RESULT): (root/folder).parent.mkdir(parents=True, exist_ok=True)
    metadata = b'Name: mediapipe\nVersion: 0.10.21\nLicense: Apache 2.0\nLicense-File: LICENSE\nRequires-Dist: numpy<2\n\nTiny metadata\n'
    wheel = zipped([('mediapipe-0.10.21.dist-info/METADATA', metadata),
                    ('mediapipe-0.10.21.dist-info/LICENSE', b'Apache License\nVersion 2.0'),
                    ('mediapipe/__init__.py', b'raise RuntimeError("NEVER_EXECUTE")'),
                    ('mediapipe/native.so', b'OPAQUE_BINARY')])
    task = zipped([('hand_detector.tflite', b'NOT_A_DECODED_MODEL'), ('hand_landmarks_detector.tflite', b'OPAQUE')])
    assets = copy.deepcopy(gate.ASSETS)
    payload = {'LICENSE': b'Apache License\nVersion 2.0', 'model-card.pdf': b'%PDF-tiny-not-parsed',
               gate.WHEEL: wheel, 'hand_landmarker.task': task}
    for name, bucket, object_name, generation, size, md5 in (
        ('task-metadata.json', 'mediapipe-models', 'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
         '1682480004222387', 7819105, 'FTGEMOo4UWcP6ZFBFqnPrQ=='),
        ('model-card-metadata.json', 'mediapipe-assets', 'Model Card Hand Tracking (Lite_Full) with Fairness Oct 2021.pdf',
         '1683223215027175', 358044, 'G6gDPVW6ruzBtatTpakf2w==')):
        payload[name] = json.dumps(dict(bucket=bucket, name=object_name, generation=generation, size=str(size), md5Hash=md5)).encode()
    wheel_row = next(x for x in assets if x['name'] == gate.WHEEL)
    payload['pypi.json'] = json.dumps({'urls': [dict(filename=gate.WHEEL, size=len(wheel), url=wheel_row['url'],
        digests={'sha256': hashlib.sha256(wheel).hexdigest()}, yanked=False,
        **{'core-metadata': {'sha256': hashlib.sha256(metadata).hexdigest()}})]}).encode()
    responses = {}
    for row in assets:
        raw = payload[row['name']]; row['bytes'] = len(raw)
        if 'sha256' in row: row['sha256'] = hashlib.sha256(raw).hexdigest()
        if 'md5' in row: row['md5'] = hashlib.md5(raw).hexdigest()
        responses[row['url']] = Response(raw, row['url'], mime=row['mime'][0])
    monkeypatch.setattr(gate, 'ASSETS', assets)
    monkeypatch.setattr(gate, 'METADATA_SHA', hashlib.sha256(metadata).hexdigest())
    def publish(part, target): os.link(part, target); part.unlink()
    monkeypatch.setattr(gate, 'publish', publish)
    return root, code, revision, payload, Opener(responses)


def report(gate, root): return json.loads((root/gate.RESULT/'report.json').read_text())


def test_tiny_real_orchestration_no_install_decode_or_inference(gate, tmp_path, monkeypatch):
    root, code, rev, payload, opener = setup(gate, tmp_path, monkeypatch)
    result = gate.acquire(root, code, rev, opener=opener)
    assert result['status'] == 'pass' and result['source_rehashed_after'] and result['artifacts_rehashed_after']
    assert result['budget_scope'] == 'acquisition_checks_public_sealing_posthash'
    assert result['receipt_publication_outer_seconds'] == 320
    assert len(opener.calls) == 7 and result['task']['entries'] == 2 and result['wheel']['entries'] == 4
    for key in ('models_loaded', 'model_nodes_decoded', 'packages_installed', 'gpu_used', 'dataset_read',
                'private_values_read', 'quality_claim', 'license_eligibility_verified',
                'task_constituent_license_verified', 'training_overlap_verified', 'challenge_overlap_verified'):
        assert result[key] is False
    assert result['wheel']['package_metadata']['requires_dist'] == ['numpy<2']
    assert len(result['wheel']['license_metadata_files']) == 2 and 'inventory' not in result['wheel']
    assert len(json.dumps(result)) < 6500
    assert result['source_binding']['entries'] == 7  # genuine empty __init__ retained
    for row in result['artifacts']:
        p = root/row['file']; assert p.read_bytes() == payload[p.name]
        assert p.stat().st_mode & 0o777 == 0o444
        assert row['sha256'] == hashlib.sha256(payload[p.name]).hexdigest()
    assert next(x for x in result['artifacts'] if x['file'].endswith('.task'))['hash_basis'] == 'caller_measured_sha256_publisher_md5'
    assert (root/gate.RESULT/'report.json').stat().st_mode & 0o777 == 0o444
    assert (root/gate.RESULT).stat().st_mode & 0o777 == 0o700
    assert all((root/folder).stat().st_mode & 0o777 == 0o555 for folder in (gate.EVIDENCE, gate.WEIGHTS))
    assert not list(root.rglob('*.part'))
    assert all(dict(req.header_items()) == {'Accept-encoding': 'identity'} and timeout == 20 for req, timeout in opener.calls)


@pytest.mark.parametrize('fault', ['html', 'status', 'redirect', 'length', 'encoding', 'hash', 'overflow', 'truncated'])
def test_bounded_provider_failures_sealed_no_secret_output(gate, tmp_path, monkeypatch, fault):
    root, code, rev, payload, opener = setup(gate, tmp_path, monkeypatch)
    row = gate.ASSETS[0]; raw = payload[row['name']]; options = {}; url = row['url']
    if fault == 'html': options['mime'] = 'text/html'
    elif fault == 'status': options['status'] = 429
    elif fault == 'redirect': url = 'https://evil.example/?TOKEN=neverlog'
    elif fault == 'length': options['length'] = len(raw)+1
    elif fault == 'encoding': options['encoding'] = 'gzip'
    elif fault == 'hash': raw = b'X'+raw[1:]
    elif fault == 'overflow': raw += b'X'; options['length'] = row['bytes']
    else: raw = raw[:-1]; options['length'] = row['bytes']
    opener.responses[row['url']] = Response(raw, url, **options)
    with pytest.raises(gate.AcquisitionError, match='sealed receipt'): gate.acquire(root, code, rev, opener=opener)
    result = report(gate, root)
    assert result['status'] == 'fail' and result['source_rehashed_after'] and result['owned_partials_removed']
    assert len(opener.calls) == 1 and not list(root.rglob('*.part'))
    assert 'TOKEN' not in json.dumps(result) and 'evil.example' not in json.dumps(result)


@pytest.mark.parametrize('url', ['http://pypi.org/pypi/mediapipe/0.10.21/json', 'https://evil.example/x',
    'https://storage.googleapis.com/download/private?token=SECRET', 'https://pypi.org:444/x',
    'https://u:p@pypi.org/x', 'https://pypi.org/pypi/mediapipe/0.10.21/json#x'])
def test_exact_closed_https_allowlist_and_redirect_handler(gate, url):
    with pytest.raises(gate.AcquisitionError): gate.safe_url(url)
    with pytest.raises(gate.AcquisitionError):
        gate.PublicRedirect().redirect_request(urllib.request.Request(next(iter(gate.URLS))), None, 302, '', {}, url)


@pytest.mark.parametrize('occupied', ['evidence', 'weights', 'result', 'symlink'])
def test_no_resume_replace_or_foreign_cleanup(gate, tmp_path, monkeypatch, occupied):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    p = root/{'evidence': gate.EVIDENCE, 'weights': gate.WEIGHTS, 'result': gate.RESULT, 'symlink': gate.EVIDENCE}[occupied]
    if occupied == 'symlink': p.symlink_to(code, target_is_directory=True)
    else: p.mkdir(); (p/'KEEP').write_bytes(b'FOREIGN')
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert opener.calls == []
    if occupied != 'symlink': assert (p/'KEEP').read_bytes() == b'FOREIGN'


def test_source_or_artifact_mutation_after_download_never_pass(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    original = gate.zip_inventory
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        p = root/gate.EVIDENCE/'LICENSE'; p.chmod(0o644); p.write_bytes(b'ALTERED'); p.chmod(0o444)
        return result
    monkeypatch.setattr(gate, 'zip_inventory', mutate)
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert report(gate, root)['source_rehashed_after'] and not report(gate, root)['artifacts_rehashed_after']


def test_changed_metadata_stops_before_any_binary_asset(gate, tmp_path, monkeypatch):
    root, code, rev, payload, opener = setup(gate, tmp_path, monkeypatch)
    row = gate.ASSETS[0]; metadata = json.loads(payload[row['name']]); metadata['urls'][0]['yanked'] = True
    raw = json.dumps(metadata).encode(); row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    opener.responses[row['url']] = Response(raw, row['url'], mime='application/json')
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert len(opener.calls) == 4 and not (root/gate.EVIDENCE/gate.WHEEL).exists()
    assert not (root/gate.WEIGHTS/'hand_landmarker.task').exists()


def test_source_mutation_and_inclusive_posthash_budget(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    original = gate.source_binding; calls = []
    def delayed(*args):
        calls.append(1)
        result = original(*args)
        if len(calls) == 2: time.sleep(.03)
        return result
    monkeypatch.setattr(gate, 'source_binding', delayed); monkeypatch.setattr(gate, 'BUDGET', .02)
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert report(gate, root)['status'] == 'fail' and report(gate, root)['elapsed_seconds'] > .02


def test_missing_or_mutated_source_receipt_blocks_network(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    p = code/'src/world_reward/__init__.py'; p.chmod(0o644)
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert opener.calls == [] and not report(gate, root)['source_rehashed_after']


def test_source_bytes_changed_after_work_are_rehashed(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    original = gate.zip_inventory
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        p = code/'src/world_reward/__init__.py'; p.chmod(0o644); p.write_bytes(b'changed'); p.chmod(0o444)
        return result
    monkeypatch.setattr(gate, 'zip_inventory', mutate)
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert not report(gate, root)['source_rehashed_after']


def test_cancelled_stream_seals_failure_and_removes_own_partial(gate, tmp_path, monkeypatch):
    root, code, rev, payload, opener = setup(gate, tmp_path, monkeypatch)
    row = gate.ASSETS[-1]
    class Cancelled(Response):
        def read(self, count=-1): gate.cancel(None, None)
    opener.responses[row['url']] = Cancelled(payload[row['name']], row['url'])
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    result = report(gate, root)
    assert result['owned_partials_removed'] and result['status'] == 'fail' and result['source_rehashed_after']


@pytest.mark.parametrize('name', ['../escape', '/escape', 'a/../b', 'a\\b', 'a:b', 'a//b', './a'])
def test_unsafe_zip_members_rejected_no_extraction(gate, tmp_path, name):
    p = tmp_path/'bad.zip'; p.write_bytes(zipped([(name, b'opaque')]))
    with pytest.raises(gate.AcquisitionError): gate.zip_inventory(p, time.monotonic()+5)
    assert not (tmp_path/'escape').exists()


def test_zip_duplicate_symlink_crc_and_budget(gate, tmp_path):
    p = tmp_path/'bad.zip'
    with pytest.warns(UserWarning): p.write_bytes(zipped([('a', b'A'), ('a', b'B')]))
    with pytest.raises(gate.AcquisitionError): gate.zip_inventory(p, time.monotonic()+5)
    p.write_bytes(zipped([('axb', b'opaque')]).replace(b'axb', b'a\x00b'))
    with pytest.raises(gate.AcquisitionError): gate.zip_inventory(p, time.monotonic()+5)
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        info = zipfile.ZipInfo('link'); info.create_system = 3; info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, b'/outside')
    p.write_bytes(out.getvalue())
    with pytest.raises(gate.AcquisitionError): gate.zip_inventory(p, time.monotonic()+5)
    raw = zipped([('a', b'UNIQUE_PAYLOAD')]); p.write_bytes(raw.replace(b'UNIQUE_PAYLOAD', b'UNIQUE_PAYLOAX', 1))
    with pytest.raises(zipfile.BadZipFile): gate.zip_inventory(p, time.monotonic()+5)
    p.write_bytes(zipped([('a', b'A')]))
    with pytest.raises(gate.AcquisitionError): gate.zip_inventory(p, time.monotonic()-1)


def test_midstream_failure_removes_only_owned_partial_preserves_receipt(gate, tmp_path, monkeypatch):
    root, code, rev, payload, opener = setup(gate, tmp_path, monkeypatch)
    row = gate.ASSETS[-1]
    class Broken(Response):
        def read(self, count=-1):
            if self.tell(): raise OSError('SECRET_URL_never_emit')
            return super().read(1)
    opener.responses[row['url']] = Broken(payload[row['name']], row['url'])
    with pytest.raises(gate.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    result = report(gate, root)
    assert len(result['artifacts']) == 6 and result['owned_partials_removed']
    assert all((root/folder).stat().st_mode & 0o777 == 0o555 for folder in (gate.EVIDENCE, gate.WEIGHTS))
    assert (root/gate.EVIDENCE/gate.WHEEL).exists() and not list(root.rglob('*.part'))
    assert 'SECRET_URL' not in json.dumps(result)


def test_source_bound_root_bootstrap_creates_only_fresh_leaves(gate, tmp_path, monkeypatch):
    parent = tmp_path/'shared-parent'; parent.mkdir(); parent.chmod(0o755)
    prior = gate.state(parent); uid, gid = os.getuid(), os.getgid(); calls = []
    monkeypatch.setattr(gate.os, 'getuid', lambda: 0)
    monkeypatch.setattr(gate.os, 'chown', lambda p, u, g: calls.append((p, u, g)))
    leaves = [parent/'owned-evidence', parent/'owned-model']
    lease = gate.create_namespace_lease(leaves, 1000, 1000, 'c'*64)
    assert len(calls) == 2 and all(p.stat().st_mode & 0o777 == 0o700 for p in leaves)
    assert gate.state(parent)[2] == prior[2] and parent.stat().st_uid == uid
    assert lease['source_closure_sha256'] == 'c'*64
    with pytest.raises(gate.AcquisitionError):
        gate.create_namespace_lease(leaves, 1000, 1000, 'c'*64)
    assert len(calls) == 2  # No chown, overwrite or resume of existing leaves.


def lease_for(gate, folders, closure):
    return dict(schema='world_reward.fresh_namespace_lease.v1', source_closure_sha256=closure,
                directories=[dict(path=str(p), device=p.stat().st_dev, inode=p.stat().st_ino,
                                  uid=os.getuid(), gid=os.getgid(), mode=0o700) for p in folders])


def test_precreated_empty_leased_leaves_are_used_once_and_sealed(gate, tmp_path, monkeypatch):
    root, code, rev, payload, opener = setup(gate, tmp_path, monkeypatch)
    folders = [root/gate.EVIDENCE, root/gate.WEIGHTS]
    for p in folders:
        p.mkdir(); p.chmod(0o700); os.chown(p, -1, os.getgid())
    closure = gate.source_binding(root, code, rev)['closure_sha256']
    lease = lease_for(gate, folders, closure)
    result = gate.acquire(root, code, rev, opener=opener, namespace_lease=lease)
    assert result['status'] == 'pass' and result['namespace_lease'] == lease
    assert all(p.stat().st_mode & 0o777 == 0o555 for p in folders)
    with pytest.raises(gate.AcquisitionError):
        gate.acquire(root, code, rev, opener=opener, namespace_lease=lease)
    assert len(opener.calls) == 7


@pytest.mark.parametrize('fault', ['closure', 'inode', 'uid', 'mode', 'occupied', 'boolean_inode', 'extra'])
def test_bootstrap_lease_mutations_block_network_and_foreign_cleanup(gate, tmp_path, monkeypatch, fault):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    folders = [root/gate.EVIDENCE, root/gate.WEIGHTS]
    for p in folders:
        p.mkdir(); p.chmod(0o700); os.chown(p, -1, os.getgid())
    closure = gate.source_binding(root, code, rev)['closure_sha256']
    lease = lease_for(gate, folders, closure)
    if fault == 'closure': lease['source_closure_sha256'] = 'd'*64
    elif fault == 'inode': lease['directories'][0]['inode'] += 1
    elif fault == 'uid': lease['directories'][0]['uid'] += 1
    elif fault == 'mode': folders[0].chmod(0o755)
    elif fault == 'occupied': (folders[0]/'KEEP').write_bytes(b'FOREIGN')
    elif fault == 'boolean_inode': lease['directories'][0]['inode'] = True
    else: lease['directories'][0]['extra'] = 'unapproved'
    with pytest.raises(gate.AcquisitionError):
        gate.acquire(root, code, rev, opener=opener, namespace_lease=lease)
    assert not opener.calls and report(gate, root)['status'] == 'fail'
    if fault == 'occupied': assert (folders[0]/'KEEP').read_bytes() == b'FOREIGN'
    assert folders[1].stat().st_mode & 0o777 == 0o700  # Unaccepted lease not cleaned/sealed.


def test_frozen_actual_pins_and_source_only_shell(gate):
    assert sum(r['bytes'] for r in gate.ASSETS) < 44_000_000 and gate.BUDGET == 300
    task = gate.ASSETS[-1]; assert task['bytes'] == 7819105 and 'sha256' not in task
    wheel = next(r for r in gate.ASSETS if r['name'] == gate.WHEEL)
    assert wheel['bytes'] == 35622638 and wheel['sha256'] == '05dc4a9e593655a79558d05d6227d31018c2537a4bd3362b51e230cf22aecfe3'
    shell = (ROOT/'infra/run_mediapipe_hands_acquire.sh').read_text()
    assert 'python3 -I -B' in shell and '--kill-after=10s 320s' in shell
    assert not any(s in shell for s in ('docker', 'pip install', 'nvidia-smi', 'az ', 'torch'))
