"""Tiny manufactured wheel/metadata streams only; no Azure/network/install."""
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

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'infra'))
    spec = importlib.util.spec_from_file_location('mp_dependencies_test', REPO/'infra/mediapipe_cpu_dependencies_acquire.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def digest(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def zip_bytes(rows):
    import zipfile
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        for name, raw in rows: archive.writestr(name, raw)
    return out.getvalue()


class Response(io.BytesIO):
    def __init__(self, raw, url, mime='application/octet-stream'):
        super().__init__(raw); self.url = url; self.status = 200
        self.headers = email.message.Message(); self.headers['Content-Type'] = mime
        self.headers['Content-Length'] = str(len(raw))
    def geturl(self): return self.url
    def read(self, count=-1):
        assert 0 <= count <= 1 << 20
        return super().read(count)


class Opener:
    def __init__(self, responses): self.responses = responses; self.calls = []
    def open(self, request, timeout):
        self.calls.append(request.full_url); return self.responses[request.full_url]


def readonly_snapshot(root, revision, job, files):
    code = root/'jobs'/revision/job/'code'; code.mkdir(parents=True)
    for name, raw in files.items():
        p = code/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    for p in sorted(code.rglob('*'), reverse=True): p.chmod(0o555 if p.is_dir() else 0o444)
    code.chmod(0o555)
    for name, raw in [('revision', revision+'\n'), ('source-sha256', 'c'*64+'\n')]:
        p = code.parent/name; p.write_text(raw); p.chmod(0o444)
    return code


def setup(gate, tmp_path, monkeypatch):
    root = tmp_path/'root'; root.mkdir(); old_rev, rev = 'a'*40, 'b'*40
    mp = gate.mp
    metadata = b'Name: mediapipe\nVersion: 0.10.21\nLicense: Apache 2.0\nLicense-File: LICENSE\n\n'
    original_wheel = zip_bytes([('mediapipe-0.10.21.dist-info/METADATA', metadata),
                               ('mediapipe-0.10.21.dist-info/LICENSE', b'Apache License Version 2.0')])
    original_payload = {}
    originals = copy.deepcopy(mp.ASSETS)
    for asset in originals:
        raw = original_wheel if asset['name'] == mp.WHEEL else b'OPAQUE-'+asset['name'].encode()
        original_payload[asset['name']] = raw; asset.update(bytes=len(raw))
        if 'sha256' in asset: asset['sha256'] = digest(raw)['sha256']
        if 'md5' in asset: asset['md5'] = hashlib.md5(raw).hexdigest()
        p = root/asset['folder']/asset['name']; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw); p.chmod(0o444)
    monkeypatch.setattr(mp, 'ASSETS', originals)
    source_files = {'infra/mediapipe_hands_acquire.py': (REPO/'infra/mediapipe_hands_acquire.py').read_bytes(),
                    'infra/run_mediapipe_hands_acquire.sh': (REPO/'infra/run_mediapipe_hands_acquire.sh').read_bytes(),
                    'src/world_reward/__init__.py': b''}
    old_code = readonly_snapshot(root, old_rev, mp.JOB, source_files)
    old_binding = gate.source_binding(root, old_code, old_rev, old=True)
    old_artifacts = [dict(file=a['folder']+'/'+a['name'], **digest(original_payload[a['name']])) for a in originals]
    old_receipt = dict(stage='mediapipe_hands_source_model_acquisition', status='pass', source_binding=old_binding,
        source_rehashed_after=True, artifacts_rehashed_after=True, owned_partials_removed=True, elapsed_seconds=1.,
        models_loaded=False, model_nodes_decoded=False, packages_installed=False, gpu_used=False, dataset_read=False,
        private_values_read=False, quality_claim=False, license_eligibility_verified=False,
        task_constituent_license_verified=False, training_overlap_verified=False, challenge_overlap_verified=False,
        artifacts=old_artifacts)
    path = root/mp.RESULT/'report.json'; path.parent.mkdir(parents=True); path.write_text(json.dumps(old_receipt)); path.chmod(0o444)
    pins = dict(schema='world_reward.mediapipe_hands_acquire_pins.v1', producer_revision=old_rev,
                report=digest(path.read_bytes()), helper=digest(source_files['infra/mediapipe_hands_acquire.py']))
    rows, payload = [], {}
    for index in range(26):
        name, version = ('mediapipe', '0.10.21') if index == 0 else (f'dep{index:02}', '1.0')
        declared = ['LICENSE'] if index != 2 else ['../LICENSE']
        raw = metadata if index == 0 else (f'Name: {name}\nVersion: {version}\nLicense: MIT\nLicense-File: {declared[0]}\n\n').encode()
        filename = mp.WHEEL if index == 0 else f'{name}-{version}-py3-none-any.whl'
        wheel = original_wheel if index == 0 else zip_bytes([(f'{name}-{version}.dist-info/METADATA', raw),
            (f'{name}-{version}.dist-info/licenses/LICENSE', b'NOTICE'), (f'{name}/module.py', b'NEVER_EXECUTE')])
        url = mp.WHEEL_URL if index == 0 else f'https://files.pythonhosted.org/packages/ab/cd/test/{filename}'
        row = dict(name=name, version=version, filename=filename, url=url, **digest(wheel),
                   acquisition='reference_existing_mediapipe_hands_acquire_v1' if index == 0 else 'new_Azure_wheel_only',
                   metadata=dict(url=url+'.metadata', **digest(raw), requires_dist=[], license=dict(
                       declared='Apache 2.0' if index == 0 else 'MIT', expression=None, license_files=declared)))
        rows.append(row); payload[url+'.metadata'] = raw
        if index: payload[url] = wheel
    manifest = dict(schema='world_reward.mediapipe_cpu_dependencies.v1', packages=rows, package_count=26,
                    new_wheel_bytes=sum(r['bytes'] for r in rows[1:]))
    encoded_manifest = json.dumps(manifest).encode(); monkeypatch.setattr(gate, 'MANIFEST_PIN', digest(encoded_manifest))
    source_files.update({'infra/mediapipe_cpu_dependencies_acquire.py': (REPO/'infra/mediapipe_cpu_dependencies_acquire.py').read_bytes(),
        'infra/run_mediapipe_cpu_dependencies_acquire.sh': (REPO/'infra/run_mediapipe_cpu_dependencies_acquire.sh').read_bytes(),
        gate.MANIFEST: encoded_manifest, gate.PRIOR_PINS: json.dumps(pins).encode()})
    code = readonly_snapshot(root, rev, gate.JOB, source_files)
    monkeypatch.setattr(gate, '__file__', str(code/'infra/mediapipe_cpu_dependencies_acquire.py'))
    monkeypatch.setattr(mp, '__file__', str(code/'infra/mediapipe_hands_acquire.py'))
    for folder in (gate.EVIDENCE, gate.RESULT): (root/folder).parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mp, 'publish', lambda part, target: (os.link(part, target), part.unlink()))
    return root, code, rev, rows, Opener({u: Response(b, u) for u, b in payload.items()})


def report(gate, root): return json.loads((root/gate.RESULT/'report.json').read_text())


def test_fresh_full_dependency_orchestration_reuses_original_mp_wheel(gate, tmp_path, monkeypatch):
    root, code, rev, rows, opener = setup(gate, tmp_path, monkeypatch)
    result = gate.acquire(root, code, rev, opener=opener)
    assert result['status'] == 'pass' and len(result['wheels']) == 26 and len(result['artifacts']) == 51
    assert gate.mp.WHEEL_URL not in opener.calls and len(opener.calls) == 51
    assert all(result[k] for k in ('source_rehashed_after', 'prior_rehashed_after', 'artifacts_rehashed_after'))
    assert result['budget_scope'] == 'acquisition_checks_public_sealing_posthash'
    assert (root/gate.EVIDENCE).stat().st_mode & 0o777 == 0o555
    assert all(p.stat().st_mode & 0o777 == 0o444 for p in (root/gate.EVIDENCE).iterdir())
    assert (root/gate.RESULT/'report.json').stat().st_mode & 0o777 == 0o444
    assert result['wheels'][2]['declared_license_files']['../LICENSE']
    assert not list(root.rglob('*.part'))
    assert all(result[k] is False for k in ('models_loaded', 'packages_installed', 'gpu_used', 'dataset_read',
        'private_values_read', 'quality_claim', 'license_eligibility_verified', 'training_overlap_verified', 'challenge_overlap_verified'))
    assert gate.dependency_source(root, code, rev) == result['source_binding']


def namespace(gate, root, code, rev):
    directory = root/gate.EVIDENCE; directory.mkdir(mode=0o700); directory.chmod(0o700)
    os.chown(directory, os.getuid(), os.getgid())  # emulate root bootstrap's explicit target group
    s = directory.stat()
    return dict(schema='world_reward.fresh_namespace_lease.v1',
                source_closure_sha256=gate.source_binding(root, code, rev)['closure_sha256'],
                directories=[dict(path=str(directory), device=s.st_dev, inode=s.st_ino,
                    uid=os.getuid(), gid=os.getgid(), mode=0o700)])


def test_source_bound_unprivileged_bootstrap_lease_and_public_sealing(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    lease = namespace(gate, root, code, rev)
    result = gate.acquire(root, code, rev, opener=opener, namespace_lease=lease)
    assert result['status'] == 'pass' and result['namespace_lease'] == lease
    assert (root/gate.EVIDENCE).stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize('fault', ['inode', 'source', 'uid', 'occupied', 'mode'])
def test_lease_cannot_reuse_or_mutate_a_namespace(gate, tmp_path, monkeypatch, fault):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch); lease = namespace(gate, root, code, rev)
    if fault == 'inode': lease['directories'][0]['inode'] += 1
    elif fault == 'source': lease['source_closure_sha256'] = '0'*64
    elif fault == 'uid': lease['directories'][0]['uid'] += 1
    elif fault == 'occupied': (root/gate.EVIDENCE/'KEEP').write_bytes(b'FOREIGN')
    else: (root/gate.EVIDENCE).chmod(0o755)
    with pytest.raises(gate.mp.AcquisitionError): gate.acquire(root, code, rev, opener=opener, namespace_lease=lease)
    assert opener.calls == []
    if fault == 'occupied': assert (root/gate.EVIDENCE/'KEEP').read_bytes() == b'FOREIGN'


@pytest.mark.parametrize('fault', ['receipt', 'source', 'artifact', 'readonly', 'manifest', 'task_md5'])
def test_prior_and_independent_pins_fail_before_network(gate, tmp_path, monkeypatch, fault):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    path = {'receipt': root/gate.mp.RESULT/'report.json', 'source': root/'jobs'/('a'*40)/gate.mp.JOB/'code/infra/mediapipe_hands_acquire.py',
            'artifact': root/gate.mp.EVIDENCE/'LICENSE', 'readonly': code/'src/world_reward/__init__.py',
            'manifest': code/gate.MANIFEST, 'task_md5': root/gate.mp.WEIGHTS/'hand_landmarker.task'}[fault]
    path.chmod(0o644)
    if fault != 'readonly': path.write_bytes(b'CHANGED'); path.chmod(0o444)
    with pytest.raises(gate.mp.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert opener.calls == [] and report(gate, root)['status'] == 'fail'


def test_dependency_stream_failure_keeps_completed_wheel_and_owned_part_cleanup(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    url = list(opener.responses)[3]
    class Broken(Response):
        def read(self, count=-1):
            if self.tell(): raise OSError('PRIVATE_URL_TOKEN_neverlog')
            return super().read(1)
    opener.responses[url] = Broken(opener.responses[url].getvalue(), url)
    with pytest.raises(gate.mp.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    result = report(gate, root)
    assert result['status'] == 'fail' and result['owned_partials_removed'] and result['prior_rehashed_after']
    assert len(result['artifacts']) == 3 and not list(root.rglob('*.part'))
    assert 'TOKEN' not in json.dumps(result)


def test_prior_artifact_mutation_during_work_never_pass(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    original = gate.wheel_record
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs); path = root/gate.mp.WEIGHTS/'hand_landmarker.task'
        path.chmod(0o644); path.write_bytes(b'ALTERED'); path.chmod(0o444); return result
    monkeypatch.setattr(gate, 'wheel_record', mutate)
    with pytest.raises(gate.mp.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert not report(gate, root)['prior_rehashed_after']


@pytest.mark.parametrize('folder', ['evidence', 'result'])
def test_preserve_unknown_namespace_never_resume(gate, tmp_path, monkeypatch, folder):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    path = root/(gate.EVIDENCE if folder == 'evidence' else gate.RESULT); path.mkdir(); (path/'KEEP').write_bytes(b'FOREIGN')
    with pytest.raises(gate.mp.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert not opener.calls and (path/'KEEP').read_bytes() == b'FOREIGN'


def test_inclusive_posthash_deadline_and_exact_url_restore(gate, tmp_path, monkeypatch):
    root, code, rev, _, opener = setup(gate, tmp_path, monkeypatch)
    urls = gate.mp.URLS; original = gate.verify_prior; calls = []
    def slow(*args):
        result = original(*args); calls.append(1)
        if len(calls) == 2: time.sleep(.03)
        return result
    monkeypatch.setattr(gate, 'verify_prior', slow); monkeypatch.setattr(gate, 'BUDGET', .02)
    with pytest.raises(gate.mp.AcquisitionError): gate.acquire(root, code, rev, opener=opener)
    assert report(gate, root)['elapsed_seconds'] > .02 and gate.mp.URLS == urls


def test_declared_notice_missing_blocks_no_false_license_clearance(gate, tmp_path, monkeypatch):
    root, code, rev, rows, _ = setup(gate, tmp_path, monkeypatch)
    row = rows[1]; p = tmp_path/'one.whl'; m = tmp_path/'one.metadata'
    raw = b'Name: dep01\nVersion: 1.0\nLicense: MIT\nLicense-File: LICENSE\n\n'
    m.write_bytes(raw); p.write_bytes(zip_bytes([('dep01-1.0.dist-info/METADATA', raw)]))
    with pytest.raises(gate.mp.AcquisitionError, match='notice missing'):
        gate.wheel_record(p, m, row, time.monotonic()+5)


def test_exact_real_manifest_byte_pin_and_no_install_shell(gate):
    assert digest((REPO/gate.MANIFEST).read_bytes()) == gate.MANIFEST_PIN
    shell = (REPO/'infra/run_mediapipe_cpu_dependencies_acquire.sh').read_text()
    assert 'python3 -I -B' in shell and '--kill-after=10s 320s' in shell
    assert not any(w in shell for w in ('docker', 'pip install', 'nvidia-smi', 'az ', 'torch'))
