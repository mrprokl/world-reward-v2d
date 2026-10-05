import hashlib
from pathlib import Path

import pytest

import joint_pair_dwpose_acquire as producer


def test_original_assets_unchanged():
    assert len(producer.acq.ASSETS) == 9
    assert producer.acq.ASSETS[0][1:3] == (
        134399116, '724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843')
    assert hashlib.sha256(Path(producer.acq.__file__).read_bytes()).hexdigest() == producer.notices.SOURCE_SHA
    assert producer.BASE != producer.acq.BASE
    assert producer.REPORT != producer.acq.REPORT


def test_partial_acquisition_is_not_success(tmp_path, monkeypatch):
    rows = (('one.bin', 1, hashlib.sha256(b'a').hexdigest(), 'https://huggingface.co/one'),
            ('two.bin', 1, hashlib.sha256(b'b').hexdigest(), 'https://huggingface.co/two'))
    monkeypatch.setattr(producer.acq, 'ASSETS', rows)
    calls = []
    def download(url, target, size, sha):
        calls.append(url)
        if len(calls) == 2: raise ValueError('fixed failure')
        target.write_bytes(b'a'); target.chmod(0o444)
    report = {'status': 'fail', 'files': []}
    with pytest.raises(ValueError, match='fixed failure'):
        producer.acquire(tmp_path, report, lambda: None, download, lambda: ['primary'])
    assert len(calls) == 2 and len(report['files']) == 1
    assert report['status'] == 'fail' and 'final_asset_pins' not in report
    assert not (tmp_path / 'two.bin').exists()


def test_final_inventory_requires_all_original_assets(tmp_path):
    with pytest.raises(ValueError, match='Complete original'):
        producer.final_assets(tmp_path, [], [])


def test_final_assets_reject_mutation_and_symlinks(tmp_path, monkeypatch):
    sha = hashlib.sha256(b'a').hexdigest()
    monkeypatch.setattr(producer.acq, 'ASSETS', (('one', 1, sha, 'https://huggingface.co/one'),))
    rows = [{'file': 'one', 'bytes': 1, 'sha256': sha}]
    target = tmp_path / 'one'
    target.write_bytes(b'b'); target.chmod(0o444)
    with pytest.raises(ValueError, match='changed'):
        producer.final_assets(tmp_path, rows, [])
    target.unlink(); (tmp_path / 'other').write_bytes(b'a'); target.symlink_to('other')
    with pytest.raises(ValueError, match='Canonical'):
        producer.final_assets(tmp_path, rows, [])


def test_final_notices_include_both_exact_packages(tmp_path, monkeypatch):
    sha = hashlib.sha256(b'a').hexdigest()
    monkeypatch.setattr(producer.acq, 'ASSETS', (('one', 1, sha, 'https://huggingface.co/one'),))
    (tmp_path / 'one').write_bytes(b'a'); (tmp_path / 'one').chmod(0o444)
    rows = [{'file': 'one', 'bytes': 1, 'sha256': sha}]
    audits = [{'package': name, 'retained_texts': []} for name in ('onnxruntime', 'flatbuffers')]
    assert producer.final_assets(tmp_path, rows, audits) == {'one': {'bytes': 1, 'sha256': sha}}
    with pytest.raises(ValueError, match='Both unchanged'):
        producer.final_assets(tmp_path, rows, audits[:1])


def test_azure_cpu_only_new_namespace():
    shell = Path('infra/run_joint_pair_dwpose_acquire.sh').read_text()
    assert 'runuser -u scenesmith' in shell and '610s' in shell
    assert 'docker run' not in shell and '--gpus' not in shell
    assert 'dwpose_native_v1' not in shell
