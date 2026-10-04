"""Manufactured original snapshots only; no cloud, media or old-code execution."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'infra'))
    import cari_export_source
    import cari_clip_inputs
    return cari_export_source, cari_clip_inputs


def setup(gate, tmp_path):
    source, public = gate
    root = tmp_path / 'root'; code = tmp_path / 'current'
    root.mkdir(); code.mkdir()
    revision = 'a' * 40
    original = root / 'jobs' / revision / 'run_cari_shared_stage_queued' / 'code'
    original.mkdir(parents=True)
    names = {*source.NUMERICAL_SOURCES, 'infra/cari_clip_inputs.py', 'infra/run_cari_full_export.sh', 'src/world_reward/__init__.py'}
    for name in names:
        raw = (ROOT / name).read_bytes()
        for directory in (original, code):
            p = directory / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw); p.chmod(0o444)
    for name in ('infra/cari_export_source.py', 'infra/cari_shared_episode_loader.py'):
        p = code / name; p.write_bytes((ROOT / name).read_bytes()); p.chmod(0o444)
    # Non-numerical wrapper changes are not mistaken for historical producer bytes.
    (code / 'infra/run_cari_full_export.sh').chmod(0o644)
    (code / 'infra/run_cari_full_export.sh').write_text('changed current wrapper, never executed\n')
    (code / 'infra/run_cari_full_export.sh').chmod(0o444)
    for p in (original, *original.rglob('*')):
        if p.is_dir(): p.chmod(0o555)
    digest = 'b' * 64
    (original.parent / 'revision').write_text(revision + '\n'); (original.parent / 'source-sha256').write_text(digest + '\n')
    for name in ('revision', 'source-sha256'): (original.parent / name).chmod(0o444)
    spec = public.PublicClipSpec(21, 97, 'front', 2, 4)
    export = dict(bytes=12, sha256='c' * 64, producer_revision=revision, script_sha256=source._read(original / 'infra/cari_full_export.py')[1]['sha256'])
    pins = dict(schema='world_reward.cari_export_source_pins.v1', clip_spec=asdict(spec), export=export,
        producer_revision=revision, job='run_cari_shared_stage_queued', source_archive_sha256=digest,
        files={name: source._read(original / name, empty=True)[1] for name in names})
    pin = code / 'configs/cari_clip_000021_export_source_pins.json'; pin.parent.mkdir(); pin.write_text(json.dumps(pins)); pin.chmod(0o444)
    return root, code, original, spec, dict(export=export), pin, pins


def write(path, raw):
    path.chmod(0o644); path.write_bytes(raw); path.chmod(0o444)


def test_complete_original_ledger_and_separate_current_numerics(gate, tmp_path):
    source, _ = gate; root, code, original, spec, export, _, pins = setup(gate, tmp_path)
    observed, proof = source.verify_export_source(root, code, spec, export)
    assert observed == original and proof['files_verified'] == len(pins['files'])
    assert proof['historical_code_executed'] is False and proof['quality_verified'] is False
    assert set(proof['current_numeric_sources']) == set(source.NUMERICAL_SOURCES)
    assert proof == source.verify_export_source(root, code, spec, export)[1]
    assert 'numpy' not in source.__dict__ and 'torch' not in source.__dict__


@pytest.mark.parametrize('fault', ['revision', 'archive', 'extra_parent', 'extra_file', 'extra_directory', 'missing_file', 'byte', 'mode', 'symlink', 'hardlink', 'foreign_job', 'report', 'clipbool', 'schema', 'ledger', 'numeric', 'input_numeric', 'consumer_writable'])
def test_original_full_snapshot_and_current_compatibility_fail_closed(gate, tmp_path, fault):
    source, _ = gate; root, code, original, spec, export, pin, pins = setup(gate, tmp_path)
    if fault == 'revision': write(original.parent / 'revision', b'c' * 40 + b'\n')
    elif fault == 'archive': write(original.parent / 'source-sha256', b'c' * 64 + b'\n')
    elif fault == 'extra_parent': (original.parent / 'other').write_bytes(b'extra')
    elif fault == 'extra_file':
        original.chmod(0o755); (original / 'extra.py').write_bytes(b'extra'); (original / 'extra.py').chmod(0o444); original.chmod(0o555)
    elif fault == 'extra_directory':
        original.chmod(0o755); (original / 'extra').mkdir(); (original / 'extra').chmod(0o555); original.chmod(0o555)
    elif fault == 'missing_file':
        p = original / 'infra/run_cari_full_export.sh'; p.parent.chmod(0o755); p.unlink(); p.parent.chmod(0o555)
    elif fault == 'byte': write(original / 'infra/cari_full_export.py', b'changed original')
    elif fault == 'mode': (original / 'infra/cari_full_export.py').chmod(0o644)
    elif fault == 'symlink':
        p = original / 'infra/run_cari_full_export.sh'; p.parent.chmod(0o755); p.unlink(); p.symlink_to(code / 'infra/run_cari_full_export.sh'); p.parent.chmod(0o555)
    elif fault == 'hardlink':
        import os
        os.link(original / 'infra/cari_full_export.py', tmp_path / 'external_alias')
    elif fault == 'foreign_job': pins['job'] = 'run_cari_full_export'
    elif fault == 'report': export['export'] = dict(export['export'], sha256='d' * 64)
    elif fault == 'clipbool': pins['clip_spec']['episode_index'] = True
    elif fault == 'schema': pins['unexpected'] = None
    elif fault == 'ledger': pins['files'].pop('src/world_reward/__init__.py')
    elif fault == 'numeric': write(code / 'infra/cari_full_export.py', b'changed numerical consumer')
    elif fault == 'consumer_writable': (code / 'infra/cari_export_source.py').chmod(0o644)
    else:
        p = code / 'infra/cari_clip_inputs.py'; raw = p.read_bytes().replace(b'def inferred_camera(spec):', b'def inferred_camera(spec):\n    altered = True')
        write(p, raw)
    if fault in ('foreign_job', 'clipbool', 'schema', 'ledger'): write(pin, json.dumps(pins).encode())
    with pytest.raises((ValueError, FileNotFoundError, KeyError)): source.verify_export_source(root, code, spec, export)


def test_only_input_enumeration_can_change_not_geometry_or_gauge(gate, tmp_path):
    source, _ = gate; root, code, _, spec, export, _, _ = setup(gate, tmp_path)
    p = code / 'infra/cari_clip_inputs.py'; raw = p.read_bytes().replace(b'def source_profile(pins):', b'def source_profile(pins):\n    independent_current_enum = True')
    write(p, raw)
    assert source.verify_export_source(root, code, spec, export)[1]['input_numeric_definitions'] == list(source.INPUT_NUMERICS)


def test_independent_real_pin_map_matches_original_git_snapshot():
    import io, lzma, subprocess, tarfile
    spec = importlib.util.spec_from_file_location('export_source_archive', ROOT / 'infra/azure_job.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pins = json.loads((ROOT / 'configs/cari_clip_000021_export_source_pins.json').read_text())
    raw = subprocess.check_output(['git', 'archive', '--format=tar', pins['producer_revision'], 'infra', 'src', 'configs', 'pyproject.toml'], cwd=ROOT)
    archive, _ = module.runtime_archive(raw, 'infra/run_cari_shared_stage_queued.sh')
    assert hashlib.sha256(lzma.compress(archive, preset=6)).hexdigest() == pins['source_archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(archive)) as entries:
        observed = {m.name: dict(bytes=m.size, sha256=hashlib.sha256(entries.extractfile(m).read()).hexdigest(), mode=m.mode & 0o555) for m in entries}
    assert observed == pins['files'] and len(observed) == 200
