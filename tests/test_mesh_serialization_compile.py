"""Manufactured/mock contracts only: never compile or invoke Docker locally."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

INFRA = Path(__file__).resolve().parents[1] / 'infra'
spec = importlib.util.spec_from_file_location('serialization_compile_gate', INFRA / 'mesh_serialization_compile.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def tetra():
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def test_literal_remote_socket_precedes_optin_exec():
    text = (INFRA / 'run_volume_qem_build.sh').read_text()
    branch = text.split('if [[ "${1:-}" == --serialization ]]; then')[1].split('\nfi')[0]
    assert branch.index('export DOCKER_HOST="unix://${WR_ROOT:?}/docker.sock"') < branch.index('exec python3')
    assert '[[ $# == 1 || $# == 2 && "$2" == --geometry ]] || exit 2' in branch
    assert 'image-volume-qem.json' in text and '660s docker build' in text


def test_protocol_and_historical_sources_are_exact():
    code = INFRA.parent
    protocol = gate.protocol(code)
    pins = protocol['source_authentication']
    import hashlib
    for name, pin in [('mesh_volume_qem.cpp', 'original_volume_cpp_sha256'),
                      ('mesh_guarded_qem.cpp', 'original_base_cpp_sha256')]:
        assert hashlib.sha256((INFRA / name).read_bytes()).hexdigest() == pins[pin]
    raw = (INFRA / 'mesh_volume_qem.cpp').read_bytes()
    prefix = raw[:raw.index(b'\nint main(int argc, char** argv) {')]
    assert {'bytes': len(prefix), 'sha256': hashlib.sha256(prefix).hexdigest()} == pins['derived_core_prefix']
    assert protocol['phase1_controls']['orientation_scalar_parity']['cases'] == 128


def test_scalar_reference_distinguishes_numeric_seams_and_new_pairs():
    v, f = tetra()
    second = v.copy(); second[second == 0.] = -0.
    exact = gate.scalar_reference(np.r_[v, second], np.r_[f, f+4])
    assert exact['serialization_safe'] and exact['nonexact_collision_pairs'] == 0
    second[0, 0] = 2.**-40
    collision = gate.scalar_reference(np.r_[v, second], np.r_[f, f+4])
    assert collision['source_float32_exactly_active'] and not collision['serialization_safe']
    assert collision['nonexact_key_collision_pairs'] == 1


@pytest.mark.parametrize('kind', ['inactive', 'overflow', 'nonfinite'])
def test_scalar_rejection_or_negative_evidence_is_not_repair(kind):
    v, f = tetra(); before = v.copy()
    if kind == 'inactive':
        v = v * .01 + 1e6
        assert not gate.scalar_reference(v, f)['source_float32_exactly_active']
    elif kind == 'overflow':
        with pytest.raises(ValueError, match='int64'):
            gate.scalar_reference(v + 2.**40, f)
    else:
        v[0, 0] = np.nan
        with pytest.raises(ValueError, match='Nonfinite'):
            gate.scalar_reference(v, f)
    assert np.array_equal(before, tetra()[0])


def test_full_frozen_parity_operator_with_fake_native(tmp_path, monkeypatch):
    scratch = tmp_path.resolve(); calls = []
    def fake(args, **kwargs):
        assert args[1] == '--preflight' and kwargs['timeout'] == 10
        lines = Path(args[2]).read_text().splitlines()
        v = np.array([[float(x) for x in l[2:].split()] for l in lines if l.startswith('v ')])
        f = np.array([[int(x)-1 for x in l[2:].split()] for l in lines if l.startswith('f ')])
        result = gate.scalar_reference(v, f); calls.append(result)
        return SimpleNamespace(returncode=0 if result['source_float32_exactly_active'] else 2,
            stdout=json.dumps(result).encode() if result['source_float32_exactly_active'] else b'')
    monkeypatch.setattr(gate.subprocess, 'run', fake)
    result = gate.parity(Path('/never-executed'), scratch, lambda: 10)
    assert result['cases'] == len(calls) == 128 and result['mismatches'] == 0
    assert result['active_cases'] == 124 and not list(scratch.iterdir())


def test_frozen_parity_fails_on_scalar_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(gate.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=b'{}'))
    with pytest.raises(ValueError, match='scalar parity mismatch'):
        gate.parity(Path('/never-executed'), tmp_path.resolve(), lambda: 10)


def test_no_new_install_or_challenge_or_default_runtime_mutation():
    text = (INFRA / 'mesh_serialization_compile.py').read_text()
    for forbidden in ['--gpus', 'apt-get', 'pip install', 'eval_private', 'docker build', 'nvidia-smi']:
        assert forbidden not in text
    for required in ['--cap-drop', 'no-new-privileges', "'--memory', '16g'", "'--cpus', '4'",
                     "'--network', 'none'", "'--read-only'", 'originals_rehashed_after', 'source_rehashed_after']:
        assert required in text
    assert "simplification_validated=False" in text and "adoption=False" in text
    assert "'/tmp:rw,exec,nosuid,nodev,size=1g'" in text


def test_source_is_in_original_runtime_archive_literal_closure():
    import sys
    sys.path.insert(0, str(INFRA))
    import azure_job
    files = {str(p.relative_to(INFRA.parent)): p.read_bytes() for base in ['infra', 'src', 'configs']
             for p in (INFRA.parent / base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files['pyproject.toml'] = (INFRA.parent / 'pyproject.toml').read_bytes()
    paths = azure_job.runtime_bundle_paths(files, 'infra/run_volume_qem_build.sh')
    assert set(gate.HELPERS) <= set(paths)
    assert 'src/world_reward/exact_triangle_predicates.py' in paths


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_invalid_metadata_fails(raw):
    with pytest.raises(ValueError):
        gate.strict(raw)


def test_no_arbitrary_local_execution():
    with pytest.raises(ValueError, match='No arbitrary'):
        gate.main(['--anything'])


def test_geometric_phase_requires_separate_frozen_protocol_and_prior_scope():
    config = gate.geometry_protocol(INFRA.parent)
    assert config['native_calls_maximum'] == 4 and config['fixed_source_scale'] == '2**-16'
    pins = json.loads((INFRA.parent / gate.PHASE1_PINS).read_text())
    assert pins['source_files'] == 155 and pins['binary_retained'] is False
    assert pins['predicate_parity_only'] is True and pins['simplification_validated'] is False
    text = (INFRA / 'mesh_serialization_compile.py').read_text()
    assert 'source(old, revision, historical=True)' in text
    assert 'spec.loader.exec_module(previous)' not in text
    assert "if geometry_phase:" in text and 'phase1_qualification(code)' in text
    assert "'--native-geometry' if geometry_phase else '--native'" in text
    assert "'original_runtime']" in text and "binary_continuity_claim=False" in text


def test_only_single_official_helper_added_to_geometry_container_not_assets():
    text = (INFRA / 'mesh_serialization_compile.py').read_text()
    assert "helper = ROOT / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'" in text
    assert "extra_mounts = ['--mount', f'type=bind,src={helper},dst={helper},readonly'] if geometry_phase else []" in text
    assert 'src={ROOT},dst={ROOT}' not in text
