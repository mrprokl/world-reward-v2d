"""Storage/provenance seams only; no native model, dataset or GPU fixtures."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'infra'), str(REPO / 'src')]
import object_budget_solid as producer
import object_pose_smoke as pose
import surface_geometry_loader as loader


@pytest.fixture
def full4d(monkeypatch):
    revision = 'a' * 40
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{revision}/outputs')
    monkeypatch.setenv('WR_PIN_ROOT', str(producer.ROOT / f'experiments/full4d-v1-{revision}/pins'))
    return revision


def test_every_episode_routes_to_isolated_generated_surface(full4d):
    for episode in range(30):
        paths = loader.paths(episode, 'b' * 40, full4d, 'c' * 40)
        assert len(paths) == 15
        assert paths['report'].startswith(f'experiments/full4d-v1-{full4d}/outputs/episode_{episode:06d}/')
        assert paths['source_glb'].endswith('/object_grounded/object.glb')
        assert producer.episode_directory(episode) == producer.ROOT / f'experiments/full4d-v1-{full4d}/outputs/episode_{episode:06d}'
        assert 'results/surface-qslim' in paths['qualification_native']
    assert loader.source_helpers() == loader.SOURCE_HELPERS + ('src/world_reward/artifact_paths.py',)


def test_legacy_namespace_and_helper_set_unchanged(monkeypatch):
    monkeypatch.delenv('WR_OUTPUT_PREFIX', raising=False)
    monkeypatch.delenv('WR_PIN_ROOT', raising=False)
    assert producer.episode_directory(2) == producer.ROOT / 'outputs/episode_000002'
    assert loader.paths(2, 'a' * 40, 'b' * 40, 'c' * 40)['object'] == 'outputs/episode_000002/object_grounded/report.json'
    assert loader.source_helpers() == loader.SOURCE_HELPERS


def source_fixture(monkeypatch, tmp_path):
    revision = 'a' * 40
    root = tmp_path / 'root'
    code = root / 'jobs' / revision / 'run_full4d_sample' / 'code'
    code.mkdir(parents=True)
    for name in ('revision', 'source-sha256'):
        (code.parent / name).write_text('tiny fixture')
    monkeypatch.setattr(producer, 'ROOT', root)
    monkeypatch.setattr(producer, '__file__', str(code / 'infra/object_budget_solid.py'))
    calls = []
    rt = SimpleNamespace(source=lambda *args: calls.append(args) or {'producer_revision': revision})
    return code, revision, calls, rt


def test_honest_full4d_source_entry_is_pinned(full4d, monkeypatch, tmp_path):
    code, revision, calls, rt = source_fixture(monkeypatch, tmp_path)
    result = producer.surface_source(code, revision, rt)
    assert result['source_entry'] == 'run_full4d_sample'
    assert calls[0][3] == 'run_full4d_sample'
    assert 'src/world_reward/artifact_paths.py' in calls[0][4]


@pytest.mark.parametrize('prefix,control', [
    ('experiments/full4d-v1-' + 'b' * 40 + '/outputs', False),
    ('experiments/other-v1-' + 'a' * 40 + '/outputs', False),
    ('experiments/full4d-v1-' + 'a' * 40 + '/outputs', True),
])
def test_other_entry_revision_or_control_cannot_claim_new_source(monkeypatch, tmp_path, prefix, control):
    code, revision, calls, rt = source_fixture(monkeypatch, tmp_path)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', prefix)
    with pytest.raises(ValueError):
        producer.surface_source(code, revision, rt, control=control)
    assert calls == []


def test_mapping_capacity_recognizes_full4d_only_mapping_role(full4d):
    path = producer.ROOT / f'experiments/full4d-v1-{full4d}/outputs/episode_000029/object_budget_surface_{full4d}/mapping.json'
    assert loader._mapping_role(path)
    assert not loader._mapping_role(path.with_name('candidate_geometry.npz'))


def test_runtime_surface_pin_is_used_before_any_geometry(full4d, monkeypatch):
    calls = []
    def reject(path, **kwargs):
        calls.append(Path(path))
        raise ValueError('Absent independent producer pin')
    monkeypatch.setattr(loader, 'identity', reject)
    with pytest.raises(ValueError, match='Absent independent'):
        pose._load_surface_mesh(producer.ROOT, 17, 'b' * 64, Path('/not/read'),
                                Path('/not/read'), 1., Path('/not/write'), Path('/not/write'))
    assert calls == [producer.ROOT / f'experiments/full4d-v1-{full4d}/pins/surface_mesh_000017_pins.json']


def test_native_policy_and_time_budgets_unchanged():
    assert producer.SURFACE_SECONDS == 600 and producer.SURFACE_HOST_SECONDS == 700
    assert producer.SURFACE_IMAGE == loader.IMAGE
    args = pose._argument_parser().parse_args(['--episode', '29', '--full-video', '--mesh-source', 'surface'])
    assert args.episode == 29 and args.full_video and not args.query_requalification


@pytest.mark.parametrize('ending', [b'', b'\n'])
def test_real_docker_cid_bytes_optional_newline_are_accepted(monkeypatch, tmp_path, ending):
    import subprocess
    import mediapipe_cpu_runtime_verify as rt
    cid = tmp_path / 'cid'
    cid.write_bytes(b'a' * 64 + ending)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=b'', stderr=b'error: no such object: ' + b'a' * 64))
    producer.surface_cleanup('owned', 'b' * 40, cid, rt)


@pytest.mark.parametrize('ending', [b'\\n', b'\r\n', b'\n\n', b' '])
def test_docker_cid_other_suffixes_are_rejected(monkeypatch, tmp_path, ending):
    import mediapipe_cpu_runtime_verify as rt
    cid = tmp_path / 'cid'
    cid.write_bytes(b'a' * 64 + ending)
    with pytest.raises(ValueError):
        producer.surface_cleanup('owned', 'b' * 40, cid, rt)
