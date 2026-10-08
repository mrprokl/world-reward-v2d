"""Tiny manufactured receipts test publication, never native correctness."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'infra'), str(REPO / 'src')]
import full4d_pins as pins


def write(path, raw, mode=0o444):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def fixture(monkeypatch, tmp_path, role='forward'):
    revision = 'a' * 40
    root = tmp_path / 'root'
    code = root / 'jobs' / revision / 'run_full4d_sample' / 'code'
    code.mkdir(parents=True)
    write(code.parent / 'revision', (revision + '\n').encode())
    write(code.parent / 'source-sha256', ('b' * 64 + '\n').encode())
    monkeypatch.setattr(pins, 'ROOT', root)
    monkeypatch.setenv('WR_OUTPUT_PREFIX', f'experiments/full4d-v1-{revision}/outputs')
    monkeypatch.setenv('WR_PIN_ROOT', str(root / f'experiments/full4d-v1-{revision}/pins'))
    pinroot = root / f'experiments/full4d-v1-{revision}/pins'
    pinroot.mkdir(parents=True)
    monkeypatch.setattr(pins, 'pin_path', lambda code, episode, role: pinroot / f'{role}_{episode:06d}.json')
    script, stage, names = pins.SHARED[role]
    script_id = write(code / 'infra' / script, b'# Manufactured source fixture\n')
    code.chmod(0o555)
    base = root / f'experiments/full4d-v1-{revision}/outputs/episode_000017'
    out = base / f'cari_shared_{role}_v1'
    out.mkdir(parents=True)
    rows = {name: write(out / name, ('fixture:' + name).encode(), mode=0o644)
            for name in names if name != 'report.json'}
    report = dict(stage=stage, status='pass', phase='complete', episode_index=17, frames=96,
                  clip_spec=dict(episode_index=17, total_frames=96, camera_name='front_stereo_camera_left', height=1152, width=1536),
                  producer_revision=revision, script_sha256=script_id['sha256'], input_track='track_1',
                  ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], private_truth_read=False,
                  original_frame_indices=list(range(96)), source_inputs_assets_rehashed=True,
                  source_helpers_rehashed=True, source_helpers={'infra/' + script: script_id}, output_files=rows)
    write(out / 'report.json', json.dumps(report).encode(), mode=0o644)
    return root, code, revision, out, pinroot, report


@pytest.mark.parametrize('role', ['prepare', 'forward', 'refined', 'export'])
def test_exact_shared_schema_and_readonly_exclusive_output(monkeypatch, tmp_path, role):
    root, code, rev, out, pinroot, report = fixture(monkeypatch, tmp_path, role)
    value = pins.shared_pin(role, root, code, rev, 17, 96)
    assert value['schema'] == f'world-reward-cari-shared-{role}-pins-v1'
    assert set(value[role + '_files']) == pins.SHARED[role][2]
    assert set(value) == {'schema', 'clip_spec', role, role + '_files'}
    assert json.loads((pinroot / f'shared_{role}_000017.json').read_bytes()) == value
    assert out.stat().st_mode & 0o777 == 0o555
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in out.iterdir())
    with pytest.raises(FileExistsError):
        pins.shared_pin(role, root, code, rev, 17, 96)


@pytest.mark.parametrize('mutation', ['foreign', 'symlink', 'hardlink', 'failed', 'revision', 'script', 'gt', 'frames', 'helper', 'manifest'])
def test_invalid_shared_producer_stops_before_sealing_or_publication(monkeypatch, tmp_path, mutation):
    root, code, rev, out, pinroot, report = fixture(monkeypatch, tmp_path)
    if mutation == 'foreign':
        write(out / 'unowned.txt', b'foreign', mode=0o644)
    elif mutation == 'symlink':
        (out / 'coconet.pth').unlink()
        (out / 'coconet.pth').symlink_to(code / 'infra/cari_full_forward.py')
    elif mutation == 'hardlink':
        (out / 'other').hardlink_to(out / 'coconet.pth')
    else:
        if mutation == 'failed': report['status'] = 'fail'
        elif mutation == 'revision': report['producer_revision'] = 'f' * 40
        elif mutation == 'script': report['script_sha256'] = 'f' * 64
        elif mutation == 'gt': report['ground_truth_used'] = True
        elif mutation == 'frames': report['original_frame_indices'][-1] = 0
        elif mutation == 'helper': report['source_helpers']['infra/cari_full_forward.py']['sha256'] = '0' * 64
        elif mutation == 'manifest': report['output_files']['coconet.pth']['sha256'] = '0' * 64
        (out / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        pins.shared_pin('forward', root, code, rev, 17, 96)
    assert list(pinroot.iterdir()) == []
    assert (out / 'report.json').stat().st_mode & 0o777 == 0o644


def test_historical_namespace_cannot_be_sealed(monkeypatch, tmp_path):
    source = tmp_path / 'historical/report.json'
    expected = write(source, b'unchanged historical receipt', mode=0o644)
    base = tmp_path / 'experiments/new/outputs/episode_000001'
    with pytest.raises(ValueError):
        pins._seal(source, base, expected)
    assert source.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize('case', ['empty', 'writable', 'hardlink', 'oversize', 'symlink'])
def test_identity_rejects_unsafe_or_unbounded_files(tmp_path, case):
    source = tmp_path / 'source'
    write(source, b'x')
    if case == 'empty': source.chmod(0o644); source.write_bytes(b''); source.chmod(0o444)
    elif case == 'writable': source.chmod(0o644)
    elif case == 'hardlink': (tmp_path / 'alias').hardlink_to(source)
    elif case == 'symlink': source.unlink(); source.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError):
        pins.identity(source, maximum=0 if case == 'oversize' else pins.MAX_BYTES)


def test_module_imports_without_numpy_or_torch():
    result = subprocess.run([sys.executable, '-B', '-c',
        "import sys;sys.path[:0]=['infra','src'];import full4d_pins;assert 'numpy' not in sys.modules;assert 'torch' not in sys.modules"],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shared_stage_constants_match_actual_source_without_importing_models():
    import ast
    for script, stage, _ in pins.SHARED.values():
        tree = ast.parse((REPO / 'infra' / script).read_bytes())
        values = [ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'STAGE'
                                                         for target in node.targets)]
        assert values == [stage]


def test_strict_json_duplicate_or_nonfinite_rejected():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            pins._strict(raw)


def test_input_pins_keep_exact_fifteen_sources_and_seal_only_new_files(monkeypatch, tmp_path):
    root, code, rev, _, pinroot, _ = fixture(monkeypatch, tmp_path)
    spec = pins.inputs.PublicClipSpec(17, 96, 'front_stereo_camera_left', 1152, 1536)
    base = pins.episode_output(root, 17)
    names = pins.inputs.source_paths(spec, object_source='surface')
    for name in names:
        write(root / name, b'tiny manufactured payload', mode=0o644)
    script = code / 'infra/cari_prepare.py'
    script_id = write(script, b'# Input fixture source\n')
    report = dict(status='pass', stage='world_reward_native_cari_inputs', episode_index=17,
                  producer_revision=rev, script_sha256=script_id['sha256'], frames=96,
                  input_track='track_1', object_source='surface', original_frame_coverage_verified=True,
                  ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    write(root / pins.inputs.relative_paths(spec)['input_report'], json.dumps(report).encode(), mode=0o644)
    calls = []
    monkeypatch.setattr(pins.inputs, 'validate_reports', lambda *args: calls.append(args))
    value = pins.input_pin(root, code, rev, 17, 96)
    assert len(value['source_files']) == 15 and set(value['source_files']) == names
    assert value['schema'] == 'world-reward-cari-clip-input-pins-v3' and value['object_source'] == 'surface'
    pins.inputs.validate_pins(spec, value)
    assert len(calls) == 1 and all((root / name).stat().st_mode & 0o777 == 0o444 for name in names)
    assert list(pinroot.iterdir()) == [pinroot / 'input_000017.json']
    assert all((root / name).is_relative_to(base) for name in names)


def test_surface_pins_authenticate_native_and_never_seal_historical_reports(monkeypatch, tmp_path):
    root, code, rev, _, pinroot, _ = fixture(monkeypatch, tmp_path)
    code.chmod(0o755)
    base = pins.episode_output(root, 17)
    proposal = base / ('object_budget_surface_' + rev)
    proposal.mkdir()
    firstrev, qrev = 'c' * 40, 'd' * 40
    write(code / 'configs/surface_identity_qualification_pins.json', json.dumps({'producer_revision': firstrev}).encode())
    write(code / 'configs/surface_qslim_qualification_pins.json', json.dumps({'producer_revision': qrev}).encode())
    loader_path = code / 'infra/surface_geometry_loader.py'
    write(loader_path, (REPO / 'infra/surface_geometry_loader.py').read_bytes())
    helpers = pins._literal_tuple(code, 'infra/surface_geometry_loader.py', 'SOURCE_HELPERS') + ('src/world_reward/artifact_paths.py',)
    for name in helpers:
        if not (code / name).exists():
            write(code / name, b'# Manufactured helper fixture\n')
    script_id = write(code / 'infra/object_budget_solid.py', b'# Surface producer fixture\n')
    outputs = {name: write(proposal / name, name.encode())
               for name in pins.SURFACE_FILES - {'report.json', 'native.json'}}
    bound = dict(source_entry='run_full4d_sample', producer_revision=rev,
                 helpers={'infra/object_budget_solid.py': script_id})
    common = dict(status='pass', phase='complete', episode_index=17, producer_revision=rev,
                  source_binding=bound, source_binding_after=bound, source_rehashed_after=True,
                  inputs_qualification_rehashed_after=True, runtime_rehashed_after=True, outputs=outputs)
    native = common | dict(stage='world_reward_object_budget_surface_native_v1', input_track='track_1',
                           input_sha256='f' * 64, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
                           metric_scale_baked_once=0.5, object_scale=1.)
    native_id = write(proposal / 'native.json', json.dumps(native).encode())
    host = common | dict(stage='world_reward_object_budget_surface_host_v1', native=native, native_identity=native_id,
                         owned_container_removed=True, owned_scratch_removed=True)
    write(proposal / 'report.json', json.dumps(host).encode())
    for name in pins.SURFACE_AUXILIARY:
        write(proposal / name, b'audited owned auxiliary')
    proposal.chmod(0o555)
    for name in ('report.json', 'object.glb', 'transform.json', 'intrinsics.json'):
        write(base / 'object_grounded' / name, b'tiny ancestry', mode=0o644)
    write(base / 'scale_smoke/report.json', b'tiny alignment', mode=0o644)
    first = root / f'results/surface-identity-qualify-{firstrev}/native.json'
    write(first, b'original native receipt')
    write(root / f'results/surface-qslim-qualify-{qrev}/native.json', b'original native receipt')
    historical = root / f'results/surface-qslim-qualify-{qrev}/report.json'
    write(historical, b'original failed host receipt', mode=0o644)
    write(root / 'results/surface-qslim-independent-v2/report.json', b'original independent receipt')
    code.chmod(0o555)
    value = pins.surface_pin(root, code, rev, 17)
    assert value['schema'] == 'world_reward.surface_mesh_pins.v1' and len(value['files']) == 15
    assert set(value['source_helpers']) == set(helpers)
    assert value['report']['script_sha256'] == script_id['sha256']
    assert historical.stat().st_mode & 0o777 == 0o644
    assert historical.read_bytes() == b'original failed host receipt'
    assert list(pinroot.iterdir()) == [pinroot / 'surface_mesh_000017.json']
