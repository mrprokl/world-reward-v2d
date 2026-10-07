"""Tiny orchestration contracts; no actual Docker or media execution."""
from pathlib import Path
import json
import pytest
import qwen4d_initializers as pilot


@pytest.fixture
def command(monkeypatch):
    revision = 'a'*40
    monkeypatch.setenv('WR_CODE_REVISION', revision)
    prefix = f'experiments/qwen4d-v1-{revision}/outputs'
    monkeypatch.setenv('WR_OUTPUT_PREFIX', prefix)
    code = pilot.ROOT/f'jobs/{revision}/{pilot.ENTRY}/code'
    cfg = json.loads((Path(__file__).resolve().parents[1]/pilot.CONFIG).read_text())
    outputs = pilot.ROOT/prefix
    preview = outputs.parent/'previews/episode_000008'
    def make(preview_mode=False):
        return pilot.worker_command(code, outputs, preview, cfg, 8,
            'infra/qwen4d_preview.py' if preview_mode else 'infra/body_smoke.py', (),
            cfg['body_image'], 'world-reward-tiny-test', revision)
    return make, outputs, preview


def test_frozen_scope_and_stage_order():
    assert [s[0] for s in pilot.STAGES] == ['body_smoke', 'depth_smoke', 'scale_smoke', 'object_grounded']
    assert pilot.STAGES[0][2] == ('--inference-type', 'body')
    assert pilot.STAGES[3][2] == ('--aligned-pointmap',)
    assert all('--full-video' not in s[2] for s in pilot.STAGES)


def test_candidate_writes_only_isolated_outputs(command):
    make, outputs, _ = command
    args = make()
    mounts = [args[i+1] for i, a in enumerate(args) if a == '--mount']
    assert [m for m in mounts if not m.endswith(',readonly')] == [f'type=bind,src={outputs},dst={outputs}']
    assert all('track_2' not in m and 'track_3' not in m for m in mounts)
    assert all(str(pilot.ROOT/'outputs') not in m for m in mounts)
    assert args[args.index('--network')+1] == 'none'
    assert '--read-only' in args and '--gpus' in args
    assert 'MOMENTUM_ENABLED=0' in args


def test_preview_is_cpu_only_saved_baseline_readonly(command):
    make, outputs, preview = command
    args = make(True)
    mounts = [args[i+1] for i, a in enumerate(args) if a == '--mount']
    assert '--gpus' not in args
    assert f'type=bind,src={outputs},dst={outputs},readonly' in mounts
    baseline = pilot.ROOT/'outputs/episode_000008/cari_shared_export_v1'
    assert f'type=bind,src={baseline},dst={baseline},readonly' in mounts
    assert [m for m in mounts if not m.endswith(',readonly')] == [f'type=bind,src={preview},dst={preview}']


def test_launcher_has_bounded_no_credentials_environment():
    text = (Path(__file__).resolve().parents[1]/'infra/run_qwen4d_initializers.sh').read_text()
    assert '/usr/bin/env -i' in text and '4500s' in text
    assert 'WR_CODE_REVISION="$REV"' in text
    assert 'set +x' in text
