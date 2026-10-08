"""Tiny manufactured immutable receipts; no local source RGB or inference."""
import json
import pytest
import vl_localization_preview as recovery
from mediapipe_cpu_runtime_verify import identity


def saved(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(recovery, 'INFERENCE', root)
    def write(name, value):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
        path.chmod(0o444)
        return identity(path)
    rp = write('report.json', dict(status='fail', phase='preview', producer_revision=recovery.PRODUCER,
                                 source_rehashed_after=True))
    monkeypatch.setattr(recovery, 'FAILED_REPORT_SHA', rp['sha256'])
    names = {name: write('episode_000009/' + name, {})
             for name in ('observations.json', 'sam-proposed-input.json')}
    write('inference.json', dict(status='complete_diagnostic_not_quality_pass', producer_revision=recovery.PRODUCER,
         requested_images=152, sam_executed=False, baseline_modified=False, quality_verified=False,
         inputs_rehashed_after=True, model_rehashed_after=True, episodes={'9': names}))
    write('batch-control.json', {})
    write('observations-inputs.json', {})
    return root


def test_saved_pins_are_stable_and_complete(tmp_path, monkeypatch):
    root = saved(tmp_path, monkeypatch)
    before = recovery.pins()
    assert len(before) == 6 and before == recovery.pins()
    assert str(root / 'episode_000009/sam-proposed-input.json') in before


def test_saved_report_identity_required(tmp_path, monkeypatch):
    saved(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, 'FAILED_REPORT_SHA', '0' * 64)
    with pytest.raises(ValueError, match='Exact original failed'): recovery.pins()


def test_saved_observation_change_rejected(tmp_path, monkeypatch):
    root = saved(tmp_path, monkeypatch)
    path = root / 'episode_000009/observations.json'
    path.chmod(0o644); path.write_text('{"changed":true}'); path.chmod(0o444)
    with pytest.raises(ValueError, match='Original observations changed'): recovery.pins()


def test_native_rejects_local_execution_before_rendering(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.sys, 'platform', 'darwin')
    with pytest.raises(ValueError, match='Offline CPU-only'): recovery.native(tmp_path, tmp_path)
