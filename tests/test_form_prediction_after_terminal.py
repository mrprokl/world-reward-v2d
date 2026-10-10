import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import azure_job
import form_prediction_after_terminal as gate
import form_prediction_transfer as transfer

PRED = 'a'*40
DEV = 'b'*40
REV = 'c'*40
SEQS = ['one', 'two', 'three', 'four']
UNIT = 'world-reward-form-native-sealed-recovery-1010.service'


def seal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    return transfer.seal_json(path, value)


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'root'; (root/'results').mkdir(parents=True)
    monkeypatch.setattr(transfer, 'ROOT', root)
    monkeypatch.setattr(transfer.dev, 'DATA', tmp_path/'public')
    monkeypatch.setattr(transfer, 'cohorts', lambda _: (SEQS, 'd'*40))
    native = dict(producer_revision=PRED, markers={'revision': {'bytes': 41, 'sha256': 'e'*64}},
        closure_sha256='f'*64)
    public_path = transfer.dev.DATA/DEV/'public-transfer.json'
    public = dict(status='pass', dev_revision=DEV, sequences=SEQS,
        private_references_transferred=False, heavy_data_local=False,
        manifest_identity=dict(bytes=10, sha256='0'*64))
    pp = seal(public_path, public)
    cohort = dict(schema='world_reward.form_hoi_external_cohort.v1', status='complete', stage='all',
        producer_revision=PRED, dev_revision=DEV, sequence_order=SEQS, sequences=[],
        ground_truth_used=False, private_truth_read=False, full_4D_accuracy_verified=False,
        reserved_acquired=0, source_binding=native,
        public_transfer=dict(path=str(public_path), pin=pp, manifest_identity=public['manifest_identity'],
            private_references_transferred=False))
    for sid in SEQS:
        inp = transfer.dev.DATA/DEV/sid/'inputs/input.json'
        ip = seal(inp, dict(public_RGB_only=True))
        row = dict(sequence_id=sid, status='complete', input=dict(path=str(inp), pin=ip), stages={})
        for stage in gate.NATIVE_STAGES:
            path = root/'results'/('form-hoi-external-predict-'+PRED)/sid/stage/'report.json'
            if stage.startswith('fit_'):
                report = dict(schema='world_reward.form_external_prediction_stage.v1', status='complete',
                    stage=stage, producer_revision=PRED, sequence_id=sid, dataset='nvidia/form-hoi',
                    input_pin=ip, original_frame_indices=list(range(96)), requested_steps=300,
                    full_predictions_sealed_before_evaluation=True, source_rehashed_after=True,
                    oracle_modes=[], ground_truth_used=False, private_truth_read=False, hand_labeled_test=False,
                    reference_inputs_mounted=False, training_overlap_verified=False, production_adopted=False,
                    source_binding=native, artifacts={'eval_geometry.npz': dict(bytes=100, sha256='1'*64)})
            else:
                report = dict(status='complete', original_reused_producer='2'*40)
            row['stages'][stage] = dict(path=str(path), pin=seal(path, report))
        cohort['sequences'].append(row)
    path = root/'results'/f'form-hoi-external-cohort-{PRED}-all.json'
    cp = seal(path, cohort)
    return root, path, cohort, cp, native


def replace(path, value):
    path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')
    path.chmod(0o444)
    return transfer.identity(path)


def test_all_four_and_eight_final_report_pins_verified_without_geometry_decode(tmp_path, monkeypatch):
    root, path, cohort, cp, native = fixture(tmp_path, monkeypatch)
    result, tracked = gate.cohort_admission(root, PRED, DEV, native)
    assert result == dict(path=str(path), pin=cp, final_stage_reports=8,
        all_four_paired_predictions_sealed=True)
    assert len(tracked) == 14
    assert not list(root.glob('**/eval_geometry.npz'))
    assert all('eval_private' not in p.parts for p, _ in tracked)
    transfer.rehash(tracked)


@pytest.mark.parametrize('fault', ['fail', 'localize_only', 'missing_clip', 'reroll', 'reorder',
    'not_started', 'missing_B', 'foreign_report_path', 'foreign_source', 'private_truth',
    'reserved', 'wrong_DEV', 'changed_public_transfer', 'stage_pin', 'false_shapeclaim'])
def test_incomplete_failed_foreign_or_unsealed_cohort_never_admitted(tmp_path, monkeypatch, fault):
    root, path, c, cp, native = fixture(tmp_path, monkeypatch)
    if fault == 'fail': c['status'] = 'fail'
    elif fault == 'localize_only': c['stage'] = 'localize'
    elif fault == 'missing_clip': c['sequences'].pop()
    elif fault == 'reroll': c['sequences'][-1]['sequence_id'] = 'reserved-other'
    elif fault == 'reorder': c['sequences'].reverse()
    elif fault == 'not_started': c['sequences'][-1]['status'] = 'not_started'
    elif fault == 'missing_B': c['sequences'][-1]['stages'].pop('fit_B')
    elif fault == 'foreign_report_path': c['sequences'][-1]['stages']['fit_B']['path'] = '/tmp/other.json'
    elif fault == 'foreign_source': c['source_binding'] = dict(native, closure_sha256='0'*64)
    elif fault == 'private_truth': c['private_truth_read'] = True
    elif fault == 'reserved': c['reserved_acquired'] = 1
    elif fault == 'wrong_DEV': c['dev_revision'] = '4'*40
    elif fault == 'changed_public_transfer': c['public_transfer']['pin']['sha256'] = '0'*64
    elif fault == 'stage_pin': c['sequences'][-1]['stages']['fit_B']['pin']['sha256'] = '0'*64
    elif fault == 'false_shapeclaim': c['full_4D_accuracy_verified'] = True
    replace(path, c)
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.cohort_admission(root, PRED, DEV, native)


@pytest.mark.parametrize('fault', ['producer', 'GT', 'incomplete', 'steps', 'indices', 'unsealed', 'source'])
def test_complete_cohort_cannot_hide_invalid_last_B(tmp_path, monkeypatch, fault):
    root, path, c, cp, native = fixture(tmp_path, monkeypatch)
    record = c['sequences'][-1]['stages']['fit_B']; report_path = Path(record['path'])
    report = json.loads(report_path.read_bytes())
    if fault == 'producer': report['producer_revision'] = '2'*40
    elif fault == 'GT': report['reference_inputs_mounted'] = True
    elif fault == 'incomplete': report['status'] = 'fail'
    elif fault == 'steps': report['requested_steps'] = 299
    elif fault == 'indices': report['original_frame_indices'].pop()
    elif fault == 'unsealed': report['full_predictions_sealed_before_evaluation'] = False
    elif fault == 'source': report['source_binding'] = dict(native, closure_sha256='0'*64)
    record['pin'] = replace(report_path, report); replace(path, c)
    with pytest.raises(ValueError):
        gate.cohort_admission(root, PRED, DEV, native)


def configure_run(monkeypatch, native):
    binding = dict(producer_revision=REV, entries=10, closure_sha256='7'*64)
    calls = []
    monkeypatch.setattr(transfer, 'source', lambda *args: binding)
    monkeypatch.setattr(gate, 'native_source', lambda _: native)
    terminal = dict(unit=UNIT, LoadState='loaded', ActiveState='inactive', SubState='dead',
        Result='success', ExecMainStatus='0', MainPID='0')
    def wait(unit, **kwargs): calls.append(('wait', unit, kwargs)); return terminal
    monkeypatch.setattr(gate, 'wait_success_unit', wait)
    monkeypatch.setattr(gate, 'snapshot', lambda *args: {k: v for k, v in terminal.items() if k != 'unit'})
    return calls, binding


def test_gate_seals_real_json_path_fields_source_and_actual_cohort_pin(tmp_path, monkeypatch):
    root, path, c, cp, native = fixture(tmp_path, monkeypatch)
    calls, binding = configure_run(monkeypatch, native)
    report = gate.admit(root, PRED, DEV, REV, UNIT)
    receipt = root/'results'/f'form-prediction-terminal-gate-{REV}.json'
    assert json.loads(receipt.read_bytes()) == report
    assert report['status'] == 'pass' and report['source_binding'] == binding
    assert report['cohort']['pin'] == cp and isinstance(report['cohort']['path'], str)
    assert report['publication_started'] is False and report['GPU_lease_acquired'] is False
    assert calls[0][2]['expected_command'] == gate.command_tokens(PRED, DEV)
    assert not receipt.stat().st_mode & 0o222
    with pytest.raises(ValueError, match='One-shot'):
        gate.admit(root, PRED, DEV, REV, UNIT)
    assert len(calls) == 1


def test_failed_predecessor_seals_one_redacted_failure_no_publish(tmp_path, monkeypatch):
    root, path, c, cp, native = fixture(tmp_path, monkeypatch)
    configure_run(monkeypatch, native)
    def failed(*args, **kwargs): raise RuntimeError('SECRET-do-not-print')
    monkeypatch.setattr(gate, 'wait_success_unit', failed)
    monkeypatch.setattr(gate, 'cohort_admission', lambda *args: pytest.fail('No cohort selection after failure'))
    with pytest.raises(RuntimeError):
        gate.admit(root, PRED, DEV, REV, UNIT)
    raw = (root/'results'/f'form-prediction-terminal-gate-{REV}.json').read_bytes()
    assert b'SECRET' not in raw
    report = json.loads(raw)
    assert report['status'] == 'fail' and report['error_type'] == 'RuntimeError'
    assert not report['publication_started']


def test_source_change_across_wait_fails_before_publish(tmp_path, monkeypatch):
    root, path, c, cp, native = fixture(tmp_path, monkeypatch)
    calls, binding = configure_run(monkeypatch, native)
    sources = iter([binding, dict(binding, closure_sha256='0'*64)])
    monkeypatch.setattr(transfer, 'source', lambda *args: next(sources))
    with pytest.raises(ValueError, match='immutable'):
        gate.admit(root, PRED, DEV, REV, UNIT)
    assert json.loads((root/'results'/f'form-prediction-terminal-gate-{REV}.json').read_bytes())['status'] == 'fail'


def test_wrapper_original_call_unchanged_optional_publish_only_and_actual_closure():
    root = Path(__file__).resolve().parents[1]
    wrapper = root/'infra/run_form_prediction_transfer.sh'
    subprocess.run(['bash', '-n', str(wrapper)], check=True)
    text = wrapper.read_text()
    assert '"$3" == publish' in text and '[[ $# == 5' in text
    assert '"$CODE/infra/form_prediction_after_terminal.py" "$2" "$4" "$5"' in text
    assert 'shift 2' in text and text.index('shift 2') < text.index('exec env -i')
    assert '960s python3 -B "$CODE/infra/form_prediction_transfer.py" "$@"' in text
    assert '--gpus' not in text and 'flock' not in text and 'docker' not in text
    files = {str(p.relative_to(root)): p.read_bytes() for folder in ('infra', 'src', 'configs')
        for p in (root/folder).rglob('*') if p.is_file() and
        (p.suffix in ('.py', '.sh', '.json', '.toml', '.cpp', '.hpp', '.h') or p.name.startswith('Dockerfile.'))}
    selected = azure_job.runtime_bundle_paths(files, 'infra/run_form_prediction_transfer.sh')
    assert set(gate.HELPERS) <= set(selected)
    assert 'infra/form_hoi_external_eval.py' not in selected
    tree = root/'infra/form_prediction_after_terminal.py'
    assert 'import numpy' not in tree.read_text() and 'import torch' not in tree.read_text()
