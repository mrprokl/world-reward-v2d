"""Public metadata admission before the unchanged private prediction publisher.

No GPU lease, inference, geometry decoding, reference access or quality-based
selection. All four A/B records must be sealed after the entire producer exits.
"""
import argparse
import json
import os
import sys

import form_prediction_transfer as transfer
from terminal_success import owned_unit, snapshot, unit_ready, wait_success_unit


HELPERS = (*transfer.HELPERS, 'infra/form_prediction_after_terminal.py', 'infra/terminal_success.py')
NATIVE_ENTRY = 'run_form_hoi_external_predict'
NATIVE_HELPERS = ('infra/form_hoi_external_predict.py', 'infra/run_form_hoi_external_predict.sh',
    'configs/form_hoi_external_predict_v1.json')
NATIVE_STAGES = ('localize', 'track', 'body_depth', 'object', 'prepare', 'forward', 'fit_A', 'fit_B')


def command_tokens(predrev, devrev):
    code = transfer.ROOT/'jobs'/predrev/NATIVE_ENTRY/'code'
    return (f'WR_CODE={code}', f'WR_CODE_REVISION={predrev}',
        str(code/'infra/run_form_hoi_external_predict.sh'), '--cohort-stage all', f'--dev-revision {devrev}')


def native_source(predrev):
    return transfer.source(transfer.ROOT, transfer.ROOT/'jobs'/predrev/NATIVE_ENTRY/'code',
        predrev, NATIVE_ENTRY, NATIVE_HELPERS)


def cohort_admission(code, predrev, devrev, native):
    seqs, _ = transfer.cohorts(code)
    path = transfer.ROOT/'results'/f'form-hoi-external-cohort-{predrev}-all.json'
    cohort_pin = transfer.identity(path, transfer.MAX_REPORT)
    cohort = transfer.pinned(path, cohort_pin, transfer.MAX_REPORT)
    transfer.require(cohort.get('schema') == 'world_reward.form_hoi_external_cohort.v1' and
        cohort.get('status') == 'complete' and cohort.get('stage') == 'all' and
        cohort.get('producer_revision') == predrev and cohort.get('dev_revision') == devrev and
        cohort.get('sequence_order') == seqs and
        all(cohort.get(key) is False for key in ('ground_truth_used', 'private_truth_read', 'full_4D_accuracy_verified')) and
        type(cohort.get('reserved_acquired')) is int and cohort['reserved_acquired'] == 0 and
        type(cohort.get('sequences')) is list and len(cohort['sequences']) == 4 and
        [row.get('sequence_id') for row in cohort['sequences']] == seqs,
        'Complete original all-four public-only cohort required')
    binding = cohort.get('source_binding', {})
    transfer.require(all(binding.get(key) == native[key] for key in
        ('producer_revision', 'markers', 'closure_sha256')), 'Cohort must bind authenticated actual native source')
    public_path = transfer.dev.DATA/devrev/'public-transfer.json'
    public_pin = transfer.identity(public_path, transfer.MAX_METADATA)
    public = transfer.pinned(public_path, public_pin, transfer.MAX_METADATA)
    transfer.pin(public.get('manifest_identity'), 65536)
    transfer.require(public.get('status') == 'pass' and public.get('dev_revision') == devrev and
        public.get('sequences') == seqs and public.get('private_references_transferred') is False and
        public.get('heavy_data_local') is False and cohort.get('public_transfer') == dict(
            path=str(public_path), pin=public_pin, manifest_identity=public.get('manifest_identity'),
            private_references_transferred=False), 'Original public input transfer receipt differs')
    tracked = [(path, cohort_pin), (public_path, public_pin)]
    for row in cohort['sequences']:
        sid = row['sequence_id']
        inp = transfer.dev.DATA/devrev/sid/'inputs/input.json'
        ip = transfer.identity(inp, 16384)
        transfer.require(row.get('status') == 'complete' and row.get('input') == dict(path=str(inp), pin=ip) and
            type(row.get('stages')) is dict and set(row['stages']) == set(NATIVE_STAGES),
            'Every sequence and its complete native stage inventory must succeed')
        tracked.append((inp, ip))
        for variant, stage in transfer.STAGES.items():
            report_path = transfer.ROOT/'results'/('form-hoi-external-predict-'+predrev)/sid/stage/'report.json'
            declared = row['stages'][stage]
            transfer.require(type(declared) is dict and set(declared) == {'path', 'pin'} and
                declared['path'] == str(report_path), 'Final stage must match original cohort path and pin')
            report = transfer.pinned(report_path, declared['pin'], transfer.MAX_REPORT)
            gp = transfer.pin(report.get('artifacts', {}).get('eval_geometry.npz'), transfer.MAX_GEOMETRY)
            transfer.stage_contract(report, predrev, sid, variant, ip, gp, native)
            tracked.append((report_path, declared['pin']))
    transfer.rehash(tracked)
    return dict(path=str(path), pin=cohort_pin, final_stage_reports=8,
        all_four_paired_predictions_sealed=True), tracked


def admit(code, predrev, devrev, revision, unit):
    result_path = transfer.ROOT/'results'/f'form-prediction-terminal-gate-{revision}.json'
    transfer.require(not result_path.exists(), 'One-shot scheduling gate; never overwrite or retry a failed receipt')
    report = dict(schema='world_reward.form_prediction_terminal_gate.v1', status='fail',
        producer_revision=revision, prediction_revision=predrev, dev_revision=devrev, unit=owned_unit(unit),
        GPU_lease_acquired=False, ground_truth_read=False, geometry_decoded=False, publication_started=False)
    try:
        before = transfer.source(transfer.ROOT, code, revision, transfer.ENTRY, HELPERS)
        native = native_source(predrev)
        expected = command_tokens(predrev, devrev)
        state = wait_success_unit(unit, expected_command=expected)
        report['terminal_state'] = state
        cohort, tracked = cohort_admission(code, predrev, devrev, native)
        report['cohort'] = cohort
        transfer.require(unit_ready(snapshot(owned_unit(unit), expected)), 'Producer changed before publication')
        transfer.rehash(tracked)
        transfer.require(native_source(predrev) == native and transfer.source(
            transfer.ROOT, code, revision, transfer.ENTRY, HELPERS) == before,
            'Actual native and publication source must remain immutable across scheduling')
        report.update(status='pass', source_binding=before, native_source_binding=native)
    except Exception as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        transfer.seal_json(result_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('unit', type=owned_unit)
    parser.add_argument('prediction_revision', type=transfer.revision)
    parser.add_argument('dev_revision', type=transfer.revision)
    args = parser.parse_args()
    transfer.require(sys.platform == 'linux' and os.geteuid() == 0 and
        os.uname().nodename == 'scenesmith-ncc-h100-01', 'CPU-only Azure01 publication scheduler required')
    result = admit(transfer.canonical(os.environ['WR_CODE']), args.prediction_revision, args.dev_revision,
        transfer.revision(os.environ['WR_CODE_REVISION']), args.unit)
    print(json.dumps(dict(status='publication_admitted', cohort=result['cohort'], unit=args.unit)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='fail', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
