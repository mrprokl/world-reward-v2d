"""Fresh Azure-only acquisition of the already pinned DWPose CPU assets.

No inference, global package installation or modification of historical runs.
The known Flatbuffers wheel has an external primary license, not an invented
embedded license. Original acquisition and audit functions are reused unchanged.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwpose_acquire as acq
import dwpose_wheel_audit as notices
import mediapipe_cpu_runtime_verify as rt

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_joint_pair_dwpose_acquire'
BASE = 'weights/dwpose_joint_pair_v1'
REPORT = 'results/dwpose-joint-pair-acquisition-v1.json'
BUDGET = 600
HELPERS = ('infra/joint_pair_dwpose_acquire.py',
           'infra/run_joint_pair_dwpose_acquire.sh', 'infra/dwpose_acquire.py',
           'infra/dwpose_wheel_audit.py', 'infra/mediapipe_cpu_runtime_verify.py')


def source(code, revision):
    result = rt.source(ROOT, code, revision, ENTRY, HELPERS)
    rt.require(result['helpers']['infra/dwpose_acquire.py']['sha256'] == notices.SOURCE_SHA,
               'Unchanged pinned acquisition implementation required')
    return result


def final_assets(base, rows, wheel_audits):
    expected = {name: dict(bytes=size, sha256=sha) for name, size, sha, _ in acq.ASSETS}
    rt.require({r['file']: {k: r[k] for k in ('bytes', 'sha256')} for r in rows} == expected,
               'Complete original nine-asset inventory required')
    files = {}
    for name, pin in expected.items():
        rt.require(rt.identity(base / name, 200_000_000) == pin, 'Original asset changed')
        files[name] = pin
    for row in wheel_audits:
        for item in row['retained_texts']:
            name = str(acq.safe_relative(item['file']))
            p = base / 'notices' / row['package'] / name
            pin = {k: item[k] for k in ('bytes', 'sha256')}
            rt.require(rt.identity(p, 2_000_000) == pin, 'Retained primary notice changed')
            files['notices/' + row['package'] + '/' + name] = pin
    rt.require([x['package'] for x in wheel_audits] == ['onnxruntime', 'flatbuffers'],
               'Both unchanged wheel notice audits required')
    return files


def acquire(base, report, persist, downloader=acq.download, metadata=acq.primary_metadata):
    report['primary_metadata'] = metadata()
    persist()
    for name, size, sha, url in acq.ASSETS:
        report.update(phase='public_pinned_download', active_file=name)
        persist()
        target = base / str(acq.safe_relative(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        downloader(url, target, size, sha)
        report['files'].append(dict(file=name, bytes=size, sha256=sha, url=url))
        persist()
    ort = next(r for r in acq.ASSETS if r[0].startswith('wheels/onnxruntime-'))
    flat = next(r for r in acq.ASSETS if r[0].startswith('wheels/flatbuffers-'))
    ort_audit = acq.audit_wheel(base / ort[0], 'onnxruntime', base / 'notices/onnxruntime')
    rt.require(ort_audit['member_count'] == 353, 'Original ORT wheel inventory required')
    ort_audit.update(embedded_license_present=True, external_primary_license_verified=False)
    flat_audit = notices.audit_flatbuffers(base / flat[0], base / 'licenses/flatbuffers-LICENSE',
                                         base / 'notices/flatbuffers', acq)
    report['wheel_audits'] = [ort_audit, flat_audit]
    report['final_asset_pins'] = final_assets(base, report['files'], report['wheel_audits'])
    report.pop('active_file', None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    code = Path(os.environ['WR_CODE'])
    revision = os.environ['WR_CODE_REVISION']
    rt.require(os.geteuid() == 1000 and os.uname().sysname == 'Linux'
               and os.uname().nodename == 'world-reward-ncc-h100-02'
               and os.environ.get('WR_ROOT') == str(ROOT)
               and os.environ.get('WR_OUTPUT_RESERVED') == '1', 'Reserved Azure VM02 UID1000 required')
    base, path = rt.canonical(ROOT / BASE), rt.canonical(ROOT / REPORT)
    rt.require(base.is_dir() and not any(base.iterdir()) and not path.exists(),
               'Fresh independent acquisition, never historical overwrite')
    before = source(code, revision)
    report = dict(schema='world_reward.joint_pair_dwpose_acquisition.v1', status='fail',
                  producer_revision=revision, source_binding=before, phase='start', files=[],
                  budget_seconds=BUDGET, network='public_pinned_https_only', device='cpu',
                  source_license='Apache-2.0', publisher_model_license='Apache-2.0',
                  ort_license='MIT', flatbuffers_license='Apache-2.0',
                  training_sources=['COCO-WholeBody', 'UBody'],
                  training_rights_verified=False, training_overlap_verified=False,
                  challenge_overlap_verified=False, license_clearance_verified=False,
                  gpu_used=False, packages_installed=False, upstream_source_executed=False,
                  inference_performed=False, accuracy_verified=False, adopted=False,
                  challenge_inputs_used=False, ground_truth_used=False, hand_labeled_test=False,
                  historical_outputs_modified=False, credentials_used=False, oracle_modes=[])
    started = time.monotonic()
    with path.open('x') as stream:
        def persist():
            import json
            report['elapsed_seconds'] = time.monotonic() - started
            stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.truncate(); stream.flush(); os.fsync(stream.fileno())

        def expired(*_):
            raise TimeoutError('Pinned CPU acquisition budget exhausted')

        signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGTERM, expired)
        signal.alarm(BUDGET)
        try:
            persist(); acquire(base, report, persist)
            rt.require(source(code, revision) == before, 'Complete source closure changed')
            report.update(status='pass', phase='complete', source_and_assets_rehashed_after=True)
        except Exception as error:
            report.update(status='fail', error_type=type(error).__name__,
                          error='Pinned acquisition failed; URLs and credentials not logged')
            raise RuntimeError('Fresh pinned DWPose acquisition failed') from None
        finally:
            signal.alarm(0)
            for p in base.rglob('*'):
                if p.is_file(): p.chmod(0o444)
            for p in sorted((x for x in base.rglob('*') if x.is_dir()), reverse=True):
                p.chmod(0o555)
            base.chmod(0o555); persist(); path.chmod(0o444)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
