"""One isolated source certificate; no adoption or historical gate replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_certified_solid_source'
QUALIFICATION = 'configs/certified_solid_qualification_pins.json'
LINEAGE = 'configs/certified_solid_source_lineage_pins.json'
ORIGINAL_PINS = 'configs/mesh_precision_diagnostic_v1.json'
HOST_SECONDS, NATIVE_SECONDS = 420, 315


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def helpers(code):
    require(code.is_absolute() and code.resolve() == code and
            not any(p.is_symlink() for p in (code, *code.parents)), 'Canonical source required')
    sys.path.insert(0, str(code / 'infra'))
    import certified_solid_build as build
    return build


def binding(code, revision, build):
    require(re.fullmatch('[0-9a-f]{40}', revision) and
            code == ROOT / 'jobs' / revision / ENTRY / 'code' and
            Path(__file__).resolve() == code / 'infra/certified_solid_source_job.py', 'Exact diagnostic source namespace required')
    rows = {}
    for p in (code, *sorted(code.rglob('*'))):
        require(p.resolve() == p and not p.is_symlink() and not p.stat().st_mode & 0o222, 'Full readonly source required')
        if p.is_file():
            rows[p.relative_to(code).as_posix()] = build.identity(p, readonly=True, maximum=2 << 20, empty=True)
        else:
            require(p.is_dir(), 'Regular source closure required')
    markers = {k: build.identity(code.parent / k, readonly=True, maximum=128) for k in ('revision', 'source-sha256')}
    require((code.parent / 'revision').read_bytes() == (revision + '\n').encode() and
            re.fullmatch('[0-9a-f]{64}\n', (code.parent / 'source-sha256').read_text()), 'Exact dispatch markers required')
    return dict(files=rows, markers=markers, producer_revision=revision)


def original_binding(code, build, episode):
    lineage = build.strict_json((code / LINEAGE).read_bytes())
    require(set(lineage) == {'schema', 'episode_index', 'producer_revision', 'entrypoint', 'files', 'archive_sha256', 'ledger_sha256'} and
            lineage['schema'] == 'world_reward.certified_solid_source_lineage.v1' and
            type(lineage['episode_index']) is int and lineage['episode_index'] == episode and
            re.fullmatch('[0-9a-f]{40}', lineage['producer_revision']) and
            lineage['entrypoint'] in ('run_track1_frontends', 'run_track1_frontends_queued', 'run_track1_initializers_only'),
            'Independent original source lineage required')
    source = ROOT / 'jobs' / lineage['producer_revision'] / lineage['entrypoint'] / 'code'
    digest = hashlib.sha256(); count = 0
    for p in (source, *sorted(source.rglob('*'))):
        require(p.resolve() == p and not p.is_symlink() and not p.stat().st_mode & 0o222, 'Original source changed/aliased')
        if p.is_file():
            pin = build.identity(p, readonly=True, maximum=2 << 20, empty=True)
            digest.update(p.relative_to(source).as_posix().encode() + b'\0' + str(p.stat().st_mode & 0o777).encode() + b'\0' + bytes.fromhex(pin['sha256']))
            count += 1
        else:
            require(p.is_dir(), 'Original source is not regular')
    require(count == lineage['files'] and digest.hexdigest() == lineage['ledger_sha256'] and
            (source.parent / 'revision').read_bytes() == (lineage['producer_revision'] + '\n').encode() and
            (source.parent / 'source-sha256').read_bytes() == (lineage['archive_sha256'] + '\n').encode(), 'Original full Git closure differs')
    return lineage


def qualification(code, build):
    pins = build.strict_json((code / QUALIFICATION).read_bytes())
    require(pins['schema'] == 'world_reward.certified_solid_qualification_pins.v1' and pins['qualified_controls'] == 15,
            'Actual qualified native pins required')
    out = ROOT / 'results' / ('certified-solid-build-' + pins['producer_revision'])
    paths = {name: out / leaf for name, leaf in (('report', 'report.json'), ('native', 'native.json'), ('binary', 'certified_solid_query'))}
    measured = {name: build.identity(p, readonly=True, maximum=2 << 20) for name, p in paths.items()}
    require(all(measured[k] == pins[k] for k in paths), 'Qualified report/native/binary byte pins differ')
    host = build.strict_json(paths['report'].read_bytes()); native = build.strict_json(paths['native'].read_bytes())
    require(host['status'] == native['status'] == 'pass' and host['phase'] == native['phase'] == 'complete' and
            host['native'] == native and host['retained_binary'] == native['compiled_binary'] == pins['binary'] and
            host['child_image_id'] == native['image_id'] == pins['child_image_id'] and
            host['parent_image_id'] == pins['parent_image_id'] == build.PARENT and
            host['producer_revision'] == pins['producer_revision'] and
            host['source_binding'] == native['source_binding'] == host['source_binding_after'] == native['source_binding_after'] and
            native['source_binding']['native_source'] == pins['native_source'], 'Actual completed build source/runtime differs')
    require(all(host[k] is True for k in ('source_rehashed_after', 'parent_rehashed_after', 'build_inputs_rehashed_after',
            'disposable_build_inputs_removed', 'child_image_parent_verified')), 'Build integrity qualification incomplete')
    for record in (host, native):
        require(all(record[k] is False for k in ('gpu_used', 'gt_used', 'adoption', 'production_mesh_validated',
                'reconstruction_accuracy_verified', 'competition_eligibility_verified')), 'Build scope differs')
    build.validate_controls(native['controls'], pins['native_source']['sha256'])
    return pins, paths, measured


def native(code, revision, episode, work, build):
    started = time.monotonic()
    require(os.getuid() == 1000 and os.environ.get('WR_NATIVE_NETWORK') == 'none' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and work.stat().st_uid == 1000 and
            stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()), 'Isolated fresh UID1000 CPU work required')
    before = binding(code, revision, build)
    sys.path.insert(0, str(code / 'src'))
    import certified_solid_source as source
    import mesh_precision_diagnostic as precision
    report = dict(stage='certified_solid_source_native_job_v1', status='fail', episode_index=episode,
                  source_binding=before, gpu_used=False, adoption=False, gt_used=False, reconstruction_accuracy_verified=False)
    try:
        pins, paths, measured = qualification(code, build)
        require(os.environ.get('WR_CPU_IMAGE_ID') == pins['child_image_id'], 'Actual qualified CPU image differs')
        original, provenance = precision.bindings(ROOT, episode, code / ORIGINAL_PINS)
        report.update(qualified_runtime=pins, original_source=provenance)
        try:
            report['certificate'] = source.certify_source(original['glb'], paths['binary'], pins['native_source']['sha256'])
        except source.SourceCertificationError as error:
            report['certificate'] = error.report
            raise
        require(qualification(code, build)[2] == measured and
                precision.bindings(ROOT, episode, code / ORIGINAL_PINS)[1] == provenance, 'Qualified/source evidence changed')
        report.update(status='pass', phase='complete')
    except Exception as error:
        report['failure_type'] = type(error).__name__
    finally:
        try:
            report['source_binding_after'] = binding(code, revision, build)
            require(report['source_binding_after'] == before, 'Diagnostic source changed')
            report['source_rehashed_after'] = True
        except Exception as error:
            report.update(status='fail', posthash_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic() - started
        if report['elapsed_seconds'] > NATIVE_SECONDS:
            report.update(status='fail', failure_type='NativeInclusiveDeadline')
        build.write_json(work / 'native.json', report)
    return 0 if report['status'] == 'pass' else 1


def host(code, revision, episode, build):
    started = time.monotonic()
    require(sys.platform == 'linux' and os.getuid() == 0 and
            os.environ.get('DOCKER_HOST') == 'unix://' + str(ROOT / 'docker.sock'), 'Owned Linux control host required')
    before = binding(code, revision, build)
    pins, paths, measured = qualification(code, build)
    lineage = original_binding(code, build, episode)
    original = build.strict_json((code / ORIGINAL_PINS).read_bytes())
    require(original['episode_index'] == episode and original['original_producer_revision'] == lineage['producer_revision'], 'Source provenance episode/revision differs')
    source_paths = [ROOT / row['path'] for row in original['files'].values()]
    for p, row in zip(source_paths, original['files'].values()):
        require(build.identity(p, maximum=256 << 20) == {k: row[k] for k in ('bytes', 'sha256')}, 'Original source evidence differs')
    image = pins['child_image_id']
    require(build.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], 30).decode().strip() == image, 'Actual CPU image absent/differs')
    out = ROOT / 'results' / f'certified-solid-source-{episode:06d}-{revision}'
    require(not out.exists() and not out.is_symlink(), 'Fresh diagnostic output required')
    out.mkdir(mode=0o755); os.chmod(out, 0o755)
    work = out / 'disposable'; work.mkdir(mode=0o700); os.chown(work, 1000, 1000)
    cid = out / '.container.cid'; name = f'wr-certified-source-{episode:06d}-{revision}'
    report = dict(stage='certified_solid_source_host_job_v1', status='fail', episode_index=episode, producer_revision=revision,
                  source_binding=before, original_lineage=lineage, qualified_runtime=pins, image_id=image,
                  gpu_used=False, gt_used=False, adoption=False, reconstruction_accuracy_verified=False,
                  historical_gate_reclassified=False)
    old_handler = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Host diagnostic deadline')))
    signal.alarm(max(1, int(HOST_SECONDS - (time.monotonic() - started))))
    try:
        mounts = []
        for p in (code, code.parent / 'revision', code.parent / 'source-sha256', *paths.values(), *source_paths):
            mounts.extend(['--mount', f'type=bind,src={p},dst={p},readonly'])
        argv = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label', 'world_reward.certified_solid.owner=' + revision,
                '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--cpus', '4', '--memory', '16g', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=256m', *mounts,
                '--mount', f'type=bind,src={work},dst={work}', '--entrypoint', '/usr/bin/env', image, '-i',
                'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'MPLCONFIGDIR=/tmp/mpl',
                'WR_CODE=' + str(code), 'WR_CODE_REVISION=' + revision, 'WR_CPU_IMAGE_ID=' + image,
                'WR_NATIVE_NETWORK=none', 'CUDA_VISIBLE_DEVICES=-1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                'PYTHONDONTWRITEBYTECODE=1', 'python3', '-I', '-B', str(code / 'infra/certified_solid_source_job.py'),
                '--native', '--episode', str(episode)]
        build.run(argv, NATIVE_SECONDS + 5, out / 'disposable-native.log')
        report.update(status='pass', phase='complete')
    except Exception as error:
        report['failure_type'] = type(error).__name__
        log = out / 'disposable-native.log'
        if log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 1000))
                report['technical_failure_tail'] = stream.read(1000).decode('utf-8', errors='replace')
    finally:
        if report['status'] != 'pass':
            signal.alarm(60)  # Cleanup-only grace; an over-budget run remains FAIL.
        try:
            build.cleanup_container(name, image, revision, cid)
            report['owned_container_removed'] = True
        except Exception as error:
            report.update(status='fail', cleanup_failure_type=type(error).__name__)
        try:
            if (work / 'native.json').exists():
                report['native'] = build.strict_json((work / 'native.json').read_bytes())
                require(report['native']['source_binding'] == before and report['native']['source_rehashed_after'] is True and
                        report['native']['source_binding_after'] == before, 'Native diagnostic source receipt differs')
                require(report['native']['stage'] == 'certified_solid_source_native_job_v1' and
                        type(report['native']['episode_index']) is int and report['native']['episode_index'] == episode and
                        all(report['native'][k] is False for k in ('gpu_used', 'gt_used', 'adoption', 'reconstruction_accuracy_verified')),
                        'Native diagnostic receipt scope/episode differs')
                if report['native']['status'] != 'pass':
                    report['status'] = 'fail'
                else:
                    certificate = report['native']['certificate']
                    require(certificate['status'] == 'pass' and certificate['phase'] == 'complete' and
                            certificate['native_attempts'] == 1 and certificate['native_returned'] is True and
                            certificate['represented_embedding_certified'] is True and certificate['source_binary_helpers_rehashed_after'] is True and
                            all(certificate[k] is False for k in ('ground_truth_used', 'adoption', 'qem_executed',
                                'geometry_repaired', 'orientation_changed', 'packing_validated', 'reconstruction_accuracy_verified')),
                            'Complete native represented geometry certificate required')
            else:
                report['status'] = 'fail'
            require(qualification(code, build)[2] == measured and original_binding(code, build, episode) == lineage, 'Original/runtime source changed afterward')
            for p, row in zip(source_paths, original['files'].values()):
                require(build.identity(p, maximum=256 << 20) == {k: row[k] for k in ('bytes', 'sha256')}, 'Original artifact changed afterward')
            report['source_binding_after'] = binding(code, revision, build)
            require(report['source_binding_after'] == before, 'Diagnostic source changed afterward')
            report['source_original_runtime_rehashed_after'] = True
        except Exception as error:
            report.update(status='fail', posthash_failure_type=type(error).__name__)
        if work.exists(): shutil.rmtree(work)
        (out / 'disposable-native.log').unlink(missing_ok=True)
        report['disposable_removed'] = not work.exists()
        report['elapsed_seconds'] = time.monotonic() - started
        if report['elapsed_seconds'] > HOST_SECONDS: report.update(status='fail', failure_type='InclusiveDeadline')
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
        build.write_json(out / 'report.json', report); os.chmod(out, 0o555)
    return 0 if report['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--episode', type=int, choices=range(30), required=True)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    build = helpers(code)
    work = ROOT / 'results' / f'certified-solid-source-{args.episode:06d}-{revision}' / 'disposable'
    return native(code, revision, args.episode, work, build) if args.native else host(code, revision, args.episode, build)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('certified_solid_source_job FAIL: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
