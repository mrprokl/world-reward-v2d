"""Four frozen new CPU controls; qualification is not adoption or HOI accuracy."""
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
ENTRY = 'run_solid_chart_v2_qualify'
BUILD_PINS = 'configs/solid_chart_v2_build_pins.json'
NATIVE_SECONDS, HOST_SECONDS = 1800, 1900
OFFICIAL = 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
OFFICIAL_PIN = dict(bytes=2031, sha256='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0')
HELPERS = ('infra/solid_chart_v2_qualify.py', 'infra/run_solid_chart_v2_qualify.sh',
    'infra/solid_chart_v2_build.py', 'infra/certified_solid_build.py', 'infra/certified_solid_source_job.py',
    'infra/oriented_solid_compiler.py', 'infra/oriented_solid_controls_v2.py',
    'infra/mesh_conditioned_chart_v2_source.py', 'infra/mesh_conditioned_chart_v2.hpp',
    'src/world_reward/mesh_conditioning_v2.py', BUILD_PINS)


def require(ok, reason):
    if not ok: raise ValueError(reason)


def official_identity(build):
    """Authenticate historical 0644 bytes; isolation is the unchanged RO bind.

    Never chmod the official producer or mistake its original mode for a
    writable container input. Every host/native pre/postcheck uses this gate.
    """
    path = ROOT/OFFICIAL
    before = path.lstat()
    require(stat.S_IMODE(before.st_mode) == 0o644, 'Original official helper mode differs')
    pin = build.identity(path, readonly=False)
    after = path.lstat()
    require(pin == OFFICIAL_PIN and all(getattr(before, k) == getattr(after, k) for k in
        ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
        'Original unmodified official helper changed')
    return pin | dict(original_mode=0o644)


def helpers(code):
    require(code.is_absolute() and code.resolve() == code and not any(p.is_symlink() for p in (code, *code.parents)), 'Canonical source required')
    sys.path[:0] = [str(code/'infra'), str(code/'src')]
    import certified_solid_build as build
    import certified_solid_source_job as certificate
    import solid_chart_v2_build as built
    return build, certificate, built


def snapshot(code, revision, build):
    rows = {}; ledger = hashlib.sha256()
    for p in (code, *sorted(code.rglob('*'))):
        m = p.lstat()
        require(p.resolve() == p and not p.is_symlink() and not m.st_mode & 0o222, 'Complete canonical readonly source required')
        if p.is_dir(): continue
        row = build.identity(p, readonly=True, maximum=2 << 20, empty=True)
        name = p.relative_to(code).as_posix(); rows[name] = row
        ledger.update(name.encode()+b'\0'+str(stat.S_IMODE(m.st_mode)&0o555).encode()+b'\0'+bytes.fromhex(row['sha256']))
    require(set(p.name for p in code.parent.iterdir()) == {'code', 'revision', 'source-sha256'}, 'Exact immutable snapshot required')
    markers = {n: build.identity(code.parent/n, readonly=True, maximum=128) for n in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Original dispatch markers differ')
    return rows, dict(producer_revision=revision, markers=markers, source_files=len(rows),
        source_files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        source_readonly_ledger_sha256=ledger.hexdigest())


def binding(code, revision, build):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/ENTRY/'code' and
            Path(__file__).resolve() == code/'infra/solid_chart_v2_qualify.py', 'Actual new qualification entry required')
    rows, bound = snapshot(code, revision, build)
    require(set(HELPERS) <= set(rows), 'Complete native qualification closure required')
    return bound | dict(helpers={n: rows[n] for n in HELPERS})


def built_qualification(code, build, certificate, built):
    """Authenticate independently pinned artifacts and historical source before calls."""
    pin_identity = build.identity(code/BUILD_PINS, readonly=True, maximum=16384)
    pins = build.strict_json((code/BUILD_PINS).read_bytes())
    keys = {'schema', 'producer_revision', 'report', 'native', 'binary', 'image_id', 'source_archive_sha256',
            'source_files', 'source_files_sha256', 'source_readonly_ledger_sha256'}
    require(set(pins) == keys and pins['schema'] == 'world_reward.solid_chart_v2_build_pins.v1' and
            re.fullmatch('[0-9a-f]{40}', pins['producer_revision']) and type(pins['source_files']) is int and pins['source_files'] > 0 and
            all(type(pins[k]) is str and re.fullmatch('[0-9a-f]{64}', pins[k]) for k in
                ('source_archive_sha256', 'source_files_sha256', 'source_readonly_ledger_sha256')), 'Completed independent build pins required')
    producer = ROOT/'jobs'/pins['producer_revision']/'run_solid_chart_v2_build'/'code'
    rows, bound = snapshot(producer, pins['producer_revision'], build)
    require(all(bound[k] == pins[k] for k in ('source_files', 'source_files_sha256', 'source_readonly_ledger_sha256')) and
            (producer.parent/'source-sha256').read_bytes() == (pins['source_archive_sha256']+'\n').encode(), 'Original full source ledger differs')
    out = ROOT/'results'/('solid-chart-v2-build-'+pins['producer_revision'])
    require(out.resolve() == out and not out.is_symlink() and stat.S_IMODE(out.lstat().st_mode) == 0o555, 'Immutable build directory required')
    paths = {k: out/n for k, n in (('report', 'report.json'), ('native', 'native.json'), ('binary', 'mesh_conditioned_chart_v2'))}
    for k, p in paths.items():
        require(build.identity(p, readonly=True, maximum=2 << 20) == pins[k] and
                stat.S_IMODE(p.lstat().st_mode) == (0o555 if k == 'binary' else 0o444), 'Frozen built artifact differs')
    host, native = (build.strict_json(paths[k].read_bytes()) for k in ('report', 'native'))
    actual_binding = {k: bound[k] for k in ('producer_revision', 'markers', 'source_files', 'source_files_sha256')} | dict(helpers={n: rows[n] for n in built.HELPERS})
    require(host['stage'] == 'solid_chart_v2_build_host_v1' and native['stage'] == 'solid_chart_v2_build_native_v1' and
            host['status'] == native['status'] == 'pass' and host['phase'] == native['phase'] == 'complete' and
            host['native'] == native and host['source_binding'] == host['source_binding_after'] ==
            native['source_binding'] == native['source_binding_after'] == actual_binding and
            host['retained_binary'] == native['build']['binary'] == pins['binary'] and
            native['image_id'] == host['image_identity']['child_id'] == pins['image_id'] and
            all(r[k] is False for r in (host, native) for k in ('gpu_used', 'gt_used', 'native_backend_qualified', 'adopted')) and
            all(type(r['mesh_calls']) is int and r['mesh_calls'] == 0 for r in (host, native)) and
            all(r[k] is True for r in (host, native) for k in ('source_rehashed_after', 'qualified_inputs_rehashed_after')) and
            host['owned_container_removed'] is host['owned_scratch_removed'] is True and
            0 < host['elapsed_seconds'] <= 900 and 0 < native['elapsed_seconds'] <= 600, 'Completed original source-bound build required')
    cgal, cgal_paths, cgal_measured = certificate.qualification(code, build)
    require(cgal['child_image_id'] == pins['image_id'] and native['qualified_inputs']['cgal'] == cgal_measured and
            build.identity(code/'infra/certified_solid_query.cpp', readonly=True) == cgal['native_source'], 'Qualified exact query/image differs')
    require(all(rows[n] == build.identity(code/n, readonly=True) for n in (
        'infra/mesh_conditioned_chart_v2_source.py', 'infra/mesh_conditioned_chart_v2.hpp', 'src/world_reward/mesh_conditioning_v2.py')),
        'Current new chart differs from original built source')
    paths_tuple = (out, *cgal_paths.values(), producer, producer.parent/'revision', producer.parent/'source-sha256')
    return pins, paths['binary'], cgal_paths['binary'], paths_tuple, dict(pins_identity=pin_identity, built_artifacts={k: pins[k] for k in paths},
        original_source=bound, cgal=cgal_measured, build=native['build'], original_runtime=native['original_runtime'])


def cohort_manifest(controls):
    return [dict(control=n, source_array_sha256=list(meta['source_array_sha256']), vertices=len(mesh[0]), faces=len(mesh[1]),
        expected_parents=meta['expected_parents'].tolist(), expected_signs=meta['expected_signs'].tolist(),
        specification_sha256=meta['specification_sha256']) for n, mesh, meta in controls]


def validate_controls(report, manifest):
    require(report['status'] == 'pass' and report['phase'] == 'complete' and report['conditioning_version'] == 2 and
            report['stage'] == 'oriented_solid_compiler_controls_v2' and report['maximum_native_calls'] == 4 and
            report['budget_seconds'] == 1800 and report['owned_scratch_removed'] is True and
            report['adoption'] is report['reconstruction_accuracy_verified'] is False and len(report['controls']) == len(manifest) == 4,
            'Complete four-control qualification required')
    for row, source in zip(report['controls'], manifest):
        require(row['control'] == source['control'] and row['status'] == 'pass' and row['phase'] == 'complete' and
                list(row['source_array_sha256']) == list(source['source_array_sha256']) and type(row['native_attempts']) is int and row['native_attempts'] == 1 and
                row['native_returned'] is True and type(row['native_returncode']) is int and row['native_returncode'] == 0 and
                type(row['native_mapping']['serialization']['committed_collapses']) is int and
                row['native_mapping']['serialization']['committed_collapses'] > 0 and row['source_arrays_unchanged'] is True and
                type(row['output_vertices']) is type(row['output_faces']) is int and
                row['output_vertices'] == row['output_faces'] == 4096 and row['metric_scale_baked_once'] == .375 and
                set(row['stages']) == {'physical_source', 'native_candidate', 'float32_glb', 'default8_exact_weld',
                    'unmodified_official_pack', 'original_grounding_metric_bake'}, 'Full unchanged packed pipeline required')


def native(code, revision, work, build, certificate, built):
    start = time.monotonic(); left = lambda: built.remaining(start, NATIVE_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 1000 and work.stat().st_uid == 1000 and
            stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()) and
            os.environ.get('WR_NATIVE_NETWORK') == 'none' and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Fresh isolated CPU scratch required')
    before = binding(code, revision, build)
    proof = None
    report = dict(stage='solid_chart_v2_qualification_native_v1', status='fail', phase='prerequisites', source_binding=before,
        gpu_used=False, gt_used=False, adoption=False, reconstruction_accuracy_verified=False, maximum_qem_calls=4,
        maximum_query_calls=40, native_seconds=NATIVE_SECONDS, qem_seconds=1200, query_seconds=180, competition_eligibility_verified=False)
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Inclusive native qualification deadline')))
    signal.alarm(max(1, int(left())))
    try:
        pins, binary, query, _, proof = built_qualification(code, build, certificate, built)
        require(os.environ.get('WR_CPU_IMAGE_ID') == pins['image_id'], 'Actual qualified CPU image required')
        import mesh_conditioned_chart_v2_source as generator
        import oriented_solid_controls_v2 as controls
        import oriented_solid_compiler as compiler
        import object_budget_conditioned as cache
        import numpy as np
        require(np.__version__.startswith('1.26.'), 'Qualified NumPy1.26 required'); report['numpy_version'] = np.__version__
        generated, derivation = generator.derive_cached_backend(code/'infra/mesh_conditioned_qem.cpp', code/'infra/mesh_conditioned_chart_v2.hpp')
        require(proof['build']['derivation'] == derivation and proof['build']['build_info'] ==
                build.strict_json(build.run([str(binary), '--build-info'], min(10, left()))), 'Built new source/header/policy ABI differs')
        original = cache.cache_qualification(ROOT, code)[1]
        runtime = cache.runtime_identity(ROOT, code, original, left)
        require(runtime == proof['original_runtime'], 'Original cached compiler runtime differs')
        official = ROOT/OFFICIAL; report['official_helper'] = official_identity(build)
        source = controls.fixtures(); manifest = cohort_manifest(source)
        require(tuple(n for n, _, _ in source) == controls.FIXTURE_NAMES and len(source) == 4, 'Only frozen new v2 cohort allowed')
        report.update(qualified_build=proof, image_id=pins['image_id'], control_manifest=manifest,
            control_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), phase='native_controls')
        try:
            report['controls'] = compiler.geometry_controls(binary, query, certificate.qualification(code, build)[0]['native_source']['sha256'],
                work, official, controls=source, conditioning_version=2)
        except compiler.SolidCompilerError as error: report['controls'] = error.report; raise
        validate_controls(report['controls'], manifest)
        require(cohort_manifest(source) == manifest and all(tuple(compiler.array_hashes(mesh)) == meta['source_array_sha256'] for _, mesh, meta in source), 'Whole original sources changed')
        require(cache.runtime_identity(ROOT, code, original, left) == runtime and official_identity(build) == report['official_helper'],
                'Runtime/official artifacts changed')
        report.update(status='pass', phase='complete', qualified_procedural_controls=4,
                      source_arrays_unchanged=True)
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        try:
            report['source_binding_after'] = binding(code, revision, build)
            require(report['source_binding_after'] == before, 'Qualification source changed')
            report['source_rehashed_after'] = True
            if proof is not None:
                require(built_qualification(code, build, certificate, built)[4] == proof, 'Original built artifacts changed')
                report['artifacts_rehashed_after'] = True
            if 'official_helper' in report:
                require(official_identity(build) == report['official_helper'], 'Official helper changed after native')
                report['official_rehashed_after'] = True
        except Exception as error: report.update(status='fail', posthash_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-start
        if report['elapsed_seconds'] > NATIVE_SECONDS: report.update(status='fail', failure_type='NativeInclusiveDeadline')
        build.write_json(work/'native.json', report); signal.signal(signal.SIGALRM, old)
    return 0 if report['status'] == 'pass' else 1


def host(code, revision, build, certificate, built):
    start = time.monotonic(); left = lambda: built.remaining(start, HOST_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 0 and os.environ.get('DOCKER_HOST') == 'unix://'+str(ROOT/'docker.sock'), 'Owned Linux control required')
    before = binding(code, revision, build); pins, _, _, paths, proof = built_qualification(code, build, certificate, built)
    qualified, _, _ = built.qualified(code, build, certificate); image = built.image_identity(qualified, build, left)
    name = 'wr-solid-chart-v2-qualify-'+revision
    require(not build.run(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], min(10, left())).strip(), 'Foreign/preexisting container')
    out = ROOT/'results'/('solid-chart-v2-qualify-'+revision); require(not out.exists() and not out.is_symlink(), 'Fresh qualification output required')
    out.mkdir(mode=0o755); os.chmod(out, 0o755); work = out/'disposable'; work.mkdir(mode=0o700); os.chown(work, 1000, 1000); owner = work.lstat()
    cid = out/'.container.cid'; report = dict(stage='solid_chart_v2_qualification_host_v1', status='fail', phase='native_controls',
        producer_revision=revision, source_binding=before, qualified_build=proof, image_identity=image,
        gpu_used=False, gt_used=False, adoption=False, reconstruction_accuracy_verified=False, competition_eligibility_verified=False)
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Inclusive host qualification deadline'))); signal.alarm(max(1, int(left())))
    try:
        _, inherited_paths, _ = built.qualified(code, build, certificate)
        official = ROOT/OFFICIAL; report['official_helper'] = official_identity(build)
        mounts = []
        for p in dict.fromkeys((code, code.parent/'revision', code.parent/'source-sha256', *paths, *inherited_paths, official)):
            mounts.extend(['--mount', f'type=bind,src={p},dst={p},readonly'])
        argv = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label', 'world_reward.certified_solid.owner='+revision,
            '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '4', '--memory', '16g',
            '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=512m', *mounts, '--mount', f'type=bind,src={work},dst={work}',
            '--entrypoint', '/usr/bin/env', pins['image_id'], '-i', 'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'TMPDIR=/tmp',
            'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision, 'WR_CPU_IMAGE_ID='+pins['image_id'], 'WR_NATIVE_NETWORK=none',
            'CUDA_VISIBLE_DEVICES=-1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
            'python3', '-I', '-B', str(code/'infra/solid_chart_v2_qualify.py'), '--native']
        build.run(argv, min(NATIVE_SECONDS+10, left()), out/'native.log')
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(60)
        try:
            if (work/'native.json').is_file():
                build.identity(work/'native.json', readonly=True, maximum=2 << 20)
                raw = (work/'native.json').read_bytes(); result = build.strict_json(raw); report['native'] = result; build.seal(out/'native.json', raw)
            build.cleanup_container(name, pins['image_id'], revision, cid); report['owned_container_removed'] = True
            require(binding(code, revision, build) == before and built_qualification(code, build, certificate, built)[4] == proof and
                    built.image_identity(qualified, build, left) == image, 'Source/built/image changed after native')
            report.update(source_binding_after=before, source_rehashed_after=True, artifacts_rehashed_after=True)
            require(official_identity(build) == report['official_helper'], 'Official helper changed after container')
            report['official_rehashed_after'] = True
            if 'native' in report:
                require(result['source_binding'] == result['source_binding_after'] == before and result['source_rehashed_after'] is True and
                    result['stage'] == 'solid_chart_v2_qualification_native_v1' and result['qualified_build'] == proof and
                    result['image_id'] == pins['image_id'] and all(result[k] is False for k in ('gpu_used', 'gt_used', 'adoption', 'reconstruction_accuracy_verified')),
                    'Native source/runtime/qualification scope differs')
                require(result['official_helper'] == report['official_helper'] and result['official_rehashed_after'] is True,
                    'Native original official helper evidence differs')
                require('failure_type' not in report and result['status'] == 'pass' and result['phase'] == 'complete' and
                    result['artifacts_rehashed_after'] is result['source_arrays_unchanged'] is True and 0 < result['elapsed_seconds'] <= NATIVE_SECONDS,
                    'Completed in-budget native qualification required')
                require(result['control_manifest_sha256'] == hashlib.sha256(json.dumps(result['control_manifest'], sort_keys=True).encode()).hexdigest(),
                        'Frozen pre-native cohort manifest differs')
                validate_controls(result['controls'], result['control_manifest'])
                report.update(status='pass', phase='complete', qualified_procedural_controls=4)
            else: raise ValueError('Missing native qualification receipt')
        except Exception as error: report.update(status='fail', posthash_failure_type=type(error).__name__)
        try:
            require(work.lstat().st_ino == owner.st_ino and work.lstat().st_dev == owner.st_dev and not work.is_symlink(), 'Owned scratch replaced')
            for p in work.rglob('*'):
                require(not p.is_symlink() and (p.is_dir() or p.is_file() and p.stat().st_nlink == 1 and p.name in
                    {'native.json', 'input.obj', 'candidate.obj', 'mapping.json', 'candidate.glb'}), 'Unknown scratch; refuse deletion')
            shutil.rmtree(work); report['owned_scratch_removed'] = True
        except Exception as error: report.update(status='fail', cleanup_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-start
        if report['elapsed_seconds'] > HOST_SECONDS: report.update(status='fail', failure_type='HostInclusiveDeadline')
        build.write_json(out/'report.json', report); os.chmod(out, 0o555); signal.alarm(0); signal.signal(signal.SIGALRM, old)
    return 0 if report['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native', action='store_true'); args = parser.parse_args()
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    require(os.environ['WR_ROOT'] == str(ROOT) and code == ROOT/'jobs'/revision/ENTRY/'code' and
            Path(__file__).resolve() == code/'infra/solid_chart_v2_qualify.py', 'Exact own entry required')
    build, certificate, built = helpers(code)
    return native(code, revision, ROOT/'results'/('solid-chart-v2-qualify-'+revision)/'disposable', build, certificate, built) if args.native else host(code, revision, build, certificate, built)


if __name__ == '__main__': raise SystemExit(main())
