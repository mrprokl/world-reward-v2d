"""One offline CPU build of the new chart backend, NOT native qualification.

Historical qualified binaries/receipts remain unchanged. Only --build-info is
executed: no mesh, QEM, predicate fixture, model, GPU, data or installation.
"""
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
ENTRY = 'run_solid_chart_v2_build'
HOST_SECONDS, NATIVE_SECONDS = 900, 600
CGAL_PINS = 'configs/certified_solid_qualification_pins.json'
CACHE_PINS = 'configs/mesh_conditioned_cache_qualification_pins.json'
CONDITIONED_PROTOCOL = 'configs/mesh_conditioned_qem_protocol_v1.json'
HELPERS = ('infra/solid_chart_v2_build.py', 'infra/run_solid_chart_v2_build.sh',
    'infra/certified_solid_build.py', 'infra/certified_solid_source_job.py',
    'infra/object_budget_conditioned.py', 'infra/mesh_serialization_compile.py',
    'infra/mesh_conditioned_chart_v2_source.py', 'infra/mesh_conditioned_chart_v2.hpp',
    'infra/mesh_conditioned_qem.cpp', 'infra/mesh_serialization_qem.cpp',
    'infra/mesh_volume_qem.cpp', 'infra/mesh_guarded_qem.cpp',
    'src/world_reward/mesh_conditioning_v2.py', 'src/world_reward/mesh_conditioning.py',
    CGAL_PINS, CACHE_PINS, CONDITIONED_PROTOCOL,
    'configs/mesh_conditioned_cache_protocol_v1.json',
    'configs/mesh_conditioned_qem_qualification_pins.json',
    'configs/mesh_serialization_compiler_protocol_v1.json')
WORK_FILES = {'wr_volume_core.hpp', 'wr_serialization_core.hpp', 'mesh_conditioned_chart_v2.hpp',
              'mesh_conditioned_chart_v2.cpp', 'mesh_conditioned_chart_v2', 'compile.log', 'native.json'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def helpers(code):
    require(code.is_absolute() and code.resolve() == code and
            not any(p.is_symlink() for p in (code, *code.parents)), 'Canonical code required')
    sys.path[:0] = [str(code/'infra'), str(code/'src')]
    import certified_solid_build as build
    import certified_solid_source_job as certificate
    return build, certificate


def binding(code, revision, build):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/ENTRY/'code'
            and Path(__file__).resolve() == code/'infra/solid_chart_v2_build.py', 'Actual new entrypoint required')
    rows = {}
    for p in (code, *sorted(code.rglob('*'))):
        require(p.resolve() == p and not p.is_symlink() and not p.lstat().st_mode & 0o222,
                'Complete canonical readonly source required')
        if p.is_dir():
            continue
        rows[p.relative_to(code).as_posix()] = build.identity(p, readonly=True, maximum=2 << 20, empty=True)
    require(set(HELPERS) <= set(rows), 'Complete build source closure required')
    require(set(p.name for p in code.parent.iterdir()) == {'code', 'revision', 'source-sha256'},
            'Exact immutable dispatch snapshot required')
    markers = {n: build.identity(code.parent/n, readonly=True, maximum=128) for n in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
    return dict(producer_revision=revision, markers=markers, helpers={n: rows[n] for n in HELPERS},
                source_files=len(rows), source_files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def cache_leaves(code, build):
    """Host stdlib byte gate; the full existing validator runs natively later."""
    pins = build.strict_json((code/CACHE_PINS).read_bytes())
    require(pins['schema'] == 'world_reward.mesh_conditioned_cache_qualification.v1' and
            re.fullmatch('[0-9a-f]{40}', pins['producer_revision']) and
            pins['exact_implementation_regression_verified'] is True, 'Actual qualified cache required')
    out = ROOT/'results'/('mesh-conditioned-cache-'+pins['producer_revision'])
    require(out.resolve() == out and not out.is_symlink() and stat.S_IMODE(out.lstat().st_mode) == 0o555,
            'Immutable qualified cache directory required')
    paths = {k: out/n for k, n in (('host_report', 'report.json'), ('native_report', 'native.json'),
                                   ('retained_binary', 'mesh_conditioned_qem'))}
    for key, p in paths.items():
        require(build.identity(p, readonly=True, maximum=2 << 20) == pins[key] and
                stat.S_IMODE(p.lstat().st_mode) == (0o555 if key == 'retained_binary' else 0o444),
                'Qualified cache artifact byte/mode mismatch')
    host, native = (build.strict_json(paths[k].read_bytes()) for k in ('host_report', 'native_report'))
    require(host['status'] == native['status'] == 'pass' and native['phase'] == 'complete' and
            host['stage'] == 'mesh_conditioned_cache_host_v1' and native['stage'] == 'mesh_conditioned_cache_native_v1' and
            host['original_image_id'] == build.PARENT and host['native_identity'] == pins['native_report'] and
            host['retained_binary'] == native['build']['binary'] == pins['retained_binary'], 'Cache evidence differs')
    return paths, dict(pins_identity=build.identity(code/CACHE_PINS, readonly=True),
                      artifacts={k: pins[k] for k in paths}, build_info=native['build']['build_info'],
                      compiler=native['build']['compiler'])


def qualified(code, build, certificate):
    pins, paths, measured = certificate.qualification(code, build)
    for p in paths.values():
        require(stat.S_IMODE(p.lstat().st_mode) == (0o555 if p.name == 'certified_solid_query' else 0o444),
                'Qualified CGAL artifact mode differs')
    require(stat.S_IMODE(paths['binary'].parent.lstat().st_mode) == 0o555, 'Immutable CGAL directory required')
    cache_paths, cache = cache_leaves(code, build)
    image_receipt = ROOT/'results/image-volume-qem.json'
    source_pins = build.strict_json((code/'configs/mesh_serialization_compiler_protocol_v1.json').read_bytes())['source_authentication']
    require(build.identity(image_receipt, readonly=True) == source_pins['original_build_receipt'], 'Historical image receipt differs')
    return pins, (*paths.values(), *cache_paths.values(), image_receipt), dict(cgal=measured, cache=cache,
        historical_image_receipt=source_pins['original_build_receipt'])


def image_identity(pins, build, remaining):
    images = {}
    for image in (pins['parent_image_id'], pins['child_image_id']):
        require(build.run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], min(10, remaining())).decode().strip() == image,
                'Actual qualified image differs')
        layers = build.strict_json(build.run(['docker', 'image', 'inspect', image, '--format', '{{json .RootFS.Layers}}'], min(10, remaining())))
        require(type(layers) is list and layers and all(type(x) is str and re.fullmatch('sha256:[0-9a-f]{64}', x) for x in layers), 'Image rootfs differs')
        images[image] = layers
    parent, child = (images[pins[k]] for k in ('parent_image_id', 'child_image_id'))
    require(len(child) > len(parent) and child[:len(parent)] == parent, 'Qualified child must inherit actual parent layers')
    return dict(parent_id=pins['parent_image_id'], child_id=pins['child_image_id'], parent_layers=len(parent),
                child_layers=len(child), rootfs_sha256=hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest())


def remaining(start, cap):
    seconds = cap-(time.monotonic()-start)
    require(seconds > 0, 'Inclusive CPU deadline')
    return seconds


def compile_binary(code, work, cache, build, remaining_seconds):
    import mesh_conditioned_chart_v2_source as generator
    import mesh_serialization_compile as inherited
    config = inherited.conditioned_protocol(code)['source_authentication']
    generated, derivation = generator.derive_cached_backend(code/'infra/mesh_conditioned_qem.cpp', code/'infra/mesh_conditioned_chart_v2.hpp')
    before_sources = {str(p): build.identity(p, readonly=False) for p in (inherited.VOLUME/'mesh_volume_qem.cpp',
        code/'infra/mesh_serialization_qem.cpp', code/'infra/mesh_conditioned_qem.cpp', code/'infra/mesh_conditioned_chart_v2.hpp')}
    prefixes = ((inherited.VOLUME/'mesh_volume_qem.cpp', 'wr_volume_core.hpp', b'\nint main(int argc, char** argv) {', config['derived_core_prefix']),
                (code/'infra/mesh_serialization_qem.cpp', 'wr_serialization_core.hpp', b'\nint main(int argc,char** argv) {',
                 dict(sha256=config['serialization_core_prefix_sha256'])))
    for source, name, marker, pin in prefixes:
        raw = source.read_bytes()
        require(raw.count(marker) == 1, 'Unique inherited source boundary required')
        prefix = raw[:raw.index(marker)]
        require(hashlib.sha256(prefix).hexdigest() == pin['sha256'] and
                ('bytes' not in pin or len(prefix) == pin['bytes']), 'Exact inherited core prefix differs')
        build.seal(work/name, prefix, 0o400)
    build.seal(work/'mesh_conditioned_chart_v2.cpp', generated, 0o400)
    build.seal(work/'mesh_conditioned_chart_v2.hpp', (code/'infra/mesh_conditioned_chart_v2.hpp').read_bytes(), 0o400)
    generated_sources = {n: build.identity(work/n, readonly=True) for n in WORK_FILES if n.endswith(('.hpp', '.cpp'))}
    require(generated_sources['mesh_conditioned_chart_v2.cpp'] == derivation['generated_source'] and
            generated_sources['mesh_conditioned_chart_v2.hpp'] == derivation['header'], 'Actual generated source/header differs')
    compiler = shutil.which('c++')
    require(compiler is not None, 'Existing compiler required; no install fallback')
    version = build.run([compiler, '--version'], min(10, remaining_seconds())).decode()
    require(version == cache['compiler'], 'Inherited qualified compiler version differs')
    macros = dict(WR_SOURCE_SHA256=config['original_base_cpp_sha256'], WR_BASE_SOURCE_SHA256=config['original_base_cpp_sha256'],
        WR_VOLUME_SOURCE_SHA256=config['original_volume_cpp_sha256'], WR_SERIALIZATION_SOURCE_SHA256=config['original_serialization_cpp_sha256'],
        WR_VOLUME_CORE_PREFIX_SHA256=config['derived_core_prefix']['sha256'], WR_SERIALIZATION_CORE_PREFIX_SHA256=config['serialization_core_prefix_sha256'],
        **derivation['compiler_macros'])
    binary = work/'mesh_conditioned_chart_v2'
    argv = [compiler, '-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off', '-DEIGEN_DONT_PARALLELIZE', '-DEIGEN_MPL2_ONLY',
            *[f'-D{k}={v}' if type(v) is int else f'-D{k}="{v}"' for k, v in macros.items()],
            '-I'+str(work), '-I'+str(inherited.BASE/'source/libigl/include'), '-I'+str(inherited.BASE/'source/eigen'),
            '-I/usr/include', str(work/'mesh_conditioned_chart_v2.cpp'), '-o', str(binary)]
    build.run(argv, remaining_seconds(), work/'compile.log')
    os.chmod(binary, 0o555)
    info = build.strict_json(build.run([str(binary), '--build-info'], min(10, remaining_seconds())))
    expected = cache['build_info'] | dict(chart_version=2, origin_search_performed=False,
        backbone_source_sha256=generator.BACKBONE_SHA256, chart_header_sha256=derivation['header']['sha256'],
        chart_policy_sha256=macros['WR_CONDITIONED_CHART_V2_POLICY_SHA256'], source_sha256=derivation['generated_source']['sha256'])
    require(info == expected and all(type(info[k]) is type(v) for k, v in expected.items()), 'Exact new build-info ABI differs')
    require({str(p): build.identity(Path(p), readonly=False) for p in before_sources} == before_sources and
            {n: build.identity(work/n, readonly=True) for n in generated_sources} == generated_sources, 'Build source changed during compilation')
    return binary, dict(derivation=derivation, compiler=version, compile_flags=argv[1:7], build_info=info,
                        generated_sources=generated_sources,
                        binary=build.identity(binary, readonly=True), native_backend_qualified=False, adopted=False)


def native(code, revision, work, build, certificate):
    started = time.monotonic(); left = lambda: remaining(started, NATIVE_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 1000 and work.stat().st_uid == 1000 and
            stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()) and
            os.environ.get('WR_NATIVE_NETWORK') == 'none' and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Fresh isolated CPU work required')
    before = binding(code, revision, build)
    report = dict(stage='solid_chart_v2_build_native_v1', status='fail', phase='prerequisites', source_binding=before,
                  gpu_used=False, gt_used=False, mesh_calls=0, native_backend_qualified=False, adopted=False)
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Native build deadline')))
    signal.alarm(max(1, int(left())))
    try:
        pins, _, proof = qualified(code, build, certificate)
        require(os.environ.get('WR_CPU_IMAGE_ID') == pins['child_image_id'], 'Actual qualified CPU image required')
        import object_budget_conditioned as cache
        _, full = cache.cache_qualification(ROOT, code)
        runtime = cache.runtime_identity(ROOT, code, full, left)
        import mesh_serialization_compile as inherited
        require(inherited.original(code, inherited.conditioned_protocol(code)) == full['original_runtime'],
                'Unchanged original headers/source/binary provenance required')
        report.update(image_id=pins['child_image_id'], qualified_inputs=proof, historical_cache_qualification=full,
                      original_runtime=runtime, phase='compile')
        _, report['build'] = compile_binary(code, work, proof['cache'], build, left)
        require(cache.cache_qualification(ROOT, code)[1] == full and cache.runtime_identity(ROOT, code, full, left) == runtime and
                qualified(code, build, certificate)[2] == proof, 'Qualified sources/runtime changed after compilation')
        report.update(status='pass', phase='complete', qualified_inputs_rehashed_after=True)
    except Exception as error:
        report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        try:
            report['source_binding_after'] = binding(code, revision, build)
            require(report['source_binding_after'] == before, 'Complete new build source changed')
            report['source_rehashed_after'] = True
        except Exception as error:
            report.update(status='fail', posthash_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-started
        if report['elapsed_seconds'] > NATIVE_SECONDS: report.update(status='fail', failure_type='NativeInclusiveDeadline')
        build.write_json(work/'native.json', report)
        signal.signal(signal.SIGALRM, old)
    return 0 if report['status'] == 'pass' else 1


def host(code, revision, build, certificate):
    started = time.monotonic(); left = lambda: remaining(started, HOST_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 0 and os.environ.get('DOCKER_HOST') == 'unix://'+str(ROOT/'docker.sock'), 'Owned Linux Docker control required')
    before = binding(code, revision, build); pins, paths, proof = qualified(code, build, certificate)
    image = image_identity(pins, build, left)
    name = 'wr-solid-chart-v2-build-'+revision
    occupied = build.run(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], min(10, left()))
    require(not occupied.strip(), 'Refuse preexisting/foreign build container')
    out = ROOT/'results'/('solid-chart-v2-build-'+revision)
    require(not out.exists() and not out.is_symlink(), 'Fresh build output required')
    out.mkdir(mode=0o755); os.chmod(out, 0o755)
    work = out/'disposable'; work.mkdir(mode=0o700); os.chown(work, 1000, 1000)
    owner = work.lstat()
    cid = out/'.container.cid'
    report = dict(stage='solid_chart_v2_build_host_v1', status='fail', phase='native_build', producer_revision=revision,
                  source_binding=before, qualified_inputs=proof, image_identity=image, gpu_used=False, gt_used=False,
                  mesh_calls=0, native_backend_qualified=False, adopted=False)
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Host build deadline')))
    signal.alarm(max(1, int(left())))
    try:
        mounts = []
        for p in (code, code.parent/'revision', code.parent/'source-sha256', *paths):
            mounts.extend(['--mount', f'type=bind,src={p},dst={p},readonly'])
        argv = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label', 'world_reward.certified_solid.owner='+revision,
                '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--cpus', '4', '--memory', '16g', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=512m', *mounts,
                '--mount', f'type=bind,src={work},dst={work}', '--entrypoint', '/usr/bin/env', pins['child_image_id'], '-i',
                'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'TMPDIR=/tmp', 'CUDA_VISIBLE_DEVICES=-1',
                'WR_NATIVE_NETWORK=none', 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision, 'WR_CPU_IMAGE_ID='+pins['child_image_id'],
                'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1',
                'python3', '-I', '-B', str(code/'infra/solid_chart_v2_build.py'), '--native']
        build.run(argv, min(NATIVE_SECONDS+10, left()), out/'native.log')
    except Exception as error:
        report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(60)  # Cleanup-only grace; cannot qualify an over-budget build.
        try:
            build.cleanup_container(name, pins['child_image_id'], revision, cid)
            report['owned_container_removed'] = True
            require(qualified(code, build, certificate)[2] == proof and image_identity(pins, build, left) == image and
                    binding(code, revision, build) == before, 'Host source/images/artifacts changed')
            report.update(source_binding_after=before, source_rehashed_after=True, qualified_inputs_rehashed_after=True)
            if (work/'native.json').is_file():
                result = build.strict_json((work/'native.json').read_bytes()); report['native'] = result
                build.seal(out/'native.json', (work/'native.json').read_bytes())
                require(result['stage'] == 'solid_chart_v2_build_native_v1' and result['source_binding'] == result['source_binding_after'] == before and
                        result['source_rehashed_after'] is True and result['qualified_inputs'] == proof and
                        result['native_backend_qualified'] is result['adopted'] is result['gpu_used'] is result['gt_used'] is False and
                        type(result['mesh_calls']) is int and result['mesh_calls'] == 0, 'Native build source/scope differs')
                require('failure_type' not in report and result['status'] == 'pass' and result['phase'] == 'complete' and
                        0 < result['elapsed_seconds'] <= NATIVE_SECONDS and result['qualified_inputs_rehashed_after'] is True,
                        'Completed build-only native receipt required')
                require(result['image_id'] == pins['child_image_id'] and result['build']['native_backend_qualified'] is
                        result['build']['adopted'] is False, 'Build-only runtime scope differs')
                require(build.identity(work/'mesh_conditioned_chart_v2', readonly=True) == result['build']['binary'], 'Built binary changed')
                build.seal(out/'mesh_conditioned_chart_v2', (work/'mesh_conditioned_chart_v2').read_bytes(), 0o555)
                report.update(status='pass', phase='complete', retained_binary=build.identity(out/'mesh_conditioned_chart_v2', readonly=True))
            else: raise ValueError('Missing native build receipt')
        except Exception as error:
            report.update(status='fail', posthash_failure_type=type(error).__name__)
        try:
            require(work.lstat().st_ino == owner.st_ino and work.lstat().st_dev == owner.st_dev and
                    stat.S_IMODE(work.lstat().st_mode) == 0o700 and not work.is_symlink(), 'Owned scratch root changed; refuse deletion')
            for p in work.rglob('*'):
                require(not p.is_symlink() and p.is_file() and p.name in WORK_FILES and p.stat().st_nlink == 1, 'Unknown build scratch; refuse deletion')
            shutil.rmtree(work); report['owned_scratch_removed'] = True
        except Exception as error: report.update(status='fail', cleanup_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-started
        if report['elapsed_seconds'] > HOST_SECONDS: report.update(status='fail', failure_type='HostInclusiveDeadline')
        if report['status'] != 'pass' and (out/'mesh_conditioned_chart_v2').exists(): (out/'mesh_conditioned_chart_v2').unlink()
        build.write_json(out/'report.json', report); os.chmod(out, 0o555)
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
    return 0 if report['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--native', action='store_true'); args = parser.parse_args()
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    require(os.environ['WR_ROOT'] == str(ROOT) and code == ROOT/'jobs'/revision/ENTRY/'code' and
            Path(__file__).resolve() == code/'infra/solid_chart_v2_build.py', 'Actual own source entry required')
    build, certificate = helpers(code)
    return native(code, revision, ROOT/'results'/('solid-chart-v2-build-'+revision)/'disposable', build, certificate) if args.native else host(code, revision, build, certificate)


if __name__ == '__main__':
    raise SystemExit(main())
