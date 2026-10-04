"""Offline standalone surface-QSlim build only; no mesh, QEM or qualification.

Reuse the pinned existing CPU compiler/image/header environment without CGAL,
downloads, dependency installation or historical source/namespace modification.
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
import subprocess
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_surface_qslim_build'
CPP = 'infra/surface_qslim.cpp'
PROTOCOL = 'configs/mesh_serialization_compiler_protocol_v1.json'
HELPERS = ('infra/surface_qslim_build.py', 'infra/run_surface_qslim_build.sh', CPP,
           'infra/certified_solid_build.py', PROTOCOL)
IMAGE = 'sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137'
BASE = Path('/opt/world-reward/guarded-qem')
BUILD_RECEIPT = 'results/image-volume-qem.json'
LIBIGL = '40e7900ccbd767f1f360e0eb10f0f1a6432e0993'
EIGEN = '3147391d946bb4b6c68edd901f2add6ac1f31f8c'
NATIVE_SECONDS, HOST_SECONDS = 600, 900
FLAGS = ('-std=c++17', '-O2', '-frounding-math', '-fno-fast-math', '-ffp-contract=off',
         '-DEIGEN_DONT_PARALLELIZE', '-DEIGEN_MPL2_ONLY')
WORK_FILES = {'surface_qslim', 'compile.log', 'native.json'}


def expected_info(source_sha256):
    return dict(stage='surface_qslim_build_info_v1', source_sha256=source_sha256, libigl_revision=LIBIGL,
        eigen_revision=EIGEN, target_vertices=4096, target_faces=4096, boundary_policy='fixed_original_vertices',
        intersection_blocking='upstream_floating_point', prepared_only=True, adoption=False)


def require(value, reason):
    if not value: raise ValueError(reason)


def modules(code):
    sys.path.insert(0, str(code/'infra'))
    import certified_solid_build as build
    return build


def remaining(start, limit):
    seconds = limit-(time.monotonic()-start)
    require(seconds > 0, 'Inclusive build deadline')
    return seconds


def binding(code, revision, build):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/ENTRY/'code'
            and Path(__file__).resolve() == code/'infra/surface_qslim_build.py', 'Actual immutable own entry required')
    rows, ledger = {}, hashlib.sha256()
    for path in (code, *sorted(code.rglob('*'))):
        s = path.lstat()
        require(path.resolve() == path and not path.is_symlink() and
                stat.S_IMODE(s.st_mode) in ((0o555,) if path.is_dir() else (0o444, 0o555)), 'Readonly canonical closure required')
        if path.is_dir(): continue
        name = path.relative_to(code).as_posix()
        rows[name] = build.identity(path, readonly=True, maximum=2 << 20, empty=True)
        ledger.update(name.encode()+b'\0'+str(stat.S_IMODE(s.st_mode)).encode()+b'\0'+bytes.fromhex(rows[name]['sha256']))
    require(set(HELPERS) <= set(rows) and set(p.name for p in code.parent.iterdir()) == {'code', 'revision', 'source-sha256'},
            'Complete source/dispatch snapshot required')
    markers = {name: build.identity(code.parent/name, readonly=True, maximum=128) for name in ('revision', 'source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode() and
            re.fullmatch(b'[0-9a-f]{64}\n', (code.parent/'source-sha256').read_bytes()), 'Dispatch markers differ')
    return dict(producer_revision=revision, helpers={name: rows[name] for name in HELPERS}, markers=markers,
                source_files=len(rows), source_files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                source_readonly_ledger_sha256=ledger.hexdigest())


def prerequisite(code, build):
    config = build.strict_json((code/PROTOCOL).read_bytes())['source_authentication']
    require(config['original_image_id'] == IMAGE and config['libigl_revision'] == LIBIGL and
            config['eigen_revision'] == EIGEN, 'Pinned original compiler environment required')
    path = ROOT/BUILD_RECEIPT
    require(build.identity(path) == config['original_build_receipt'], 'Independent historical build receipt differs')
    report = build.strict_json(path.read_bytes())
    require(report['stage'] == 'world_reward_volume_qem_build' and report['status'] == 'pass' and
            report['image_id'] == IMAGE and report['libigl_revision'] == LIBIGL and report['eigen_revision'] == EIGEN and
            type(report['compiler']) is str and report['compiler'] and
            re.fullmatch('[0-9a-f]{64}', report['source_inventory_sha256']), 'Historical environment fields differ')
    return config, report, dict(receipt=build.identity(path), image_id=IMAGE,
        compiler=report['compiler'], inherited_inventory_sha256=report['source_inventory_sha256'])


def header_environment(config, original, build, left):
    source = BASE/'source'
    require(build.identity(BASE/'build.json')['sha256'] == original['inherited_build_report_sha256'], 'Inherited header authority differs')
    authority = build.strict_json((BASE/'build.json').read_bytes())
    require(authority['status'] == 'pass' and authority['libigl_revision'] == LIBIGL and authority['eigen_revision'] == EIGEN,
            'Pinned source revisions differ')
    inventory = {}
    for path in (source, *sorted(source.rglob('*'))):
        left(); require(path.resolve() == path and not path.is_symlink(), 'Canonical original headers required')
        if path.is_dir(): continue
        inventory[path.relative_to(source).as_posix()] = build.identity(path, maximum=2 << 20)['sha256']
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == authority['source_inventory_sha256'] == original['source_inventory_sha256'] and
            all(inventory.get(name) == pin for name, pin in authority['pinned_primary_sha256'].items()), 'Full libigl/Eigen source inventory differs')
    boost = Path('/usr/include/boost')
    require(build.identity(boost/'multiprecision/cpp_int.hpp')['sha256'] == config['boost_cpp_int_header_sha256'], 'Pinned cpp_int differs')
    boost_inventory = {}
    for path in (boost, *sorted(boost.rglob('*'))):
        left(); require(path.resolve() == path and not path.is_symlink(), 'Canonical Boost source required')
        if path.is_dir(): continue
        boost_inventory[path.relative_to(boost).as_posix()] = build.identity(path, maximum=2 << 20)['sha256']
    return dict(libigl_revision=LIBIGL, eigen_revision=EIGEN, headers=len(inventory), header_inventory_sha256=digest,
                boost_headers=len(boost_inventory), boost_inventory_sha256=hashlib.sha256(json.dumps(boost_inventory, sort_keys=True).encode()).hexdigest())


def compile_binary(code, work, original, build, left):
    compiler = shutil.which('c++'); require(compiler is not None, 'Existing compiler missing; no install fallback')
    version = build.run([compiler, '--version'], min(10, left())).decode()
    require(version == original['compiler'], 'Qualified compiler version differs')
    pin = build.identity(code/CPP, readonly=True, maximum=2 << 20)
    binary = work/'surface_qslim'
    argv = [compiler, *FLAGS, f'-DWR_SURFACE_QSLIM_SOURCE_SHA256="{pin["sha256"]}"',
            '-I'+str(BASE/'source/libigl/include'), '-I'+str(BASE/'source/eigen'), str(code/CPP), '-o', str(binary)]
    build.run(argv, left(), work/'compile.log'); binary.chmod(0o555)
    info = build.strict_json(build.run([str(binary), '--build-info'], min(10, left())))
    expected = expected_info(pin['sha256'])
    require(info == expected and all(type(info[key]) is type(value) for key, value in expected.items()), 'New build-info ABI differs')
    require(build.identity(code/CPP, readonly=True) == pin, 'Compiler source changed')
    return dict(compiler=version, flags=list(FLAGS), source=pin, binary=build.identity(binary, readonly=True),
                build_info=info, native_backend_qualified=False, qem_calls=0, adopted=False)


def validate_native(report, before):
    expected = dict(stage='surface_qslim_build_native_v1', status='pass', phase='complete', image_id=IMAGE,
                    gpu_used=False, mesh_calls=0, qem_calls=0, native_backend_qualified=False, adopted=False,
                    source_rehashed_after=True, headers_rehashed_after=True)
    require(type(report) is dict and all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items())
            and report.get('source_binding') == before and type(report.get('elapsed_seconds')) in (int, float)
            and 0 <= report['elapsed_seconds'] <= NATIVE_SECONDS, 'Native build incomplete')
    measured = report.get('build'); require(type(measured) is dict and measured.get('flags') == list(FLAGS)
            and measured.get('source') == before['helpers'][CPP] and type(measured.get('qem_calls')) is int and measured['qem_calls'] == 0
            and measured.get('native_backend_qualified') is False and measured.get('adopted') is False,
            'Build-only source/scope differs')
    info, expected = measured.get('build_info'), expected_info(measured['source']['sha256'])
    require(type(info) is dict and info == expected and all(type(info[k]) is type(v) for k, v in expected.items()),
            'Bound native build-info missing/different')


def seal(path, data, mode=0o444):
    with path.open('xb') as output:
        os.fchmod(output.fileno(), mode); output.write(data); output.flush(); os.fsync(output.fileno())


def native(code, revision, work, build):
    started = time.monotonic(); left = lambda: remaining(started, NATIVE_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == work.stat().st_uid == 1000 and
            stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()) and
            os.environ.get('WR_CPU_IMAGE_ID') == IMAGE and os.environ.get('WR_NATIVE_NETWORK') == 'none' and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Isolated fresh CPU build required')
    require(os.listdir('/sys/class/net') == ['lo'], 'Actual native network isolation required')
    before = binding(code, revision, build)
    report = dict(stage='surface_qslim_build_native_v1', status='fail', phase='headers', source_binding=before,
                  image_id=IMAGE, gpu_used=False, mesh_calls=0, qem_calls=0, native_backend_qualified=False, adopted=False)
    old = {sig: signal.signal(sig, lambda *_: (_ for _ in ()).throw(TimeoutError('Build deadline'))) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(NATIVE_SECONDS)
    try:
        config, original, report['original_environment'] = prerequisite(code, build)
        headers = header_environment(config, original, build, left); report['headers'] = headers
        report['phase'] = 'compile'; report['build'] = compile_binary(code, work, original, build, left)
        require(header_environment(config, original, build, left) == headers and prerequisite(code, build)[2] == report['original_environment'], 'Original environment changed')
        report.update(status='pass', phase='complete', headers_rehashed_after=True)
    except Exception as error:
        report['failure_type'] = type(error).__name__
        if (work/'compile.log').exists(): report['compile_failure_tail'] = (work/'compile.log').read_bytes()[-1000:].decode(errors='replace')
    finally:
        try:
            try: require(binding(code, revision, build) == before, 'Own source changed'); report['source_rehashed_after'] = True
            except Exception as error: report.update(status='fail', posthash_failure_type=type(error).__name__)
            report['elapsed_seconds'] = time.monotonic()-started
            if report['elapsed_seconds'] > NATIVE_SECONDS: report.update(status='fail', failure_type='NativeInclusiveDeadline')
            seal(work/'native.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
            left()  # Receipt sealing belongs to the inclusive native deadline.
        finally:
            signal.alarm(0)
            for sig, handler in old.items(): signal.signal(sig, handler)
    return 0 if report['status'] == 'pass' else 1


def image_identity(build, left):
    require(build.run(['docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], min(10, left())).decode().strip() == IMAGE, 'Actual original image missing/different')
    layers = build.strict_json(build.run(['docker', 'image', 'inspect', IMAGE, '--format', '{{json .RootFS.Layers}}'], min(10, left())))
    require(type(layers) is list and layers and all(re.fullmatch('sha256:[0-9a-f]{64}', item) for item in layers), 'Image RootFS differs')
    return dict(image_id=IMAGE, rootfs_layers=len(layers), rootfs_sha256=hashlib.sha256(json.dumps(layers).encode()).hexdigest())


def cid_state(cid):
    require(cid.resolve() == cid and not any(p.is_symlink() for p in (cid, *cid.parents)), 'Canonical CID required')
    s = cid.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.getuid() and s.st_size in (64, 65),
            'Owned regular CID required')
    raw = cid.read_bytes(); require(re.fullmatch(b'[0-9a-f]{64}\n?', raw), 'Exact CID required')
    fields = ('st_dev', 'st_ino', 'st_uid', 'st_gid', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(s, f) == getattr(cid.lstat(), f) for f in fields), 'CID changed while reading')
    return raw.strip().decode(), tuple(getattr(s, f) for f in fields)


def cleanup(name, revision, cid, left):
    identifier, owned = cid_state(cid)
    def inspect():
        result = subprocess.run(['docker', 'container', 'inspect', identifier], capture_output=True, timeout=min(10, left()))
        require(cid_state(cid) == (identifier, owned), 'CID changed during cleanup')
        return result
    def absent(result):
        variants = tuple((prefix+identifier).encode() for prefix in ('Error: No such container: ',
            'Error response from daemon: No such container: ', 'Error: No such object: ', 'error: no such object: '))
        return result.returncode == 1 and result.stdout.strip() in (b'', b'[]') and result.stderr.strip() in variants
    result = inspect()
    if result.returncode:
        require(absent(result), 'Daemon error is not exact CID absence'); return
    rows = json.loads(result.stdout)
    require(len(rows) == 1 and rows[0]['Id'] == identifier and rows[0]['Name'] == '/'+name and rows[0]['Image'] == IMAGE and
            rows[0]['Config']['Labels'].get('world_reward.surface_qslim.owner') == revision, 'Refuse foreign container cleanup')
    result = subprocess.run(['docker', 'rm', '-f', identifier], capture_output=True, timeout=min(15, left()))
    require(result.returncode == 0 and cid_state(cid) == (identifier, owned), 'Owned cleanup failed/CID changed')
    require(absent(inspect()), 'Owned exact CID absence not verified')


def remove_work(work, owner):
    s = work.lstat(); require(work.resolve() == work and not work.is_symlink() and (s.st_dev, s.st_ino) == owner, 'Owned scratch changed')
    entries = list(work.iterdir()); require({p.name for p in entries} <= WORK_FILES, 'Unexpected scratch file')
    require(all(not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode) and p.lstat().st_nlink == 1 for p in entries), 'Scratch contains aliases')
    for path in entries: path.unlink()
    work.rmdir()


def host(code, revision, build):
    started = time.monotonic(); left = lambda: remaining(started, HOST_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 0 and os.environ.get('DOCKER_HOST') == 'unix://'+str(ROOT/'docker.sock'), 'Private owned Docker host required')
    before = binding(code, revision, build); _, _, proof = prerequisite(code, build); image = image_identity(build, left)
    name = 'wr-surface-qslim-build-'+revision
    require(not build.run(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], min(10, left())).strip(), 'Preexisting container refused')
    out = ROOT/'results'/('surface-qslim-build-'+revision)
    require(not out.exists() and not out.is_symlink(), 'Fresh results required')
    out.mkdir(mode=0o755); out.chmod(0o755)
    work = out/'disposable'; work.mkdir(mode=0o700); work.chmod(0o700); os.chown(work, 1000, 1000)
    owner = (work.stat().st_dev, work.stat().st_ino); cid = out/'.container.cid'
    report = dict(stage='surface_qslim_build_host_v1', status='fail', phase='native_build', producer_revision=revision,
        source_binding=before, image_identity=image, original_environment=proof, gpu_used=False, qem_calls=0,
        native_backend_qualified=False, adopted=False)
    old = {sig: signal.signal(sig, lambda *_: (_ for _ in ()).throw(TimeoutError('Host deadline'))) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.alarm(max(1, int(left())))
    native_report = None; published = None
    try:
        command = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label', 'world_reward.surface_qslim.owner='+revision,
            '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--cpus', '4', '--memory', '16g', '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m',
            '--mount', f'type=bind,src={code.parent},dst={code.parent},readonly',
            '--mount', f'type=bind,src={ROOT/BUILD_RECEIPT},dst={ROOT/BUILD_RECEIPT},readonly',
            '--mount', f'type=bind,src={work},dst={work}', '--entrypoint', '/usr/bin/env', IMAGE, '-i',
            'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'CUDA_VISIBLE_DEVICES=-1',
            'OMP_NUM_THREADS=1', 'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code), 'WR_CODE_REVISION='+revision,
            'WR_CPU_IMAGE_ID='+IMAGE, 'WR_NATIVE_NETWORK=none', 'python3', '-I', '-B', str(code/HELPERS[0]), '--native']
        build.run(command, min(NATIVE_SECONDS+30, left()), out/'native.log')
        build.identity(work/'native.json', readonly=True, maximum=2 << 20)
        native_report = build.strict_json((work/'native.json').read_bytes())
        validate_native(native_report, before)
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(0); cleanup_start = time.monotonic(); clean_left = lambda: remaining(cleanup_start, 40)
        try: cleanup(name, revision, cid, clean_left); report['owned_container_removed'] = True
        except Exception as error: report.update(failure_type=type(error).__name__, cleanup_failed=True)
        try: signal.alarm(max(1, int(left())))
        except Exception as error: report.update(failure_type=type(error).__name__)
        try:
            if native_report is None and (work/'native.json').exists():
                build.identity(work/'native.json', readonly=True, maximum=2 << 20)
                native_report = build.strict_json((work/'native.json').read_bytes())
            if native_report is not None: seal(out/'native.json', (work/'native.json').read_bytes()); report['native_identity'] = build.identity(out/'native.json', readonly=True)
            require(binding(code, revision, build) == before and prerequisite(code, build)[2] == proof and image_identity(build, left) == image, 'Source/runtime postchecks differ')
            report['source_rehashed_after'] = report['runtime_rehashed_after'] = True
            require(report.get('owned_container_removed') is True and not report.get('failure_type'), 'Build/cleanup failed')
            pin = native_report['build']['binary']; require(build.identity(work/'surface_qslim', readonly=True) == pin, 'Built binary differs')
            with (out/'surface_qslim').open('xb') as target:
                s = os.fstat(target.fileno()); published = (s.st_dev, s.st_ino)
                os.fchmod(target.fileno(), 0o555); target.write((work/'surface_qslim').read_bytes()); target.flush(); os.fsync(target.fileno())
            require(build.identity(out/'surface_qslim', readonly=True) == pin, 'Retained binary differs')
            report.update(status='pass', phase='complete', retained_binary=pin)
        except Exception as error: report.update(status='fail', failure_type=type(error).__name__)
        try:
            require(report.get('owned_container_removed') is True, 'No scratch cleanup with unknown container')
            remove_work(work, owner); report['owned_scratch_removed'] = True
        except Exception as error: report.update(status='fail', scratch_failure_type=type(error).__name__)
        try:
            for control in (cid, out/'native.log'):
                if control.exists():
                    build.identity(control, empty=True); control.chmod(0o444)
        except Exception as error: report.update(status='fail', control_seal_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-started
        if report['elapsed_seconds'] > HOST_SECONDS: report.update(status='fail', failure_type='HostInclusiveDeadline')
        if report['status'] != 'pass' and published:
            p = out/'surface_qslim'; s = p.lstat()
            if not p.is_symlink() and (s.st_dev, s.st_ino) == published: p.unlink()
            report.pop('retained_binary', None)
        try:
            seal(out/'report.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode()); out.chmod(0o555)
            left()  # Sealing/postchecks, not just compilation, are inside the host deadline.
        finally:
            signal.alarm(0)
            for sig, handler in old.items(): signal.signal(sig, handler)
    return 0 if report['status'] == 'pass' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--native', action='store_true'); args = parser.parse_args(argv)
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    require(os.environ['WR_ROOT'] == str(ROOT) and code == ROOT/'jobs'/revision/ENTRY/'code', 'Exact build namespace required')
    build = modules(code)
    return native(code, revision, ROOT/'results'/('surface-qslim-build-'+revision)/'disposable', build) if args.native else host(code, revision, build)


if __name__ == '__main__': raise SystemExit(main())
