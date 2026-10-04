"""Source-bound CPU build and procedural qualification; not geometry adoption."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_certified_solid_build'
CONFIG = 'configs/certified_solid_protocol_v1.json'
PARENT = 'sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137'
BUDGET = 2400
SHA = re.compile(r'[0-9a-f]{64}')
REV = re.compile(r'[0-9a-f]{40}')
CGAL = dict(url='https://github.com/CGAL/cgal/releases/download/v6.0.1/CGAL-6.0.1-library.tar.xz',
            bytes=5077192, sha256='c752737f91d1af71fa96038f0e37945ce82a5f1fffb6200172cfcdd77755a356')
SUM = dict(url='https://github.com/CGAL/cgal/releases/download/v6.0.1/sha256sum.txt',
           bytes=1003, sha256='c35a4c4c3c779585e194b3b172f145ccac8d9b3125dc5819762bbfa2ec122ec1')
DEBS = (
    dict(package='libgmpxx4ldbl', version='2:6.2.1+dfsg-3ubuntu1', filename='libgmpxx4ldbl_6.2.1+dfsg-3ubuntu1_amd64.deb',
         url='https://archive.ubuntu.com/ubuntu/pool/main/g/gmp/libgmpxx4ldbl_6.2.1+dfsg-3ubuntu1_amd64.deb',
         bytes=9580, sha256='73e8145633a86c8f01466bb42c5b0734665268f7b2656e82365c1659525f7874'),
    dict(package='libgmp-dev', version='2:6.2.1+dfsg-3ubuntu1', filename='libgmp-dev_6.2.1+dfsg-3ubuntu1_amd64.deb',
         url='https://archive.ubuntu.com/ubuntu/pool/main/g/gmp/libgmp-dev_6.2.1+dfsg-3ubuntu1_amd64.deb',
         bytes=336684, sha256='e4ce547c5c5e4efd98854d06559349b3a03272eb343f1bd8e4ccac7b783229a3'),
    dict(package='libmpfr-dev', version='4.1.0-3build3', filename='libmpfr-dev_4.1.0-3build3_amd64.deb',
         url='https://archive.ubuntu.com/ubuntu/pool/main/m/mpfr4/libmpfr-dev_4.1.0-3build3_amd64.deb',
         bytes=271446, sha256='ee3fc55ad08686f89a1839b5673e38c2dd269b0335bc80f8e881e4b641259116'),
)
FLAGS = ['-O2', '-std=c++17', '-ffp-contract=off', '-frounding-math', '-fno-fast-math']
EXPECTED = dict(schema='world_reward.certified_solid_protocol.v1', parent_image_id=PARENT,
                cgal_version='6.0.1', cgal_archive=CGAL, cgal_checksums=SUM,
                dependencies=[{**p, 'architecture': 'amd64'} for p in DEBS],
                installed_libraries={'libgmp10': '2:6.2.1+dfsg-3ubuntu1', 'libmpfr6': '4.1.0-3build3'},
                compiler_flags=FLAGS, linker_flags=['-lgmpxx', '-lgmp', '-lmpfr'],
                budgets_seconds={'inclusive': BUDGET, 'dependency_download': 180, 'cgal_download': 120,
                                 'child_build': 180, 'compile': 600, 'controls': 900},
                capacities={'expanded_bytes': 128 << 20, 'member_bytes': 16 << 20, 'members': 100000},
                controls=15, cpu_count=4, memory_bytes=16 << 30,
                rights={'native_query_spdx': 'Apache-2.0', 'linked_cgal_scope': 'GPL-3.0-or-later OR commercial license',
                        'binary_is_apache_only': False, 'cgal_relicense_performed': False,
                        'competition_eligibility_verified': False},
                scope={'production_mesh_validated': False, 'reconstruction_accuracy_verified': False,
                       'adoption': False, 'gpu_used': False, 'gt_used': False})
CONTROL_NAMES = ('two_cavities_identity', 'two_cavities_similarity', 'disconnected_hollows_identity',
                 'disconnected_hollows_similarity', 'island_in_void_identity', 'island_in_void_similarity',
                 'contact', 'crossing', 'self_crossing', 'open', 'duplicate', 'degenerate', 'trailing',
                 'bounds', 'nested_positive')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def identity(path, *, readonly=False, maximum=64 << 20, empty=False):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and
            not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical regular artifact required')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and
            (0 if empty else 1) <= before.st_size <= maximum and
            (not readonly or not before.st_mode & 0o222), 'Artifact type/mode/size differs')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_uid', 'st_gid', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, f) == getattr(after, f) for f in fields), 'Artifact changed during hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def source_binding(code, revision):
    require(REV.fullmatch(revision) is not None and code == ROOT / 'jobs' / revision / ENTRY / 'code' and
            Path(__file__).resolve() == code / 'infra/certified_solid_build.py', 'Canonical immutable calling source differs')
    files = {}
    for p in sorted(code.rglob('*')):
        if p.is_dir() and not p.is_symlink():
            require(not p.stat().st_mode & 0o222, 'Source directory writable')
        else:
            files[p.relative_to(code).as_posix()] = identity(p, readonly=True, maximum=2 << 20, empty=True)
    require(files and not code.stat().st_mode & 0o222, 'Empty/writable code closure')
    markers = {n: identity(code.parent / n, readonly=True, maximum=128) for n in ('revision', 'source-sha256')}
    require((code.parent / 'revision').read_bytes() == (revision + '\n').encode() and
            SHA.fullmatch((code.parent / 'source-sha256').read_text().strip()), 'Dispatch markers differ')
    protocol = strict_json((code / CONFIG).read_bytes())
    require(json.dumps(protocol, sort_keys=True, separators=(',', ':')) ==
            json.dumps(EXPECTED, sort_keys=True, separators=(',', ':')), 'Frozen acquisition/build protocol differs')
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return dict(producer_revision=revision, files=files, files_sha256=hashlib.sha256(encoded).hexdigest(),
                markers=markers, protocol_identity=files[CONFIG],
                native_source=files['infra/certified_solid_query.cpp'])


def seal(path, data, mode=0o444):
    with Path(path).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.chmod(path, mode)


def write_json(path, value):
    seal(path, (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode())


def safe_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ''
    require(parsed.scheme == 'https' and not parsed.username and not parsed.password and parsed.port in (None, 443) and
            (host in ('github.com', 'archive.ubuntu.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com')
             or host.endswith('.githubusercontent.com')), 'Nonpublic HTTPS download route')


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        safe_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def download(pin, path, deadline, opener=None):
    safe_url(pin['url'])
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), Redirect())
    require(time.monotonic() < deadline, 'Download deadline')
    count, digest = 0, hashlib.sha256()
    request = urllib.request.Request(pin['url'], headers={'Accept-Encoding': 'identity'})
    with opener.open(request, timeout=min(30, deadline - time.monotonic())) as response, Path(path).open('xb') as output:
        require(response.status == 200 and response.headers.get('Content-Encoding', 'identity') == 'identity', 'Download response differs')
        length = response.headers.get('Content-Length')
        require(length is None or length == str(pin['bytes']), 'Download length differs')
        while True:
            require(time.monotonic() < deadline, 'Download deadline')
            block = response.read(min(1 << 20, pin['bytes'] - count + 1))
            if not block:
                break
            count += len(block)
            require(count <= pin['bytes'], 'Download exceeds pinned capacity')
            output.write(block); digest.update(block)
        output.flush(); os.fsync(output.fileno())
    require(count == pin['bytes'] and digest.hexdigest() == pin['sha256'], 'Download byte/SHA differs')
    return dict(bytes=count, sha256=digest.hexdigest())


def extract_headers(archive, destination, capacities):
    rows, paths, expanded = [], {}, 0
    with tarfile.open(archive, 'r:xz') as tar:
        for member in tar:
            name = member.name.rstrip('/')
            p = PurePosixPath(name)
            require(name and not p.is_absolute() and p.as_posix() == name and
                    all(part not in ('', '.', '..') for part in p.parts) and
                    not any(c in name for c in ('\\', '\x00')) and p.parts[0] == 'CGAL-6.0.1', 'Unsafe CGAL path/root')
            require(member.isdir() or member.isfile(), 'CGAL links/special files forbidden')
            require(name not in paths and not any(paths.get(str(parent)) == 'file' for parent in p.parents), 'CGAL duplicate/ancestor collision')
            require(not member.isfile() or not any(n.startswith(name + '/') for n in paths), 'CGAL descendant collision')
            require(0 <= member.size <= capacities['member_bytes'], 'CGAL member capacity')
            expanded += member.size
            require(expanded <= capacities['expanded_bytes'] and len(rows) < capacities['members'], 'CGAL expanded/member capacity')
            paths[name] = 'dir' if member.isdir() else 'file'; rows.append(member)
        # Consume the XZ remainder too, verifying its checksum rather than just TAR EOF.
        while tar.fileobj.read(1 << 20):
            pass
        required = ('CGAL-6.0.1/include/CGAL/version.h', 'CGAL-6.0.1/include/CGAL/Exact_predicates_exact_constructions_kernel.h')
        require(all(paths.get(n) == 'file' for n in required), 'CGAL expected header layout differs')
        destination.mkdir(mode=0o755)
        os.chmod(destination, 0o755)
        retained = {}
        for m in rows:
            if m.isfile() and m.name.startswith('CGAL-6.0.1/include/'):
                relative = PurePosixPath(m.name).relative_to('CGAL-6.0.1/include')
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                # Host umask is 077; isolated UID1000 must traverse the bound
                # public headers without a broad mount or elevated capability.
                for parent in (target.parent, *target.parent.parents):
                    if parent == destination:
                        break
                    require(parent.is_relative_to(destination) and not parent.is_symlink(), 'Header parent differs')
                    os.chmod(parent, 0o755)
                with tar.extractfile(m) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                os.chmod(target, 0o444)
                retained[relative.as_posix()] = identity(target)
    return dict(members=len(rows), expanded_bytes=expanded, header_files=len(retained),
                header_inventory_sha256=hashlib.sha256(json.dumps(retained, sort_keys=True).encode()).hexdigest())


def validate_controls(report, source_sha):
    require(type(report) is dict, 'Control receipt mapping required')
    require(report.get('stage') == 'certified_solid_procedural_controls_v1' and report.get('status') == 'pass' and
            report.get('phase') == 'complete' and report.get('native_source_sha256') == source_sha and
            report.get('source_binary_rehashed_after') is True and report.get('maximum_calls') == 15 and
            report.get('call_seconds') == 60 and report.get('inclusive_seconds') == 900,
            'Native controls incomplete/source differs')
    rows = report.get('records')
    require(type(rows) is list and len(rows) == 15 and all(type(r) is dict for r in rows) and tuple(r.get('name') for r in rows) == CONTROL_NAMES and
            all(r.get('status') == 'pass' for r in rows) and
            type(report.get('elapsed_seconds')) in (int, float) and 0 <= report['elapsed_seconds'] <= 900,
            'All fifteen ordered controls must pass')
    require(all(report.get(k) is False for k in ('challenge_inputs_used', 'gt_used', 'qem_executed', 'adoption',
                                                'production_mesh_validated', 'reconstruction_accuracy_verified')), 'Qualification scope differs')


def remaining(start, cap=BUDGET):
    value = cap - (time.monotonic() - start)
    require(value > 0, 'Inclusive build deadline')
    return value


def run(argv, seconds, log=None):
    if log is None:
        result = subprocess.run(argv, capture_output=True, timeout=seconds, check=False)
        require(len(result.stdout) <= 2 << 20 and len(result.stderr) <= 1 << 20, 'Command receipt output capacity')
    else:
        with Path(log).open('xb') as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds, check=False)
        require(Path(log).stat().st_size <= 8 << 20, 'Build log capacity')
    require(result.returncode == 0, 'Owned command failed')
    return result.stdout if log is None else None


PROBE = """import json,sys,subprocess,numpy; p=subprocess.check_output(['dpkg-query','-W','-f=${Package}\\t${Version}\\n'],text=True); print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'compiler':subprocess.check_output(['c++','-dumpfullversion'],text=True).strip(),'cmake':subprocess.check_output(['cmake','--version'],text=True).splitlines()[0],'packages':dict(row.split('\\t') for row in p.splitlines())},sort_keys=True))"""


def probe(image, seconds):
    return strict_json(run(['docker', 'run', '--rm', '--network', 'none', '--read-only', '--user', '1000:1000',
                           '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--entrypoint', '/usr/bin/env', image,
                           '-i', 'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp',
                           'CUDA_VISIBLE_DEVICES=-1', 'PYTHONDONTWRITEBYTECODE=1', 'python3', '-I', '-B', '-c', PROBE], seconds))


def compare_environment(parent, child):
    require(parent['numpy'] == '1.26.3' and all(child[k] == parent[k] for k in ('python', 'numpy', 'compiler', 'cmake')), 'Parent numerical/compiler environment changed')
    before, after = parent['packages'], child['packages']
    require(all(before.get(k) == v for k, v in EXPECTED['installed_libraries'].items()), 'Existing GMP/MPFR differs')
    allowed = {p['package']: p['version'] for p in DEBS}
    require(all(after.get(k) == v for k, v in before.items()) and
            all(after.get(k) == v for k, v in allowed.items()) and set(after) - set(before) <= set(allowed), 'Unexpected package mutation')
    return dict(python=child['python'], numpy=child['numpy'], compiler=child['compiler'], cmake=child['cmake'],
                parent_packages_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                child_packages_sha256=hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(),
                installed_dependencies=allowed, parent_existing_packages_unchanged=True)


def native(code, revision, work, headers):
    start = time.monotonic()
    require(sys.platform == 'linux' and os.getuid() == 1000 and re.fullmatch(r'sha256:[0-9a-f]{64}', os.environ.get('WR_CPU_IMAGE_ID', '')) and
            os.environ.get('CUDA_VISIBLE_DEVICES') == '-1' and os.environ.get('WR_NATIVE_NETWORK') == 'none', 'Native isolated CPU context differs')
    require(work.is_dir() and not work.is_symlink() and work.stat().st_uid == 1000 and
            stat.S_IMODE(work.stat().st_mode) == 0o700 and not any(work.iterdir()), 'Fresh native work lease differs')
    before = source_binding(code, revision)
    report = dict(stage='certified_solid_build_native_v1', status='fail', phase='compile', source_binding=before,
                  image_id=os.environ['WR_CPU_IMAGE_ID'], device='cpu', gpu_used=False, gt_used=False, adoption=False,
                  production_mesh_validated=False, reconstruction_accuracy_verified=False, competition_eligibility_verified=False)
    try:
        binary = work / 'compiled-query'
        argv = ['c++', *FLAGS, '-DWR_SOLID_QUERY_SOURCE_SHA256="' + before['native_source']['sha256'] + '"',
                '-I' + str(headers), str(code / 'infra/certified_solid_query.cpp'), '-o', str(binary), '-lgmpxx', '-lgmp', '-lmpfr']
        run(argv, min(600, remaining(start)), work / 'compile.log')
        report['compile_flags'] = FLAGS; report['compiled_binary'] = identity(binary)
        report['phase'] = 'procedural_controls'
        sys.path.insert(0, str(code / 'infra'))
        from certified_solid_controls import run_controls, ControlError
        try:
            report['controls'] = run_controls(binary, before['native_source']['sha256'])
        except ControlError as error:
            report['controls'] = error.report
            raise
        validate_controls(report['controls'], before['native_source']['sha256'])
        require(identity(binary) == report['compiled_binary'], 'Compiled binary changed')
        report.update(status='pass', phase='complete', qualified_controls=15)
    except Exception as error:
        report['failure_type'] = type(error).__name__
        log = work / 'compile.log'
        if report['phase'] == 'compile' and log.is_file():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 1000))
                report['compiler_failure_tail'] = stream.read(1000).decode('utf-8', errors='replace')
    finally:
        try:
            report['source_binding_after'] = source_binding(code, revision)
            require(report['source_binding_after'] == before, 'Native source changed')
            report['source_rehashed_after'] = True
        except Exception as error:
            report.update(status='fail', failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic() - start
        if report['elapsed_seconds'] > 1500:
            report.update(status='fail', failure_type='NativeDeadline')
        write_json(work / 'native.json', report)
    return 0 if report['status'] == 'pass' else 1


def cleanup_container(name, image, revision, cid):
    result = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, timeout=20, check=False)
    if result.returncode:
        require(result.returncode == 1 and b'No such' in result.stderr, 'Owned container state unknown')
        return
    rows = strict_json(result.stdout)
    require(len(rows) == 1 and rows[0]['Name'] == '/' + name and rows[0]['Image'] == image and
            rows[0]['Config']['Labels'].get('world_reward.certified_solid.owner') == revision and
            cid.exists() and cid.read_text().strip() == rows[0]['Id'], 'Refuse cleanup of foreign container')
    subprocess.run(['docker', 'kill', name], capture_output=True, timeout=20, check=False)
    require(subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20, check=False).returncode == 0,
            'Owned container cleanup failed')


def host(code, revision, *, reuse_qualified_runtime=False):
    start = time.monotonic()
    require(sys.platform == 'linux' and os.getuid() == 0 and os.environ.get('DOCKER_HOST') == 'unix://' + str(ROOT / 'docker.sock'), 'Linux owned Docker host required')
    before = source_binding(code, revision)
    out = ROOT / 'results' / ('certified-solid-build-' + revision)
    require(not out.exists() and not out.is_symlink(), 'Fresh qualification namespace required')
    out.mkdir(mode=0o700); scratch = out / 'disposable'; scratch.mkdir(mode=0o755)
    scratch_owner = scratch.lstat()
    context, work = scratch / 'context', scratch / 'work'
    context.mkdir(mode=0o755); work.mkdir(mode=0o700); os.chown(work, 1000, 1000)
    name = 'wr-certified-solid-' + revision
    tag = 'world-reward/certified-solid:' + revision
    cid = out / '.container.cid'
    image = None
    report = dict(stage='certified_solid_build_host_v1', status='fail', phase='dependency_acquisition', producer_revision=revision,
                  source_binding=before, parent_image_id=PARENT, inclusive_seconds=BUDGET, device='cpu',
                  gpu_used=False, gt_used=False, adoption=False, production_mesh_validated=False,
                  reconstruction_accuracy_verified=False, competition_eligibility_verified=False,
                  corresponding_sources={'query': {**before['native_source'],
                      'url': 'https://raw.githubusercontent.com/mrprokl/world-reward-v2d/' + revision + '/infra/certified_solid_query.cpp'},
                      'cgal_archive': CGAL, 'cgal_checksums': SUM})
    def expired(_sig, _frame):
        raise TimeoutError('Inclusive build deadline')
    previous_handler = signal.signal(signal.SIGALRM, expired); signal.alarm(BUDGET)
    try:
        parent_id = run(['docker', 'image', 'inspect', PARENT, '--format', '{{.Id}}'], min(30, remaining(start))).decode().strip()
        require(parent_id == PARENT, 'Actual parent image differs')
        parent_before = probe(PARENT, min(60, remaining(start)))
        if reuse_qualified_runtime:
            # Authenticate the original completed numerical runtime, not the
            # current (unqualified) query. Only the freshly compiled query may
            # become qualified by the unchanged fifteen controls below.
            sys.path.insert(0, str(code / 'infra'))
            import certified_solid_source_job as certificate
            prior_pins, _, prior_measured = certificate.qualification(code, sys.modules[__name__])
            image = prior_pins['child_image_id']
            require(run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                        min(30, remaining(start))).decode().strip() == image,
                    'Original qualified runtime image differs')
            report['reused_qualified_runtime'] = dict(pins=prior_pins, artifacts=prior_measured,
                current_query_qualified_by_reuse=False, image_rebuilt=False, dependencies_installed=False)
        else:
            require(subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True, timeout=20, check=False).returncode == 1,
                    'Child image tag already exists')
            depdir = context / 'deps'; depdir.mkdir(mode=0o755)
            deadline = min(start + BUDGET, time.monotonic() + 180)
            report['dependency_archives'] = {p['package']: download(p, depdir / p['filename'], deadline) for p in DEBS}
        deadline = min(start + BUDGET, time.monotonic() + 120)
        archive, sums = context / 'cgal.tar.xz', context / 'sha256sum.txt'
        report['cgal_archive'] = download(CGAL, archive, deadline)
        report['cgal_checksums'] = download(SUM, sums, deadline)
        rows = [row.split() for row in sums.read_text().splitlines()]
        require(sum(row == [CGAL['sha256'], 'CGAL-6.0.1-library.tar.xz'] or
                    row == [CGAL['sha256'], '*CGAL-6.0.1-library.tar.xz'] for row in rows) == 1,
                'Primary checksum concordance differs')
        report['cgal_headers'] = extract_headers(archive, context / 'include', EXPECTED['capacities'])
        context_before = {p.relative_to(context).as_posix(): identity(p) for p in sorted(context.rglob('*')) if p.is_file()}
        report['downloaded_context_sha256'] = hashlib.sha256(json.dumps(context_before, sort_keys=True).encode()).hexdigest()
        if not reuse_qualified_runtime:
            # Only verified DEBs enter the child, never data/model/CGAL layers.
            dockerfile = context / 'Dockerfile'
            dockerfile.write_text('FROM ' + PARENT + '\nCOPY deps /tmp/wr-deps\nRUN dpkg -i /tmp/wr-deps/*.deb && rm -rf /tmp/wr-deps\n')
            report['derived_dockerfile'] = identity(dockerfile)
            context_before['Dockerfile'] = report['derived_dockerfile']
            report['phase'] = 'offline_child_build'
            run(['/usr/bin/env', 'DOCKER_BUILDKIT=0', 'docker', 'build', '--pull=false', '--network', 'none', '--force-rm', '--rm', '--cpu-period', '100000', '--cpu-quota', '400000', '--memory', '16g',
                 '--label', 'world_reward.certified_solid.owner=' + revision, '--tag', tag, '--file', str(dockerfile), str(context)],
                min(180, remaining(start)), scratch / 'docker-build.log')
            image = run(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'], min(30, remaining(start))).decode().strip()
        require(re.fullmatch(r'sha256:[0-9a-f]{64}', image) and image != PARENT, 'Actual child image differs')
        report['child_image_id'] = image
        report['environment'] = compare_environment(parent_before, probe(image, min(60, remaining(start))))
        report['phase'] = 'native_compile_and_controls'
        # Parent directory must be traversable for the exact RW work bind, without broad ROOT mounts.
        os.chmod(out, 0o755)
        argv = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label', 'world_reward.certified_solid.owner=' + revision,
                '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                '--cpus', '4', '--memory', '16g', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=512m',
                '--mount', f'type=bind,src={code},dst={code},readonly',
                '--mount', f'type=bind,src={code.parent / "revision"},dst={code.parent / "revision"},readonly',
                '--mount', f'type=bind,src={code.parent / "source-sha256"},dst={code.parent / "source-sha256"},readonly',
                '--mount', f'type=bind,src={context / "include"},dst=/opt/wr-cgal/include,readonly',
                '--mount', f'type=bind,src={work},dst={work}', '--entrypoint', '/usr/bin/env', image, '-i',
                'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'TMPDIR=/tmp',
                'WR_CODE_REVISION=' + revision, 'WR_CODE=' + str(code), 'WR_CPU_IMAGE_ID=' + image, 'WR_NATIVE_NETWORK=none',
                'CUDA_VISIBLE_DEVICES=-1', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
                'python3', '-I', '-B', str(code / 'infra/certified_solid_build.py'), '--native', '--work', str(work)]
        run(argv, min(1505, remaining(start)), scratch / 'native.log')
        cleanup_container(name, image, revision, cid)
        report['native'] = strict_json((work / 'native.json').read_bytes())
        require(report['native']['status'] == 'pass' and report['native']['source_binding'] == before and
                report['native']['source_rehashed_after'] is True and report['native']['image_id'] == image,
                'Native qualification/source differs')
        validate_controls(report['native']['controls'], before['native_source']['sha256'])
        require(identity(work / 'compiled-query') == report['native']['compiled_binary'], 'Qualified binary differs')
        parent_after = probe(PARENT, min(60, remaining(start)))
        require(parent_after == parent_before, 'Parent image environment changed')
        require(run(['docker', 'image', 'inspect', PARENT, '--format', '{{.Id}}'], min(30, remaining(start))).decode().strip() == PARENT,
                'Parent image identity changed')
        report['parent_rehashed_after'] = True
        report['phase'] = 'publication'
        shutil.copyfile(work / 'compiled-query', out / 'certified_solid_query')
        os.chmod(out / 'certified_solid_query', 0o555)
        report['retained_binary'] = identity(out / 'certified_solid_query', readonly=True)
        require(report['retained_binary'] == report['native']['compiled_binary'], 'Retained binary identity differs')
        shutil.copyfile(work / 'native.json', out / 'native.json'); os.chmod(out / 'native.json', 0o444)
        report.update(status='pass', phase='complete', qualified_controls=15, child_image_parent_verified=True)
    except Exception as error:
        report['failure_type'] = type(error).__name__
        if (work / 'native.json').is_file():
            report['native'] = strict_json((work / 'native.json').read_bytes())
    finally:
        timed_out = report.get('failure_type') == 'TimeoutError'
        if timed_out:
            signal.alarm(60)  # Bounded failure-only cleanup; never qualifies an over-budget run.
        try:
            if image:
                cleanup_container(name, image, revision, cid)
            if 'context_before' in locals():
                context_after = {p.relative_to(context).as_posix(): identity(p) for p in sorted(context.rglob('*')) if p.is_file()}
                require(context_after == context_before, 'Acquired headers/DEBs/build context changed')
                report['build_inputs_rehashed_after'] = True
            if reuse_qualified_runtime and 'prior_measured' in locals():
                require(certificate.qualification(code, sys.modules[__name__])[2] == prior_measured and
                        run(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'],
                            min(30, remaining(start))).decode().strip() == image,
                        'Original qualified runtime changed during fresh compilation')
                report['reused_runtime_rehashed_after'] = True
            if 'parent_before' in locals() and time.monotonic() - start < BUDGET:
                require(probe(PARENT, min(60, remaining(start))) == parent_before, 'Parent changed after run')
                report['parent_rehashed_after'] = True
            report['source_binding_after'] = source_binding(code, revision)
            require(report['source_binding_after'] == before, 'Host source changed')
            report['source_rehashed_after'] = True
            report['elapsed_seconds'] = time.monotonic() - start
            require(report['elapsed_seconds'] <= BUDGET, 'Inclusive build deadline after cleanup')
        except Exception as error:
            report.update(status='fail', failure_type=type(error).__name__)
        # An integrity rejection must not bypass removal of this invocation's
        # disposable inputs. Never clean an aliased/replaced/foreign namespace.
        try:
            if scratch.exists():
                current = scratch.lstat()
                require(scratch.resolve() == scratch and not scratch.is_symlink() and
                        (current.st_dev, current.st_ino, current.st_uid) ==
                        (scratch_owner.st_dev, scratch_owner.st_ino, scratch_owner.st_uid),
                        'Refuse cleanup of replaced disposable namespace')
                shutil.rmtree(scratch)
            report['disposable_build_inputs_removed'] = True
        except Exception as error:
            report.update(status='fail', cleanup_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic() - start
        if report['elapsed_seconds'] > BUDGET:
            report.update(status='fail', failure_type='InclusiveDeadline')
        if report['status'] != 'pass':
            signal.alarm(60)  # Failed-run cleanup only; never changes qualification deadline.
            if image and not reuse_qualified_runtime:
                try:
                    records = strict_json(run(['docker', 'image', 'inspect', tag], 20))
                    require(len(records) == 1 and records[0]['Id'] == image and
                            records[0]['Config']['Labels'].get('world_reward.certified_solid.owner') == revision,
                            'Refuse cleanup of foreign child image')
                    run(['docker', 'image', 'rm', tag], 30)
                    report['unqualified_child_tag_removed'] = True
                except Exception as error:
                    report['child_cleanup_failure_type'] = type(error).__name__
            report['docker_build_intermediate_cleanup_verified'] = False
            for filename in ('certified_solid_query', 'native.json'):
                p = out / filename
                if p.exists():
                    p.unlink()
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - start
        write_json(out / 'report.json', report)
        os.chmod(out, 0o555)
        signal.signal(signal.SIGALRM, previous_handler)
    return 0 if report['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--work', type=Path)
    parser.add_argument('--reuse-qualified-runtime', action='store_true',
        help='Authenticate existing qualified CPU image; fresh compile and all15 controls still required')
    args = parser.parse_args()
    code = Path(os.environ.get('WR_CODE', ''))
    revision = os.environ.get('WR_CODE_REVISION', '')
    require(not args.native or args.work == ROOT / 'results' / ('certified-solid-build-' + revision) / 'disposable/work', 'Native work namespace differs')
    require(args.native or args.work is None, 'Host accepts no work override')
    require(not args.native or not args.reuse_qualified_runtime, 'Runtime reuse is a host-only operation')
    return native(code, revision, args.work, Path('/opt/wr-cgal/include')) if args.native else host(
        code, revision, reuse_qualified_runtime=args.reuse_qualified_runtime)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print('certified_solid_build FAIL: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1)
