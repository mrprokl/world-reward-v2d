"""Offline phase1 compiler/predicate controls, not geometry qualification."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_volume_qem_build'
PROTOCOL = 'configs/mesh_serialization_compiler_protocol_v1.json'
CPP = 'infra/mesh_serialization_qem.cpp'
HELPERS = (CPP, 'infra/mesh_serialization_compile.py', 'infra/run_volume_qem_build.sh',
           'infra/mesh_volume_qem.cpp', 'infra/mesh_guarded_qem.cpp',
           'src/world_reward/mesh_serialization.py', PROTOCOL)
VOLUME = Path('/opt/world-reward/volume-qem')
BASE = Path('/opt/world-reward/guarded-qem')


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path, maximum=32 << 20, *, readonly=True, empty=False):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical file required')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and (not readonly or not before.st_mode & 0o222)
            and (0 if empty else 1) <= before.st_size <= maximum, 'Bounded original file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    after = path.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in
                ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')),
            'File changed during hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def strict(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def protocol(code):
    value = strict((code / PROTOCOL).read_bytes())
    require(value['schema'] == 'world_reward.mesh_serialization_compiler_protocol.v1'
            and value['phase'] == 'compile_predicates_parity_only'
            and value['phase1_controls']['scalar_parity'] == dict(cases=128, seed=8401,
                reference='independent_Python_scalar_dyadics_and_cpp_int_exact_core', mismatches_allowed=0)
            and value['phase1_controls']['compile_seconds'] == 600
            and value['phase1_controls']['inclusive_total_seconds'] == 900
            and value['phase1_controls']['cpu_count'] == 4
            and value['phase1_controls']['memory_gib'] == 16
            and value['claims']['adoption'] is False, 'Frozen phase1 controls required')
    return value


def source(code, revision):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT / 'jobs' / revision / ENTRY / 'code'
            and code.resolve() == code and Path(__file__) == code / HELPERS[1], 'Exact dispatched source required')
    rows = {}
    for path in (code, *sorted(code.rglob('*'))):
        require(path.resolve() == path and not path.is_symlink() and not path.lstat().st_mode & 0o222,
                'Complete readonly source required')
        if path.is_dir():
            continue
        rows[str(path.relative_to(code))] = identity(path, empty=True)
    require(set(HELPERS) <= set(rows), 'Complete adapter closure required')
    markers = {n: identity(code.parent / n, 100) for n in ('revision', 'source-sha256')}
    require((code.parent / 'revision').read_bytes() == (revision + '\n').encode()
            and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent / 'source-sha256').read_bytes()), 'Original markers required')
    return dict(producer_revision=revision, source_files=len(rows), markers=markers,
                helpers={n: rows[n] for n in HELPERS},
                source_files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def write(path, raw):
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def original(code, config):
    pins = config['source_authentication']
    def frozen(path):
        # These original image layers are immutable via read-only root; COPY
        # may retain writable mode bits, unlike our readonly mounted source.
        return identity(path, readonly=False)
    require(frozen(VOLUME / 'mesh_volume_qem.cpp')['sha256'] == pins['original_volume_cpp_sha256']
            == identity(code / 'infra/mesh_volume_qem.cpp')['sha256']
            and frozen(BASE / 'mesh_guarded_qem.cpp')['sha256'] == pins['original_base_cpp_sha256']
            == identity(code / 'infra/mesh_guarded_qem.cpp')['sha256']
            and frozen(VOLUME / 'mesh_volume_qem')['sha256'] == pins['original_binary_sha256'],
            'Original native source/binary identity differs')
    inherited = strict((BASE / 'build.json').read_bytes())
    built = strict((VOLUME / 'build.json').read_bytes())
    require(inherited['status'] == built['status'] == 'pass'
            and built['binary_sha256'] == pins['original_binary_sha256']
            and built['source_cpp_sha256'] == pins['original_volume_cpp_sha256']
            and inherited['libigl_revision'] == pins['libigl_revision']
            and inherited['eigen_revision'] == pins['eigen_revision']
            and frozen(BASE / 'mesh_guarded_qem')['sha256'] == inherited['binary_sha256'],
            'Complete original build chain differs')
    inventory = {str(p.relative_to(BASE / 'source')): frozen(p)['sha256']
                 for p in sorted((BASE / 'source').rglob('*')) if not p.is_dir()}
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == inherited['source_inventory_sha256'] == built['source_inventory_sha256']
            and all(inventory.get(n) == d for n, d in inherited['pinned_primary_sha256'].items()),
            'Complete frozen libigl/Eigen header inventory differs')
    boost = Path('/usr/include/boost')
    require(frozen(boost / 'multiprecision/cpp_int.hpp')['sha256'] == pins['boost_cpp_int_header_sha256'],
            'Original cpp_int header differs')
    # Retain a scalar inventory hash, not thousands of source rows in the receipt.
    boost_inventory = {str(p.relative_to(boost)): frozen(p)['sha256']
                       for p in sorted(boost.rglob('*')) if not p.is_dir()}
    return dict(original_build=frozen(VOLUME / 'build.json'), base_build=frozen(BASE / 'build.json'),
                original_binary=frozen(VOLUME / 'mesh_volume_qem'), inherited_inventory_sha256=digest,
                boost_files=len(boost_inventory), boost_inventory_sha256=hashlib.sha256(json.dumps(
                    boost_inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest())


def compile_binary(code, scratch, config, remaining):
    pins = config['source_authentication']
    raw = (VOLUME / 'mesh_volume_qem.cpp').read_bytes()
    marker = b'\nint main(int argc, char** argv) {'
    require(raw.count(marker) == 1, 'Unique original main boundary required')
    prefix = raw[:raw.index(marker)]
    require(dict(bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest()) == pins['derived_core_prefix'],
            'Authenticated exact original volume prefix differs')
    core = scratch / 'wr_volume_core.hpp'
    write(core, prefix)
    compiler = shutil.which('c++')
    require(compiler is not None, 'Existing compiler required, no install fallback')
    version = subprocess.check_output([compiler, '--version'], text=True, timeout=remaining())
    binary = scratch / 'mesh_serialization_qem'
    macros = {'WR_SOURCE_SHA256': pins['original_base_cpp_sha256'],
              'WR_BASE_SOURCE_SHA256': pins['original_base_cpp_sha256'],
              'WR_VOLUME_SOURCE_SHA256': pins['original_volume_cpp_sha256'],
              'WR_SERIALIZATION_SOURCE_SHA256': identity(code / CPP)['sha256'],
              'WR_VOLUME_CORE_PREFIX_SHA256': pins['derived_core_prefix']['sha256']}
    command = [compiler, '-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off',
               '-DEIGEN_DONT_PARALLELIZE', '-DEIGEN_MPL2_ONLY',
               *[f'-D{k}="{v}"' for k, v in macros.items()], '-I' + str(scratch),
               '-I' + str(BASE / 'source/libigl/include'), '-I' + str(BASE / 'source/eigen'),
               str(code / CPP), '-o', str(binary)]
    start = time.monotonic()
    child = subprocess.run(command, capture_output=True, timeout=min(600, remaining()))
    require(child.returncode == 0, 'Compiler rejected new adapter: ' + child.stderr[-1200:].decode(errors='replace'))
    info = strict(subprocess.check_output([str(binary), '--build-info'], timeout=remaining()))
    require(info['source_sha256'] == macros['WR_SERIALIZATION_SOURCE_SHA256']
            and info['volume_core_prefix_sha256'] == macros['WR_VOLUME_CORE_PREFIX_SHA256']
            and info['volume_source_sha256'] == pins['original_volume_cpp_sha256']
            and info['base_source_sha256'] == pins['original_base_cpp_sha256']
            and info['native_cost_and_placement_unchanged'] is True
            and info['volume_relative_limit'] == .05 and info['adopted'] is False, 'Native compiled ABI differs')
    return binary, dict(compiler=version, compile_command=command, binary=identity(binary, readonly=False),
                        derived_core=identity(core), build_info=info, elapsed_seconds=time.monotonic()-start)


def scalar_reference(vertices, faces):
    """Independent Python arbitrary integers after exact F32 roundtrip."""
    import struct
    from fractions import Fraction
    points = [[struct.unpack('f', struct.pack('f', float(x)))[0] for x in p] for p in vertices]
    used = sorted({int(v) for face in faces for v in face})
    keys = {}
    for v in used:
        require(all(math.isfinite(x) for x in points[v]), 'Nonfinite float32')
        key = tuple(round(float(x) * 1e8) for x in points[v])
        require(all(-(2 ** 63) <= x < 2 ** 63 for x in key), 'Unsupported int64 key')
        keys[v] = key
    def normal(face):
        p = [[Fraction.from_float(x) for x in points[v]] for v in face]
        a, b = ([p[1][j]-p[0][j] for j in range(3)], [p[2][j]-p[0][j] for j in range(3)])
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    active = all(any(normal(face)) for face in faces)
    pairs = [(a, b) for i, a in enumerate(used) for b in used[i+1:]
             if keys[a] == keys[b] and tuple(vertices[a]) != tuple(vertices[b])]
    collapsed = sum(points[a] == points[b] for a, b in pairs)
    return dict(source_float32_exactly_active=active, nonexact_collision_pairs=len(pairs),
                nonexact_key_collision_pairs=len(pairs)-collapsed,
                float32_collapsed_distinct_position_pairs=collapsed, active_vertices=len(used),
                active_faces=len(faces), keys_in_int64_range=True, serialization_safe=active and not pairs)


def obj(path, vertices, faces):
    raw = ''.join('v ' + ' '.join(format(float(x), '.17g') for x in p) + '\n' for p in vertices)
    raw += ''.join('f ' + ' '.join(str(int(x)+1) for x in f) + '\n' for f in faces)
    write(path, raw.encode())


def parity(binary, scratch, remaining):
    import random
    import numpy as np
    from world_reward.mesh_serialization import POLICY_SHA256, serialization_preflight
    rng = random.Random(8401)
    tetra_f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    rows = []
    for index in range(128):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        v *= 2. ** rng.randint(-32, 8)
        v += np.array([rng.randint(-4, 4) / 512. for _ in range(3)])
        f = tetra_f.copy()
        if index % 8 == 0:
            # A distinct near-origin second shell tests global, cross-shell keys.
            second = v.copy(); second[0, 0] += 2. ** -40
            v = np.r_[v, second]; f = np.r_[f, tetra_f+4]
        if index % 8 == 1:
            second = v.copy(); second[second == 0.] = -0.
            v = np.r_[v, second]; f = np.r_[f, tetra_f+4]
        expected = scalar_reference(v, f)
        before = v.tobytes(), f.tobytes()
        python = dict(serialization_preflight(v, f))
        require(python['float32_triangles_exactly_active'] == expected['source_float32_exactly_active'],
                'Python dyadic/filtered noncollinearity disagree')
        path = scratch / ('parity-%03d.obj' % index); obj(path, v, f)
        result = subprocess.run([str(binary), '--preflight', str(path)], capture_output=True, timeout=remaining())
        if expected['source_float32_exactly_active']:
            require(result.returncode == 0, 'Native preflight rejected active manufactured source')
            native = strict(result.stdout)
            require(all(type(native.get(k)) is type(x) and native[k] == x for k, x in expected.items()),
                    'Native/Python scalar parity mismatch')
        else:
            require(result.returncode == 2 and not result.stdout, 'Inactive F32 source not rejected')
        require((v.tobytes(), f.tobytes()) == before, 'Procedural source modified')
        rows.append(dict(case=index, input=identity(path), expected=expected))
        path.unlink()
    return dict(cases=len(rows), seed=8401, mismatches=0, policy_sha256=POLICY_SHA256,
                active_cases=sum(r['expected']['source_float32_exactly_active'] for r in rows),
                controls_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def orientation_parity(binary, remaining):
    import random
    import struct
    from fractions import Fraction
    rng = random.Random(8401)
    rows = []
    def normal(triangle):
        p = [[Fraction.from_float(struct.unpack('f', struct.pack('f', x))[0]) for x in v]
             for v in triangle]
        a, b = ([p[1][j]-p[0][j] for j in range(3)], [p[2][j]-p[0][j] for j in range(3)])
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    for i in range(128):
        before = [[rng.randint(-8, 8)*2.**rng.randint(-149, 5) for _ in range(3)] for _ in range(3)]
        after = [p.copy() for p in before]
        if i % 4 == 0:
            after[1], after[2] = after[2], after[1]
        elif i % 4 == 1:
            after[2] = after[1].copy()
        elif i % 4 == 2:
            after[1][0] += 2.**-30
        a, b = normal(before), normal(after)
        expected = dict(before_float32_exactly_active=any(a), after_float32_exactly_active=any(b),
                        exact_normal_dot_positive=sum(x*y for x, y in zip(a, b)) > 0, adopted=False)
        coords = [format(x, '.17g') for triangle in (before, after) for p in triangle for x in p]
        child = subprocess.run([str(binary), '--triangle-predicates', *coords], capture_output=True, timeout=remaining())
        require(child.returncode == 0 and strict(child.stdout) == expected, 'Exact native orientation scalar mismatch')
        rows.append(dict(coordinates=coords, expected=expected))
    return dict(cases=len(rows), seed=8401, mismatches=0,
                controls_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def controls(binary, scratch, remaining):
    import numpy as np
    import trimesh
    key_cases = [1./512, 3./512, 5./512, -1./512, -3./512, -5./512, -0.]
    for x in key_cases:
        expected = [int(round(float(np.float32(x))*1e8)), 0, 0]
        child = subprocess.run([str(binary), '--position-key', format(x, '.17g'), '0', '0'],
                               capture_output=True, timeout=remaining())
        require(child.returncode == 0 and strict(child.stdout) == dict(key=expected, adopted=False),
                'Native exact default-weld ties-to-even key mismatch')
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rejects = [('nonfinite', np.r_[v, [[np.nan, 0., 0.]]], f),
               ('float32_collinear', v * .01 + 1e6, f),
               ('int64_overflow', v + 2.**40, f), ('malformed', v, np.array([[0, 1, 2]]))]
    for name, positions, faces in rejects:
        path = scratch / (name + '.obj'); obj(path, positions, faces)
        result = subprocess.run([str(binary), '--preflight', str(path)], capture_output=True, timeout=remaining())
        require(result.returncode == 2 and not result.stdout, 'Predeclared invalid control not rejected')
        path.unlink()
    sphere = trimesh.creation.icosphere(subdivisions=1)
    a, b, m = (scratch/n for n in ('identity.obj', 'identity-out.obj', 'identity-mapping.json'))
    obj(a, sphere.vertices, sphere.faces); start = identity(a)
    result = subprocess.run([str(binary), str(a), str(b), str(m)], capture_output=True, timeout=remaining())
    require(result.returncode == 0, 'Underbudget identity control failed')
    mapping = strict(m.read_bytes())
    require(mapping['serialization']['committed_collapses'] == 0
            and mapping['serialization']['serialization_safe'] is True
            and mapping['native_volume']['I'] == list(range(len(sphere.vertices)))
            and mapping['native_volume']['J'] == list(range(len(sphere.faces)))
            and a.read_bytes() == b.read_bytes() and identity(a) == start,
            'Identity control reduced/reordered/changed geometry or mappings')
    for p in (a, b, m):
        p.unlink()
    return dict(expected_rejections=len(rejects), key_rounding_cases=len(key_cases),
                identity_control='icosphere_subdivision_level_1',
                identity_native_calls=1, committed_collapses=0, geometry_quality_validated=False)


def native(code, revision, out):
    require(sys.platform == 'linux' and os.geteuid() == 0
            and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'Offline restricted LinuxCPU required')
    sys.path[:0] = [str(code / 'src')]
    before = source(code, revision); config = protocol(code); started = time.monotonic()
    deadline = float(os.environ['WR_PHASE1_DEADLINE'])
    def remaining():
        value = deadline - time.monotonic()
        require(value > 0, 'Inclusive phase1 deadline exhausted')
        return value
    report = dict(stage='mesh_serialization_compiler_native_v1', status='fail', phase='authentication',
                  source_binding=before, predicate_parity_only=True, simplification_validated=False,
                  geometry_quality_validated=False, production_mesh_used=False, challenge_performance_verified=False,
                  gpu_used=False, adoption=False)
    prior = None
    failure = None
    try:
        remaining(); prior = original(code, config); report['original_runtime'] = prior
        with tempfile.TemporaryDirectory(prefix='serialization-phase1-', dir='/tmp') as tmp:
            scratch = Path(tmp); report['phase'] = 'compile'
            binary, build = compile_binary(code, scratch, config, remaining); report['build'] = build
            report['phase'] = 'scalar_parity'; report['parity'] = parity(binary, scratch, remaining)
            report['orientation_parity'] = orientation_parity(binary, remaining)
            report['phase'] = 'controls'; report['controls'] = controls(binary, scratch, remaining)
            require(identity(binary, readonly=False) == build['binary'], 'New compiler binary changed')
            require(report['parity']['policy_sha256'] == config['source_authentication']['position_weld_policy_sha256'],
                    'Audited serialization policy changed')
        report.update(status='pass', phase='complete', owned_scratch_removed=True)
    except Exception as exc:
        failure = exc; report.update(error_type=type(exc).__name__, error=str(exc)[-1500:])
    finally:
        try:
            require(source(code, revision) == before and (prior is None or original(code, config) == prior),
                    'Original source/header/build/binary changed')
            remaining(); report['originals_rehashed_after'] = True
        except Exception as exc:
            failure = failure or exc; report.update(post_error_type=type(exc).__name__)
        remaining()
        report.update(status='fail' if failure else report['status'], elapsed_seconds=time.monotonic()-started)
        write(out / 'native.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
        remaining()
    if failure:
        raise RuntimeError('Phase1 gate failed; see tiny native receipt')


def host(code, revision):
    started = time.monotonic(); deadline = started + 900
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
            'Actual VM01 CPU build driver required')
    before = source(code, revision); config = protocol(code); pins = config['source_authentication']
    out = ROOT / 'results' / ('mesh-serialization-compiler-' + revision)
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(), 'Fresh result namespace required')
    out.mkdir(mode=0o700); out.chmod(0o700)
    image = pins['original_image_id']; name = 'world-reward-serialization-' + revision[:12]
    def control(args):
        result = subprocess.run(['docker', *args], capture_output=True, timeout=min(15, max(.1, deadline-time.monotonic())))
        require(result.returncode == 0 and len(result.stdout) <= 32 << 10, 'Bounded container control failed')
        return result.stdout
    report = dict(stage='mesh_serialization_compiler_phase1_v1', status='fail', source_binding=before,
                  protocol_identity=identity(code / PROTOCOL), original_image_id=image,
                  predicate_parity_only=True, simplification_validated=False, geometry_quality_validated=False,
                  production_mesh_used=False, challenge_performance_verified=False, adoption=False,
                  gpu_used=False, total_budget_seconds=900, compile_budget_seconds=600)
    failure = None; launched = False
    host_build = ROOT / 'results/image-volume-qem.json'; prior_build = identity(host_build, readonly=False)
    cidfile = out / '.container.cid'
    def interrupted(*_):
        raise TimeoutError('Compiler phase1 interrupted; owned cleanup only')
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        require(prior_build == pins['original_build_receipt'], 'Original host build receipt differs')
        require(control(['image', 'inspect', image, '--format', '{{.Id}}']).decode().strip() == image,
                'Exact existing image required')
        require(not control(['ps', '-aq', '--filter', 'name=^/'+name+'$']).strip(), 'Compiler namespace occupied')
        command = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(cidfile),
                   '--label', 'world_reward.serialization.owner='+revision, '--network', 'none', '--read-only',
                   '--user', '0:0', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   '--cpus', '4', '--memory', '16g', '--tmpfs', '/tmp:rw,nosuid,size=1g',
                   '--mount', f'type=bind,src={code},dst={code},readonly',
                   '--mount', f'type=bind,src={code.parent}/revision,dst={code.parent}/revision,readonly',
                   '--mount', f'type=bind,src={code.parent}/source-sha256,dst={code.parent}/source-sha256,readonly',
                   '--mount', f'type=bind,src={out},dst={out}', '--entrypoint', '/usr/bin/env', image, '-i',
                   'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1',
                   'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=-1',
                   'WR_PHASE1_DEADLINE='+str(deadline), '/opt/conda/bin/python', '-I', '-B', str(code / HELPERS[1]),
                   '--native', str(code), revision, str(out)]
        launched = True
        with (out / '.native.log').open('xb') as stream:
            os.fchmod(stream.fileno(), 0o400)
            result = subprocess.run(command, stdout=stream, stderr=stream, timeout=max(.1, deadline-time.monotonic()-20))
        require(result.returncode == 0, 'Native compiler/predicate gate failed')
        native_report = strict((out / 'native.json').read_bytes())
        require(native_report['status'] == 'pass' and native_report['source_binding'] == before
                and native_report['originals_rehashed_after'] is True
                and native_report['parity']['cases'] == 128 and native_report['parity']['mismatches'] == 0
                and native_report['parity']['policy_sha256'] == pins['position_weld_policy_sha256']
                and native_report['orientation_parity']['cases'] == 128
                and native_report['orientation_parity']['mismatches'] == 0
                and native_report['controls']['identity_native_calls'] == 1
                and native_report['controls']['committed_collapses'] == 0
                and native_report['controls']['expected_rejections'] == 4
                and native_report['controls']['key_rounding_cases'] == 7
                and native_report['phase'] == 'complete'
                and all(native_report[k] is False for k in ('simplification_validated', 'geometry_quality_validated',
                    'production_mesh_used', 'challenge_performance_verified', 'gpu_used', 'adoption')),
                'Complete native phase1 receipt required')
        report['native_identity'] = identity(out / 'native.json'); report['status'] = 'pass'
    except Exception as exc:
        failure = exc; report.update(error_type=type(exc).__name__, error=str(exc)[-500:])
    finally:
        # Protect bounded owned cleanup from a second termination signal.
        for s in handlers:
            signal.signal(s, signal.SIG_IGN)
        try:
            if launched:
                cid = cidfile.read_text().strip(); require(re.fullmatch('[0-9a-f]{64}', cid), 'Owned CID required')
                ids = control(['ps', '-aq', '--no-trunc', '--filter', 'id='+cid]).decode().split()
                require(ids in ([], [cid]), 'Owned container query ambiguous')
                if ids:
                    projection = control(['inspect', cid, '--format', '{{.Image}}|{{.Name}}|{{index .Config.Labels "world_reward.serialization.owner"}}']).decode().strip()
                    require(projection == image+'|/'+name+'|'+revision, 'Cannot clean foreign container')
                    control(['rm', '-f', cid])
                require(not control(['ps', '-aq', '--filter', 'id='+cid]).strip(), 'Owned container survives')
                cidfile.unlink()
            require(source(code, revision) == before and identity(host_build, readonly=False) == prior_build
                    and control(['image', 'inspect', image, '--format', '{{.Id}}']).decode().strip() == image,
                    'Original source/build/image changed')
            report.update(source_rehashed_after=True, original_build_rehashed_after=True, owned_container_removed=True)
            require(time.monotonic() < deadline, 'Inclusive phase1 cleanup budget exhausted')
        except Exception as exc:
            failure = failure or exc; report.update(post_error_type=type(exc).__name__)
        # Detailed logs are disposable; native receipt retains bounded failure reason.
        if (out / '.native.log').is_file():
            (out / '.native.log').unlink()
        report.update(status='fail' if failure else report['status'], elapsed_seconds=time.monotonic()-started)
        write(out / 'report.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
        require(time.monotonic() < deadline, 'Inclusive final receipt publication deadline exceeded')
        for s, handler in handlers.items():
            signal.signal(s, handler)
    require(failure is None, 'Compiler phase1 failed; inspect original tiny receipts')
    print(json.dumps(dict(stage=report['stage'], status=report['status'], predicate_parity_only=True,
                         adoption=False, elapsed_seconds=report['elapsed_seconds'])))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == '--native':
        require(len(args) == 4, 'Exact internal native arguments required')
        return native(Path(args[1]), args[2], Path(args[3]))
    require(not args, 'No arbitrary compiler arguments')
    return host(Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(stage='mesh_serialization_compiler_phase1_v1', status='fail', error_type=type(exc).__name__)))
        sys.exit(1)
