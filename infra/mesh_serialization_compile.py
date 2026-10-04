"""Offline source-bound compiler; explicit fresh geometry modes, never adoption."""
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
REPLAY = 'configs/mesh_serialization_compiler_replay_v1.json'
PHASE1_PINS = 'configs/mesh_serialization_compiler_phase1_pins.json'
GEOMETRY_PROTOCOL = 'configs/mesh_serialization_geometry_protocol_v1.json'
CPP = 'infra/mesh_serialization_qem.cpp'
CONDITIONED_CPP = 'infra/mesh_conditioned_qem.cpp'
CONDITIONED_PROTOCOL = 'configs/mesh_conditioned_qem_protocol_v1.json'
CACHE_PROTOCOL = 'configs/mesh_conditioned_cache_protocol_v1.json'
CACHE_PINS = 'configs/mesh_conditioned_qem_qualification_pins.json'
CACHE_HELPERS = (CACHE_PROTOCOL, CACHE_PINS, 'infra/mesh_conditioned_cache.py')
CONDITIONED_HELPERS = (CONDITIONED_CPP, CONDITIONED_PROTOCOL,
                       'infra/mesh_conditioned_geometry.py', 'src/world_reward/mesh_conditioning.py', PHASE1_PINS)
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


def source(code, revision, *, historical=False, conditioned=False, conditioned_cache=False):
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT / 'jobs' / revision / ENTRY / 'code'
            and code.resolve() == code and (historical or Path(__file__) == code / HELPERS[1]),
            'Exact dispatched source required')
    rows = {}
    for path in (code, *sorted(code.rglob('*'))):
        require(path.resolve() == path and not path.is_symlink() and not path.lstat().st_mode & 0o222,
                'Complete readonly source required')
        if path.is_dir():
            continue
        rows[str(path.relative_to(code))] = identity(path, empty=True)
    helpers = HELPERS + CONDITIONED_HELPERS if conditioned or conditioned_cache else HELPERS
    if conditioned_cache:
        helpers += CACHE_HELPERS
    require(set(helpers) <= set(rows), 'Complete adapter closure required')
    markers = {n: identity(code.parent / n, 100) for n in ('revision', 'source-sha256')}
    require((code.parent / 'revision').read_bytes() == (revision + '\n').encode()
            and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent / 'source-sha256').read_bytes()), 'Original markers required')
    return dict(producer_revision=revision, source_files=len(rows), markers=markers,
                helpers={n: rows[n] for n in helpers},
                source_files_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())


def write(path, raw):
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def technical_replay(code):
    """Authenticate actual prior execution failure; no scientific reroll."""
    value = strict((code / REPLAY).read_bytes()); rev = value['original_producer_revision']
    require(value['schema'] == 'world_reward.mesh_serialization_compiler_technical_replay.v1'
            and re.fullmatch('[0-9a-f]{40}', rev) and value['replays_maximum'] == 1
            and value['original_failure_remains_fail'] is True
            and identity(code / CPP) == value['unchanged_cpp']
            and identity(code / PROTOCOL) == value['unchanged_protocol'], 'Frozen one technical replay required')
    old = ROOT / 'jobs' / rev / ENTRY / 'code'
    rows = {}
    for path in (old, *sorted(old.rglob('*'))):
        require(path.resolve() == path and not path.is_symlink() and not path.lstat().st_mode & 0o222,
                'Original failed source remains readonly')
        if not path.is_dir():
            rows[str(path.relative_to(old))] = identity(path, empty=True)
    require(len(rows) == value['original_source_files'] and hashlib.sha256(json.dumps(
            rows, sort_keys=True).encode()).hexdigest() == value['original_source_files_sha256']
            and (old.parent / 'revision').read_bytes() == (rev+'\n').encode(), 'Original failed whole source differs')
    out = ROOT / 'results' / ('mesh-serialization-compiler-'+rev)
    records = {}
    for name, pin in [('report.json', 'original_host_report'), ('native.json', 'original_native_report')]:
        require(identity(out/name) == value[pin], 'Actual original failure receipt differs')
        records[name] = strict((out/name).read_bytes())
    native = records['native.json']; host = records['report.json']
    require(native['status'] == host['status'] == 'fail' and native['phase'] == 'compile'
            and native['error_type'] == 'PermissionError' and 'Permission denied' in native['error']
            and 'parity' not in native and 'controls' not in native
            and native['originals_rehashed_after'] is True and host['owned_container_removed'] is True,
            'Original failure must precede all scientific controls')
    return dict(pins_identity=identity(code / REPLAY), original_revision=rev,
                original_host=value['original_host_report'], original_native=value['original_native_report'],
                original_source_files_sha256=value['original_source_files_sha256'], original_failure_preserved=True)


def phase1_qualification(code):
    """Bind the complete actual earlier PASS, not today's consumer source."""
    pins = strict((code / PHASE1_PINS).read_bytes()); revision = pins['producer_revision']
    require(pins['schema'] == 'world_reward.mesh_serialization_compiler_phase1_pins.v1'
            and re.fullmatch('[0-9a-f]{40}', revision)
            and identity(code / CPP) == pins['source_cpp']
            and identity(code / PROTOCOL) == pins['protocol']
            and pins['simplification_validated'] is False and pins['binary_retained'] is False,
            'Actual phase1 pins and unchanged source/policy required')
    old = ROOT / 'jobs' / revision / ENTRY / 'code'
    require((old.parent/'source-sha256').read_bytes() == (pins['source_archive_sha256']+'\n').encode(),
            'Exact earlier source archive required')
    # Hash old source as inert bytes; never execute an unauthenticated old module.
    bound = source(old, revision, historical=True)
    require(bound['source_files'] == pins['source_files']
            and bound['source_files_sha256'] == pins['source_files_sha256'], 'Complete earlier source closure differs')
    out = ROOT / 'results' / ('mesh-serialization-compiler-'+revision)
    for name, pin in [('report.json', 'host_report'), ('native.json', 'native_report')]:
        require(identity(out/name) == pins[pin], 'Independent earlier actual PASS receipt differs')
    host = strict((out/'report.json').read_bytes()); native = strict((out/'native.json').read_bytes())
    require(host['status'] == native['status'] == 'pass' and host['source_binding'] == native['source_binding'] == bound
            and host['native_identity'] == pins['native_report']
            and host['owned_container_removed'] is True and host['source_rehashed_after'] is True
            and native['originals_rehashed_after'] is True and native['owned_scratch_removed'] is True
            and native['original_runtime'] == pins['original_runtime']
            and native['build']['binary'] == pins['measured_temporary_binary']
            and native['parity']['cases'] == native['orientation_parity']['cases'] == 128
            and native['parity']['mismatches'] == native['orientation_parity']['mismatches'] == 0
            and native['controls']['committed_collapses'] == 0
            and native['controls']['key_rounding_cases'] == 7
            and host['authorized_technical_replay'] == technical_replay(old), 'Complete earlier scope required')
    return dict(pins_identity=identity(code / PHASE1_PINS), source_binding=bound,
                host_report=pins['host_report'], native_report=pins['native_report'],
                binary_continuity_claim=False, simplification_previously_validated=False)


def geometry_protocol(code):
    config = strict((code / GEOMETRY_PROTOCOL).read_bytes())
    require(config['schema'] == 'world_reward.mesh_serialization_geometry_protocol.v1'
            and config['native_calls_maximum'] == 4 and config['native_seconds_per_call'] == 900
            and config['compile_seconds'] == 600 and config['inclusive_total_seconds'] == 5400
            and config['cpu_count'] == 4 and config['memory_gib'] == 16
            and config['receipt_publication_grace_seconds'] == 10
            and config['fixed_source_scale'] == '2**-16' and config['metric_bake_scale'] == .375
            and config['claims']['adoption'] is False, 'Exact separately frozen geometry controls required')
    return config


def conditioned_protocol(code):
    value = strict((code / CONDITIONED_PROTOCOL).read_bytes())
    require(value['schema'] == 'world_reward.mesh_conditioned_qem_protocol.v1'
            and value['native_calls_maximum'] == 8 and value['native_seconds_per_call'] == 600
            and value['compile_seconds'] == 600 and value['inclusive_total_seconds'] == 5400
            and value['cpu_count'] == 4 and value['memory_gib'] == 16
            and value['receipt_publication_grace_seconds'] == 10 and value['claims']['adoption'] is False,
            'Exact fresh conditioned-algorithm controls required')
    original_pins = protocol(code)['source_authentication']
    pins = value['source_authentication']
    require(all(pins.get(k) == v for k, v in original_pins.items())
            and identity(code / CPP)['sha256'] == pins['original_serialization_cpp_sha256']
            and re.fullmatch('[0-9a-f]{64}', pins['serialization_core_prefix_sha256']),
            'Conditioning must retain authenticated original sources and policy')
    return value


def cache_protocol(code):
    value = strict((code / CACHE_PROTOCOL).read_bytes())
    require(value['schema'] == 'world_reward.mesh_conditioned_cache_protocol.v1'
            and value['phase'] == 'exact_cached_implementation_regression'
            and value['qualification_pins'] == CACHE_PINS
            and value['compile_seconds_per_binary'] == 600 and value['inclusive_total_seconds'] == 5400
            and value['cpu_count'] == 4 and value['memory_gib'] == 16
            and value['receipt_publication_grace_seconds'] == 10
            and value['native_calls_maximum'] == 8 and value['native_seconds_per_call'] == 450
            and value['slow_binary_identity'] == dict(bytes=694736,
                sha256='b754e56111b38b079c6ba9bec185c3858bf81e847bd5d193a2483bf6dd0b118a')
            and value['implementation']['fast_compile_macro'] == 'WR_CONDITIONED_CACHE=1'
            and value['implementation']['default_macro_value'] == 0
            and all(value['controls'][k] is True for k in ('all_source_gates_before_any_native',
                'candidate_obj_byte_exact_required', 'mapping_json_byte_exact_required',
                'full_births_and_volumes_and_counters_equal', 'physical_stage_metrics_equal',
                'all_native_failures_stop', 'timeout_or_technical_unavailable_stops'))
            and value['controls']['new_or_replayed_failed_fixture'] is False
            and value['publication'] == dict(retain_qualified_fast_binary_on_azure=True,
                filename='mesh_conditioned_qem', mode='0555',
                qualified_only_after_complete_native_and_host_receipts=True, remove_on_failure=True,
                receipt_mode='0444', directory_mode='0555')
            and all(value['claims'][k] is False for k in ('adoption', 'production_mesh_validated',
                'challenge_performance_verified', 'universal_equivalence')),
            'Exact cached implementation regression protocol required')
    return value


def conditioned_qualification(code):
    """Authenticate old PASS as inert bytes, including its actual source cohort."""
    pins = strict((code / CACHE_PINS).read_bytes()); revision = pins['producer_revision']
    require(pins['schema'] == 'world_reward.mesh_conditioned_qem_qualification.v1'
            and re.fullmatch('[0-9a-f]{40}', revision) and pins['source_files'] == 173
            and pins['qualified_procedural_controls'] == 4 and pins['binary_retained'] is False
            and all(pins[k] is False for k in ('physical_geometry_rescaled', 'production_mesh_validated',
                'challenge_performance_verified', 'adoption')) and pins['new_numeric_algorithm'] is True
            and pins['measured_temporary_binary'] == cache_protocol(code)['slow_binary_identity'],
            'Actual complete conditioned qualification required')
    old = ROOT / 'jobs' / revision / ENTRY / 'code'
    bound = source(old, revision, historical=True, conditioned=True)
    require(bound['source_files'] == pins['source_files'] and bound['source_files_sha256'] == pins['source_files_sha256']
            and (old.parent/'source-sha256').read_bytes() == (pins['source_archive_sha256']+'\n').encode()
            and identity(old / CONDITIONED_CPP) == pins['source_cpp']
            and identity(old / CONDITIONED_PROTOCOL) == pins['protocol'] == identity(code / CONDITIONED_PROTOCOL),
            'Complete qualified old source/archive differs')
    # Reused math and fixture production must be exactly the qualified bytes.
    reused = ('infra/mesh_conditioned_geometry.py', 'infra/mesh_serialization_geometry.py',
              'infra/exact_mesh_geometry.py', 'src/world_reward/mesh_conditioning.py',
              'src/world_reward/mesh_serialization.py', 'src/world_reward/exact_triangle_predicates.py', CPP, PROTOCOL)
    require(all(identity(code/n) == identity(old/n) for n in reused), 'Qualified fixture or geometric math changed')
    out = ROOT / 'results' / ('mesh-conditioned-qem-' + revision)
    for name, pin in (('report.json', 'host_report'), ('native.json', 'native_report')):
        require(identity(out/name) == pins[pin], 'Actual conditioned PASS receipt differs')
    host = strict((out/'report.json').read_bytes()); native = strict((out/'native.json').read_bytes())
    phase1 = strict((old/PHASE1_PINS).read_bytes()); prior = host['phase1_qualification']
    require(host['stage'] == 'mesh_conditioned_qem_host_v1' and native['stage'] == 'mesh_conditioned_qem_native_v1'
            and host['status'] == native['status'] == 'pass' and native['phase'] == 'complete'
            and host['source_binding'] == native['source_binding'] == bound
            and host['native_identity'] == pins['native_report']
            and host['geometry_qualification_status'] == 'pass'
            and host['owned_container_removed'] is True and host['source_rehashed_after'] is True
            and host['original_build_rehashed_after'] is True and native['originals_rehashed_after'] is True
            and native['owned_scratch_removed'] is True and native['original_runtime'] == phase1['original_runtime']
            and native['build']['binary'] == pins['measured_temporary_binary']
            and native['build']['build_info']['source_sha256'] == pins['source_cpp']['sha256']
            and host['conditioned_protocol_identity'] == native['conditioned_protocol_identity'] == pins['protocol']
            and prior['pins_identity'] == identity(old/PHASE1_PINS)
            and prior['source_binding']['source_files_sha256'] == phase1['source_files_sha256']
            and prior['source_binding']['source_files'] == phase1['source_files']
            and prior['source_binding']['producer_revision'] == phase1['producer_revision']
            and prior['host_report'] == phase1['host_report'] and prior['native_report'] == phase1['native_report']
            and prior['binary_continuity_claim'] is False and prior['simplification_previously_validated'] is False
            and native['parity']['cases'] == native['orientation_parity']['cases'] == 128
            and native['parity']['mismatches'] == native['orientation_parity']['mismatches'] == 0
            and native['controls']['expected_rejections'] == 4 and native['controls']['key_rounding_cases'] == 7
            and native['controls']['identity_native_calls'] == 1 and native['controls']['committed_collapses'] == 0
            and all(record[k] is False for record in (host, native) for k in ('simplification_validated',
                'geometry_quality_validated', 'production_mesh_used', 'challenge_performance_verified', 'gpu_used', 'adoption')),
            'Complete old host/native/subqualification scope required')
    geometry = native['geometry']; fixtures = geometry['paired_fixtures']
    require(geometry['stage'] == 'mesh_conditioned_geometry_controls_v1' and geometry['status'] == 'pass'
            and geometry['sources_rehashed_after'] is True and geometry['owned_scratch_removed'] is True
            and geometry['adoption'] is False and geometry['maximum_native_calls'] == 8
            and geometry['native_budget_seconds'] == 600 and len(fixtures) == 4
            and all(len(f['comparisons']) == 2 and [r['method'] for r in f['comparisons']] == ['original', 'conditioned']
                and f['comparisons'][1]['status'] == 'pass' and f['comparisons'][1]['committed_collapses'] > 0 for f in fixtures),
            'Four actual successful conditioned sources required')
    expected = {f['fixture']: f['source_array_sha256'] for f in fixtures}
    require(len(expected) == 4 and all(isinstance(n, str) and isinstance(h, (list, tuple)) and len(h) == 2
            and all(isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) for x in h) for n, h in expected.items()),
            'Actual qualified source array identities required')
    return dict(pins_identity=identity(code/CACHE_PINS), source_binding=bound, source_cpp=pins['source_cpp'],
                host_report=pins['host_report'], native_report=pins['native_report'],
                measured_binary=pins['measured_temporary_binary'], expected_sources=expected,
                original_runtime=native['original_runtime'], qualified_build=native['build'], old_code=str(old),
                reused_sources={n: identity(old/n) for n in reused})


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


def compile_binary(code, scratch, config, remaining, *, conditioned=False, physical_cache=False):
    require(not physical_cache or conditioned, 'Physical cache requires conditioned compilation')
    pins = config['source_authentication']
    raw = (VOLUME / 'mesh_volume_qem.cpp').read_bytes()
    marker = b'\nint main(int argc, char** argv) {'
    require(raw.count(marker) == 1, 'Unique original main boundary required')
    prefix = raw[:raw.index(marker)]
    require(dict(bytes=len(prefix), sha256=hashlib.sha256(prefix).hexdigest()) == pins['derived_core_prefix'],
            'Authenticated exact original volume prefix differs')
    core = scratch / 'wr_volume_core.hpp'
    write(core, prefix)
    serialization_core = None
    if conditioned:
        raw = (code / CPP).read_bytes()
        marker = b'\nint main(int argc,char** argv) {'
        require(raw.count(marker) == 1 and hashlib.sha256(raw).hexdigest()
                == pins['original_serialization_cpp_sha256'], 'Exact original serialization source required')
        prefix = raw[:raw.index(marker)]
        require(hashlib.sha256(prefix).hexdigest() == pins['serialization_core_prefix_sha256'],
                'Authenticated exact serialization prefix differs')
        serialization_core = scratch / 'wr_serialization_core.hpp'
        write(serialization_core, prefix)
    compiler = shutil.which('c++')
    require(compiler is not None, 'Existing compiler required, no install fallback')
    version = subprocess.check_output([compiler, '--version'], text=True, timeout=remaining())
    binary = scratch / ('mesh_conditioned_qem' if conditioned else 'mesh_serialization_qem')
    macros = {'WR_SOURCE_SHA256': pins['original_base_cpp_sha256'],
              'WR_BASE_SOURCE_SHA256': pins['original_base_cpp_sha256'],
              'WR_VOLUME_SOURCE_SHA256': pins['original_volume_cpp_sha256'],
              'WR_SERIALIZATION_SOURCE_SHA256': identity(code / CPP)['sha256'],
              'WR_VOLUME_CORE_PREFIX_SHA256': pins['derived_core_prefix']['sha256']}
    if conditioned:
        macros.update(WR_CONDITIONED_SOURCE_SHA256=identity(code / CONDITIONED_CPP)['sha256'],
                      WR_SERIALIZATION_CORE_PREFIX_SHA256=pins['serialization_core_prefix_sha256'])
    command = [compiler, '-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off',
               '-DEIGEN_DONT_PARALLELIZE', '-DEIGEN_MPL2_ONLY',
               *(['-DWR_CONDITIONED_CACHE=1'] if physical_cache else []),
               *[f'-D{k}="{v}"' for k, v in macros.items()], '-I' + str(scratch),
               '-I' + str(BASE / 'source/libigl/include'), '-I' + str(BASE / 'source/eigen'),
               str(code / (CONDITIONED_CPP if conditioned else CPP)), '-o', str(binary)]
    start = time.monotonic()
    child = subprocess.run(command, capture_output=True, timeout=min(600, remaining()))
    require(child.returncode == 0, 'Compiler rejected new adapter: ' + child.stderr[-1200:].decode(errors='replace'))
    info = strict(subprocess.check_output([str(binary), '--build-info'], timeout=remaining()))
    require(info['source_sha256'] == macros['WR_CONDITIONED_SOURCE_SHA256' if conditioned
                                          else 'WR_SERIALIZATION_SOURCE_SHA256']
            and info['volume_core_prefix_sha256'] == macros['WR_VOLUME_CORE_PREFIX_SHA256']
            and info['volume_source_sha256'] == pins['original_volume_cpp_sha256']
            and info['base_source_sha256'] == pins['original_base_cpp_sha256']
            and info['native_cost_and_placement_unchanged'] is (not conditioned)
            and info['volume_relative_limit'] == .05 and info['adopted'] is False, 'Native compiled ABI differs')
    if conditioned:
        require(info['serialization_source_sha256'] == macros['WR_SERIALIZATION_SOURCE_SHA256']
                and info['serialization_core_prefix_sha256'] == pins['serialization_core_prefix_sha256']
                and info['native_qslim_implementation_reused'] is True and info['new_numeric_algorithm'] is True
                and info['cost_normalization'] is True and info['physical_geometry_rescaled'] is False
                and info['block_intersections'] is True and info['target_faces'] == 4096
                and info['libigl_revision'] == pins['libigl_revision']
                and info['eigen_revision'] == pins['eigen_revision'], 'Conditioned native algorithm ABI differs')
        require((info.get('physical_coordinate_cache') is True if physical_cache
                 else 'physical_coordinate_cache' not in info or info['physical_coordinate_cache'] is False),
                'Compiled physical-cache policy differs')
    report = dict(compiler=version, compile_command=command, binary=identity(binary, readonly=False),
                  derived_core=identity(core), build_info=info, elapsed_seconds=time.monotonic()-start)
    if conditioned:
        report['derived_serialization_core'] = identity(serialization_core)
    return binary, report


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


def native(code, revision, out, *, geometry_phase=False, conditioned=False, conditioned_cache=False):
    require(sum((geometry_phase, conditioned, conditioned_cache)) <= 1, 'Conditioned and historical geometry modes are exclusive')
    require(sys.platform == 'linux' and os.geteuid() == 0
            and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'Offline restricted LinuxCPU required')
    sys.path[:0] = [str(code / 'infra'), str(code / 'src')]
    source_options = {'conditioned_cache': True} if conditioned_cache else ({'conditioned': True} if conditioned else {})
    before = source(code, revision, **source_options)
    config = conditioned_protocol(code) if conditioned or conditioned_cache else protocol(code); started = time.monotonic()
    deadline = float(os.environ['WR_PHASE1_DEADLINE'])
    def remaining():
        value = deadline - time.monotonic()
        require(value > 0, 'Inclusive phase1 deadline exhausted')
        return value
    report = dict(stage='mesh_serialization_compiler_native_v1', status='fail', phase='authentication',
                  source_binding=before, predicate_parity_only=True, simplification_validated=False,
                  geometry_quality_validated=False, production_mesh_used=False, challenge_performance_verified=False,
                  gpu_used=False, adoption=False)
    if conditioned:
        require(out == ROOT / 'results' / ('mesh-conditioned-qem-' + revision), 'Exact conditioned result path required')
        report.update(stage='mesh_conditioned_qem_native_v1', predicate_parity_only=False,
                      new_numeric_algorithm=True, physical_geometry_rescaled=False,
                      conditioned_protocol_identity=identity(code / CONDITIONED_PROTOCOL))
    if conditioned_cache:
        require(out == ROOT / 'results' / ('mesh-conditioned-cache-' + revision), 'Exact cache result path required')
        cache_protocol(code)
        report.update(stage='mesh_conditioned_cache_native_v1', predicate_parity_only=False,
                      physical_coordinate_cache=True, cache_protocol_identity=identity(code/CACHE_PROTOCOL))
    prior = None
    failure = None; retained_owner = None; qualification = None
    try:
        remaining(); prior = original(code, config); report['original_runtime'] = prior
        if conditioned_cache:
            qualification = conditioned_qualification(code); report['conditioned_qualification'] = qualification
            require(prior == qualification['original_runtime'], 'Qualified conditioned runtime differs')
        elif geometry_phase or conditioned:
            if geometry_phase:
                geometry_protocol(code)
            phase1 = strict((code / PHASE1_PINS).read_bytes())
            require(prior == phase1['original_runtime'] and identity(code / CPP) == phase1['source_cpp']
                    and identity(code / PROTOCOL) == phase1['protocol'], 'Qualified source/runtime changed before phase2')
        with tempfile.TemporaryDirectory(prefix='serialization-phase1-', dir='/tmp') as tmp:
            scratch = Path(tmp); report['phase'] = 'compile'
            if conditioned_cache:
                slow_work, fast_work = scratch/'slow', scratch/'fast'; slow_work.mkdir(); fast_work.mkdir()
                old = Path(qualification['old_code'])
                report['phase'] = 'compile_slow'
                slow, slow_build = compile_binary(old, slow_work, conditioned_protocol(old), remaining, conditioned=True)
                report['slow_build'] = slow_build
                require(slow_build['binary'] == qualification['measured_binary']
                        and slow_build['compiler'] == qualification['qualified_build']['compiler'],
                        'Recompiled slow binary/compiler differs from actual qualification')
                report['phase'] = 'compile_cached'
                binary, build = compile_binary(code, fast_work, config, remaining, conditioned=True, physical_cache=True)
                old_info = dict(slow_build['build_info']); fast_info = dict(build['build_info'])
                for info in (old_info, fast_info):
                    info.pop('source_sha256'); info.pop('physical_coordinate_cache', None)
                require(old_info == fast_info and build['compiler'] == slow_build['compiler'],
                        'Cached build changed native algorithm ABI or compiler')
            else:
                binary, build = (compile_binary(code, scratch, config, remaining, conditioned=True) if conditioned
                                 else compile_binary(code, scratch, config, remaining))
            report['build'] = build
            report['phase'] = 'scalar_parity'; report['parity'] = parity(binary, scratch, remaining)
            report['orientation_parity'] = orientation_parity(binary, remaining)
            report['phase'] = 'controls'; report['controls'] = controls(binary, scratch, remaining)
            require(identity(binary, readonly=False) == build['binary'], 'New compiler binary changed')
            require(report['parity']['policy_sha256'] == config['source_authentication']['position_weld_policy_sha256'],
                    'Audited serialization policy changed')
            if conditioned_cache:
                from mesh_conditioned_cache import GeometryControlError, cache_controls
                report['phase'] = 'cache_regression_controls'
                helper = ROOT / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
                try:
                    report['geometry'] = cache_controls(slow, binary, scratch, remaining, official_helper=helper,
                                                        expected_sources=qualification['expected_sources'])
                except GeometryControlError as exc:
                    report['geometry'] = exc.report
                    raise
                require(report['geometry']['status'] == 'pass'
                        and report['geometry']['exact_implementation_regression_verified'] is True,
                        'Cached regression did not qualify publication')
                require(identity(slow, readonly=False) == slow_build['binary']
                        and identity(binary, readonly=False) == build['binary'], 'Regression binary changed')
                report['phase'] = 'publish_qualified_binary'; remaining()
                target = out/'mesh_conditioned_qem'
                with target.open('xb') as stream:
                    os.fchmod(stream.fileno(), 0o555); retained_owner = os.fstat(stream.fileno())
                    with binary.open('rb') as source_stream:
                        shutil.copyfileobj(source_stream, stream, 1 << 20)
                    stream.flush(); os.fsync(stream.fileno())
                require(identity(target) == build['binary'], 'Retained fast binary differs')
                report['retained_binary'] = identity(target); remaining()
            elif conditioned:
                from mesh_conditioned_geometry import GeometryControlError, geometry_controls
                report['phase'] = 'conditioned_geometry_controls'
                helper = ROOT / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
                try:
                    report['geometry'] = geometry_controls(binary, scratch, remaining, official_helper=helper)
                except GeometryControlError as exc:
                    report['geometry'] = exc.report
                    raise
                require(identity(binary, readonly=False) == build['binary'], 'New binary changed during conditioned geometry')
            elif geometry_phase:
                from mesh_serialization_geometry import GeometryControlError, geometry_controls
                report['phase'] = 'geometry_controls'
                helper = ROOT / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
                try:
                    report['geometry'] = geometry_controls(binary, scratch, remaining, official_helper=helper)
                except GeometryControlError as exc:
                    report['geometry'] = exc.report
                    raise
                require(identity(binary, readonly=False) == build['binary'], 'New binary changed during geometry')
        report.update(status='pass', phase='complete', owned_scratch_removed=True)
    except Exception as exc:
        failure = exc; report.update(error_type=type(exc).__name__, error=str(exc)[-1500:])
    finally:
        try:
            after = source(code, revision, **source_options)
            require(after == before and (prior is None or original(code, config) == prior),
                    'Original source/header/build/binary changed')
            if conditioned_cache and qualification is not None:
                require(conditioned_qualification(code) == qualification, 'Original conditioned qualification changed')
                if retained_owner is not None:
                    require(identity(out/'mesh_conditioned_qem') == report['build']['binary'], 'Published binary changed')
            remaining(); report['originals_rehashed_after'] = True
        except Exception as exc:
            failure = failure or exc; report.update(post_error_type=type(exc).__name__)
        report.update(status='fail' if failure else report['status'], elapsed_seconds=time.monotonic()-started)
        if failure and retained_owner is not None:
            target = out/'mesh_conditioned_qem'; now = target.lstat()
            require((now.st_dev, now.st_ino) == (retained_owner.st_dev, retained_owner.st_ino)
                    and stat.S_ISREG(now.st_mode) and now.st_nlink == 1, 'Owned published binary replaced')
            target.unlink(); report.pop('retained_binary', None)
        write(out / 'native.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
        # Publication is an explicit separately bounded evidence grace. A
        # computation timeout must retain FAIL/partial geometry, not drop it.
    if failure:
        raise RuntimeError('Phase1 gate failed; see tiny native receipt')


def host(code, revision, *, geometry_phase=False, conditioned=False, conditioned_cache=False):
    require(sum((geometry_phase, conditioned, conditioned_cache)) <= 1, 'Conditioned and historical geometry modes are exclusive')
    extended = geometry_phase or conditioned or conditioned_cache
    started = time.monotonic(); budget = 5400 if extended else 900; deadline = started + budget
    publication_deadline = deadline + (10 if extended else 0)
    require(sys.platform == 'linux' and os.geteuid() == 0 and os.uname().nodename == 'scenesmith-ncc-h100-01',
            'Actual VM01 CPU build driver required')
    source_options = {'conditioned_cache': True} if conditioned_cache else ({'conditioned': True} if conditioned else {})
    before = source(code, revision, **source_options)
    config = conditioned_protocol(code) if conditioned or conditioned_cache else protocol(code); pins = config['source_authentication']
    prefix = ('mesh-conditioned-cache-' if conditioned_cache else ('mesh-conditioned-qem-' if conditioned else
              ('mesh-serialization-geometry-' if geometry_phase else 'mesh-serialization-compiler-')))
    out = ROOT / 'results' / (prefix + revision)
    require(out.parent.is_dir() and not out.exists() and not out.is_symlink(), 'Fresh result namespace required')
    out.mkdir(mode=0o700); out.chmod(0o700); out_owner = out.stat()
    image = pins['original_image_id']; name = 'world-reward-serialization-' + revision[:12]
    def control(args):
        result = subprocess.run(['docker', *args], capture_output=True, timeout=min(15, max(.1, deadline-time.monotonic())))
        require(result.returncode == 0 and len(result.stdout) <= 32 << 10, 'Bounded container control failed')
        return result.stdout
    report = dict(stage='mesh_serialization_compiler_phase1_v1', status='fail', source_binding=before,
                  protocol_identity=identity(code / PROTOCOL), original_image_id=image,
                  predicate_parity_only=True, simplification_validated=False, geometry_quality_validated=False,
                  production_mesh_used=False, challenge_performance_verified=False, adoption=False,
                  gpu_used=False, total_budget_seconds=budget, compile_budget_seconds=600,
                  receipt_publication_grace_seconds=10 if extended else 0)
    if conditioned:
        report.update(stage='mesh_conditioned_qem_host_v1', predicate_parity_only=False,
                      new_numeric_algorithm=True, physical_geometry_rescaled=False,
                      conditioned_protocol_identity=identity(code / CONDITIONED_PROTOCOL))
    if conditioned_cache:
        cache_protocol(code)
        report.update(stage='mesh_conditioned_cache_host_v1', predicate_parity_only=False,
                      physical_coordinate_cache=True, cache_protocol_identity=identity(code/CACHE_PROTOCOL),
                      compile_budget_seconds_per_binary=600)
    failure = None; launched = False; retained_owner = None
    host_build = ROOT / 'results/image-volume-qem.json'; prior_build = identity(host_build, readonly=False)
    cidfile = out / '.container.cid'
    def interrupted(*_):
        raise TimeoutError('Compiler phase1 interrupted; owned cleanup only')
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        if conditioned_cache:
            report['conditioned_qualification'] = conditioned_qualification(code)
        elif conditioned:
            report['phase1_qualification'] = phase1_qualification(code)
        elif geometry_phase:
            report['phase1_qualification'] = phase1_qualification(code)
            geometry_protocol(code); report['geometry_protocol_identity'] = identity(code / GEOMETRY_PROTOCOL)
        else:
            report['authorized_technical_replay'] = technical_replay(code)
        require(prior_build == pins['original_build_receipt'], 'Original host build receipt differs')
        require(control(['image', 'inspect', image, '--format', '{{.Id}}']).decode().strip() == image,
                'Exact existing image required')
        require(not control(['ps', '-aq', '--filter', 'name=^/'+name+'$']).strip(), 'Compiler namespace occupied')
        helper = ROOT / 'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
        extra_mounts = ['--mount', f'type=bind,src={helper},dst={helper},readonly'] if geometry_phase else []
        if conditioned or conditioned_cache:
            extra_mounts = ['--mount', f'type=bind,src={helper},dst={helper},readonly']
        if conditioned_cache:
            old = Path(report['conditioned_qualification']['old_code'])
            old_out = ROOT / 'results' / ('mesh-conditioned-qem-' + report['conditioned_qualification']['source_binding']['producer_revision'])
            for path in (old, old.parent/'revision', old.parent/'source-sha256', old_out/'report.json', old_out/'native.json'):
                extra_mounts += ['--mount', f'type=bind,src={path},dst={path},readonly']
        command = ['docker', 'run', '--rm', '--name', name, '--cidfile', str(cidfile),
                   '--label', 'world_reward.serialization.owner='+revision, '--network', 'none', '--read-only',
                   '--user', '0:0', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                   # Docker tmpfs defaults noexec: this isolated compiler needs
                   # its own generated binary executable, never model/data mounts.
                   '--cpus', '4', '--memory', '16g', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
                   '--mount', f'type=bind,src={code},dst={code},readonly',
                   '--mount', f'type=bind,src={code.parent}/revision,dst={code.parent}/revision,readonly',
                   '--mount', f'type=bind,src={code.parent}/source-sha256,dst={code.parent}/source-sha256,readonly',
                   '--mount', f'type=bind,src={out},dst={out}', *extra_mounts, '--entrypoint', '/usr/bin/env', image, '-i',
                   'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'PYTHONDONTWRITEBYTECODE=1',
                   'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=-1',
                   'WR_PHASE1_DEADLINE='+str(deadline), '/opt/conda/bin/python', '-I', '-B', str(code / HELPERS[1]),
                   '--native-conditioned-cache' if conditioned_cache else
                   ('--native-conditioned' if conditioned else ('--native-geometry' if geometry_phase else '--native')),
                   str(code), revision, str(out)]
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
        if conditioned_cache:
            result = native_report['geometry']; qualification = report['conditioned_qualification']
            require(native_report['stage'] == 'mesh_conditioned_cache_native_v1'
                    and native_report['predicate_parity_only'] is False
                    and native_report['physical_coordinate_cache'] is True
                    and native_report['cache_protocol_identity'] == report['cache_protocol_identity']
                    and native_report['conditioned_qualification'] == qualification
                    and native_report['owned_scratch_removed'] is True
                    and native_report['original_runtime'] == qualification['original_runtime']
                    and native_report['slow_build']['binary'] == qualification['measured_binary']
                    and native_report['slow_build']['compiler'] == native_report['build']['compiler']
                    == qualification['qualified_build']['compiler']
                    and native_report['build']['build_info']['physical_coordinate_cache'] is True
                    and result['stage'] == 'mesh_conditioned_cache_controls_v1' and result['status'] == 'pass'
                    and result['sources_rehashed_after'] is True and result['owned_scratch_removed'] is True
                    and result['adoption'] is False and result['maximum_native_calls'] == 8
                    and result['native_budget_seconds'] == 450 and result['previous_successful_sources_only'] is True
                    and result['failed_controls_replayed'] is False and result['exact_implementation_regression_verified'] is True
                    and len(result['paired_fixtures']) == 4
                    and {f['fixture']: f['source_array_sha256'] for f in result['paired_fixtures']} == qualification['expected_sources']
                    and all(f['candidate_and_mapping_byte_exact'] is True and f['physical_stage_evidence_equal'] is True
                            and len(f['comparisons']) == 2 and [r['implementation'] for r in f['comparisons']] == ['slow', 'cached']
                            and all(r['method'] == 'conditioned' and r['status'] == 'pass'
                                    and r['committed_collapses'] > 0 for r in f['comparisons']) for f in result['paired_fixtures']),
                    'Complete byte-exact cached implementation regression required')
            target = out/'mesh_conditioned_qem'; retained_owner = target.lstat()
            require(stat.S_IMODE(retained_owner.st_mode) == 0o555
                    and identity(target) == native_report['retained_binary'] == native_report['build']['binary'],
                    'Qualified retained fast binary required')
            report.update(retained_binary=native_report['retained_binary'], geometry_qualification_status='pass')
        if conditioned:
            result = native_report['geometry']
            require(native_report['stage'] == 'mesh_conditioned_qem_native_v1'
                    and native_report['predicate_parity_only'] is False and native_report['new_numeric_algorithm'] is True
                    and native_report['physical_geometry_rescaled'] is False
                    and native_report['conditioned_protocol_identity'] == report['conditioned_protocol_identity']
                    and result['stage'] == 'mesh_conditioned_geometry_controls_v1' and result['status'] == 'pass'
                    and result['sources_rehashed_after'] is True and result['owned_scratch_removed'] is True
                    and result['adoption'] is False and result['maximum_native_calls'] == 8
                    and result['native_budget_seconds'] == 600 and len(result['paired_fixtures']) == 4
                    and all(len(f['comparisons']) == 2 and [r['method'] for r in f['comparisons']]
                            == ['original', 'conditioned'] and f['comparisons'][1]['status'] == 'pass'
                            and f['comparisons'][1]['committed_collapses'] > 0 for f in result['paired_fixtures']),
                    'Complete fresh conditioned geometry evidence required')
            report['geometry_qualification_status'] = result['status']
        if geometry_phase:
            result = native_report['geometry']
            require(result['status'] in ('pass', 'inconclusive') and result['sources_rehashed_after'] is True
                    and result['owned_scratch_removed'] is True and result['adoption'] is False
                    and len(result['paired_fixtures']) == 2
                    and all(len(f['comparisons']) == 2 and f['comparisons'][1]['status'] == 'pass'
                            for f in result['paired_fixtures']), 'Complete paired geometry evidence required')
            report['geometry_qualification_status'] = result['status']
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
            after = source(code, revision, **source_options)
            require(after == before and identity(host_build, readonly=False) == prior_build
                    and control(['image', 'inspect', image, '--format', '{{.Id}}']).decode().strip() == image,
                    'Original source/build/image changed')
            if conditioned_cache:
                require(conditioned_qualification(code) == report['conditioned_qualification'], 'Original conditioned proof changed')
                if retained_owner is not None:
                    require(identity(out/'mesh_conditioned_qem') == report['retained_binary'], 'Published fast binary changed')
            elif conditioned or geometry_phase:
                require(phase1_qualification(code) == report['phase1_qualification'], 'Original phase1 proof changed')
            else:
                require(technical_replay(code) == report['authorized_technical_replay'], 'Original failure replay proof changed')
            report.update(source_rehashed_after=True, original_build_rehashed_after=True, owned_container_removed=True)
            require(time.monotonic() < deadline, 'Inclusive phase1 cleanup budget exhausted')
        except Exception as exc:
            failure = failure or exc; report.update(post_error_type=type(exc).__name__)
        # Detailed logs are disposable; native receipt retains bounded failure reason.
        if (out / '.native.log').is_file():
            (out / '.native.log').unlink()
        if conditioned_cache and failure and (out/'mesh_conditioned_qem').exists():
            target = out/'mesh_conditioned_qem'; now = target.lstat()
            retained_owner = retained_owner or now  # Fresh owned output scope, after owned-container cleanup.
            directory = out.lstat()
            require(stat.S_ISDIR(directory.st_mode) and (directory.st_dev, directory.st_ino)
                    == (out_owner.st_dev, out_owner.st_ino)
                    and stat.S_ISREG(now.st_mode) and now.st_nlink == 1 and now.st_uid == out_owner.st_uid
                    and (now.st_dev, now.st_ino) == (retained_owner.st_dev, retained_owner.st_ino),
                    'Owned cache artifact replaced')
            target.unlink(); report.pop('retained_binary', None)
        report.update(status='fail' if failure else report['status'], elapsed_seconds=time.monotonic()-started)
        write(out / 'report.json', (json.dumps(report, sort_keys=True, allow_nan=False)+'\n').encode())
        try:
            require(time.monotonic() < publication_deadline, 'Bounded final receipt publication deadline exceeded')
            if conditioned_cache and failure is None:
                artifacts = (out/'native.json', out/'report.json', out/'mesh_conditioned_qem')
                identities = [identity(p) for p in artifacts]
                for path in artifacts[:2]:
                    path.chmod(0o444)
                out.chmod(0o555)
                require([identity(p) for p in artifacts] == identities
                        and stat.S_IMODE(out.stat().st_mode) == 0o555
                        and all(stat.S_IMODE(p.stat().st_mode) == m for p, m in zip(artifacts, (0o444, 0o444, 0o555)))
                        and time.monotonic() < publication_deadline, 'Bounded immutable publication failed')
        except Exception:
            if conditioned_cache and retained_owner is not None:
                out.chmod(0o700); target = out/'mesh_conditioned_qem'
                now = target.lstat()
                require((now.st_dev, now.st_ino) == (retained_owner.st_dev, retained_owner.st_ino), 'Owned binary replaced')
                target.unlink()
            raise
        for s, handler in handlers.items():
            signal.signal(s, handler)
    require(failure is None, 'Compiler phase1 failed; inspect original tiny receipts')
    print(json.dumps(dict(stage=report['stage'], status=report['status'], predicate_parity_only=report['predicate_parity_only'],
                         adoption=False, elapsed_seconds=report['elapsed_seconds'])))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ('--native', '--native-geometry', '--native-conditioned', '--native-conditioned-cache'):
        require(len(args) == 4, 'Exact internal native arguments required')
        if args[0] == '--native-conditioned-cache':
            return native(Path(args[1]), args[2], Path(args[3]), conditioned_cache=True)
        if args[0] == '--native-conditioned':
            return native(Path(args[1]), args[2], Path(args[3]), conditioned=True)
        return native(Path(args[1]), args[2], Path(args[3]), geometry_phase=args[0] == '--native-geometry')
    if args == ['--conditioned-cache']:
        return host(Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'], conditioned_cache=True)
    if args == ['--conditioned']:
        return host(Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'], conditioned=True)
    if args == ['--geometry']:
        return host(Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'], geometry_phase=True)
    require(not args, 'No arbitrary compiler arguments')
    return host(Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        stage = ('mesh_conditioned_cache_host_v1' if any(a in ('--conditioned-cache', '--native-conditioned-cache') for a in sys.argv[1:])
                 else ('mesh_conditioned_qem_host_v1' if any(a in ('--conditioned', '--native-conditioned')
                      for a in sys.argv[1:]) else 'mesh_serialization_compiler_phase1_v1'))
        print(json.dumps(dict(stage=stage, status='fail', error_type=type(exc).__name__)))
        sys.exit(1)
