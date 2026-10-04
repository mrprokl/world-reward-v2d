"""Read one frozen CPU-qualified mesh; never run QEM or reconstruct on GPU.

Whole source/embedding/birth/volume proofs belong to the SHA-bound CPU producer.
This consumer checks those receipts, exact stored topology and scale arithmetic;
it does not relabel them as newly measured embedding or reconstruction accuracy.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np
from volume_mesh_pin_inventory import identity as _identity, strict_json

CODE = Path(__file__).resolve().parent.parent
SCHEMA = 'world_reward.conditioned_mesh_pins.v1'
STAGE = 'world_reward_cpu_conditioned_object_mesh'
PROCESSOR = 'infra/object_budget_conditioned.py'
CACHE_PINS = 'configs/mesh_conditioned_cache_qualification_pins.json'
STAGES = ('physical_source', 'native_candidate', 'float32_glb', 'default8_exact_weld',
          'unmodified_official_pack', 'original_grounding_metric_bake')
MATH = ('infra/exact_mesh_geometry.py', 'infra/object_budget_endpoint.py', 'infra/mesh_serialization_geometry.py',
        'src/world_reward/exact_triangle_predicates.py', 'src/world_reward/mesh_serialization.py',
        'src/world_reward/mesh_conditioning.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path, *, readonly=True):
    path = Path(path); before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and (not readonly or not before.st_mode & 0o222)
            and 0 < before.st_size <= 32 << 20, 'Bounded readonly single-link conditioned artifact required')
    return _identity(path)


def _pin(value, *, empty=False):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and (0 if empty else 1) <= value['bytes'] <= 32 << 20
            and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'Exact bounded byte/SHA identity required')


def paths(episode, cache_revision):
    base = f'outputs/episode_{episode:06d}'; cached = 'results/mesh-conditioned-cache-' + cache_revision
    return dict(report=base+'/object_budget_conditioned/report.json', geometry=base+'/object_budget_conditioned/geometry.npz',
        glb=base+'/object_budget_conditioned/object_fixed_canonical.glb', object=base+'/object_grounded/report.json',
        alignment=base+'/scale_smoke/report.json', source_glb=base+'/object_grounded/object.glb',
        transform=base+'/object_grounded/transform.json', intrinsics=base+'/object_grounded/intrinsics.json',
        build='results/image-volume-qem.json', cache_host=cached+'/report.json', cache_native=cached+'/native.json')


def _fields(record, expected):
    require(type(record) is dict and all(type(record.get(k)) is type(v) and record[k] == v for k, v in expected.items()),
            'Exact conditioned producer/source fields required')


def _measurement(value, limit):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= limit,
            'Finite frozen 1% surface/5% shell-volume gates required')


def _source(record, pins):
    binding = record['source_binding']; rows = binding['helpers']
    require(binding['producer_revision'] == pins['report']['producer_revision'] and type(rows) is dict
            and type(binding['source_files']) is int and binding['source_files'] == len(rows),
            'Whole original CPU source ledger required')
    for name, pin in rows.items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Canonical source ledger names required'); _pin(pin, empty=True)
    require(hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest() == binding['source_files_sha256']
            and rows[PROCESSOR]['sha256'] == pins['report']['script_sha256']
            and binding['markers']['revision'] == dict(bytes=41,
                sha256=hashlib.sha256((binding['producer_revision']+'\n').encode()).hexdigest()),
            'Independent CPU processor/whole-source digest differs')
    _pin(binding['markers']['source-sha256'])
    required = (*MATH, CACHE_PINS, 'configs/object_budget_conditioned_protocol_v1.json',
                'configs/mesh_conditioned_cache_protocol_v1.json', 'infra/mesh_conditioned_qem.cpp')
    require(all(identity(CODE/n) == rows[n] for n in required), 'Qualified CPU policy or reused pure math differs')
    return binding


def verify_pinned_artifacts(root, pins, episode, input_sha, object_sha, alignment_sha, scale):
    require(type(episode) is int and 0 <= episode < 30 and type(scale) is float and math.isfinite(scale) and scale > 0,
            'Selected Track1 episode and finite positive original scale required')
    require(all(type(s) is str and re.fullmatch('[0-9a-f]{64}', s) for s in (input_sha, object_sha, alignment_sha)),
            'Independent input/source SHA identities required')
    require(type(pins) is dict and set(pins) == {'schema', 'episode_index', 'input_sha256', 'metric_scale_baked_once', 'report', 'files'},
            'Mandatory exact conditioned mesh pins required; no fallback')
    _fields(pins, dict(schema=SCHEMA, episode_index=episode, input_sha256=input_sha, metric_scale_baked_once=scale))
    producer = pins['report']
    require(type(producer) is dict and set(producer) == {'bytes', 'sha256', 'producer_revision', 'script_sha256'}
            and re.fullmatch('[0-9a-f]{40}', producer['producer_revision'])
            and re.fullmatch('[0-9a-f]{64}', producer['script_sha256']), 'Independent conditioned CPU processor pins required')
    _pin({k: producer[k] for k in ('bytes', 'sha256')})
    base = f'outputs/episode_{episode:06d}/object_budget_conditioned/report.json'
    require(type(pins['files']) is dict and len(pins['files']) == 11 and pins['files'].get(base)
            == {k: producer[k] for k in ('bytes', 'sha256')}, 'Exactly eleven conditioned lineage artifacts required')
    # Authenticate the producer receipt before using its cache revision to form the fixed allowlist.
    require(identity(root/base) == pins['files'][base], 'Frozen proposal receipt differs')
    report = strict_json((root/base).read_text()); revision = report['cache_qualification']['source_binding']['producer_revision']
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Qualified cache revision required')
    names = paths(episode, revision)
    require(set(pins['files']) == set(names.values()), 'Exact fixed artifact allowlist required; no broad paths')
    for pin in pins['files'].values(): _pin(pin)
    new_roles = {'report', 'geometry', 'glb', 'cache_host', 'cache_native'}
    before = {name: identity(root/name, readonly=name in {names[k] for k in new_roles}) for name in sorted(pins['files'])}
    require(before == pins['files'] and before[names['object']]['sha256'] == object_sha
            and before[names['alignment']]['sha256'] == alignment_sha, 'All eleven original artifacts must match before interpretation')
    _fields(report, dict(stage=STAGE, status='pass', phase='complete', episode_index=episode,
        producer_revision=producer['producer_revision'], script_sha256=producer['script_sha256'],
        input_track='track_1', input_sha256=input_sha, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        target_faces=4096, target_vertices=4096, components_deleted=False, holes_filled=False, normals_repaired=False,
        frame_poses_changed=False, native_cost_and_placement_unchanged=False, new_numeric_algorithm=True, cost_normalization=True,
        adoption_performed=False, challenge_performance_verified=False, metric_scale_accuracy_verified=False,
        source_arrays_unchanged=True, object_scale=1., metric_scale_baked_once=scale, owned_scratch_removed=True,
        source_rehashed_after=True, cache_rehashed_after=True, runtime_rehashed_after=True,
        helpers_rehashed_after=True, inputs_rehashed_after=True, native_attempts=1, native_returned=True, native_exit_code=0))
    _source(report, pins)
    records = {role: strict_json((root/names[role]).read_text()) for role in ('object', 'alignment', 'build', 'cache_host', 'cache_native')}
    for role, stage in (('object', 'sam3d_objects_grounded_fixed_frame'), ('alignment', 'predicted_human_anchored_moge2_pointmaps')):
        _fields(records[role], dict(stage=stage, status='pass', episode_index=episode, input_track='track_1',
            input_sha256=input_sha, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[]))
    source = report['source_hashes']
    require(source['video'] == input_sha and source['object_report'] == object_sha and source['alignment_report'] == alignment_sha,
            'Original grounding ancestry differs')
    obj = records['object']; transform = strict_json((root/names['transform']).read_text())
    require(obj['pointmap_grounding']['alignment_report_sha256'] == alignment_sha
            and obj['scale_source'] == 'already_human_anchored_MoGe2_no_second_scalar' and obj['transform'] == transform,
            'Original source transform/grounding differs')
    _fields(obj, dict(frame_index=0))
    _fields(records['alignment'], dict(coordinate_frame='OpenCV_x_right_y_down_z_forward',
                                      pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied'))
    values = transform['scale']
    require(type(values) is list and len(values) == 3 and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in values)
            and float(values[0]) == scale and all(abs(x-values[0]) <= 1e-5*abs(values[0]) for x in values),
            'Original isotropic scalar cannot be averaged or rebaked')
    for role, source_name, field in (('source_glb', 'object.glb', 'object_sha256'),
                                   ('transform', 'transform.json', 'transform_sha256'), ('intrinsics', 'intrinsics.json', 'intrinsics_sha256')):
        require(before[names[role]]['sha256'] == source[source_name] == obj[field], 'Original constant object artifact differs')
    require(set(report['stages']) == set(STAGES), 'Every original physical serialization stage required')
    for name in STAGES[1:]:
        value = report['stages'][name]; shells = value['birthface_matched_shells']
        require(type(shells) is list and shells and value['scale_or_pose_fitted'] is False, 'All full-support shell birth proofs required')
        _measurement(value['sampled_bidirectional_chamfer_diagonal_ratio'], .01); _measurement(value['net_volume_relative_error'], .05)
        for shell in shells: _measurement(shell['relative_volume_error'], .05)
    require(report['candidate_serialization']['position_weld_admissible'] is True
            and report['candidate_serialization']['float32_triangles_exactly_active'] is True,
            'Actual candidate serialization safety required')
    _fields(report['conditioning'], dict(roundtrip_numerically_exact=True, source_arrays_modified=False, source_geometry_repaired=False))
    _fields(report['official_helper_identity'], dict(bytes=2031, sha256='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0'))
    _fields(report['official_pack_fidelity'], dict(oriented_triangles_exact=True, official_helper_simplification_invoked=False,
                                                nonexact_merge_or_face_deletion=False))
    for role, key in (('geometry', 'geometry_sha256'), ('glb', 'canonical_glb_sha256')):
        require(before[names[role]]['sha256'] == report[key]
                and report['outputs'][Path(names[role]).name] == before[names[role]], 'Frozen CPU output differs')
    _cache(records, report, revision, before, names)
    require({name: identity(root/name, readonly=name in {names[k] for k in new_roles}) for name in before} == before,
            'Original proposal artifacts changed during proof checks')
    return report, before, names


def _cache(records, report, revision, observed, names):
    cp = strict_json((CODE/CACHE_PINS).read_text()); q = report['cache_qualification']
    host, native = records['cache_host'], records['cache_native']
    require(cp['schema'] == 'world_reward.mesh_conditioned_cache_qualification.v1' and cp['producer_revision'] == revision
            and cp['host_report'] == observed[names['cache_host']] == q['host_report']
            and cp['native_report'] == observed[names['cache_native']] == q['native_report']
            and q['pins_identity'] == identity(CODE/CACHE_PINS), 'Independent actual cache pins/receipts differ')
    expected_markers = {name: dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name, raw in
        (('revision', (revision+'\n').encode()), ('source-sha256', (cp['source_archive_sha256']+'\n').encode()))}
    require(host['stage'] == 'mesh_conditioned_cache_host_v1' and native['stage'] == 'mesh_conditioned_cache_native_v1'
            and host['status'] == native['status'] == 'pass' and native['phase'] == 'complete'
            and host['source_binding'] == native['source_binding'] == q['source_binding']
            and q['source_binding']['producer_revision'] == revision and q['source_binding']['source_files'] == cp['source_files']
            and q['source_binding']['source_files_sha256'] == cp['source_files_sha256']
            and q['source_binding']['markers'] == expected_markers
            and host['native_identity'] == cp['native_report']
            and native['build']['binary'] == native['retained_binary'] == host['retained_binary'] == cp['retained_binary'] == q['retained_binary']
            and native['build']['build_info'] == q['build_info'] and q['build_info']['physical_coordinate_cache'] is True
            and native['original_runtime'] == q['original_runtime']
            and all(host[k] is True for k in ('source_rehashed_after', 'original_build_rehashed_after', 'owned_container_removed'))
            and all(native[k] is True for k in ('originals_rehashed_after', 'owned_scratch_removed'))
            and all(r[k] is False for r in (host, native) for k in ('gpu_used', 'production_mesh_used', 'challenge_performance_verified', 'adoption')),
            'Actual completed CPU cache evidence required; no GPU build-info execution')
    require(q['source_binding']['helpers']['infra/mesh_conditioned_qem.cpp'] == cp['source_cpp']
            and q['build_info']['source_sha256'] == cp['source_cpp']['sha256']
            and q['build_info']['physical_geometry_rescaled'] is False
            and q['build_info']['native_cost_and_placement_unchanged'] is False
            and q['build_info']['cost_normalization'] is True and q['build_info']['volume_relative_limit'] == .05
            and report['runtime_identity']['actual_fast_build_info'] == q['build_info'], 'Authenticated fixed-chart algorithm ABI differs')
    require(native['geometry']['status'] == 'pass' and native['geometry']['exact_implementation_regression_verified'] is True
            and cp['exact_implementation_regression_verified'] is True and cp['native_calls'] == 8
            and records['build']['image_id'] == host['original_image_id'] == report['image_id']
            and records['build']['status'] == 'pass'
            and report['runtime_identity']['image_receipt'] == observed[names['build']], 'CPU runtime/cache ancestry differs')


def load(root, episode, input_sha, object_report_sha, alignment_sha, scale, *, pins=None):
    root = Path(root)
    report, before, names = verify_pinned_artifacts(root, pins, episode, input_sha, object_report_sha, alignment_sha, scale)
    # Pure readers/math only after producer and exact helper source authentication.
    geometry = importlib.import_module('exact_mesh_geometry'); endpoint = importlib.import_module('object_budget_endpoint')
    for module in (geometry, endpoint):
        require(Path(module.__file__).resolve().parent.parent == CODE, 'Actual source-bound geometry math required')
    glb = root/names['glb']
    with np.load(root/names['geometry'], allow_pickle=False) as data:
        require(set(data.files) == {'vertices', 'faces', 'episode_index', 'object_scale', 'grounded_scale_baked'}, 'Exact CPU NPZ payload required')
        v, f = data['vertices'].copy(), data['faces'].copy()
        require(v.dtype == np.float64 and f.dtype == np.int64 and v.shape == f.shape == (4096, 3)
                and np.isfinite(v).all() and np.all((f >= 0) & (f < 4096)), 'Exact finite official 4096 F64/I64 arrays required')
        require(all(data[n].shape == () and data[n].dtype == dtype and data[n].item() == value for n, dtype, value in
            (('episode_index', np.dtype('int64'), episode), ('object_scale', np.dtype('float64'), 1.),
             ('grounded_scale_baked', np.dtype('float64'), scale))), 'Original metric scalar already baked; never scale again')
    canonical = endpoint._load_mesh(glb)
    compact, fidelity = geometry.verify_pack_fidelity((canonical[0].astype(v.dtype)*scale, canonical[1]), v, f)
    active = np.flatnonzero(~np.all(f == 0, axis=1)).astype(np.int64)
    from world_reward.mesh_geometry import normalize_degenerate_faces
    normalized, cleanup = normalize_degenerate_faces(v, f)
    require(np.array_equal(active, normalized), 'Legacy normalization would delete meaningful qualified faces')
    topology = geometry.exact_mesh_topology(*compact)
    def array_hash(array):
        return hashlib.sha256(json.dumps(dict(dtype=array.dtype.str, shape=array.shape), sort_keys=True).encode()
                              + b'\0' + np.ascontiguousarray(array).tobytes()).hexdigest()
    historical = report['stages']['original_grounding_metric_bake']['candidate_topology']
    require(topology == historical and [array_hash(a) for a in compact]
            == report['stages']['original_grounding_metric_bake']['stored_array_sha256'], 'Stored metric topology/arrays differ from CPU-qualified stage')
    require({name: identity(root/name, readonly=name in {names[k] for k in ('report','geometry','glb','cache_host','cache_native')})
             for name in before} == before, 'Pinned artifacts changed during numerical loading')
    receipt = dict(backend='frozen_conditioned_cached_qem', cpu_report_sha256=before[names['report']]['sha256'],
        geometry_sha256=before[names['geometry']]['sha256'], cpu_producer_revision=pins['report']['producer_revision'],
        cpu_script_sha256=pins['report']['script_sha256'], original_artifacts_rehashed=True,
        metric_scale_already_baked=True, resimplification_performed=False, actual_topology_verified=True,
        metric_oriented_triangle_fidelity=fidelity, independent_embedding_reverified_here=False,
        source_geometry_repaired=False, challenge_performance_verified=False, cpu_source_binding=report['source_binding'])
    return v, f, active, cleanup, glb, receipt
