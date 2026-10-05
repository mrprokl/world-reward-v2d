"""One qualified whole-solid budget proposal; no repair, retry or adoption.

Original public RGB is hashed for ancestry, never decoded. Only predicted object
geometry and metadata are consumed. Native six-stage gates remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import time

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_object_budget_solid'
PINS = 'configs/solid_chart_v2_qualification_pins.json'
BALANCED_PINS = 'configs/solid_chart_v2_balanced_qualification_pins.json'
NATIVE_SECONDS, HOST_SECONDS = 1800, 1900
HELPERS = ('infra/object_budget_solid.py', 'infra/run_object_budget_solid.sh', PINS,
    'infra/solid_chart_v2_qualify.py', 'infra/solid_chart_v2_build.py', 'infra/certified_solid_build.py',
    'infra/certified_solid_source_job.py', 'infra/oriented_solid_compiler.py', 'infra/object_budget_endpoint.py', 'infra/body_smoke.py')
OUTPUTS = ('object_fixed_canonical.glb', 'geometry.npz')
STAGES = frozenset(('physical_source', 'native_candidate', 'float32_glb', 'default8_exact_weld',
    'unmodified_official_pack', 'original_grounding_metric_bake'))


def require(ok, reason):
    if not ok: raise ValueError(reason)


def helpers(code):
    require(code.is_absolute() and code.resolve() == code and not any(p.is_symlink() for p in (code, *code.parents)), 'Canonical own source required')
    sys.path[:0] = [str(code/'infra'), str(code/'src')]
    import solid_chart_v2_qualify as qual
    build, certificate, built = qual.helpers(code)
    import body_smoke as body
    return qual, build, certificate, built, body


def binding(code, revision, qual, build, *, query_requalification=False):
    require(type(query_requalification) is bool, 'Explicit query profile required')
    require(re.fullmatch('[0-9a-f]{40}', revision) and code == ROOT/'jobs'/revision/ENTRY/'code' and
        Path(__file__).resolve() == code/'infra/object_budget_solid.py', 'Exact new production entry required')
    rows, result = qual.snapshot(code, revision, build)
    names = tuple(BALANCED_PINS if n == PINS else n for n in HELPERS) if query_requalification else HELPERS
    require(set(names) <= set(rows), 'Complete own source closure required')
    return result | dict(helpers={n: rows[n] for n in names})


def qualification(code, qual, build, certificate, built, *, query_requalification=False):
    """Compose a fixed qualified query with its unchanged original QEM build."""
    require(type(query_requalification) is bool, 'Explicit query profile required')
    options = dict(query_requalification=True) if query_requalification else {}
    selected = BALANCED_PINS if query_requalification else PINS
    pin_identity = build.identity(code/selected, readonly=True, maximum=16384)
    pins = build.strict_json((code/selected).read_bytes())
    keys = {'schema', 'producer_revision', 'report', 'native', 'build_pins', 'source_archive_sha256', 'source_files',
        'source_files_sha256', 'source_readonly_ledger_sha256', 'qualified_controls', 'native_qem_calls', 'native_query_calls',
        'adoption', 'reconstruction_accuracy_verified'}
    require(set(pins) == keys and pins['schema'] == 'world_reward.solid_chart_v2_qualification_pins.v1' and
        re.fullmatch('[0-9a-f]{40}', pins['producer_revision']) and pins['adoption'] is pins['reconstruction_accuracy_verified'] is False and
        type(pins['source_files']) is int and pins['source_files'] > 0 and all(type(pins[k]) is str and
        re.fullmatch('[0-9a-f]{64}', pins[k]) for k in ('source_archive_sha256', 'source_files_sha256', 'source_readonly_ledger_sha256')) and
        all(type(pins[k]) is int and pins[k] == n for k, n in (('qualified_controls', 4), ('native_qem_calls', 4), ('native_query_calls', 40))),
        'Actual completed independent procedural qualification required')
    producer = ROOT/'jobs'/pins['producer_revision']/'run_solid_chart_v2_qualify/code'
    rows, source = qual.snapshot(producer, pins['producer_revision'], build)
    require(all(source[k] == pins[k] for k in ('source_files', 'source_files_sha256', 'source_readonly_ledger_sha256')) and
        (producer.parent/'source-sha256').read_bytes() == (pins['source_archive_sha256']+'\n').encode(), 'Original qualification source ledger differs')
    qual_names = qual.HELPERS + ((qual.BALANCED_PINS, 'infra/certified_solid_query.cpp') if query_requalification else ())
    old_binding = source | dict(helpers={n: rows[n] for n in qual_names})
    prefix = 'solid-chart-v2-query-requalify-' if query_requalification else 'solid-chart-v2-qualify-'
    out = ROOT/'results'/(prefix+pins['producer_revision'])
    require(out.resolve() == out and not out.is_symlink() and stat.S_IMODE(out.lstat().st_mode) == 0o555, 'Original sealed qualification required')
    paths = {k: out/(k+'.json') for k in ('report', 'native')}
    for k, path in paths.items(): require(build.identity(path, readonly=True, maximum=2 << 20) == pins[k] and
        stat.S_IMODE(path.lstat().st_mode) == 0o444, 'Qualified receipt changed')
    host, native = (build.strict_json(paths[k].read_bytes()) for k in ('report', 'native'))
    require(host['stage'] == 'solid_chart_v2_qualification_host_v1' and native['stage'] == 'solid_chart_v2_qualification_native_v1' and
        host['status'] == native['status'] == 'pass' and host['phase'] == native['phase'] == 'complete' and host['native'] == native and
        host['source_binding'] == host['source_binding_after'] == native['source_binding'] == native['source_binding_after'] == old_binding and
        all(r[k] is True for r in (host, native) for k in ('source_rehashed_after', 'artifacts_rehashed_after', 'official_rehashed_after')) and
        all(r[k] is False for r in (host, native) for k in ('gpu_used', 'gt_used', 'adoption', 'reconstruction_accuracy_verified', 'competition_eligibility_verified')) and
        host['owned_container_removed'] is host['owned_scratch_removed'] is True and native['source_arrays_unchanged'] is True and
        host['qualified_procedural_controls'] == native['qualified_procedural_controls'] == 4 and
        0 < host['elapsed_seconds'] <= 1900 and 0 < native['elapsed_seconds'] <= 1800, 'Incomplete qualification lifecycle')
    qual.validate_controls(native['controls'], native['control_manifest'])
    require(native['control_manifest_sha256'] == hashlib.sha256(json.dumps(native['control_manifest'], sort_keys=True).encode()).hexdigest(), 'Frozen cohort manifest differs')
    buildpins, binary, query, buildpaths, buildproof = qual.built_qualification(code, build, certificate, built, **options)
    require(build.identity(code/qual.BUILD_PINS, readonly=True) == pins['build_pins'] and
        host['qualified_build'] == native['qualified_build'] == buildproof and native['image_id'] == buildpins['image_id'] and
        all(rows[n] == build.identity(code/n, readonly=True, empty=True) for n in qual_names), 'Actual qualified compiler/source differs')
    if query_requalification:
        require(host.get('query_requalification') == native.get('query_requalification') == buildproof['query_requalification'],
            'Active query composition qualification differs')
    mounts = (out, producer, producer.parent/'revision', producer.parent/'source-sha256', *buildpaths)
    proof = dict(pins_identity=pin_identity, source=source, report=pins['report'], native=pins['native'], build=buildproof)
    if query_requalification: proof['query_requalification'] = buildproof['query_requalification']
    return buildpins, binary, query, mounts, proof


def successful_queries(document, expected):
    records = []
    def visit(value):
        if isinstance(value, dict):
            if 'query_attempted' in value:
                require(value['query_attempted'] is value['query_returned'] is True and type(value['query_returncode']) is int and
                    value['query_returncode'] == 0 and value['native_certificate']['status'] == 'pass' and
                    set(value['forest']) == {'component_keys', 'signs', 'inside', 'parents', 'depths'}, 'Incomplete exact query/forest')
                query_record(value)
                records.append(value)
            for child in value.values(): visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value: visit(child)
    visit(document); require(len(records) == expected, 'Actual exact-query census differs')
    return len(records)


def query_record(row):
    """Metadata gate only; exact geometry predicates remain the qualified native."""
    c, f, top = row['native_certificate'], row['forest'], row['topology']; n = c['component_count']
    require(type(n) is int and 1 <= n <= 256 and c['schema'] == 'world_reward.certified_solid_query.v1' and c['cgal_version'] == '6.0.1' and
        all(type(c[k]) is int and c[k] >= 4 for k in ('vertices','faces')) and
        all(c[k] is True for k in ('closed_oriented_vertex_manifold_verified','all_original_faces_retained','all_original_vertices_referenced',
            'exact_nondegenerate_triangles_verified','component_self_intersections_absent','inter_component_surface_contacts_absent')) and
        all(c[k] is False for k in ('geometry_repaired','orientation_changed','qem_executed','forest_adjudicated','reconstruction_accuracy_verified')) and
        type(f['component_keys']) in (tuple, list) and len(set(f['component_keys'])) == len(f['component_keys']) == n and
        all(type(k) is str and k for k in f['component_keys']) and all(type(f[k]) is list and len(f[k]) == n for k in ('signs','parents','depths','inside')) and
        c['inside'] == f['inside'] and len(c['components']) == n and top['vertices'] == top['active_vertices'] == c['vertices'] and top['faces'] == c['faces'] and
        top['closed_oriented_vertex_manifold'] is True and len(top['components']) == n, 'Full exact certificate/topology differs')
    for i in range(n):
        inside = f['inside'][i]; parent = f['parents'][i]; depth = f['depths'][i]; component = c['components'][i]
        require(type(inside) is list and len(inside) == n and all(type(x) is bool for x in inside) and not inside[i] and
            type(depth) is int and depth == sum(inside) and type(parent) is int and type(f['signs'][i]) is int and
            f['signs'][i] == (1 if depth % 2 == 0 else -1) and component['original_component_id'] == i and
            component['exact_volume_sign'] == f['signs'][i] and component['original_vertices'] >= 4 and component['original_faces'] >= 4 and
            (parent == -1 if depth == 0 else 0 <= parent < n and inside[parent] and f['depths'][parent] == depth-1), 'Typed original material forest differs')


def validate_compiler_proof(c):
    successful_queries(c, 8); source = c['stages']['physical_source']; metric = c['metric_source_certificate']
    require(source['forest'] == metric['forest'] and source['conditioning']['roundtrip_numerically_exact'] is True and
        source['float32_orientation']['exact_positive_normal_dot'] is True and source['conditioning_header']['bytes'] > 0,
        'Full physical/metric source forest and original F32 chart required')
    references = [(source['float32_certificate'], source), (metric, source)] + [
        (c['stages'][s], metric if s == 'original_grounding_metric_bake' else source) for s in STAGES if s != 'physical_source']
    for row, reference in references:
        f, original, permutation = row['forest'], reference['forest'], row['candidate_to_source_components']; n = len(original['signs'])
        require(type(permutation) is list and all(type(x) is int for x in permutation) and sorted(permutation) == list(range(n)) and
            len(f['signs']) == n and all(f['signs'][i] == original['signs'][s] and f['depths'][i] == original['depths'][s] and
                (-1 if f['parents'][i] == -1 else permutation[f['parents'][i]]) == original['parents'][s] and
                all(f['inside'][i][j] == original['inside'][s][permutation[j]] for j in range(n)) for i,s in enumerate(permutation)),
            'All source shells/signs/depths/parents/inside must survive birth bijection')
        if 'fidelity' in row:
            fidelity = row['fidelity']; shells = fidelity['birthface_matched_shells']
            # scipy component IDs follow minimum original vertex, while the
            # certified forest follows first loaded face. Never equate labels.
            scipy_to_forest = [s['original_component_id'] for s in sorted(reference['native_certificate']['components'],
                key=lambda s:s['witness_original_vertex'])]
            require(fidelity['scale_or_pose_fitted'] is False and 0 <= fidelity['sampled_bidirectional_chamfer_diagonal_ratio'] <= .01 and
                0 <= fidelity['net_volume_relative_error'] <= .05 and len(shells) == n and
                sorted(s['source_component'] for s in shells) == list(range(n)) and all(0 <= s['relative_volume_error'] <= .05 and
                s['volume_sign'] == original['signs'][scipy_to_forest[s['source_component']]] and type(s['euler']) is int for s in shells) and
                sorted(s['euler'] for s in shells) == sorted(s['euler'] for s in reference['topology']['components']) ==
                sorted(s['euler'] for s in fidelity['candidate_topology']['components']), 'Original per-shell Euler/volume/CD gates required')
        else: require(row is source['float32_certificate'] or row is metric, 'Measured candidate fidelity missing')


def publish(path, raw, owned):
    """Register the exclusively owned inode before any fallible write/fsync."""
    descriptor = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o444); m = os.fstat(descriptor)
    count = 0; digest = hashlib.sha256(); owned[path.name] = (m.st_dev,m.st_ino,dict(bytes=0,sha256=digest.hexdigest()))
    try:
        while count < len(raw):
            chunk = raw[count:count+(1<<20)]; written = os.write(descriptor,chunk); require(written > 0,'Short publication write')
            digest.update(chunk[:written]); count += written
            owned[path.name] = (m.st_dev,m.st_ino,dict(bytes=count,sha256=digest.hexdigest()))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally: os.close(descriptor)


def input_binding(episode, body, build):
    require(type(episode) is int and 0 <= episode < 30, 'Original Track1 episode0..29 required')
    inputs = body._validate_inputs(ROOT, episode_index=episode); base = ROOT/f'outputs/episode_{episode:06d}'
    paths = (ROOT/'results/input-manifest.json', ROOT/'data/track_1/meta/episodes.jsonl', inputs['video'],
        base/'automatic_masks/report.json', base/'automatic_masks/prompts.json',
        *(base/n for n in ('object_grounded/report.json', 'object_grounded/object.glb', 'object_grounded/transform.json',
            'object_grounded/intrinsics.json', 'scale_smoke/report.json', 'body_smoke/report.json', 'depth_smoke/report.json')))
    pins = {str(p): build.identity(p, maximum=32 << 30) for p in paths}
    maskroot = inputs['human_masks']; masks = tuple(sorted(maskroot.glob('*.png')))
    require(maskroot.resolve() == maskroot and not maskroot.is_symlink() and len(masks) == inputs['total_frames'], 'Original full-T masks required')
    pins.update({str(p): build.identity(p, maximum=32 << 20) for p in masks})
    return inputs, (*paths, maskroot), dict(files=pins, episode_index=episode, total_frames=inputs['total_frames'],
        dataset_revision=inputs['dataset_revision'], video_sha256=inputs['video_sha256'], input_video_hashed=True, media_decoded=False)


def produce(episode, binary, query, query_sha, work, official, left, report, build):
    import numpy as np
    import object_budget_endpoint as endpoint
    import oriented_solid_compiler as compiler
    require(np.__version__.startswith('1.26.'), 'Qualified NumPy1.26 required')
    inputs, sources, scale = endpoint.prerequisites(ROOT, episode)
    source, source_proof = compiler.load_source(ROOT/f'outputs/episode_{episode:06d}/object_grounded/object.glb')
    report.update(source_hashes=sources, source_geometry=source_proof, original_grounded_scale=scale, input_sha256=inputs['video_sha256'])
    compiler_work = work/'compiler'; compiler_work.mkdir(mode=0o700)
    try:
        metric, report['compiler'] = compiler.compile_solid(source, binary, query, query_sha, compiler_work, official,
            metric_scale=scale, remaining=left, conditioning_version=2)
    except compiler.SolidCompilerError as error: report['compiler'] = error.report; raise
    glb = work/OUTPUTS[0]; candidate = compiler_work/'candidate.glb'
    require(build.identity(candidate) == report['compiler']['glb_identity'], 'Certified GLB changed')
    build.seal(glb, candidate.read_bytes()); require(build.identity(glb, readonly=True) == report['compiler']['glb_identity'], 'Copied GLB differs')
    npz = work/OUTPUTS[1]
    with npz.open('xb') as stream:
        np.savez_compressed(stream, vertices=metric[0], faces=metric[1], episode_index=np.array(episode),
            object_scale=np.array(1.), grounded_scale_baked=np.array(scale)); stream.flush(); os.fsync(stream.fileno())
    os.chmod(npz, 0o444)
    with np.load(npz, allow_pickle=False) as saved:
        require(set(saved.files) == {'vertices', 'faces', 'episode_index', 'object_scale', 'grounded_scale_baked'} and
            np.array_equal(saved['vertices'], metric[0]) and np.array_equal(saved['faces'], metric[1]) and
            saved['episode_index'].item() == episode and saved['object_scale'].item() == 1. and saved['grounded_scale_baked'].item() == scale,
            'Saved metric geometry differs')
    require(endpoint.prerequisites(ROOT, episode)[1:] == (sources, scale), 'Original grounded ancestry changed')
    report.update(frame_poses_changed=False, object_scale=1., metric_scale_baked_once=scale)
    return (glb, npz)


def validate_native(result, episode, source, proof, inputs):
    require(result['stage'] == 'world_reward_object_budget_solid_native_v1' and result['status'] == 'pass' and result['phase'] == 'complete' and
        result['source_binding'] == result['source_binding_after'] == source and result['qualification'] == proof and result['input_binding'] == inputs and
        type(result['episode_index']) is int and result['episode_index'] == episode and result['input_track'] == 'track_1' and
        result['input_sha256'] == inputs['video_sha256'] and result['source_hashes']['video'] == inputs['video_sha256'] and
        all(result[k] is False for k in ('gpu_used', 'ground_truth_used', 'hand_labeled_test', 'adoption', 'reconstruction_accuracy_verified',
            'competition_eligibility_verified', 'media_decoded', 'frame_poses_changed')) and result['oracle_modes'] == [] and
        all(result[k] is True for k in ('input_video_hashed', 'source_rehashed_after', 'inputs_qualification_rehashed_after', 'runtime_rehashed_after')) and
        all(type(result[k]) is int and result[k] == n for k, n in (('budget_seconds', 1800), ('qem_seconds', 1200), ('query_seconds', 180),
            ('maximum_qem_calls', 1), ('maximum_query_calls', 8))) and 0 < result['elapsed_seconds'] <= 1800 and
        type(result['native_query_calls']) is int and result['native_query_calls'] == 8 and
        set(result['outputs']) == set(OUTPUTS), 'Complete source-bound native proposal required')
    c = result['compiler']; scale = result['original_grounded_scale']
    require(type(scale) is float and math.isfinite(scale) and scale > 0 and result['metric_scale_baked_once'] == scale and result['object_scale'] == 1. and
        c['stage'] == 'oriented_solid_compiler_v2' and type(c['conditioning_version']) is int and c['conditioning_version'] == 2 and
        c['status'] == 'pass' and c['phase'] == 'complete' and type(c['native_attempts']) is int and c['native_attempts'] == 1 and
        c['native_returned'] is True and type(c['native_returncode']) is int and c['native_returncode'] == 0 and c['source_arrays_unchanged'] is True and
        c['artifacts_after'] == c['artifacts_before'] and set(c['stages']) == STAGES and
        c['metric_scale_baked_once'] == scale and c['output_vertices'] == c['output_faces'] == 4096 and
        all(c[k] is False for k in ('geometry_repaired', 'components_deleted', 'orientation_changed', 'cost_backend_changed',
            'ground_truth_used', 'adoption', 'reconstruction_accuracy_verified', 'frame_poses_changed', 'metric_scale_accuracy_verified')) and
        c['glb_identity'] == result['outputs'][OUTPUTS[0]], 'Whole six-stage unchanged compiler proof required')
    validate_compiler_proof(c)
    require(all(k in c['stages']['physical_source'] for k in ('conditioning', 'float32_orientation', 'float32_certificate', 'conditioning_header')) and
        c['stages']['physical_source']['conditioning']['roundtrip_numerically_exact'] is True and
        c['stages']['physical_source']['float32_orientation']['exact_positive_normal_dot'] is True and
        c['native_mapping']['conditioning']['chart_version'] == 2 and c['native_mapping']['conditioning']['source_roundtrip_numerically_exact'] is True and
        type(c['native_mapping']['serialization']['committed_collapses']) is int and c['native_mapping']['serialization']['committed_collapses'] >= 0,
        'Original F64/F32/source chart and native mapping proof required')


def native(episode, code, revision, work, qual, build, certificate, built, body, *, query_requalification=False):
    require(type(query_requalification) is bool, 'Explicit query profile required')
    options = dict(query_requalification=True) if query_requalification else {}
    start = time.monotonic(); left = lambda: built.remaining(start, NATIVE_SECONDS)
    require(work.resolve() == work and not work.is_symlink() and sys.platform == 'linux' and os.getuid() == work.stat().st_uid == 1000 and stat.S_IMODE(work.stat().st_mode) == 0o700 and
        not any(work.iterdir()) and os.environ.get('WR_NATIVE_NETWORK') == 'none' and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Fresh isolated CPU work required')
    source = binding(code, revision, qual, build, **options); report = dict(stage='world_reward_object_budget_solid_native_v1', status='fail', phase='prerequisites',
        episode_index=episode, source_binding=source, gpu_used=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        adoption=False, reconstruction_accuracy_verified=False, competition_eligibility_verified=False, input_track='track_1',
        input_video_hashed=True, media_decoded=False, budget_seconds=1800, qem_seconds=1200, query_seconds=180, maximum_qem_calls=1, maximum_query_calls=8)
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Inclusive production deadline'))); signal.alarm(max(1, int(left())))
    term = signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGALRM))
    before = None
    try:
        pins, binary, query, _, proof = qualification(code, qual, build, certificate, built, **options)
        if query_requalification: report['query_requalification'] = proof['query_requalification']
        require(os.environ.get('WR_CPU_IMAGE_ID') == pins['image_id'], 'Actual qualified CPU image required')
        _, _, inputs = input_binding(episode, body, build); before = (proof, inputs, qual.official_identity(build))
        import object_budget_conditioned as cache
        runtime = cache.runtime_identity(ROOT, code, cache.cache_qualification(ROOT, code)[1], left)
        require(runtime == proof['build']['original_runtime'], 'Qualified inherited runtime changed')
        report.update(qualification=proof, input_binding=inputs, image_id=pins['image_id'], phase='whole_solid')
        paths = produce(episode, binary, query, certificate.qualification(code, build, **options)[0]['native_source']['sha256'], work, ROOT/qual.OFFICIAL, left, report, build)
        report['native_query_calls'] = successful_queries(report['compiler'], 8)
        report['outputs'] = {p.name: build.identity(p, readonly=True) for p in paths}
        require(cache.runtime_identity(ROOT, code, cache.cache_qualification(ROOT, code)[1], left) == runtime, 'Runtime changed')
        report.update(status='pass', phase='complete', runtime_rehashed_after=True)
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        try:
            require(binding(code, revision, qual, build, **options) == source, 'Own source changed'); report.update(source_binding_after=source, source_rehashed_after=True)
            if before is not None:
                require((qualification(code, qual, build, certificate, built, **options)[4], input_binding(episode, body, build)[2], qual.official_identity(build)) == before, 'Original inputs/qualification changed')
                report['inputs_qualification_rehashed_after'] = True
            for name, pin in report.get('outputs', {}).items(): require(build.identity(work/name, readonly=True) == pin, 'Output changed')
        except Exception as error: report.update(status='fail', posthash_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-start
        if report['elapsed_seconds'] > NATIVE_SECONDS: report.update(status='fail', failure_type='InclusiveDeadline')
        build.write_json(work/'native.json', report); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term)
    return 0 if report['status'] == 'pass' else 1


def host(episode, code, revision, qual, build, certificate, built, body, *, query_requalification=False):
    require(type(query_requalification) is bool, 'Explicit query profile required')
    options = dict(query_requalification=True) if query_requalification else {}
    start = time.monotonic(); left = lambda: built.remaining(start, HOST_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 0 and os.environ.get('DOCKER_HOST') == 'unix://'+str(ROOT/'docker.sock'), 'Owned CPU Docker control required')
    source = binding(code, revision, qual, build, **options); pins, _, _, paths, proof = qualification(code, qual, build, certificate, built, **options)
    original_code = ROOT/'jobs'/proof['build']['original_source']['producer_revision']/'run_solid_chart_v2_build/code' if query_requalification else code
    _, inputpaths, inputs = input_binding(episode, body, build); original, inherited, runtime_proof = built.qualified(original_code, build, certificate)
    if query_requalification:
        require(runtime_proof == proof['query_requalification']['original_qualified_inputs'], 'Original QEM runtime qualification differs')
    image = built.image_identity(original, build, left); official = qual.official_identity(build)
    prefix = 'object_budget_solid_balanced_' if query_requalification else 'object_budget_solid_'
    name = f'wr-object-budget-solid-{episode:06d}-'+('balanced-' if query_requalification else '')+revision
    require(not build.run(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$'], min(10, left())).strip(), 'Foreign/preexisting container')
    out = ROOT/f'outputs/episode_{episode:06d}'/(prefix+revision); require(out.resolve() == out and
        not any(p.is_symlink() for p in out.parents) and not out.exists() and not out.is_symlink(), 'Fresh proposal required')
    out.mkdir(mode=0o755); os.chmod(out, 0o755); work = out/'disposable'; work.mkdir(mode=0o700); os.chown(work, 1000, 1000); owner = work.lstat()
    cid = out/'.container.cid'; published = {}; report = dict(stage='world_reward_object_budget_solid_host_v1', status='fail', phase='native', episode_index=episode,
        producer_revision=revision, source_binding=source, qualification=proof, input_binding=inputs, image_identity=image,
        gpu_used=False, ground_truth_used=False, adoption=False, reconstruction_accuracy_verified=False, competition_eligibility_verified=False,
        input_video_hashed=True, media_decoded=False, maximum_qem_calls=1, maximum_query_calls=8)
    if query_requalification: report['query_requalification'] = proof['query_requalification']
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Host deadline'))); signal.alarm(max(1, int(left())))
    term = signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGALRM))
    try:
        mounts = []
        for p in dict.fromkeys((code, code.parent/'revision', code.parent/'source-sha256', *paths, *inherited, *inputpaths, ROOT/qual.OFFICIAL)):
            mounts.extend(['--mount', f'type=bind,src={p},dst={p},readonly'])
        argv = ['docker', 'run', '--name', name, '--cidfile', str(cid), '--label', 'world_reward.certified_solid.owner='+revision,
            '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '4', '--memory', '16g',
            '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=512m', *mounts, '--mount', f'type=bind,src={work},dst={work}', '--entrypoint', '/usr/bin/env', pins['image_id'],
            '-i', 'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp', 'TMPDIR=/tmp', 'WR_ROOT='+str(ROOT), 'WR_CODE='+str(code),
            'WR_CODE_REVISION='+revision, 'WR_CPU_IMAGE_ID='+pins['image_id'], 'WR_NATIVE_NETWORK=none', 'CUDA_VISIBLE_DEVICES=-1',
            'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1', 'python3', '-I', '-B', str(code/'infra/object_budget_solid.py'), '--episode', str(episode), '--native']
        if query_requalification: argv.append('--query-requalification')
        build.run(argv, min(1810, left()), out/'native.log')
    except Exception as error: report['failure_type'] = type(error).__name__
    finally:
        signal.alarm(60)
        try:
            build.cleanup_container(name, pins['image_id'], revision, cid); report['owned_container_removed'] = True
        except Exception as error: report.update(status='fail', container_cleanup_failure_type=type(error).__name__)
        try:
            if (work/'native.json').is_file():
                pin = build.identity(work/'native.json', readonly=True, maximum=2 << 20); raw = (work/'native.json').read_bytes()
                require(dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == pin, 'Native receipt changed while reading')
                result = build.strict_json(raw); report['native'] = result; build.seal(out/'native.json', raw)
            require(report.get('owned_container_removed') is True, 'Owned container cleanup required')
            require(binding(code, revision, qual, build, **options) == source and qualification(code, qual, build, certificate, built, **options)[4] == proof and
                input_binding(episode, body, build)[2] == inputs and built.image_identity(original, build, left) == image and qual.official_identity(build) == official,
                'Source/inputs/qualified runtime changed')
            if query_requalification:
                require(built.qualified(original_code, build, certificate)[2] == runtime_proof and
                    result.get('query_requalification') == proof['query_requalification'], 'Original runtime/active query proof changed')
            report.update(source_binding_after=source, source_rehashed_after=True, inputs_qualification_rehashed_after=True)
            require('failure_type' not in report, 'Native process must exit zero'); validate_native(result, episode, source, proof, inputs); left()
            for n in OUTPUTS:
                require(build.identity(work/n, readonly=True) == result['outputs'][n], 'Accepted candidate changed')
                publish(out/n, (work/n).read_bytes(), published)
                require(build.identity(out/n, readonly=True) == result['outputs'][n], 'Published candidate differs')
            report.update(status='pass', phase='complete', outputs=result['outputs'])
        except Exception as error: report.update(status='fail', posthash_failure_type=type(error).__name__)
        try:
            require(report.get('owned_container_removed') is True, 'Container cleanup unverified; retain scratch')
            require(work.lstat().st_ino == owner.st_ino and work.lstat().st_dev == owner.st_dev and work.lstat().st_uid == owner.st_uid and
                stat.S_IMODE(work.lstat().st_mode) == 0o700 and not work.is_symlink(), 'Owned work replaced')
            allowed = {'compiler', 'native.json', *OUTPUTS, *(f'compiler/{n}' for n in ('input.obj', 'candidate.obj', 'mapping.json', 'candidate.glb'))}
            for p in work.rglob('*'):
                m = p.lstat(); require(p.relative_to(work).as_posix() in allowed and not p.is_symlink() and
                    m.st_uid == owner.st_uid and
                    (stat.S_ISDIR(m.st_mode) and p.name == 'compiler' or stat.S_ISREG(m.st_mode) and m.st_nlink == 1), 'Unknown work; refuse deletion')
            shutil.rmtree(work); report['owned_scratch_removed'] = True
        except Exception as error: report.update(status='fail', cleanup_failure_type=type(error).__name__)
        try:
            for n, (dev, ino, pin) in published.items():
                p = out/n; m = p.lstat(); require((m.st_dev, m.st_ino) == (dev, ino) and build.identity(p, readonly=True) == pin,
                    'Final published output changed')
        except Exception as error: report.update(status='fail', output_posthash_failure_type=type(error).__name__)
        report['elapsed_seconds'] = time.monotonic()-start
        if report['elapsed_seconds'] > HOST_SECONDS: report.update(status='fail', failure_type='HostInclusiveDeadline')
        if report['status'] != 'pass':
            try:
                for n, (dev, ino, pin) in published.items():
                    p = out/n; m = p.lstat(); require((m.st_dev, m.st_ino) == (dev, ino) and
                        build.identity(p, readonly=True, empty=True) == pin, 'Published output replaced; refuse deletion'); p.unlink()
                report['owned_partial_outputs_removed'] = True
            except Exception as error: report['rollback_failure_type'] = type(error).__name__
            report.pop('outputs', None)
        build.write_json(out/'report.json', report); os.chmod(out, 0o555); signal.alarm(0); signal.signal(signal.SIGALRM, old); signal.signal(signal.SIGTERM, term)
    return 0 if report['status'] == 'pass' else 1


SURFACE_IMAGE = 'sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0'
SURFACE_SECONDS, SURFACE_HOST_SECONDS = 600, 700
SURFACE_OUTPUTS = ('object_fixed_canonical.glb', 'geometry.npz', 'candidate_geometry.npz', 'mapping.json')
SURFACE_HELPERS = ('infra/object_budget_solid.py', 'infra/run_object_budget_solid.sh',
    'src/world_reward/surface_budget.py', 'src/world_reward/surface_identity.py',
    'src/world_reward/exact_triangle_predicates.py', 'src/world_reward/mesh_serialization.py',
    'infra/surface_qslim_qualify.py', 'infra/surface_identity_qualify.py', 'infra/surface_qslim.cpp',
    'infra/official_pack_geometry.py', 'infra/mesh_precision_diagnostic.py', 'infra/body_smoke.py',
    'configs/surface_identity_qualification_pins.json', 'configs/surface_qslim_build_pins.json',
    'configs/surface_qslim_qualification_pins.json')
SURFACE_CONTROL = 'configs/surface_consumer_control_protocol_v1.json'
SURFACE_CONTROL_PIN = dict(bytes=2844, sha256='8f92c9da55a4a87d7ceb64c384ff53d7aca5e9c716c878fc20214a2b5b9b02eb')
SURFACE_TRIMESH_SOURCES = {
    'scene/transforms.py': dict(bytes=28938, sha256='f38beb118974172c42270d035f3bb77eb5d374de8c251aab7bce1a33afbe8ea2'),
    'transformations.py': dict(bytes=74801, sha256='644b112736124b7803c028a248279926d10f748649634006d493a90722eae360'),
    'scene/scene.py': dict(bytes=54380, sha256='6dd01efd09edae58f9d3643d73f9ca943904cb353a52f9d8f6633213b684e15b'),
    'base.py': dict(bytes=110040, sha256='13d002a80f14bfa33cf5e49fab19b60083356c2377d92fc98a701d0d2b3e8706'),
    'exchange/gltf/__init__.py': dict(bytes=79987, sha256='0bebabb3a28a9e75773ddf4135a51198dc107ad61bbbcf4de280191417e2159c'),
    'exchange/load.py': dict(bytes=21516, sha256='6b313c1f0ff9295e1cdf6a5567588f5df7d5c02eea12353fc9d15740d19a9e4e'),
}


def surface_profile(domain, control=None):
    require(type(domain) is str and domain == 'surface' and (control is None or type(control) is str and control=='surface_consumer_v1'),
        'Only explicitly selected surface profile/control allowed')


def surface_trimesh_sources(rt, trimesh):
    directory = Path(trimesh.__file__).resolve().parent
    rows = {n:rt.identity(directory/n, 2<<20, readonly=False) for n in SURFACE_TRIMESH_SOURCES}
    require(rows == SURFACE_TRIMESH_SOURCES, 'Actual raw/native scene graph and fix_rigid source differs')
    return rows


def surface_control_protocol(code, rt):
    config=rt.pinned(code/SURFACE_CONTROL,SURFACE_CONTROL_PIN,16<<10)
    require(config['schema']=='world_reward.surface_consumer_control_protocol.v1' and config['frames']==96 and
        config['QEM_calls']==0 and config['metric_scale_baked_once']==.5, 'Frozen prospective control required')
    return config


def surface_report(path, report, build, left, *, finalize=lambda:None):
    """An over-budget sealing operation cannot leave an adoptable PASS receipt."""
    left()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
    owner=os.fstat(fd)
    def write():
        raw=(json.dumps(report,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n').encode()
        os.lseek(fd,0,os.SEEK_SET);os.ftruncate(fd,0);offset=0
        while offset<len(raw):
            count=os.write(fd,raw[offset:]);require(count>0,'Short receipt write');offset+=count
        os.fchmod(fd,0o444);os.fsync(fd)
    try:
        write(); finalize(); left()
    except Exception:
        report.update(status='fail',failure_type='ReceiptSealingFailure')
        require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(owner.st_dev,owner.st_ino),'Owned receipt descriptor differs')
        write()  # Same still-open owned inode, not a replaceable filesystem path.
        raise
    finally:os.close(fd)


def surface_modules(code):
    require(code.resolve() == code, 'Canonical current code required')
    sys.path[:0] = [str(code/'infra'), str(code/'src')]
    import mediapipe_cpu_runtime_verify as rt
    import surface_qslim_qualify as q
    import certified_solid_build as build
    import body_smoke as body
    return rt, q, build, body


def surface_source(code, revision, rt, *, control=False):
    require(Path(__file__).resolve() == code/'infra/object_budget_solid.py' and
        set(p.name for p in code.parent.iterdir()) == {'code', 'revision', 'source-sha256'}, 'Actual immutable dispatcher required')
    names = SURFACE_HELPERS + ((SURFACE_CONTROL, 'src/world_reward/surface_pose_geometry.py') if control else ())
    return rt.source(ROOT, code, revision, ENTRY, names)


def surface_qualification(code, rt, q):
    """Authenticate qualified native controls without reclassifying failed hosts."""
    path = code/'configs/surface_qslim_qualification_pins.json'
    pinid = rt.identity(path); pins = rt.strict(path.read_bytes())
    q.equal(pins, dict(schema='world_reward.surface_qslim_qualification_pins.v1', historical_host_status='fail',
        native_loader_calls=4, official_budget_calls=2, QEM_calls=2, committed_collapses=4224,
        independent_native_receipt_audit=True, independent_geometry_replay=False, actual_original_CID_absence_verified=True,
        whole_Parquet_qualified=False, full_surface_fidelity_certified=False, embedding_certified=False,
        adoption=False, reconstruction_accuracy_verified=False), 'Original qualified surface scope required')
    rev = pins['producer_revision']; require(type(rev) is str and re.fullmatch('[0-9a-f]{40}', rev), 'Original revision required')
    producer = ROOT/'jobs'/rev/q.ENTRY/'code'; old = q.host_proof(ROOT, producer, rev)
    ledger = hashlib.sha256(); rows = {}
    for p in sorted(producer.rglob('*')):
        s = p.lstat(); require(stat.S_IMODE(s.st_mode) in ((0o555,) if p.is_dir() else (0o444, 0o555)), 'Historical readonly ledger differs')
        if p.is_dir(): continue
        name = p.relative_to(producer).as_posix(); rows[name] = rt.identity(p, 2<<20, empty=True)
        ledger.update(name.encode()+b'\0'+str(stat.S_IMODE(s.st_mode)).encode()+b'\0'+bytes.fromhex(rows[name]['sha256']))
    require(len(rows) == pins['source_files'] and ledger.hexdigest() == pins['source_readonly_ledger_sha256'] and
        (producer.parent/'source-sha256').read_bytes() == (pins['source_archive_sha256']+'\n').encode() and
        all(rows[n] == rt.identity(code/n, 2<<20, empty=True) for n in q.HELPERS if not n.endswith('.sh')),
        'Original full source/current numerical helpers differ')
    out = ROOT/'results'/('surface-qslim-qualify-'+rev)
    native = rt.pinned(out/'native.json', pins['native'], 2<<20)
    host = rt.pinned(out/'report.json', pins['historical_host_report'], 2<<20)
    q.validate_native(native, old['native'], rev)
    # Historical CID-mode cleanup failed before seal() read the native JSON.
    # Authenticate its separate native_identity; do not invent an embedded PASS.
    require(host['status'] == 'fail' and host['native_report'] is None
        and host['native_identity'] == pins['native'] and host['source_proof'] == old,
        'Original outer failure must remain failed and unchanged')
    audit = rt.pinned(ROOT/pins['independent_audit_path'], pins['independent_audit_report'], 16<<10)
    q.equal(audit, dict(stage='world_reward.surface_qslim_independent_receipt_audit.v1', status='pass', producer_revision=rev,
        original_host_status='fail', original_native_status='pass', original_host_receipt_unchanged=True,
        original_exact_CID_absence_verified=True, own_exact_CID_absence_verified=True,
        source_runtime_rehashed_after=True, actual_official_runtime_versions_exact=True, all3_full_source_manifests_exact=True,
        full_source_files=pins['source_files'], original_QEM_calls=2, original_committed_collapses_total=4224,
        native_binary_calls=0, QEM_calls=0, mesh_budget_calls=0, geometry_replay=False, retained_receipt_only=True,
        original_CID_mode='0644_unchanged'), 'Independent original native qualification audit incomplete')
    binary, built = q.build_proof(ROOT, code)
    buildrev = rt.strict((code/q.BUILD_PINS).read_bytes())['producer_revision']
    buildcode = ROOT/'jobs'/buildrev/'run_surface_qslim_build/code'
    buildnative = rt.pinned(ROOT/'results'/('surface-qslim-build-'+buildrev)/'native.json', built['native'], 2<<20)
    build_helpers = tuple(buildnative['source_binding']['helpers'])
    require(rt.source(ROOT,buildcode,buildrev,'run_surface_qslim_build',build_helpers)['markers'] ==
        buildnative['source_binding']['markers'], 'Original build markers differ')
    for name, pin in buildnative['source_binding']['helpers'].items():
        require(rt.identity(buildcode/name,2<<20,empty=True)==pin,'Original build helper changed')
    buildrows, buildledger = {}, hashlib.sha256()
    for p in sorted(buildcode.rglob('*')):
        mode = stat.S_IMODE(p.lstat().st_mode)
        require(mode in ((0o555,) if p.is_dir() else (0o444,0o555)), 'Original build source mode differs')
        if p.is_dir(): continue
        name=p.relative_to(buildcode).as_posix(); buildrows[name]=rt.identity(p,2<<20,empty=True)
        buildledger.update(name.encode()+b'\0'+str(mode).encode()+b'\0'+bytes.fromhex(buildrows[name]['sha256']))
    require(buildnative['source_binding']['source_files']==len(buildrows) and
        buildnative['source_binding']['source_files_sha256']==hashlib.sha256(json.dumps(buildrows,sort_keys=True).encode()).hexdigest() and
        buildnative['source_binding']['source_readonly_ledger_sha256']==buildledger.hexdigest(), 'Complete original build source differs')
    runtime = q.modules()[2].load_runtime(ROOT, code)
    require(runtime['pins']['image_id'] == SURFACE_IMAGE, 'Exact qualified official CPU runtime required')
    phase1 = q.phase1(ROOT, code, historical=True)
    first = rt.strict((code/q.PHASE1_PINS).read_bytes())
    firstcode = ROOT/'jobs'/first['producer_revision']/'run_surface_identity_qualify/code'
    paths = [*q.mount_paths(ROOT, code), producer.parent, firstcode.parent, buildcode.parent, ROOT/q.PHASE1_LOG,
        out/'native.json', out/'report.json', ROOT/pins['independent_audit_path']]
    leafs = {str(p): rt.identity(p, 64<<20, readonly=False) for p in paths if p.is_file()}
    proof = dict(pins_identity=pinid, producer_revision=rev, native=pins['native'], historical_host=pins['historical_host_report'],
        independent_audit=pins['independent_audit_report'], original_host_status='fail',
        original_source_proof=old, original_source_ledger_sha256=ledger.hexdigest(), build=built,
        phase1=phase1, runtime=runtime, leaf_identities=leafs)
    return binary, tuple(dict.fromkeys(paths)), proof


def surface_raw_source(path, rt, official, np, trimesh):
    """Keep the complete native source; no exact weld or unreferenced compaction."""
    import mesh_precision_diagnostic as precision
    from official_pack_geometry import canonical_oriented_triangles
    from types import SimpleNamespace
    before = rt.identity(path, 256<<20, readonly=False); local, world, records = precision.raw_glb(path)
    scene = trimesh.load(path, force='mesh', process=False)
    require(type(scene) is trimesh.Trimesh, 'Full triangular raw scene required')
    native_root, _ = official.native_mesh_sources(ROOT); scope = dict(np=np, trimesh=trimesh, Path=Path)
    scope['load_object_mesh'] = official.isolated_source_function(native_root/'lib_mhr/contact.py', 'load_object_mesh', scope)
    load = official.isolated_source_function(native_root/'learning/training/mhr_opt_refineout.py', '_load_object_vertices', scope)
    v, f = load(path); rv, rf, authority = official.load_mesh_authorities(ROOT, path, SimpleNamespace(object_vertices=v, object_faces=f))
    require(v.dtype == np.float32 and f.dtype == np.int64 and len(v) == sum(len(x) for x,_ in local) and
        len(f) == sum(len(x) for _,x in local) and np.array_equal(f, rf) and
        np.array_equal(v, rv.astype(np.float32)) and
        precision.triangle_hash(world) == precision.triangle_hash(rv[rf]) and
        rt.identity(path, 256<<20, readonly=False) == before, 'Raw/native source loss, scene projection change or untracked geometry')
    return v, f, dict(source_identity=before, raw_oriented_triangles_sha256=precision.triangle_hash(world),
        full_scene_authority=authority, raw_accessors=records, native_float32_conversion=True,
        source_vertices_preserved=True, source_faces_preserved=True, welding_performed=False,
        geometry_repaired=False)


def surface_produce(episode, code, binary, work, left, report, rt, q, build, *, authored=None):
    import numpy as np
    import trimesh
    import object_budget_endpoint as endpoint
    from world_reward.surface_budget import prepare_surface_budget, SurfaceBudgetError, surface_record
    from world_reward.surface_identity import SurfaceIdentity
    from world_reward.mesh_serialization import serialization_preflight
    from official_pack_geometry import verify_exact_dual_surfaces, canonical_oriented_triangles, nonzero_triangle_mask
    official = q.modules()[2]; official.check_runtime_packages(report['qualification']['runtime']['pins']['versions'])
    installed = surface_trimesh_sources(rt, trimesh); report['installed_trimesh_sources'] = installed
    if authored is None:
        inputs, sources, scale = endpoint.prerequisites(ROOT, episode)
        report.update(source_hashes=sources, original_grounded_scale=scale, input_sha256=inputs['video_sha256'], phase='raw_source')
        v, f, report['source_geometry'] = surface_raw_source(ROOT/f'outputs/episode_{episode:06d}/object_grounded/object.glb', rt, official, np, trimesh)
    else:
        source_glb, scale = authored
        v, f, report['source_geometry'] = surface_raw_source(source_glb, rt, official, np, trimesh)
        report.update(original_grounded_scale=scale, phase='raw_source')
    compiler = dict(status='fail', phase='source_domain'); report['compiler'] = compiler
    def simplify(sv, sf):
        q.write_obj(work/'input.obj', sv, sf)
        counts = dict(preflight_attempts=0, preflight_returns=0, qem_attempts=0, qem_returns=0)
        compiler['native_counts'] = counts
        q.run_native(binary, ['--preflight', work/'input.obj'], left, counts, 'preflight')
        q.run_native(binary, [work/'input.obj', work/'candidate.obj', work/'native_mapping.json'], left, counts, 'qem')
        u, g = q.read_obj(work/'candidate.obj'); mapping = rt.strict((work/'native_mapping.json').read_bytes())
        return u, g, mapping
    try: proposal = prepare_surface_budget(v, f, simplify=simplify, verify_mapping=q.verify_mapping)
    except SurfaceBudgetError as error: compiler.update(error.report); raise
    compiler.update(proposal.proof); left()
    canonical_v = (proposal.vertices.astype(np.float64)*scale).astype(np.float32)
    canonical_f = proposal.faces.copy()
    require(np.isfinite(canonical_v).all(), 'Original scale/F32 conversion overflow')
    for face in canonical_f: q.orientation(proposal.vertices[face], canonical_v[face])
    metric = SurfaceIdentity(canonical_v, canonical_f)
    serialized = dict(serialization_preflight(canonical_v, canonical_f))
    require(serialized['serialized_triangles_numerically_preserved_by_weld'] is True and
        serialized['welded_triangles_exactly_active'] is True, 'Default8 weld would change represented candidate')
    glb = work/SURFACE_OUTPUTS[0]; trimesh.Trimesh(canonical_v, canonical_f, process=False).export(glb); glb.chmod(0o444)
    actual_v, actual_f, authority = surface_raw_source(glb, rt, official, np, trimesh)
    require(np.array_equal(actual_v, canonical_v) and np.array_equal(actual_f, canonical_f), 'Canonical native loader changed full candidate rows')
    sys.path.insert(0, str(ROOT/'vendor/v2d_submission_kit'))
    from v2dlb.mesh_budget import budget_mesh
    require(Path(budget_mesh.__code__.co_filename) == ROOT/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py', 'Original official helper required')
    pv, pf = budget_mesh(str(glb), faces=4096, vertices=4096)
    exact = verify_exact_dual_surfaces(actual_v, actual_f, actual_v.astype(np.float64), actual_f, pv, pf)
    active = nonzero_triangle_mask(pv, pf)
    require(pv.shape == pf.shape == (4096,3) and pv.dtype == np.float64 and pf.dtype == np.int64 and
        np.all(pf[~active] == 0) and np.count_nonzero(active) == len(canonical_f), 'Exact official4096/padding only')
    used = np.unique(pf[active]); packed = SurfaceIdentity(pv[used], np.searchsorted(used, pf[active]).astype(np.int64))
    require(q.modules()[0].surface_signature(metric) == q.modules()[0].surface_signature(packed),
        'Original meaningful components/boundaries must not disappear in official packing')
    payload_v = np.concatenate((canonical_v.astype(np.float64), np.repeat(canonical_v[:1].astype(np.float64), 4096-len(canonical_v), axis=0)))
    payload_f = np.concatenate((canonical_f, np.zeros((4096-len(canonical_f),3),np.int64)))
    compiler.update(canonical_surface=surface_record(canonical_v, canonical_f), canonical_vertex_count=len(canonical_v),
        canonical_face_count=len(canonical_f), metric_scale_baked_once=scale, object_scale=1., default8_serialization=serialized,
        native_authority=authority, official_pack_fidelity=exact, official_budget_calls=1,
        native_loader_calls_including_authorities=4, canonical_glb_identity=rt.identity(glb),
        status='pass', phase='complete', orphan_vertices_dropped=False, frame_poses_changed=False,
        payload_role='native_canonical_arrays_plus_only_repeat_first_vertex_zero_faces',
        official_arrays_role='independent_original_packer_surface_audit_not_retained_payload',
        official_unreferenced_source_vertices=metric.diagnostics['unused_vertices_preserved'])
    candidate = dict(source_vertices=proposal.source_vertices, source_faces=proposal.source_faces,
        candidate_vertices=proposal.vertices, candidate_faces=proposal.faces, canonical_vertices=canonical_v, canonical_faces=canonical_f)
    for name, arrays in ((SURFACE_OUTPUTS[1], dict(vertices=payload_v, faces=payload_f, episode_index=np.array(episode, np.int64),
            object_scale=np.array(1., np.float64), grounded_scale_baked=np.array(scale, np.float64))), (SURFACE_OUTPUTS[2], candidate)):
        with (work/name).open('xb') as stream: np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        (work/name).chmod(0o444)
        with np.load(work/name, allow_pickle=False) as stored:
            require(set(stored.files) == set(arrays) and all(stored[n].dtype == a.dtype and np.array_equal(stored[n], a)
                for n,a in arrays.items()), 'Saved full arrays changed')
    build.write_json(work/SURFACE_OUTPUTS[3], proposal.mapping); compiler['mapping_identity'] = rt.identity(work/SURFACE_OUTPUTS[3])
    if authored is None: require(endpoint.prerequisites(ROOT, episode)[1:] == (sources, scale), 'Source ancestry changed')
    require(surface_trimesh_sources(rt,trimesh)==installed, 'Installed raw/native geometry source changed')
    left(); report.update(frame_poses_changed=False, metric_scale_baked_once=scale, object_scale=1.)


def surface_validate(result, episode, source, proof, inputs):
    require(result['stage'] == 'world_reward_object_budget_surface_native_v1' and result['domain'] == 'surface' and
        result['status'] == 'pass' and result['phase'] == 'complete' and result['episode_index'] == episode and
        result['source_binding'] == result['source_binding_after'] == source and result['qualification'] == proof and
        result['input_binding'] == inputs and result['input_sha256'] == inputs['video_sha256'] and
        set(result['outputs']) == set(SURFACE_OUTPUTS) and all(result[k] is True for k in
        ('source_rehashed_after','inputs_qualification_rehashed_after','runtime_rehashed_after')) and
        all(result[k] is False for k in ('gpu_used','ground_truth_used','hand_labeled_test','adoption','reconstruction_accuracy_verified',
        'competition_eligibility_verified','media_decoded','frame_poses_changed')) and result['oracle_modes'] == [] and
        0 < result['elapsed_seconds'] <= SURFACE_SECONDS, 'Complete surface native proof required')
    c = result['compiler']; require(c['schema'] == 'world_reward.surface_budget.v1' and c['status'] == 'pass' and c['phase'] == 'complete' and
        c['method'] in ('identity','qslim') and type(c['qem_calls']) is int and c['qem_calls'] == int(c['method']=='qslim') and
        c['source_arrays_frozen_before_gates'] is c['source_arrays_unchanged'] is True and c['volume_or_closure_required'] is False and
        c['embedding_certified'] is c['full_surface_fidelity_certified'] is False and c['canonical_glb_identity'] == result['outputs'][SURFACE_OUTPUTS[0]] and
        c['mapping_identity'] == result['outputs'][SURFACE_OUTPUTS[3]] and c['official_budget_calls'] == 1 and
        c['official_pack_fidelity']['raw_oriented_triangles_exact'] is True,
        'Actual surface identity/QSlim/packing scope differs')


def surface_control_produce(code, binary, work, left, report, rt, q, build):
    """One frozen NEW authored source through the same production graph."""
    import numpy as np
    import trimesh
    from world_reward.surface_pose_geometry import compact_surface, serialized_mesh, camera_roundtrip
    from official_pack_geometry import canonical_oriented_triangles
    import mesh_precision_diagnostic as precision
    config = surface_control_protocol(code,rt)
    surface_profile('surface','surface_consumer_v1')
    v=np.array(config['source']['vertices'],np.float32);f=np.array(config['source']['faces'],np.int64)
    angle=.20;A=np.eye(4);A[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]];A[:3,3]=[.03,-.02,.01];A=A.astype(np.float32).astype(np.float64)
    t=np.arange(96,dtype=np.int64);a=.10*np.sin(2*np.pi*t/95);R=np.zeros((96,3,3));R[:,0,0]=R[:,1,1]=np.cos(a);R[:,0,1]=-np.sin(a);R[:,1,0]=np.sin(a);R[:,2,2]=1
    translation=np.column_stack((.01*np.sin(2*np.pi*t/95),.005*np.sin(4*np.pi*t/95),3+.01*np.sin(np.pi*t/95)))
    poses=np.broadcast_to(np.eye(4),(96,4,4)).copy();poses[:,:3,:3]=R;poses[:,:3,3]=translation;saved=(poses@np.linalg.inv(A)).astype(np.float32)
    allarrays=(v,f,t,A,R,translation,saved); frozen=tuple(a.tobytes() for a in allarrays)
    report['control_source_array_sha256']=[hashlib.sha256(a).hexdigest()for a in frozen]
    report['all_control_arrays_frozen_before_native_calls']=True
    report['control_protocol']=SURFACE_CONTROL_PIN
    source=work/'authored_source.glb';trimesh.Trimesh(v,f,process=False).export(source);source.chmod(0o444)
    surface_produce(-1,code,binary,work,left,report,rt,q,build,authored=(source,.5))
    c=report['compiler'];require(c['method']=='identity'and c['qem_calls']==0,'Control must never simplify')
    with np.load(work/'candidate_geometry.npz',allow_pickle=False)as stored_arrays:
        cv,cf=stored_arrays['canonical_vertices'].copy(),stored_arrays['canonical_faces'].copy()
    with np.load(work/'geometry.npz',allow_pickle=False)as stored_arrays:
        pv,pf=stored_arrays['vertices'].copy(),stored_arrays['faces'].copy()
    u,g,active,top=compact_surface(pv,pf,reference_vertices=cv.astype(np.float64),reference_faces=cf)
    mesh=trimesh.Trimesh(cv,cf,process=False);scene=trimesh.Scene();scene.add_geometry(mesh,node_name='object',geom_name='object',transform=A)
    aligned=work/'authored_aligned.glb';scene.export(aligned);aligned.chmod(0o444)
    local,world,_=precision.raw_glb(aligned);loaded=trimesh.load(aligned,force='mesh',process=False)
    official=q.modules()[2];native_root,_=official.native_mesh_sources(ROOT);scope=dict(np=np,trimesh=trimesh,Path=Path)
    scope['load_object_mesh']=official.isolated_source_function(native_root/'lib_mhr/contact.py','load_object_mesh',scope)
    load=official.isolated_source_function(native_root/'learning/training/mhr_opt_refineout.py','_load_object_vertices',scope)
    nv,nf=load(aligned);projected=trimesh.transformations.fix_rigid(A,max_deviance=1e-5)
    effective=trimesh.load(aligned,force='scene',process=False).graph['object'][0]
    nv,nf,serialized=serialized_mesh(u,g,local_vertices=local[0][0].astype(np.float64),local_faces=local[0][1],raw_world_triangles=world,
        loaded_vertices=loaded.vertices,loaded_faces=loaded.faces,native_vertices=nv,native_faces=nf,raw_matrix=A,effective_matrix=effective,projected_matrix=projected)
    error=camera_roundtrip(u,g,R,translation,saved,nv,nf);negatives=[]
    def rejected(name,callback):
        try:callback()
        except ValueError:negatives.append(name)
        else:raise ValueError('Required negative accepted: '+name)
    original=canonical_oriented_triangles(u,g)
    def same(faces):
        require(np.array_equal(original,canonical_oriented_triangles(u,faces)),'Meaningful oriented surface changed')
    rejected('drop_tiny_component',lambda:same(g[:-1]));flipped=g.copy();flipped[0]=flipped[0,::-1];rejected('flip_original_triangle',lambda:same(flipped))
    bad=pf.copy();bad[len(cf)]=[0,1,2];rejected('nonzero_padding',lambda:compact_surface(pv,bad,reference_vertices=u,reference_faces=g))
    rejected('wrong_profile_or_pin',lambda:surface_profile('solid','surface_consumer_v1'))
    rejected('drop_final_frame',lambda:camera_roundtrip(u,g,R,translation,saved[:-1],nv,nf))
    changed=saved.copy();changed[-1,0,3]+=.01;rejected('alter_saved_pose',lambda:camera_roundtrip(u,g,R,translation,changed,nv,nf))
    require(negatives==config['negative_controls']and tuple(a.tobytes() for a in allarrays)==frozen and
        surface_trimesh_sources(rt,trimesh)==report['installed_trimesh_sources'], 'Control negatives/source changed')
    report.update(control=dict(status='pass',frames=96,frame_index=t.tolist(),source_arrays_unchanged=True,QEM_calls=0,
        serialized_mesh=serialized,camera_roundtrip_max_error_m=error,negative_controls_rejected=negatives,
        canonical_surface=top,body_or_contact_qualified=False,full_HDF5_route_qualified=False,whole_Parquet_qualified=False,
        accuracy_verified=False,adoption=False),frame_poses_changed=False)
    left()


def surface_control_validate(result, source, proof):
    require(result['stage']=='world_reward_surface_consumer_control_native_v1'and result['status']=='pass'and result['phase']=='complete'and
        result['source_binding']==result['source_binding_after']==source and result['qualification']==proof and
        result['control_protocol']==SURFACE_CONTROL_PIN and result['control']['status']=='pass'and result['control']['frames']==96 and
        result['control']['QEM_calls']==0 and result['compiler']['method']=='identity'and result['compiler']['qem_calls']==0 and
        result['control']['frame_index']==list(range(96)) and result['control']['negative_controls_rejected']==
        ['drop_tiny_component','flip_original_triangle','nonzero_padding','wrong_profile_or_pin','drop_final_frame','alter_saved_pose'] and
        result['all_control_arrays_frozen_before_native_calls']is True and result['control']['source_arrays_unchanged']is True and
        result['control']['camera_roundtrip_max_error_m']<=1e-5 and all(result[k] is False for k in
        ('gpu_used','ground_truth_used','adoption','reconstruction_accuracy_verified','competition_eligibility_verified','media_decoded','input_video_hashed')) and
        result['input_track']=='authored_operator_only' and result['oracle_modes']==[] and
        result['source_rehashed_after']is result['inputs_qualification_rehashed_after']is result['runtime_rehashed_after']is True and
        0<result['elapsed_seconds']<=SURFACE_SECONDS,'Whole fresh surface consumer control required')


def surface_native(episode, code, revision, work, rt, q, build, body, *, control=False):
    started = time.monotonic(); left = lambda: q.modules()[3].remaining(started, SURFACE_SECONDS)
    require(sys.platform == 'linux' and os.getuid() == 1000 and work.resolve() == work and work.stat().st_uid == 1000 and
        stat.S_IMODE(work.stat().st_mode) == 0o700 and not list(work.iterdir()) and
        {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'} and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Restricted offline CPU required')
    source = surface_source(code, revision, rt, control=control); report = dict(stage='world_reward_surface_consumer_control_native_v1'if control else 'world_reward_object_budget_surface_native_v1', status='fail',
        phase='qualification', domain='surface', episode_index=episode, producer_revision=revision, source_binding=source,
        input_track='authored_operator_only'if control else 'track_1', gpu_used=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        adoption=False, reconstruction_accuracy_verified=False, competition_eligibility_verified=False,
        input_video_hashed=not control, media_decoded=False, budget_seconds=SURFACE_SECONDS, maximum_qem_calls=0 if control else 1)
    old = {s: signal.signal(s, lambda *_: (_ for _ in ()).throw(TimeoutError('Surface inclusive deadline'))) for s in (signal.SIGALRM,signal.SIGTERM)}
    signal.alarm(SURFACE_SECONDS)
    try:
        binary, _, proof = surface_qualification(code, rt, q); report['qualification'] = proof
        require(os.environ.get('WR_CPU_IMAGE_ID') == SURFACE_IMAGE, 'Actual qualified surface image required')
        inputs=None
        if not control:
            _, _, inputs = input_binding(episode, body, build); report['input_binding'] = inputs
        report['binary_runtime'] = q.binary_runtime(binary, proof['build']['source_cpp']['sha256'], left)
        report['native_binary_metadata_calls']=1
        report['geometry_binary_calls']=0 if control else None
        if control:surface_control_produce(code,binary,work,left,report,rt,q,build)
        else:surface_produce(episode, code, binary, work, left, report, rt, q, build)
        if not control:report['outputs'] = {n:rt.identity(work/n, 256<<20) for n in SURFACE_OUTPUTS}
        require(surface_source(code,revision,rt,control=control) == source and surface_qualification(code,rt,q)[2] == proof and
            (control or input_binding(episode,body,build)[2] == inputs), 'Native source/input/runtime changed')
        report.update(status='pass', phase='complete', source_binding_after=source, source_rehashed_after=True,
            inputs_qualification_rehashed_after=True, runtime_rehashed_after=True)
    except Exception as error: report.update(status='fail', failure_type=type(error).__name__, failure_reason=str(error)[:400])
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        try: surface_report(work/'native.json',report,build,left)
        finally:
            signal.alarm(0)
            for s,h in old.items(): signal.signal(s,h)
    return 0 if report['status']=='pass' else 1


def surface_cleanup(name, revision, cid, rt):
    import subprocess
    pin = rt.identity(cid,65,readonly=False); raw = cid.read_bytes()
    require(cid.stat().st_uid == os.getuid() and re.fullmatch(b'[0-9a-f]{64}\\n?',raw), 'Exact owned CID required')
    identifier = raw.decode().strip()
    def inspect():
        result = subprocess.run(['docker','container','inspect',identifier],capture_output=True,timeout=10)
        require(rt.identity(cid,65,readonly=False)==pin,'CID changed');return result
    def absent(r):
        return r.returncode==1 and r.stdout.strip() in (b'',b'[]') and r.stderr.strip() in tuple((p+identifier).encode() for p in
            ('Error: No such object: ','error: no such object: ','Error: No such container: ','Error response from daemon: No such container: '))
    result=inspect()
    if result.returncode: require(absent(result),'Daemon failure is not CID absence');return
    rows=rt.strict(result.stdout);require(len(rows)==1 and rows[0]['Id']==identifier and rows[0]['Name']=='/'+name and rows[0]['Image']==SURFACE_IMAGE and
        rows[0]['Config']['Labels'].get('world_reward.surface_budget.owner')==revision,'Foreign container refused')
    require(subprocess.run(['docker','rm','-f',identifier],capture_output=True,timeout=15).returncode==0 and absent(inspect()),'Owned exact cleanup failed')


def surface_remove_work(work, owner, rt, *, control=False):
    require(rt.canonical(work)==work and (work.lstat().st_dev,work.lstat().st_ino)==(owner.st_dev,owner.st_ino), 'Scratch owner changed')
    allowed={*SURFACE_OUTPUTS,'native.json','input.obj','candidate.obj','native_mapping.json',*(('authored_source.glb','authored_aligned.glb')if control else ())}
    require(all(p.name in allowed and not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode) and p.lstat().st_nlink==1
        and p.lstat().st_uid==owner.st_uid for p in work.iterdir()), 'Unknown scratch refused')
    shutil.rmtree(work)


def surface_host(episode, code, revision, rt, q, build, body, *, control=False):
    started=time.monotonic(); left=lambda:q.modules()[3].remaining(started,SURFACE_HOST_SECONDS)
    require(sys.platform=='linux' and os.getuid()==0 and os.environ.get('DOCKER_HOST')=='unix://'+str(ROOT/'docker.sock'),'Private Linux CPU daemon required')
    source=surface_source(code,revision,rt,control=control);binary,paths,proof=surface_qualification(code,rt,q)
    inputpaths=();inputs=None
    if not control:_,inputpaths,inputs=input_binding(episode,body,build)
    image=rt.image(SURFACE_IMAGE)
    require(image['Id']==SURFACE_IMAGE,'Actual qualified image required')
    name='wr-object-budget-surface-'+str(episode).zfill(6)+'-'+revision
    require(not build.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],min(10,left())).strip(),'Preexisting container refused')
    out=ROOT/'results'/('object-budget-surface-control-'+revision)if control else ROOT/f'outputs/episode_{episode:06d}'/('object_budget_surface_'+revision)
    require(rt.canonical(out)==out and not out.exists(),'Fresh surface-only output required')
    out.mkdir(mode=0o755);out.chmod(0o755);work=out/'disposable';work.mkdir(mode=0o700);work.chmod(0o700);os.chown(work,1000,1000)
    owner=work.lstat();cid=out/'.container.cid';published={};report=dict(stage='world_reward_surface_consumer_control_host_v1'if control else 'world_reward_object_budget_surface_host_v1',status='fail',phase='native',
        domain='surface',episode_index=episode,producer_revision=revision,source_binding=source,qualification=proof,input_binding=inputs,
        image_identity=image,gpu_used=False,ground_truth_used=False,adoption=False,reconstruction_accuracy_verified=False,competition_eligibility_verified=False)
    old={s:signal.signal(s,lambda *_:(_ for _ in ()).throw(TimeoutError('Host inclusive deadline')))for s in(signal.SIGALRM,signal.SIGTERM)};signal.alarm(max(1,int(left())))
    try:
        mounts=[]
        for p in dict.fromkeys((code.parent,*paths,*inputpaths)):
            mounts.extend(['--mount',f'type=bind,src={p},dst={p},readonly'])
        argv=['docker','run','--name',name,'--cidfile',str(cid),'--label','world_reward.surface_budget.owner='+revision,
            '--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','4','--memory','16g',
            '--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=512m',*mounts,'--mount',f'type=bind,src={work},dst={work}',
            '--entrypoint','/usr/bin/env',SURFACE_IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp',
            'WR_ROOT='+str(ROOT),'WR_CODE='+str(code),'WR_CODE_REVISION='+revision,'WR_CPU_IMAGE_ID='+SURFACE_IMAGE,
            'CUDA_VISIBLE_DEVICES=-1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
            'python3','-I','-B',str(code/'infra/object_budget_solid.py')]
        argv+=['--domain','surface','--control','surface_consumer_v1','--native']if control else ['--episode',str(episode),'--domain','surface','--native']
        build.run(argv,min(SURFACE_SECONDS+10,left()),out/'native.log')
    except Exception as error:report['failure_type']=type(error).__name__
    finally:
        signal.alarm(30)
        try:surface_cleanup(name,revision,cid,rt);report['owned_container_removed']=True
        except Exception as error:report.update(status='fail',cleanup_failure_type=type(error).__name__)
        try:
            require(report.get('owned_container_removed') is True, 'Unknown live container forbids scratch cleanup')
            signal.alarm(max(1,int(left())))
            result=rt.pinned(work/'native.json',rt.identity(work/'native.json',2<<20),2<<20)
            publish(out/'native.json',(work/'native.json').read_bytes(),published);report['native']=result;report['native_identity']=rt.identity(out/'native.json')
            require(report.get('owned_container_removed')is True and 'failure_type'not in report,'Native exit0 and owned cleanup required')
            if control:surface_control_validate(result,source,proof)
            else:surface_validate(result,episode,source,proof,inputs)
            require(surface_source(code,revision,rt,control=control)==source and surface_qualification(code,rt,q)[2]==proof and (control or input_binding(episode,body,build)[2]==inputs) and rt.image(SURFACE_IMAGE)==image,'Host source/input/runtime changed')
            for n in (()if control else SURFACE_OUTPUTS):
                require(rt.identity(work/n,256<<20)==result['outputs'][n],'Candidate changed');publish(out/n,(work/n).read_bytes(),published)
            report.update(status='pass',phase='complete',source_binding_after=source,source_rehashed_after=True,
                inputs_qualification_rehashed_after=True,runtime_rehashed_after=True)
            if not control:report['outputs']=result['outputs']
        except Exception as error:report.update(status='fail',posthash_failure_type=type(error).__name__,failure_reason=str(error)[:400])
        try:
            require(report.get('owned_container_removed') is True, 'Unknown live container forbids scratch cleanup')
            surface_remove_work(work,owner,rt,control=control);report['owned_scratch_removed']=True
            for n,(dev,ino,pin)in published.items():require((out/n).lstat().st_dev==dev and (out/n).lstat().st_ino==ino and rt.identity(out/n,256<<20,empty=True)==pin,'Published inode replaced')
            left()
        except Exception as error:report.update(status='fail',cleanup_failure_type=type(error).__name__)
        if report['status']!='pass':
            for n,(dev,ino,pin)in published.items():
                if n=='native.json':continue
                try:
                    p=out/n;m=p.lstat();require((m.st_dev,m.st_ino)==(dev,ino)and rt.identity(p,256<<20,empty=True)==pin,'Foreign replaced output');p.unlink()
                except Exception as error:report['rollback_failure_type']=type(error).__name__
            report.pop('outputs',None)
        report['elapsed_seconds']=time.monotonic()-started
        try:
            for p in (cid,out/'native.log'):
                if p.exists():rt.identity(p,64<<20,readonly=False,empty=True);p.chmod(0o444)
            surface_report(out/'report.json',report,build,left,finalize=lambda:out.chmod(0o555))
        finally:
            signal.alarm(0)
            for s,h in old.items():signal.signal(s,h)
    return 0 if report['status']=='pass'else 1


def surface_main():
    if sys.argv[1:5]==['--domain','surface','--control','surface_consumer_v1']:
        require(sys.argv[5:]in([],['--native']),'Only explicit fresh surface control allowed')
        surface_profile(sys.argv[2],sys.argv[4])
        code,revision=Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'];require(os.environ['WR_ROOT']==str(ROOT),'Fixed root required')
        work=ROOT/'results'/('object-budget-surface-control-'+revision)/'disposable'
        runtime=surface_modules(code)
        return surface_native(-1,code,revision,work,*runtime,control=True)if sys.argv[5:]else surface_host(-1,code,revision,*runtime,control=True)
    require(len(sys.argv)in(5,6)and sys.argv[1]=='--episode'and re.fullmatch(r'(0|[1-9]|[12][0-9])',sys.argv[2])and
        sys.argv[3:5]==['--domain','surface']and sys.argv[5:]in([],['--native']),'Exact surface-only enum/episode required')
    surface_profile(sys.argv[4])
    code,revision=Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'];require(os.environ['WR_ROOT']==str(ROOT),'Fixed root required')
    runtime=surface_modules(code);episode=int(sys.argv[2]);work=ROOT/f'outputs/episode_{episode:06d}'/('object_budget_surface_'+revision)/'disposable'
    return surface_native(episode,code,revision,work,*runtime)if sys.argv[5:]else surface_host(episode,code,revision,*runtime)


def main():
    # Explicit representation seam; the historical solid route below is intact.
    if '--domain' in sys.argv:
        return surface_main()
    require(len(sys.argv) in (3, 4, 5) and sys.argv[1] == '--episode' and re.fullmatch(r'(0|[1-9]|[12][0-9])', sys.argv[2]) and
        sys.argv[3:] in ([], ['--native'], ['--query-requalification'], ['--native', '--query-requalification']),
        'Only exact episode and explicit optional native/query arguments accepted')
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--episode', type=int, choices=range(30), required=True)
    parser.add_argument('--native', action='store_true'); parser.add_argument('--query-requalification', action='store_true'); args = parser.parse_args()
    code, revision = Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    require(os.environ['WR_ROOT'] == str(ROOT), 'Fixed root required'); runtime = helpers(code)
    prefix = 'object_budget_solid_balanced_' if args.query_requalification else 'object_budget_solid_'
    options = dict(query_requalification=True) if args.query_requalification else {}
    work = ROOT/f'outputs/episode_{args.episode:06d}'/(prefix+revision)/'disposable'
    return native(args.episode, code, revision, work, *runtime, **options) if args.native else host(args.episode, code, revision, *runtime, **options)


if __name__ == '__main__': raise SystemExit(main())
