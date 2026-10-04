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


def main():
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
