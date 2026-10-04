import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'infra'), str(REPO/'src'), str(REPO/'tests')]
import object_budget_solid as gate
import solid_chart_v2_qualify as qual
import solid_chart_v2_build as built
import certified_solid_build as build
from test_solid_chart_v2_qualify import controls, write


def source(tmp_path, monkeypatch, revision='a'*40, entry=gate.ENTRY):
    root = tmp_path/'root'; code = root/'jobs'/revision/entry/'code'
    for n in set(gate.HELPERS) | set(qual.HELPERS): write(code/n)
    write(code/'src/world_reward/__init__.py', b'')
    write(code.parent/'revision', (revision+'\n').encode()); write(code.parent/'source-sha256', ('b'*64+'\n').encode())
    for p in sorted(code.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    code.chmod(0o555)
    monkeypatch.setattr(gate, 'ROOT', root); monkeypatch.setattr(qual, 'ROOT', root)
    monkeypatch.setattr(gate, '__file__', str(code/'infra/object_budget_solid.py'))
    return root, code, revision


def evidence(tmp_path, monkeypatch):
    root, code, _ = source(tmp_path, monkeypatch)
    _, producer, revision = source(tmp_path, monkeypatch, 'c'*40, 'run_solid_chart_v2_qualify')
    rows, ledger = qual.snapshot(producer, revision, build)
    bound = ledger | dict(helpers={n: rows[n] for n in qual.HELPERS})
    manifest, full = controls(); image = 'sha256:'+'d'*64
    measured = dict(original_runtime={'qualified': True})
    common = dict(status='pass', phase='complete', source_binding=bound, source_binding_after=bound,
        source_rehashed_after=True, artifacts_rehashed_after=True, official_rehashed_after=True,
        gpu_used=False, gt_used=False, adoption=False, reconstruction_accuracy_verified=False,
        competition_eligibility_verified=False, qualified_build=measured, qualified_procedural_controls=4)
    native = common | dict(stage='solid_chart_v2_qualification_native_v1', elapsed_seconds=1., source_arrays_unchanged=True,
        image_id=image, control_manifest=manifest, controls=full,
        control_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    host = common | dict(stage='solid_chart_v2_qualification_host_v1', elapsed_seconds=2., native=native,
        owned_container_removed=True, owned_scratch_removed=True)
    out = root/'results'/('solid-chart-v2-qualify-'+revision)
    hp = write(out/'report.json', json.dumps(host, sort_keys=True).encode()); npin = write(out/'native.json', json.dumps(native, sort_keys=True).encode())
    out.chmod(0o555)
    pins = dict(schema='world_reward.solid_chart_v2_qualification_pins.v1', producer_revision=revision, report=hp, native=npin,
        build_pins=build.identity(code/qual.BUILD_PINS), source_archive_sha256='b'*64,
        **{k: ledger[k] for k in ('source_files', 'source_files_sha256', 'source_readonly_ledger_sha256')},
        qualified_controls=4, native_qem_calls=4, native_query_calls=40, adoption=False, reconstruction_accuracy_verified=False)
    write(code/gate.PINS, json.dumps(pins).encode())
    monkeypatch.setattr(qual, 'built_qualification', lambda *_: ({'image_id': image}, root/'binary', root/'query', (root/'built-dir',), measured))
    return root, code, producer, pins, out


def test_qualification_independent_full_historical_ledger_before_native(tmp_path, monkeypatch):
    root, code, producer, pins, out = evidence(tmp_path, monkeypatch)
    result = gate.qualification(code, qual, build, None, None)
    assert result[4]['source']['source_files_sha256'] == pins['source_files_sha256']
    assert out in result[3] and producer in result[3] and producer.parent/'revision' in result[3]
    assert root/'built-dir' in result[3]
    assert stat.S_IMODE(out.stat().st_mode) == 0o555


@pytest.mark.parametrize('change', ['missing', 'ledger', 'archive', 'producer', 'count', 'bool_count', 'source', 'current', 'receipt', 'scope', 'cleanup', 'partial'])
def test_qualification_rejects_unverified_or_changed_proof(tmp_path, monkeypatch, change):
    _, code, producer, pins, out = evidence(tmp_path, monkeypatch)
    if change == 'missing': (code/gate.PINS).parent.chmod(0o755); (code/gate.PINS).unlink()
    elif change in ('ledger', 'archive', 'producer', 'count', 'bool_count'):
        key = {'ledger':'source_files_sha256', 'archive':'source_archive_sha256', 'producer':'producer_revision', 'count':'native_query_calls', 'bool_count':'native_qem_calls'}[change]
        pins[key] = {'producer':'e'*40, 'count':39, 'bool_count':True}.get(change, 'e'*64); write(code/gate.PINS, json.dumps(pins).encode())
    elif change in ('source', 'current'): write((producer if change == 'source' else code)/'infra/oriented_solid_compiler.py', b'mutated')
    elif change == 'receipt': write(out/'native.json', b'tampered')
    else:
        h = json.loads((out/'report.json').read_bytes())
        if change == 'scope': h['gpu_used'] = True
        elif change == 'cleanup': h['owned_container_removed'] = False
        else: h['native']['controls']['controls'].pop()
        pins['report'] = write(out/'report.json', json.dumps(h).encode()); write(code/gate.PINS, json.dumps(pins).encode())
    with pytest.raises((ValueError, FileNotFoundError)): gate.qualification(code, qual, build, None, None)


def inputs(root, episode=8, total=3):
    base = root/f'outputs/episode_{episode:06d}'
    video = root/'data/track_1/rgb/selected.mp4'; maskroot = base/'automatic_masks/masks/0'
    for path in (root/'results/input-manifest.json', root/'data/track_1/meta/episodes.jsonl', video,
        *(base/n for n in ('automatic_masks/report.json', 'automatic_masks/prompts.json', 'object_grounded/report.json',
        'object_grounded/object.glb', 'object_grounded/transform.json', 'object_grounded/intrinsics.json', 'scale_smoke/report.json',
        'body_smoke/report.json', 'depth_smoke/report.json'))): write(path, b'opaque data never decoded')
    for n in range(total): write(maskroot/f'{n:06d}.png', b'opaque mask never decoded')
    value = dict(video=video, video_sha256=build.identity(video)['sha256'], total_frames=total, dataset_revision='f'*40, human_masks=maskroot)
    return SimpleNamespace(_validate_inputs=lambda *a, **k: value), value


def test_input_binder_hashes_only_exact_public_full_t_ancestry(tmp_path, monkeypatch):
    root, _, _ = source(tmp_path, monkeypatch); body, original = inputs(root)
    _, paths, proof = gate.input_binding(8, body, build)
    assert original['video'] in paths and original['human_masks'] in paths
    assert proof['total_frames'] == 3 and proof['input_video_hashed'] is True and proof['media_decoded'] is False
    assert len(proof['files']) == 15 and not any('eval_private' in p or 'full.npz' in p for p in proof['files'])
    (original['human_masks']/'000001.png').unlink()
    with pytest.raises(ValueError): gate.input_binding(8, body, build)


def native_result(episode, source, proof, inputs, outputs):
    c = dict(stage='oriented_solid_compiler_v2', conditioning_version=2, status='pass', phase='complete', native_attempts=1,
        native_returned=True, native_returncode=0, source_arrays_unchanged=True, artifacts_before={}, artifacts_after={},
        stages=dict.fromkeys(gate.STAGES, {}), metric_scale_baked_once=.375, output_vertices=4096, output_faces=4096,
        glb_identity=outputs[gate.OUTPUTS[0]], geometry_repaired=False, components_deleted=False, orientation_changed=False,
        cost_backend_changed=False, ground_truth_used=False, adoption=False, reconstruction_accuracy_verified=False,
        frame_poses_changed=False, metric_scale_accuracy_verified=False)
    def query():
        certificate = dict(status='pass',schema='world_reward.certified_solid_query.v1',cgal_version='6.0.1',component_count=1,vertices=4,faces=4,
            inside=[[False]], components=[dict(original_component_id=0,original_vertices=4,original_faces=4,exact_volume_sign=1,witness_original_vertex=0)],
            **dict.fromkeys(('closed_oriented_vertex_manifold_verified','all_original_faces_retained','all_original_vertices_referenced',
                'exact_nondegenerate_triangles_verified','component_self_intersections_absent','inter_component_surface_contacts_absent'), True),
            **dict.fromkeys(('geometry_repaired','orientation_changed','qem_executed','forest_adjudicated','reconstruction_accuracy_verified'),False))
        return dict(query_attempted=True,query_returned=True,query_returncode=0,native_certificate=certificate,
            topology=dict(vertices=4,active_vertices=4,faces=4,closed_oriented_vertex_manifold=True,components=[{'euler':2}]),
            forest={'component_keys':['source0'], 'signs':[1], 'inside':[[False]], 'parents':[-1], 'depths':[0]}, candidate_to_source_components=[0])
    c['stages'] = {s:query() for s in gate.STAGES}
    c['stages']['physical_source'].update(conditioning={'roundtrip_numerically_exact':True}, float32_certificate=query(),
        float32_orientation={'exact_positive_normal_dot':True}, conditioning_header={'bytes':1,'sha256':'a'*64})
    c['metric_source_certificate'] = query()
    for s in gate.STAGES-{'physical_source'}:
        c['stages'][s]['fidelity'] = dict(scale_or_pose_fitted=False,sampled_bidirectional_chamfer_diagonal_ratio=0.,net_volume_relative_error=0.,
            birthface_matched_shells=[dict(source_component=0,euler=2,volume_sign=1,relative_volume_error=0.)],candidate_topology={'components':[{'euler':2}]})
    c['native_mapping'] = {'conditioning':{'chart_version':2,'source_roundtrip_numerically_exact':True},'serialization':{'committed_collapses':1}}
    return dict(stage='world_reward_object_budget_solid_native_v1', status='pass', phase='complete', episode_index=episode,
        source_binding=source, source_binding_after=source, qualification=proof, input_binding=inputs, outputs=outputs, compiler=c,
        source_hashes={'video': inputs['video_sha256']}, input_sha256=inputs['video_sha256'], input_track='track_1',
        original_grounded_scale=.375, object_scale=1., metric_scale_baked_once=.375, elapsed_seconds=1.,
        input_video_hashed=True, media_decoded=False, source_rehashed_after=True, inputs_qualification_rehashed_after=True, runtime_rehashed_after=True,
        gpu_used=False, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], adoption=False, reconstruction_accuracy_verified=False,
        competition_eligibility_verified=False, frame_poses_changed=False, budget_seconds=1800, qem_seconds=1200, query_seconds=180,
        maximum_qem_calls=1, maximum_query_calls=8, native_query_calls=8)


@pytest.mark.parametrize('change', ['none', 'gt', 'eligibility', 'count', 'dropped', 'repair', 'scale', 'geometry', 'source', 'elapsed','query','querymissing','queryextra','f32','emptyforest','missingfidelity','forest','fidelity'])
def test_native_complete_six_stage_no_repair_no_adoption_gate(change):
    outputs = dict.fromkeys(gate.OUTPUTS, {'bytes':1, 'sha256':'a'*64}); evidence = {'video_sha256':'b'*64}
    r = native_result(8, {}, {}, evidence, outputs)
    if change == 'gt': r['ground_truth_used'] = True
    elif change == 'eligibility': r['competition_eligibility_verified'] = True
    elif change == 'count': r['maximum_qem_calls'] = True
    elif change == 'dropped': r['compiler']['stages'].pop('physical_source')
    elif change == 'repair': r['compiler']['components_deleted'] = True
    elif change == 'scale': r['compiler']['metric_scale_baked_once'] = .2
    elif change == 'geometry': r['compiler']['glb_identity'] = {'bytes':1,'sha256':'e'*64}
    elif change == 'source': r['source_binding_after'] = {'changed':True}
    elif change == 'elapsed': r['elapsed_seconds'] = 1801
    elif change == 'query': r['compiler']['metric_source_certificate']['query_returncode'] = 1
    elif change == 'querymissing': r['compiler'].pop('metric_source_certificate')
    elif change == 'queryextra': r['compiler']['extra'] = r['compiler']['metric_source_certificate']
    elif change == 'f32': r['compiler']['stages']['physical_source']['float32_orientation']['exact_positive_normal_dot'] = False
    elif change == 'emptyforest': r['compiler']['stages']['native_candidate']['forest']['component_keys'] = []
    elif change == 'missingfidelity': r['compiler']['stages']['native_candidate'].pop('fidelity')
    elif change == 'forest': r['compiler']['stages']['native_candidate']['candidate_to_source_components'] = [1]
    elif change == 'fidelity': r['compiler']['stages']['native_candidate']['fidelity']['sampled_bidirectional_chamfer_diagonal_ratio'] = .02
    if change == 'none': gate.validate_native(r, 8, {}, {}, evidence)
    else:
        with pytest.raises(ValueError): gate.validate_native(r, 8, {}, {}, evidence)


def test_fidelity_scipy_vertex_order_not_certified_first_face_order():
    r = native_result(8,{}, {}, {'video_sha256':'a'*64},dict.fromkeys(gate.OUTPUTS,{'bytes':1,'sha256':'a'*64})); c=r['compiler']
    def extend(value):
        if isinstance(value,dict):
            if 'query_attempted' in value:
                value['native_certificate'].update(component_count=2,vertices=8,faces=8,inside=[[False,True],[False,False]],
                    components=[dict(original_component_id=0,original_vertices=4,original_faces=4,exact_volume_sign=-1,witness_original_vertex=4),
                        dict(original_component_id=1,original_vertices=4,original_faces=4,exact_volume_sign=1,witness_original_vertex=0)])
                value['forest'] = dict(component_keys=['inner-first-face','outer-second-face'], signs=[-1,1],inside=[[False,True],[False,False]],parents=[1,-1],depths=[1,0])
                value['topology'].update(vertices=8,active_vertices=8,faces=8,components=[{'euler':2},{'euler':2}])
                value['candidate_to_source_components']=[0,1]
                if 'fidelity' in value:
                    value['fidelity'].update(birthface_matched_shells=[dict(source_component=0,euler=2,volume_sign=1,relative_volume_error=0.),
                        dict(source_component=1,euler=2,volume_sign=-1,relative_volume_error=0.)],candidate_topology={'components':[{'euler':2},{'euler':2}]})
            for child in value.values():extend(child)
        elif isinstance(value,list):
            for child in value:extend(child)
    extend(c);gate.validate_compiler_proof(c)


def test_actual_core_components_returns_tuple_and_json_equivalent_keys():
    import certified_solid_source as certificate
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float64)
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
    _,keys,_=certificate.components(vertices,faces)
    assert type(keys) is tuple and keys == ('source-loaded-first-face-0',)
    c=native_result(8,{}, {}, {'video_sha256':'a'*64},dict.fromkeys(gate.OUTPUTS,{'bytes':1,'sha256':'a'*64}))['compiler']
    c['stages']['physical_source']['forest']['component_keys']=keys
    gate.successful_queries(c,8)
    restored=build.strict_json(json.dumps(c,sort_keys=True).encode())
    assert type(restored['stages']['physical_source']['forest']['component_keys']) is list
    gate.successful_queries(restored,8)


@pytest.mark.parametrize('keys',[{'source0'},'source0',('source0','source0'),[1],[''],np.array(['source0'])])
def test_component_keys_only_typed_unique_tuple_or_list(keys):
    c=native_result(8,{}, {}, {'video_sha256':'a'*64},dict.fromkeys(gate.OUTPUTS,{'bytes':1,'sha256':'a'*64}))['compiler']
    c['stages']['physical_source']['forest']['component_keys']=keys
    with pytest.raises(ValueError):gate.successful_queries(c,8)


def host_fixture(tmp_path, monkeypatch):
    root, code, revision = source(tmp_path, monkeypatch); body, _ = inputs(root)
    image = 'sha256:'+'d'*64; proof = {'build':{'original_runtime':{}}}
    old = root/'jobs'/('c'*40)/'run_solid_chart_v2_qualify/code'; paths = (root/'qualified-out', old, old.parent/'revision', old.parent/'source-sha256')
    monkeypatch.setattr(gate, 'qualification', lambda *_: ({'image_id':image}, root/'qem', root/'query', paths, proof))
    monkeypatch.setattr(built, 'qualified', lambda *_: ({}, (root/'cgal-dir', root/'cache-dir'), {}))
    monkeypatch.setattr(built, 'image_identity', lambda *_: {'child_id':image})
    monkeypatch.setattr(qual, 'OFFICIAL_PIN', write(root/qual.OFFICIAL, mode=0o644))
    monkeypatch.setattr(gate.sys, 'platform', 'linux'); monkeypatch.setattr(gate.os, 'getuid', lambda:0)
    monkeypatch.setattr(gate.os, 'chown', lambda *_:None); monkeypatch.setenv('DOCKER_HOST', 'unix://'+str(root/'docker.sock'))
    calls = []; cleanup = []; monkeypatch.setattr(build, 'cleanup_container', lambda *a:cleanup.append(a))
    def run(argv, seconds, log=None):
        if argv[1] == 'ps': return b''
        calls.append((argv, seconds)); work = root/'outputs/episode_000008'/('object_budget_solid_'+revision)/'disposable'
        outputs = {n:write(work/n, (n+' exact bytes').encode()) for n in gate.OUTPUTS}
        result = native_result(8, gate.binding(code, revision, qual, build), proof, gate.input_binding(8, body, build)[2], outputs)
        write(work/'native.json', json.dumps(result, sort_keys=True).encode()); write(log, b'owned log')
    monkeypatch.setattr(build, 'run', run)
    return root, code, revision, body, calls, cleanup


def test_cpu_single_native_public_only_narrow_mounts_and_sealed_outputs(tmp_path, monkeypatch):
    root, code, revision, body, calls, cleanup = host_fixture(tmp_path, monkeypatch)
    assert gate.host(8, code, revision, qual, build, None, built, body) == 0
    out = root/'outputs/episode_000008'/('object_budget_solid_'+revision); report = json.loads((out/'report.json').read_bytes())
    assert report['status'] == 'pass' and report['owned_container_removed'] and report['owned_scratch_removed']
    assert not (out/'disposable').exists() and report['gpu_used'] is report['adoption'] is report['reconstruction_accuracy_verified'] is False
    assert stat.S_IMODE(out.stat().st_mode) == 0o555
    for n in gate.OUTPUTS: assert build.identity(out/n, readonly=True) == report['native']['outputs'][n]
    argv, seconds = calls[0]; assert len(calls) == 1 and seconds <= 1810
    assert '--gpus' not in argv and argv[argv.index('--network')+1] == 'none' and argv[argv.index('--user')+1] == '1000:1000'
    mounts = [argv[i+1] for i,v in enumerate(argv) if v == '--mount']
    assert any('selected.mp4' in m and m.endswith(',readonly') for m in mounts)
    assert any('automatic_masks/masks/0' in m and m.endswith(',readonly') for m in mounts)
    assert not any('eval_private' in m or f'src={root},' in m or 'body_smoke/prediction' in m for m in mounts)
    assert sum(not m.endswith(',readonly') for m in mounts) == 1 and cleanup
    with pytest.raises(ValueError, match='Fresh'): gate.host(8, code, revision, qual, build, None, built, body)


def test_immutable_original_failed_namespace_preserved_with_fresh_revision(tmp_path,monkeypatch):
    root,code,revision,body,_,_=host_fixture(tmp_path,monkeypatch)
    old=root/'outputs/episode_000008/object_budget_solid';write(old/'report.json',b'original sealed FAIL receipt');old.chmod(0o555)
    before=build.identity(old/'report.json',readonly=True);inode=(old/'report.json').stat().st_ino
    assert gate.host(8,code,revision,qual,build,None,built,body)==0
    assert build.identity(old/'report.json',readonly=True)==before and (old/'report.json').stat().st_ino==inode and stat.S_IMODE(old.stat().st_mode)==0o555
    current=old.parent/('object_budget_solid_'+revision)
    assert json.loads((current/'report.json').read_bytes())['status']=='pass'
    with pytest.raises(ValueError,match='Fresh'):gate.host(8,code,revision,qual,build,None,built,body)
    assert build.identity(old/'report.json',readonly=True)==before


@pytest.mark.parametrize('change', ['exit', 'missing', 'cleanup', 'source', 'scope', 'extra', 'replaced', 'deadline','badjson','publishfsync','term'])
def test_host_failures_never_publish_an_accepted_proposal(tmp_path, monkeypatch, change):
    root, code, revision, body, _, _ = host_fixture(tmp_path, monkeypatch); run = build.run
    out = root/'outputs/episode_000008'/('object_budget_solid_'+revision)
    if change == 'cleanup': monkeypatch.setattr(build, 'cleanup_container', lambda *_:(_ for _ in ()).throw(ValueError('cleanup refused')))
    elif change == 'publishfsync':
        fsync = gate.os.fsync
        def fail(descriptor):
            if any((out/n).exists() for n in gate.OUTPUTS): raise OSError('owned publication fsync failed')
            fsync(descriptor)
        monkeypatch.setattr(gate.os,'fsync',fail)
    elif change == 'replaced':
        rmtree = gate.shutil.rmtree
        def replace(path):
            (out/gate.OUTPUTS[0]).unlink(); write(out/gate.OUTPUTS[0], b'foreign replacement preserved'); rmtree(path)
        monkeypatch.setattr(gate.shutil, 'rmtree', replace)
    elif change == 'deadline':
        clock = [0.]; monkeypatch.setattr(gate.time, 'monotonic', lambda:clock[0]); rmtree = gate.shutil.rmtree
        def expired(path): rmtree(path); clock[0] = 1901.
        monkeypatch.setattr(gate.shutil, 'rmtree', expired)
    else:
        def altered(argv, seconds, log=None):
            if argv[1] == 'ps': return b''
            if change == 'missing': return b''
            run(argv, seconds, log)
            if change == 'exit': raise ValueError('exit nonzero')
            if change == 'term': gate.signal.getsignal(gate.signal.SIGTERM)(gate.signal.SIGTERM,None)
            if change == 'source': write(code.parent/'revision', ('e'*40+'\n').encode())
            if change == 'extra': write(out/'disposable/unknown', b'foreign content retained')
            if change == 'badjson': write(out/'disposable/native.json',b'{invalid json')
            if change == 'scope':
                p = out/'disposable/native.json'; r = json.loads(p.read_bytes()); r['ground_truth_used'] = True; write(p, json.dumps(r).encode())
        monkeypatch.setattr(build, 'run', altered)
    assert gate.host(8, code, revision, qual, build, None, built, body) == 1
    report = json.loads((out/'report.json').read_bytes()); assert report['status'] == 'fail' and 'outputs' not in report
    if change == 'replaced': assert (out/gate.OUTPUTS[0]).read_bytes() == b'foreign replacement preserved'
    else: assert not any((out/n).exists() for n in gate.OUTPUTS)
    if change == 'extra': assert (out/'disposable/unknown').read_bytes() == b'foreign content retained'
    if change == 'badjson': assert report['owned_container_removed'] is True
    if change == 'term': assert report['owned_container_removed'] is True and report['failure_type'] == 'TimeoutError'


def test_produce_one_whole_unprocessed_source_scale_once_and_exact_five_keys(tmp_path, monkeypatch):
    import object_budget_endpoint as endpoint
    import oriented_solid_compiler as compiler
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    # Fake callbacks only: no mesh/native invocation and no challenge arrays.
    inputs = {'video_sha256':'a'*64}; sources = {'video':'a'*64}; scale = .375; calls = []
    source = (np.zeros((4,3), np.float64), np.zeros((4,3), np.int64)); metric = (np.ones((8,3),np.float32), np.zeros((8,3),np.int64))
    monkeypatch.setattr(np, '__version__', '1.26.test')
    monkeypatch.setattr(endpoint, 'prerequisites', lambda *a:(inputs,sources,scale))
    monkeypatch.setattr(compiler, 'load_source', lambda p:(source,{'unprocessed':True}))
    def compile(mesh, binary, query, sha, work, official, **kwargs):
        calls.append((mesh, kwargs)); pin = write(work/'candidate.glb', b'certified exact physical candidate')
        return metric, {'glb_identity':pin}
    monkeypatch.setattr(compiler, 'compile_solid', compile)
    report = {}; paths = gate.produce(8, tmp_path/'b', tmp_path/'q', 'a'*64, tmp_path, tmp_path/'official', lambda:100., report, build)
    assert len(calls) == 1 and calls[0][0] is source and calls[0][1]['metric_scale'] == scale and calls[0][1]['conditioning_version'] == 2
    assert paths[0].read_bytes() == b'certified exact physical candidate' and report['metric_scale_baked_once'] == scale
    with np.load(paths[1], allow_pickle=False) as saved:
        assert set(saved.files) == {'vertices','faces','episode_index','object_scale','grounded_scale_baked'}
        assert np.array_equal(saved['vertices'], metric[0]) and saved['object_scale'].item() == 1. and saved['grounded_scale_baked'].item() == scale


def test_produce_preserves_single_native_failure_report_without_rescue(tmp_path, monkeypatch):
    import object_budget_endpoint as endpoint
    import oriented_solid_compiler as compiler
    monkeypatch.setattr(np, '__version__', '1.26.test')
    monkeypatch.setattr(endpoint, 'prerequisites', lambda *a:({'video_sha256':'a'*64},{'video':'a'*64},.375))
    monkeypatch.setattr(compiler, 'load_source', lambda *a:((np.zeros((4,3)),np.zeros((4,3),np.int64)),{'all_faces':True}))
    partial = {'status':'fail','phase':'source','native_attempts':0,'failure_scope':'geometry_or_contract'}; calls = []
    def failed(*a, **k): calls.append(1); raise compiler.SolidCompilerError(partial)
    monkeypatch.setattr(compiler, 'compile_solid', failed); report = {}
    with pytest.raises(compiler.SolidCompilerError):
        gate.produce(8,tmp_path/'b',tmp_path/'q','a'*64,tmp_path,tmp_path/'official',lambda:100.,report,build)
    assert calls == [1] and report['compiler'] is partial and not any((tmp_path/n).exists() for n in gate.OUTPUTS)


@pytest.mark.parametrize('change', ['none','native_failure','input_post','scope_guard','native_tuple'])
def test_native_cpu_inclusive_seal_and_partial_failure(tmp_path, monkeypatch, change):
    import object_budget_conditioned as cache
    root, code, revision = source(tmp_path, monkeypatch); body, _ = inputs(root)
    work = root/'outputs/episode_000008'/('object_budget_solid_'+revision)/'disposable'; work.mkdir(parents=True,mode=0o700)
    original_stat = Path.stat
    def fake_stat(p, *a, **k):
        s = original_stat(p,*a,**k)
        return SimpleNamespace(st_uid=1000,st_mode=s.st_mode) if p == work else s
    monkeypatch.setattr(Path,'stat',fake_stat); monkeypatch.setattr(gate.os,'getuid',lambda:1000); monkeypatch.setattr(gate.sys,'platform','linux')
    monkeypatch.setenv('WR_NATIVE_NETWORK','none'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    image = 'sha256:'+'d'*64; monkeypatch.setenv('WR_CPU_IMAGE_ID', image)
    proof = {'build':{'original_runtime':{'same':True}}}
    monkeypatch.setattr(gate,'qualification',lambda *_:({'image_id':image},root/'qem',root/'query',(),proof))
    monkeypatch.setattr(qual,'OFFICIAL_PIN',write(root/qual.OFFICIAL,mode=0o644))
    monkeypatch.setattr(cache,'cache_qualification',lambda *_:({},{})); monkeypatch.setattr(cache,'runtime_identity',lambda *_:{'same':True})
    certificate = SimpleNamespace(qualification=lambda *_:({'native_source':{'sha256':'a'*64}},{},{})); calls=[]
    def produce(*args):
        report = args[-2]; calls.append(1)
        if change == 'native_failure': report['compiler'] = {'status':'fail','native_attempts':0}; raise ValueError('closed source failure')
        paths = tuple(work/n for n in gate.OUTPUTS)
        for p in paths: write(p,b'opaque candidate')
        report['compiler'] = native_result(8,{}, {}, {'video_sha256':'a'*64}, {n:build.identity(work/n) for n in gate.OUTPUTS})['compiler']
        if change == 'native_tuple':
            def native_keys(value):
                if isinstance(value,dict):
                    if 'forest' in value:value['forest']['component_keys']=tuple(value['forest']['component_keys'])
                    for child in value.values():native_keys(child)
                elif isinstance(value,list):
                    for child in value:native_keys(child)
            native_keys(report['compiler'])
        if change == 'input_post': write(root/'outputs/episode_000008/object_grounded/object.glb',b'changed')
        return paths
    monkeypatch.setattr(gate,'produce',produce)
    if change == 'scope_guard': monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    if change == 'scope_guard':
        with pytest.raises(ValueError): gate.native(8,code,revision,work,qual,build,certificate,built,body)
        assert calls == [] and not (work/'native.json').exists()
    else:
        assert gate.native(8,code,revision,work,qual,build,certificate,built,body) == (0 if change in ('none','native_tuple') else 1)
        report = json.loads((work/'native.json').read_bytes())
        assert report['maximum_qem_calls'] == 1 and report['maximum_query_calls'] == 8 and report['source_rehashed_after'] is True
        assert report['gpu_used'] is report['adoption'] is False and len(calls) == 1 and stat.S_IMODE((work/'native.json').stat().st_mode) == 0o444
        if change == 'native_failure': assert report['compiler']['native_attempts'] == 0


def test_bare_host_import_is_stdlib_only_and_future_missing_pins_block(tmp_path):
    code = f"import sys;sys.path[:0]=[{str(REPO/'infra')!r},{str(REPO/'src')!r}];import object_budget_solid as g;g.helpers(__import__('pathlib').Path({str(REPO)!r}));assert 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', code], capture_output=True, env={'PATH':os.defpath})
    assert result.returncode == 0, result.stderr.decode()
    assert build.identity  # Pins are never synthesized by the producer.


@pytest.mark.parametrize('args',[[],['--episode','08'],['--episode','30'],['--episode','8','--episode','8'],['--native','--episode','8'],['--episode','8','--gt']])
def test_direct_driver_invalid_args_stop_before_runtime(tmp_path,monkeypatch,args):
    monkeypatch.setattr(gate.sys,'argv',['driver',*args])
    monkeypatch.setattr(gate,'helpers',lambda *_:(_ for _ in ()).throw(AssertionError('no runtime before args')))
    with pytest.raises(ValueError): gate.main()


def test_main_native_uses_same_revision_namespace_without_path_override(tmp_path,monkeypatch):
    revision='a'*40;code=tmp_path/'jobs'/revision/gate.ENTRY/'code';captured=[]
    monkeypatch.setattr(gate,'ROOT',tmp_path);monkeypatch.setattr(gate.sys,'argv',['driver','--episode','29','--native'])
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',revision)
    monkeypatch.setattr(gate,'helpers',lambda *_:())
    monkeypatch.setattr(gate,'native',lambda *a:captured.append(a) or 0)
    assert gate.main()==0 and captured[0][3]==tmp_path/'outputs/episode_000029'/('object_budget_solid_'+revision)/'disposable'


def test_wrapper_syntax_strict_scope_and_runtime_closure():
    wrapper = REPO/'infra/run_object_budget_solid.sh'
    assert subprocess.run(['bash','-n',str(wrapper)], capture_output=True).returncode == 0
    text = wrapper.read_text(); assert '--gpus' not in text and 'run_object_budget_solid/code' in text and '1960s' in text
    from azure_job import runtime_bundle_paths
    files = {p.relative_to(REPO).as_posix():p.read_bytes() for folder in ('infra','src','configs') for p in (REPO/folder).rglob('*') if p.is_file()}
    files['pyproject.toml'] = (REPO/'pyproject.toml').read_bytes()
    paths = runtime_bundle_paths(files, wrapper.relative_to(REPO).as_posix())
    assert {'infra/object_budget_solid.py','infra/oriented_solid_compiler.py','infra/solid_chart_v2_qualify.py','src/world_reward/mesh_conditioning_v2.py'} <= set(paths)
    for args in ([], ['--episode','30'], ['--episode','08'], ['--episode','8','--episode','8'], ['--native'], ['--episode','8','--gt']):
        assert subprocess.run(['bash',str(wrapper),*args], capture_output=True, env={'PATH':os.defpath}).returncode == 2


def test_owned_publication_mode_explicit_under_restrictive_umask(tmp_path):
    old=os.umask(0o077)
    try:
        path=tmp_path/'geometry.npz';owned={};gate.publish(path,b'opaque bytes',owned)
    finally:os.umask(old)
    assert stat.S_IMODE(path.stat().st_mode)==0o444 and build.identity(path,readonly=True)==owned[path.name][2]
