"""Inert reader of an independently pinned CPU whole-surface proposal.

No compiler, QEM, CGAL, native loader, model, image inspection or official packer
is executed. Original failed qualification hosts remain failed. Represented
surface preservation is not an embedding, Hausdorff or accuracy certificate.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import re
import stat

import numpy as np
from solid_geometry_loader import identity as _solid_identity, strict_json, require, _hex, _pin, _fields
from world_reward.artifact_paths import episode_relative, output_prefix, pin_path as artifact_pin_path

CODE = Path(__file__).resolve().parent.parent
SCHEMA = 'world_reward.surface_mesh_pins.v1'
PROCESSOR = 'infra/object_budget_solid.py'
PHASE1 = 'configs/surface_identity_qualification_pins.json'
QUALIFICATION = 'configs/surface_qslim_qualification_pins.json'
BUILD = 'configs/surface_qslim_build_pins.json'
IMAGE = 'sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0'
SOURCE_HELPERS = ('infra/surface_geometry_loader.py', 'infra/solid_geometry_loader.py',
    'src/world_reward/surface_pose_geometry.py', 'src/world_reward/surface_identity.py',
    'src/world_reward/exact_triangle_predicates.py', 'src/world_reward/raw_shape_proposal.py',
    'src/world_reward/shared_scene.py', 'src/world_reward/contracts.py', 'src/world_reward/submission.py',
    'infra/official_pack_geometry.py', 'infra/mesh_link_gate.py', 'infra/mesh_endpoint_gate.py',
    'infra/mesh_precision_diagnostic.py', 'infra/object_budget_endpoint.py',
    'infra/body_smoke.py', 'src/world_reward/data.py')
OUTPUTS = ('object_fixed_canonical.glb', 'geometry.npz', 'candidate_geometry.npz', 'mapping.json')
MAPPING_MAX_BYTES = 256 << 20  # Same bounded ledger capacity as its CPU producer.


def source_helpers():
    """Bind namespace handling for new experiments; legacy pin schema stays exact."""
    return SOURCE_HELPERS + (() if output_prefix() == 'outputs' else
                             ('src/world_reward/artifact_paths.py',))


def _mapping_role(path):
    path=Path(path)
    return (path.name=='mapping.json' and path.parent.parent.parent.name=='outputs'
        and re.fullmatch(r'episode_0000(?:0[0-9]|1[0-9]|2[0-9])',path.parent.parent.name) is not None
        and re.fullmatch(r'object_budget_surface_[0-9a-f]{40}',path.parent.name) is not None)


def identity(path, *, readonly=True):
    """Original 32MiB identities, except the complete revision-bound surface ledger."""
    path=Path(path)
    if not _mapping_role(path):return _solid_identity(path,readonly=readonly)
    require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink() for p in(path,*path.parents)),
        'Canonical original surface mapping required')
    before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and 0<before.st_size<=MAPPING_MAX_BYTES
        and (not readonly or not before.st_mode&0o222),'Bounded single-link original surface mapping required')
    digest=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    after=path.lstat()
    require(all(getattr(before,k)==getattr(after,k)for k in
        ('st_dev','st_ino','st_size','st_mode','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns')),
        'Original surface mapping changed while hashing')
    return dict(bytes=before.st_size,sha256=digest.hexdigest())


def recheck(ledger):
    """Byte/stat rehash; historical writable metadata is never chmod'ed here."""
    require(all(_source_identity(path) == pin for path, pin in ledger.items()),
            'Original immutable inputs or source changed during surface consumption')


def _source_identity(path):
    # Complete Git closures contain legitimate empty __init__.py files.
    path=Path(path)
    if path.stat().st_size: return identity(path,readonly=False)
    a=path.lstat()
    require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents))
        and stat.S_ISREG(a.st_mode) and a.st_nlink==1,'Canonical original empty source required')
    require(path.read_bytes()==b'' and all(getattr(a,k)==getattr(path.lstat(),k) for k in
        ('st_dev','st_ino','st_size','st_mode','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns')),'Empty source changed')
    return dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest())


def paths(episode, qualification_revision, producer_revision, identity_revision=None):
    require(type(episode) is int and 0 <= episode < 30, 'Exact Track1 episode required')
    for revision in (qualification_revision, producer_revision, identity_revision): _hex(revision, 40)
    base = episode_relative(episode)
    proposal = base + '/object_budget_surface_' + producer_revision
    q = 'results/surface-qslim-qualify-' + qualification_revision
    return dict(report=proposal+'/report.json', native=proposal+'/native.json',
        geometry=proposal+'/geometry.npz', glb=proposal+'/object_fixed_canonical.glb',
        candidate=proposal+'/candidate_geometry.npz', mapping=proposal+'/mapping.json',
        object=base+'/object_grounded/report.json', source_glb=base+'/object_grounded/object.glb',
        transform=base+'/object_grounded/transform.json', intrinsics=base+'/object_grounded/intrinsics.json',
        alignment=base+'/scale_smoke/report.json',
        phase1_native='results/surface-identity-qualify-'+identity_revision+'/native.json',
        qualification_native=q+'/native.json', qualification_host=q+'/report.json',
        qualification_audit='results/surface-qslim-independent-v2/report.json')


def _module(name, relative):
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == CODE/relative, 'Actual pinned pure reader source required')
    return module


def _snapshot(root, revision, entry, bound):
    """Hash the complete original code without importing/executing any of it."""
    code = root/'jobs'/revision/entry/'code'
    require(code.is_absolute() and code.resolve() == code and not code.is_symlink()
        and {p.name for p in code.parent.iterdir()} == {'code','revision','source-sha256'}, 'Exact original snapshot required')
    entries, ledger = {}, {}
    for path in (code, *sorted(code.rglob('*'))):
        info = path.lstat()
        require(path.resolve() == path and not path.is_symlink() and not info.st_mode & 0o222,
                'Full original source must be canonical and readonly')
        if stat.S_ISDIR(info.st_mode): entries[str(path.relative_to(code))] = {'directory':True}
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size <= 2_000_000,
                    'Bounded complete original source required')
            before = info
            raw = path.read_bytes(); after = path.lstat()
            require(all(getattr(before,k)==getattr(after,k) for k in
                ('st_dev','st_ino','st_size','st_mode','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns')), 'Original source changed while read')
            pin = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            entries[str(path.relative_to(code))] = pin; ledger[path] = pin
    markers = {n:identity(code.parent/n) for n in ('revision','source-sha256')}
    require((code.parent/'revision').read_bytes() == (revision+'\n').encode(), 'Original revision differs')
    _hex((code.parent/'source-sha256').read_text().strip())
    require(bound['producer_revision'] == revision and bound['markers'] == markers
        and bound['entries'] == len(entries)
        and bound['closure_sha256'] == hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
        and all(entries[n] == pin for n,pin in bound['helpers'].items()), 'Full original producer closure differs')
    ledger.update({code.parent/n:pin for n,pin in markers.items()})
    return ledger


def _configs():
    ids = {n:identity(CODE/n) for n in (PHASE1,QUALIFICATION,BUILD)}
    first,q,b = (strict_json((CODE/n).read_bytes()) for n in (PHASE1,QUALIFICATION,BUILD))
    _fields(first,dict(schema='world_reward.surface_identity_qualification_pins.v1', historical_host_status='fail',
        native_loader_calls=6,official_budget_calls=3,QEM_calls=0,independent_native_audit=True,adoption=False))
    _fields(q,dict(schema='world_reward.surface_qslim_qualification_pins.v1',historical_host_status='fail',
        native_loader_calls=4,official_budget_calls=2,QEM_calls=2,independent_native_receipt_audit=True,
        independent_geometry_replay=False,actual_original_CID_absence_verified=True,adoption=False))
    _fields(b,dict(schema='world_reward.surface_qslim_build_pins.v1'))
    for row in (first,q,b): _hex(row['producer_revision'],40)
    return ids,first,q,b


def verify_pinned_artifacts(root,pins,episode,input_sha,object_sha,alignment_sha,scale):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and not root.is_symlink(), 'Canonical root required')
    require(type(scale) is float and np.isfinite(scale) and scale > 0, 'Original positive clip scale required')
    for sha in (input_sha,object_sha,alignment_sha): _hex(sha)
    require(type(pins) is dict and set(pins) == {'schema','episode_index','input_sha256',
        'metric_scale_baked_once','report','files','source_helpers'}, 'Independent exact surface pins required')
    _fields(pins,dict(schema=SCHEMA,episode_index=episode,input_sha256=input_sha,metric_scale_baked_once=scale))
    pin_path=artifact_pin_path(CODE,episode,'surface_mesh')
    pin_identity=identity(pin_path)
    require(strict_json(pin_path.read_bytes())==pins,'Caller pins differ from independent readonly proposal pins')
    producer = pins['report']
    require(set(producer) == {'bytes','sha256','producer_revision','script_sha256'}, 'Original producer pin required')
    _pin({k:producer[k] for k in ('bytes','sha256')}); _hex(producer['producer_revision'],40); _hex(producer['script_sha256'])
    configs,first,q,b = _configs(); names=paths(episode,q['producer_revision'],producer['producer_revision'],first['producer_revision'])
    require(set(pins['files']) == set(names.values()) and q['independent_audit_path'] == names['qualification_audit'],
            'Exact fifteen surface/source/qualification artifacts required')
    for name,pin in pins['files'].items(): _pin(pin,maximum=MAPPING_MAX_BYTES if name==names['mapping'] else 32<<20)
    ro = {names[k] for k in ('report','native','geometry','glb','candidate','mapping','phase1_native','qualification_native','qualification_audit')}
    before = {n:identity(root/n,readonly=n in ro) for n in names.values()}
    require(before == pins['files'] and before[names['report']] == {k:producer[k] for k in ('bytes','sha256')}
        and before[names['object']]['sha256'] == object_sha and before[names['alignment']]['sha256'] == alignment_sha,
        'Independent bytes must match before interpretation')
    require(stat.S_IMODE((root/names['report']).parent.stat().st_mode) == 0o555, 'Sealed proposal namespace required')
    require(set(pins['source_helpers']) == set(source_helpers()), 'Exact current pure reader helper pins required')
    ledger = {root/n:pin for n,pin in before.items()} | {CODE/n:pin for n,pin in configs.items()} | {pin_path:pin_identity}
    for n,pin in pins['source_helpers'].items(): _pin(pin); require(identity(CODE/n) == pin,'Current pinned math differs'); ledger[CODE/n]=pin
    records = {k:strict_json((root/names[k]).read_bytes()) for k in
        ('report','native','object','alignment','transform','qualification_native','qualification_host','qualification_audit','phase1_native')}
    host,native=records['report'],records['native']
    common=dict(status='pass',phase='complete',domain='surface',episode_index=episode,producer_revision=producer['producer_revision'],
        source_rehashed_after=True,inputs_qualification_rehashed_after=True,runtime_rehashed_after=True,
        gpu_used=False,ground_truth_used=False,adoption=False,reconstruction_accuracy_verified=False,competition_eligibility_verified=False)
    _fields(host,common | dict(stage='world_reward_object_budget_surface_host_v1',owned_container_removed=True,owned_scratch_removed=True))
    _fields(native,common | dict(stage='world_reward_object_budget_surface_native_v1',input_track='track_1',input_sha256=input_sha,
        hand_labeled_test=False,media_decoded=False,frame_poses_changed=False,input_video_hashed=True,budget_seconds=600,
        maximum_qem_calls=1,object_scale=1.,metric_scale_baked_once=scale))
    require(native['oracle_modes'] == [] and host['native'] == native and host['native_identity'] == before[names['native']]
        and host['source_binding'] == host['source_binding_after'] == native['source_binding'] == native['source_binding_after']
        and host['qualification'] == native['qualification'] and host['input_binding'] == native['input_binding']
        and native['input_binding']['video_sha256'] == input_sha and host['image_identity']['Id'] == IMAGE,
        'Complete original native/host ancestry required')
    for row,limit in ((host,700),(native,600)):
        require(type(row['elapsed_seconds']) in (int,float) and 0 < row['elapsed_seconds'] <= limit,'Inclusive original budget differs')
    require(host['outputs'] == native['outputs'] == {n:before[names[k]] for n,k in zip(OUTPUTS,('glb','geometry','candidate','mapping'))},
            'All frozen output byte identities differ')
    bound=host['source_binding']; require(bound['helpers'][PROCESSOR]['sha256'] == producer['script_sha256'],'Original script differs')
    entry = bound.get('source_entry', 'run_object_budget_solid')
    require(entry == ('run_object_budget_solid' if output_prefix() == 'outputs' else CODE.parent.name)
        and (output_prefix() == 'outputs' or entry in {'run_full4d_sample', 'run_gemini_full4d'}),
            'A new experiment may not relabel a legacy surface producer')
    if entry != 'run_object_budget_solid':
        require(entry in {'run_full4d_sample', 'run_gemini_full4d'} and output_prefix() ==
                'experiments/full4d-v1-'+producer['producer_revision']+'/outputs',
                'Exact experiment source entry and matching revision required')
    ledger.update(_snapshot(root,producer['producer_revision'],entry,bound))
    qualification=host['qualification']
    require(qualification['pins_identity'] == configs[QUALIFICATION] and qualification['native'] == q['native'] == before[names['qualification_native']]
        and qualification['historical_host'] == q['historical_host_report'] == before[names['qualification_host']]
        and qualification['independent_audit'] == q['independent_audit_report'] == before[names['qualification_audit']]
        and qualification['original_host_status'] == 'fail' and qualification['build']['binary'] == b['binary']
        and qualification['build']['source_cpp'] == b['source_cpp'] and qualification['phase1']['native'] == first['native'] == before[names['phase1_native']],
        'Independent existing qualification lineages differ')
    oldhost,oldnative,audit=records['qualification_host'],records['qualification_native'],records['qualification_audit']
    require(oldhost['status'] == 'fail' and oldhost['native_report'] is None
        and oldhost['native_identity'] == before[names['qualification_native']] and oldnative['status'] == 'pass'
        and oldnative['phase'] == 'complete' and oldnative['producer_revision'] == q['producer_revision']
        and oldhost['source_proof'] == qualification['original_source_proof']
        and oldnative['source_proof']==qualification['original_source_proof']['native'], 'Original HOSTFAIL may not be relabelled')
    ledger.update(_snapshot(root,q['producer_revision'],'run_surface_qslim_qualify',oldnative['source_proof']['source_binding']))
    first_native=records['phase1_native']
    ledger.update(_snapshot(root,first['producer_revision'],'run_surface_identity_qualify',first_native['source_proof']['source_binding']))
    _fields(audit,dict(stage='world_reward.surface_qslim_independent_receipt_audit.v1',status='pass',producer_revision=q['producer_revision'],
        original_host_status='fail',original_native_status='pass',original_host_receipt_unchanged=True,
        original_exact_CID_absence_verified=True,source_runtime_rehashed_after=True,geometry_replay=False,retained_receipt_only=True))
    _fields(first_native,dict(status='pass',phase='complete',producer_revision=first['producer_revision']))
    require(first_native['controls']['QEM_calls']==0 and first_native['controls']['native_loader_calls_including_authority_replays']==6,
        'Actual original identity controls differ')
    obj,alignment=records['object'],records['alignment']; source=native['source_hashes']
    for row,stage in ((obj,'sam3d_objects_grounded_fixed_frame'),(alignment,'predicted_human_anchored_moge2_pointmaps')):
        _fields(row,dict(stage=stage,status='pass',episode_index=episode,input_track='track_1',input_sha256=input_sha,
            ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]))
    _fields(obj,dict(frame_index=0))
    _fields(alignment,dict(coordinate_frame='OpenCV_x_right_y_down_z_forward',
        pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied'))
    require(source['video'] == input_sha and source['object_report'] == object_sha and source['alignment_report'] == alignment_sha
        and obj['transform'] == records['transform'] and obj['pointmap_grounding']['alignment_report_sha256'] == alignment_sha
        and obj['scale_source'] == 'already_human_anchored_MoGe2_no_second_scalar','Original physical grounding differs')
    scales=records['transform']['scale']
    require(type(scales) is list and len(scales)==3 and all(type(x) in (int,float) and np.isfinite(x) and x>0 for x in scales)
        and float(scales[0]) == scale and all(abs(x-scales[0]) <= 1e-5*abs(scales[0]) for x in scales), 'No averaged or second scale')
    for role,key,field in (('source_glb','object.glb','object_sha256'),('transform','transform.json','transform_sha256'),('intrinsics','intrinsics.json','intrinsics_sha256')):
        require(before[names[role]]['sha256'] == source[key] == obj[field], 'Original fixed artifact differs')
    c=native['compiler']
    _fields(native['source_geometry'],dict(source_identity=before[names['source_glb']],
        native_float32_conversion=True,source_vertices_preserved=True,source_faces_preserved=True,
        welding_performed=False,geometry_repaired=False))
    _fields(c,dict(schema='world_reward.surface_budget.v1',status='pass',phase='complete',domain='surface',
        source_arrays_frozen_before_gates=True,source_arrays_unchanged=True,geometry_repaired=False,
        components_deleted=False,orientation_repaired=False,volume_or_closure_required=False,embedding_certified=False,
        full_surface_fidelity_certified=False,adoption=False,reconstruction_accuracy_verified=False,
        metric_scale_baked_once=scale,object_scale=1.,official_budget_calls=1,
        payload_role='native_canonical_arrays_plus_only_repeat_first_vertex_zero_faces',orphan_vertices_dropped=False))
    require(c['method'] in ('identity','qslim') and type(c['qem_calls']) is int and c['qem_calls']==int(c['method']=='qslim')
        and c['canonical_glb_identity']==before[names['glb']] and c['mapping_identity']==before[names['mapping']], 'Surface branch/output evidence differs')
    fidelity=c['official_pack_fidelity']
    _fields(fidelity,dict(raw_oriented_triangles_exact=True,native_fp32_quantization_exact=True,
        original_nonzero_triangles_preserved=True,oriented_triangles_exact=True,additional_scale_or_alignment=False,geometry_repaired=False))
    _fields(c['default8_serialization'],dict(serialized_triangles_numerically_preserved_by_weld=True,welded_triangles_exactly_active=True))
    recheck(ledger)
    return host,native,ledger,names


def load(root,episode,input_sha,object_report_sha,alignment_sha,scale,*,pins=None):
    root=Path(root); host,native,ledger,names=verify_pinned_artifacts(root,pins,episode,input_sha,object_report_sha,alignment_sha,scale)
    pure=_module('world_reward.surface_pose_geometry','src/world_reward/surface_pose_geometry.py')
    from world_reward.raw_shape_proposal import _identity
    with np.load(root/names['candidate'],allow_pickle=False) as data:
        require(set(data.files)=={'source_vertices','source_faces','candidate_vertices','candidate_faces','canonical_vertices','canonical_faces'},'Exact six-array original/canonical evidence required')
        arrays={n:data[n].copy() for n in data.files}
    sv,sf,cv,cf,rv,rf=(arrays[n] for n in ('source_vertices','source_faces','candidate_vertices','candidate_faces','canonical_vertices','canonical_faces'))
    require(sv.dtype==rv.dtype==np.float32 and cv.dtype in (np.dtype('float32'),np.dtype('float64'))
        and all(x.dtype==np.int64 for x in (sf,cf,rf)) and np.array_equal(cf,rf)
        and np.array_equal((cv.astype(np.float64)*scale).astype(np.float32),rv),'Original candidate/clip-constant once-baked canonical geometry differs')
    c=native['compiler']; require(c['source_vertices']==_identity(sv) and c['source_faces']==_identity(sf)
        and c['candidate_vertices']==_identity(cv) and c['candidate_faces']==_identity(cf)
        and c['canonical_surface']['vertices']==_identity(rv) and c['canonical_surface']['faces']==_identity(rf)
        and c['canonical_vertex_count']==len(rv) and c['canonical_face_count']==len(rf),'Frozen complete array evidence differs')
    require((c['method']=='identity') == (len(sv)<=4096 and len(sf)<=4096), 'Exactly one generic source-count dispatch required')
    with np.load(root/names['geometry'],allow_pickle=False) as data:
        require(set(data.files)=={'vertices','faces','episode_index','object_scale','grounded_scale_baked'},'Exact five-array metric payload required')
        v,f=data['vertices'].copy(),data['faces'].copy()
        require(v.shape==f.shape==(4096,3) and v.dtype==np.float64 and f.dtype==np.int64,'Original canonical padded budgets required')
        require(all(data[n].shape==() and data[n].dtype==dtype and data[n].item()==value for n,dtype,value in
            (('episode_index',np.dtype('int64'),episode),('object_scale',np.dtype('float64'),1.),('grounded_scale_baked',np.dtype('float64'),scale))),'No second metric scale')
    compact,faces,active,topology=pure.compact_surface(v,f,reference_vertices=rv.astype(np.float64),reference_faces=rf)
    record=c['canonical_surface']
    require(record['component_keys']==topology['component_keys'] and record['boundary_loops']==topology['boundary_loops']
        and record['boundary_components']==topology['boundary_components'] and record['components']==topology['diagnostics']['components']
        and record['unused_vertices_preserved']==topology['canonical_unused_vertices_preserved'],
        'Original canonical components/boundaries/orphan evidence differs')
    precision=_module('mesh_precision_diagnostic','infra/mesh_precision_diagnostic.py')
    local,world,records=precision.raw_glb(root/names['glb'])
    require(len(local)==1 and all(r['node_transform_identity'] for r in records)
        and local[0][0].dtype==np.float32 and local[0][0].shape==rv.shape and local[0][0].tobytes()==rv.tobytes()
        and local[0][1].dtype==np.int64 and local[0][1].shape==faces.shape and local[0][1].tobytes()==faces.tobytes()
        and precision.triangle_hash(world)==precision.triangle_hash(compact[faces]), 'Canonical metric GLB changed original full rows/faces/transform')
    mapping=strict_json((root/names['mapping']).read_bytes())
    require(len(mapping['I'])==len(cv) and len(mapping['J'])==len(cf)
        and all(type(x)is int and 0<=x<len(sv) for x in mapping['I'])
        and all(type(x)is int and 0<=x<len(sf) for x in mapping['J']), 'Full original vertex/face births required')
    if c['method']=='identity':
        require(mapping['schema']=='surface-identity-mapping-v1' and mapping['I']==list(range(len(sv)))
            and mapping['J']==list(range(len(sf))) and sv.tobytes()==cv.tobytes() and sf.tobytes()==cf.tobytes(), 'Identity route changed original geometry')
    else:
        _fields(mapping,dict(schema='surface-qslim-mapping-v1',source_vertices=len(sv),source_faces=len(sf),
            output_vertices=len(cv),output_faces=len(cf),target_vertices=4096,target_faces=4096,
            boundary_policy='fixed_original_vertices',intersection_blocking='upstream_floating_point',
            initial_embedding_certified=False,volume_or_closure_required=False,serialization_qualification_completed=False,adoption=False))
        require(mapping['source_sha256']==strict_json((CODE/BUILD).read_bytes())['source_cpp']['sha256']
            and c['oriented_birth_quotient_replay_exact'] is True and cv.dtype==np.float64
            and len(set(mapping['I']))==len(cv) and len(set(mapping['J']))==len(cf)
            and type(mapping['committed_collapses'])is int and mapping['committed_collapses']==len(sv)-len(cv)>0
            and len(mapping['ledger'])==mapping['committed_collapses'],
            'Authenticated CPU whole QSlim birth replay/source/collapse census differs')
    recheck(ledger)
    v=np.frombuffer(v.tobytes(),dtype=v.dtype).reshape(v.shape)
    f=np.frombuffer(f.tobytes(),dtype=f.dtype).reshape(f.shape)
    cleanup=dict(faces=len(f),active_faces=len(active),padding_faces=len(f)-len(active),meaningful_faces_removed=0,normal_repair_performed=False)
    receipt=dict(backend='frozen_surface_identity_qslim',source_domain='surface',
        committed_pins_sha256=identity(artifact_pin_path(CODE,episode,'surface_mesh'))['sha256'],
        producer_report_sha256=ledger[root/names['report']]['sha256'],cpu_native_report_sha256=ledger[root/names['native']]['sha256'],
        cpu_producer_revision=pins['report']['producer_revision'],cpu_script_sha256=pins['report']['script_sha256'],
        metric_scale_baked_once=scale,metric_scale_already_baked=True,no_cpu_solver_executed=True,
        geometry_operations_applied=False,source_geometry_repaired=False,original_artifacts_rehashed=True,
        canonical_vertices_count=len(compact),canonical_faces_count=len(faces),canonical_vertex_count=len(compact),canonical_face_count=len(faces),
        surface_topology=topology,official_pack_fidelity=c['official_pack_fidelity'],cpu_source_binding=host['source_binding'],
        embedding_verified=False,full_surface_fidelity_certified=False,challenge_performance_verified=False,adoption_authorized=False)
    return v,f,active,cleanup,root/names['glb'],receipt


def preflight_geometry_and_poses(path,*,expected_vertices=None,expected_faces=None,expected_v=None,expected_f=None,
        expected_episode=None,expected_scale=None,topology_budget=None):
    """Lossless frozen pose NPZ, full frame indices and unchanged canonical padding."""
    before=identity(path); pure=_module('world_reward.surface_pose_geometry','src/world_reward/surface_pose_geometry.py')
    ev=expected_vertices if expected_vertices is not None else expected_v
    ef=expected_faces if expected_faces is not None else expected_f
    with np.load(path,allow_pickle=False) as data:
        require(set(data.files)=={'vertices','faces','frame_index','rotation','translation','object_scale'},'Exact full pose payload required')
        v,f,r,t,indices=(data[n].copy() for n in ('vertices','faces','rotation','translation','frame_index'))
        require(v.dtype==np.float64 and f.dtype==np.int64 and v.shape==f.shape==(4096,3)
            and v.tobytes()==ev.tobytes() and f.tobytes()==ef.tobytes(),'Frozen pose changed complete canonical padded geometry')
        require(indices.dtype==np.int64 and indices.ndim==1 and len(indices)>0 and np.array_equal(indices,np.arange(len(indices)))
            and data['object_scale'].shape==() and data['object_scale'].dtype==np.float64 and data['object_scale'].item()==1.,'Full original timeline/once-baked metric scale required')
    require(r.dtype==t.dtype==np.float64 and r.shape==(len(indices),3,3) and t.shape==(len(indices),3)
        and np.isfinite(r).all() and np.isfinite(t).all() and np.allclose(r@r.swapaxes(-1,-2),np.eye(3),atol=1e-5,rtol=0)
        and np.allclose(np.linalg.det(r),1.,atol=1e-5,rtol=0),'Finite full proper poses required')
    require(type(topology_budget)is dict and topology_budget['source_domain']=='surface'
        and type(topology_budget['metric_scale_baked_once']) is float and topology_budget['metric_scale_baked_once']>0
        and (expected_scale is None or expected_scale==1.),'Explicit frozen surface geometry proof and pose scale1 required')
    compact,faces,active,top=pure.compact_surface(v,f,canonical_vertex_count=topology_budget['canonical_vertices_count'],
        canonical_face_count=topology_budget['canonical_faces_count'])
    require(identity(path)==before,'Pose payload changed while read')
    v,f,r,t=(np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape) for a in (v,f,r,t))
    return v,f,active,r,t,(compact,faces),top,{Path(path):before}
