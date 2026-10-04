"""Inert consumer of one independently pinned CPU whole-solid proposal.

No compiler, CGAL query, model or build-info is executed here. Exact embedding
and birth-forest proofs remain claims of the authenticated CPU producer; the
stored metric geometry and padding are checked independently by pure readers.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import stat

import numpy as np

CODE = Path(__file__).resolve().parent.parent
SCHEMA = 'world_reward.solid_mesh_pins.v1'
PROCESSOR = 'infra/object_budget_solid.py'
QUALIFICATION = 'configs/solid_chart_v2_qualification_pins.json'
BUILD = 'configs/solid_chart_v2_build_pins.json'
CGAL = 'configs/certified_solid_qualification_pins.json'
BALANCED_QUALIFICATION = 'configs/solid_chart_v2_balanced_qualification_pins.json'
BALANCED_CGAL = 'configs/certified_solid_balanced_qualification_pins.json'
SOURCE_HELPERS = ('infra/exact_mesh_geometry.py', 'infra/object_budget_endpoint.py',
    'infra/mesh_link_gate.py', 'infra/mesh_endpoint_gate.py', 'infra/guarded_mesh_gate.py',
    'infra/body_smoke.py', 'src/world_reward/data.py', 'src/world_reward/exact_triangle_predicates.py',
    'src/world_reward/oriented_solid_forest.py', 'src/world_reward/mesh_conditioning_v2.py',
    'infra/mesh_conditioned_chart_v2.hpp')
STAGES = ('physical_source', 'native_candidate', 'float32_glb', 'default8_exact_weld',
          'unmodified_official_pack', 'original_grounding_metric_bake')
TRUE = ('closed_oriented_vertex_manifold_verified', 'all_original_faces_retained',
    'all_original_vertices_referenced', 'exact_nondegenerate_triangles_verified',
    'component_self_intersections_absent', 'inter_component_surface_contacts_absent')
FALSE = ('geometry_repaired', 'orientation_changed', 'qem_executed', 'forest_adjudicated',
         'reconstruction_accuracy_verified')
OFFICIAL = dict(bytes=2031, sha256='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0')


def require(ok, message):
    if not ok: raise ValueError(message)


def _hex(value, length=64):
    require(type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value), 'Exact hex identity required')
    return value


def _pin(value, maximum=32 << 20):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'} and type(value['bytes']) is int
            and 0 < value['bytes'] <= maximum, 'Exact bounded byte identity required')
    _hex(value['sha256']); return value


def strict_json(raw):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, 'Duplicate JSON field'); out[key] = value
        return out
    result = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    json.dumps(result, allow_nan=False); return result


def identity(path, *, readonly=True):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
            'Canonical artifact required')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= 32 << 20
            and (not readonly or not before.st_mode & 0o222), 'Bounded single-link readonly artifact required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    after = path.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in
        ('st_dev','st_ino','st_size','st_mode','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns')), 'Artifact changed')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def _fields(row, values):
    require(type(row) is dict and all(type(row.get(k)) is type(v) and row[k] == v for k,v in values.items()),
            'Exact producer fields required')


def _module(name,path):
    module=importlib.import_module(name)
    require(Path(module.__file__).resolve()==CODE/path, 'Actual source-bound pure module required')
    return module


def _number(value, *, maximum=None, positive=False):
    require(type(value) in (int,float) and math.isfinite(value) and (value > 0 if positive else value >= 0)
            and (maximum is None or value <= maximum), 'Finite bounded measurement required')


def paths(episode, qualification_revision, producer_revision, *, query_requalification=False):
    require(type(query_requalification) is bool, 'Explicit query profile required')
    require(type(episode) is int and 0 <= episode < 30, 'Exact Track1 episode required')
    _hex(qualification_revision,40); _hex(producer_revision,40)
    base = f'outputs/episode_{episode:06d}'
    q = ('results/solid-chart-v2-query-requalify-' if query_requalification else 'results/solid-chart-v2-qualify-') + qualification_revision
    proposal=base+('/object_budget_solid_balanced_' if query_requalification else '/object_budget_solid_')+producer_revision
    return dict(report=proposal+'/report.json', native=proposal+'/native.json',
        geometry=proposal+'/geometry.npz', glb=proposal+'/object_fixed_canonical.glb',
        object=base+'/object_grounded/report.json', source_glb=base+'/object_grounded/object.glb',
        transform=base+'/object_grounded/transform.json', intrinsics=base+'/object_grounded/intrinsics.json',
        alignment=base+'/scale_smoke/report.json', qualification_host=q+'/report.json', qualification_native=q+'/native.json')


def _source(binding, producer):
    _fields(binding, dict(producer_revision=producer['producer_revision']))
    require(type(binding['source_files']) is int and binding['source_files'] >= len(binding['helpers']) > 0,
            'Full source count and selected helper ledger required')
    for key in ('source_files_sha256', 'source_readonly_ledger_sha256'): _hex(binding[key])
    for name, pin in binding['helpers'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts, 'Source path differs'); _pin(pin)
    require(binding['helpers'][PROCESSOR]['sha256'] == producer['script_sha256'], 'Original processor differs')
    markers = binding['markers']; require(set(markers) == {'revision','source-sha256'}, 'Original markers required')
    require(markers['revision'] == dict(bytes=41, sha256=hashlib.sha256((producer['producer_revision']+'\n').encode()).hexdigest()),
            'Original revision marker differs'); _pin(markers['source-sha256'])


def _query(row, query_sha):
    OrientedSolidForest=_module('world_reward.oriented_solid_forest','src/world_reward/oriented_solid_forest.py').OrientedSolidForest
    _fields(row, dict(query_attempted=True, query_returned=True, query_returncode=0)); _pin(row['query_input'],256 << 20)
    c, f, top = row['native_certificate'], row['forest'], row['topology']; count = c['component_count']
    require(type(count) is int and 1 <= count <= 256, 'Complete component count required')
    _fields(c, dict(schema='world_reward.certified_solid_query.v1', status='pass', source_sha256=query_sha,
        cgal_version='6.0.1', represented_coordinates='EPECK exact values of original parsed IEEE754 binary64; no perturbation',
        **{k:True for k in TRUE}, **{k:False for k in FALSE}))
    require(all(type(c[k]) is int and 4 <= c[k] <= limit for k,limit in (('vertices',1000000),('faces',2000000))) and
            type(f) is dict and set(f) == {'component_keys','signs','inside','parents','depths'} and
            type(f['component_keys']) is list and len(f['component_keys']) == count and
            all(type(x) is int for key in ('signs','parents','depths') for x in f[key]) and
            all(type(row) is list and all(type(x) is bool for x in row) for row in f['inside']), 'Typed whole forest required')
    forest = OrientedSolidForest(tuple(f['component_keys']), np.asarray(f['signs'],np.int64), np.asarray(f['inside'],bool))
    require(forest.parents.tolist() == f['parents'] and forest.depths.tolist() == f['depths'] and c['inside'] == f['inside'],
            'Exact parent/depth/containment differs')
    require(len(c['components']) == count and top['vertices'] == top['active_vertices'] == c['vertices']
            and top['faces'] == c['faces'] and top['closed_oriented_vertex_manifold'] is True
            and len(top['components']) == count, 'Full topology counts differ')
    for i, component in enumerate(c['components']):
        require(type(component['original_component_id']) is int and component['original_component_id'] == i and
            type(component['exact_volume_sign']) is int and component['exact_volume_sign'] == f['signs'][i] and
            all(type(component[k]) is int and component[k] >= 4 for k in ('original_vertices','original_faces')) and
            type(component['witness_original_vertex']) is int and 0 <= component['witness_original_vertex'] < c['vertices'],
            'Original component/witness differs')
    require(sum(x['original_vertices'] for x in c['components'])==c['vertices'] and
        sum(x['original_faces'] for x in c['components'])==c['faces'], 'Every original indexed component must be counted')
    for component in top['components']:
        require(type(component['euler']) is int and type(component['volume_sign']) is int
                and component['volume_sign'] in (-1,1), 'Signed shell topology required')
        require(type(component['signed_volume']) in (int,float), 'Signed volume numeric type differs')
        _number(abs(component['signed_volume']), positive=True)
        require(np.sign(component['signed_volume']) == component['volume_sign'], 'Volume orientation differs')
    require(sorted(f['signs']) == sorted(x['volume_sign'] for x in top['components']), 'Exact/floating signs differ')
    require(type(row['stored_array_sha256']) is list and len(row['stored_array_sha256']) == 2, 'Stored represented arrays required')
    for digest in row['stored_array_sha256']: _hex(digest)
    return forest


def _match(row, reference, query_sha, *, measured):
    candidate = _query(row, query_sha); original = _query(reference, query_sha)
    p = row['candidate_to_source_components']; n = len(original.signs)
    require(type(p) is list and all(type(x) is int for x in p) and sorted(p) == list(range(n)), 'Full birth component bijection required')
    p = np.asarray(p,np.int64)
    require(len(candidate.signs) == n and np.array_equal(candidate.signs,original.signs[p]) and
        np.array_equal(candidate.depths,original.depths[p]) and np.array_equal(candidate.inside,original.inside[np.ix_(p,p)]),
        'Original material forest changed')
    parents = np.full(n,-1,np.int64); active = candidate.parents >= 0; parents[active] = p[candidate.parents[active]]
    require(np.array_equal(parents,original.parents[p]), 'Original parent relation changed')
    if not measured: return
    f = row['fidelity']; _fields(f, dict(scale_or_pose_fitted=False))
    _number(f['sampled_bidirectional_chamfer_diagonal_ratio'],maximum=.01); _number(f['net_volume_relative_error'],maximum=.05)
    shells = f['birthface_matched_shells']
    require(type(shells) is list and len(shells) == n and all(type(x['source_component']) is int for x in shells)
            and sorted(x['source_component'] for x in shells) == list(range(n)), 'Every original shell volume required')
    for shell in shells:
        require(type(shell['euler']) is int and type(shell['volume_sign']) is int
            and shell['volume_sign'] in (-1,1), 'Euler/orientation differs')
        a,b = shell['source_volume'],shell['candidate_volume']; require(type(a) in (int,float) and type(b) in (int,float), 'Volume numeric types differ')
        _number(abs(a),positive=True); _number(abs(b),positive=True)
        require(np.sign(a) == np.sign(b) == shell['volume_sign'], 'Signed volume changed')
        _number(shell['relative_volume_error'],maximum=.05)
        require(shell['relative_volume_error'] == abs(b-a)/abs(a), 'Reported volume error differs')
    require(sorted(x['euler'] for x in shells) == sorted(x['euler'] for x in reference['topology']['components']) ==
            sorted(x['euler'] for x in row['topology']['components']), 'Original shell Euler changed')
    # Legacy component_labels is vertex-order keyed; CGAL keys are first-face
    # ordered. Never reinterpret legacy shell IDs as CGAL component identities.
    key=lambda x,volume:(x['euler'],x['volume_sign'],x[volume])
    require(sorted(key(x,'source_volume') for x in shells)==sorted(key(x,'signed_volume') for x in reference['topology']['components'])
        and sorted(key(x,'candidate_volume') for x in shells)==sorted(key(x,'signed_volume') for x in row['topology']['components']),
        'All measured signed shell volumes must match independently ordered topology')
    av=sum(x['source_volume'] for x in shells); bv=sum(x['candidate_volume'] for x in shells)
    _number(av,positive=True); _number(bv,positive=True)
    require(f['net_volume_relative_error']==abs(bv-av)/av, 'Reported whole signed volume error differs')
    require(f['candidate_topology'] == row['topology'] and f['source_topology'] == reference['topology'], 'Measured topology differs')


def _compiler(c, scale, query_sha):
    _fields(c, dict(stage='oriented_solid_compiler_v2',conditioning_version=2,status='pass',phase='complete',
        native_attempts=1,native_returned=True,native_returncode=0,source_arrays_unchanged=True,output_vertices=4096,output_faces=4096,
        metric_scale_baked_once=scale,**{k:False for k in ('geometry_repaired','components_deleted','orientation_changed','cost_backend_changed',
            'ground_truth_used','adoption','reconstruction_accuracy_verified','frame_poses_changed','metric_scale_accuracy_verified')}))
    require(c['artifacts_before'] == c['artifacts_after'] and set(c['stages']) == set(STAGES), 'Full compiler lifecycle required')
    source, metric = c['stages']['physical_source'],c['metric_source_certificate']; _query(source,query_sha)
    _match(source['float32_certificate'],source,query_sha,measured=False); _match(metric,source,query_sha,measured=False)
    for name in STAGES[1:]: _match(c['stages'][name],metric if name == STAGES[-1] else source,query_sha,measured=True)
    _fields(source['conditioning'],dict(roundtrip_numerically_exact=True,source_geometry_repaired=False,source_arrays_modified=False))
    _fields(source['float32_orientation'],dict(exact_positive_normal_dot=True,area_tolerance_used=False))
    chart = c['native_mapping']['conditioning']; _fields(chart,dict(chart_version=2,source_roundtrip_numerically_exact=True,chart_refitted=False,adopted=False))
    POLICY_SHA256=_module('world_reward.mesh_conditioning_v2','src/world_reward/mesh_conditioning_v2.py').POLICY_SHA256
    require(source['conditioning_header']==identity(CODE/'infra/mesh_conditioned_chart_v2.hpp') and
        chart['header_sha256']==source['conditioning_header']['sha256'] and chart['policy_sha256']==POLICY_SHA256==source['conditioning']['policy_sha256'],
        'Actual immutable chart header/policy differs')
    require(type(c['native_mapping']['serialization']['committed_collapses']) is int and
            c['native_mapping']['serialization']['committed_collapses'] >= 0, 'Actual native mapping required')
    s,m=c['native_mapping']['serialization'],c['native_mapping']['native_volume']
    _fields(s,dict(serialization_safe=True)); _fields(m,dict(committed_collapses=s['committed_collapses'],
        native_cost_and_placement_unchanged=False,cost_normalization=True,final_shell_volumes_verified=True,volume_relative_limit=.05))
    require(type(s['serialization_vetoes']) is int and s['serialization_vetoes']>=0, 'Actual serialization transaction counts required')
    _fields(c['official_pack_fidelity'],dict(oriented_triangles_exact=True,official_helper_simplification_invoked=False,nonexact_merge_or_face_deletion=False))
    queries=[]
    def visit(value):
        if type(value) is dict:
            if 'query_attempted' in value: queries.append(value)
            for child in value.values(): visit(child)
        elif type(value) is list:
            for child in value: visit(child)
    visit(c); require(len(queries) == 8, 'Exactly eight original full geometry queries required')


def _query_composition(row, original, active, configs):
    """Metadata-only composition; old QEM inputs never become the new query."""
    require(type(row) is dict and set(row)=={'original_query','active_query','original_qualified_inputs',
        'original_qem_build_unchanged','original_qem_recompiled','procedural_controls_required'}, 'Exact query composition required')
    _fields(row,dict(original_qem_build_unchanged=True,original_qem_recompiled=False,procedural_controls_required=15))
    for key,record,config in (('original_query',original,CGAL),('active_query',active,BALANCED_CGAL)):
        _fields(record,dict(schema='world_reward.certified_solid_qualification_pins.v1',qualified_controls=15))
        _hex(record['producer_revision'],40); _pin(record['native_source'])
        for name in ('report','native','binary'): _pin(record[name])
        item=row[key]
        require(type(item) is dict and set(item)=={'pins_identity','artifacts','native_source','producer_revision'}, 'Explicit query identity required')
        require(item==dict(pins_identity=configs[config],artifacts={n:record[n] for n in ('report','native','binary')},
            native_source=record['native_source'],producer_revision=record['producer_revision']), 'Pinned original/active query differs')
    require(original['parent_image_id']==active['parent_image_id'] and original['child_image_id']==active['child_image_id'] and
        original['producer_revision']!=active['producer_revision'] and original['native_source']['sha256']!=active['native_source']['sha256'],
        'Distinct qualified query under unchanged runtime required')
    inherited=row['original_qualified_inputs']
    require(type(inherited) is dict and set(inherited)=={'cgal','cache','historical_image_receipt'} and
        inherited['cgal']==row['original_query']['artifacts'], 'Original qualified inputs must retain old query')
    _pin(inherited['historical_image_receipt']); cache=inherited['cache']
    require(type(cache) is dict and set(cache)=={'pins_identity','artifacts','build_info','compiler'} and
        type(cache['artifacts']) is dict and set(cache['artifacts'])=={'host_report','native_report','retained_binary'}, 'Original qualified cache required')
    _pin(cache['pins_identity'])
    for value in cache['artifacts'].values(): _pin(value)


def verify_pinned_artifacts(root,pins,episode,input_sha,object_sha,alignment_sha,scale,*,query_requalification=False):
    require(type(query_requalification) is bool, 'Explicit query profile required')
    root=Path(root); require(root.is_absolute() and root.resolve()==root and not root.is_symlink(), 'Canonical root required')
    require(type(episode) is int and 0 <= episode < 30 and type(scale) is float, 'Exact episode and original scale required'); _number(scale,positive=True)
    for value in (input_sha,object_sha,alignment_sha): _hex(value)
    require(type(pins) is dict and set(pins)=={'schema','episode_index','input_sha256','metric_scale_baked_once','report','files','source_helpers'}, 'Explicit independent solid pins required')
    _fields(pins,dict(schema=SCHEMA,episode_index=episode,input_sha256=input_sha,metric_scale_baked_once=scale))
    producer=pins['report']; require(type(producer) is dict and set(producer)=={'bytes','sha256','producer_revision','script_sha256'}, 'Independent processor identity required')
    _pin({k:producer[k] for k in ('bytes','sha256')}); _hex(producer['producer_revision'],40); _hex(producer['script_sha256'])
    selected = BALANCED_QUALIFICATION if query_requalification else QUALIFICATION
    config_names=(selected,BUILD,CGAL)+((BALANCED_CGAL,) if query_requalification else ())
    configs={name:identity(CODE/name) for name in config_names}
    q,b,g=(strict_json((CODE/name).read_bytes()) for name in (selected,BUILD,CGAL))
    active=strict_json((CODE/BALANCED_CGAL).read_bytes()) if query_requalification else g
    _fields(q,dict(schema='world_reward.solid_chart_v2_qualification_pins.v1',qualified_controls=4,native_qem_calls=4,native_query_calls=40,adoption=False,reconstruction_accuracy_verified=False))
    _hex(q['producer_revision'],40); names=paths(episode,q['producer_revision'],producer['producer_revision'],query_requalification=query_requalification)
    require(type(pins['files']) is dict and set(pins['files'])==set(names.values()), 'Exact eleven source/qualification artifacts required')
    for pin in pins['files'].values(): _pin(pin)
    ro={names[k] for k in ('report','native','geometry','glb','qualification_host','qualification_native')}
    before={name:identity(root/name,readonly=name in ro) for name in names.values()}; require(before==pins['files'], 'All artifacts must match before interpretation')
    require(all(stat.S_IMODE((root/name).stat().st_mode)==0o444 for name in ro), 'Sealed proposal/qualification receipts must remain0444')
    require(before[names['report']]=={k:producer[k] for k in ('bytes','sha256')} and
        before[names['object']]['sha256']==object_sha and before[names['alignment']]['sha256']==alignment_sha, 'Independent source report differs')
    require(stat.S_IMODE((root/names['report']).parent.stat().st_mode)==0o555, 'Sealed proposal directory required')
    records={role:strict_json((root/names[role]).read_bytes()) for role in ('report','native','object','alignment','transform','qualification_host','qualification_native')}
    host,native=records['report'],records['native']; _source(host['source_binding'],producer)
    require(host['native']==native and host['source_binding']==host['source_binding_after']==native['source_binding']==native['source_binding_after'], 'Full host/native source equality required')
    _fields(host,dict(stage='world_reward_object_budget_solid_host_v1',status='pass',phase='complete',episode_index=episode,producer_revision=producer['producer_revision'],
        gpu_used=False,ground_truth_used=False,adoption=False,reconstruction_accuracy_verified=False,competition_eligibility_verified=False,
        owned_container_removed=True,owned_scratch_removed=True,source_rehashed_after=True,inputs_qualification_rehashed_after=True,media_decoded=False))
    _fields(native,dict(stage='world_reward_object_budget_solid_native_v1',status='pass',phase='complete',episode_index=episode,input_track='track_1',input_sha256=input_sha,
        native_query_calls=8,budget_seconds=1800,qem_seconds=1200,query_seconds=180,maximum_qem_calls=1,maximum_query_calls=8,object_scale=1.,original_grounded_scale=scale,metric_scale_baked_once=scale,
        **{k:False for k in ('gpu_used','ground_truth_used','hand_labeled_test','adoption','reconstruction_accuracy_verified','competition_eligibility_verified','media_decoded','frame_poses_changed')},
        **{k:True for k in ('input_video_hashed','source_rehashed_after','inputs_qualification_rehashed_after','runtime_rehashed_after')}))
    require(native['oracle_modes']==[] and host['input_binding']==native['input_binding'] and native['input_binding']['video_sha256']==input_sha,
            'Original video-only input differs'); _number(host['elapsed_seconds'],positive=True,maximum=1900); _number(native['elapsed_seconds'],positive=True,maximum=1800)
    require(host['outputs']==native['outputs']=={Path(names[k]).name:before[names[k]] for k in ('geometry','glb')}, 'Frozen output identities differ')
    require(type(pins['source_helpers']) is dict and set(pins['source_helpers'])==set(SOURCE_HELPERS), 'Exact reused pure source allowlist required')
    for name,pin in pins['source_helpers'].items(): _pin(pin); require(identity(CODE/name)==pin, 'Reused math source differs')
    qualification=host['qualification']; require(native['qualification']==qualification and qualification['pins_identity']==configs[selected] and
        qualification['report']==q['report']==before[names['qualification_host']] and qualification['native']==q['native']==before[names['qualification_native']], 'Actual qualification byte evidence differs')
    qhost,qnative=records['qualification_host'],records['qualification_native']; require(qhost['native']==qnative and
        qhost['source_binding']==qhost['source_binding_after']==qnative['source_binding']==qnative['source_binding_after'], 'Qualification source changed')
    require(all(qualification['source'][k]==qhost['source_binding'][k]==q[k] for k in
        ('producer_revision','source_files','source_files_sha256','source_readonly_ledger_sha256')), 'Whole qualification source ledger differs')
    for row,stage in ((qhost,'solid_chart_v2_qualification_host_v1'),(qnative,'solid_chart_v2_qualification_native_v1')):
        _fields(row,dict(stage=stage,status='pass',phase='complete',qualified_procedural_controls=4,source_rehashed_after=True,artifacts_rehashed_after=True,
            official_rehashed_after=True,**{k:False for k in ('gpu_used','gt_used','adoption','reconstruction_accuracy_verified','competition_eligibility_verified')}))
    _fields(qhost,dict(owned_container_removed=True,owned_scratch_removed=True)); _fields(qnative,dict(source_arrays_unchanged=True))
    require(qualification['build']==qhost['qualified_build']==qnative['qualified_build'] and qualification['build']['pins_identity']==configs[BUILD]
        and q['build_pins']==configs[BUILD] and qualification['build']['built_artifacts']=={k:b[k] for k in ('report','native','binary')}
        and qnative['image_id']==native['image_id']==host['image_identity']['child_id']==b['image_id']==g['child_image_id'], 'Qualified actual build/image differs')
    _fields(g,dict(schema='world_reward.certified_solid_qualification_pins.v1',qualified_controls=15))
    require(qualification['build']['cgal']=={k:g[k] for k in ('report','native','binary')}, 'Qualified exact query differs')
    if query_requalification:
        qr=qualification['build']['query_requalification']; _query_composition(qr,g,active,configs)
        require(all(row.get('query_requalification')==qr for row in (host,native,qualification,qhost,qnative)),
            'Full original/active query composition must match every producer proof')
        require(active['child_image_id']==b['image_id'], 'Active query runtime differs')
    source=native['source_hashes']; obj,alignment=records['object'],records['alignment']
    for row,stage in ((obj,'sam3d_objects_grounded_fixed_frame'),(alignment,'predicted_human_anchored_moge2_pointmaps')):
        _fields(row,dict(stage=stage,status='pass',episode_index=episode,input_track='track_1',input_sha256=input_sha,ground_truth_used=False,hand_labeled_test=False,oracle_modes=[]))
    require(source['video']==input_sha and source['object_report']==object_sha and source['alignment_report']==alignment_sha and obj['transform']==records['transform']
        and obj['pointmap_grounding']['alignment_report_sha256']==alignment_sha and obj['scale_source']=='already_human_anchored_MoGe2_no_second_scalar', 'Original grounding ancestry differs')
    _fields(obj,dict(frame_index=0)); _fields(alignment,dict(coordinate_frame='OpenCV_x_right_y_down_z_forward',pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied'))
    values=records['transform']['scale']; require(type(values) is list and len(values)==3 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in values)
        and float(values[0])==scale and all(abs(x-values[0])<=1e-5*abs(values[0]) for x in values), 'Original scale cannot be averaged/rebaked')
    for role,key,field in (('source_glb','object.glb','object_sha256'),('transform','transform.json','transform_sha256'),('intrinsics','intrinsics.json','intrinsics_sha256')):
        require(before[names[role]]['sha256']==source[key]==obj[field], 'Original constant artifact differs')
    c=native['compiler']; _compiler(c,scale,active['native_source']['sha256']); require(c['glb_identity']==before[names['glb']], 'Canonical GLB differs')
    proof=native['source_geometry']; weld=proof['exact_welding']; source_top=c['stages']['physical_source']['topology']
    require(proof['source_identity']==before[names['source_glb']], 'Original raw source identity differs')
    _hex(proof['raw_oriented_triangles_sha256'])
    _fields(weld,dict(faces_preserved=source_top['faces'],welded_vertices=source_top['vertices'],face_order_and_coordinates_exact=True,
        faces_removed=0,position_merging='exact equality only',source_modified=False))
    require({name:identity(root/name,readonly=name in ro) for name in before}==before and {name:identity(CODE/name) for name in configs}==configs and
        {name:identity(CODE/name) for name in pins['source_helpers']}==pins['source_helpers'],
            'Artifacts changed during inert proof validation')
    return host,native,before,names


def load(root,episode,input_sha,object_report_sha,alignment_sha,scale,*,pins=None,query_requalification=False):
    require(type(query_requalification) is bool, 'Explicit query profile required')
    root=Path(root)
    options=dict(query_requalification=True) if query_requalification else {}
    host,native,before,names=verify_pinned_artifacts(root,pins,episode,input_sha,object_report_sha,alignment_sha,scale,**options)
    geometry=_module('exact_mesh_geometry','infra/exact_mesh_geometry.py'); endpoint=_module('object_budget_endpoint','infra/object_budget_endpoint.py')
    with np.load(root/names['geometry'],allow_pickle=False) as data:
        require(set(data.files)=={'vertices','faces','episode_index','object_scale','grounded_scale_baked'}, 'Exact five-array payload required')
        v,f=data['vertices'].copy(),data['faces'].copy()
        require(v.dtype==np.float64 and f.dtype==np.int64 and v.shape==f.shape==(4096,3) and np.isfinite(v).all()
            and np.all((f>=0)&(f<4096)), 'Finite F64/I64 official4096 arrays required')
        require(all(data[n].shape==() and data[n].dtype==dtype and data[n].item()==value for n,dtype,value in
            (('episode_index',np.dtype('int64'),episode),('object_scale',np.dtype('float64'),1.),('grounded_scale_baked',np.dtype('float64'),scale))), 'Original metric scale already baked')
    canonical=endpoint._load_mesh(root/names['glb']); compact,fidelity=geometry.verify_pack_fidelity((canonical[0]*scale,canonical[1]),v,f)
    active=np.flatnonzero(~np.all(f==0,axis=1)).astype(np.int64); topology=geometry.exact_mesh_topology(*compact)
    def digest(a): return hashlib.sha256(json.dumps(dict(dtype=a.dtype.str,shape=a.shape),sort_keys=True).encode()+b'\0'+np.ascontiguousarray(a).tobytes()).hexdigest()
    stage=native['compiler']['stages'][STAGES[-1]]
    require(topology==stage['topology']==stage['fidelity']['candidate_topology'] and [digest(a) for a in compact]==stage['stored_array_sha256'], 'Stored metric geometry differs from CPU-qualified stage')
    require({name:identity(root/name,readonly=name in {names[k] for k in ('report','native','geometry','glb','qualification_host','qualification_native')}) for name in before}==before,
            'Original artifacts changed during payload read')
    require({name:identity(CODE/name) for name in pins['source_helpers']}==pins['source_helpers'], 'Pure reader source changed during payload read')
    cleanup=dict(faces=len(f),active_faces=len(active),padding_faces=len(f)-len(active),meaningful_faces_removed=0,normal_repair_performed=False)
    receipt=dict(backend='frozen_oriented_solid_chart_v2',cpu_report_sha256=before[names['report']]['sha256'],native_report_sha256=before[names['native']]['sha256'],
        cpu_producer_revision=pins['report']['producer_revision'],cpu_script_sha256=pins['report']['script_sha256'],original_artifacts_rehashed=True,
        metric_scale_already_baked=True,resimplification_performed=False,actual_topology_verified=True,metric_oriented_triangle_fidelity=fidelity,
        independent_embedding_reverified_here=False,source_geometry_repaired=False,challenge_performance_verified=False,cpu_source_binding=host['source_binding'])
    if query_requalification: receipt['query_requalification']=host['query_requalification']
    return v,f,active,cleanup,root/names['glb'],receipt
