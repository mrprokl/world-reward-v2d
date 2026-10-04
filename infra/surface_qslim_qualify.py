"""Fresh open-surface QSlim controls; native/packing qualification, NOT accuracy.

Original phase1 native PASS is independently authenticated; its failed host
receipt/log remains failed. No datasets, models, GPU, geometry repair or adoption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_surface_qslim_qualify'
STAGE = 'world_reward_surface_qslim_runtime_qualification_v1'
BUDGET, OUTER_SECONDS = 300, 302
IMAGE = 'sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0'
BUILD_PINS = 'configs/surface_qslim_build_pins.json'
PHASE1_PINS = 'configs/surface_identity_qualification_pins.json'
PHASE1_LOG = 'results/surface-identity-real-loader-v1.log'
NAMES = ('fresh_curved_holed_patch', 'fresh_disjoint_holed_patches', 'boundary_excess_inapplicable')
HELPERS = ('infra/surface_qslim_qualify.py', 'infra/run_surface_qslim_qualify.sh',
    'infra/surface_identity_qualify.py', 'infra/surface_qslim_build.py', 'infra/surface_qslim.cpp',
    'infra/official_track1_pack_gate.py', 'infra/official_pack_geometry.py',
    'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/surface_identity.py',
    'src/world_reward/exact_triangle_predicates.py', 'src/world_reward/mesh_serialization.py', BUILD_PINS, PHASE1_PINS,
    'configs/official_pack_runtime_pins.json')


def modules():
    import surface_identity_qualify as q
    import mediapipe_cpu_runtime_verify as rt
    import official_track1_pack_gate as official
    import surface_qslim_build as build
    return q, rt, official, build


def equal(actual, expected, message):
    _, rt, _, _ = modules()
    rt.require(type(actual) is dict and all(type(actual.get(k)) is type(v) and actual[k] == v
                                          for k, v in expected.items()), message)


def phase1(root, code, *, historical=False):
    q, rt, _, _ = modules()
    pinpath = code / PHASE1_PINS; pinid = rt.identity(pinpath, 16384)
    pins = rt.strict(pinpath.read_bytes())
    expected = dict(schema='world_reward.surface_identity_qualification_pins.v1', historical_host_status='fail',
        native_loader_calls=6, official_budget_calls=3, QEM_calls=0, independent_native_audit=True,
        whole_Parquet_qualified=False, adoption=False, reconstruction_accuracy_verified=False)
    rt.require(set(pins) == set(expected) | {'producer_revision', 'native', 'historical_log',
        'source_archive_sha256', 'source_readonly_ledger_sha256', 'source_files', 'fixture_manifest_sha256'}, 'Exact phase1 pin schema required')
    equal(pins, expected, 'Original phase1 scope differs')
    rev = pins['producer_revision']; rt.require(type(rev) is str and re.fullmatch('[0-9a-f]{40}', rev), 'Original phase1 revision required')
    path = root / 'results' / ('surface-identity-qualify-' + rev) / 'native.json'
    report = rt.pinned(path, pins['native'])
    equal(report, dict(stage=q.STAGE, status='pass', phase='complete', producer_revision=rev,
        source_runtime_rehashed_after=True, owned_scratch_removed=True, GPU_used=False,
        datasets_read=False, models_read=False, private_values_read=False, body_model_constructed=False,
        adoption=False, reconstruction_accuracy_verified=False), 'Actual original native qualification incomplete')
    c = report['controls']
    rt.require(c['fixture_manifest_sha256'] == pins['fixture_manifest_sha256']
        and hashlib.sha256(json.dumps(c['fixture_manifest'],sort_keys=True).encode()).hexdigest()==pins['fixture_manifest_sha256'] and len(c['fixture_manifest']) == 6
        and c['all_fixture_sources_frozen_before_measurement'] is True and c['all_fixture_sources_rehashed_after'] is True
        and c['native_loader_calls_including_authority_replays'] == 6 and c['counts']['budget_returns'] == 3
        and c['QEM_calls'] == 0 and c['negative_GLBS_written'] is False
        and [r['name'] for r in c['records']] == list(q.NAMES)
        and all(r['meaningful_faces_preserved'] is True and r['component_boundary_geometry_exact'] is True for r in c['records']), 'Phase1 full fixed controls missing')
    result = dict(pins=pinid, native=pins['native'], producer_revision=rev, original_host_status='fail')
    if historical:
        oldcode = root / 'jobs' / rev / q.ENTRY / 'code'
        before = q.host_proof(root, oldcode, rev)
        rt.require(before == report['source_proof'], 'Complete original phase1 source/runtime proof differs')
        ledger, count = hashlib.sha256(), 0
        for p in sorted(oldcode.rglob('*')):
            s = p.lstat()
            if p.is_dir(): rt.require(stat.S_IMODE(s.st_mode) == 0o555, 'Historical source directory mode differs'); continue
            rt.require(stat.S_IMODE(s.st_mode) in (0o444, 0o555), 'Historical source file mode differs')
            identity = rt.identity(p, 2 << 20, empty=True); count += 1
            ledger.update(p.relative_to(oldcode).as_posix().encode() + b'\0' + str(stat.S_IMODE(s.st_mode)).encode() + b'\0' + bytes.fromhex(identity['sha256']))
        rt.require(count == pins['source_files'] and ledger.hexdigest() == pins['source_readonly_ledger_sha256']
            and (oldcode.parent / 'source-sha256').read_text().strip() == pins['source_archive_sha256'], 'Historical complete source ledger differs')
        rt.require(rt.identity(root / PHASE1_LOG, readonly=False) == pins['historical_log'], 'Original failed host log changed')
        result['historical_source_verified'] = True
    return result


def build_proof(root, code):
    _, rt, _, build = modules()
    pinpath = code / BUILD_PINS; pinid = rt.identity(pinpath, 16384); pins = rt.strict(pinpath.read_bytes())
    rt.require(type(pins) is dict and set(pins) == {'schema', 'producer_revision', 'report', 'native', 'binary', 'source_cpp'}
        and pins['schema'] == 'world_reward.surface_qslim_build_pins.v1'
        and re.fullmatch('[0-9a-f]{40}', str(pins['producer_revision'])), 'Independent actual build pins required')
    directory = root / 'results' / ('surface-qslim-build-' + pins['producer_revision'])
    rt.require(stat.S_IMODE(directory.stat().st_mode) == 0o555, 'Retained build namespace must be sealed')
    host = rt.pinned(directory / 'report.json', pins['report'], 2 << 20)
    native = rt.pinned(directory / 'native.json', pins['native'], 2 << 20)
    equal(host, dict(stage='surface_qslim_build_host_v1', status='pass', phase='complete',
        producer_revision=pins['producer_revision'], source_rehashed_after=True, runtime_rehashed_after=True,
        owned_container_removed=True, owned_scratch_removed=True, qem_calls=0, gpu_used=False,
        native_backend_qualified=False, adopted=False), 'Complete actual build-only host receipt required')
    build.validate_native(native, host['source_binding'])
    rt.require(host['native_identity'] == pins['native'] and host['retained_binary'] == pins['binary']
        and native['build']['binary'] == pins['binary'] and native['build']['source'] == pins['source_cpp']
        and rt.identity(code / 'infra/surface_qslim.cpp') == pins['source_cpp'], 'Compiled source/binary does not match current original source')
    binary = directory / 'surface_qslim'
    rt.require(stat.S_IMODE(binary.stat().st_mode) == 0o555 and rt.identity(binary, 32 << 20) == pins['binary'], 'Retained ELF differs')
    return binary, dict(pins=pinid, report=pins['report'], native=pins['native'], binary=pins['binary'], source_cpp=pins['source_cpp'])


def runtime_proof(root, code, revision):
    q, rt, official, _ = modules()
    binding = rt.source(root, code, revision, ENTRY, HELPERS)
    rt.require({p.name for p in code.parent.iterdir()} == {'code', 'revision', 'source-sha256'}, 'Own dispatch namespace differs')
    runtime = official.load_runtime(root, code)
    rt.require(runtime['pins']['image_id'] == IMAGE, 'Original actual official CPU runtime required')
    _, build = build_proof(root, code)
    return dict(source_binding=binding, runtime=runtime, official_sources=official.official_sources(root),
                native_mesh_sources=official.native_mesh_sources(root)[1], build=build, phase1=phase1(root, code))


def host_proof(root, code, revision):
    return dict(native=runtime_proof(root, code, revision), phase1_history=phase1(root, code, historical=True))


def mount_paths(root, code):
    q, rt, _, _ = modules(); pins = rt.strict((code / BUILD_PINS).read_bytes())
    first = rt.strict((code / PHASE1_PINS).read_bytes())
    build = root / 'results' / ('surface-qslim-build-' + pins['producer_revision'])
    return [*q.mount_paths(root, code), build,
            root / 'results' / ('surface-identity-qualify-' + first['producer_revision']) / 'native.json']


def fixtures(np):
    def patch(n, a, b, translation=0):
        cells = [(i, j) for i in range(n) for j in range(n) if not (a <= i < b and a <= j < b)]
        ids = sorted({p for i, j in cells for p in ((i,j),(i+1,j),(i+1,j+1),(i,j+1))}); lookup = {p:k for k,p in enumerate(ids)}
        v = np.array([((i-n/2)/32+translation, (j-n/2)/32, (((i-n/2)/32)**2+((j-n/2)/32)**2)/16) for i,j in ids], np.float32)
        f = np.array([(lookup[p],lookup[q],lookup[r]) for i,j in cells for p,q,r in
            (((i,j),(i+1,j),(i+1,j+1)),((i,j),(i+1,j+1),(i,j+1)))], np.int64)
        return v,f
    a, af = patch(64,24,40); b,bf=patch(48,20,28); c,cf=patch(48,20,28,4)
    s=np.arange(512)/256
    rim=np.concatenate([np.column_stack(pair) for pair in ((-1+s,-np.ones(512)),(np.ones(512),-1+s),(1-s,np.ones(512)),(-np.ones(512),1-s))])
    v=np.vstack((np.column_stack((rim,np.zeros(2048))),np.column_stack((rim,np.ones(2048))))).astype(np.float32)
    f=np.array([t for i in range(2048) for t in ((i,(i+1)%2048,(i+1)%2048+2048),(i,(i+1)%2048+2048,i+2048))],np.int64)
    return tuple(zip(NAMES, ((a,af),(np.vstack((b,c)),np.vstack((bf,cf+len(b)))),(v,f))))


def topology(v, f):
    import numpy as np
    from world_reward.surface_identity import _topology
    from world_reward.exact_triangle_predicates import validate_exact_triangle_non_degeneracy
    validate_exact_triangle_non_degeneracy(v.astype(np.float64),f)
    keys, labels, loops, components, stats, _, _ = _topology(f,len(v))
    return SimpleNamespace(vertices=v,faces=f,component_keys=keys,face_components=labels,
        boundary_loops=loops,boundary_components=components,stats=stats)


def orientation(before,after):
    from fractions import Fraction
    def normal(t):
        p=[[Fraction(float(x)) for x in row] for row in t]
        a=[p[1][k]-p[0][k] for k in range(3)];b=[p[2][k]-p[0][k] for k in range(3)]
        return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    a,b=normal(before),normal(after)
    if sum(x*y for x,y in zip(a,b))<=0:raise ValueError('Exact triangle orientation reversed/collapsed')


def verify_mapping(v, f, u, g, mapping):
    import numpy as np
    _, rt, _, build = modules()
    def ints(value, length):
        rt.require(type(value) is list and len(value)==length and all(type(n) is int and -(1<<63)<=n<(1<<63) for n in value), 'Exact complete integer lineage required'); return np.array(value,np.int64)
    equal(mapping,dict(schema='surface-qslim-mapping-v1',source_sha256=rt.identity(Path(__file__).with_name('surface_qslim.cpp'),readonly=False)['sha256'],
        source_vertices=len(v),source_faces=len(f),output_vertices=len(u),output_faces=len(g),target_vertices=4096,target_faces=4096,
        boundary_policy='fixed_original_vertices',intersection_blocking='upstream_floating_point',initial_embedding_certified=False,
        volume_or_closure_required=False,serialization_qualification_completed=False,adoption=False), 'Native lineage scope differs')
    rt.require(u.dtype==np.float64 and g.dtype==np.int64 and u.shape[1:]==g.shape[1:]==(3,)
        and 0<len(g)<=4096 and 3<=len(u)<=4096 and np.isfinite(u).all() and np.min(g)>=0 and np.max(g)<len(u), 'Native output arrays invalid')
    source,candidate=topology(v,f),topology(u,g)
    boundary_vertices={n for loop in source.boundary_loops for n in loop}
    j=ints(mapping['J'],len(g)); i=ints(mapping['I'],len(u)); supplied=ints(mapping['original_vertex_to_output'],len(v))
    rt.require(np.min(j)>=0 and np.max(j)<len(f) and len(np.unique(j))==len(j)
        and np.min(i)>=0 and np.max(i)<len(v) and len(np.unique(i))==len(i), 'Face/vertex survivor birth range differs')
    rt.require(type(mapping['ledger']) is list and all(type(mapping[key]) is int and mapping[key]>=0 for key in ('native_attempts','native_failed','committed_collapses','veto_link','veto_geometry','veto_component','veto_intersection')),'Exact native call-counter types required')
    parent=np.arange(len(v)); live=f.copy(); positions=v.astype(np.float64).copy(); removed=set()
    for row in mapping['ledger']:
        rt.require(type(row) is dict and set(row)=={'survivor','removed_vertex','placement','removed_faces'}, 'Exact committed-event schema required')
        s,d=row['survivor'],row['removed_vertex']; p=np.asarray(row['placement'])
        rt.require(type(s) is int and type(d) is int and 0<=s<d<len(v) and parent[s]==s and parent[d]==d
            and s not in boundary_vertices and d not in boundary_vertices
            and p.shape==(3,) and p.dtype.kind in 'fi' and np.isfinite(p).all(), 'Committed contraction invalid')
        ids=ints(row['removed_faces'],2)
        expected=set(np.flatnonzero(np.any(live==s,axis=1)&np.any(live==d,axis=1)))-removed
        rt.require(set(ids.tolist())==expected and len(expected)==2, 'Committed removed faces differ from edge incidence')
        neighbors=[]
        for vertex in (s,d):
            active_rows=np.array([fid not in removed for fid in range(len(live))])
            ring=live[active_rows & np.any(live==vertex,axis=1)];neighbors.append(set(ring.ravel().tolist())-{vertex})
        opposite={int(n) for fid in expected for n in live[fid] if n not in (s,d)}
        rt.require(neighbors[0]&neighbors[1]==opposite and len(opposite)==2,'Committed edge violates real interior link condition')
        affected=set(np.flatnonzero(np.any(live==s,axis=1)|np.any(live==d,axis=1)))-removed-expected
        for fid in affected:
            old=positions[live[fid]].copy();new=old.copy();new[np.isin(live[fid],[s,d])]=p
            orientation(old,new);orientation(old.astype(np.float32),new.astype(np.float32));orientation(new,new.astype(np.float32))
        removed.update(expected); parent[d]=s; positions[s]=positions[d]=p; live[live==d]=s
    def root(n):
        while parent[n]!=n:n=parent[n]
        return n
    survivors={root(int(birth)):k for k,birth in enumerate(i)}
    rt.require(len(survivors)==len(i) and all(root(int(birth))==birth for birth in i), 'I is not surviving original birth')
    replay=np.array([survivors.get(root(n),-1) for n in range(len(v))],np.int64)
    rt.require(mapping['native_attempts']==mapping['native_failed']+mapping['committed_collapses']
        and mapping['native_failed']>=sum(mapping[key] for key in ('veto_link','veto_geometry','veto_component','veto_intersection'))
        and mapping['committed_collapses']==len(v)-len(u), 'Native committed/veto call census differs')
    rt.require(np.array_equal(replay,supplied) and np.min(replay)>=0 and np.array_equal(g,replay[f[j]])
        and np.array_equal(u,positions[i]) and set(j.tolist())==set(range(len(f)))-removed
        and mapping['committed_collapses']==len(mapping['ledger'])>0, 'Whole oriented J/I/quotient/placement replay differs')
    correspond=ints(mapping['candidate_component_to_source'],len(candidate.component_keys))
    rt.require(len(source.component_keys)==len(candidate.component_keys) and sorted(correspond.tolist())==list(range(len(source.component_keys))), 'Component bijection differs')
    for c,sc in enumerate(correspond):
        rt.require(np.all(source.face_components[j[candidate.face_components==c]]==sc)
            and candidate.stats[c]['euler_characteristic']==source.stats[sc]['euler_characteristic'], 'Component/Euler lineage changed')
    original_edges={(int(a),int(b)) for loop in source.boundary_loops for a,b in zip(loop,(*loop[1:],loop[0]))}
    actual_edges={(int(i[a]),int(i[b])) for loop in candidate.boundary_loops for a,b in zip(loop,(*loop[1:],loop[0]))}
    rt.require(original_edges==actual_edges and all(u[k].tobytes()==v[b].astype(np.float64).tobytes()
        for k,b in enumerate(i) if any(b in loop for loop in source.boundary_loops)), 'Fixed boundary orientation/coordinate bytes changed')
    return source,candidate


def binary_runtime(binary, source_sha, remaining):
    _, rt, _, build=modules(); before=rt.identity(binary,32<<20)
    with binary.open('rb') as stream: header=stream.read(20)
    rt.require(header[:6]==b'\x7fELF\x02\x01' and int.from_bytes(header[18:20],'little')==62, 'Actual ELF64 little-endian x86_64 required')
    deps=subprocess.run(['ldd',str(binary)],capture_output=True,timeout=min(5,remaining()))
    rt.require(deps.returncode==0 and b'not found' not in deps.stdout+deps.stderr, 'Retained binary ABI dependencies unavailable; no install/fallback')
    result=subprocess.run([str(binary),'--build-info'],capture_output=True,timeout=min(5,remaining()))
    rt.require(result.returncode==0,'Retained binary not executable in original official CPU runtime')
    info=rt.strict(result.stdout); equal(info,build.expected_info(source_sha),'Actual retained build-info differs')
    rt.require(rt.identity(binary,32<<20)==before,'Retained binary changed')
    return dict(ELF64_x86_64=True,dependencies_resolved=True,build_info=info,qem_calls=0)


def run_native(binary, argv, remaining, counts, kind, expected=0):
    _,rt,_,_=modules(); counts[kind+'_attempts']+=1
    result=subprocess.run([str(binary),*map(str,argv)],capture_output=True,timeout=remaining())
    counts[kind+'_returns']+=1
    if result.returncode!=expected:
        raise ValueError('Native '+kind+' exit '+str(result.returncode)+': '+result.stderr[-300:].decode(errors='replace'))
    return result


def write_obj(path,v,f):
    with path.open('x') as stream:
        for row in v:stream.write('v '+' '.join(format(float(x),'.17g') for x in row)+'\n')
        for row in f:stream.write('f '+' '.join(str(int(x)+1) for x in row)+'\n')
    path.chmod(0o444)


def read_obj(path):
    import numpy as np
    v,f=[],[]
    for line in path.read_text().splitlines():
        row=line.split()
        if row[0]=='v' and len(row)==4:v.append([float(x) for x in row[1:]])
        elif row[0]=='f' and len(row)==4:f.append([int(x)-1 for x in row[1:]])
        else:raise ValueError('Malformed native OBJ output')
    return np.array(v,np.float64),np.array(f,np.int64)


def controls(np,trimesh,budget_mesh,load_native,authorities,binary,scratch,remaining,progress):
    q,rt,_,_=modules(); rows=[(name,*(np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape) for a in pair)) for name,pair in fixtures(np)]
    manifest=q._array_manifest(rows); progress.update(fixture_manifest=manifest,fixture_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),all_sources_frozen_before_measurement=True,records=[])
    artifacts={}
    counts=dict(preflight_attempts=0,preflight_returns=0,qem_attempts=0,qem_returns=0,native_loader_attempts=0,native_loader_returns=0,authority_attempts=0,authority_returns=0,budget_attempts=0,budget_returns=0);progress['counts']=counts
    try:
        for index,(name,v,f) in enumerate(rows):
            progress.update(current_case=name,current_phase='source_preflight');remaining();s=topology(v,f)
            wanted=((4000,7680,1,2,320),(4704,8960,2,4,448),(4096,4096,1,2,4096))[index]
            rt.require((len(v),len(f),len(s.component_keys),len(s.boundary_loops),sum(map(len,s.boundary_loops)))==wanted,'Fixed procedural source counts differ')
            path=scratch/(name+'.obj');write_obj(path,v,f);artifacts[path]=rt.identity(path,32<<20)
            pre=run_native(binary,['--preflight',path],remaining,counts,'preflight',4 if index==2 else 0)
            if index<2:
                equal(rt.strict(pre.stdout),dict(source_vertices=len(v),source_faces=len(f),components=wanted[2],boundary_vertices=wanted[4],unused_vertices=0,float32_triangles_exactly_active=True,oriented_vertex_manifold=True,volume_or_closure_required=False,qem_calls=0,adoption=False),'Native preflight did not verify the complete authored source')
            else:
                rt.require(pre.stderr.strip()==b'INAPPLICABLE: Fixed referenced boundary vertices reach the phase2 target','Negative did not fail the exact pre-QEM policy')
        for name,v,f in rows[:2]:
            progress.update(current_case=name,current_phase='QEM');obj=scratch/(name+'.lod.obj');mapping=scratch/(name+'.json')
            run_native(binary,[scratch/(name+'.obj'),obj,mapping],remaining,counts,'qem')
            for artifact in (obj,mapping):rt.identity(artifact,32<<20,readonly=False);artifact.chmod(0o444);artifacts[artifact]=rt.identity(artifact,32<<20)
            u,g=read_obj(obj);m=rt.strict(mapping.read_bytes());source,candidate=verify_mapping(v,f,u,g,m)
            for fid in range(len(g)):orientation(u[g[fid]],u[g[fid]].astype(np.float32))
            path=scratch/(name+'.glb');trimesh.Trimesh(vertices=u,faces=g,process=False).export(path);path.chmod(0o444);artifacts[path]=rt.identity(path,32<<20)
            progress['current_phase']='original_native_loader';counts['native_loader_attempts']+=1;nv,nf=load_native(path);counts['native_loader_returns']+=1
            counts['authority_attempts']+=1;rv,rf,proof=authorities(path,SimpleNamespace(object_vertices=nv,object_faces=nf));counts['authority_returns']+=1
            rt.require(proof['original_native_FP32_loader_replayed'] is True and proof['native_geometry_byte_exact'] is True and proof['full_scene_instances_verified'] is True and proof['model_imports'] is False,'Actual original complete-scene/native authority required')
            counts['budget_attempts']+=1;pv,pf=budget_mesh(str(path),faces=4096,vertices=4096);counts['budget_returns']+=1
            from official_pack_geometry import canonical_oriented_triangles,nonzero_triangle_mask,verify_exact_dual_surfaces
            from world_reward.surface_identity import SurfaceIdentity
            exact=verify_exact_dual_surfaces(nv,nf,rv,rf,pv,pf)
            from world_reward.mesh_serialization import serialization_preflight
            weld=dict(serialization_preflight(u,g));rt.require(weld['serialized_triangles_numerically_preserved_by_weld'] is True and weld['welded_triangles_exactly_active'] is True,'Default8 changed represented LOD surface')
            stored=u.astype(np.float32);rt.require(np.array_equal(canonical_oriented_triangles(stored,g),canonical_oriented_triangles(nv,nf)),'LOD nativeF32 surface differs')
            active=nonzero_triangle_mask(pv,pf);rt.require(pv.shape==pf.shape==(4096,3) and np.all(pf[~active]==0),'Only official all-zero padding allowed')
            ids,inverse=np.unique(pf[active],return_inverse=True);packed=SurfaceIdentity(pv[ids],inverse.reshape(-1,3).astype(np.int64));serialized=SurfaceIdentity(stored,g)
            rt.require(q.surface_signature(serialized)==q.surface_signature(packed),'Full serialized component/boundary geometry differs after original default8 packing')
            diagnostic=[]
            for c,sc in enumerate(m['candidate_component_to_source']):
                sv=np.unique(f[source.face_components==sc]);uv=np.unique(g[candidate.face_components==c]);
                sm=trimesh.Trimesh(vertices=v,faces=f[source.face_components==sc],process=False);cm=trimesh.Trimesh(vertices=u,faces=g[candidate.face_components==c],process=False)
                distances=[]
                for mesh,points in ((cm,v[sv]),(sm,u[uv])):
                    for start in range(0,len(points),128):remaining();distances.extend(trimesh.proximity.closest_point(mesh,points[start:start+128])[1].tolist())
                bv=sorted({n for loop,lc in zip(source.boundary_loops,source.boundary_components) if lc==sc for n in loop})
                boundary_d=[]
                for start in range(0,len(bv),128):remaining();boundary_d.extend(trimesh.proximity.closest_point(cm,v[bv[start:start+128]])[1].tolist())
                diagnostic.append(dict(source_component=sc,vertices_tested=len(sv)+len(uv),boundary_vertices_tested=len(bv),mean_unsigned_vertex_surface_distance=float(np.mean(distances)),max_unsigned_vertex_surface_distance=float(np.max(distances)),max_boundary_vertex_to_surface_distance=float(max(boundary_d,default=0))))
            progress['records'].append(dict(name=name,source_vertices=len(v),source_faces=len(f),output_vertices=len(u),output_faces=len(g),committed_collapses=m['committed_collapses'],mapping_sha256=rt.identity(mapping,32<<20,readonly=False)['sha256'],oriented_birth_quotient_replay_exact=True,fixed_boundary_geometry_exact=True,serialized_packed_components_boundaries_exact=True,exact_dual_surfaces=exact,default8_serialization=weld,float64_to_float32_quantized_vertex_count=int(np.count_nonzero(np.any(u!=stored.astype(np.float64),axis=1))),quantization_orientation_exact=True,vertex_distance_diagnostic_only=True,Hausdorff_or_full_surface_certified=False,per_component_vertex_diagnostics=diagnostic))
            remaining()
        progress.update(current_phase='complete',QEM_calls=counts['qem_returns'],native_loader_calls_including_authority_replays=counts['native_loader_returns']+counts['authority_returns'],negative_QEM_calls=0,negative_GLBS_written=False)
    finally:
        rt.require(q._array_manifest(rows)==manifest,'Complete authored sources changed');progress['all_sources_rehashed_after']=True
        rt.require(all(rt.identity(path,32<<20)==pin for path,pin in artifacts.items()),'Created input/output artifacts changed')
        progress['created_artifacts_rehashed_after']=True;progress['created_artifacts']={path.name:pin for path,pin in artifacts.items()}
    return progress


def output(root, revision):
    return root / 'results' / ('surface-qslim-qualify-' + revision)


def native(root,code,revision):
    started=time.monotonic();q,rt,official,_=modules();out=output(root,revision)
    def remaining():
        left=BUDGET-(time.monotonic()-started)
        if left<=0:raise TimeoutError('Inclusive300s QSlim qualification exhausted')
        return left
    def timeout(*_):raise TimeoutError('Inclusive300s QSlim qualification exhausted')
    old={sig:signal.signal(sig,timeout) for sig in (signal.SIGALRM,signal.SIGTERM)};signal.alarm(BUDGET)
    report=dict(stage=STAGE,status='fail',phase='preflight',producer_revision=revision,budget_seconds=BUDGET,
        GPU_used=False,datasets_read=False,models_read=False,private_values_read=False,
        reconstruction_accuracy_verified=False,adoption=False,positive_QEM_calls_maximum=2)
    before=None;scratch=None
    try:
        rt.require(sys.platform=='linux' and os.geteuid()==1000 and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
            and os.environ.get('WR_IMAGE_ID')==IMAGE and rt.canonical(out)==out and out.is_dir()
            and not list(out.iterdir()) and out.stat().st_uid==1000 and stat.S_IMODE(out.stat().st_mode)==0o755,
            'Original restricted offline CPU and empty reserved output required')
        before=runtime_proof(root,code,revision);report['source_proof']=before
        report['runtime_versions']=official.check_runtime_packages(before['runtime']['pins']['versions'])
        binary,_=build_proof(root,code);report['binary_runtime']=binary_runtime(binary,before['build']['source_cpp']['sha256'],remaining)
        import numpy as np
        import trimesh
        import rtree  # Existing native distance dependency; no install/fallback after QEM.
        report['vertex_distance_dependency_version']=rtree.__version__
        native_root,_=official.native_mesh_sources(root);scope=dict(np=np,trimesh=trimesh,Path=Path)
        scope['load_object_mesh']=official.isolated_source_function(native_root/'lib_mhr/contact.py','load_object_mesh',scope)
        load=official.isolated_source_function(native_root/'learning/training/mhr_opt_refineout.py','_load_object_vertices',scope)
        if any(n=='v2dlb' or n.startswith('v2dlb.') for n in sys.modules):raise ValueError('Fresh official namespace required')
        kit=root/'vendor/v2d_submission_kit';sys.path.insert(0,str(kit))
        from v2dlb.mesh_budget import budget_mesh
        rt.require(Path(budget_mesh.__code__.co_filename)==kit/'v2dlb/mesh_budget.py','Original unmodified official helper required')
        report.update(phase='controls',controls={})
        with tempfile.TemporaryDirectory(prefix='wr-surface-qslim-',dir='/tmp') as temporary:
            scratch=Path(temporary)
            controls(np,trimesh,budget_mesh,load,lambda p,ep:official.load_mesh_authorities(root,p,ep),
                     binary,scratch,remaining,report['controls'])
        official.check_runtime_packages(report['runtime_versions']);remaining();report.update(status='pass',phase='complete')
    except Exception as error:
        report.update(status='fail',error_type=type(error).__name__,error=str(error)[:400])
    finally:
        report['owned_scratch_removed']=scratch is None or not scratch.exists()
        try:
            remaining();rt.require(before is not None and runtime_proof(root,code,revision)==before,'Sources/build/runtime changed')
            report['source_runtime_rehashed_after']=True;remaining()
        except Exception as error:report.update(status='fail',postflight_error_type=type(error).__name__)
        report['elapsed_seconds']=time.monotonic()-started
        if report['elapsed_seconds']>BUDGET:report.update(status='fail',budget_exhausted=True)
        try:q.write(out/'native.json',report);remaining()
        finally:
            signal.alarm(0)
            for sig,handler in old.items():signal.signal(sig,handler)
    return 0 if report['status']=='pass' else 1


def container_absence(root,revision,failed=False):
    _,rt,_,_=modules();path=Path(str(output(root,revision))+'.container.cid')
    if failed and not path.exists() and not path.is_symlink():return dict(verified=False,CID_identity=None)
    pin=rt.identity(path,65,readonly=False);rt.require(path.stat().st_uid==0 and stat.S_IMODE(path.stat().st_mode)==0o400,'Actual sealed owned host CID required')
    raw=path.read_bytes();rt.require(re.fullmatch(b'[0-9a-f]{64}\n?',raw),'Exact owned CID required');cid=raw.decode().strip()
    result=subprocess.run(['docker','inspect',cid,'--format','{{.Id}}'],capture_output=True,timeout=5)
    absent=(f'Error: No such object: {cid}',f'error: no such object: {cid}',f'Error: No such container: {cid}',f'Error response from daemon: No such container: {cid}')
    rt.require(result.returncode==1 and result.stdout.strip() in (b'',b'[]') and result.stderr.decode().strip() in absent
        and rt.identity(path,65,readonly=False)==pin,'Independent actual CID absence required; daemon failures are not absence')
    return dict(verified=True,CID_identity=pin)


def validate_native(report,proof,revision):
    equal(report,dict(stage=STAGE,status='pass',phase='complete',producer_revision=revision,source_proof=proof,
        source_runtime_rehashed_after=True,owned_scratch_removed=True,GPU_used=False,datasets_read=False,models_read=False,
        private_values_read=False,reconstruction_accuracy_verified=False,adoption=False),'Actual complete native qualification scope required')
    _,rt,_,_=modules();c=report['controls']
    rt.require(type(report['elapsed_seconds']) in (float,int) and 0<report['elapsed_seconds']<=BUDGET
        and c['all_sources_frozen_before_measurement'] is True and c['all_sources_rehashed_after'] is True and c['created_artifacts_rehashed_after'] is True
        and c['fixture_manifest_sha256']==hashlib.sha256(json.dumps(c['fixture_manifest'],sort_keys=True).encode()).hexdigest()
        and len(c['fixture_manifest'])==3 and [r['name'] for r in c['records']]==list(NAMES[:2])
        and c['QEM_calls']==2 and c['negative_QEM_calls']==0 and c['negative_GLBS_written'] is False
        and all(type(value) is int for value in c['counts'].values())
        and c['counts']==dict(preflight_attempts=3,preflight_returns=3,qem_attempts=2,qem_returns=2,
            native_loader_attempts=2,native_loader_returns=2,authority_attempts=2,authority_returns=2,budget_attempts=2,budget_returns=2)
        and c['native_loader_calls_including_authority_replays']==4
        and all(r['oriented_birth_quotient_replay_exact'] is True and r['fixed_boundary_geometry_exact'] is True
            and r['serialized_packed_components_boundaries_exact'] is True and r['quantization_orientation_exact'] is True
            and r['Hausdorff_or_full_surface_certified'] is False for r in c['records']), 'All fixed cases/native calls/serialized geometry required')


def seal(root,code,revision,before,status,cleanup_verified):
    q,rt,_,_=modules();out=output(root,revision);proof=rt.strict(before.encode())
    absence=dict(verified=False,CID_identity=None);raw=None;error=None;post=False;good=False
    try:
        rt.require(host_proof(root,code,revision)==proof,'Host complete source/oldfailure/build changed');post=True
        absence=container_absence(root,revision,failed=status!=0)
        path=out/'native.json';raw=rt.strict(path.read_bytes()) if path.exists() else None
        good=status==0 and cleanup_verified and absence['verified'] and type(raw) is dict
        if good:validate_native(raw,proof['native'],revision)
    except Exception as failure:error=type(failure).__name__;good=False
    rt.require({p.name for p in out.iterdir()}<= {'native.json'},'Geometry or extra files escaped owned scratch')
    path=out/'native.json'
    report=dict(stage=STAGE+'_host',status='pass' if good else 'fail',producer_revision=revision,
        source_proof=proof,native_report=raw,native_identity=rt.identity(path,2<<20) if path.exists() else None,
        source_runtime_rehashed_after=post,owned_container_removed=absence['verified'],container_absence=absence,
        native_budget_seconds=BUDGET,docker_outer_seconds=OUTER_SECONDS,
        budget_scope='300s inclusive native; bounded host source/daemon/cleanup separate',
        GPU_used=False,datasets_read=False,models_read=False,reconstruction_accuracy_verified=False,adoption=False)
    if error:report['host_failure_type']=error
    q.write(out/'report.json',report)
    rt.require({p.name for p in out.iterdir()}==({'native.json','report.json'} if path.exists() else {'report.json'})
        and all(stat.S_IMODE(p.stat().st_mode)==0o444 for p in out.iterdir()),'Only readonly actual receipts may remain')
    out.chmod(0o555)
    return 0 if good else (status or 1)


def main():
    code=Path(os.environ['WR_CODE']);sys.path[:0]=[str(code/'infra'),str(code/'src')]
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('action',choices=('proof','mounts','native','seal'))
    parser.add_argument('--before');parser.add_argument('--status',type=int);parser.add_argument('--cleanup-verified',type=int,choices=(0,1))
    args=parser.parse_args();root=Path(os.environ['WR_ROOT']);rev=os.environ['WR_CODE_REVISION']
    _,rt,_,_=modules();rt.require(root==ROOT and re.fullmatch('[0-9a-f]{40}',rev)
        and code==root/'jobs'/rev/ENTRY/'code' and Path(__file__).resolve()==code/'infra/surface_qslim_qualify.py','Own actual source entry required')
    if args.action=='proof':print(json.dumps(host_proof(root,code,rev),sort_keys=True));return 0
    if args.action=='mounts':print('\n'.join(map(str,mount_paths(root,code))));return 0
    if args.action=='native':return native(root,code,rev)
    if args.before is None or args.status is None or not 0<=args.status<=255 or args.cleanup_verified is None:parser.error('Actual before/status/cleanup required')
    return seal(root,code,rev,args.before,args.status,args.cleanup_verified==1)


if __name__=='__main__':raise SystemExit(main())
