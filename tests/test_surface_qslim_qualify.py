"""Data-free Python/stub contracts only; no native ELF/official/QEM execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def q(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'infra'))
    spec=importlib.util.spec_from_file_location('surface_qslim_qualify_test',ROOT/'infra/surface_qslim_qualify.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_fixed_sources_full_arrays_manifest_before_measurement(q):
    rows=q.fixtures(np)
    assert [n for n,_ in rows]==list(q.NAMES)
    assert [(len(v),len(f)) for _,(v,f) in rows]==[(4000,7680),(4704,8960),(4096,4096)]
    for n,(v,f) in rows:
        assert v.dtype==np.float32 and f.dtype==np.int64
        s=q.topology(v,f)
        assert len(s.boundary_loops)==(4 if n==q.NAMES[1] else 2)
    assert q.BUDGET==300 and q.OUTER_SECONDS==302
    text=(ROOT/'infra/surface_qslim_qualify.py').read_text()
    assert text.index('manifest=q._array_manifest(rows)')<text.index("run_native(binary,['--preflight'")
    assert text.index('for index,(name,v,f) in enumerate(rows)')<text.index('for name,v,f in rows[:2]')


def tiny_valid_collapse(q):
    # Closed curved bipyramid, boundaryless link-valid ring contraction 1 -> 0.
    v=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)
    f=np.array([[4,0,1],[4,1,2],[4,2,3],[4,3,0],
                [5,1,0],[5,2,1],[5,3,2],[5,0,3]],np.int64)
    parent=np.array([0,0,2,3,4,5]);i=np.array([0,2,3,4,5]);index={int(b):k for k,b in enumerate(i)}
    full=np.array([index[int(b)] for b in parent]);j=np.array([1,2,3,5,6,7]);g=full[f[j]]
    u=v[i].astype(np.float64);u[0]=[.5,.5,0]
    source=q.topology(v,f);candidate=q.topology(u,g)
    mapping=dict(schema='surface-qslim-mapping-v1',source_sha256=hashlib.sha256((ROOT/'infra/surface_qslim.cpp').read_bytes()).hexdigest(),
        source_vertices=len(v),source_faces=len(f),output_vertices=len(u),output_faces=len(g),target_vertices=4096,target_faces=4096,
        boundary_policy='fixed_original_vertices',intersection_blocking='upstream_floating_point',initial_embedding_certified=False,
        volume_or_closure_required=False,serialization_qualification_completed=False,adoption=False,
        native_attempts=1,native_failed=0,veto_link=0,veto_geometry=0,veto_component=0,veto_intersection=0,
        J=j.tolist(),I=i.tolist(),original_vertex_to_output=full.tolist(),committed_collapses=1,
        candidate_component_to_source=[0],ledger=[dict(survivor=0,removed_vertex=1,placement=[.5,.5,0],removed_faces=[0,4])])
    return v,f,u,g,mapping


def test_independent_whole_oriented_birth_and_quotient_replay(q):
    v,f,u,g,m=tiny_valid_collapse(q)
    before=[a.tobytes() for a in (v,f,u,g)]
    source,candidate=q.verify_mapping(v,f,u,g,m)
    assert len(source.component_keys)==len(candidate.component_keys)==1
    assert [a.tobytes() for a in (v,f,u,g)]==before


@pytest.mark.parametrize('fault',['birth','face_order','quotient','position','removed','count','flip','bool_birth'])
def test_bad_native_ledger_or_geometry_abstains(q,fault):
    v,f,u,g,m=tiny_valid_collapse(q)
    if fault=='birth':m['I'][0]=1
    if fault=='face_order':g=g[::-1].copy()
    if fault=='quotient':m['original_vertex_to_output'][1]=1
    if fault=='position':u[0,0]=.6
    if fault=='removed':m['ledger'][0]['removed_faces']=[0,1]
    if fault=='count':m['native_attempts']=2
    if fault=='flip':g[0]=g[0,::-1]
    if fault=='bool_birth':m['J'][0]=True
    with pytest.raises(ValueError):q.verify_mapping(v,f,u,g,m)


def test_exact_orientation_rejects_f32_collapse_no_epsilon(q):
    t=np.array([[1.,0,0],[1+2**-25,0,0],[1,2**-25,0]],np.float64)
    with pytest.raises(ValueError):q.orientation(t,t.astype(np.float32))
    with pytest.raises(ValueError):q.orientation(t,t[::-1])


def test_binary_abi_proof_binds_elf_dependencies_and_original_info(q,monkeypatch,tmp_path):
    path=tmp_path/'surface_qslim';raw=bytearray(40);raw[:6]=b'\x7fELF\x02\x01';raw[18:20]=(62).to_bytes(2,'little')
    path.write_bytes(raw);path.chmod(0o555)
    _,_,_,build=q.modules();sha='a'*64;calls=[]
    def run(argv,**kwargs):
        calls.append(argv);assert kwargs['timeout']<=5
        return SimpleNamespace(returncode=0,stdout=b'libc => /lib/libc.so\n' if argv[0]=='ldd' else json.dumps(build.expected_info(sha)).encode(),stderr=b'')
    monkeypatch.setattr(q.subprocess,'run',run)
    report=q.binary_runtime(path,sha,lambda:100)
    assert report['qem_calls']==0 and len(calls)==2 and calls[-1][-1]=='--build-info'


@pytest.mark.parametrize('failure',['dependency','build_info','execution','bad_elf'])
def test_unsupported_binary_runtime_never_calls_qem(q,monkeypatch,tmp_path,failure):
    path=tmp_path/'surface_qslim';raw=bytearray(40);raw[:6]=b'\x7fELF\x02\x01';raw[18:20]=(62).to_bytes(2,'little')
    if failure=='bad_elf':raw[0]=0
    path.write_bytes(raw);path.chmod(0o555);calls=[]
    def run(argv,**kw):
        calls.append(argv)
        if argv[0]=='ldd':return SimpleNamespace(returncode=0,stdout=b'libbad => not found' if failure=='dependency' else b'ok',stderr=b'')
        return SimpleNamespace(returncode=1 if failure=='execution' else 0,stdout=b'{}',stderr=b'')
    monkeypatch.setattr(q.subprocess,'run',run)
    with pytest.raises(ValueError):q.binary_runtime(path,'a'*64,lambda:100)
    assert all('--preflight' not in a and not any(str(x).endswith('.obj') for x in a) for a in calls)


def test_native_call_counts_keep_failure_attempts_return_and_bound_tail(q,monkeypatch,tmp_path):
    counts=dict(qem_attempts=0,qem_returns=0)
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=3,stdout=b'',stderr=b'z'*1000))
    with pytest.raises(ValueError) as error:q.run_native(tmp_path/'binary',['a.obj','b.obj','m.json'],lambda:10,counts,'qem')
    assert counts==dict(qem_attempts=1,qem_returns=1) and len(str(error.value))<400


def test_missing_independent_build_pins_stop_before_binary(q,tmp_path):
    with pytest.raises(FileNotFoundError):q.build_proof(tmp_path,tmp_path)


def test_read_obj_plain_exact_arrays_and_malformed(q,tmp_path):
    p=tmp_path/'output.obj';p.write_text('v -0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    v,f=q.read_obj(p);assert v.dtype==np.float64 and f.dtype==np.int64 and np.signbit(v[0,0])
    p.write_text('v 1 2\n')
    with pytest.raises(ValueError):q.read_obj(p)


def test_real_namespace_and_narrow_mount_shell_no_models_dataset_gpu():
    path=ROOT/'infra/run_surface_qslim_qualify.sh';text=path.read_text()
    subprocess.run(['bash','-n',str(path)],check=True)
    for phrase in ('run_surface_qslim_qualify/code','--network none','--read-only','--cap-drop ALL',
                   '--user 1000:1000','--memory 16g','--cpus 4','302s docker run','noexec','CUDA_VISIBLE_DEVICES=-1'):
        assert phrase in text
    assert '--gpus' not in text and 'validation/' not in text and 'weights/' not in text
    assert 'No such object' in text and 'No such container' in text and '|| true' not in text


def test_quantization_is_exact_once_and_distance_is_not_hausdorff(q):
    text=(ROOT/'infra/surface_qslim_qualify.py').read_text()
    assert 'stored=u.astype(np.float32)' in text
    assert 'canonical_oriented_triangles(stored,g)' in text
    assert "'Hausdorff_or_full_surface_certified'] is False" in text
    assert 'serialized_triangles_numerically_preserved_by_weld' in text
    assert 'max_boundary_vertex_to_surface_distance' in text
    assert '1e-5' not in text and 'rtol' not in text and 'atol' not in text
    assert 'counts[\'native_loader_returns\']+counts[\'authority_returns\']' in text


def test_two_adjacent_contractions_exclude_deleted_rows_from_future_links(q):
    v,f,_,_,m=tiny_valid_collapse(q)
    i=np.array([0,2,3,5]);j=np.array([2,5,6,7]);parent=np.array([0,0,2,3,0,5]);index={int(b):k for k,b in enumerate(i)}
    quotient=np.array([index[int(b)] for b in parent]);g=quotient[f[j]];u=v[i].astype(np.float64);u[0]=[.25,.25,.5]
    m.update(I=i.tolist(),J=j.tolist(),original_vertex_to_output=quotient.tolist(),output_vertices=len(u),output_faces=len(g),native_attempts=2,committed_collapses=2)
    m['ledger'].append(dict(survivor=0,removed_vertex=4,placement=[.25,.25,.5],removed_faces=[1,3]))
    source,candidate=q.verify_mapping(v,f,u,g,m)
    assert source.stats[0]['euler_characteristic']==candidate.stats[0]['euler_characteristic']==2


def test_partial_controls_failure_keeps_frozen_manifest_attempts_and_no_qem(q,monkeypatch,tmp_path):
    progress={};calls=[]
    def fake(binary,argv,remaining,counts,kind,expected=0):
        counts[kind+'_attempts']+=1;counts[kind+'_returns']+=1;calls.append(kind)
        raise ValueError('manufactured first preflight failure')
    monkeypatch.setattr(q,'run_native',fake)
    with pytest.raises(ValueError,match='manufactured'):
        q.controls(np,None,None,None,None,tmp_path/'binary',tmp_path,lambda:100,progress)
    assert calls==['preflight'] and progress['all_sources_rehashed_after'] is True
    assert len(progress['fixture_manifest'])==3 and progress['counts']['qem_attempts']==0
    assert progress['current_phase']=='source_preflight' and progress['records']==[]


def test_native_deadline_does_not_retry_or_hide_attempt(q,monkeypatch,tmp_path):
    counts=dict(qem_attempts=0,qem_returns=0)
    def timeout(*a,**kw):raise subprocess.TimeoutExpired(a[0],kw['timeout'])
    monkeypatch.setattr(q.subprocess,'run',timeout)
    with pytest.raises(subprocess.TimeoutExpired):q.run_native(tmp_path/'binary',['i.obj','o.obj','m.json'],lambda:1,counts,'qem')
    assert counts==dict(qem_attempts=1,qem_returns=0)


def test_phase1_wrong_policy_stops_before_original_runtime_read(q,tmp_path):
    config=tmp_path/q.PHASE1_PINS;config.parent.mkdir(parents=True)
    config.write_text(json.dumps(dict(schema='wrong',historical_host_status='pass')));config.chmod(0o444)
    with pytest.raises(ValueError,match='schema'):q.phase1(tmp_path,tmp_path)


def test_host_fail_keeps_original_status_and_actual_failed_scope(q,monkeypatch,tmp_path):
    out=q.output(tmp_path,'a'*40);out.mkdir(parents=True)
    monkeypatch.setattr(q,'host_proof',lambda *a: {'native': {'fake_fixture': True}})
    monkeypatch.setattr(q,'container_absence',lambda *a,**kw:dict(verified=True,CID_identity={'fixture':True}))
    status=q.seal(tmp_path,tmp_path,'a'*40,json.dumps({'native': {'fake_fixture':True}}),7,True)
    report=json.loads((out/'report.json').read_text())
    assert status==7 and report['status']=='fail' and report['native_report'] is None
    assert report['owned_container_removed'] is True and out.stat().st_mode & 0o777==0o555
    assert (out/'report.json').stat().st_mode & 0o777==0o444


def test_third_native_preflight_failure_stops_after_two_valid_preflights(q,monkeypatch,tmp_path):
    rows=q.fixtures(np); before=[(v.tobytes(),f.tobytes()) for _,(v,f) in rows]
    first=q.modules()[0]; manifest=first._array_manifest([(name,*pair) for name,pair in rows])
    progress={}; calls=[]; real_topology=q.topology
    monkeypatch.setattr(q,'fixtures',lambda _: rows)
    def checked_topology(v,f):
        assert progress['all_sources_frozen_before_measurement'] is True
        assert progress['fixture_manifest']==manifest
        assert not v.flags.writeable and not f.flags.writeable
        assert all(not np.shares_memory(v,pair[0]) and not np.shares_memory(f,pair[1]) for _,pair in rows)
        return real_topology(v,f)
    monkeypatch.setattr(q,'topology',checked_topology)
    binary=tmp_path/'manufactured-ELF-call-only'
    def native(argv,**kwargs):
        assert argv[:2]==[str(binary),'--preflight'] and len(argv)==3
        assert 0<kwargs['timeout']<=100 and kwargs['capture_output'] is True
        index=len(calls); calls.append(argv)
        assert Path(argv[2]).name==q.NAMES[index]+'.obj'
        if index==2:
            return SimpleNamespace(returncode=2,stdout=b'',stderr=b'FAIL: manufactured unexpected negative preflight')
        nv,nf,nc,nb=((4000,7680,1,320),(4704,8960,2,448))[index]
        report=dict(source_vertices=nv,source_faces=nf,components=nc,boundary_vertices=nb,
            unused_vertices=0,float32_triangles_exactly_active=True,oriented_vertex_manifold=True,
            volume_or_closure_required=False,qem_calls=0,adoption=False)
        return SimpleNamespace(returncode=0,stdout=json.dumps(report).encode(),stderr=b'')
    monkeypatch.setattr(q.subprocess,'run',native)
    with pytest.raises(ValueError,match='Native preflight exit 2'):
        q.controls(np,None,None,None,None,binary,tmp_path,lambda:100,progress)
    assert len(calls)==3 and progress['counts']==dict(preflight_attempts=3,preflight_returns=3,
        qem_attempts=0,qem_returns=0,native_loader_attempts=0,native_loader_returns=0,
        authority_attempts=0,authority_returns=0,budget_attempts=0,budget_returns=0)
    assert progress['current_case']==q.NAMES[2] and progress['current_phase']=='source_preflight'
    assert progress['records']==[] and progress['all_sources_rehashed_after'] is True
    assert progress['created_artifacts_rehashed_after'] is True
    assert set(progress['created_artifacts'])=={name+'.obj' for name in q.NAMES}
    assert {p.name for p in tmp_path.iterdir()}=={name+'.obj' for name in q.NAMES}
    assert before==[(v.tobytes(),f.tobytes()) for _,(v,f) in rows]


@pytest.mark.parametrize('corruption',['boundary_position','boundary_orientation'])
def test_valid_open_interior_contraction_rejects_boundary_corruption(q,corruption):
    # Remove a nonincident top face from the tiny bipyramid: vertices2/3/4
    # form one fixed boundary, while the actual contracted edge0--1 is interior.
    v,f,u,g,m=tiny_valid_collapse(q)
    f=np.delete(f,2,axis=0); j=np.array([1,2,4,5,6],np.int64)
    quotient=np.asarray(m['original_vertex_to_output'],np.int64); g=quotient[f[j]]
    m.update(source_faces=len(f),output_faces=len(g),J=j.tolist())
    m['ledger'][0]['removed_faces']=[0,3]
    arrays=[np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape) for a in (v,f,u,g)]
    source,candidate=q.verify_mapping(*arrays,m)
    assert len(source.boundary_loops)==len(candidate.boundary_loops)==1
    assert set(source.boundary_loops[0])=={2,3,4}
    before=[a.tobytes() for a in arrays]
    bad_u,bad_g=arrays[2].copy(),arrays[3].copy()
    if corruption=='boundary_position':
        boundary_output=m['I'].index(2); bad_u[boundary_output,2]+=.125
    else:
        # Reverse every surviving face: manifoldness remains, but oriented
        # boundary/face lineage cannot be relabeled as the original surface.
        bad_g=bad_g[:,::-1].copy()
    with pytest.raises(ValueError):q.verify_mapping(arrays[0],arrays[1],bad_u,bad_g,m)
    assert before==[a.tobytes() for a in arrays]
