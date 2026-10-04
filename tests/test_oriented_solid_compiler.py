"""Tiny represented geometry and mocked native processes; no compile/Azure/media."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
import oriented_solid_compiler as p


def tetra(offset=(0.,0.,0.)):
    return (np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])+offset,
            np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64))


def native_report(raw,sha):
    tokens=raw.decode().split();nv,nf,count=map(int,tokens[1:4])
    v=np.array(tokens[4:4+3*nv],np.float64).reshape(nv,3)
    f=np.array(tokens[4+3*nv:],np.int64).reshape(nf,4);rows=[]
    for i in range(count):
        ids=np.unique(f[f[:,3]==i,:3]);tri=v[f[f[:,3]==i,:3]]
        volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
        rows.append(dict(original_component_id=i,original_vertices=len(ids),original_faces=int(np.count_nonzero(f[:,3]==i)),
            witness_original_vertex=int(ids[0]),exact_volume_sign=1 if volume>0 else -1,diagnostic_signed_volume=volume))
    return dict(schema='world_reward.certified_solid_query.v1',status='pass',source_sha256=sha,cgal_version='6.0.1',
        vertices=nv,faces=nf,component_count=count,components=rows,inside=np.zeros((count,count),bool).tolist(),
        represented_coordinates='EPECK exact values of original parsed IEEE754 binary64; no perturbation',
        **{k:True for k in p.certificate.TRUE_FLAGS},**{k:False for k in p.certificate.FALSE_FLAGS})


def document(mesh,conditioning_version=1):
    chart=p.conditioning_function(conditioning_version)(*mesh);m=p.identity_mapping(mesh)
    m.update(native_cost_and_placement_unchanged=False,cost_normalization=True,final_shell_volumes_verified=True,
             volume_relative_limit=.05,committed_collapses=0)
    conditioning=dict(origin=chart.origin.tolist(),scale=chart.scale,scale_exponent=chart.scale_exponent,
        source_roundtrip_vertices=len(mesh[0]),chart_scale_positive=True,source_roundtrip_numerically_exact=True,
        physical_geometry_rescaled=False,new_numeric_algorithm=True,native_qslim_implementation_reused=True,
        chart_refitted=False,adopted=False)
    if conditioning_version==2:
        conditioning.update(chart_version=2,origin_modes=list(chart.origin_modes),
            policy_sha256=p.chart_v2_helper.POLICY_SHA256,
            header_sha256=p.certificate.identity(p.CHART_V2_HEADER)['sha256'],
            source_roundtrip_byte_exact=chart.diagnostics['roundtrip_byte_exact'],
            origin_search_performed=False,native_backend_qualified=False)
    return dict(conditioning=conditioning,serialization=dict(serialization_safe=True,committed_collapses=0,serialization_vetoes=0),native_volume=m)


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    work=tmp_path/'work';work.mkdir();binary=tmp_path/'qem';binary.write_bytes(b'fake not executed')
    query=tmp_path/'query';query.write_bytes(b'fake not executed')
    helper=tmp_path/'mesh_budget.py';helper.write_text('import numpy as np\ndef budget_mesh(path,faces,vertices):\n    v=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float32)\n    f=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)\n    return np.r_[v,np.repeat(v[:1],4096-len(v),axis=0)],np.r_[f,np.zeros((4096-len(f),3),np.int64)]\n')
    sha=hashlib.sha256((REPO/'infra/certified_solid_query.cpp').read_bytes()).hexdigest()
    monkeypatch.setattr(p.endpoint,'BUDGET_HELPER_SHA',p.certificate.identity(helper)['sha256'])
    mesh=tetra();calls=[]
    def run(argv,**kw):
        if len(argv)==1:
            calls.append(('query',kw['timeout']))
            return SimpleNamespace(returncode=0,stdout=json.dumps(native_report(kw['input'],sha)).encode(),stderr=b'')
        calls.append(('qem',kw['timeout']));assert p.geometry.read_obj(Path(argv[1]))[0].shape==(4,3)
        p.geometry.write_obj(Path(argv[2]),*mesh);Path(argv[3]).write_text(json.dumps(document(mesh)))
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    monkeypatch.setattr(p.subprocess,'run',run)
    class Mesh:
        def __init__(self,*args,**kw):
            self.vertices=kw.get('vertices',args[0] if args else None)
            self.faces=kw.get('faces',args[1] if args else None)
            calls.append(('trimesh',kw.get('process')))
        def export(self,path):Path(path).write_bytes(b'fake source-bound GLB never rendered')
    monkeypatch.setitem(sys.modules,'trimesh',SimpleNamespace(Trimesh=Mesh,__version__='fixture-not-a-qualified-runtime'))
    monkeypatch.setattr(p.endpoint,'_load_mesh',lambda path:tuple(a.copy() for a in mesh))
    return mesh,binary,query,sha,work,helper,calls


def test_complete_six_stage_path_actual_pure_fidelity_and_full_births(runtime):
    mesh,binary,query,sha,work,helper,calls=runtime;before=tuple(a.tobytes() for a in mesh)
    (v,f),report=p.compile_solid(mesh,binary,query,sha,work,helper,metric_scale=.375)
    assert report['status']=='pass' and tuple(report['stages'])==p.STAGES
    assert len([x for x in calls if x[0]=='query'])==8 and len([x for x in calls if x[0]=='qem'])==1
    assert all(x[1]<=180 for x in calls if x[0]=='query')
    assert all(x[1]<=1200 for x in calls if x[0]=='qem')
    assert ('trimesh',False) in calls and ('trimesh',True) in calls
    assert v.shape==f.shape==(4096,3) and report['metric_scale_baked_once']==.375
    assert report['artifacts_before']==report['artifacts_after'] and report['source_arrays_unchanged']
    assert before==tuple(a.tobytes() for a in mesh)
    assert all(row['candidate_to_source_components']==[0] for row in list(report['stages'].values())[1:])
    assert report['adoption'] is report['reconstruction_accuracy_verified'] is False


@pytest.mark.parametrize('failure',['contact','runtime','bad_birth','native_exit','metric_forest'])
def test_failure_is_not_repair_or_fallback_and_preserves_partial_evidence(runtime,monkeypatch,failure):
    mesh,binary,query,sha,work,helper,calls=runtime;original=p.subprocess.run
    def run(argv,**kw):
        if len(argv)==1 and failure in ('contact','runtime'):
            reason='Components intersect or touch' if failure=='contact' else 'std::bad_alloc'
            return SimpleNamespace(returncode=1,stdout=b'',stderr=('certified_solid_query FAIL: '+reason+'\n').encode())
        result=original(argv,**kw)
        if len(argv)==4 and failure=='bad_birth':
            path=Path(argv[3]);d=json.loads(path.read_text());d['native_volume']['I'][0]=99;path.write_text(json.dumps(d))
        if len(argv)==4 and failure=='native_exit':return SimpleNamespace(returncode=2,stdout=b'',stderr=b'closed native queue')
        if len(argv)==1 and failure=='metric_forest' and len([c for c in calls if c[0]=='query'])==7:
            data=json.loads(result.stdout);data['components'][0]['exact_volume_sign']=-1
            return SimpleNamespace(returncode=0,stdout=json.dumps(data).encode(),stderr=b'')
        return result
    monkeypatch.setattr(p.subprocess,'run',run)
    with pytest.raises(p.SolidCompilerError) as error:p.compile_solid(mesh,binary,query,sha,work,helper,metric_scale=.375)
    report=error.value.report
    assert report['status']=='fail' and report['source_arrays_unchanged'] and report['adoption'] is False
    assert len([x for x in calls if x[0]=='qem'])<=1
    if failure in ('contact','runtime'):assert report['native_attempts']==0
    if failure=='native_exit':assert report['native_stderr_tail']=='closed native queue'


@pytest.mark.parametrize('which',['face_drop','vertex_merge','flip'])
def test_default_weld_vertex_face_permutation_preserves_births_but_damage_fails(which):
    mesh=tetra();mapping=p.identity_mapping(mesh);order=np.array([3,1,0,2]);inverse=np.argsort(order)
    after=(mesh[0][order],inverse[mesh[1]][::-1])
    result=p.reordered_mapping(mesh,after,mapping)
    assert result['I']==order.tolist() and result['J']==[3,2,1,0]
    bad=(after[0].copy(),after[1].copy())
    if which=='face_drop':bad=(bad[0],bad[1][:-1])
    elif which=='vertex_merge':bad[0][1]=bad[0][0]
    else:bad[1][0]=bad[1][0,::-1]
    with pytest.raises(ValueError):p.reordered_mapping(mesh,bad,mapping)


def test_native_component_ids_never_substitute_original_face_births(runtime):
    mesh,binary,query,sha,work,helper,_=runtime
    other,of=tetra((4.,0.,0.));source=(np.r_[mesh[0],other],np.r_[mesh[1],of+4])
    labels,forest=p.certify_arrays(source,query,sha,lambda:180,{})
    # Candidate reverses whole components while vertices preserve their native source births.
    candidate=(np.r_[other,mesh[0]],source[1].copy());mapping=p.identity_mapping(source)
    mapping['I']=[4,5,6,7,0,1,2,3];mapping['J']=[4,5,6,7,0,1,2,3]
    record={};p.matched_stage(source,labels,forest,candidate,mapping,query,sha,lambda:180,record)
    assert record['candidate_to_source_components']==[1,0]
    mapping['J'][0]=0
    with pytest.raises(ValueError):p.matched_stage(source,labels,forest,candidate,mapping,query,sha,lambda:180,{})


def test_f32_source_loss_and_orphans_stop_before_native(runtime,monkeypatch):
    mesh,binary,query,sha,work,helper,calls=runtime
    changed=mesh[0].copy();changed+=2.**25  # F32 merges unit-separated positions.
    with pytest.raises(p.SolidCompilerError):p.compile_solid((changed,mesh[1]),binary,query,sha,work,helper,metric_scale=.375)
    assert not any(x[0]=='qem' for x in calls)
    with pytest.raises(ValueError):p.certificate.components(np.r_[mesh[0],[[9.,9.,9.]]],mesh[1])


def test_raw_glb_oriented_multiset_and_loaded_face_order(runtime,monkeypatch):
    mesh,binary,query,sha,work,helper,_=runtime;path=work/'source.glb';path.write_bytes(b'fixture')
    monkeypatch.setattr(p.precision,'raw_glb',lambda path:([(mesh[0],mesh[1])],mesh[0][mesh[1]],[]))
    loaded,proof=p.load_source(path)
    np.testing.assert_array_equal(loaded[0][loaded[1]],mesh[0][mesh[1]])
    assert proof['exact_welding']['faces_removed']==0 and 'multiset' in proof['order_scope']
    monkeypatch.setattr(p.precision,'raw_glb',lambda path:([(mesh[0],mesh[1])],mesh[0][mesh[1][:,::-1]],[]))
    with pytest.raises(ValueError):p.load_source(path)


def test_deadline_no_hidden_reset_and_missing_qualified_binary(runtime):
    mesh,binary,query,sha,work,helper,_=runtime
    with pytest.raises(TimeoutError):p.deadline(lambda:0)()
    with pytest.raises(p.SolidCompilerError) as error:p.compile_solid(mesh,work/'missing',query,sha,work,helper,metric_scale=.375)
    assert error.value.report['native_attempts']==0


def test_source_default8_collision_is_diagnostic_not_deleted():
    mesh=tetra();v=mesh[0]*2.**-28;info=p.serialization_preflight(v,mesh[1])
    assert info['float32_triangles_exactly_active'] and not info['position_weld_admissible']
    p.float32_orientation((v,mesh[1]))


@pytest.mark.parametrize('conditioning_version',[1,2])
def test_all_four_exact_source_preflights_before_any_qem_and_stop_no_retry(tmp_path,monkeypatch,conditioning_version):
    import oriented_solid_controls as controls
    items=controls.fixtures();events=[]
    def preflight(mesh,binary,sha,remaining,record,*,conditioning_version=1):
        assert conditioning_version in (1,2)
        found=next(meta for _,candidate,meta in items if candidate[0] is mesh[0])
        events.append('preflight')
        return found['face_components'],SimpleNamespace(parents=found['expected_parents'],signs=found['expected_signs']),None
    def compile(*args,**kwargs):
        assert events==['preflight']*4
        assert kwargs['conditioning_version']==conditioning_version
        events.append('qem');raise p.SolidCompilerError(dict(status='fail',native_attempts=1))
    monkeypatch.setattr(p,'preflight_source',preflight);monkeypatch.setattr(p,'compile_solid',compile)
    with pytest.raises(p.SolidCompilerError) as error:p.geometry_controls(Path('/mock/qem'),Path('/mock/query'),'a'*64,
        tmp_path,Path('/mock/helper'),controls=items,conditioning_version=conditioning_version)
    assert events==['preflight']*4+['qem'] and error.value.report['owned_scratch_removed']
    assert len(error.value.report['controls'])==4 and not (tmp_path/'oriented-solid-controls').exists()


def test_bad_last_source_forest_blocks_all_qem(tmp_path,monkeypatch):
    import oriented_solid_controls as controls
    items=controls.fixtures();events=[]
    def preflight(mesh,binary,sha,remaining,record,*,conditioning_version=1):
        assert conditioning_version==1
        found=next(meta for _,candidate,meta in items if candidate[0] is mesh[0]);events.append('query')
        parents=found['expected_parents'].copy()
        if len(events)==4:parents[0]=1
        return found['face_components'],SimpleNamespace(parents=parents,signs=found['expected_signs']),None
    monkeypatch.setattr(p,'preflight_source',preflight)
    monkeypatch.setattr(p,'compile_solid',lambda *a,**kw:pytest.fail('QEM before all four sources qualify'))
    with pytest.raises(p.SolidCompilerError):p.geometry_controls(Path('/mock/qem'),Path('/mock/query'),'a'*64,
        tmp_path,Path('/mock/helper'),controls=items)
    assert len(events)==4


def test_explicit_v2_whole_pipeline_bound_to_source_chart_and_actual_header(runtime,monkeypatch):
    mesh,binary,query,sha,work,helper,calls=runtime;original=p.subprocess.run
    def run(argv,**kw):
        result=original(argv,**kw)
        if len(argv)==4:Path(argv[3]).write_text(json.dumps(document(mesh,2)))
        return result
    monkeypatch.setattr(p.subprocess,'run',run)
    _,report=p.compile_solid(mesh,binary,query,sha,work,helper,metric_scale=.375,conditioning_version=2)
    assert report['stage']=='oriented_solid_compiler_v2' and report['conditioning_version']==2
    assert tuple(report['stages'])==p.STAGES
    assert report['native_mapping']['conditioning']['origin_modes']==['zero']*3
    assert report['native_mapping']['conditioning']['chart_version']==2
    assert report['native_mapping']['conditioning']['native_backend_qualified'] is False
    assert report['stages'][p.STAGES[0]]['conditioning']['schema']==p.chart_v2_helper.SCHEMA
    assert report['stages'][p.STAGES[0]]['conditioning_header']==p.certificate.identity(p.CHART_V2_HEADER)
    for path in (p.chart_v2_helper.__file__,p.CHART_V2_HEADER,p.chart_v2_source.__file__):
        assert str(Path(path)) in report['artifacts_before']
    assert report['artifacts_before']==report['artifacts_after']
    assert len([c for c in calls if c[0]=='qem'])==1


@pytest.mark.parametrize('bad',['header','policy','version','bool_version','modes','search','qualified',
                                'byte_roundtrip','extra','origin_bool'])
def test_v2_document_requires_actual_bound_header_policy_and_complete_chart_evidence(bad):
    mesh=tetra();chart=p.prepare_conditioning_v2(*mesh);doc=document(mesh,2);c=doc['conditioning']
    if bad=='header':c['header_sha256']='0'*64
    if bad=='policy':c['policy_sha256']='0'*64
    if bad=='version':c['chart_version']=1
    if bad=='bool_version':c['chart_version']=True
    if bad=='modes':c['origin_modes'][0]='sterbenz_midpoint'
    if bad=='search':c['origin_search_performed']=True
    if bad=='qualified':c['native_backend_qualified']=True
    if bad=='byte_roundtrip':c['source_roundtrip_byte_exact']=False
    if bad=='extra':c['new_unknown_field']=True
    if bad=='origin_bool':c['origin'][0]=False
    with pytest.raises(ValueError):
        p.validate_conditioned_mapping(mesh,mesh,doc,chart,conditioning_version=2)


def test_legacy_document_is_not_implicitly_accepted_as_v2_and_default_is_legacy():
    mesh=tetra()
    assert p.conditioning_function(1) is p.prepare_conditioning
    assert p.conditioning_function(2) is p.prepare_conditioning_v2
    legacy=document(mesh)
    assert p.validate_conditioned_mapping(mesh,mesh,legacy,p.prepare_conditioning(*mesh))[0]['mapping_complete']
    with pytest.raises(ValueError):
        p.validate_conditioned_mapping(mesh,mesh,legacy,p.prepare_conditioning_v2(*mesh),conditioning_version=2)


@pytest.mark.parametrize('version',[True,False,0,3,'2',2.])
def test_invalid_conditioning_version_fails_before_preflight_or_scratch_creation(tmp_path,version):
    with pytest.raises(ValueError):
        p.geometry_controls(Path('/missing/qem'),Path('/missing/query'),'a'*64,tmp_path,
                            Path('/missing/helper'),controls=(),conditioning_version=version)
    assert not any(tmp_path.iterdir())


def test_original_chart_failure_and_frozen_source_remain_identifiable(monkeypatch):
    import oriented_solid_controls as controls
    _,source,metadata=controls.fixtures()[2]
    before=tuple(a.tobytes() for a in source)
    assert not p.prepare_conditioning(*source).diagnostics['roundtrip_numerically_exact']
    monkeypatch.setattr(p,'certify_arrays',lambda *a:(None,None))
    with pytest.raises(ValueError,match='Source fixed chart must roundtrip exactly'):
        p.preflight_source(source,Path('/not-executed'),'a'*64,lambda:180,{})
    assert before==tuple(a.tobytes() for a in source)
    assert tuple(p.array_hashes(source))==metadata['source_array_sha256']
