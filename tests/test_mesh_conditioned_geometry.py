"""Tiny fresh surface and native orchestration spies; never compile or full QA."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).parents[1]/'infra'),str(Path(__file__).parents[1]/'src')]
import mesh_conditioned_geometry as controls


def test_tiny_symmetric_surface_unique_seam_poles_exact_antipodes_outward():
    v,f=controls.radial_mesh(8,6)
    assert v.shape==(42,3)and f.shape==(80,3)and v.dtype==np.float64 and f.dtype==np.int64
    assert len(np.unique(v,axis=0))==len(v)and np.count_nonzero(np.all(v[:,:2]==0,axis=1))==2
    np.testing.assert_array_equal(v.min(axis=0),-v.max(axis=0))
    points={tuple(p)for p in v};assert all(tuple(-p)in points for p in v)
    top=controls.geometry.exact_mesh_topology(v,f)
    assert len(top['components'])==1 and top['components'][0]['euler']==2 and top['components'][0]['volume_sign']==1


def test_declared_shapes_scales_rigid_transforms_and_chart_no_surface_snap(monkeypatch):
    original=controls.radial_mesh;calls=[]
    def tiny(azimuth=96,polar=48):calls.append((azimuth,polar));return original(8,6)
    monkeypatch.setattr(controls,'radial_mesh',tiny)
    sources=controls.fixtures()
    assert calls==[(96,48),(32,16)]and tuple(n for n,_,_ in sources)==controls.FIXTURE_NAMES
    assert controls.SCALES==(2.**-14,4.)and controls.TRANSLATION==(4.,-2.,1.)
    base,_=original(8,6)
    for i,(_,mesh,cavity)in enumerate(sources):
        scale=controls.SCALES[i%2];rotation=np.array(controls.ROTATIONS[i%2]);v,f=mesh
        assert np.linalg.det(rotation)==1 and np.array_equal(rotation.T@rotation,np.eye(3))
        np.testing.assert_array_equal(v[:len(base)],base@rotation.T*scale+np.array(controls.TRANSLATION)*scale)
        assert cavity==(i>=2)
        chart=controls.prepare_conditioning(v,f)
        assert chart.diagnostics['roundtrip_numerically_exact']and chart.scale>0
        np.testing.assert_array_equal(chart.decode(chart.encode(v)),v)
        if cavity:
            center=np.array(controls.INNER_TRANSLATION)@rotation.T*scale+np.array(controls.TRANSLATION)*scale
            np.testing.assert_allclose(v[len(base):].mean(axis=0),center,rtol=1e-14,atol=0)
            assert not np.array_equal(center,np.array(controls.TRANSLATION)*scale)


@pytest.mark.parametrize('azimuth,polar',[(3,6),(6,6),(8,3),(8,5),(True,6)])
def test_malformed_sampling_does_not_refine_or_drop(azimuth,polar):
    with pytest.raises(ValueError):controls.radial_mesh(azimuth,polar)


def setup(tmp_path,monkeypatch):
    old,new,helper=(tmp_path/n for n in('old','new','helper'))
    for p in(old,new,helper):p.write_bytes(b'x')
    monkeypatch.setattr(controls,'ORIGINAL_BINARY',old)
    monkeypatch.setattr(controls,'identity',lambda p:dict(bytes=2031,sha256=controls.endpoint.BUDGET_HELPER_SHA))
    monkeypatch.setattr(controls.geometry,'validate_legacy_sources',lambda:{'legacy':'pin'})
    monkeypatch.setattr(controls.importlib.util,'spec_from_file_location',lambda *a:SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m:None)))
    monkeypatch.setattr(controls.importlib.util,'module_from_spec',lambda s:SimpleNamespace())
    v,f=controls.radial_mesh(8,6);f=np.tile(f,(52,1))  # Tiny procedural mocked topology gate, not qualified geometry.
    meshes=tuple((name,(v.copy(),f.copy()),i>=2)for i,name in enumerate(controls.FIXTURE_NAMES))
    monkeypatch.setattr(controls,'fixtures',lambda:meshes)
    monkeypatch.setattr(controls,'topology_and_embedding',lambda *a:{'tiny_spy':True})
    monkeypatch.setattr(controls,'serialization_preflight',lambda *a:{'position_weld_admissible':True})
    return new,helper,meshes


def test_four_fresh_paired_sources_same_bytes_original_failure_retained_and_new_zero_veto_pass(tmp_path,monkeypatch):
    new,helper,_=setup(tmp_path,monkeypatch);seen=[]
    def branch(source,cavity,binary,method,work,h,remaining,record):
        seen.append((method,controls.array_hashes(source)));record['native_attempts']=1
        if method=='original':raise ValueError('frozen comparator geometric failure')
        record.update(status='pass',committed_collapses=5,serialization_vetoes=0)
    monkeypatch.setattr(controls,'branch',branch)
    report=controls.geometry_controls(new,tmp_path,lambda:600,official_helper=helper)
    assert report['status']=='pass'and report['maximum_native_calls']==8 and report['native_budget_seconds']==600
    assert report['conditioning_mechanism_exercised']and not report['serialization_veto_mechanism_exercised']
    assert len(report['paired_fixtures'])==4 and len(seen)==8
    assert all(seen[i][1]==seen[i+1][1]for i in range(0,8,2))
    assert report['sources_rehashed_after']and report['owned_scratch_removed']and not report['adoption']
    assert not(tmp_path/'conditioned-geometry-controls').exists()


@pytest.mark.parametrize('fault',['conditioned','technical','timeout','source_mutation','roundtrip'])
def test_any_new_failure_or_original_technical_stops_without_reroll_and_retains_partial(tmp_path,monkeypatch,fault):
    new,helper,meshes=setup(tmp_path,monkeypatch);sentinel=tmp_path/'foreign';sentinel.write_bytes(b'keep');calls=[]
    if fault=='roundtrip':
        monkeypatch.setattr(controls,'prepare_conditioning',lambda *a:SimpleNamespace(scale=1.,diagnostics={'roundtrip_numerically_exact':False}))
    def branch(source,cavity,binary,method,work,h,remaining,record):
        calls.append(method);record['native_attempts']=1
        if fault=='technical':raise PermissionError('no executable')
        if fault=='timeout':raise controls.subprocess.TimeoutExpired('native',600)
        if fault=='conditioned'and method=='conditioned':raise ValueError('new failed gate')
        if fault=='source_mutation':source[0][0,0]+=1
        record.update(status='pass',committed_collapses=2,serialization_vetoes=1)
    monkeypatch.setattr(controls,'branch',branch)
    with pytest.raises(controls.GeometryControlError)as exc:controls.geometry_controls(new,tmp_path,lambda:600,official_helper=helper)
    report=exc.value.report
    assert report['status']=='fail'and report['owned_scratch_removed']and sentinel.read_bytes()==b'keep'
    assert len(calls)<=2 and not report['reroll_performed']
    if fault!='source_mutation':assert report['sources_rehashed_after']
    if fault=='roundtrip':assert not calls


def test_namespace_collision_and_binary_alias_fail_before_any_geometry(tmp_path):
    (tmp_path/'conditioned-geometry-controls').mkdir()
    with pytest.raises(ValueError,match='Fresh'):controls.geometry_controls(tmp_path/'new',tmp_path,lambda:600,official_helper=tmp_path/'helper')


@pytest.mark.parametrize('timeout',[False,True])
def test_native_bounded600_failure_preserves_input_proof_and_scalar_record(tmp_path,monkeypatch,timeout):
    monkeypatch.setattr(controls.geometry,'write_obj',lambda p,*a:p.write_bytes(b'own input'))
    def execute(args,**kw):
        assert kw['timeout']==600 and kw['capture_output']is True
        if timeout:raise controls.subprocess.TimeoutExpired(args,600)
        return SimpleNamespace(returncode=3,stderr=b'failure '*100)
    monkeypatch.setattr(controls.subprocess,'run',execute);record={}
    with pytest.raises((ValueError,controls.subprocess.TimeoutExpired)):
        controls.branch((np.zeros((4,3)),np.array([[0,1,2]])),False,tmp_path/'binary','original',tmp_path,None,lambda:700,record)
    assert record['native_attempts']==1 and record['native_input']['bytes']==9 and record['native_elapsed_seconds']>=0
    if not timeout:assert record['native_returned']and record['native_exit_code']==3 and len(record['native_stderr_tail'])<=500


@pytest.mark.parametrize('conditioned',[False,True])
def test_real_branch_contract_keeps_mapping_private_and_reuses_all_physical_gates(tmp_path,monkeypatch,conditioned):
    source=controls.radial_mesh(8,6);calls=[];record={}
    mapping=dict(native_cost_and_placement_unchanged=not conditioned,cost_normalization=conditioned,
        final_shell_volumes_verified=True,volume_relative_limit=.05,committed_collapses=3,I=[0],J=[0])
    payload=mapping
    if conditioned:
        payload=dict(conditioning=dict(chart_scale_positive=True,physical_geometry_rescaled=False,
            source_roundtrip_numerically_exact=True,new_numeric_algorithm=True,native_qslim_implementation_reused=True),
            serialization=dict(serialization_safe=True,serialization_vetoes=0,committed_collapses=3),native_volume=mapping)
    monkeypatch.setattr(controls.geometry,'write_obj',lambda p,*a:p.write_bytes(b'owninput'))
    def native(args,**kwargs):
        Path(args[2]).write_bytes(b'candidate');Path(args[3]).write_text(__import__('json').dumps(payload))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controls.subprocess,'run',native)
    monkeypatch.setattr(controls.geometry,'read_obj',lambda p:source)
    monkeypatch.setattr(controls,'serialization_preflight',lambda *a:{'position_weld_admissible':True})
    def stage(before,after,m,cavity,remaining):calls.append((m,after));return {'gate':'unchanged'}
    monkeypatch.setattr(controls,'stage',stage)
    class Mesh:
        def __init__(self,*a,**kwargs):assert kwargs=={'process':False}
        def export(self,p):p.write_bytes(b'GLB')
    monkeypatch.setitem(sys.modules,'trimesh',SimpleNamespace(Trimesh=Mesh))
    stored=(source[0].astype(np.float32).astype(np.float64),source[1])
    monkeypatch.setattr(controls.endpoint,'_load_mesh',lambda p:stored)
    monkeypatch.setattr(controls.endpoint,'exact_weld',lambda *a:(*stored,{}))
    monkeypatch.setattr(controls.original,'export_birth_mapping',lambda *a:mapping)
    monkeypatch.setattr(controls.original,'packed_mapping',lambda *a:mapping)
    monkeypatch.setattr(controls.geometry,'verify_pack_fidelity',lambda mesh,pv,pf:(mesh,{'byte_fidelity':True}))
    def pack(path,**kwargs):assert kwargs=={'faces':4096,'vertices':4096};return stored
    controls.branch(source,False,tmp_path/'binary','conditioned'if conditioned else'original',tmp_path,
        SimpleNamespace(budget_mesh=pack),lambda:600,record)
    assert record['status']=='pass'and len(calls)==4 and record['metric_scale_baked_once']==.375
    assert 'I'not in record['native_volume']and 'J'not in record['native_volume']
    assert all(c[0]is mapping or c[0]==mapping for c in calls)
    if conditioned:assert record['serialization_vetoes']==0


def test_literal_reuse_fresh_fixture_parameters_and_physical_pipeline_source():
    source=(Path(__file__).parents[1]/'infra/mesh_conditioned_geometry.py').read_text();ast.parse(source)
    assert controls.AZIMUTH==96 and controls.POLAR==48 and controls.INNER_AZIMUTH==32 and controls.INNER_POLAR==16
    assert controls.METRIC_SCALE==.375 and controls.POWER==.8
    assert 'original.fixtures('not in source and 'graded_corner('not in source and 'capsule('not in source
    assert 'source_float32_topology=stored_top' in source and 'roundtrip_numerically_exact' in source
    assert "helper.budget_mesh(str(glb),faces=4096,vertices=4096)"in source
    assert 'original.export_birth_mapping(candidate,exported,mapping)'in source and 'original.packed_mapping(exported,compact,emap)'in source
    assert 'new_numeric_algorithm=True'in source and 'serialization_veto_mechanism_exercised=bool(vetoes)'in source
    assert (controls.POLAR-1)*controls.AZIMUTH+2==4514 and 2*controls.AZIMUTH*(controls.POLAR-1)==9024
    assert (controls.INNER_POLAR-1)*controls.INNER_AZIMUTH+2==482
