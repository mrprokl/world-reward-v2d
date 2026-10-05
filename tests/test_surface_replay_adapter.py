"""Whole original-versus-filtered verifier controls, no native execution."""
import ast
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'infra'))
import surface_qslim_qualify as q
from surface_replay_adapter import verifier, require_orientation


def fixture():
    v=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1],
        [4,0,0],[4.015625,0,0],[4,.015625,0]],np.float32)
    f=np.array([[4,0,1],[4,1,2],[4,2,3],[4,3,0],[5,1,0],[5,2,1],[5,3,2],[5,0,3],[6,7,8]],np.int64)
    i=np.array([0,2,3,5,6,7,8]);j=np.array([2,5,6,7,8]);quotient=np.array([0,0,1,2,0,3,4,5,6])
    g=quotient[f[j]];u=v[i].astype(np.float64);u[0]=[.25,.25,.5]
    m=dict(schema='surface-qslim-mapping-v1',source_sha256=hashlib.sha256((ROOT/'infra/surface_qslim.cpp').read_bytes()).hexdigest(),
        source_vertices=len(v),source_faces=len(f),output_vertices=len(u),output_faces=len(g),target_vertices=4096,target_faces=4096,
        boundary_policy='fixed_original_vertices',intersection_blocking='upstream_floating_point',initial_embedding_certified=False,
        volume_or_closure_required=False,serialization_qualification_completed=False,adoption=False,
        native_attempts=2,native_failed=0,veto_link=0,veto_geometry=0,veto_component=0,veto_intersection=0,
        J=j.tolist(),I=i.tolist(),original_vertex_to_output=quotient.tolist(),committed_collapses=2,
        candidate_component_to_source=[0,1],ledger=[dict(survivor=0,removed_vertex=1,placement=[.5,.5,0],removed_faces=[0,4]),
        dict(survivor=0,removed_vertex=4,placement=[.25,.25,.5],removed_faces=[1,3])])
    return v,f,u,g,m


def record(value):
    return [(row.vertices.tobytes(),row.faces.tobytes(),row.face_components.tobytes(),
        row.component_keys,row.boundary_loops,row.boundary_components,[dict(x)for x in row.stats])for row in value]


def test_complete_original_gates_positive_and_source_unchanged():
    v,f,u,g,m=fixture();before=[x.tobytes()for x in (v,f,u,g)];raw=copy.deepcopy(m)
    assert record(q.verify_mapping(v,f,u,g,m)) == record(verifier(q)(v,f,u,g,m))
    assert before==[x.tobytes()for x in (v,f,u,g)] and m==raw


@pytest.mark.parametrize('fault',['birth','face_order','quotient','position','removed','count','flip','bool_birth',
    'component','component_duplicate','delete_tiny','boundary_position','ledger_extra','nonroot','bad_placement','bad_counter'])
def test_full_acceptance_and_rejection_match_original(fault):
    v,f,u,g,m=fixture()
    if fault=='birth':m['I'][0]=1
    elif fault=='face_order':g=g[::-1].copy()
    elif fault=='quotient':m['original_vertex_to_output'][1]=1
    elif fault=='position':u[0,0]=.6
    elif fault=='removed':m['ledger'][0]['removed_faces']=[0,1]
    elif fault=='count':m['native_attempts']=3
    elif fault=='flip':g[0]=g[0,::-1]
    elif fault=='bool_birth':m['J'][0]=True
    elif fault=='component':m['candidate_component_to_source']=[1,0]
    elif fault=='component_duplicate':m['candidate_component_to_source']=[0,0]
    elif fault=='delete_tiny':g=g[:-1];m['output_faces']=len(g);m['J']=m['J'][:-1]
    elif fault=='boundary_position':u[-1,0]=4.01
    elif fault=='ledger_extra':m['ledger'][0]['hidden']=1
    elif fault=='nonroot':m['ledger'][1]['survivor']=1
    elif fault=='bad_placement':m['ledger'][0]['placement']=[.5,.5,-100]
    elif fault=='bad_counter':m['native_failed']=True
    for verify in (q.verify_mapping,verifier(q)):
        with pytest.raises(ValueError):verify(v,f,u,g,m)


def test_original_source_tamper_not_adapted(tmp_path):
    from types import SimpleNamespace
    path=tmp_path/'surface_qslim_qualify.py';path.write_bytes((ROOT/'infra/surface_qslim_qualify.py').read_bytes()+b'\n')
    with pytest.raises(ValueError):verifier(SimpleNamespace(__file__=str(path)))


def test_exact_batch_rejects_zero_and_float32_collapse():
    a=np.array([[[1.,0,0],[1.+2**-30,0,0],[1,1,0]]])
    require_orientation(a,a)
    for b in (a.astype(np.float32),a[:,::-1],np.zeros_like(a)):
        with pytest.raises(ValueError):require_orientation(a,b)


def test_native_qslim_source_and_qualification_helpers_not_changed():
    from surface_replay_adapter import SOURCE_SHA256
    assert hashlib.sha256((ROOT/'infra/surface_qslim_qualify.py').read_bytes()).hexdigest()==SOURCE_SHA256
    tree=ast.parse((ROOT/'infra/surface_replay_adapter.py').read_bytes())
    imports={n.name for row in ast.walk(tree)if isinstance(row,ast.Import)for n in row.names}
    assert not imports&{'subprocess','torch','trimesh','requests'}
