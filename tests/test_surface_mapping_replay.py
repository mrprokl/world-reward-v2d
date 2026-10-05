"""Tiny exact-reference replay controls; no native solver or challenge geometry."""
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward.surface_mapping_replay import replay_surface_mapping

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'd6ec4443d71fdf3b717a9b37855b8c6697196fcc'
SOURCE_SHA = '59bf5bbc9ae4eae77ed0757fb3a8c9f9475458ea14b8c74c0cdcfca6996b2a74'
REGION_SHA = 'f348940aa1b93037cc8f9a83d943e0935b4ca620ac2b9119b1fcb9d531d24030'


@pytest.fixture(scope='module')
def original():
    raw = subprocess.check_output(['git','show',REVISION+':infra/surface_qslim_qualify.py'],cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    text=raw.decode(); fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='verify_mapping')
    begin=next(n.lineno for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='parent' for t in n.targets))
    end=next(n.lineno for n in fn.body if isinstance(n,ast.FunctionDef)and n.name=='root')
    region='\n'.join(text.splitlines()[begin-1:end-1])+'\n'
    assert hashlib.sha256(region.encode()).hexdigest()==REGION_SHA
    def require(ok,msg):
        if not ok:raise ValueError(msg)
    def ints(value,n):
        require(type(value)is list and len(value)==n and all(type(x)is int and -(1<<63)<=x<(1<<63) for x in value),'integer lineage')
        return np.array(value,np.int64)
    # Execute only the immutable original arithmetic region, never its driver.
    scope={'np':np,'rt':SimpleNamespace(require=require),'ints':ints}
    exec('def reference(v,f,mapping,boundary_vertices,orientation):\n'+region+
        '    return parent,positions,live,frozenset(removed)\n',scope)
    sys.path.insert(0,str(ROOT/'infra'))
    import surface_qslim_qualify as q
    return scope['reference'],q.orientation


def fixture():
    n=8;a=np.arange(n)*2*np.pi/n
    v=np.vstack((np.column_stack((np.cos(a),np.sin(a),np.zeros(n))),[0,0,1],[0,0,-1])).astype(np.float32)
    f=np.array([r for k in range(n)for r in ((n,k,(k+1)%n),(n+1,(k+1)%n,k))],np.int64)
    # Separate small open component must remain whole and untouched.
    v=np.vstack((v,[[4.,0,0],[4.015625,0,0],[4.,.015625,0]])).astype(np.float32)
    f=np.vstack((f,[[10,11,12]])).astype(np.int64)
    p=(v[0].astype(np.float64)+v[1])/2
    ledger=[dict(survivor=0,removed_vertex=1,placement=p.tolist(),removed_faces=[0,1]),
        dict(survivor=0,removed_vertex=2,placement=((p+v[2])/2).tolist(),removed_faces=[2,3])]
    return v,f,ledger,{10,11,12}


def compare(original,v,f,ledger,boundary,**kwargs):
    ref,orientation=original;expected=ref(v,f,{'ledger':ledger},boundary,orientation)
    state=replay_surface_mapping(v,f,ledger,boundary_vertices=boundary,orientation=kwargs.pop('orientation',orientation),**kwargs)
    for actual,old in zip((state.parent,state.positions,state.live),expected[:3]):
        assert actual.dtype==old.dtype and actual.shape==old.shape and actual.tobytes()==old.tobytes()
        assert not actual.flags.writeable
    assert state.removed_faces==expected[3]
    roots=[]
    for vertex in range(len(v)):
        while expected[0][vertex]!=vertex:vertex=int(expected[0][vertex])
        roots.append(vertex)
    assert state.quotient.dtype==np.dtype('int64') and state.quotient.tolist()==roots
    assert not state.quotient.flags.writeable
    return state


@pytest.mark.parametrize('count',[0,1,2])
def test_full_terminal_state_adjacent_contractions_and_tiny_component(original,count):
    v,f,ledger,boundary=fixture();before=(v.tobytes(),f.tobytes(),copy.deepcopy(ledger))
    state=compare(original,v,f,ledger[:count],boundary)
    assert state.live[-1].tolist()==[10,11,12] and state.positions[-3:].tobytes()==v[-3:].astype(np.float64).tobytes()
    assert (v.tobytes(),f.tobytes(),ledger)==before
    v[:]=9;f[:]=0
    assert state.positions[10,0]==4 and state.live[-1,0]==10
    with pytest.raises(ValueError):state.positions.setflags(write=True)


def test_disjoint_contraction_state_equality(original):
    v,f,ledger,boundary=fixture();last=len(v)
    v=np.vstack((v,v[:10]+[8,0,0])).astype(np.float32)
    f=np.vstack((f,f[:16]+last)).astype(np.int64)
    row=dict(survivor=last,removed_vertex=last+1,placement=(np.array(ledger[0]['placement'])+[8,0,0]).tolist(),removed_faces=[17,18])
    compare(original,v,f,[ledger[0],row],boundary)


def test_direct_parent_chain_and_compressed_quotient(original):
    v,f,_,boundary=fixture()
    p=(v[1].astype(np.float64)+v[2])/2
    ledger=[dict(survivor=1,removed_vertex=2,placement=p.tolist(),removed_faces=[2,3]),
        dict(survivor=0,removed_vertex=1,placement=((v[0]+p)/2).tolist(),removed_faces=[0,1])]
    state=compare(original,v,f,ledger,boundary)
    assert state.parent[:3].tolist()==[0,0,1] and state.quotient[:3].tolist()==[0,0,0]
    assert state.live[2].tolist()==[8,0,0]  # Historically removed face remapped again.


@pytest.mark.parametrize('fault',['bool','reversed','range','nonroot','boundary','extra','removed_duplicate',
    'removed_bool','removed_wrong','removed_tuple','third_edge_face','placement_nan','placement_shape',
    'placement_text','flip','bad_link','inactive_face'])
def test_rejection_equivalent_to_immutable_reference(original,fault):
    v,f,ledger,boundary=fixture()
    if fault=='bool':ledger[0]['survivor']=False
    elif fault=='reversed':ledger[0]['survivor']=2
    elif fault=='range':ledger[0]['removed_vertex']=100
    elif fault=='nonroot':ledger[1]['survivor']=1
    elif fault=='boundary':boundary.add(0)
    elif fault=='extra':ledger[0]['hidden']=1
    elif fault=='removed_duplicate':ledger[0]['removed_faces']=[0,0]
    elif fault=='removed_bool':ledger[0]['removed_faces']=[False,1]
    elif fault=='removed_wrong':ledger[0]['removed_faces']=[0,3]
    elif fault=='removed_tuple':ledger[0]['removed_faces']=(0,1)
    elif fault=='third_edge_face':f=np.vstack((f,[[0,1,10]])).astype(np.int64)
    elif fault=='placement_nan':ledger[0]['placement'][0]=np.nan
    elif fault=='placement_shape':ledger[0]['placement']=[0,1]
    elif fault=='placement_text':ledger[0]['placement']=['0','0','0']
    elif fault=='flip':ledger[0]['placement']=[-2.,-2.,0]
    elif fault=='bad_link':f[4,1]=0;f[5,2]=1
    elif fault=='inactive_face':ledger[1]['removed_faces']=[0,1]
    ref,orientation=original
    with pytest.raises(ValueError):ref(v,f,{'ledger':ledger},boundary,orientation)
    with pytest.raises(ValueError):replay_surface_mapping(v,f,ledger,boundary_vertices=boundary,orientation=orientation)


def test_batch_exact_three_semantics_and_no_scalar_calls(original):
    v,f,ledger,boundary=fixture();calls=[];orientation=original[1]
    def batch(old,new):
        calls.append((old.dtype,new.dtype,len(old)))
        for a,b in zip(old,new):orientation(a,b)
    state=compare(original,v,f,ledger,boundary,orientation_batch=batch,
        orientation=lambda *_:pytest.fail('Scalar callback must not run in batch mode'))
    assert len(calls)==6
    for i in (0,3):
        assert calls[i][:2]==(np.dtype('float64'),np.dtype('float64'))
        assert calls[i+1][:2]==(np.dtype('float32'),np.dtype('float32'))
        assert calls[i+2][:2]==(np.dtype('float64'),np.dtype('float32'))
    assert len(state.removed_faces)==4


def test_batch_rejection_propagates(original):
    v,f,ledger,boundary=fixture()
    def reject(*_):raise ValueError('qualified predicate rejected')
    with pytest.raises(ValueError,match='qualified predicate'):
        replay_surface_mapping(v,f,ledger,boundary_vertices=boundary,orientation=original[1],orientation_batch=reject)


def test_callbacks_receive_same_precision_operands_as_original(original):
    v,f,ledger,boundary=fixture();reference,orientation=original
    expected=[];actual=[]
    def record(destination,a,b):
        destination.append((a.dtype.str,b.dtype.str,a.tobytes(),b.tobytes()))
        orientation(a,b)
    reference(v,f,{'ledger':ledger},boundary,lambda a,b:record(expected,a,b))
    replay_surface_mapping(v,f,ledger,boundary_vertices=boundary,orientation=lambda a,b:record(actual,a,b))
    assert sorted(actual)==sorted(expected)
    actual.clear()
    def batch(a,b):
        assert a.shape==b.shape and a.shape[1:]==(3,3)
        for old,new in zip(a,b):record(actual,old,new)
    replay_surface_mapping(v,f,ledger,boundary_vertices=boundary,orientation=orientation,orientation_batch=batch)
    assert sorted(actual)==sorted(expected)


def test_empty_affected_bank_never_calls_batch(original):
    v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-1,0]],np.float32)
    f=np.array([[0,1,2],[1,0,3]],np.int64)
    ledger=[dict(survivor=0,removed_vertex=1,placement=[.5,0,0],removed_faces=[0,1])]
    compare(original,v,f,ledger,set(),orientation_batch=lambda *_:pytest.fail('Empty batch not called'))


@pytest.mark.parametrize('fault',['vertices','faces','ledger','callback'])
def test_typed_boundary_no_solver_or_alias(fault,original):
    v,f,ledger,boundary=fixture();orientation=original[1]
    if fault=='vertices':v=v.astype(np.int64)
    elif fault=='faces':f=f.astype(np.float64)
    elif fault=='ledger':ledger=tuple(ledger)
    else:orientation=None
    with pytest.raises(ValueError):replay_surface_mapping(v,f,ledger,boundary_vertices=boundary,orientation=orientation)
