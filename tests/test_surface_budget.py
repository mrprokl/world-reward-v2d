import numpy as np
import pytest

from world_reward.surface_budget import prepare_surface_budget, SurfaceBudgetError


def patch(orphan=False):
    v=np.array([[0,0,0],[.25,0,0],[.25,.125,.03125],[0,.125,0]],np.float32)
    if orphan:v=np.vstack((v,[[4,2,1]])).astype(np.float32)
    return v,np.array([[0,1,2],[0,2,3]],np.int64)


def test_identity_never_calls_native_and_owns_all_rows():
    v,f=patch(True);before=(v.tobytes(),f.tobytes())
    p=prepare_surface_budget(v,f,simplify=lambda *_:pytest.fail('No QEM'))
    assert p.proof['method']=='identity' and p.proof['qem_calls']==0
    assert p.proof['source_surface']['unused_vertices_preserved']==1
    assert p.vertices.dtype==np.float32 and p.vertices.tobytes()==before[0]
    assert p.mapping['I']==list(range(5)) and p.mapping['J']==[0,1]
    v[0]=99;assert p.vertices[0,0]==0
    with pytest.raises(ValueError):p.vertices.setflags(write=True)
    assert not p.proof['volume_or_closure_required']


@pytest.mark.parametrize('fault',['index','collinear','duplicate','winding'])
def test_rejection_retains_frozen_source_identity(fault):
    v,f=patch()
    if fault=='index':f[0,0]=9
    if fault=='collinear':v[2]=[.5,0,0]
    if fault=='duplicate':f[1]=f[0]
    if fault=='winding':f[1]=f[1,::-1]
    with pytest.raises(SurfaceBudgetError)as e:prepare_surface_budget(v,f)
    assert e.value.report['status']=='fail' and e.value.report['source_arrays_frozen_before_gates']
    assert e.value.report['source_faces']['bytes']==f.nbytes
    assert e.value.report['qem_calls']==0 and e.value.report['source_arrays_unchanged']


def test_over_budget_single_callback_and_original_verifier(monkeypatch):
    # Only the pure test target is smaller; production exposes no target option.
    import world_reward.surface_budget as core
    monkeypatch.setattr(core,'BUDGET',3)
    v,f=patch();calls=[]
    def native(sv,sf):
        assert not sv.flags.writeable and not sf.flags.writeable
        calls.append('native');return sv.astype(np.float64),sf,{'untouched_native':True}
    def verify(*args):calls.append('original_verifier');assert args[-1]=={'untouched_native':True}
    p=prepare_surface_budget(v,f,simplify=native,verify_mapping=verify)
    assert calls==['native','original_verifier']and p.proof['qem_calls']==1
    assert p.vertices.dtype==np.float64


def test_no_fallback_after_native_failure(monkeypatch):
    import world_reward.surface_budget as core
    monkeypatch.setattr(core,'BUDGET',3);v,f=patch();calls=[]
    def fail(*_):calls.append(1);raise RuntimeError('native failed')
    with pytest.raises(SurfaceBudgetError)as e:prepare_surface_budget(v,f,simplify=fail,verify_mapping=lambda *_:None)
    assert calls==[1]and e.value.report['qem_calls']==1


def test_missing_native_verifier_fails_before_call(monkeypatch):
    import world_reward.surface_budget as core
    monkeypatch.setattr(core,'BUDGET',3);v,f=patch()
    with pytest.raises(SurfaceBudgetError)as e:prepare_surface_budget(v,f,simplify=lambda *_:pytest.fail())
    assert e.value.report['qem_calls']==0


def test_over_budget_orphan_rejected_before_native_without_deleting_it(monkeypatch):
    import world_reward.surface_budget as core
    monkeypatch.setattr(core,'BUDGET',3);v,f=patch(True)
    with pytest.raises(SurfaceBudgetError)as e:
        prepare_surface_budget(v,f,simplify=lambda *_:pytest.fail('No native call'),verify_mapping=lambda *_:None)
    assert e.value.report['qem_calls']==0
    assert e.value.report['source_surface']['unused_vertices_preserved']==1
