import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def gate(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('wr_volume_gate',infra/'volume_mesh_gate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_new_cavity_parameters_closed_and_all_shells_retained(gate):
    pytest.importorskip('trimesh')
    controls=gate.fixtures();assert [x[0] for x in controls]==gate.FIXTURE_NAMES
    for i,(_,mesh) in enumerate(controls):
        topology=gate.mesh_topology(*mesh)
        assert topology['faces']>4096 and len(topology['components'])==(2 if i==0 else 7)
        assert sum(c['volume_sign']<0 for c in topology['components'])==(1 if i==0 else 6)
        hashes=[gate._array_hash(x) for x in mesh]
        again=gate.fixtures()[i][1];assert hashes==[gate._array_hash(x) for x in again]

def test_actual_volume_binary_and_private_mount_contract(gate):
    source=Path(gate.__file__).read_text();wrapper=Path(gate.__file__).with_name('run_volume_mesh_gate.sh').read_text()
    assert 'guarded.mapped_geometry' in source and 'source_intersections' in source
    assert "info.get('volume_relative_limit')!=.05" in source
    assert '--gpus' not in wrapper and '243s docker run' in wrapper
    assert not any(f'src=$ROOT/{folder}' in wrapper for folder in ('data','outputs','weights','vendor'))
    assert 'mesh_volume_qem.cpp' in wrapper and 'mesh_guarded_qem.cpp' in wrapper
    with pytest.raises(SystemExit):gate.main(['--oracle'])

def test_cavities_checked_individually_without_combined_parity(gate,monkeypatch):
    v=np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]])
    f=np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])
    mesh=np.r_[v,v*.1+[.2,0.,0.],v*.1+[-.2,0.,0.]],np.r_[f,f[:,::-1]+4,f[:,::-1]+8]
    seen=[]
    def contains(v,f,**kwargs):
        seen.append(gate.mesh_topology(v,f));return {'true_containment_verified':True}
    monkeypatch.setattr(gate,'true_hollow_containment',contains)
    result=gate.all_cavities_contained(mesh)
    assert result['all_cavities_contained'] and result['cavities_checked']==2
    assert all(len(x['components'])==2 for x in seen)
