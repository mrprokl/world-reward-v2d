"""No libigl/mesh assets/compilation: serialization and birth-mapping contracts."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def gate(monkeypatch):
    p=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(p))
    s=importlib.util.spec_from_file_location('wr_guarded_test',p/'guarded_mesh_gate.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def tetra():
    return np.array([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]]),np.array([[0,1,2],[0,3,1],[0,2,3],[1,3,2]])

def mapping(v,f):return {'source_vertices':len(v),'source_faces':len(f),'output_vertices':len(v),'output_faces':len(f),
                       'mapping_complete':True,'target_reached':True,'J':list(range(len(f))),'I':list(range(len(v)))}

def test_obj_exact_f64_roundtrip_exclusive_no_process(gate,tmp_path):
    v,f=tetra();v[0,0]=np.nextafter(v[0,0],2.);path=tmp_path/'m.obj';gate.write_obj(path,v,f)
    a,b=gate.read_obj(path);assert np.array_equal(a,v) and np.array_equal(b,f)
    with pytest.raises(FileExistsError):gate.write_obj(path,v,f)

@pytest.mark.parametrize('line',['vn 1 0 0','f -1 2 3','f 1/1 2/2 3/3','f 1 2 3 4','v nan 0 0'])
def test_obj_invalid_no_silent_healing(gate,tmp_path,line):
    v,f=tetra();p=tmp_path/'m.obj';gate.write_obj(p,v,f)
    p.write_text(p.read_text()+line+'\n')
    with pytest.raises(ValueError):gate.read_obj(p)

def test_identity_mapped_geometry_zero_error(gate):
    v,f=tetra();r=gate.mapped_geometry((v,f),(v.copy(),f.copy()),mapping(v,f))
    assert r['sampled_bidirectional_chamfer_diagonal_ratio']==0 and r['net_volume_relative_error']==0
    assert len(r['birthface_matched_shells'])==1 and not r['scale_or_pose_fitted']


def test_failed_numerical_gate_retains_actual_diagnostic_measurements(gate):
    v,f=tetra();r={}
    with pytest.raises(ValueError,match='gates failed'):
        gate.mapped_geometry((v,f),(v*.9,f),mapping(v,f),r)
    assert r['net_volume_relative_error']==pytest.approx(1-.9**3)
    assert r['birthface_matched_shells'][0]['relative_volume_error']>.05

@pytest.mark.parametrize('fault',['missing','bool_count','bad_J','bad_I','incomplete','target','reverse','scale'])
def test_bad_mapping_or_geometry_fail_closed(gate,fault):
    v,f=tetra();r=mapping(v,f);cv,cf=v.copy(),f.copy()
    if fault=='missing':r.pop('J')
    if fault=='bool_count':r['source_faces']=True
    if fault=='bad_J':r['J'][0]=99
    if fault=='bad_I':r['I'][0]=-1
    if fault=='incomplete':r['mapping_complete']=False
    if fault=='target':r['target_reached']=False
    if fault=='reverse':cf=cf[:,::-1]
    if fault=='scale':cv*=.9
    with pytest.raises(ValueError):gate.mapped_geometry((v,f),(cv,cf),r)

def test_birth_identity_cannot_pair_by_sorted_volume_or_delete_shell(gate):
    v,f=tetra();v=np.r_[v+[-3,0,0],v*.8+[3,0,0]];f=np.r_[f,f+4];r=mapping(v,f)
    assert len(gate.mapped_geometry((v,f),(v,f),r)['birthface_matched_shells'])==2
    bad=r|{'J':[0,1,2,3]*2}
    with pytest.raises(ValueError,match='merged or split'):gate.mapped_geometry((v,f),(v,f),bad)
    bad=r|{'I':list(range(4))*2}
    with pytest.raises(ValueError,match='crosses'):gate.mapped_geometry((v,f),(v,f),bad)

def test_wrapper_cpu_and_minimum_mounts(gate):
    s=Path(gate.__file__).with_name('run_guarded_mesh_gate.sh').read_text()
    assert '--gpus' not in s and '--network none' in s and '183s docker run' in s
    assert 'src=$ROOT/outputs' not in s and 'src=$ROOT/data' not in s
    assert 'mesh_guarded_qem.cpp' in s
