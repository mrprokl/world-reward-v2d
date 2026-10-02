import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def modules(monkeypatch):
 p=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(p));result=[]
 for name in ['object_motion_track','object_motion_evaluate']:
  s=importlib.util.spec_from_file_location('wr_'+name,p/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);result.append(m)
 return result

def test_real_triangle_attachment_camera_pixel_centres_not_gt(modules):
 track,_=modules
 # One perspective ray at (256.5,192.5), same as array pixel(256,192).
 points=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]])
 w=np.array([.249,.250,.501]);p=(points*w[:,None]).sum(0)
 desired=np.array([.5,.5])/640*2
 delta=desired-p[:2];points[:,:2]+=delta
 face=np.full((384,512),-1);face[192,256]=0;bary=np.zeros((384,512,3));bary[192,256]=w
 result=track.raster_attachments(points,np.array([[0,1,2]]),np.array([[256,192]]),face,bary)
 assert np.allclose(result[0],[desired[0],desired[1],2])
 bary[192,256]=[1.,0,0]
 with pytest.raises(ValueError,match='pixel-centre'):track.raster_attachments(points,np.array([[0,1,2]]),np.array([[256,192]]),face,bary)

def test_surface_cd_no_alignment_sum_cm_and_relativeSO3(modules):
 _,ev=modules;a=np.array([[0.,0,0],[0.,1.,0],[0.,0,1.]])
 assert ev.cd(a,a)==0 and ev.cd(a,a+[1.,0,0])==pytest.approx(200)
 assert ev.rotation_error(np.eye(3),np.diag([-1.,-1.,1.]))==pytest.approx(180)

def case(b,a):return {'baseline_mean_camera_cd_cm':b,'candidate_mean_camera_cd_cm':a,'baseline_fast_camera_cd_cm':b,'candidate_fast_camera_cd_cm':a,'anchor_closed_oriented_positive_volume':True}

def test_predeclared_threeobject_camera_fast_geometry_gate(modules):
 _,ev=modules;c=[case(10,9),case(20,18),case(30,27)]
 assert ev.decision(c)['synthetic_hypothesis_supported']
 c[1]['candidate_fast_camera_cd_cm']=21
 assert not ev.decision(c)['synthetic_hypothesis_supported']
 c[1]['candidate_fast_camera_cd_cm']=18;c[2]['anchor_closed_oriented_positive_volume']=False
 assert not ev.decision(c)['synthetic_hypothesis_supported']
 with pytest.raises(ValueError):ev.decision(c[:2])

def test_truth_only_evaluator_no_weights_model_mount(modules):
 track,ev=modules;tw=Path(track.__file__).with_name('run_object_motion_track.sh').read_text();ew=Path(ev.__file__).with_name('run_object_motion_evaluate.sh').read_text()
 assert 'eval_private' not in tw and 'weights' not in tw and 'vendor' not in tw
 assert '--gpus' in tw and '--gpus' not in ew and 'weights' not in ew
 assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in ew
 assert 'per-frame alignment' in Path(ev.__file__).read_text()
 for module in modules:
  with pytest.raises(SystemExit):module.main(['--oracle'])
