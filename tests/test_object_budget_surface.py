"""Tiny real-array graph/host spies: no Azure/native solver/model calls."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'infra'))
    spec=importlib.util.spec_from_file_location('surface_producer_test',ROOT/'infra/object_budget_solid.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('args',[['--episode','2','--domain','surface'],['--domain','surface','--control','surface_consumer_v1']])
def test_exact_optin_route_and_fresh_namespace(gate,monkeypatch,tmp_path,args):
    rev='a'*40;monkeypatch.setattr(gate,'ROOT',tmp_path)
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(tmp_path/'code'));monkeypatch.setenv('WR_CODE_REVISION',rev)
    monkeypatch.setattr(sys,'argv',['driver',*args]);monkeypatch.setattr(gate,'surface_modules',lambda _:())
    calls=[];monkeypatch.setattr(gate,'surface_host',lambda *a,**kw:calls.append((a,kw))or 0)
    assert gate.main()==0 and len(calls)==1
    assert calls[0][1].get('control',False)==('--control'in args)


@pytest.mark.parametrize('args',[
    ['--episode','2','--domain','solid'],['--episode','2','--domain','surface','--query-requalification'],
    ['--episode','2','--domain','surface','--domain','surface'],['--episode','02','--domain','surface'],
    ['--domain','surface','--control','surface_consumer_v1','--episode','2'],
    ['--domain','surface','--control','arbitrary']])
def test_unsafe_or_combined_surface_flags_rejected(gate,monkeypatch,args):
    monkeypatch.setattr(sys,'argv',['driver',*args])
    with pytest.raises(ValueError):gate.main()


def test_same_graph_identity_preserves_orphan_canonical_payload(gate,monkeypatch,tmp_path):
    trimesh=pytest.importorskip('trimesh',reason='Real Trimesh graph runs only in qualified Azure CPU runtime')
    import mediapipe_cpu_runtime_verify as rt
    import official_pack_geometry as packed
    from world_reward.surface_identity import SurfaceIdentity
    from surface_identity_qualify import surface_signature
    v=np.array([[0,0,0],[.25,0,0],[0,.25,0],[4,5,6]],np.float32);f=np.array([[0,1,2]],np.int64)
    metadata={};sources={'video':'1'*64}
    import object_budget_endpoint as ep
    monkeypatch.setattr(ep,'prerequisites',lambda *_:(dict(video_sha256='1'*64),sources,.5))
    native_root=tmp_path/'vendor';native_root.mkdir()
    official=SimpleNamespace(check_runtime_packages=lambda _:None)
    def raw(path,*_):
        if path.name in ('object.glb','original_source.glb'):return v,f,{'source_identity':{'sha256':'1'*64,'bytes':1}}
        mesh=trimesh.load(path,force='mesh',process=False)
        return mesh.vertices.astype(np.float32),mesh.faces.astype(np.int64),{'source_identity':rt.identity(path)}
    monkeypatch.setattr(gate,'surface_raw_source',raw)
    source_original=tmp_path/'outputs/episode_000002/object_grounded/object.glb'
    source_original.parent.mkdir(parents=True);source_original.write_bytes(b'opaque predicted source')
    monkeypatch.setattr(gate,'surface_trimesh_sources',lambda *_:gate.SURFACE_TRIMESH_SOURCES)
    # Original helper ABI spy pads a reordered/welded copy, not the retained payload.
    kit=tmp_path/'vendor/v2d_submission_kit/v2dlb';kit.mkdir(parents=True)
    helper=kit/'mesh_budget.py';helper.write_text('')
    scope={}
    exec(compile('def budget_mesh(path,faces,vertices):\n return callback(path)\n',str(helper),'exec'),scope)
    def budget(path):
        mesh=trimesh.load(path,force='mesh',process=False);mv=mesh.vertices[:3].copy();mf=mesh.faces.copy()
        return np.vstack((mv,np.repeat(mv[:1],4096-len(mv),axis=0))),np.vstack((mf,np.zeros((4096-len(mf),3),np.int64)))
    scope['callback']=budget
    monkeypatch.setitem(sys.modules,'v2dlb.mesh_budget',SimpleNamespace(budget_mesh=scope['budget_mesh']))
    monkeypatch.setattr(gate,'ROOT',tmp_path)
    q=SimpleNamespace(modules=lambda:(SimpleNamespace(surface_signature=surface_signature),None,official,None),orientation=lambda *_:None,
        verify_mapping=lambda *_:pytest.fail('Identity no QEM'))
    def write(p,r):p.write_text(__import__('json').dumps(r));p.chmod(0o444)
    build=SimpleNamespace(write_json=write)
    report={'qualification':{'runtime':{'pins':{'versions':{}}}}}
    gate.surface_produce(2,ROOT,tmp_path/'no_solver',tmp_path,lambda:100,report,rt,q,build)
    with np.load(tmp_path/'geometry.npz')as a:
        assert len(a['vertices'])==4096 and np.array_equal(a['vertices'][:4],v.astype(np.float64)*.5)
        assert np.array_equal(a['faces'][0],f[0])and np.all(a['faces'][1:]==0)
    with np.load(tmp_path/'candidate_geometry.npz')as a:
        assert set(a.files)=={'source_vertices','source_faces','candidate_vertices','candidate_faces','canonical_vertices','canonical_faces'}
        assert np.array_equal(a['source_vertices'],v)and len(a['canonical_vertices'])==4
    assert report['compiler']['qem_calls']==0 and report['compiler']['official_unreferenced_source_vertices']==1
    assert report['compiler']['official_pack_fidelity']['raw_oriented_triangles_exact']


def test_shell_static_closure_and_no_numeric_solid_changes(gate):
    shell=(ROOT/'infra/run_object_budget_solid.sh').read_text()
    assert '740s'in shell and '1960s'in shell and 'surface_consumer_v1'in shell
    assert '/src/world_reward/surface_budget.py'in shell
    assert gate.SURFACE_SECONDS==600 and gate.SURFACE_HOST_SECONDS==700
    assert gate.SURFACE_IMAGE=='sha256:1a04b1930f713ef9ffb411489e80ddebbce59a5ce26e713add4095cd9b5303f0'


def test_cleanup_rejects_daemon_failure(gate,monkeypatch,tmp_path):
    import subprocess,mediapipe_cpu_runtime_verify as rt
    cid=tmp_path/'cid';cid.write_text('a'*64)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=b'',stderr=b'Cannot connect to Docker daemon'))
    with pytest.raises(ValueError):gate.surface_cleanup('owned','b'*40,cid,rt)


def test_cleanup_accepts_exact_lowercase_separated_absence(gate,monkeypatch,tmp_path):
    import subprocess,mediapipe_cpu_runtime_verify as rt
    cid=tmp_path/'cid';cid.write_text('a'*64+'\n')
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=b'\n',stderr=('error: no such object: '+'a'*64+'\n').encode()))
    gate.surface_cleanup('owned','b'*40,cid,rt)


@pytest.mark.parametrize('domain,control',[('solid','surface_consumer_v1'),('surface',True),('surface','other')])
def test_shared_real_profile_guard_negative(gate,domain,control):
    with pytest.raises(ValueError):gate.surface_profile(domain,control)


def test_control_pin_before_recipe_and_no_relabel(gate,tmp_path):
    import mediapipe_cpu_runtime_verify as rt
    path=tmp_path/gate.SURFACE_CONTROL;path.parent.mkdir()
    path.write_bytes((ROOT/gate.SURFACE_CONTROL).read_bytes());path.chmod(0o444)
    assert gate.surface_control_protocol(tmp_path,rt)['frames']==96
    path.chmod(0o644);path.write_bytes(path.read_bytes()+b' ');path.chmod(0o444)
    with pytest.raises(ValueError):gate.surface_control_protocol(tmp_path,rt)


def test_inclusive_receipt_write_cannot_leave_pass(gate,tmp_path):
    import json
    report={'status':'pass'};calls=[]
    def left():
        calls.append(1)
        if len(calls)>1:raise TimeoutError('elapsed')
        return 1
    path=tmp_path/'report.json'
    with pytest.raises(TimeoutError):gate.surface_report(path,report,None,left)
    assert json.loads(path.read_bytes())['status']=='fail'
    with pytest.raises(FileExistsError):gate.surface_report(path,{},None,lambda:1)


@pytest.mark.parametrize('fault',['unknown','symlink','hardlink'])
def test_scratch_rejects_foreign_without_deletion(gate,tmp_path,fault):
    import os,mediapipe_cpu_runtime_verify as rt
    work=tmp_path/'work';work.mkdir();owner=work.lstat();leaf=work/'geometry.npz';leaf.write_bytes(b'owned')
    if fault=='unknown':(work/'foreign').write_bytes(b'foreign')
    elif fault=='symlink':leaf.unlink();leaf.symlink_to(tmp_path/'foreign')
    else:os.link(leaf,work/'mapping.json')
    with pytest.raises(ValueError):gate.surface_remove_work(work,owner,rt)
    assert work.exists()


def test_cleanup_found_container_binds_image_and_owner(gate,monkeypatch,tmp_path):
    import json,subprocess,mediapipe_cpu_runtime_verify as rt
    cid=tmp_path/'cid';cid.write_text('a'*64);rev='b'*40
    row={'Id':'a'*64,'Name':'/owned','Image':gate.SURFACE_IMAGE,'Config':{'Labels':{'world_reward.surface_budget.owner':rev}}}
    calls=[]
    def run(args,**kw):
        calls.append(args)
        if args[1]=='rm':return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
        if len(calls)>1:return SimpleNamespace(returncode=1,stdout=b'',stderr=('error: no such object: '+'a'*64).encode())
        return SimpleNamespace(returncode=0,stdout=json.dumps([row]).encode(),stderr=b'')
    monkeypatch.setattr(subprocess,'run',run);gate.surface_cleanup('owned',rev,cid,rt)
    assert calls[1]==['docker','rm','-f','a'*64]
    calls.clear();row['Image']='sha256:'+'c'*64
    with pytest.raises(ValueError):gate.surface_cleanup('owned',rev,cid,rt)
    assert len(calls)==1



def test_control_full96_saved_poses_survive_npz_handles(gate,monkeypatch,tmp_path):
    import json,mediapipe_cpu_runtime_verify as rt
    from world_reward.surface_pose_geometry import camera_roundtrip
    config=json.loads((ROOT/gate.SURFACE_CONTROL).read_bytes())
    v=np.array(config['source']['vertices'],np.float32)*.5;f=np.array(config['source']['faces'],np.int64)
    angle=.2;A=np.eye(4);A[:3,:3]=[[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]];A[:3,3]=[.03,-.02,.01];A=A.astype(np.float32).astype(np.float64)
    loaded=(A@np.column_stack((v,np.ones(len(v)))).T).T[:,:3]
    class Mesh:
        def __init__(self,*a,**kw):self.vertices=loaded;self.faces=f
        def export(self,path):path.write_bytes(b'authored tiny source')
    class Scene:
        graph={'object':(A,'object')}
        def add_geometry(self,*a,**kw):pass
        def export(self,path):path.write_bytes(b'authored aligned source')
    module=SimpleNamespace(Trimesh=Mesh,Scene=Scene,load=lambda *a,**kw:Scene()if kw.get('force')=='scene'else Mesh(),
        transformations=SimpleNamespace(fix_rigid=lambda *a,**kw:A))
    monkeypatch.setitem(sys.modules,'trimesh',module)
    import mesh_precision_diagnostic as precision
    monkeypatch.setattr(precision,'raw_glb',lambda _:( [(v,f)],loaded[f],[]))
    monkeypatch.setattr(gate,'surface_control_protocol',lambda *_:config)
    monkeypatch.setattr(gate,'surface_trimesh_sources',lambda *_:{'verified':'spy'})
    def produce(e,code,binary,work,left,report,*args,**kw):
        assert e==-1 and report['all_control_arrays_frozen_before_native_calls']
        np.savez(work/'candidate_geometry.npz',canonical_vertices=v,canonical_faces=f)
        np.savez(work/'geometry.npz',vertices=np.vstack((v.astype(np.float64),np.repeat(v[:1].astype(np.float64),4096-len(v),axis=0))),
            faces=np.vstack((f,np.zeros((4096-len(f),3),np.int64))))
        report.update(compiler={'method':'identity','qem_calls':0},installed_trimesh_sources={'verified':'spy'})
    monkeypatch.setattr(gate,'surface_produce',produce)
    official=SimpleNamespace(native_mesh_sources=lambda _:(tmp_path,{}),
        isolated_source_function=lambda path,name,scope:lambda _:(loaded.astype(np.float32),f))
    q=SimpleNamespace(modules=lambda:(None,None,official,None))
    import world_reward.surface_pose_geometry as pure
    seen=[]
    def serialized(*args,**kwargs):return kwargs['native_vertices'],kwargs['native_faces'],{'native_loader_spy':True}
    monkeypatch.setattr(pure,'serialized_mesh',serialized)
    def roundtrip(*args):seen.append(args[4]);return camera_roundtrip(*args)
    monkeypatch.setattr(pure,'camera_roundtrip',roundtrip)
    report={};gate.surface_control_produce(ROOT,tmp_path/'not_executed',tmp_path,lambda:100,report,rt,q,None)
    assert report['control']['frames']==96 and report['control']['frame_index']==list(range(96))
    assert report['control']['negative_controls_rejected']==config['negative_controls']
    assert type(seen[0])is np.ndarray and seen[0].dtype==np.float32 and seen[0].shape==(96,4,4)
    assert len(report['control_source_array_sha256'])==7 and report['control']['QEM_calls']==0


def test_original_source_readonly_copy_never_changes_historical_mode(gate,tmp_path):
    import mediapipe_cpu_runtime_verify as rt
    original=tmp_path/'original.glb';original.write_bytes(b'opaque bytes, no mesh interpretation')
    original.chmod(0o644);before=original.stat()
    work=tmp_path/'work';work.mkdir()
    copied,pin=gate.surface_readonly_source(original,work,rt)
    assert copied.name=='original_source.glb' and copied.read_bytes()==original.read_bytes()
    assert copied.stat().st_mode&0o777==0o444 and original.stat()==before
    assert pin==rt.identity(original,readonly=False)
    with pytest.raises(FileExistsError):gate.surface_readonly_source(original,work,rt)
