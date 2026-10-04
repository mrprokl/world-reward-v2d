"""Tiny byte/analytic mesh fixtures only: no datasets, models, CUDA or network."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import types

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
spec=importlib.util.spec_from_file_location('ycbv_objects_tests',REPO/'infra/ycbv_point_objects.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)



class TinyMesh:
    """Procedural arrays only; runtime uses actual installed trimesh separately."""
    def __init__(self,vertices,faces,process=False):self.vertices=np.asarray(vertices).copy();self.faces=np.asarray(faces).copy()
    @property
    def area(self):
        p=self.vertices[self.faces];return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).sum()/2
    @property
    def volume(self):
        p=self.vertices[self.faces];return float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
    @property
    def face_adjacency(self):return np.column_stack((np.arange(len(self.faces)-1),np.arange(1,len(self.faces))))
    @property
    def is_watertight(self):return len(self.faces)==12
    @property
    def is_winding_consistent(self):return True


def box():
    vertices=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)/2
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]],dtype=np.int64)
    return TinyMesh(vertices,faces)


@pytest.fixture
def fake_trimesh(monkeypatch):
    stub=types.SimpleNamespace(Trimesh=TinyMesh,graph=types.SimpleNamespace(connected_components=lambda adjacency,nodes:[nodes]))
    monkeypatch.setitem(sys.modules,'trimesh',stub)


def write(path,raw=b'own tiny bytes',mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True);
    if path.exists():path.chmod(0o644)
    path.write_bytes(raw);path.chmod(mode);return gate.identity(path,empty=True)


def json_file(path,value):return write(path,json.dumps(value).encode())


def pin_fixture():
    pin=dict(bytes=9,sha256='a'*64)
    producers=dict(bytes=9,sha256='a'*64,producer_revision='b'*40,script_sha256='c'*64)
    bundles=[]
    for scene in gate.SCENES:
        bundles.append(dict(scene_id=scene,
            rgb=dict(path=f'{gate.BASE}/inputs/scene_{scene:06d}_frame_000000.png',**pin),
            mask=dict(path=f'{gate.BASE}/automatic_masks_v1/scene_{scene:06d}/masks/1/000000.png',**pin),
            depth=dict(path=f'{gate.BASE}/depth_init_v1/scene_{scene:06d}_frame_000000.npz',**pin)))
    models={gate.OBJECT+'/checkpoints/'+n+'.ckpt':dict(bytes=z,sha256=h)for n,(z,h)in zip(gate.CKPTS,gate.CHECKPOINT_PINS)}
    models.update({gate.OBJECT+'/checkpoints/'+n+'.yaml':dict(bytes=z,sha256=h)for n,(z,h)in zip(gate.YAMLS,gate.YAML_PINS)})
    models.update({gate.OBJECT+'/LICENSE':gate.SOURCE_LICENSE_PIN,**{gate.WEIGHTS+'/torch_home/hub/checkpoints/'+n:pin for n in gate.REG4},gate.MOGE+'/blobs/'+'d'*64:dict(bytes=1256823446,sha256='da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f'),gate.MOGE+'/blobs/'+'e'*40:pin})
    return dict(schema='world_reward.ycbv_point_objects.pins.v1',inputs=dict(manifest=pin,acquisition_report=producers,mask_report=producers,depth_report=producers,bundles=bundles),
        runtime=dict(image_receipt=dict(path='results/ycbv-objects-runtime-'+('b'*40)+'/image.json',**pin),model_files=models,
            source_files={gate.DINO+'/'+n:pin for n in ('hubconf.py','LICENSE','MODEL_CARD.md','dinov2/__init__.py')},
            installed_sources={n:{'__init__.py':pin}for n in gate.MODULES},
            acquisition_receipts={n:pin for n in ('results/weights-acquisition.json','results/auxiliary-assets.json')},
            moge_links={gate.SNAPSHOT+'/model.pt':'../../blobs/'+'d'*64,gate.SNAPSHOT+'/README.md':'../../blobs/'+'e'*40}))


def test_fixed_primary_pins_match_original_binder_without_import():
    # Existing immutable source acts as independent literal evidence, not GPU import.
    spec=importlib.util.spec_from_file_location('wr_inventory_evidence',REPO/'infra/frontend_replica_inventory.py')
    existing=importlib.util.module_from_spec(spec);spec.loader.exec_module(existing)
    assert [(existing.CHECKPOINTS[n][1],existing.CHECKPOINTS[n][0])for n in gate.CKPTS]==list(gate.CHECKPOINT_PINS)
    assert [(existing.YAMLS[n][1],existing.YAMLS[n][0])for n in gate.YAMLS]==list(gate.YAML_PINS)
    assert gate.IMAGE==existing.IMAGES['objects'][1] and gate.OBJECT_REV==existing.OBJECT_REV and gate.DINO_REV==existing.DINO_REVS['dinov2']
    assert gate.validate_pins(pin_fixture())


@pytest.mark.parametrize('fault',['schema','extra','scene','path','bytes_bool','sha','report_extra','model','yaml','license','foreign_model','foreign_source','missing_module','install_path','moge_path'])
def test_independent_pin_contract_rejects_before_any_IO(monkeypatch,fault):
    pins=copy.deepcopy(pin_fixture());bundle=pins['inputs']['bundles'][0];runtime=pins['runtime']
    if fault=='schema':pins['schema']='other'
    elif fault=='extra':pins['truth']={}
    elif fault=='scene':bundle['scene_id']=True
    elif fault=='path':bundle['rgb']['path']='data/track_1/private.png'
    elif fault=='bytes_bool':bundle['mask']['bytes']=True
    elif fault=='sha':bundle['depth']['sha256']='A'*64
    elif fault=='report_extra':pins['inputs']['mask_report']['token']='not secret'
    elif fault=='model':runtime['model_files'][gate.OBJECT+'/checkpoints/ss_generator.ckpt']['sha256']='a'*64
    elif fault=='yaml':runtime['model_files'].pop(gate.OBJECT+'/checkpoints/pipeline.yaml')
    elif fault=='license':runtime['model_files'][gate.OBJECT+'/LICENSE']=dict(bytes=True,sha256='b'*64)
    elif fault=='foreign_model':runtime['model_files']['weights/body.pt']=dict(bytes=1,sha256='b'*64)
    elif fault=='foreign_source':runtime['source_files'][gate.DINO+'/data/private.json']=dict(bytes=1,sha256='b'*64)
    elif fault=='missing_module':runtime['installed_sources'].pop('moge')
    elif fault=='install_path':runtime['installed_sources']['moge']['../other.py']=dict(bytes=1,sha256='b'*64)
    else:runtime['moge_links']['weights/cache/other']='../../other'
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('Invalid pins reached filesystem'))
    with pytest.raises((ValueError,KeyError)):gate.validate_pins(pins)


@pytest.mark.parametrize('value',['/absolute','../up','a/../b','a//b','a\\b','a/./b','a\0b',''])
def test_relative_name_firewall(value):
    with pytest.raises(ValueError):gate.safe(value)


def test_byte_identity_receipt_before_JSON_and_duplicate_keys(tmp_path):
    file=tmp_path/'receipt';pin=write(file,b'{"ok":true}')
    assert gate.bound_json(file,pin)=={'ok':True}
    with pytest.raises(ValueError):gate.bound_json(file,dict(bytes=pin['bytes'],sha256='0'*64))
    with pytest.raises(ValueError):gate.strict('{"a":1,"a":2}')
    with pytest.raises(ValueError):gate.strict('{"a":NaN}')
    os.link(file,tmp_path/'alias')
    with pytest.raises(ValueError):gate.identity(file)


def input_fixture(tmp_path):
    pins=copy.deepcopy(pin_fixture());rows=[]
    for scene in gate.SCENES:
        for frame in range(96):
            filename=f'scene_{scene:06d}_frame_{frame:06d}.png'
            rows.append(dict(scene_id=scene,frame_id=frame,file=filename,sha256='f'*64,width=640,height=480))
    manifest=dict(schema='world-reward-ycbv-point-rgb-v1',revision='5c2c4aa229800355648cd268040aa814f8dc94f0',license='MIT',
        selection='first_three_sorted_scene_directories_first_96_contiguous_RGB_names_before_private_annotations',attribution='YCB-Video: Yu Xiang et al.; BOP conversion: Hodan et al.',images=rows)
    mask_rows=[];depth_rows=[]
    for bundle in pins['inputs']['bundles']:
        for kind in ('rgb','mask','depth'):
            actual=write(tmp_path/bundle[kind]['path'],('own '+kind+str(bundle['scene_id'])).encode());bundle[kind].update(actual)
        scene=bundle['scene_id'];manifest['images'][(scene-48)*96]['sha256']=bundle['rgb']['sha256']
        for frame in range(96):
            actual={k:bundle['mask'][k]for k in ('bytes','sha256')}if frame==0 else dict(bytes=8,sha256='a'*64)
            mask_rows.append(dict(scene_id=scene,frame_id=frame,file=f'scene_{scene:06d}/masks/1/{frame:06d}.png',rgb_sha256=bundle['rgb']['sha256'],**actual))
        depth_rows.append(dict(scene_id=scene,frame_id=0,file=Path(bundle['depth']['path']).name,rgb_sha256=bundle['rgb']['sha256'],**{k:bundle['depth'][k]for k in ('bytes','sha256')}))
    pins['inputs']['manifest']=json_file(tmp_path/gate.BASE/'inputs/manifest.json',manifest)
    receipts={}
    for role,stage,path in (('acquisition_report','external_ycbv_contiguous_rgb_only_acquisition','report.json'),('mask_report','public_ycbv_point_native_object_masks','automatic_masks_v1/report.json'),('depth_report','public_ycbv_three_frame_zero_native_MoGe2_preflight','depth_init_v1/report.json')):
        receipt=dict(stage=stage,status='pass',phase='complete',producer_revision='b'*40,script_sha256='c'*64,challenge_inputs_used=False)
        if role=='acquisition_report':receipt.update(source_rehashed_after=True,private_annotations_exported_as_inference_inputs=False,selected_frames=288,all_instances_retained=True,selection_before_private_annotation_values=True,license='MIT',public_manifest=pins['inputs']['manifest'])
        if role=='mask_report':receipt.update(all_inputs_sources_assets_outputs_rehashed=True,ground_truth_used=False,frames_completed=288,masks=mask_rows,budget_seconds=600,elapsed_seconds=2.)
        if role=='depth_report':receipt.update(sources_after_reverified=True,private_truth_read=False,outputs_completed=3,outputs=depth_rows,budget_seconds=300,GPU_budget_elapsed_seconds=3.,elapsed_seconds=3.01)
        pins['inputs'][role]=dict(**json_file(tmp_path/gate.BASE/path,receipt),producer_revision='b'*40,script_sha256='c'*64);receipts[role]=receipt
    return pins,manifest,receipts


def test_host_and_GPU_firewall_actual_producer_fields(tmp_path):
    pins,_,_=input_fixture(tmp_path)
    assert set(gate.inputs_proof(tmp_path,pins))=={'acquisition_report','mask_report','depth_report'}
    (tmp_path/gate.BASE/'report.json').unlink()
    assert set(gate.inputs_proof(tmp_path,pins,False))=={'mask_report','depth_report'}
    with pytest.raises(FileNotFoundError):gate.inputs_proof(tmp_path,pins)


@pytest.mark.parametrize('fault',['gt','mask_post','depth_post','zero_budget','missing_frame','duplicate_frame','mask_path','source_sha','RGB_mismatch'])
def test_actual_frontend_lineage_no_rescue(tmp_path,fault):
    pins,_,receipts=input_fixture(tmp_path);role='mask_report';row=receipts[role]
    if fault=='gt':row['ground_truth_used']=True
    elif fault=='mask_post':row['all_inputs_sources_assets_outputs_rehashed']=False
    elif fault=='depth_post':role='depth_report';row=receipts[role];row['sources_after_reverified']=False
    elif fault=='zero_budget':row['elapsed_seconds']=0
    elif fault=='missing_frame':row['masks'].pop()
    elif fault=='duplicate_frame':row['masks'][1]=copy.deepcopy(row['masks'][0])
    elif fault=='mask_path':row['masks'][0]['file']='../private.png'
    elif fault=='source_sha':row['script_sha256']='d'*64
    else:row['masks'][0]['rgb_sha256']='d'*64
    path=tmp_path/gate.BASE/('automatic_masks_v1/report.json'if role=='mask_report'else'depth_init_v1/report.json')
    pins['inputs'][role].update(json_file(path,row))
    with pytest.raises(ValueError):gate.inputs_proof(tmp_path,pins)


def camera():return np.array([[800.,0,320.],[0,800.,240.],[0,0,1.]])
def camera_dict():return dict(fx=800.,fy=800.,cx=320.,cy=240.,width=640,height=480)
def transform(scale=2.):return dict(rotation=[1.,0,0,0],translation=[.2,.3,2.],scale=[scale]*3)


def test_raw_canonical_geometry_bakes_scale_once_and_no_pose_or_budget(tmp_path,fake_trimesh):
    mesh=box();v=mesh.vertices.copy();f=mesh.faces.copy();path=tmp_path/'canonical.npz'
    result=gate.export_geometry(mesh,transform(),camera(),camera_dict(),path)
    with np.load(path,allow_pickle=False)as file:
        assert set(file.files)=={'vertices','faces','R0','t0','K','scale'}
        assert file['vertices'].dtype==np.float64 and file['faces'].dtype==np.int64 and file['scale'].shape==()
        assert np.array_equal(file['vertices'],v*2)and np.array_equal(file['faces'],f)
        assert np.array_equal(file['R0'],np.eye(3))and np.array_equal(file['t0'],[.2,.3,2.])and file['scale']==1.
    assert np.array_equal(mesh.vertices,v)and np.array_equal(mesh.faces,f)
    assert result['source_triangles_retained']and not result['geometry_budget_verified']
    assert result['topology_qualification']['component_count']==1
    assert result['topology_qualification']['self_intersections_verified']is False
    assert path.stat().st_mode&0o777==0o444
    with pytest.raises(FileExistsError):gate.export_geometry(mesh,transform(),camera(),camera_dict(),path)


@pytest.mark.parametrize('fault',['quaternion','translation','negative_scale','nonuniform','K','zeroarea','duplicate','nan','invalidindex'])
def test_rawgeometry_fails_without_repair(tmp_path,fault,fake_trimesh):
    mesh=box();t=transform();outK=camera_dict()
    if fault=='quaternion':t['rotation']=[2,0,0,0]
    elif fault=='translation':t['translation'][0]=float('nan')
    elif fault=='negative_scale':t['scale']=[-2]*3
    elif fault=='nonuniform':t['scale']=[2,3,2]
    elif fault=='K':outK['fx']=801
    elif fault=='zeroarea':mesh.vertices[mesh.faces[0,1]]=mesh.vertices[mesh.faces[0,0]]
    elif fault=='duplicate':mesh.faces=np.vstack((mesh.faces,mesh.faces[0]))
    elif fault=='nan':mesh.vertices[0,0]=np.nan
    else:mesh.faces[0,0]=1000
    with pytest.raises(ValueError):gate.export_geometry(mesh,t,camera(),outK,tmp_path/'canonical.npz')
    assert not(tmp_path/'canonical.npz').exists()


def test_uniform_native_roundoff_uses_first_scale_never_average():
    t=transform();t['scale']=[2.,2.000001,2.]
    R,xyz,s=gate.native_pose(t,camera(),camera_dict())
    assert s==2. and np.array_equal(R,np.eye(3))and np.array_equal(xyz,t['translation'])


def test_open_surface_all_triangles_retained_but_not_qualifiedclosed(tmp_path,fake_trimesh):
    mesh=box();mesh=TinyMesh(mesh.vertices,mesh.faces[:-1],process=False)
    result=gate.export_geometry(mesh,transform(1.),camera(),camera_dict(),tmp_path/'canonical.npz')
    assert result['faces']==11 and result['topology_qualification']['watertight']is False


def source_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path);rev='a'*40;code=tmp_path/'jobs'/rev/gate.ENTRY/'code'
    for name in(*gate.HELPERS,gate.PIN_FILE):write(code/name,b'own source '+name.encode())
    for name,raw in (('revision',(rev+'\n').encode()),('source-sha256',('b'*64+'\n').encode())):write(code.parent/name,raw,0o644)
    for folder in sorted([code,*[p for p in code.rglob('*')if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):folder.chmod(0o555)
    return code,rev


def test_complete_canonical_readonly_source_and_marker_preserved(tmp_path,monkeypatch):
    code,rev=source_fixture(tmp_path,monkeypatch);original=gate.source(code,rev)
    assert gate.source(code,rev,True)['helpers']==original['helpers']
    assert(code.parent/'revision').stat().st_mode&0o777==0o644
    (code/'infra/ycbv_point_objects.py').chmod(0o644)
    with pytest.raises(ValueError):gate.source(code,rev)


def test_foreign_recipe_not_in_GPU_closure(tmp_path,monkeypatch):
    code,rev=source_fixture(tmp_path,monkeypatch);(code/'configs').chmod(0o755);write(code/'configs/renderer_recipe.json',b'own unused recipe');(code/'configs').chmod(0o555)
    assert gate.source(code,rev)
    with pytest.raises(ValueError):gate.source(code,rev,True)


def test_missing_future_facts_fail_before_output_GPU_or_model(tmp_path,monkeypatch):
    code,rev=source_fixture(tmp_path,monkeypatch);(tmp_path/gate.BASE).mkdir(parents=True)
    monkeypatch.setattr(os,'geteuid',lambda:0)
    class Uname:nodename='scenesmith-ncc-h100-01'
    monkeypatch.setattr(os,'uname',lambda:Uname())
    monkeypatch.setattr(gate,'command',lambda *a,**k:pytest.fail('Unknown future facts reached runtime'))
    with pytest.raises((ValueError,json.JSONDecodeError)):gate.launch(tmp_path,code,rev)
    assert not(tmp_path/gate.BASE/gate.OUTPUT).exists()


def test_installed_complete_hashes_not_assumed_commit_parity(tmp_path,monkeypatch):
    pins=pin_fixture();folders={}
    for index,module in enumerate(gate.MODULES):
        folder=tmp_path/str(index);write(folder/'__init__.py',b'own source');pins['runtime']['installed_sources'][module]={'__init__.py':gate.identity(folder/'__init__.py')};folders[module]=folder
    class Spec:
        def __init__(self,path):self.submodule_search_locations=[str(path)]
    monkeypatch.setattr(gate.util,'find_spec',lambda name:Spec(folders[name]))
    assert gate.installed_proof(pins)==pins['runtime']['installed_sources']
    write(folders['moge']/'unknown.py',b'foreign source')
    with pytest.raises(ValueError):gate.installed_proof(pins)


def test_runtime_mount_graph_only_exact_snapshot_and_repoblobs(tmp_path):
    pins=pin_fixture();(tmp_path/gate.SNAPSHOT).mkdir(parents=True);(tmp_path/gate.MOGE/'blobs').mkdir()
    for name,target in pins['runtime']['moge_links'].items():(tmp_path/name).symlink_to(target)
    for name in pins['runtime']['model_files']:
        if name.startswith(gate.MOGE+'/blobs/'):write(tmp_path/name)
    paths=gate.runtime_mounts(tmp_path,pins)
    assert tmp_path/gate.SNAPSHOT in paths and tmp_path/gate.MOGE/'blobs'in paths
    assert tmp_path/gate.HF not in paths and tmp_path/'weights'not in paths
    write(tmp_path/gate.MOGE/'blobs/unknown',b'not allowed')
    with pytest.raises(ValueError):gate.runtime_mounts(tmp_path,pins)


def test_exact_docker_argv_blind_mounts_no_private_or_full_assets(tmp_path,monkeypatch):
    pins=pin_fixture();code=tmp_path/'code';out=tmp_path/'out'
    monkeypatch.setattr(gate,'runtime_mounts',lambda root,pins:[root/'exact_public_model'])
    for mode in('probe','gpu'):
        args=gate.docker_arguments(tmp_path,code,'b'*40,pins,out,'c'*64,10.,mode,'owned')
        assert args[args.index('--entrypoint')+1:args.index('--entrypoint')+4]==['/usr/bin/env',gate.IMAGE,'-i']
        assert args[args.index('--user')+1]=='0:0'and '--network'in args and args[args.index('--network')+1]=='none'
        assert '--cap-drop'in args and '--security-opt'in args and '--read-only'in args
        assert ('--gpus'in args)==(mode=='gpu')
        mounts=[args[i+1]for i,v in enumerate(args)if v=='--mount']
        assert not any('eval_private'in m or '/data,'in m or '/vendor,'in m or '/weights,'in m or 'image-sam3d-runtime.json'in m for m in mounts)
        assert not any('src='+str(tmp_path/gate.BASE/'report.json')+','in m for m in mounts)
        assert not any('src='+str(code)+','in m for m in mounts)
        assert not any('src='+str(tmp_path/gate.HF)+','in m for m in mounts)
        if mode=='probe':assert not any('/inputs/'in m or 'exact_public_model'in m for m in mounts)
        else:assert sum('/inputs/scene_'in m for m in mounts)==3
        assert 'HOME=/tmp'in args and 'HF_HUB_OFFLINE=1'in args and '/opt/conda/bin/python'in args


def test_cleanup_will_not_kill_foreign_container(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();write(out/'.container.cid',b'a'*64)
    calls=[]
    def command(args,*unused):
        calls.append(args)
        if args[1]=='ps':return b'a'*64
        return b'foreign|/wrong|other|revision'
    monkeypatch.setattr(gate,'command',command)
    with pytest.raises(ValueError):gate.cleanup(out,'b'*40,'owned',100)
    assert not any('rm'in a for a in calls)


def test_wrapper_arguments_guard_existing_lock_and_sourceclosure(tmp_path):
    script=REPO/'infra/run_ycbv_point_objects.sh'
    assert subprocess.run(['bash','-n',str(script)],capture_output=True).returncode==0
    env={'PATH':'/usr/bin:/bin','HOME':str(tmp_path),'WR_ROOT':str(tmp_path),'WR_CODE':str(tmp_path),'WR_CODE_REVISION':'a'*40}
    for argv in ([],['--help'],['--episode','0']):
        result=subprocess.run(['bash',str(script),*argv],env=env,capture_output=True)
        assert result.returncode!=0
    text=script.read_text();assert 'exec 9<'in text and '9>'not in text and '/usr/bin/env -i'in text and ' -I -B 'in text
    spec=importlib.util.spec_from_file_location('closure_evidence',REPO/'infra/azure_job.py');job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in ('infra','src','configs')for p in(REPO/folder).rglob('*')if p.is_file()and p.suffix in('.py','.sh','.json')}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    closure=job.runtime_bundle_paths(files,'infra/run_ycbv_point_objects.sh')
    assert set(gate.HELPERS)<=set(closure)
    assert not any(n in closure for n in ('infra/ycbv_point_acquire.py','infra/bridge_rgb_anchor_render.py','infra/object_smoke.py','infra/frontend_replica_inventory.py'))


@pytest.mark.parametrize('tamper',[False,True])
def test_mocked_host_end_to_end_seals_after_final_post_not_self_success(tmp_path,monkeypatch,tamper):
    code,rev=source_fixture(tmp_path,monkeypatch);pins=pin_fixture();pinpath=code/gate.PIN_FILE;pinpath.chmod(0o644);pinpath.write_text(json.dumps(pins));pinpath.chmod(0o444)
    (tmp_path/gate.BASE).mkdir(parents=True);before=gate.source(code,rev)
    monkeypatch.setattr(os,'geteuid',lambda:0)
    class Uname:nodename='scenesmith-ncc-h100-01'
    monkeypatch.setattr(os,'uname',lambda:Uname())
    monkeypatch.setattr(gate,'lock_identity',lambda *a:[1,2,3])
    image={'Id':gate.IMAGE,'Architecture':'amd64','Os':'linux','RootFS':{'Type':'layers','Layers':['sha256:'+'b'*64]}}
    monkeypatch.setattr(gate,'actual_image',lambda *a:image)
    host={'runtime':{'rootfs':image['RootFS']}}
    calls=[]
    def hostproof(*unused):
        calls.append('hostproof')
        return pins,host if calls.count('hostproof')==1 or not tamper else {'runtime':{'rootfs':{'Type':'layers','Layers':[]}}}
    monkeypatch.setattr(gate,'host_proof',hostproof)
    monkeypatch.setattr(gate,'runtime_mounts',lambda *a:[])
    def metadata(args,*unused):
        calls.append(args[0]+':'+args[1]);return b''
    monkeypatch.setattr(gate,'command',metadata)
    def subprocess_run(args,**kwargs):
        assert kwargs['pass_fds']==(9,)and kwargs['input']==gate.BOOTSTRAP and kwargs['timeout']<=900
        assert set(kwargs['env'])=={'PATH','HOME','DOCKER_HOST'}
        out=tmp_path/gate.BASE/gate.OUTPUT;cid=Path(args[args.index('--cidfile')+1]);write(cid,b'd'*64,0o400)
        if 'WR_OBJECTS_MODE=gpu'in args:
            rows=[]
            for scene in gate.SCENES:
                files={n:write(out/f'scene_{scene:06d}'/n,n.encode())for n in ('object.glb','transform.json','intrinsics.json','canonical.npz')}
                rows.append(dict(scene_id=scene,frame_id=0,files=files))
            native=dict(stage=gate.STAGE,status='pass',phase='complete',source_helpers=before['helpers'],producer_revision=rev,
                native_calls_attempted=3,native_calls_returned=3,native_calls_completed=3,outputs_completed=3,outputs=rows)
            json_file(out/'.native-report.json',native)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(gate.subprocess,'run',subprocess_run)
    if tamper:
        with pytest.raises(SystemExit):gate.launch(tmp_path,code,rev)
    else:gate.launch(tmp_path,code,rev)
    report=gate.strict((tmp_path/gate.BASE/gate.OUTPUT/'report.json').read_bytes())
    assert report['status']==('fail'if tamper else'pass')
    assert report['source_rehashed_after']is(not tamper)
    assert 0<report['GPU_budget_elapsed_seconds']<=900
    assert report['human_scalar_used']is False and report['hand_observations_used']is False
    assert report['acquisition_receipt_checked_host_only']is True and report['raw_image_receipt_mounted']is False
    assert (tmp_path/gate.BASE/gate.OUTPUT/'report.json').stat().st_mode&0o777==0o400
    with pytest.raises(ValueError):gate.launch(tmp_path,code,rev)


def test_three_native_invocations_cached_api_and_exact_original_XYZ_K(tmp_path,monkeypatch,fake_trimesh):
    pins=pin_fixture();out=tmp_path/'out';out.mkdir();points=np.zeros((3,4,3),dtype=np.float32);points[...,2]=2
    rgb=np.zeros((3,4,3),dtype=np.uint8)
    monkeypatch.setattr(gate,'bundle_arrays',lambda *a:(rgb,np.full((3,4),255,dtype=np.uint8),{'points':points},camera()))
    monkeypatch.setattr(gate,'bound_json',lambda *a:{'outputs':[{'scene_id':s,'frame_id':0,'decoded_RGB_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}for s in gate.SCENES]})
    monkeypatch.setattr(sys.modules['trimesh'],'load',lambda *a,**k:box(),raising=False)
    calls=[]
    def native(*args,**kwargs):
        calls.append((args,kwargs))
        assert kwargs==dict(seed=0,stage1_only=False,with_mesh_postprocess=False,with_texture_baking=False,with_layout_postprocess=False,use_vertex_color=True,
            pointmap_path=str(Path(args[2]).parent/'pointmap.npy'),pointmap_intrinsics_path=str(Path(args[2]).parent/'pointmap_intrinsics.json'))
        assert np.array_equal(np.load(kwargs['pointmap_path'],allow_pickle=False),points)
        assert gate.strict(Path(kwargs['pointmap_intrinsics_path']).read_bytes())==camera_dict()
        write(Path(args[2]),b'own native GLB')
        json_file(Path(args[3]),transform());json_file(Path(args[4]),camera_dict())
    report=dict(native_calls_attempted=0,native_calls_returned=0,native_calls_completed=0,outputs=[])
    gate.observe(tmp_path,pins,out,report,lambda:None,native)
    assert len(calls)==3 and all(report[k]==3 for k in ('native_calls_attempted','native_calls_returned','native_calls_completed'))
    for bundle,row,(args,kwargs)in zip(pins['inputs']['bundles'],report['outputs'],calls):
        assert row['input_bundle']==bundle and row['camera_unchanged']and args[0]==str(tmp_path/bundle['rgb']['path'])and args[1]==str(tmp_path/bundle['mask']['path'])
        assert set(row['files'])=={'object.glb','transform.json','intrinsics.json','canonical.npz'}
        assert not Path(kwargs['pointmap_path']).exists()and not Path(kwargs['pointmap_intrinsics_path']).exists()


def test_full_native_grid_K_is_unrounded_normalized_FP32_no_geometry_fill(monkeypatch):
    # Analytic memory arrays only, no saved images/model/dataset. One native grid.
    y,x=np.mgrid[:480,:640];depth=np.full((480,640),2,dtype=np.float32)
    intrinsics=np.array([[800/640,0,.5],[0,800/480,.5],[0,0,1]],dtype=np.float32)
    K=np.diag([640.,480.,1.])@intrinsics.astype(np.float64)
    points=np.stack(((x+.5-320)/K[0,0]*2,(y+.5-240)/K[1,1]*2,np.full_like(x,2)),axis=-1).astype(np.float32)
    valid=np.ones((480,640),dtype=bool);valid[0,0]=False;depth[0,0]=np.nan;points[0,0]=np.inf
    arrays=dict(depth=depth,points=points,mask=valid,intrinsics=intrinsics,frame_index=np.array(0,dtype=np.int64))
    class Npz:
        files=list(arrays)
        def __enter__(self):return self
        def __exit__(self,*unused):pass
        def __getitem__(self,key):return arrays[key]
    class Image:
        format='PNG';size=(640,480)
        def __init__(self,mask):self.mode='L'if mask else'RGB';self.array=np.full((480,640),255,dtype=np.uint8)if mask else np.zeros((480,640,3),dtype=np.uint8)
        def __enter__(self):return self
        def __exit__(self,*unused):pass
        def __array__(self,dtype=None,copy=None):return np.asarray(self.array,dtype=dtype)
    monkeypatch.setattr(np,'load',lambda *a,**k:Npz())
    monkeypatch.setitem(sys.modules,'PIL',types.SimpleNamespace(Image=types.SimpleNamespace(open=lambda p:Image('/masks/'in str(p)))))
    bundle=pin_fixture()['inputs']['bundles'][0]
    monkeypatch.setattr(gate,'identity',lambda p,*a,**k:next({k:b[k]for k in('bytes','sha256')}for b in(bundle[t]for t in('rgb','mask','depth'))if b['path']==str(p).split('/root/')[-1]))
    before={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in arrays.items()}
    rgb,mask,actual,cameraK=gate.bundle_arrays(Path('/root'),bundle)
    assert np.array_equal(cameraK,K)and cameraK[1,1]==float(intrinsics[1,1])*480 and cameraK[1,1]!=800.
    assert actual['points']is points and actual['depth']is depth and actual['mask']is valid
    assert before=={k:hashlib.sha256(v.tobytes()).hexdigest()for k,v in arrays.items()}
    assert mask.shape==(480,640)and rgb.shape==(480,640,3)


@pytest.mark.parametrize('path',['results/image-sam3d-runtime.json','results/ycbv-objects-runtime-main/image.json','results/ycbv-objects-runtime-'+('B'*40)+'/image.json','results/ycbv-objects-runtime-'+('b'*40)+'/../image.json','results/ycbv-objects-runtime-'+('b'*40)+'/image.json/extra',True])
def test_image_receipt_only_safe_measured_projection_namespace(path):
    pins=pin_fixture();pins['runtime']['image_receipt']['path']=path
    with pytest.raises(ValueError):gate.validate_pins(pins)
