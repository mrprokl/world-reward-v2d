"""Tiny frozen-export/render fixtures only, no real videos/models/cloud."""
import copy
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
import reconstruction_overview as overview
import reconstruction_preview as preview


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location('wr_overview_test',REPO/'infra/reconstruction_overview.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def setup(m,tmp_path,monkeypatch):
    root=tmp_path/'root';rev='a'*40;code=root/'jobs'/rev/m.ENTRY/'code';(root/'results').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'__file__',str(code/m.HELPERS[0]))
    for n in m.HELPERS:
        p=code/n;p.parent.mkdir(parents=True,exist_ok=True)
        if n.startswith('infra/'):p.write_bytes((REPO/n).read_bytes())
    (code.parent/'revision').write_text(rev+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    manifest=dict(track='track_1',repo_id='nvidia/video_to_data_challenge',revision='5f68335f3acc802033d1e80728c1633197521de8',files=[])
    identity=lambda b:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    for e in m.EPISODES:
        spec=dict(episode_index=e,total_frames=5,camera_name='front_stereo_camera_left',height=24,width=32)
        relative=f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{e:06d}.mp4';video=root/'data'/relative
        video.parent.mkdir(parents=True,exist_ok=True);video.write_bytes(b'NO_MEDIA_TINY_OPAQUE_'+str(e).encode());vp=identity(video.read_bytes());manifest['files'].append(dict(path=relative,**vp))
        ipath=code/m.pin_name(e,'input');inputs=dict(schema='world-reward-cari-clip-input-pins-v1',clip_spec=spec,source_files={'tiny/source':vp});ipath.write_text(json.dumps(inputs))
        base=root/f'outputs/episode_{e:06d}/cari_shared_export_v1';base.mkdir(parents=True)
        records={}
        for n in m.EXPORT_FILES-{'report.json'}:
            b=('TINY_OPAQUE_'+n).encode();(base/n).write_bytes(b);records[n]=identity(b)
        report=dict(stage='world_reward_native_cari_shared_full_video_direct_export',status='pass',phase='complete',episode_index=e,clip_spec=spec,
            producer_revision='c'*40,script_sha256='d'*64,frames=5,input_track='track_1',ground_truth_used=False,ground_truth_read=False,
            private_truth_read=False,hand_labeled_test=False,oracle_modes=[],unchanged_refined_predictions_verified=True,
            full_original_native_export_verified=True,original_frame_coverage_verified=True,source_inputs_assets_rehashed=True,source_helpers_rehashed=True,
            input_pins=identity(ipath.read_bytes()),source_files=inputs['source_files'],output_files=records,original_frame_indices=list(range(5)))
        (base/'report.json').write_text(json.dumps(report));rp=identity((base/'report.json').read_bytes());records['report.json']=rp
        pins=dict(schema='world-reward-cari-shared-export-pins-v1',clip_spec=spec,export=dict(**rp,producer_revision='c'*40,script_sha256='d'*64),export_files=records)
        (code/m.pin_name(e,'shared_export')).write_text(json.dumps(pins))
    (root/'results/input-manifest.json').write_text(json.dumps(manifest))
    for p in sorted(code.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    code.chmod(0o555)
    for name in('revision','source-sha256'):(code.parent/name).chmod(0o444)
    return root,code,rev


def test_fixed_all_fourteen_midpoint_order_no_selection():
    assert overview.EPISODES==(0,1,2,3,5,6,8,9,12,13,14,15,21,26)
    assert overview.midpoint(399)==199 and overview.midpoint(415)==207 and overview.midpoint(4)==1
    for bad in(0,2,True,5.):
        with pytest.raises(ValueError):overview.midpoint(bad)
    assert overview.tile_position(0)==(0,64)and overview.tile_position(1)==(320,64)and overview.tile_position(13)==(320,988)
    for bad in(-1,14,True):
        with pytest.raises(ValueError):overview.tile_position(bad)


def test_real_current_source_and_all_export_video_proof_tiny(gate,tmp_path,monkeypatch):
    root,code,rev=setup(gate,tmp_path,monkeypatch);p=gate.host_proof(root,code,rev)
    assert [x['episode_index']for x in p['clips']]==list(gate.EPISODES) and len(p['clips'])==14
    assert len(p['mounts'])==30 and len(p['files'])==113 and all(x['frame_index']==2 for x in p['clips'])
    assert p['source_binding']['helpers'][gate.HELPERS[0]]['bytes']>0
    assert all('/weights/'not in x and '/vendor/'not in x and '/labels/'not in x for x in p['mounts'])
    gate.recheck(gate.runtime(code),p)


def test_actual_export_schema_has_no_direct_video_sha_field(gate,tmp_path,monkeypatch):
    # The real export binds its prepared-input pins/source-files, not a new
    # top-level video SHA. The independently frozen official video is checked
    # by overview separately; no genuine producer field is invented.
    tree=ast.parse((REPO/'infra/cari_full_export.py').read_text())
    report_keywords={k.arg for node in ast.walk(tree) if isinstance(node,ast.Call)
        and (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
             and node.func.value.id=='report' and node.func.attr=='update') for k in node.keywords}
    assert {'input_pins','source_files'}<=report_keywords and 'input_sha256'not in report_keywords
    root,code,rev=setup(gate,tmp_path,monkeypatch)
    for e in gate.EPISODES:
        report=json.loads((root/f'outputs/episode_{e:06d}/cari_shared_export_v1/report.json').read_text())
        assert 'input_sha256'not in report and report['input_pins'] and report['source_files']
    assert len(gate.host_proof(root,code,rev)['clips'])==14


@pytest.mark.parametrize('fault',['export','video','input','manifest','source','sibling','extraexport','missingpin'])
def test_provenance_changes_reject_before_render(gate,tmp_path,monkeypatch,fault):
    root,code,rev=setup(gate,tmp_path,monkeypatch);e=gate.EPISODES[0];base=root/f'outputs/episode_{e:06d}/cari_shared_export_v1'
    if fault=='export':(base/'target.npy').write_bytes(b'changed')
    elif fault=='video':next((root/'data').rglob('*.mp4')).write_bytes(b'changed')
    elif fault=='input':p=code/gate.pin_name(e,'input');p.chmod(0o644);p.write_bytes(p.read_bytes()+b' ');p.chmod(0o444)
    elif fault=='manifest':p=root/'results/input-manifest.json';v=json.loads(p.read_text());v['track']='track_2';p.write_text(json.dumps(v))
    elif fault=='source':(code/gate.HELPERS[0]).chmod(0o644)
    elif fault=='sibling':(code.parent/'extra').write_bytes(b'X')
    elif fault=='extraexport':(base/'extra').write_bytes(b'X')
    else:p=code/gate.pin_name(e,'shared_export');p.parent.chmod(0o755);p.unlink();p.parent.chmod(0o555)
    with pytest.raises((ValueError,FileNotFoundError)):gate.host_proof(root,code,rev)


def test_posthash_detects_any_export_changes(gate,tmp_path,monkeypatch):
    root,code,rev=setup(gate,tmp_path,monkeypatch);p=gate.host_proof(root,code,rev)
    path=Path(p['clips'][-1]['export'])/'native_parameters.npz';path.write_bytes(b'CHANGED')
    with pytest.raises(ValueError):gate.recheck(gate.runtime(code),p)


def test_jpeg_fixed_settings_and_cap_no_quality_retry():
    class Sheet:
        calls=[]
        def save(self,buffer,**kw):self.calls.append(kw);buffer.write(b'X'*240000)
    s=Sheet();assert len(overview.encode_sheet(s))==240000 and s.calls==[dict(format='JPEG',quality=70,optimize=True)]
    class Overflow(Sheet):
        def save(self,buffer,**kw):self.calls.append(kw);buffer.write(b'X'*240001)
    with pytest.raises(ValueError,match='no quality rescue'):overview.encode_sheet(Overflow())


def test_renderer_same_unchanged_depth_and_colours_no_extra_transform():
    K=np.array([[20.,0,16.],[0,20.,12.],[0,0,1.]])
    v=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]],np.float32);f=np.array([[0,1,2]],np.int64)
    before=v.tobytes();d=preview.raster_depth(v,f,K,32,24);rgb=np.full((24,32,3),100,np.uint8)
    got=preview.overlay(rgb,d,np.full_like(d,np.inf))
    assert v.tobytes()==before and np.isfinite(d).any() and got.shape==rgb.shape
    np.testing.assert_array_equal(got[np.isfinite(d)][0],np.rint(.48*100+.52*np.asarray(preview.HUMAN_RGB)))


def test_shell_cpu_scope_syntax_and_no_free_episode_selection():
    p=REPO/'infra/run_reconstruction_overview.sh';subprocess.run(['bash','-n',str(p)],check=True)
    s=p.read_text();assert '$# == 0'in s and'210s'in s and'--episode'not in s
    native=(REPO/'infra/reconstruction_overview.py').read_text()
    assert "'--memory','3g'"in native and"'--cpus','2'"in native and"'--cap-drop','ALL'"in native
    assert '--gpus'not in native and '.crop('not in native and'quality=70,optimize=True'in native


@pytest.mark.parametrize('expired',[False,True])
def test_mock_native_all_fourteen_middle_frames_and_one_frozen_sheet(gate,tmp_path,monkeypatch,expired):
    import types
    cv2=types.ModuleType('cv2');cv2.CAP_PROP_FRAME_COUNT=1;cv2.CAP_PROP_POS_FRAMES=2;cv2.INTER_AREA=3;cv2.COLOR_BGR2RGB=4
    cv2.resize=lambda image,size,interpolation:np.tile(image[:1,:1],(size[1],size[0],1))
    cv2.cvtColor=lambda image,code:image[:,:,::-1].copy()
    cv2.VideoCapture=None
    monkeypatch.setitem(sys.modules,'cv2',cv2)
    from PIL import ImageFont
    root,code,rev=setup(gate,tmp_path,monkeypatch);out=root/'results'/('reconstruction-overview-'+rev);out.mkdir();(out/'container.cid').write_bytes(b'e'*64)
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1');monkeypatch.setenv('WR_IMAGE_ID',gate.IMAGE)
    # All network observations are a single loopback interface, no cloud.
    original_iter=Path.iterdir
    def iterdir(p):
        if str(p)=='/sys/class/net':return iter([Path('/sys/class/net/lo')])
        return original_iter(p)
    monkeypatch.setattr(Path,'iterdir',iterdir)
    # The source origin remains the frozen copied helper; existing original
    # raster/overlay implementations are not modified.
    monkeypatch.setattr(preview,'__file__',str(code/'infra/reconstruction_preview.py'))
    font=ImageFont.load_default();monkeypatch.setattr(ImageFont,'truetype',lambda *a,**k:font)
    tri=np.array([[-.1,-.1,2],[.1,-.1,2],[0,.1,2]],np.float32);faces=np.array([[0,1,2]],np.int64);original=tri.tobytes()
    def values(np,base,spec):
        K=np.array([[40.,0,16.],[0,40.,12.],[0,0,1.]])
        return np.tile(tri,(5,1,1)),faces,dict(camera_K=K,object_vertices=tri,object_faces=faces,object_rotation=np.tile(np.eye(3,dtype=np.float32),(5,1,1)),object_translation=np.zeros((5,3),np.float32))
    monkeypatch.setattr(gate,'arrays',values);calls=[]
    class Capture:
        def __init__(self,path):self.path=path;self.index=None;calls.append(self)
        def get(self,key):return 5 if key==cv2.CAP_PROP_FRAME_COUNT else self.index+1
        def set(self,key,index):assert key==cv2.CAP_PROP_POS_FRAMES;self.index=index;return True
        def read(self):return True,np.full((24,32,3),100,np.uint8)
        def release(self):self.released=True
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    if expired:
        original_encode=gate.encode_sheet
        def expire(sheet):
            raw=original_encode(sheet);monkeypatch.setattr(gate.time,'monotonic',lambda:10**12);return raw
        monkeypatch.setattr(gate,'encode_sheet',expire)
        with pytest.raises(ValueError,match='budget proof'):gate.native(root,code,rev,out)
        assert len(calls)==14 and not(out/'overview.jpg').exists() and not(out/'native.json').exists()
        return
    result=gate.native(root,code,rev,out)
    assert result['status']=='pass' and result['image_hw']==[1142,640] and result['image']['bytes']<=240000
    assert len(calls)==14 and all(x.index==2 and x.released for x in calls)
    assert [x['episode_index']for x in result['clips']]==list(gate.EPISODES) and tri.tobytes()==original
    assert result['source_rehashed_after'] and result['crop_applied']is False and result['model_execution']is False
    assert set(p.name for p in out.iterdir())=={'overview.jpg','native.json','container.cid'}


def test_mock_host_owned_absence_lowercase_and_readonly_mounts(gate,tmp_path,monkeypatch):
    from types import SimpleNamespace
    root,code,rev=setup(gate,tmp_path,monkeypatch);proof=gate.host_proof(root,code,rev);cid='e'*64;calls=[]
    def run(argv,**kw):
        calls.append(argv)
        if argv[:3]==['docker','image','inspect']:return SimpleNamespace(returncode=0,stdout=(gate.IMAGE+'\n').encode(),stderr=b'')
        if argv[:2]==['docker','inspect']:return SimpleNamespace(returncode=1,stdout=b'\n',stderr=('error: no such object: '+cid+'\n').encode())
        assert argv[:2]==['docker','run'] and '--gpus'not in argv and argv[argv.index('--memory')+1]=='3g'
        for p in proof['mounts']:assert 'type=bind,src='+p+',dst='+p+',readonly'in argv
        Path(argv[argv.index('--cidfile')+1]).write_bytes(cid.encode())
        out=root/'results'/('reconstruction-overview-'+rev);image=b'TINY_JPEG';(out/'overview.jpg').write_bytes(image);(out/'overview.jpg').chmod(0o444)
        result=dict(status='pass',phase='complete',producer_revision=rev,input_track='track_1',source_binding=proof['source_binding'],sources=proof['files'],episodes=list(gate.EPISODES),
            image_id=gate.IMAGE,image=dict(bytes=len(image),sha256=hashlib.sha256(image).hexdigest()),source_rehashed_after=True,image_hw=[1142,640],viewport_hw=[120,160],
            original_geometry_unchanged=True,geometry_operations_applied=False,ground_truth_used=False,manual_annotation=False,fitting=False,metric_evaluation=False,quality_verified=False,
            model_execution=False,gpu_used=False,crop_applied=False,budget_seconds=180,elapsed_seconds=1,clips=[dict(episode_index=e,frame_index=2)for e in gate.EPISODES])
        (out/'native.json').write_text(json.dumps(result));(out/'native.json').chmod(0o444)
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    monkeypatch.setattr(gate.subprocess,'run',run)
    value=gate.host(root,code,rev);out=root/'results'/('reconstruction-overview-'+rev)
    assert value['status']=='pass' and value['source_rehashed_after'] and value['owned_container_removed']
    assert out.stat().st_mode&0o777==0o555 and (out/'container.cid').stat().st_mode&0o777==0o444
    assert not any(a[:2]==['docker','rm']for a in calls)


def test_mock_host_wrong_image_fails_before_native_and_retains_failure(gate,tmp_path,monkeypatch):
    from types import SimpleNamespace
    root,code,rev=setup(gate,tmp_path,monkeypatch);calls=[]
    def run(argv,**kw):
        calls.append(argv);return SimpleNamespace(returncode=0,stdout=('sha256:'+'f'*64+'\n').encode(),stderr=b'')
    monkeypatch.setattr(gate.subprocess,'run',run)
    with pytest.raises(ValueError,match='immutable receipt'):gate.host(root,code,rev)
    out=root/'results'/('reconstruction-overview-'+rev);r=json.loads((out/'report.json').read_text())
    assert r['status']=='fail' and r['source_rehashed_after'] and r['owned_container_removed']is False
    assert len(calls)==1 and calls[0][:3]==['docker','image','inspect'] and not(out/'overview.jpg').exists()


@pytest.mark.parametrize('fault',['extra','malformed','symlink'])
def test_native_exact_precreated_cid_guards_before_render(gate,tmp_path,monkeypatch,fault):
    root,code,rev=setup(gate,tmp_path,monkeypatch);out=root/'results'/('reconstruction-overview-'+rev);out.mkdir();cid=out/'container.cid'
    cid.write_bytes(b'e'*64)
    if fault=='extra':(out/'foreign').write_bytes(b'X')
    elif fault=='malformed':cid.write_bytes(b'e'*63+b'Z')
    else:cid.unlink();cid.symlink_to(root/'results/input-manifest.json')
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1');monkeypatch.setenv('WR_IMAGE_ID',gate.IMAGE)
    old=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('/sys/class/net/lo')])if str(p)=='/sys/class/net'else old(p))
    with pytest.raises(ValueError):gate.native(root,code,rev,out)
    assert not(out/'overview.jpg').exists()and not(out/'native.json').exists()
