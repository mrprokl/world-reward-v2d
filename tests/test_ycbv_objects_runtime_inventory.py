"""Tiny procedural assets/git/image mocks; no real image, network, model or GPU."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
spec=importlib.util.spec_from_file_location('wr_objects_runtime_inventory_test',REPO/'infra/ycbv_objects_runtime_inventory.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)


def write(path,raw=b'own tiny bytes',mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o644)
    path.write_bytes(raw);path.chmod(mode);return gate.identity(path,empty=True)


def js(path,value):return write(path,json.dumps(value).encode())


def image():return dict(Id=gate.IMAGE,Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=['sha256:'+'a'*64,'sha256:'+'b'*64]))


def fake_assets(tmp_path,monkeypatch):
    raw=b'own procedural checkpoint bytes'
    expected=(len(raw),hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(gate.objects,'CHECKPOINT_PINS',tuple(expected for _ in gate.objects.CKPTS))
    monkeypatch.setattr(gate.objects,'YAML_PINS',tuple(expected for _ in gate.objects.YAMLS))
    for names,suffix in ((gate.objects.CKPTS,'.ckpt'),(gate.objects.YAMLS,'.yaml')):
        for name in names:write(tmp_path/gate.objects.OBJECT/'checkpoints'/(name+suffix),raw)
    write(tmp_path/gate.objects.OBJECT/'LICENSE',b'Custom model license: permission to use under stated terms; research eligibility not established.')
    acquisition={'assets':[dict(repo_id='facebook/sam-3d-objects',revision=gate.objects.OBJECT_REV,path=str(tmp_path/gate.objects.OBJECT)),
        dict(repo_id='Ruicheng/moge-vitl',revision=gate.objects.MOGE_REV,cache_dir=str(tmp_path/gate.objects.HF))]}
    records=[]
    for name in gate.objects.REG4:
        row=write(tmp_path/gate.objects.WEIGHTS/'torch_home/hub/checkpoints'/name,raw)
        records.append(dict(filename=name,hash_source='first_observed_https_download',
            url='https://dl.fbaipublicfiles.com/dinov2/dinov2_'+('vitl14'if 'vitl14'in name else'vitb14')+'/'+name,**row))
    js(tmp_path/'results/weights-acquisition.json',acquisition)
    js(tmp_path/'results/auxiliary-assets.json',dict(source_revisions={'dinov2':gate.objects.DINO_REV},checkpoints=records))
    original=image();original['Config']={'Env':['UNTRUSTED_INHERITED_VALUE=never output']}
    js(tmp_path/'results/image-sam3d-runtime.json',original)
    dino={gate.objects.DINO+'/hubconf.py':dict(bytes=4,sha256='a'*64),gate.objects.DINO+'/LICENSE':dict(bytes=4,sha256='b'*64),gate.objects.DINO+'/MODEL_CARD.md':dict(bytes=4,sha256='c'*64)}
    monkeypatch.setattr(gate,'dino_source',lambda *a:dino)
    monkeypatch.setattr(gate,'moge_graph',lambda *a:({gate.objects.MOGE+'/blobs/'+'d'*64:dict(bytes=123,sha256='d'*64)},
        {gate.objects.SNAPSHOT+'/model.pt':'../../blobs/'+'d'*64,gate.objects.SNAPSHOT+'/README.md':'../../blobs/'+'e'*40}))
    monkeypatch.setattr(gate,'image',lambda *a:image())
    return acquisition,records


def test_exact_model_inventory_actual_original_receipts_and_no_raw_image_env(tmp_path,monkeypatch):
    fake_assets(tmp_path,monkeypatch);runtime,img,evidence=gate.collect(tmp_path,1000)
    assert set(runtime)=={'model_files','source_files','moge_links','installed_sources','acquisition_receipts'}
    assert len(runtime['model_files'])==17 and set(img)=={'Id','Architecture','Os','RootFS'}
    assert evidence['original_raw_image_settings_exported']is False
    assert evidence['model_license']['content_notice_verified']and not evidence['model_license']['source_model_license_equality_assumed']
    assert 'UNTRUSTED_INHERITED_VALUE'not in json.dumps((runtime,img,evidence))
    assert evidence['DINO_checkpoint_hash_source']=='first_observed_https_download'


@pytest.mark.parametrize('fault',['weights_revision','moge_revision','path','auxrev','reg4_hash','reg4_origin','model','yaml','license','image'])
def test_original_provenance_tamper_fails_before_CPUprobe(tmp_path,monkeypatch,fault):
    acq,rows=fake_assets(tmp_path,monkeypatch)
    if fault=='weights_revision':acq['assets'][0]['revision']='0'*40;js(tmp_path/'results/weights-acquisition.json',acq)
    elif fault=='moge_revision':acq['assets'][1]['revision']='0'*40;js(tmp_path/'results/weights-acquisition.json',acq)
    elif fault=='path':acq['assets'][0]['path']=str(tmp_path/'foreign');js(tmp_path/'results/weights-acquisition.json',acq)
    elif fault=='auxrev':js(tmp_path/'results/auxiliary-assets.json',dict(source_revisions={'dinov2':'0'*40},checkpoints=rows))
    elif fault in('reg4_hash','reg4_origin'):
        rows[0]['sha256']='0'*64 if fault=='reg4_hash'else rows[0]['sha256']
        if fault=='reg4_origin':rows[0]['hash_source']='independent_release_claim'
        js(tmp_path/'results/auxiliary-assets.json',dict(source_revisions={'dinov2':gate.objects.DINO_REV},checkpoints=rows))
    elif fault=='model':write(tmp_path/gate.objects.OBJECT/'checkpoints/ss_generator.ckpt',b'changed')
    elif fault=='yaml':write(tmp_path/gate.objects.OBJECT/'checkpoints/pipeline.yaml',b'changed')
    elif fault=='license':write(tmp_path/gate.objects.OBJECT/'LICENSE',b'Not a licensing notice')
    else:
        original=image();original['Id']='sha256:'+'d'*64;js(tmp_path/'results/image-sam3d-runtime.json',original)
    with pytest.raises(ValueError):gate.collect(tmp_path,1000)


def test_dino_source_Git_blobs_revision_selected_inventory_only(tmp_path,monkeypatch):
    folder=tmp_path/gate.objects.DINO;rows=[]
    for name in ('hubconf.py','LICENSE','MODEL_CARD.md','dinov2/__init__.py','dinov2/models.py'):
        raw=('own '+name).encode();write(folder/name,raw)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();rows.append(('100644 blob '+blob+'\t'+name).encode())
    calls=[]
    def control(args,*unused):
        calls.append(args)
        if 'rev-parse'in args:return gate.objects.DINO_REV.encode()
        if 'ls-tree'in args:return b'\0'.join(rows)+b'\0'
        return b''
    monkeypatch.setattr(gate,'control',control)
    result=gate.dino_source(tmp_path,1000)
    assert len(result)==5 and all(n.startswith(gate.objects.DINO+'/')for n in result)
    assert not any('fetch'in args or 'clone'in args for args in calls)
    write(folder/'dinov2/models.py',b'modified')
    with pytest.raises(ValueError):gate.dino_source(tmp_path,1000)


@pytest.mark.parametrize('name',['../private.py','/absolute.py','dinov2/../other.py','data/private.json','dinov2/file.pkl','README.md'])
def test_only_selected_public_DINO_source(name):
    try:accepted=gate.source_selection(name)
    except ValueError:accepted=False
    assert not accepted


def test_safe_image_projection_only_and_fullorderedLayers_not_guessed(monkeypatch):
    expected=image();monkeypatch.setattr(gate,'control',lambda *a:json.dumps(expected).encode())
    assert gate.image(1000)==expected
    expected['RootFS']['Layers']=[]
    with pytest.raises(ValueError):gate.image(1000)


def test_native_probe_has_NO_mounts_model_GPU_network_credentials(tmp_path):
    args=gate.probe_arguments(tmp_path,'a'*40)
    assert '--mount'not in args and '--gpus'not in args and args[args.index('--network')+1]=='none'
    assert args[args.index('--user')+1]=='0:0'and '--cap-drop'in args and '--read-only'in args
    assert args[args.index('--entrypoint')+1:args.index('--entrypoint')+4]==['/usr/bin/env',gate.IMAGE,'-i']
    assert 'CUDA_VISIBLE_DEVICES='in args and 'HOME=/tmp'in args
    assert not any('TOKEN='in arg or 'model.pt'in arg or '/data/'in arg or 'eval_private'in arg for arg in args)
    assert 'find_spec'in gate.PROBE and 'from_pretrained'not in gate.PROBE and 'torch.load'not in gate.PROBE


def installed():return {name:{'__init__.py':dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest()),'module.py':dict(bytes=1,sha256='a'*64)}for name in gate.objects.MODULES}


@pytest.mark.parametrize('fault',['extra','missing','path','nonpython','bool','digest'])
def test_installed_source_metadata_fails_without_model_load(fault):
    value=installed()
    if fault=='extra':value['body']={}
    elif fault=='missing':value.pop('moge')
    elif fault=='path':value['moge']['../private.py']=dict(bytes=1,sha256='a'*64)
    elif fault=='nonpython':value['moge']['model.pt']=dict(bytes=1,sha256='a'*64)
    elif fault=='bool':value['moge']['module.py']['bytes']=True
    else:value['moge']['module.py']['sha256']='a'
    with pytest.raises(ValueError):gate.validate_installed(value)


def test_installed_source_empty_init_is_kept_and_metadata_is400exclusive(tmp_path):
    value=installed();assert gate.validate_installed(value)==value
    original=os.umask(0)
    try:pin=gate.exclusive(tmp_path/'runtime.json',value)
    finally:os.umask(original)
    assert (tmp_path/'runtime.json').stat().st_mode&0o777==0o400
    assert pin==gate.identity(tmp_path/'runtime.json')
    with pytest.raises(FileExistsError):gate.exclusive(tmp_path/'runtime.json',value)


def test_cleanup_rejects_foreign_container(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();write(out/'.probe.cid',b'a'*64)
    calls=[]
    def control(args,*unused):
        calls.append(args)
        return b'a'*64 if 'ps'in args else b'foreign|/foreign|other|other'
    monkeypatch.setattr(gate,'control',control)
    with pytest.raises(ValueError):gate.cleanup(out,'b'*40)
    assert not any('rm'in args for args in calls)


def source_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'ROOT',tmp_path);rev='a'*40;code=tmp_path/'jobs'/rev/gate.ENTRY/'code'
    for n in gate.HELPERS:write(code/n,('own source '+n).encode())
    for name,raw in (('revision',(rev+'\n').encode()),('source-sha256',('b'*64+'\n').encode())):write(code.parent/name,raw,0o644)
    for path in sorted([code,*[p for p in code.rglob('*')if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):path.chmod(0o555)
    return code,rev


@pytest.mark.parametrize('tamper',[False,True])
def test_mocked_CPUinventory_seals_only_after_actual_assets_source_post(tmp_path,monkeypatch,tamper,capsys):
    code,rev=source_fixture(tmp_path,monkeypatch);(tmp_path/'results').mkdir()
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',rev)
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]));monkeypatch.setattr(gate.objects,'__file__',str(code/'infra/ycbv_point_objects.py'));monkeypatch.setattr(sys,'platform','linux');monkeypatch.setattr(os,'geteuid',lambda:0)
    class Uname:nodename='scenesmith-ncc-h100-01'
    monkeypatch.setattr(os,'uname',lambda:Uname())
    runtime=dict(model_files={},source_files={},moge_links={},installed_sources={},acquisition_receipts={})
    calls=[]
    def collect(*unused):
        calls.append(1)
        return runtime.copy(),image(),{'opaque_old_image_pin':dict(bytes=1,sha256='a'*64)} if len(calls)==1 or not tamper else {'changed':True}
    monkeypatch.setattr(gate,'collect',collect)
    monkeypatch.setattr(gate,'control',lambda *a:b'')
    def run(args,**kwargs):
        assert '--gpus'not in args and '--mount'not in args
        assert set(kwargs['env'])=={'PATH','HOME','DOCKER_HOST'}and kwargs['timeout']<=300
        out=tmp_path/'results'/('ycbv-objects-runtime-'+rev);write(out/'.probe.cid',b'c'*64,0o400)
        return types.SimpleNamespace(returncode=0,stdout=json.dumps(installed()).encode())
    monkeypatch.setattr(gate.subprocess,'run',run)
    if tamper:
        with pytest.raises(SystemExit):gate.main([])
    else:gate.main([])
    out=tmp_path/'results'/('ycbv-objects-runtime-'+rev);report=gate.strict((out/'report.json').read_bytes())
    assert report['status']==('fail'if tamper else'pass')
    assert (out/'runtime.json').exists()is(not tamper)
    assert report['GPU_used']is False and report['models_loaded']is False and report['RGB_or_labels_read']is False
    stdout=capsys.readouterr().out;assert len(stdout)<4000 and 'model_files'not in stdout and 'moge_links'not in stdout
    if not tamper:
        output=gate.strict((out/'runtime.json').read_bytes());assert set(output)=={'image_receipt','model_files','source_files','moge_links','installed_sources','acquisition_receipts'}
        assert output['image_receipt']['path']==f'results/ycbv-objects-runtime-{rev}/image.json'
    with pytest.raises(ValueError):gate.main([])


def test_shell_arguments_exact_namespace_and_statically_complete_closure(tmp_path):
    script=REPO/'infra/run_ycbv_objects_runtime_inventory.sh';assert subprocess.run(['bash','-n',str(script)],capture_output=True).returncode==0
    env={'PATH':'/usr/bin:/bin','HOME':str(tmp_path),'WR_ROOT':str(tmp_path),'WR_CODE':str(tmp_path),'WR_CODE_REVISION':'a'*40}
    for args in ([],['--help'],['--episode','0']):assert subprocess.run(['bash',str(script),*args],env=env,capture_output=True).returncode!=0
    spec=importlib.util.spec_from_file_location('runtime_inventory_closure',REPO/'infra/azure_job.py');job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)
    data={str(p.relative_to(REPO)):p.read_bytes()for folder in('infra','src','configs')for p in(REPO/folder).rglob('*')if p.is_file()and p.suffix in('.py','.sh','.json')}
    data['pyproject.toml']=(REPO/'pyproject.toml').read_bytes();closure=job.runtime_bundle_paths(data,'infra/run_ycbv_objects_runtime_inventory.sh')
    assert set(gate.HELPERS)<=set(closure)
    assert not any(n in closure for n in('infra/object_smoke.py','infra/frontend_replica_inventory.py','infra/ycbv_point_acquire.py','infra/bridge_rgb_anchor_render.py'))


def test_original_MoGe1_link_graph_preserved_and_hash_not_fabricated(tmp_path,monkeypatch):
    snap=tmp_path/gate.objects.SNAPSHOT;snap.mkdir(parents=True)
    blobs=tmp_path/gate.objects.MOGE/'blobs';blobs.mkdir();terminal=tmp_path/gate.objects.HF/'blobs/ab'/('a'*64)
    write(terminal,b'own model bytes');repo=blobs/('b'*64);repo.symlink_to('../../blobs/ab/'+'a'*64)
    (snap/'model.pt').symlink_to('../../blobs/'+'b'*64)
    card=blobs/('c'*40);write(card,b'own card');(snap/'README.md').symlink_to('../../blobs/'+'c'*40)
    original=gate.identity
    def fake_identity(path,*a,**k):
        if path==terminal:return dict(bytes=1256823446,sha256='da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f')
        return original(path,*a,**k)
    monkeypatch.setattr(gate,'identity',fake_identity)
    files,links=gate.moge_graph(tmp_path)
    assert len(links)==3 and len(files)==2
    assert links[gate.objects.SNAPSHOT+'/model.pt']=='../../blobs/'+'b'*64
    assert links[gate.objects.MOGE+'/blobs/'+'b'*64]=='../../blobs/ab/'+'a'*64
    assert str(terminal.relative_to(tmp_path))in files
    repo.unlink();repo.symlink_to('/foreign/path')
    with pytest.raises(ValueError):gate.moge_graph(tmp_path)


def test_foreign_extra_MoGe1_snapshot_never_native_first_selected(tmp_path):
    (tmp_path/gate.objects.MOGE/'snapshots'/gate.objects.MOGE_REV).mkdir(parents=True)
    (tmp_path/gate.objects.MOGE/'snapshots'/'foreign').mkdir()
    with pytest.raises(ValueError):gate.moge_graph(tmp_path)


def test_foreign_writable_source_rejected_before_inventory(tmp_path,monkeypatch):
    code,rev=source_fixture(tmp_path,monkeypatch);assert gate.source(code,rev)['helpers']
    (code/gate.HELPERS[0]).chmod(0o644)
    with pytest.raises(ValueError):gate.source(code,rev)


def test_stable_metadata_excludes_atime_but_not_owner_links_or_bytes(tmp_path):
    path=tmp_path/'source.py';write(path,b'own source');before=path.lstat()
    values={k:getattr(before,k)for k in gate.STABLE_FIELDS};values.update(st_atime_ns=before.st_atime_ns+1,st_atime=before.st_atime+1)
    assert gate.stable(before)==gate.stable(types.SimpleNamespace(**values))
    for key in('st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_uid','st_gid'):
        changed=dict(values);changed[key]+=1
        assert gate.stable(before)!=gate.stable(types.SimpleNamespace(**changed))
    assert 'assert s==p.lstat()'not in gate.PROBE and 'assert stable(s)==stable(p.lstat())'in gate.PROBE
