"""Manufactured original ABI/lifecycle controls; no models, HTTP, RGB datasets."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace,ModuleType
import numpy as np
import pytest
import vcoco_hoi_observations as p
from world_reward.hoi_detr_observations import HOIDetrObservations


def fixture_rows():
    images=[dict(image_id='%032x'%i,file=f'image_{i:06d}.jpg',height=4,width=5)for i in range(16)]
    banks=[dict(file=f'image_{i:06d}.npz',original_slot=i,image_size=[4,5],person_ids=[f'person-{i}'],identity=p.public.pin(f'bank{i}'.encode()))for i in range(16)]
    return images,banks


def result(k=1):
    c=np.asarray([0,1,2]if k else [],np.int64);n=len(c)
    pairs=np.asarray([[0,1]],np.int64)if k else np.empty((0,2),np.int64)
    target=np.asarray([[1,2]],np.int64)if k else np.empty((0,2),np.int64)
    return HOIDetrObservations(0,(4,5),np.zeros((1500,3),np.float32),np.zeros((1500,4),np.float32),
        np.zeros((1500,256),np.float32),np.column_stack((np.zeros((n,4),np.float32),np.full(n,.6,np.float32))),np.arange(n,dtype=np.int64),
        np.arange(n,dtype=np.int64),np.arange(n,dtype=np.int64),c,np.tile(np.asarray([[1,1,3,3]],np.float32),(n,1)),
        np.full(n,.7,np.float32),np.full(n,.6,np.float32),pairs,np.full((k,2),-.5,np.float32),target,np.full((k,2),-.2,np.float32))


def test_all16_validate_before_any_forward_empty_and_raw_negative():
    images,banks=fixture_rows();events=[];outputs=[]
    def load(row):events.append(('validate',row['original_slot']))
    def infer(rgb):events.append(('infer',));return result(len(events)%2)
    def save(value,image,bank,ordinal):
        assert value.query_tokens.shape==(1500,256)
        assert not len(value.hand_object_logits)or (value.hand_object_logits<0).all()
        return dict(slot=bank['original_slot'],ordinal=ordinal,k=len(value.hand_object_pairs))
    p.observe(images,banks,load,lambda _:np.zeros((4,5,3),np.uint8),infer,save,lambda:None,outputs)
    assert events[:16]==[('validate',i)for i in range(16)]and len(outputs)==16 and len(events)==32
    assert [r['slot']for r in outputs]==list(range(16))and {r['k']for r in outputs}=={0,1}


def test_bad_fullbank_stops_before_forward():
    images,banks=fixture_rows();calls=[]
    def load(r):
        if r['original_slot']==15:raise ValueError('bad completebank')
    with pytest.raises(ValueError):p.observe(images,banks,load,lambda _:None,lambda _:calls.append('forward'),lambda *a:None,lambda:None,[])
    assert not calls


def test_rgb_mutation_and_bad_count_rejected():
    images,banks=fixture_rows();rgb=np.zeros((4,5,3),np.uint8)
    def infer(a):a[0,0,0]=1;return result()
    with pytest.raises(ValueError):p.observe(images,banks,lambda _:None,lambda _:rgb,infer,lambda *a:None,lambda:None,[])
    with pytest.raises(ValueError):p.observe(images[:15],banks,lambda _:None,lambda _:rgb,lambda _:result(),lambda *a:None,lambda:None,[])


def test_host_stdlib_only_original_constructor_and_no_old_profile():
    code="""import sys,importlib.abc
sys.path[:0]=['infra','src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in ('numpy','torch','transformers','onnxruntime','PIL','cv2'):raise AssertionError('ML import')
sys.meta_path.insert(0,Deny())
import vcoco_hoi_observations
print('stdlib')
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout=='stdlib\n'
    src=Path(p.__file__).read_text()
    assert 'neutral.build_original_hoi('in src and 'infer_hoi_detr_frame(built.model,rgb,0,built.operations)'in src
    assert all(x not in src for x in ('person_bank_profile','authenticate_person_banks','external_cohort','eval_private','actor_bbox','original.run(','original.gpu_model(','original.cpu_overlay('))
    assert 'urllib.request.ProxyHandler({})'in src and 'runtime.NoRedirect()'in src
    assert all(Path(n).is_file()for n in p.NATIVE_FILES)
    assert '--no-index'in src and '--no-deps'in src and '--no-build-isolation'in src


def fake_native_source(monkeypatch,tmp_path):
    code=tmp_path/'code';code.mkdir();names=('infra/vcoco_hoi_observations.py','two.py')
    monkeypatch.setattr(p,'NATIVE_FILES',names);monkeypatch.setattr(p,'__file__',str(code/names[0]))
    helpers={}
    for n in names:
        f=code/n;f.parent.mkdir(exist_ok=True);f.write_bytes(n.encode());f.chmod(0o400);helpers[n]=p.rt.identity(f)
    revision='1'*40
    for n,b in [('revision',(revision+'\n').encode()),('source-sha256',('a'*64+'\n').encode())]:
        (code.parent/n).write_bytes(b);(code.parent/n).chmod(0o400)
    source=dict(producer_revision=revision,helpers=helpers,markers={n:p.rt.identity(code.parent/n)for n in ('revision','source-sha256')})
    return code,revision,dict(source=source)


def test_native_markers_literal_whitelist_and_mutation(monkeypatch,tmp_path):
    code,rev,proof=fake_native_source(monkeypatch,tmp_path);before=p.native_source(code,rev,proof)
    assert set(before)==set(p.NATIVE_FILES)
    (code/'extra.py').write_bytes(b'foreign');(code/'extra.py').chmod(0o400)
    with pytest.raises(ValueError):p.native_source(code,rev,proof)
    (code/'extra.py').unlink();(code.parent/'revision').chmod(0o600);(code.parent/'revision').write_bytes((rev+'\\n').encode());(code.parent/'revision').chmod(0o400)
    proof['source']['markers']['revision']=p.rt.identity(code.parent/'revision')
    with pytest.raises(ValueError):p.native_source(code,rev,proof)


def native_fixture(monkeypatch,tmp_path):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out)
    images,banks=fixture_rows();pin=p.public.pin(b'proof');report=dict(schema=p.SCHEMA,stage='native_vcoco_hoi_model',status='pass',phase='complete',producer_revision='1'*40,image_id=p.IMAGE,
        proof_identity=pin,source_inputs_runtime_assets_rehashed_after=True,models_loaded=1,native_forward_calls=16,model=dict(strict_checkpoint=dict(keys=1796,strict=True,weights_only=True),checkpoint_buffer_schema=dict(ema_swapped=False)),
        reference_metadata_read=False,FIT_performed=False,ownership_verified=False,quality_verified=False,adoption=False,AMP_used=False,TF32_used=False,images=[])
    for i,(image,bank)in enumerate(zip(images,banks)):
        value=result(i%2);payload={n:getattr(value,n)for n in p.FIELDS[:15]};payload.update(image_size=np.asarray(value.image_size,np.int64),original_frame_index=np.asarray(0,np.int64),original_slot=np.asarray(i,np.int64),acquired_ordinal=np.asarray(i,np.int64))
        (out/bank['file']).write_bytes(f'authored-{i}'.encode());(out/bank['file']).chmod(0o400)
        report['images'].append(dict(image_id=image['image_id'],original_slot=i,acquired_ordinal=i,original_frame_index=0,image_size=[4,5],
            file=bank['file'],identity=p.rt.identity(out/bank['file']),arrays={n:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())for n,a in payload.items()},
            native_detections=len(value.query_ids),hand_object_pairs=len(value.hand_object_pairs),object_target_pairs=len(value.object_target_pairs),endpoint_bank_identity=bank['identity'],source_person_ids=bank['person_ids'],owl_patches=3600))
    return report,dict(images=images,banks=banks),'1'*40,pin


def test_full_native_validator(monkeypatch,tmp_path):
    report,proof,rev,pin=native_fixture(monkeypatch,tmp_path);p.validate_native(report,proof,rev,'model',pin)
    for edit in (lambda r:r.update(models_loaded=16),lambda r:r['images'][0]['arrays']['query_tokens'].update(shape=[1499,256]),
        lambda r:r.update(reference_metadata_read=True),lambda r:r['images'][0].update(source_person_ids=[]),lambda r:r['images'][0].update(owl_patches=10)):
        bad=copy.deepcopy(report);edit(bad)
        with pytest.raises(ValueError):p.validate_native(bad,proof,rev,'model',pin)


@pytest.mark.parametrize('newline',[b'',b'\n'])
def test_cleanup_exact_cid_and_name(monkeypatch,tmp_path,newline):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);cid='a'*64;rev='1'*40;name='own';(out/'model.cid').write_bytes(cid.encode()+newline)
    live=True;events=[]
    def command(args,deadline):
        nonlocal live
        events.append(args)
        if args[1]=='ps':return cid if live else ''
        if args[1]=='inspect':return p.IMAGE+'|/'+name+'|'+p.ENTRY+'|'+rev
        if args[1]=='rm':live=False;return cid
        raise AssertionError(args)
    monkeypatch.setattr(p.public.bank,'command',command);p.cleanup('model',name,rev,100)
    assert not live and len([a for a in events if a[1]=='ps'])==3
    assert (out/'model.cid').stat().st_mode&0o777==0o400


def test_cleanup_rejects_cid_survivor_and_foreign_rename(monkeypatch,tmp_path):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);cid='a'*64;(out/'model.cid').write_bytes(cid.encode());removed=[]
    def command(a,d):
        if a[1]=='ps':return cid
        if a[1]=='inspect':return p.IMAGE+'|/foreign|'+p.ENTRY+'|'+'1'*40
        if a[1]=='rm':removed.append(a);return cid
    monkeypatch.setattr(p.public.bank,'command',command)
    with pytest.raises(ValueError):p.cleanup('model','own','1'*40,100)
    assert not removed
    monkeypatch.setattr(p.public.bank,'command',lambda a,d:p.IMAGE+'|/own|'+p.ENTRY+'|'+'1'*40 if a[1]=='inspect'else cid)
    with pytest.raises(ValueError):p.cleanup('model','own','1'*40,100)


def test_launch_cpu_and_gpu_individual_public_mounts(monkeypatch,tmp_path):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);code=tmp_path/'code';code.mkdir();monkeypatch.setattr(p,'NATIVE_FILES',('x.py',))
    for f in (code/'x.py',code.parent/'revision',code.parent/'source-sha256',tmp_path/'input.jpg'):
        f.write_bytes(b'x');f.chmod(0o400)
    proof=dict(files={str(tmp_path/'input.jpg'):p.rt.identity(tmp_path/'input.jpg')});commands=[]
    monkeypatch.setattr(p.public.bank,'command',lambda *a:'');monkeypatch.setattr(p,'cleanup',lambda *a:None)
    monkeypatch.setattr(p,'validate_native',lambda *a:None)
    def run(args,**kwargs):
        commands.append(args);phase=args[args.index('--phase')+1];p.rt.write(out/(phase+'.json'),p.encode({'status':'pass'}));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run)
    p.launch(code,'1'*40,'overlay',proof,1000000000.);p.launch(code,'1'*40,'model',proof,1000000000.)
    cpu,gpu=commands
    assert '--gpus'not in cpu and gpu[gpu.index('--gpus')+1]=='driver=nvidia,count=all'
    assert cpu[cpu.index('--memory')+1]=='16g'and gpu[gpu.index('--memory')+1]=='64g'
    assert cpu[cpu.index('--network')+1]==gpu[gpu.index('--network')+1]=='none'
    assert 'type=bind,src='+str(out)+'/.overlay,dst='+str(out)+'/.overlay,readonly'in gpu
    assert '--no-index'not in gpu
    assert all('eval_private'not in a and 'cohort'not in a for a in gpu)


def test_arguments_do_not_allow_native_input_overrides():
    assert p.arguments([]).phase is None
    with pytest.raises(ValueError):p.arguments(['--proof-bytes','1','--proof-sha256','a'*64])
    with pytest.raises(ValueError):p.arguments(['--phase','model'])
    a=p.arguments(['--phase','model','--proof-bytes','1','--proof-sha256','a'*64,'--deadline','99','--code','/readonly/code','--revision','1'*40])
    assert a.pin==dict(bytes=1,sha256='a'*64)


def test_full_fake_model_child_one_constructor_all16_roundtrip(monkeypatch,tmp_path):
    import hoi_detr_native_model as neutral
    import coco_endpoint_evaluate as saved
    import world_reward.hoi_detr_observations as observations
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out)
    (out/'.overlay/source').mkdir(parents=True);(out/'.overlay/site').mkdir()
    code,rev,proof=fake_native_source(monkeypatch,tmp_path)
    images,banks=fixture_rows();policy=__import__('json').loads(Path('configs/hoi_detr_model_qualify_v1.json').read_bytes())
    leaf=tmp_path/'original-input';leaf.write_bytes(b'authored');leaf.chmod(0o400)
    proof.update(files={str(leaf):p.rt.identity(leaf)},images=images,banks=banks,started_monotonic=p.time.monotonic(),
        runtime=dict(configuration={},manifest=dict(artifacts=[]),report=dict(native_extension_identity=p.public.pin(b'extension'))),derived_source_inventory=[],overlay=dict(site_inventory=[]))
    proof_path=out/'model_proof.json';p.rt.write(proof_path,p.encode(proof));pin=p.rt.identity(proof_path)
    pinned=p.rt.pinned
    monkeypatch.setattr(p.rt,'pinned',lambda path,wanted,maximum:pinned(path,wanted,maximum)if path==proof_path else policy)
    monkeypatch.setattr(p.original,'check_inventory',lambda *a:None)
    events=[];backend=SimpleNamespace(allow_tf32=False)
    torch=SimpleNamespace(manual_seed=lambda n:events.append('seed'),cuda=SimpleNamespace(manual_seed_all=lambda n:events.append('cudaseed'),
        synchronize=lambda:events.append('sync'),empty_cache=lambda:events.append('empty')))
    monkeypatch.setattr(p.runtime,'versions',lambda _: (torch,np,{}))
    versions={**policy['extra_base_distributions'],'fairscale':'0.4.13','terminaltables':'3.1.10'}
    monkeypatch.setattr(p.importlib.metadata,'version',lambda n:versions[n])
    mmcv=ModuleType('mmcv');mmcv.__path__=[];mmcv.__file__='/opt/world-reward-hoi-mmcv/source/mmcv/__init__.py'
    ext=ModuleType('mmcv._ext');ext.__file__=str(leaf);mmcv._ext=ext
    monkeypatch.setitem(sys.modules,'mmcv',mmcv);monkeypatch.setitem(sys.modules,'mmcv._ext',ext)
    proof['runtime']['report']['native_extension_identity']=p.rt.identity(leaf)
    proof_path.chmod(0o600);proof_path.write_bytes(p.encode(proof));proof_path.chmod(0o400);pin=p.rt.identity(proof_path)
    monkeypatch.setattr(saved,'validate_npz',lambda path,row:events.append(('validate',row['original_slot'])))
    provenance=dict(strict_checkpoint=dict(keys=1796,strict=True,weights_only=True),checkpoint_buffer_schema=dict(ema_swapped=False))
    def build(**kwargs):
        assert events[:16]==[('validate',i)for i in range(16)]
        assert kwargs['native_config']==policy['native_config']and kwargs['checkpoint_buffers']==policy['checkpoint_buffers']
        events.append('construct');return SimpleNamespace(model='real-original-contract',operations='ops',provenance=provenance)
    monkeypatch.setattr(neutral,'build_original_hoi',build)
    def infer(model,rgb,frame,ops):
        assert model=='real-original-contract'and ops=='ops'and frame==0
        events.append('forward');return result(len([x for x in events if x=='forward'])%2)
    monkeypatch.setattr(observations,'infer_hoi_detr_frame',infer)
    monkeypatch.setattr(p.public.bank.rgb_inputs,'decode_rgb',lambda *a,**k:np.zeros((4,5,3),np.uint8))
    monkeypatch.setattr(p.os,'geteuid',lambda:0)
    original_iter=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([Path('lo')])if str(path)=='/sys/class/net'else original_iter(path))
    monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.setenv('WR_IMAGE_ID',p.IMAGE);report=p.native(code,rev,'model',pin,p.time.monotonic()+10)
    assert report['status']=='pass'and report['models_loaded']==1 and report['native_forward_calls']==16
    assert events.count('construct')==1 and events.count('forward')==16
    assert report['source_inputs_runtime_assets_rehashed_after']is True
    p.validate_native(report,proof,rev,'model',pin)
    saved_report=pinned(out/'model.json',p.rt.identity(out/'model.json'),2 << 20)
    assert saved_report['status']=='pass'
    for i,row in enumerate(report['images']):
        with np.load(out/row['file'],allow_pickle=False)as z:
            assert len(z.files)==19 and z['query_logits'].shape==(1500,3)and z['query_tokens'].shape==(1500,256)
            assert z['original_slot']==i and z['acquired_ordinal']==i
            assert not len(z['hand_object_logits'])or(z['hand_object_logits']<0).all()


@pytest.mark.parametrize('which',['success','late','foreign'])
def test_complete_fake_host_finally_and_publication(monkeypatch,tmp_path,which):
    out=tmp_path/'out';monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setattr(p.sys,'platform','linux');monkeypatch.setattr(p.os,'geteuid',lambda:0)
    monkeypatch.setattr(p.os,'uname',lambda:SimpleNamespace(nodename='world-reward-ncc-h100-02'))
    before=dict(source=dict(helpers={}),states='frozen');monkeypatch.setattr(p,'source',lambda *a:before)
    images,banks=fixture_rows();proof_input=dict(source={'original':'true'},states='unchanged')
    monkeypatch.setattr(p.public,'export_inputs',lambda *a:(dict(files={}),{},proof_input))
    policy=__import__('json').loads(Path('configs/hoi_detr_model_qualify_v1.json').read_bytes())
    monkeypatch.setattr(p.rt,'pinned',lambda *a:policy if str(a[0]).endswith(p.original.PROTOCOL)else dict(images=banks))
    qual=dict(image=dict(Id=p.IMAGE),report_identity=p.public.pin(b'runtime'),frozen={});models=dict(report_identity=p.public.pin(b'assets'),frozen={})
    monkeypatch.setattr(p.original,'authenticate_runtime',lambda *a:qual);monkeypatch.setattr(p.original,'authenticate_acquisition',lambda *a:(models,{}))
    monkeypatch.setattr(p,'qualified_model',lambda:dict(files={}))
    def fetch(mp,acq,folder,row,deadline,*rest):
        p.rt.write(folder/row['file'],b'authored')
        return dict(file=row['file'],bytes=row['bytes'],sha256=row['sha256'])
    monkeypatch.setattr(p.runtime,'fetch',fetch)
    def derive(rt,acq,manifest,target,policy):target.mkdir();return dict(remaining_AST_unchanged=True)
    monkeypatch.setattr(p.original,'derive_source',derive);monkeypatch.setattr(p.original,'inventory',lambda *a:[])
    monkeypatch.setattr(p.public.bank,'public_inputs',lambda *a:dict(images=images))
    lock=tmp_path/'jobs/.world-reward-h100.lock';lock.parent.mkdir();lock.write_bytes(b'');lock.chmod(0o444)
    monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p.runtime,'command',lambda *a:'');events=[]
    def launch(code,rev,phase,proof,deadline):
        events.append(phase)
        p.rt.write(out/(phase+'.json'),p.encode(dict(status='pass')))
        if phase=='overlay':(out/'.overlay/site').mkdir();return dict(site_inventory=[])
        if which=='foreign':p.rt.write(out/'unexpected-secret.bin',b'foreign')
        return dict(images=[])
    monkeypatch.setattr(p,'launch',launch);monkeypatch.setattr(p,'cleanup',lambda *a:events.append('cleanup'))
    if which=='late':monkeypatch.setattr(p,'BUDGET',0)
    r=p.run(tmp_path/'code','1'*40)
    assert events[:2]==['overlay','model']and events[-2:]==['cleanup','cleanup']
    assert r['status']==('pass'if which=='success'else 'fail')
    saved=__import__('json').loads((out/'report.json').read_bytes());assert saved['status']==r['status']
    assert not(out/'.overlay').exists()and not(out/'.scratch').exists()
    assert r['source_inputs_runtime_assets_rehashed_after']is True and r['owned_cleanup_verified']is True
    if which=='foreign':assert(out/'unexpected-secret.bin').read_bytes()==b'foreign'


def test_offline_cpu_overlay_original_source_and_no_image_install(monkeypatch,tmp_path):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);scratch=out/'.scratch';scratch.mkdir();(out/'.overlay/source').mkdir(parents=True)
    policy=__import__('json').loads(Path('configs/hoi_detr_model_qualify_v1.json').read_bytes())
    for row in [policy['fairscale'],policy['fairscale']['publisher_license'],*[r for w in policy['import_wheels']for r in (w,w['publisher_license'])]]:
        p.rt.write(scratch/row['file'],b'authored')
    def unpack(rt,acq,path,target,policy,deadline,check):
        rt.write(target/'setup.py',b'authored_original_setup');check(deadline)
    monkeypatch.setattr(p.original,'unpack_fairscale',unpack)
    commands=[]
    def run(args,**kwargs):
        commands.append((args,kwargs))
        assert kwargs['env']['PIP_NO_INDEX']=='1'and kwargs['env']['BUILD_CUDA_EXTENSIONS']=='0'
        assert '--no-index'in args and '--no-deps'in args and '--no-cache-dir'in args
        if 'wheel'in args:
            wheel=Path(args[args.index('--wheel-dir')+1])/'fairscale-0.4.13-py3-none-any.whl'
            wheel.write_bytes(b'authored_pure_wheel');wheel.chmod(0o644)
        else:p.rt.write(Path(args[args.index('--target')+1])/'installed.py',b'authored_pure_overlay')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run);monkeypatch.setattr(p.runtime,'wheel_notice',lambda *a:dict(publisher_notice_checked=True))
    monkeypatch.setattr(p.runtime,'versions',lambda _: (SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),np,dict(original=True)))
    modules={n:ModuleType(n)for n in ('fairscale','fairscale.nn','fairscale.nn.checkpoint','mmcv','mmdet','mmdet.models','mmdet.models.builder','mmdet.datasets','mmdet.datasets.pipelines','projects','projects.models')}
    for n,m in modules.items():
        m.__path__=[];monkeypatch.setitem(sys.modules,n,m)
        if '.'in n:parent,leaf=n.rsplit('.',1);setattr(modules[parent],leaf,m)
    cfg=dict(load_from=policy['native_config']['original_load_from'],model=dict(train_cfg=[],backbone=dict(use_act_checkpoint=True),query_head=dict(num_query=1500,num_classes=3)))
    modules['mmcv'].Config=SimpleNamespace(fromfile=lambda path,import_custom_modules:SimpleNamespace(_cfg_dict=cfg))
    monkeypatch.setattr(sys,'path',list(sys.path))
    report=p.overlay_cpu(dict(runtime=dict(configuration={})),policy,p.time.monotonic()+10)
    assert len(commands)==2 and report['cuda_initialized']is report['base_installed']is False
    assert report['full_import_closure_qualified']is True and report['site_inventory'][0]['file']=='installed.py'
    assert cfg['load_from']is None and cfg['model']['train_cfg']is None
    assert (out/'.scratch/fairscale/setup.py').read_bytes()==b'authored_original_setup'
    assert commands[0][0][0:7]==[sys.executable,'-I','-B','-m','pip','--isolated','wheel']
    assert '--no-build-isolation'in commands[0][0]and '--no-compile'in commands[1][0]
    assert (scratch/'wheels/fairscale-0.4.13-py3-none-any.whl').stat().st_mode&0o777==0o444
    assert (scratch/'wheels').stat().st_mode&0o777==0o555
