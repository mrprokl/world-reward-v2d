"""Authored ABI48/raw1500 arrays and mocked lifecycle; no native models/HTTP."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType,SimpleNamespace

import numpy as np
import pytest
import vcoco_full_hoi_run as p
import test_vcoco_hoi_observations as authored
import test_vcoco_public_observation_context as endpoint_fixture
import test_vcoco_cardinality_observation_adapters as adapter_fixture


def seal(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);raw=value if type(value)is bytes else p.encode(value)
    if path.exists():path.chmod(0o600)
    path.write_bytes(raw);path.chmod(0o400);return p.rt.identity(path)


def policy():return json.loads(Path('configs/hoi_detr_model_qualify_v1.json').read_bytes())


def test_real_host_and_native_bootstrap_import_stdlib_no_private_or_old_run():
    root=Path(p.__file__).resolve().parents[1]
    script="""import importlib.abc,runpy,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','torch','transformers','PIL','vcoco_hoi_observations','vcoco_fit_cal_endpoint_run','vcoco_fit_cal_acquisition_inputs','vcoco_fit_cal_acquire','vcoco_fit_cal_context','vcoco_role_reference','sealed_callback_publication'}:raise ImportError('Forbidden native import')
sys.meta_path.insert(0,Deny())
runpy.run_path(sys.argv[1]+'/infra/vcoco_full_hoi_run.py',run_name='authored')
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',script,str(root)],capture_output=True)
    assert r.returncode==0,r.stderr
    assert p.BUDGET==1800 and p.MAX_BANK==8<<20 and len(p.public.HELPERS)==17
    assert set(p.public.HELPERS)<=set(p.NATIVE_FILES)and all(Path(n).exists()for n in p.HELPERS)
    assert all(not any(x in n for x in('vcoco_hoi_observations','endpoint_run','acquisition_inputs','role_reference','sealed_callback'))for n in p.NATIVE_FILES)
    tree=ast.parse(Path(p.__file__).read_text());assert not any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)for t in n.targets)for n in ast.walk(tree))
    s=Path(p.__file__).read_text();assert 'neutral.build_original_hoi('in s and 'infer_hoi_detr_frame(built.model,rgb,0,built.operations)'in s
    assert all(x not in s for x in('original.run(','original.gpu_model(','original.cpu_overlay(','person_bank_profile','authenticate_person_banks','manual_prompt','topk('))


def native_source_fixture(tmp_path,monkeypatch):
    code=tmp_path/'code';code.mkdir();revision='1'*40
    names=('infra/vcoco_full_hoi_run.py','two.py');monkeypatch.setattr(p,'NATIVE_FILES',names)
    monkeypatch.setattr(p,'__file__',str(code/names[0]));monkeypatch.setattr(p.public,'HELPERS',names)
    monkeypatch.setattr(p.public,'source_identity',lambda *a:None)
    pins={}
    for n in names:pins[n]=seal(code/n,n.encode())
    markers={n:seal(code.parent/n,raw)for n,raw in(('revision',(revision+'\n').encode()),('source-sha256',('a'*64+'\n').encode()))}
    return code,revision,dict(revision=revision,image_id=p.IMAGE,native_files=pins,markers=markers)


def test_native_source_exact_mount_whitelist_markers_and_changes(tmp_path,monkeypatch):
    code,rev,proof=native_source_fixture(tmp_path,monkeypatch);p.native_source(code,rev,proof)
    seal(code/'extra.py',b'private')
    with pytest.raises(ValueError):p.native_source(code,rev,proof)
    (code/'extra.py').unlink();proof['markers']['revision']=seal(code.parent/'revision',(rev+'\\n').encode())
    with pytest.raises(ValueError):p.native_source(code,rev,proof)


def test_host_endpoint_actual_pins_projection_only_and_no_numeric_reader(tmp_path,monkeypatch):
    context,args,images,rows,host,native=endpoint_fixture.fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(p,'ENDPOINT',context.endpoint_banks);monkeypatch.setattr(p,'DATA',context.inputs)
    monkeypatch.setattr(p,'ENDPOINT_REV',args['endpoint_source']['producer_revision'])
    pins={n:p.rt.identity(context.endpoint_banks/n)for n in('report.json','native.json','proof.json')};monkeypatch.setattr(p,'ENDPOINT_PINS',pins)
    import vcoco_fit_cal_endpoint_run as endpoint
    monkeypatch.setattr(endpoint,'__file__',str(args['code']/'infra/vcoco_fit_cal_endpoint_run.py'))
    events=[];monkeypatch.setattr(endpoint,'validate_native',lambda *a:events.append('metadata_only'))
    monkeypatch.setattr(p.rt,'source',lambda *a:args['endpoint_source'])
    old=p.ROOT/'jobs'/p.ENDPOINT_REV/endpoint.ENTRY/'code';fake=tmp_path/'old';fake.mkdir()
    monkeypatch.setattr(p,'ROOT',tmp_path);old=tmp_path/'jobs'/p.ENDPOINT_REV/endpoint.ENTRY/'code';old.mkdir(parents=True)
    projection,proof=p.endpoint_inputs(args['code'],args['current_source'],time.monotonic()+30)
    assert events==['metadata_only']and set(projection)=={'schema','inputs','manifest_identity','banks'}and len(projection['banks'])==48
    assert all(set(r)==p.public.BANK_FIELDS for r in projection['banks'])
    raw=p.encode(projection);assert not any(x in raw for x in(b'source_binding',b'qualification',b'private',b'split'))
    assert len(proof['files'])==52 and proof['source']==args['endpoint_source']
    host['status']='fail';pins['report.json']=seal(context.endpoint_banks/'report.json',host)
    with pytest.raises(ValueError):p.endpoint_inputs(args['code'],args['current_source'],time.monotonic()+30)


def valid_native(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out)
    images,banks=authored.fixture_rows();images*=3;banks*=3
    projection=dict(inputs=dict(images=[]),banks=[]);report=dict(schema=p.SCHEMA,stage='native_full48_hoi_model',status='pass',phase='complete',
        producer_revision='1'*40,image_id=p.IMAGE,proof_identity=p.encode({'pin':1}),models_loaded=1,models_released=True,native_forward_calls=48,completed_bank_count=48,
        source_inputs_runtime_assets_rehashed_after=True,model=dict(strict_checkpoint=dict(keys=1796,strict=True,weights_only=True),checkpoint_buffer_schema=dict(ema_swapped=False)),
        images=[],**{k:False for k in p.FLAGS})
    for i in range(48):
        image=dict(image_id=f'{i:032x}',height=4,width=5);bank=dict(file=f'image_{i:06d}.npz',identity=dict(bytes=1,sha256='b'*64),person_ids=[f'person-{i}'],image_size=[4,5])
        projection['inputs']['images'].append(image);projection['banks'].append(bank)
        value=authored.result(i%2);arrays={n:getattr(value,n)for n in p.adapter.HOI_FIELDS[:15]}
        arrays.update(image_size=np.asarray([4,5],np.int64),original_frame_index=np.asarray(0,np.int64),original_slot=np.asarray(i,np.int64),acquired_ordinal=np.asarray(i,np.int64))
        pin=seal(out/bank['file'],f'authored-{i}'.encode())
        report['images'].append(dict(image_id=image['image_id'],original_slot=i,acquired_ordinal=i,original_frame_index=0,image_size=[4,5],
            endpoint_bank_identity=bank['identity'],source_person_ids=bank['person_ids'],owl_patches=3600,file=bank['file'],identity=pin,
            arrays=p.public.endpoint.array_identities(arrays),native_detections=len(value.query_ids),hand_object_pairs=len(value.hand_object_pairs),object_target_pairs=len(value.object_target_pairs)))
    return report,projection


def test_full48_metadata_validator_no_fake_pass_or_1500_drop(tmp_path,monkeypatch):
    report,projection=valid_native(tmp_path,monkeypatch);p.validate_native(report,{},'1'*40,'model',report['proof_identity'],projection)
    edits=[lambda r:r['images'].pop(),lambda r:r.update(models_loaded=48),lambda r:r.update(native_forward_calls=True),lambda r:r.update(FIT_performed=True),
        lambda r:r['images'][47].update(original_slot=16),lambda r:r['images'][0].update(source_person_ids=[]),lambda r:r['images'][0]['arrays']['query_tokens'].update(shape=[1499,256]),
        lambda r:r['model']['checkpoint_buffer_schema'].update(ema_swapped=True)]
    for edit in edits:
        bad=deepcopy(report);edit(bad)
        with pytest.raises(ValueError):p.validate_native(bad,{},'1'*40,'model',report['proof_identity'],projection)


@pytest.mark.parametrize('phase',['overlay','model'])
def test_launch_leaf_public_whitelist_cpu_gpu_bounds_and_no_old_receipts(tmp_path,monkeypatch,phase):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);work=out/'.work';work.mkdir();monkeypatch.setattr(p,'WORK',work)
    code=tmp_path/'code';code.mkdir();monkeypatch.setattr(p,'NATIVE_FILES',('x.py',))
    data=tmp_path/'inputs';data.mkdir();end=tmp_path/'endpoint';end.mkdir();banks=tmp_path/'virtual';monkeypatch.setattr(p,'DATA',data);monkeypatch.setattr(p,'ENDPOINT',end);monkeypatch.setattr(p,'BANKS',banks)
    for path in(code/'x.py',code.parent/'revision',code.parent/'source-sha256',data/'manifest.json',data/'image_000000.jpg',end/'image_000000.npz'):seal(path,b'x')
    asset=tmp_path/'checkpoint';seal(asset,b'weight');proof={'files':{str(asset):p.rt.identity(asset)}}
    projection=dict(inputs=dict(images=[dict(file='image_000000.jpg')]),banks=[dict(file='image_000000.npz')]);commands=[];events=[]
    monkeypatch.setattr(p.runtime,'command',lambda *a:'');monkeypatch.setattr(p,'cleanup',lambda *a:events.append('cleanup'));monkeypatch.setattr(p,'validate_native',lambda *a:None)
    def run(cmd,**kw):
        commands.append(cmd);seal(out/(phase+'.json'),{'status':'pass'});return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run);p.launch(code,'1'*40,phase,proof,projection,time.monotonic()+30)
    cmd=commands[0];assert events==['cleanup']and cmd[cmd.index('--memory')+1]==('64g'if phase=='model'else '16g')
    assert ('--gpus'in cmd)==(phase=='model')and cmd[cmd.index('--network')+1]=='none'
    if phase=='model':
        assert f'type=bind,src={end}/image_000000.npz,dst={banks}/image_000000.npz,readonly'in cmd
        assert f'type=bind,src={data}/manifest.json,dst={data}/manifest.json,readonly'in cmd
        assert f'type=bind,src={work},dst={work},readonly'in cmd
    assert all(not any(n in a for n in('report.json','vcoco_fit_cal_acquisition_inputs','vcoco_fit_cal_context','role_reference'))for a in cmd)
    assert kw_env(commands,p.runtime.SAFE_ENV) is None


def kw_env(commands,env):assert env['DOCKER_HOST']=='unix://'+str(p.ROOT/'docker.sock')


@pytest.mark.parametrize('newline',[b'',b'\n'])
def test_owned_cleanup_exact_cid_and_name_after_remove(tmp_path,monkeypatch,newline):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);cid='a'*64;(out/'model.cid').write_bytes(cid.encode()+newline)
    live=True;events=[]
    def command(a,d):
        nonlocal live
        events.append(a)
        if a[1]=='ps':return cid if live else ''
        if a[1]=='inspect':return p.IMAGE+'|/own|'+p.ENTRY+'|'+'1'*40
        if a[1]=='rm':live=False;return cid
        raise AssertionError(a)
    monkeypatch.setattr(p.runtime,'command',command);p.cleanup('model','own','1'*40,time.monotonic()+5)
    assert not live and len([a for a in events if a[1]=='ps'])==3 and(out/'model.cid').stat().st_mode&0o777==0o400


def test_cleanup_foreign_or_renamed_surviving_cid_never_deleted(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);cid='a'*64;(out/'model.cid').write_bytes(cid.encode());removed=[]
    def cmd(a,d):
        if a[1]=='ps':return cid
        if a[1]=='inspect':return p.IMAGE+'|/foreign|'+p.ENTRY+'|'+'1'*40
        if a[1]=='rm':removed.append(a);return cid
    monkeypatch.setattr(p.runtime,'command',cmd)
    with pytest.raises(ValueError):p.cleanup('model','own','1'*40,time.monotonic()+5)
    assert not removed
    monkeypatch.setattr(p.runtime,'command',lambda a,d:p.IMAGE+'|/own|'+p.ENTRY+'|'+'1'*40 if a[1]=='inspect'else cid)
    with pytest.raises(ValueError):p.cleanup('model','own','1'*40,time.monotonic()+5)


def forward_fixture(tmp_path,monkeypatch):
    ref=adapter_fixture.setup(tmp_path,monkeypatch);monkeypatch.setattr(p,'OUTPUT',ref.context.output);work=ref.context.output/'.work';(work/'site').mkdir(parents=True);(work/'source').mkdir()
    monkeypatch.setattr(p,'WORK',work);monkeypatch.setattr(p,'reference',lambda *a:ref);monkeypatch.setattr(p.original,'check_inventory',lambda *a:None)
    events=[];torch=SimpleNamespace(manual_seed=lambda _:events.append('seed'),cuda=SimpleNamespace(manual_seed_all=lambda _:events.append('cudaseed'),empty_cache=lambda:events.append('empty'),synchronize=lambda:events.append('sync')))
    monkeypatch.setattr(p.runtime,'versions',lambda _: (torch,np,{}));cfg=policy();versions={**cfg['extra_base_distributions'],'fairscale':'0.4.13',**{r['name']:r['version']for r in cfg['import_wheels']}}
    monkeypatch.setattr(p.importlib.metadata,'version',lambda n:versions[n]);leaf=tmp_path/'extension';ext_pin=seal(leaf,b'original-extension')
    mmcv=ModuleType('mmcv');mmcv.__file__='/opt/world-reward-hoi-mmcv/source/mmcv/__init__.py';mmcv.__path__=[]
    ext=ModuleType('mmcv._ext');ext.__file__=str(leaf);mmcv._ext=ext;monkeypatch.setitem(sys.modules,'mmcv',mmcv);monkeypatch.setitem(sys.modules,'mmcv._ext',ext)
    import hoi_detr_native_model as neutral
    import world_reward.hoi_detr_observations as observations
    provenance=dict(strict_checkpoint=dict(keys=1796,strict=True,weights_only=True),checkpoint_buffer_schema=dict(ema_swapped=False))
    def build(**kw):events.append('construct');return SimpleNamespace(model='original',operations='original',provenance=provenance)
    monkeypatch.setattr(neutral,'build_original_hoi',build)
    def infer(*args):events.append('forward');return adapter_fixture.hoi(empty=events.count('forward')%2==0)
    monkeypatch.setattr(observations,'infer_hoi_detr_frame',infer);monkeypatch.setattr(sys,'path',list(sys.path))
    proof=dict(overlay=dict(site_inventory=[]),runtime_configuration={},extension_identity=ext_pin)
    return ref,proof,cfg,events


def test_one_original_constructor_then48_roundtrips_all1500_and_k0(tmp_path,monkeypatch):
    ref,proof,cfg,events=forward_fixture(tmp_path,monkeypatch);report=dict(images=[])
    p.forward(ref._code,proof,cfg,time.monotonic()+30,report)
    assert events.count('construct')==1 and events.count('forward')==report['native_forward_calls']==48 and report['models_released']is True
    assert len(report['images'])==48 and report['images'][0]['hand_object_pairs']==4 and report['images'][1]['hand_object_pairs']==0
    p.adapter.validate_saved_records(ref,'hoi',report['images'],time.monotonic()+30)
    assert all(len(r['arrays'])==19 for r in report['images'])


@pytest.mark.parametrize('fault',['bank','forward'])
def test_prevalidation_or_failure_never_fakes_complete_and_releases(tmp_path,monkeypatch,fault):
    ref,proof,cfg,events=forward_fixture(tmp_path,monkeypatch);report=dict(images=[])
    if fault=='bank':
        monkeypatch.setattr(p,'reference',lambda *a:(_ for _ in()).throw(ValueError('last fullbank invalid')))
        with pytest.raises(ValueError):p.forward(ref._code,proof,cfg,time.monotonic()+30,report)
        assert 'construct'not in events and 'forward'not in events
    else:
        import world_reward.hoi_detr_observations as observations
        def fail(*args):events.append('forward');raise RuntimeError('upstream failure')
        monkeypatch.setattr(observations,'infer_hoi_detr_frame',fail)
        with pytest.raises(RuntimeError):p.forward(ref._code,proof,cfg,time.monotonic()+30,report)
        assert report['models_released']is True and not report['images']and report['native_forward_calls']==1 and 'empty'in events


def test_native_fail_receipt_is_sealed_and_diagnostic_safe(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir(mode=0o700);monkeypatch.setattr(p,'OUTPUT',out)
    code,rev,proof=native_source_fixture(tmp_path,monkeypatch);proof['unexpected_private']='secret'
    proof.update(files={},started_monotonic=time.monotonic());pin=seal(out/'model_proof.json',proof)
    report=p.native(code,rev,'model',pin,time.monotonic()+5)
    assert report['status']=='fail'and report['phase']=='authentication'and report['failure_stage']=='authentication'and report['error_type']=='ValueError'
    saved=json.loads((out/'model.json').read_bytes());assert saved['status']=='fail'and saved['native_forward_calls']==0
    assert 'secret'not in(out/'model.json').read_text()and(out/'model.json').stat().st_mode&0o777==0o400


def test_native_only_args_and_duplicate_or_context_override_rejected():
    assert p.arguments([]).phase is None
    with pytest.raises(ValueError):p.arguments(['--proof-bytes','1','--proof-sha256','a'*64])
    with pytest.raises(ValueError):p.arguments(['--phase','model'])
    with pytest.raises(ValueError):p.arguments(['--phase','model','--phase','model'])
    a=p.arguments(['--phase','model','--proof-bytes','1','--proof-sha256','a'*64,'--deadline','99','--code','/own/code','--revision','1'*40])
    assert a.pin==dict(bytes=1,sha256='a'*64)


def test_cpu_pure_overlay_real0644_wheel_sealed_before_pin(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();work=out/'.work';scratch=work/'scratch';scratch.mkdir(parents=True);(work/'source').mkdir()
    monkeypatch.setattr(p,'WORK',work);cfg=policy()
    for row in[cfg['fairscale'],cfg['fairscale']['publisher_license'],*[r for w in cfg['import_wheels']for r in(w,w['publisher_license'])]]:seal(scratch/row['file'],b'authored')
    monkeypatch.setattr(p.original,'unpack_fairscale',lambda rt,acq,path,target,*a:seal(target/'setup.py',b'original_setup'))
    commands=[]
    def run(cmd,**kw):
        commands.append((cmd,kw));assert kw['env']['PIP_NO_INDEX']=='1'and kw['env']['BUILD_CUDA_EXTENSIONS']=='0'
        if 'wheel'in cmd:
            path=Path(cmd[cmd.index('--wheel-dir')+1])/'fairscale-0.4.13-py3-none-any.whl';path.write_bytes(b'pure_wheel');path.chmod(0o644)
        else:seal(Path(cmd[cmd.index('--target')+1])/'installed.py',b'original_pure')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess,'run',run);monkeypatch.setattr(p.runtime,'wheel_notice',lambda *a:dict(notice=True))
    monkeypatch.setattr(p.runtime,'versions',lambda _: (SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False)),np,{}))
    names=('fairscale','fairscale.nn','fairscale.nn.checkpoint','mmcv','mmdet','mmdet.models','mmdet.models.builder','mmdet.datasets','mmdet.datasets.pipelines','projects','projects.models')
    modules={n:ModuleType(n)for n in names}
    for n,module in modules.items():
        module.__path__=[];monkeypatch.setitem(sys.modules,n,module)
        if '.'in n:parent,name=n.rsplit('.',1);setattr(modules[parent],name,module)
    configuration=dict(load_from=cfg['native_config']['original_load_from'],model=dict(train_cfg=[],backbone=dict(use_act_checkpoint=True),query_head=dict(num_query=1500,num_classes=3)))
    modules['mmcv'].Config=SimpleNamespace(fromfile=lambda path,import_custom_modules:SimpleNamespace(_cfg_dict=configuration))
    monkeypatch.setattr(sys,'path',list(sys.path));result=p.overlay(cfg,dict(runtime_configuration={}),time.monotonic()+5)
    assert len(commands)==2 and commands[0][0][:7]==[sys.executable,'-I','-B','-m','pip','--isolated','wheel']
    assert result['cuda_initialized']is result['base_installed']is False and configuration['load_from']is None
    assert (scratch/'wheels').stat().st_mode&0o777==0o555 and(scratch/'wheels/fairscale-0.4.13-py3-none-any.whl').stat().st_mode&0o777==0o444
    assert '--no-build-isolation'in commands[0][0]and '--no-compile'in commands[1][0]


@pytest.mark.parametrize('fault',['success','failed_child','posthash','foreign'])
def test_host_finally_flat_sealing_cleanup_and_failure_preserved(tmp_path,monkeypatch,fault):
    out=tmp_path/'result';root=tmp_path/'root';root.mkdir();monkeypatch.setattr(p,'ROOT',root);monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setattr(p,'WORK',out/'.work')
    lock=root/'jobs/.world-reward-h100.lock';lock.parent.mkdir();lock.write_bytes(b'');lock.chmod(0o444)
    src=dict(binding=dict(helpers={n:dict(bytes=1,sha256='a'*64)for n in p.NATIVE_FILES},markers={}),states='frozen')
    projection=dict(inputs=dict(images=[]),banks=[]);endpoint_proof=dict(source={})
    calls=[]
    def source(*a):
        calls.append('source')
        if fault=='posthash'and calls.count('source')>1:raise ValueError('authored changed')
        return src
    monkeypatch.setattr(p,'source',source);monkeypatch.setattr(p,'endpoint_inputs',lambda *a:(projection,endpoint_proof))
    cfg=policy();runtime=dict(configuration={},manifest=dict(artifacts=[]),report=dict(native_extension_identity={}),image=dict(Id=p.IMAGE),frozen={},report_identity={})
    assets=dict(frozen={},report_identity={});prior=dict(files={});monkeypatch.setattr(p,'qualified_inputs',lambda *a:(cfg,runtime,assets,{},prior))
    monkeypatch.setattr(p.runtime,'command',lambda *a:'');monkeypatch.setattr(p.runtime,'fetch',lambda mp,acq,data,row,*a:seal(data/row['file'],b'authored'))
    def derive(rt,acq,manifest,target,cfg):target.mkdir();seal(target/'original.py',b'original');return dict(remaining_AST_unchanged=True)
    monkeypatch.setattr(p.original,'derive_source',derive);events=[]
    def launch(code,revision,phase,proof,projection,deadline):
        events.append(phase);value=dict(status='pass',phase='complete',models_loaded=0,native_forward_calls=0,completed_bank_count=0)
        seal(out/(phase+'_proof.json'),proof);seal(out/(phase+'.json'),value)
        if phase=='overlay':(out/'.work/site').mkdir();value.update(site_inventory=[],wheel_identity={},notices={})
        else:
            if fault=='failed_child':value.update(status='fail',phase='all48_original_forwards',error_type='RuntimeError');seal(out/'model.json',value);raise ValueError('closed native failure')
            if fault=='foreign':seal(out/'foreign.bin',b'foreign remains')
            value['images']=[]
        return value,p.rt.identity(out/(phase+'_proof.json'))
    monkeypatch.setattr(p,'launch',launch);monkeypatch.setattr(p,'cleanup',lambda *a:events.append('cleanup'))
    start=time.monotonic()
    if fault=='foreign':
        with pytest.raises(ValueError):p.host(tmp_path/'code','1'*40,start,start+15)
        assert(out/'foreign.bin').read_bytes()==b'foreign remains';return
    result=p.host(tmp_path/'code','1'*40,start,start+15)
    assert events==['overlay','model','cleanup','cleanup']and not(out/'.work').exists()
    assert result['status']==('pass'if fault=='success'else 'fail')and result['outputs_sealed']is True
    saved=json.loads((out/'report.json').read_bytes());assert saved['status']==result['status']and out.stat().st_mode&0o777==0o500
    if fault=='failed_child':assert saved['model_summary']['error_type']=='RuntimeError'and saved['model_summary']['status']=='fail'
