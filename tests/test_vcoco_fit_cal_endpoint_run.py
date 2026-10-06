"""Authored tiny inputs and mocked Docker/model lifecycle; no actual inference."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import vcoco_fit_cal_endpoint_run as p
import test_vcoco_fit_cal_endpoint_observations as authored


def seal(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=p.original.encode(value)if not isinstance(value,bytes)else value
    path.write_bytes(raw);path.chmod(0o400)
    return p.rt.identity(path)


def configured(tmp_path,monkeypatch):
    context,inputs,input_pin=authored.fixture(tmp_path)
    context.output.chmod(0o700)
    monkeypatch.setattr(p,'OUTPUT',context.output);monkeypatch.setattr(p,'DATA',context.inputs)
    monkeypatch.setattr(p,'CONTEXT',context)
    code=tmp_path/'code';code.mkdir()
    assets=tmp_path/'asset';asset_pin=seal(assets,b'authored assets')
    policy={'policy':'source-only authored'}
    pins={n:dict(bytes=1,sha256='1'*64)for n in p.NATIVE_FILES}
    source={'producer_revision':'a'*40,'helpers':pins,'markers':{}}
    before={n:dict(pin=pin,state=(1,2))for n,pin in pins.items()}
    runtime={'wheel_RECORD_identities':{'transformers':dict(bytes=1,sha256='2'*64)}}
    proof=dict(source=source,native_files=pins,assets={str(assets):asset_pin},owl_runtime=runtime,
        inputs_identity=input_pin,images=48,image_id=p.IMAGE,profile=deepcopy(p.seam.PROFILE))
    proof_pin=seal(context.output/'proof.json',proof)
    monkeypatch.setattr(p,'native_source',lambda *a:deepcopy(before))
    monkeypatch.setattr(p.original,'configuration',lambda *a:policy)
    original_pinned=p.rt.pinned
    monkeypatch.setattr(p.rt,'pinned',lambda path,pin,*a:policy if str(path).endswith(p.original.owl.CONFIG)else original_pinned(path,pin,*a))
    monkeypatch.setattr(p.original,'expected_asset_files',lambda _:({},str(assets)))
    monkeypatch.setattr(p.original.owl,'installed',lambda _:deepcopy(runtime))
    monkeypatch.setattr(p.original,'installed_grounding',lambda _:dict(source='authored RECORD'))
    monkeypatch.setenv('WR_IMAGE_ID',p.IMAGE);monkeypatch.setenv('WR_ENDPOINT_STARTED',str(time.monotonic()))
    real_iter=Path.iterdir
    monkeypatch.setattr(Path,'iterdir',lambda path:iter([Path('lo')])if str(path)=='/sys/class/net'else real_iter(path))
    return context,inputs,code,proof,proof_pin,before,runtime


def test_fixed_native_allowlist_host_import_boundary_and_original_pins():
    root=Path(p.__file__).resolve().parents[1]
    assert p.BUDGET==600 and p.seam.PROFILE['images']==48 and p.IMAGE==p.original.IMAGE
    assert set(p.original.NATIVE_FILES)<=set(p.NATIVE_FILES)
    assert all(not any(s in n for s in('acquisition','context','role_reference','sealed_callback'))for n in p.NATIVE_FILES)
    assert all(p.rt.identity(root/n,readonly=False)==pin for n,pin in p.seam.REUSE.items())
    tree=ast.parse(Path(p.__file__).read_text())
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('vcoco_fit_cal_acquisition_inputs'in ast.unparse(n)or'sealed_callback'in ast.unparse(n)for n in imports)
    assert not any(isinstance(n,ast.Assign)and any(isinstance(t,ast.Attribute)for t in n.targets)for n in ast.walk(tree))


def test_real_stdlib_import_with_private_and_numeric_modules_denied():
    root=Path(p.__file__).resolve().parents[1]
    script="""import importlib.abc,runpy,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','torch','transformers','PIL','vcoco_fit_cal_acquisition_inputs','vcoco_fit_cal_acquire','vcoco_fit_cal_context','sealed_callback_publication'}:
   raise ImportError('Forbidden native import')
sys.meta_path.insert(0,Deny())
runpy.run_path(sys.argv[1],run_name='authored_native_import')
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',script,str(root/'infra/vcoco_fit_cal_endpoint_run.py')],capture_output=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('argv',[['--proof-bytes','1'],['--native'],['--native','--proof-bytes','0','--proof-sha256','a'*64],
    ['--native','--proof-bytes','1','--proof-sha256','bad'],['--images','48'],['--input-directory','/tmp/private'],
    ['--native','--native','--proof-bytes','1','--proof-sha256','a'*64]])
def test_no_private_path_profile_count_or_unpinned_native_cli(argv):
    with pytest.raises((ValueError,SystemExit)):p.arguments(argv)


def test_host_and_native_cli_are_exact():
    assert not p.arguments([]).native
    assert p.arguments(['--native','--proof-bytes','1','--proof-sha256','a'*64]).native


def test_host_projects_only_six_public_keys_with_actual_helper_union(tmp_path,monkeypatch):
    import vcoco_fit_cal_acquisition_inputs as inputs_helper
    context,inputs,input_pin=authored.fixture(tmp_path);monkeypatch.setattr(p,'CONTEXT',context);monkeypatch.setattr(p,'DATA',context.inputs)
    code=tmp_path/'code';seen=[]
    monkeypatch.setattr(inputs_helper,'__file__',str(code/'infra/vcoco_fit_cal_acquisition_inputs.py'))
    def authenticate(*args):
        seen.append(args)
        return [dict(r,path=str(context.inputs/r['file']))for r in inputs['images']],{'public_manifest_identity':input_pin}
    monkeypatch.setattr(inputs_helper,'authenticate_actual_acquisition',authenticate)
    value,proof=p.host_inputs(code,'a'*40,lambda:None)
    assert value==inputs and all(set(r)==p.seam.public.KEYS for r in value['images'])
    assert set(inputs_helper.acquisition.HELPERS)<=set(seen[0][3])and seen[0][:3]==(code,'a'*40,p.ENTRY)
    assert proof=={'public_manifest_identity':input_pin}


def test_actual_native_all48_two_loads_lossless_outputs_and_release(tmp_path,monkeypatch):
    context,inputs,code,proof,pin,_,_=configured(tmp_path,monkeypatch);events=[];released=[]
    models=authored.fake_model(monkeypatch,events)
    models[2].tensor_ops.cuda.empty_cache=lambda:released.append('release')
    loads=[]
    def load(_):loads.append(1);return models
    monkeypatch.setattr(p.original,'load_models',load)
    report=p.native(code,'a'*40,pin,time.monotonic()+30)
    assert report['status']=='pass'and events==list(range(48))and len(loads)==1 and released==['release']
    assert report['model_loads']==2 and report['person_forward_calls']==report['box_calls']==48
    p.validate_native(report,proof,'a'*40,pin,inputs)
    assert report['images'][1]['person_retained_rows']==0
    assert (context.output/'native.json').stat().st_mode&0o777==0o400
    assert p.rt.strict((context.output/'native.json').read_bytes())==report


@pytest.mark.parametrize('failure',['before_model','forward','input_post','runtime_post'])
def test_native_failure_no_retry_release_and_source_rehash(tmp_path,monkeypatch,failure):
    context,inputs,code,proof,pin,_,runtime=configured(tmp_path,monkeypatch);events=[];released=[];loads=[]
    models=authored.fake_model(monkeypatch,events,fail=2 if failure=='forward'else None)
    models[2].tensor_ops.cuda.empty_cache=lambda:released.append(1)
    def load(_):
        loads.append(1)
        if failure=='input_post':
            path=context.inputs/inputs['images'][-1]['file'];path.chmod(0o600);path.write_bytes(b'changed');path.chmod(0o400)
        return models
    monkeypatch.setattr(p.original,'load_models',load)
    if failure=='before_model':monkeypatch.setattr(p.original.owl,'installed',lambda _:dict(wrong=True))
    if failure=='runtime_post':
        calls=[]
        def installed(_):calls.append(1);return runtime if len(calls)==1 else dict(wrong=True)
        monkeypatch.setattr(p.original.owl,'installed',installed)
    report=p.native(code,'a'*40,pin,time.monotonic()+30)
    assert report['status']=='fail'and len(loads)==(0 if failure=='before_model'else 1)
    assert released==([]if failure=='before_model'else[1])
    if failure=='forward':assert events==[0,1,2]and len(report['images'])==2
    assert (context.output/'native.json').exists()
    assert 'Secret'not in p.original.encode(report).decode()


@pytest.mark.parametrize('failure',['inference','surviving_cid','foreign_cid'])
def test_cleanup_checks_exact_cid_and_name_without_foreign_deletion(tmp_path,monkeypatch,failure):
    out=tmp_path/'out';out.mkdir(mode=0o700);monkeypatch.setattr(p,'OUTPUT',out);cid='c'*64;seal(out/'.container.cid',(cid+'\n').encode())
    removed=[];name='authored';revision='a'*40
    def command(args,_):
        if args[1]=='inspect':return ('foreign'if failure=='foreign_cid'else p.IMAGE)+'|/'+name+'|'+p.ENTRY+'|'+revision
        if args[1]=='rm':removed.append(args[-1]);return cid
        if args[-1]=='id='+cid:return cid if not removed or failure=='surviving_cid'else ''
        return ''
    monkeypatch.setattr(p,'command',command)
    if failure=='inference':p.cleanup(name,revision,time.monotonic()+10);assert removed==[cid]
    else:
        with pytest.raises(ValueError):p.cleanup(name,revision,time.monotonic()+10)
        assert removed==([]if failure=='foreign_cid'else[cid])


@pytest.mark.parametrize('fault',['partial','wrongslot','shape','count','flag','savedbytes'])
def test_host_complete_native_gate_rejects_partial_or_forged_receipt(tmp_path,monkeypatch,fault):
    context,inputs,code,proof,pin,_,_=configured(tmp_path,monkeypatch);models=authored.fake_model(monkeypatch,[])
    monkeypatch.setattr(p.original,'load_models',lambda _:models);report=p.native(code,'a'*40,pin,time.monotonic()+30)
    if fault=='partial':report['images'].pop()
    elif fault=='wrongslot':report['images'][1]['original_slot']=0
    elif fault=='shape':report['images'][0]['arrays']['person_model_logits']['shape'][1]=899
    elif fault=='count':report['model_loads']=True
    elif fault=='flag':report['actor_selection_performed']=True
    else:
        path=context.output/'image_000000.npz';path.chmod(0o600);path.write_bytes(b'changed');path.chmod(0o400)
    with pytest.raises(ValueError):p.validate_native(report,proof,'a'*40,pin,inputs)


def test_output_inventory_blocks_extra_or_writable_saved_bank(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir();monkeypatch.setattr(p,'OUTPUT',out);seal(out/'proof.json',b'proof')
    cid=out/'.container.cid';cid.write_bytes(b'c'*64);cid.chmod(0o600)
    assert set(p.output_identities())=={'proof.json','.container.cid'}
    seal(out/'foreign',b'foreign')
    with pytest.raises(ValueError):p.output_identities()


def test_wrapper_fixed_host_no_optional_profile_or_environment_inheritance():
    text=Path(p.__file__).with_name('run_vcoco_fit_cal_endpoint_observations.sh').read_text()
    assert 'world-reward-ncc-h100-02'in text and '615'in text and 'env -i'in text and '-I -B'in text
    assert 'umask 077'in text and '$# == 0'in text


@pytest.mark.parametrize('failure',[None,'native','source_post','cleanup','publication'])
def test_real_host_injected_native_deadline_mounts_owned_sealing_and_failure(tmp_path,monkeypatch,failure):
    import sealed_callback_publication as publisher
    context,inputs,code,_,_,before,runtime=configured(tmp_path,monkeypatch)
    (context.output/'proof.json').unlink();context.output.rmdir()
    root=tmp_path/'hostroot';jobs=root/'jobs';jobs.mkdir(parents=True)
    lock=jobs/'.world-reward-h100.lock';lock.write_bytes(b'lock');monkeypatch.setattr(p,'ROOT',root)
    binding={'producer_revision':'a'*40,'helpers':{n:r['pin']for n,r in before.items()},'markers':{}}
    for n in ('revision','source-sha256'):seal(code.parent/n,b'authored marker')
    for n in p.NATIVE_FILES:seal(code/n,b'authored code')
    state={'binding':binding,'states':{'allcode':'authored'}};calls=[]
    def source(*_):
        calls.append(1)
        return state if len(calls)==1 or failure!='source_post'else {'binding':binding,'states':{'allcode':'mutated'}}
    monkeypatch.setattr(p,'source',source)
    acquisition={'public_manifest_identity':p.rt.identity(context.inputs/'manifest.json'),'input_records':48}
    monkeypatch.setattr(p,'host_inputs',lambda *args:(inputs,deepcopy(acquisition)))
    prior={'owl':{'native_runtime':runtime}}
    monkeypatch.setattr(p.original,'qualifications',lambda *a,**k:deepcopy(prior))
    asset=tmp_path/'asset';assets={str(asset):p.rt.identity(asset)}
    monkeypatch.setattr(p.original,'asset_files',lambda *a:assets)
    monkeypatch.setattr(p,'command',lambda args,_:p.IMAGE if args[:3]==['docker','image','inspect']else '')
    models=authored.fake_model(monkeypatch,[],fail=2 if failure=='native'else None)
    models[2].tensor_ops.cuda.empty_cache=lambda:None
    monkeypatch.setattr(p.original,'load_models',lambda _:models)
    commands=[]
    def docker(args,**kw):
        commands.append((args,kw));(context.output/'.container.cid').write_bytes(b'c'*64)
        pin=p.rt.identity(context.output/'proof.json')
        value=p.native(code,'a'*40,pin,time.monotonic()+30)
        return SimpleNamespace(returncode=0 if value['status']=='pass'else 1)
    monkeypatch.setattr(p.subprocess,'run',docker)
    if failure=='cleanup':monkeypatch.setattr(p,'cleanup',lambda *a:(_ for _ in()).throw(ValueError('SECRET_SENTINEL')))
    if failure=='publication':
        def interrupted(_):raise TimeoutError('SECRET_SENTINEL')
        monkeypatch.setattr(p,'sync',interrupted)
    started=time.monotonic();report=p.host(code,'a'*40,started,started+60)
    assert report['status']==('pass'if failure is None else 'fail')
    assert len(commands)==1 and len(calls)==2 and report['outputs_sealed']==(failure not in('publication','cleanup'))
    assert 'SECRET_SENTINEL'not in p.original.encode(report).decode()
    assert p.rt.strict((context.output/'report.json').read_bytes())==report
    assert (context.output/'report.json').stat().st_mode&0o777==0o400
    args,kw=commands[0];mounts=[args[i+1]for i,v in enumerate(args)if v=='--mount']
    assert '--gpus'in args and '64g'in args and '4'in args and '--network'in args and 'none'in args
    assert kw['env']['DOCKER_HOST']=='unix://'+str(root/'docker.sock')
    assert kw['timeout']<=40 and args.count('--mount')==len(p.NATIVE_FILES)+2+len(assets)+49+1
    assert not any(s in value for value in mounts for s in ('acquisition/','role_reference','context.py','acquisition_inputs.py'))
    if failure is None:
        assert len(report['native_images'])==48 and report['source_inputs_assets_runtime_rehashed_after']is True
        assert (context.output/'.container.cid').stat().st_mode&0o777==0o400
    if failure=='native':assert report['native_summary']['status']=='fail'and report['native_summary']['phase']=='all48_original_forward'
