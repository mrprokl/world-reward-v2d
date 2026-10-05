"""Tiny source/lifecycle fixtures only; never real model, Docker or GPU."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

import owlv2_native_qualify as p


def config():return json.loads((Path(__file__).resolve().parents[1]/p.CONFIG).read_bytes())


def seal(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o400)
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def test_procedural_rgb_three_shapes_deterministic_no_reference_scene():
    a=p.procedural_rgb();b=p.procedural_rgb()
    assert [r.shape for r in a]==[(48,80,3),(80,48,3),(64,64,3)]
    assert all(r.dtype==np.uint8 and np.array_equal(r,s) for r,s in zip(a,b))
    assert len({hashlib.sha256(r.tobytes()).hexdigest() for r in a})==3
    a[0][:]=0;assert b[0].any()


def test_exact_known_runtime_and_adapter_and_no_placeholder_acquisition():
    cfg=config();root=Path(__file__).resolve().parents[1]
    raw=(root/'src/world_reward/owlv2_object_observations.py').read_bytes()
    assert cfg['adapter_identity']==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert cfg['image_id']==p.IMAGE and cfg['budget_seconds']==120 and cfg['memory_bytes']==64<<30
    assert cfg['versions']['transformers']=='4.53.3' and cfg['versions']['torch']=='2.5.1+cu124'
    assert cfg['model_loads']==1 and cfg['image_embed_calls']==3 and cfg['patches_per_frame']==3600
    assert 'mandatory_independent_actual_CLI' in cfg['acquisition_receipt']
    assert set(p.NATIVE_FILES)=={'infra/owlv2_native_qualify.py','infra/mediapipe_cpu_runtime_verify.py',p.CONFIG,
        'src/world_reward/__init__.py','src/world_reward/owlv2_object_observations.py'}
    assert not any('hocap' in n or 'sam2' in n or 'acquire' in n for n in p.NATIVE_FILES)
    assert str(p.ASSETS).endswith('/owlv2_objectness_v2') and str(p.ACQUIRE_REPORT).endswith('/owlv2-objectness-assets-v2.json')
    assert p.ACQUIRE_HELPERS[2]=='configs/owlv2_assets_v2.json'


def test_strict_all_heads_and_original_integer_buffers_without_cast_or_ignored_keys():
    def tensor(dtype='float32',shape=(2,),floating=True):
        return SimpleNamespace(dtype=dtype,shape=shape,is_floating_point=lambda:floating)
    expected={'vision':tensor(),'unused_text':tensor(),'position':tensor('int64',floating=False)}
    calls=[]
    model=SimpleNamespace(state_dict=lambda:expected,load_state_dict=lambda state,strict:
        (calls.append((state,strict)) or SimpleNamespace(missing_keys=[],unexpected_keys=[])))
    state=dict(expected);p.strict_state(model,state,SimpleNamespace(float32='float32'))
    assert len(calls)==1 and calls[0][0] is state and calls[0][1] is True
    for modified in (dict(state,vision=tensor('float16')),dict(state,vision=tensor(shape=(3,))),
                     {k:v for k,v in state.items() if k!='unused_text'},dict(state,extra=tensor())):
        with pytest.raises(ValueError):p.strict_state(model,modified,SimpleNamespace(float32='float32'))
    assert len(calls)==1


def acquisition_fixture(tmp_path,monkeypatch):
    cfg=config();assets=tmp_path/'assets';assets.mkdir();expected={}
    for name in cfg['assets']:expected[name]=seal(assets/name,('original '+name).encode())
    cfg['assets']=expected;source={'helpers':{p.ACQUIRE_HELPERS[2]:dict(bytes=1,sha256='a'*64)}}
    receipt=dict(schema='world_reward.owlv2_assets_receipt.v1',status='pass',producer_revision='b'*40,
        source_binding=source,configuration_identity=source['helpers'][p.ACQUIRE_HELPERS[2]],
        source_rehashed_after=True,artifacts_rehashed_after=True,owned_partials_removed=True,outputs_sealed=True,
        checkpoint_decoded=False,models_loaded=False,dataset_read=False,gpu_used=False,publisher_revision=cfg['publisher_revision'],
        primary_model_license_recorded=True,final_asset_pins=expected)
    report=tmp_path/'report.json';rp=seal(report,p.encode(receipt));native_cfg={'assets':[dict(file=k,**v) for k,v in expected.items()]}
    monkeypatch.setattr(p,'ASSETS',assets);monkeypatch.setattr(p,'ACQUIRE_REPORT',report);monkeypatch.setattr(p.rt,'source',lambda *a:source)
    original=p.rt.pinned
    monkeypatch.setattr(p.rt,'pinned',lambda path,pin,maximum=1<<20: native_cfg if Path(path).name==p.ACQUIRE_HELPERS[2].split('/')[-1] else original(path,pin,maximum))
    return cfg,receipt,report,rp


def test_actual_acquisition_proof_required_before_model(tmp_path,monkeypatch):
    cfg,receipt,report,rp=acquisition_fixture(tmp_path,monkeypatch)
    proof=p.acquisition('b'*40,rp,cfg)
    assert proof['assets']==cfg['assets'] and proof['report_identity']==rp


@pytest.mark.parametrize('failure',['status','source','decode','gpu','posthash','publisher','pin','asset','extra'])
def test_acquisition_mismatch_is_fail_not_model_load(tmp_path,monkeypatch,failure):
    cfg,receipt,report,rp=acquisition_fixture(tmp_path,monkeypatch)
    if failure=='asset':
        f=p.ASSETS/'config.json';f.chmod(0o600);f.write_bytes(b'changed');f.chmod(0o400)
    elif failure=='extra':(p.ASSETS/'extra').write_bytes(b'foreign')
    elif failure=='pin':rp=dict(rp,sha256='0'*64)
    else:
        if failure=='status':receipt['status']='fail'
        elif failure=='source':receipt['source_binding']={}
        elif failure=='decode':receipt['checkpoint_decoded']=True
        elif failure=='gpu':receipt['gpu_used']=True
        elif failure=='posthash':receipt['artifacts_rehashed_after']=False
        else:receipt['publisher_revision']='0'*40
        report.chmod(0o600);report.write_bytes(p.encode(receipt));report.chmod(0o400);rp=dict(bytes=report.stat().st_size,sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    with pytest.raises(ValueError):p.acquisition('b'*40,rp,cfg)


def test_owned_cid_cleanup_no_foreign_removal(tmp_path,monkeypatch):
    cid='c'*64;path=tmp_path/'cid';path.write_text(cid);commands=[];revision='a'*40;name='owned'
    def command(args,deadline):
        commands.append(args)
        if args[:2]==['docker','inspect']:return p.IMAGE+'|/'+name+'|'+p.ENTRY+'|'+revision
        if args[:3]==['docker','ps','-aq']:return cid if 'id='+cid in args else ''
        return ''
    monkeypatch.setattr(p,'command',command);p.cleanup(path,name,revision,time.monotonic()+10)
    assert ['docker','rm','-f',cid] in commands and path.stat().st_mode&0o777==0o400
    commands.clear()
    monkeypatch.setattr(p,'command',lambda args,deadline: 'foreign' if args[1]=='inspect' else cid)
    with pytest.raises(ValueError):p.cleanup(path,name,revision,time.monotonic()+10)


def test_closed_runtime_original_record_semantics_in_source():
    source=Path(p.__file__).read_text()
    assert "load_file(str(ASSETS/'model.safetensors'),device='cpu')" in source
    assert 'load_state_dict(state,strict=True)' in source and 'from_pretrained' not in source
    assert 'Owlv2Config.from_json_file' in source and 'Owlv2ImageProcessor(**rt.strict' in source
    assert "entries[0].hash.mode=='sha256'" in source and "entries[0].size==wanted['bytes']" in source
    assert 'infer_owlv2_object_frame(model,rgb,i,operations)' in source
    assert 'torch.autocast(' not in source and 'torch.backends.cuda.matmul.allow_tf32=False' in source
    assert 'str(exc)' not in source and 'traceback' not in source


def dispatch_fixture(tmp_path,monkeypatch,*,native_fail=False):
    root=tmp_path/'root';code=root/'code';code.mkdir(parents=True);out=root/'results'/'out';out.parent.mkdir();assets=root/'assets';assets.mkdir()
    (root/'jobs').mkdir();(root/'jobs/.world-reward-h100.lock').write_bytes(b'lock')
    cfg=config();cfg['assets']={};source={'helpers':{},'markers':{},'producer_revision':'a'*40};commands=[];native_files={}
    for name in p.NATIVE_FILES:
        raw=(b'{"config":true}' if name==p.CONFIG else b'#code');source['helpers'][name]=seal(code/name,raw);native_files[name]=source['helpers'][name]
    for name in ('revision','source-sha256'):source['markers'][name]=seal(code.parent/name,b'originalmarker')
    monkeypatch.setattr(p,'ROOT',root);monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setattr(p,'ASSETS',assets)
    monkeypatch.setattr(p.rt,'source',lambda *a:source);monkeypatch.setattr(p,'configuration',lambda *a:cfg)
    monkeypatch.setattr(p,'runtime_proof',lambda *a,**k:{'runtime':'original'});monkeypatch.setattr(p,'acquisition',lambda *a:{'assets':'proof'})
    monkeypatch.setattr(p,'command',lambda args,d:(commands.append(args) or ''))
    monkeypatch.setattr(p,'cleanup',lambda *a:commands.append(['cleanup']))
    def run(cmd,**kwargs):
        commands.append(cmd);(out/'.container.cid').write_text('c'*64)
        proof=json.loads((out/'proof.json').read_bytes());proof_pin=dict(bytes=(out/'proof.json').stat().st_size,sha256=hashlib.sha256((out/'proof.json').read_bytes()).hexdigest())
        rows=[]
        for i in range(3):rows.append(dict(original_frame_index=i,file=f'procedural_{i:02d}.npz',patches=3600,identity=seal(out/f'procedural_{i:02d}.npz',b'bank'+bytes([i]))))
        report=dict(schema=cfg['schema'],stage='native_owlv2_procedural',producer_revision='a'*40,image_id=p.IMAGE,
            status='fail' if native_fail else 'pass',phase='complete',source_runtime_assets_rehashed_after=True,proof_identity=proof_pin,
            model_loads=1,image_embed_calls=3,objectness_calls=3,box_calls=3,frames=rows)
        seal(out/'native.json',p.encode(report));return SimpleNamespace(returncode=1 if native_fail else 0)
    monkeypatch.setattr(p.subprocess,'run',run)
    return code,cfg,source,out,commands


def test_host_individual_readonly_mounts_one_gpu_and_sealed_receipt(tmp_path,monkeypatch):
    code,cfg,source,out,commands=dispatch_fixture(tmp_path,monkeypatch)
    result=p.dispatch(code,'a'*40,cfg,'b'*40,dict(bytes=1,sha256='c'*64))
    assert result['status']=='pass' and result['owned_cleanup_verified'] and result['source_runtime_assets_rehashed_after']
    cmd=next(c for c in commands if c[:2]==['docker','run'])
    assert cmd[cmd.index('--gpus')+1]=='device=0' and cmd[cmd.index('--memory')+1]=='64g' and '--network' in cmd
    readonly=[cmd[i+1] for i,v in enumerate(cmd) if v=='--mount' and cmd[i+1].endswith(',readonly')]
    assert len(readonly)==len(p.NATIVE_FILES)+2 and not any(f'src={code},' in x for x in readonly)
    assert '-I' in cmd and '-B' in cmd and 'HF_HUB_OFFLINE=1' in cmd
    assert out.stat().st_mode&0o777==0o500 and all(f.stat().st_mode&0o777==0o400 for f in out.iterdir())
    assert 'acquire' not in ','.join(readonly)


def test_host_native_fail_retained_not_qualified_or_suppressed(tmp_path,monkeypatch):
    code,cfg,source,out,commands=dispatch_fixture(tmp_path,monkeypatch,native_fail=True)
    result=p.dispatch(code,'a'*40,cfg,'b'*40,dict(bytes=1,sha256='c'*64))
    assert result['status']=='fail' and result['owned_cleanup_verified'] and result['outputs_sealed']
    assert (out/'native.json').exists() and result['native_exit_status']==1 and result['error_type']=='ValueError'


@pytest.mark.parametrize('field',['producer_revision','image_id','stage','schema','phase','proof'])
def test_host_rejects_foreign_or_mutated_native_proof(tmp_path,monkeypatch,field):
    code,cfg,source,out,commands=dispatch_fixture(tmp_path,monkeypatch)
    original=p.subprocess.run
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        if field=='proof':
            file=out/'proof.json';file.chmod(0o600);file.write_bytes(b'{"changed":true}');file.chmod(0o400)
        else:
            file=out/'native.json';value=json.loads(file.read_bytes());value[field]='foreign'
            file.chmod(0o600);file.write_bytes(p.encode(value));file.chmod(0o400)
        return result
    monkeypatch.setattr(p.subprocess,'run',changed)
    result=p.dispatch(code,'a'*40,cfg,'b'*40,dict(bytes=1,sha256='c'*64))
    assert result['status']=='fail' and result['owned_cleanup_verified'] and result['outputs_sealed']


def test_budget_seal_cannot_turn_late_native_success_into_pass(tmp_path,monkeypatch):
    code,cfg,source,out,commands=dispatch_fixture(tmp_path,monkeypatch);clock=[0.]
    monkeypatch.setattr(p.time,'monotonic',lambda:clock[0]);original=p.subprocess.run
    def delayed(*args,**kwargs):
        result=original(*args,**kwargs);clock[0]=121.;return result
    monkeypatch.setattr(p.subprocess,'run',delayed)
    result=p.dispatch(code,'a'*40,cfg,'b'*40,dict(bytes=1,sha256='c'*64))
    assert result['status']=='fail' and result['post_error_type']=='TimeoutError' and result['outputs_sealed']
    assert json.loads((out/'host.json').read_bytes())['status']=='fail'


def test_deadline_and_sanitized_unknown_exception():
    with pytest.raises(TimeoutError):p.check(time.monotonic()-1)
    class SecretError(Exception):pass
    assert p.error(SecretError('secret'))=='other' and p.error(RuntimeError('secret'))=='RuntimeError'


def test_native_authentication_failure_retained_and_sanitized_before_model(tmp_path,monkeypatch):
    code=tmp_path/'code';code.mkdir();out=tmp_path/'out';out.mkdir();cfg=config()
    pin=seal(out/'proof.json',p.encode(dict(source={},native_files={},assets=cfg['assets'],image_id=p.IMAGE)))
    monkeypatch.setenv('WR_IMAGE_ID','foreign')
    original=p.rt.require
    def rejected(condition,message):
        if not condition:raise RuntimeError('SECRET_TOKEN=sensitive signedURL env payload')
        original(condition,message)
    monkeypatch.setattr(p.rt,'require',rejected)
    result=p.native(code,'a'*40,out,cfg,pin,time.monotonic()+10)
    assert result['status']=='fail' and result['phase']=='authentication' and result['model_loads']==0
    assert result['error_type']=='RuntimeError' and 'SECRET_TOKEN' not in (out/'native.json').read_text()
    assert set(x.name for x in out.iterdir())=={'proof.json','native.json'}


def test_script_syntax_and_frozen_host_env():
    root=Path(__file__).resolve().parents[1];compile((root/p.NATIVE_FILES[0]).read_bytes(),p.NATIVE_FILES[0],'exec')
    wrapper=(root/'infra/run_owlv2_native_qualify.sh').read_text()
    assert 'env -i' in wrapper and '130s' in wrapper and '-I -B' in wrapper
    assert '--acquisition-report-bytes' in wrapper and '--acquisition-report-sha256' in wrapper
