"""Pure mocked provenance/lifecycle and optional tiny CPU tensors; no GPU/model."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location('wr_masa_native_test',REPO/'infra/masa_native_qualify.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def load_rt():
    spec=importlib.util.spec_from_file_location('wr_masa_native_rt_test',REPO/'infra/mediapipe_cpu_runtime_verify.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def seal(code):
    for p in sorted(code.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    code.chmod(0o555)


def source(rt,root,revision,entry,helpers):
    code=root/'jobs'/revision/entry/'code';code.mkdir(parents=True)
    for n in helpers:
        p=code/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/n).read_bytes())
    (code/'empty.py').write_bytes(b'')
    for n,s in(('revision',revision+'\n'),('source-sha256','f'*64+'\n')):
        (code.parent/n).write_text(s);(code.parent/n).chmod(0o444)
    seal(code);return code,rt.source(root,code,revision,entry,helpers)


def setup(gate,tmp_path,monkeypatch):
    rt=load_rt();root=tmp_path/'root';(root/'results').mkdir(parents=True);(root/'jobs').mkdir()
    data=tmp_path/'data';data.mkdir();revision='a'*40;rv='b'*40;av='c'*40
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'DATA',data)
    code,own=source(rt,root,revision,gate.ENTRY,gate.HELPERS)
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    cfg=json.loads((code/gate.CONFIG).read_bytes())
    acq_files=('infra/masa_acquire.py','configs/masa_acquisition_v1.json')
    old,acq_source=source(rt,root,av,'run_masa_acquire',acq_files)
    upstream=json.loads((old/'configs/masa_acquisition_v1.json').read_bytes());assets=[]
    for row in upstream['assets']:
        raw=('TINY_'+row['file']).encode();row.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        p=data/row['file'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
        assets.append(dict(row))
    p=old/'configs/masa_acquisition_v1.json';p.chmod(0o644);p.write_text(json.dumps(upstream));p.chmod(0o444)
    acq_source=rt.source(root,old,av,'run_masa_acquire',acq_files);seal(data)
    acq=dict(schema='world_reward.masa_acquisition_receipt.v1',stage='masa_native_r50_source_weight_acquisition',status='pass',phase='complete',
      producer_revision=av,models_loaded=False,weights_decoded=False,packages_installed=False,rgb_or_datasets_used=False,
      ground_truth_used=False,gpu_used=False,challenge_inputs_used=False,oracle_modes=[],quality_evaluated=False,adopted=False,
      source_rehashed_after=True,artifacts_rehashed_after=True,public_sealed=True,owned_partials_removed=True,
      source_binding=acq_source,source_scope=upstream['source_scope'],assets=assets)
    cfg['acquisition']['producer_revision']=av
    cfg['acquisition']['source_helper']=rt.identity(old/'infra/masa_acquire.py')
    cfg['acquisition']['config']=rt.identity(old/'configs/masa_acquisition_v1.json');acq['config_identity']=cfg['acquisition']['config']
    ap=root/'results/masa-acquisition-v1.json';rt.write(ap,json.dumps(acq).encode(),0o444);cfg['acquisition']['report']=rt.identity(ap)
    rold,rsource=source(rt,root,rv,'run_masa_runtime_build',('infra/masa_runtime_build.py','configs/masa_runtime_v1.json'))
    rc=dict(wheels=[dict(name='torch',version='2.1.2+cu118',filename='torch.whl',bytes=4,sha256=hashlib.sha256(b'TINY').hexdigest())],
            publisher_notices=[dict(file='LICENSE',bytes=4,sha256=hashlib.sha256(b'MIT!').hexdigest())])
    p=rold/'configs/masa_runtime_v1.json';p.chmod(0o644);p.write_text(json.dumps(rc));p.chmod(0o444)
    rsource=rt.source(root,rold,rv,'run_masa_runtime_build',('infra/masa_runtime_build.py','configs/masa_runtime_v1.json'))
    folder=root/'results'/('masa-runtime-build-'+rv);(folder/'wheels').mkdir(parents=True);rt.write(folder/'wheels/torch.whl',b'TINY',0o444)
    (folder/'notices').mkdir();rt.write(folder/'notices/LICENSE',b'MIT!',0o444)
    image=dict(image_id='sha256:'+'d'*64,layers=['sha256:'+'e'*64])
    runtime=dict(schema='world_reward.masa_runtime_build.v1',stage='masa_author_isolated_runtime_build',status='pass',phase='complete',producer_revision=rv,
      gpu_used=False,operator_qualified=False,model_constructed=False,model_or_dataset_read=False,quality_verified=False,adopted=False,
      license_eligibility_verified=False,system_site_packages=False,source_rehashed_after=True,artifacts_rehashed_after=True,
      base_rechecked_after=True,owned_containers_removed=True,owned_partial_cleanup=True,source_binding=rsource,child_image=image,
      target_tag='world-reward/masa-author-runtime:'+rv,config_identity=rt.identity(rold/'configs/masa_runtime_v1.json'),
      cpu_import=dict(versions={'torch':'2.1.2+cu118'},python='3.11',isolated_venv=True,cuda_initialized=False,cuda_build='11.8',
         extension_imported=True,operator_executed=False,model_constructed=False),wheels={'torch':dict(identity=rt.identity(folder/'wheels/torch.whl'))},
      publisher_notices={'LICENSE':dict(rt.identity(folder/'notices/LICENSE'),publisher_md5=None)})
    rt.write(folder/'report.json',json.dumps(runtime).encode(),0o444);pin=rt.identity(folder/'report.json')
    p=code/gate.CONFIG;p.chmod(0o644);p.write_text(json.dumps(cfg));p.chmod(0o444)
    monkeypatch.setattr(gate,'CONFIG_PIN',rt.identity(p));own=gate.current_source(rt,code,revision)
    args=SimpleNamespace(revision=revision,runtime_revision=rv,runtime_report_bytes=pin['bytes'],runtime_report_sha256=pin['sha256'],image_id=image['image_id'])
    lock=root/'jobs/.world-reward-h100.lock';lock.write_bytes(b'');lock.chmod(0o444)
    return rt,root,code,args,cfg


def test_frozen_source_pipeline_no_oracle_model_settings(gate):
    raw=(REPO/gate.CONFIG).read_bytes();c=json.loads(raw)
    assert gate.CONFIG_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert c['budget_seconds']==600 and len(c['source_execution']['registry_leaves'])==4
    assert c['source_execution']['given_dets']and not c['source_execution']['load_public_dets']
    assert c['procedural_model_control']['boxes_xyxy'][0]==c['procedural_model_control']['boxes_xyxy'][2]
    assert c['checkpoint_policy']['strict_state_load']and not c['checkpoint_policy']['state_key_rewrite']


def test_actual_authentication_complete_historical_source_and_assets(gate,tmp_path,monkeypatch):
    rt,root,code,args,c=setup(gate,tmp_path,monkeypatch);proof=gate.authenticate(rt,code,args)
    assert len(proof['old_source_parents'])==2 and len(proof['files'])==22
    assert proof['image_id']==args.image_id and proof['source_binding']==gate.current_source(rt,code,args.revision)
    runtime_path=root/'results'/('masa-runtime-build-'+args.runtime_revision)/'report.json'
    assert proof['runtime_report']==dict(bytes=args.runtime_report_bytes,sha256=args.runtime_report_sha256)==proof['files'][str(runtime_path)]
    assert proof['runtime_report']!=proof['files'][str(runtime_path.parent/'notices/LICENSE')]


@pytest.mark.parametrize('fault',['missing_md5','unexpected_md5','extra_field'])
def test_authentication_requires_actual_notice_producer_schema(gate,tmp_path,monkeypatch,fault):
    rt,root,code,args,c=setup(gate,tmp_path,monkeypatch)
    p=root/'results'/('masa-runtime-build-'+args.runtime_revision)/'report.json';r=json.loads(p.read_bytes())
    notice=r['publisher_notices']['LICENSE']
    if fault=='missing_md5':del notice['publisher_md5']
    elif fault=='unexpected_md5':notice['publisher_md5']='f'*32
    else:notice['extra_field']=True
    p.chmod(0o644);p.write_text(json.dumps(r));p.chmod(0o444);pin=rt.identity(p)
    args.runtime_report_bytes=pin['bytes'];args.runtime_report_sha256=pin['sha256']
    with pytest.raises(ValueError):gate.authenticate(rt,code,args)


@pytest.mark.parametrize('fault',['runtime_fail','source_changed','weight_changed','wheel_changed','notice_changed','image_changed','report_unpinned'])
def test_authentication_rejects_before_model_or_gpu(gate,tmp_path,monkeypatch,fault):
    rt,root,code,args,c=setup(gate,tmp_path,monkeypatch)
    folder=root/'results'/('masa-runtime-build-'+args.runtime_revision)
    if fault in('runtime_fail','image_changed'):
        p=folder/'report.json';r=json.loads(p.read_bytes())
        if fault=='runtime_fail':r['status']='fail'
        else:r['child_image']['image_id']='sha256:'+'f'*64
        p.chmod(0o644);p.write_text(json.dumps(r));p.chmod(0o444);pin=rt.identity(p)
        args.runtime_report_bytes=pin['bytes'];args.runtime_report_sha256=pin['sha256']
    if fault=='source_changed':
        p=root/'jobs'/args.runtime_revision/'run_masa_runtime_build/code/empty.py';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    if fault=='weight_changed':
        p=gate.DATA/'weights/masa_r50.pth';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    if fault=='wheel_changed':
        p=folder/'wheels/torch.whl';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    if fault=='notice_changed':
        p=folder/'notices/LICENSE';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    if fault=='report_unpinned':args.runtime_report_sha256='f'*64
    with pytest.raises(ValueError):gate.authenticate(rt,code,args)


def test_strict_state_layout_no_prefix_or_partial_fallback(gate):
    class Tensor:
        def __init__(self,shape=(2,),dtype='float32'):self.shape=shape;self.dtype=dtype
    class T:
        is_tensor=staticmethod(lambda v:isinstance(v,Tensor))
        isfinite=staticmethod(lambda v:SimpleNamespace(all=lambda:True))
    expected={'layer.weight':Tensor()};state={'layer.weight':Tensor()}
    actual,proof=gate.strict_state(T,{'state_dict':state,'meta':{}},expected)
    assert actual is state and proof['strict']and not proof['prefix_rewrite']
    for bad in({'module.layer.weight':Tensor()},{'state_dict':state,'ema_state_dict':state},{'state_dict':{}},
                {'state_dict':{'layer.weight':Tensor((3,))}}, {'state_dict':{'layer.weight':Tensor(dtype='float64')}}):
        with pytest.raises(ValueError):gate.strict_state(T,bad,expected)


def test_cli_requires_all_independent_runtime_pins_and_rejects_duplicates(gate):
    good=['--revision','a'*40,'--runtime-revision','b'*40,'--runtime-report-bytes','2000',
          '--runtime-report-sha256','c'*64,'--image-id','sha256:'+'d'*64]
    assert gate.parse(good).runtime_revision=='b'*40
    with pytest.raises(ValueError):gate.parse(good+['--image-id','sha256:'+'e'*64])
    with pytest.raises(ValueError):gate.parse(good+['--image-id=sha256:'+'e'*64])
    with pytest.raises(ValueError):gate.parse(['--revision','a'*40])
    with pytest.raises(ValueError):gate.parse(good+['--out','/tmp/foreign'])


def test_operator_gate_is_ordered_before_checkpoint_and_uses_original_defaults(gate):
    tree=ast.parse((REPO/'infra/masa_native_qualify.py').read_bytes())
    native=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_native')
    segment=ast.get_source_segment((REPO/'infra/masa_native_qualify.py').read_text(),native)
    assert segment.index('operators(torch,c,report)')<segment.index('native_model(torch,c,deadline,report)')
    op=ast.get_source_segment((REPO/'infra/masa_native_qualify.py').read_text(),next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='operators'))
    assert 'RoIAlign(output_size=7,sampling_ratio=0)'in op
    constructor=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='RoIAlign')
    assert {k.arg for k in constructor.keywords}=={'output_size','sampling_ratio'}
    assert 'torch.equal(non,repeat)'in op and 'torch.cuda.synchronize()'in op


def test_native_model_static_contract_no_tracker_or_original_package_import(gate):
    raw=(REPO/'infra/masa_native_qualify.py').read_text();tree=ast.parse(raw)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native_model');s=ast.get_source_segment(raw,fn)
    assert 'Compose(cfg.inference_pipeline)'in s and 'training=False'in s and 'scale.repeat(2)'in s
    assert 'model.backbone.forward(inputs[:,0].contiguous())'in s
    assert s.count('model.track_head.predict(')==3 and '.tracker.track('not in s and 'model.test_step('not in s
    assert 'weights_only=True'in s and 'strict=True'in s and 'revert_sync_batchnorm(MODELS.build(cfg.model))'in s
    assert 'import masa'not in s and 'fix_upsample='not in s
    assert 'mean=torch.tensor([123.675,116.28,103.53]'in s
    assert 'std=torch.tensor([58.395,57.12,57.375]'in s
    assert 'padded_h=(original_resize.shape[0]+31)//32*32'in s and 'torch.equal(inputs[0,0],expected)'in s


def test_observability_is_ast_neutral_for_native_math_and_runtime(gate):
    tree=ast.parse((REPO/'infra/masa_native_qualify.py').read_bytes())
    pins={'operators':'d2d6fa2535808b189d04816fdb813c1e3159a6704dfa6a030313a1ec38c4dcd3',
          'native_model':'7114a80269be6e2a2d2ce2fc273e3d6d8d32bcd74e799cf27f6d92383f855ee1',
          'strict_state':'aaf5767b0bd4ab5bce8a1121075923f51bd56961f6fb2748a390eee6506ff97d',
          'authenticate':'587d4dfd92106154ebf6ceba37213015c567421e5c624859a88d880271eb4bdd',
          'command':'27998ae76ac5265b4011a2bb6bb746152ea72d0e3c965e351ac1ff7f8e8e60d6',
          'image':'cd189a335a9d50ef5eb49fa58e764fb7332ba51613334faa45de9fa3690c88e9'}
    class RemoveDiagnostics(ast.NodeTransformer):
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript)
                and isinstance(node.targets[0].value,ast.Name)and node.targets[0].value.id=='progress'
                and isinstance(node.targets[0].slice,ast.Constant)and node.targets[0].slice.value in ('subgate','operator_subgate')):
                allowed=gate.OPERATOR_SUBGATES if node.targets[0].slice.value=='operator_subgate'else gate.FAILURE_GATES
                assert isinstance(node.value,ast.Constant)and node.value.value in allowed
                return None
            return node
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef)and fn.name in pins:
            actual=RemoveDiagnostics().visit(copy.deepcopy(fn))
            if fn.name=='operators':
                assert actual.args.args[-1].arg=='progress'
                actual.args.args.pop()
            assert hashlib.sha256(ast.dump(actual,include_attributes=False).encode()).hexdigest()==pins[fn.name]


def test_fixed_operator_subgates_precede_unchanged_native_calls(gate):
    raw=(REPO/'infra/masa_native_qualify.py').read_text();tree=ast.parse(raw)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='operators')
    markers=sorted((n.lineno,n.value.value)for n in ast.walk(fn)if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Subscript)and isinstance(n.targets[0].slice,ast.Constant)
        and n.targets[0].slice.value=='operator_subgate')
    assert [label for _,label in markers]==['imports','dcn_construct','dcn_zero_offset','dcn_nonzero',
                                           'roi_construct','roi_cuda','roi_cpu','roi_compare','complete']
    checks={'ModulatedDeformConv2d(2,3,3,padding=1,bias=True)':'dcn_construct',
            'RoIAlign(output_size=7,sampling_ratio=0)':'roi_construct',
            'roi(x,boxes.cuda())':'roi_cuda','roi(x.cpu(),boxes)':'roi_cpu',
            "torch.allclose(gpu.cpu(),cpu,**g['roi_align_cuda_vs_cpu'])":'roi_compare'}
    calls=[n for n in ast.walk(fn)if isinstance(n,ast.Call)]
    for expression,label in checks.items():
        call=next(n for n in calls if ast.get_source_segment(raw,n)==expression)
        assert [value for line,value in markers if line<call.lineno][-1]==label
    conv_calls=sorted((n for n in calls if ast.get_source_segment(raw,n)=='conv(x,offset,mask)'),key=lambda n:n.lineno)
    assert len(conv_calls)==3
    assert [[value for line,value in markers if line<call.lineno][-1]for call in conv_calls]==[
        'dcn_zero_offset','dcn_nonzero','dcn_nonzero']


@pytest.mark.parametrize('operator_subgate',sorted(('imports','dcn_construct','dcn_zero_offset','dcn_nonzero',
                                                 'roi_construct','roi_cuda','roi_cpu','roi_compare')))
def test_native_operator_failure_keeps_fixed_subgate_and_first_operators_gate(gate,tmp_path,monkeypatch,operator_subgate):
    rt,root,code,args,c=setup(gate,tmp_path,monkeypatch);out=tmp_path/'native';out.mkdir()
    proof=gate.authenticate(rt,code,args);rt.write(out/'proof.json',json.dumps(proof).encode(),0o444)
    torch=SimpleNamespace(__version__='2.1.2+cu118',cuda=SimpleNamespace(is_available=lambda:True,
      get_device_capability=lambda:(9,0),get_device_name=lambda:'H100'))
    monkeypatch.setitem(gate.sys.modules,'torch',torch);monkeypatch.setattr(gate.sys,'prefix',gate.VENV)
    def operators(torch,policy,progress):
        progress['operator_subgate']=operator_subgate
        raise RuntimeError('SECRET_CUDA_DIAGNOSTIC_PAYLOAD')
    def model(*a):raise AssertionError('Model must not run after operator failure')
    monkeypatch.setattr(gate,'operators',operators);monkeypatch.setattr(gate,'native_model',model)
    with pytest.raises(ValueError,match='Native contract failed'):
        gate.run_native(code,args.revision,out,rt.identity(out/'proof.json'),gate.time.monotonic()+600)
    report=json.loads((out/'native.json').read_bytes())
    assert report['failure_gate']=='operators'and report['failure_class']=='RuntimeError'
    assert report['operator_subgate']==operator_subgate and operator_subgate in gate.OPERATOR_SUBGATES
    assert not report['model_constructed']and not report['weights_decoded']and not report['model_loaded']
    assert report['source_artifacts_rehashed_after']and 'SECRET'not in json.dumps(report)


@pytest.mark.parametrize('kind',[ImportError,ModuleNotFoundError,ValueError,TypeError,KeyError,AttributeError,RuntimeError,
                               OSError,FileNotFoundError,PermissionError,TimeoutError,MemoryError,AssertionError,
                               subprocess.TimeoutExpired])
def test_failure_type_allowlist_never_reads_exception_text(gate,kind):
    secret='SECRET_TOKEN_MUST_NOT_BE_SERIALIZED'
    exc=kind(['command',secret],1,output=secret,stderr=secret)if kind is subprocess.TimeoutExpired else kind(secret)
    report={};gate.failure(report,'weights_decode',exc)
    assert report==dict(failure_gate='weights_decode',failure_class=kind.__name__)
    assert secret not in json.dumps(report)
    gate.failure(report,'host_cleanup',TypeError(secret))
    assert report==dict(failure_gate='weights_decode',failure_class=kind.__name__)


def test_unknown_exception_and_gate_are_other_without_text_or_classname(gate):
    class Unknown(RuntimeError):
        def __str__(self):raise AssertionError('Exception text was read')
    Unknown.__name__='SECRET_CUSTOM_EXCEPTION_CLASS'
    report={};gate.failure(report,'SECRET_UNTRUSTED_GATE',Unknown('SECRET_EXCEPTION_PAYLOAD'))
    assert report==dict(failure_gate='other',failure_class='other')and 'SECRET'not in json.dumps(report)


def test_terminal_failure_message_contains_only_allowlisted_diagnostics(gate):
    class Unknown(RuntimeError):
        def __str__(self):raise AssertionError('Exception text was read')
    def main():raise Unknown('SECRET_TERMINAL_PAYLOAD')
    terminal=ast.parse((REPO/'infra/masa_native_qualify.py').read_bytes()).body[-1];stderr=io.StringIO()
    namespace=dict(vars(gate),__name__='__main__',main=main,sys=SimpleNamespace(stderr=stderr))
    with pytest.raises(SystemExit):exec(compile(ast.Module(body=[terminal],type_ignores=[]),'<safe-terminal>','exec'),namespace)
    assert stderr.getvalue()=='MASA native contract failed {"failure_class": "other", "failure_gate": "bootstrap"}\n'


@pytest.mark.parametrize('subgate',[None,'operators','registry','config','model_construct','weights_decode','strict_state',
                                   'model_device','preprocess','encoder','embedding'])
def test_mock_native_failure_subgate_and_class_without_gpu_or_payload(gate,tmp_path,monkeypatch,subgate):
    rt,root,code,args,c=setup(gate,tmp_path,monkeypatch);out=tmp_path/'native';out.mkdir()
    proof=gate.authenticate(rt,code,args);rt.write(out/'proof.json',json.dumps(proof).encode(),0o444)
    fake_torch=SimpleNamespace(__version__='2.1.2+cu118',cuda=SimpleNamespace(is_available=lambda:True,
      get_device_capability=lambda:(9,0),get_device_name=lambda:'H100'))
    monkeypatch.setitem(gate.sys.modules,'torch',fake_torch);monkeypatch.setattr(gate.sys,'prefix',gate.VENV)
    class Unknown(RuntimeError):
        def __str__(self):raise AssertionError('Exception text was read')
    def operators(*a):
        if subgate=='operators':raise ImportError('SECRET_OPERATOR_PAYLOAD')
        return {'checkpoint_read':False}
    def model(torch,policy,deadline,progress):
        if subgate:
            progress['subgate']=subgate
            if subgate=='embedding':raise Unknown('SECRET_MODEL_PAYLOAD')
            raise ValueError('SECRET_MODEL_PAYLOAD')
        progress.update(model_constructed=True,weights_decoded=True,model_loaded=True)
        return {'native_encoder_calls':1,'native_embedding_calls':3}
    monkeypatch.setattr(gate,'operators',operators);monkeypatch.setattr(gate,'native_model',model)
    if subgate:
        with pytest.raises(ValueError,match='Native contract failed'):
            gate.run_native(code,args.revision,out,rt.identity(out/'proof.json'),gate.time.monotonic()+600)
        report=json.loads((out/'native.json').read_bytes())
        assert report['status']=='fail'and report['failure_gate']==subgate
        assert report['failure_class']==('ImportError'if subgate=='operators'else'other'if subgate=='embedding'else'ValueError')
        assert 'SECRET'not in json.dumps(report)
    else:
        report=gate.run_native(code,args.revision,out,rt.identity(out/'proof.json'),gate.time.monotonic()+600)
        assert report['status']=='pass'and report['subgate']=='complete'
        assert 'failure_gate'not in report and 'failure_class'not in report
    assert report['source_artifacts_rehashed_after']


@pytest.mark.parametrize('fault',[None,'native_fail','source_posthash','foreign_entry','mutated_native','cleanup_error','missing_cid','late_publication'])
def test_mock_dispatch_actual_complete_native_receipt_and_cleanup(gate,tmp_path,monkeypatch,fault):
    rt,root,code,args,c=setup(gate,tmp_path,monkeypatch);calls=[]
    monkeypatch.setattr(gate.os,'chown',lambda *a:None)
    monkeypatch.setattr(gate,'acquire_lock_fd',lambda path:os.open(path,os.O_RDONLY))
    monkeypatch.setattr(gate,'image',lambda *a:dict(image_id=args.image_id,layers=['sha256:'+'e'*64]))
    if fault=='late_publication':
        real_time=gate.time.monotonic;clock=[real_time()];monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0])
        real_fchmod=os.fchmod
        def fchmod(fd,mode):
            real_fchmod(fd,mode)
            if mode==0o444:
                st=os.fstat(fd)
                out=root/'results'/('masa-native-qualification-'+args.revision)/'report.json'
                if out.exists()and out.stat().st_ino==st.st_ino:clock[0]+=601
        monkeypatch.setattr(gate.os,'fchmod',fchmod)
    def command(argv,deadline,log=None,pass_fds=()):
        calls.append(argv)
        if argv[:2]==['docker','run']:
            out=Path(argv[argv.index('--out')+1]);proof=rt.strict((out/'proof.json').read_bytes());pin=rt.identity(out/'proof.json')
            n=dict(stage='masa_native_data_free_contract',status='pass',phase='complete',producer_revision=args.revision,image_id=args.image_id,
              protocol_identity=gate.CONFIG_PIN,proof_identity=pin,source_binding=proof['source_binding'],source_artifacts_rehashed_after=True,
              quality_verified=False,adoption=False,challenge_inputs_used=False,ground_truth_used=False,operators={'checkpoint_read':False},
              model_loaded=True,model_constructed=True,weights_decoded=True,gpu_used=True,tracking_correctness_verified=False,
              identity_or_physical_ownership_verified=False,
              model={'native_encoder_calls':1,'native_embedding_calls':3})
            if fault=='native_fail':n.update(status='fail',phase='strict_checkpoint_model')
            rt.write(out/'native.json',json.dumps(n).encode(),0o444)
            if fault!='missing_cid':Path(argv[argv.index('--cidfile')+1]).write_text('d'*64+'\n')
            if log:log.write_bytes(b'Tiny bounded native log')
            if fault=='source_posthash':
                p=gate.DATA/'weights/masa_r50.pth';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
            if fault=='foreign_entry':(out/'foreign').write_bytes(b'foreign stays writable')
        if argv[:2]==['docker','inspect']:
            if fault=='cleanup_error':return subprocess.CompletedProcess(argv,2,b'',b'daemon unavailable')
            if fault=='mutated_native':
                p=root/'results'/('masa-native-qualification-'+args.revision)/'native/native.json'
                p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
            return subprocess.CompletedProcess(argv,1,b'\n',('error: no such container: '+'d'*64+'\n').encode())
        return subprocess.CompletedProcess(argv,0,b'',b'')
    monkeypatch.setattr(gate,'command',command)
    if fault is None:report=gate.dispatch(args,code)
    else:
        with pytest.raises(ValueError):gate.dispatch(args,code)
        report=json.loads((root/'results'/('masa-native-qualification-'+args.revision)/'report.json').read_bytes())
    assert report['status']==('pass'if fault is None else'fail')
    assert report['owned_cleanup_verified']==(fault!='cleanup_error')
    assert report['source_artifacts_rehashed_after']==(fault!='source_posthash')
    assert report.get('owned_output_sealed')==(fault not in('foreign_entry','mutated_native','missing_cid'))
    if fault:
        expected={'native_fail':'native_receipt','source_posthash':'host_posthash','foreign_entry':'output_seal',
                  'mutated_native':'output_seal','cleanup_error':'host_cleanup','missing_cid':'output_seal',
                  'late_publication':'host_deadline'}
        assert report['failure_gate']==expected[fault]
        assert report['failure_class']==('TimeoutError'if fault=='late_publication'else'ValueError')
    else:assert report['subgate']=='complete'and 'failure_gate'not in report and 'failure_class'not in report
    if fault=='late_publication':assert report['error']=='inclusive_host_deadline_exceeded'and report['elapsed_seconds']>=600
    if fault=='foreign_entry':
        p=root/'results'/('masa-native-qualification-'+args.revision)/'native/foreign'
        assert p.read_bytes()==b'foreign stays writable'and stat.S_IMODE(p.stat().st_mode)&0o222
    docker=next(x for x in calls if x[:2]==['docker','run'])
    assert '--gpus'in docker and '--network'in docker and gate.VENV+'/bin/python'in docker
    assert '/tmp:rw,exec,nosuid,size=512m'in docker and '-i'in docker and 'LD_LIBRARY_PATH='in docker
    assert not any('track2'in x.lower()or 'track3'in x.lower()for x in docker)


@pytest.mark.parametrize('case',['uppercase','lowercase','daemon','foreign','extra','nonempty','wrong_rc','timeout','wrong_image'])
def test_exact_cleanup_streams_never_remove_foreign_or_unknown(gate,tmp_path,monkeypatch,case):
    rt=load_rt();cid=tmp_path/'container.cid';cid.write_bytes(('d'*64+'\n').encode());calls=[]
    prefixes={'uppercase':'Error: No such container: ','lowercase':'error: no such object: ',
              'daemon':'Error response from daemon: No such container: '}
    def invoke(argv,*args,**kwargs):
        calls.append(argv)
        if argv[:2]!=['docker','inspect']:return subprocess.CompletedProcess(argv,0,b'',b'')
        if case=='timeout':raise subprocess.TimeoutExpired(argv,1)
        if case=='wrong_image':
            actual=dict(Id='d'*64,Name='/owned',Image='sha256:'+'f'*64,Config={'Labels':{'world_reward.masa_native.owner':'a'*40}})
            return subprocess.CompletedProcess(argv,0,json.dumps(actual).encode(),b'')
        prefix=prefixes.get(case,prefixes['lowercase']);value='e'*64 if case=='foreign'else'd'*64
        stderr=(prefix+value+('\nextra'if case=='extra'else'')+'\n').encode()
        return subprocess.CompletedProcess(argv,2 if case=='wrong_rc'else 1,b'garbage'if case=='nonempty'else b'\n',stderr)
    monkeypatch.setattr(gate,'command',invoke)
    if case in prefixes:
        gate.cleanup(rt,cid,'owned','sha256:'+'b'*64,'a'*40,999999)
        assert stat.S_IMODE(cid.stat().st_mode)==0o444
    else:
        with pytest.raises((ValueError,subprocess.TimeoutExpired)):gate.cleanup(rt,cid,'owned','sha256:'+'b'*64,'a'*40,999999)
        assert stat.S_IMODE(cid.stat().st_mode)&0o222
    assert not any(x[:3]==['docker','rm','-f']for x in calls)


def test_shell_syntax_and_source_closure(gate):
    subprocess.run(['bash','-n',str(REPO/'infra/run_masa_native_qualify.sh')],check=True)
    spec=importlib.util.spec_from_file_location('wr_masa_native_bundle',REPO/'infra/azure_job.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    files={p.relative_to(REPO).as_posix():p.read_bytes()for p in(REPO/'infra').glob('*')if p.is_file()}
    for folder in('src','configs'):
        for p in(REPO/folder).rglob('*'):
            if p.is_file():files[p.relative_to(REPO).as_posix()]=p.read_bytes()
    assert set(gate.HELPERS)<=set(m.runtime_bundle_paths(files,'infra/run_masa_native_qualify.sh'))
