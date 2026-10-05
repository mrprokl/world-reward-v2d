"""No Torch/GPU: tiny procedural NumPy contracts and mocked lifecycle only."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('tiny_gpu_probe_test',REPO/'infra/coherent_pair_gpu_probe.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
RTSPEC=importlib.util.spec_from_file_location('tiny_gpu_rt_test',REPO/'infra/mediapipe_cpu_runtime_verify.py')
rt=importlib.util.module_from_spec(RTSPEC);RTSPEC.loader.exec_module(rt)


def test_host_module_import_blocks_numpy_torch_and_model_stack():
    script=f'''import importlib.abc,importlib.util,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','torch','transformers'):raise ImportError('numeric host forbidden')
sys.meta_path.insert(0,Block())
s=importlib.util.spec_from_file_location('gpu_host',{str(REPO/'infra/coherent_pair_gpu_probe.py')!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert m.manifest()['budget_seconds']==1200
'''
    subprocess.run([sys.executable,'-I','-B','-c',script],check=True,capture_output=True)


def test_fixed_manifest_limits_and_helper_pins_no_data_or_alias_config():
    v=m.manifest()
    assert [r['native_pairs'] for r in v['cases']]==[4,9,4,0,0,4,4,0]
    assert max(r['persons'] for r in v['cases'])<=3 and max(r['objects'] for r in v['cases'])<=3
    assert v['alphas']==[0.,.3125] and v['repeats']==2
    assert v['torch_memory_bytes']==v['docker_host_memory_bytes']==6*1024**3
    assert v['oracle_rtol']==v['oracle_atol']==1e-12 and v['fd_step']==1e-6 and v['fd_atol']==1e-7
    for name,(size,digest) in m.REUSED.items():
        assert rt.identity(REPO/name,readonly=False)==dict(bytes=size,sha256=digest)
    assert not any(x.endswith('.json') for x in m.HELPERS)


def test_new_tiny_fixtures_keep_all_proposals_pairs_and_source_support():
    from world_reward.coherent_pair_learning import PairScale
    from world_reward.coherent_pair_cache import prepare_pair_cache
    from world_reward.coherent_pair_packed import prepare_marginal_packed
    from world_reward.coherent_pair_packed_score import score_pair_marginal_packed
    banks=list(m.fixtures(np));assert len(banks)==8
    for bank,(n,o,k,variant) in zip(banks,m.CASES):
        e=bank.evidence;assert e.original_frame_index==7 and e.image_size==(48,64)
        assert e.features.shape==(n*2*o,10) and e.scope['hoi_native_pairs']==k*k
        packed=prepare_marginal_packed(prepare_pair_cache(bank,PairScale(np.asarray(m.manifest()['scales']),np.ones(12,bool))))
        result=score_pair_marginal_packed(packed,np.asarray(m.manifest()['theta']),temperature=.8125,alpha=.3125)
        assert result.native_scores_a.shape==(n,2,o)
        if variant in ('unsupported','empty_persons','empty_objects','empty'):assert not result.supported.any()
        if variant=='aliases':assert result.scores_a.shape==(1,1) and len(result.identity['native_pair_slots'])==9
        if variant=='missing':assert np.isnan(e.features[~e.feature_supported]).all()
        if not k:assert not result.native_route_supported.any()


def test_ast_actual_snapshot_repeats_fd_envelope_and_limited_mounts():
    source=(REPO/m.HELPERS[0]).read_text();tree=ast.parse(source)
    funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    imports=[n for n in ast.walk(tree) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)]
    assert len(imports)==1 and imports[0] in list(ast.walk(funcs['native']))
    assert "('_arrays','_layouts','_pairs')" in source
    assert "a.tobytes()!=b.tobytes()" in source
    assert "(-3*first['scores_b'][active]+4*p['scores_b'][active]-q['scores_b'][active])/(2*STEP)" in source
    assert 't.cuda.set_per_process_memory_fraction(MEMORY/props.total_memory,0)' in source
    assert 'all_prepared_snapshots_rehashed_after' in source
    host=ast.get_source_segment(source,funcs['host'])
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in host and "'--gpus','all'" in host
    assert 'for path in leaves(code)' in host and 'readonly' in host
    literals=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert not any(x.startswith('/weights') or x.startswith('/data/') or x=='pip' for x in literals)


@pytest.mark.parametrize('difference',['dtype','shape','value','nan_payload','none'])
def test_bit_comparison_including_nan_payload(difference):
    a=np.array([np.nan,1.],np.float64);b=a.copy()
    if difference=='dtype':b=b.astype(np.float32)
    elif difference=='shape':b=b.reshape(1,2)
    elif difference=='value':b[1]=2.
    elif difference=='nan_payload':b.view(np.uint64)[0]^=np.uint64(1)
    if difference=='none':m.bits(a,b)
    else:
        with pytest.raises(ValueError):m.bits(a,b)


@pytest.mark.parametrize('fault',['none','late','hash','sync'])
def test_publish_same_inode_sealed_and_late_failure(tmp_path,monkeypatch,fault):
    p=tmp_path/'receipt.json';value=dict(status='pass',decision='TINY_PASS')
    if fault=='hash':
        class Bad:
            identity=staticmethod(lambda *args,**kw:dict(bytes=1,sha256='0'*64))
            require=staticmethod(rt.require)
        helper=Bad
    else:helper=rt
    if fault=='sync':
        real=m.os.fsync;count=[0]
        def fsync(fd):
            count[0]+=1
            if count[0]==2:raise OSError('controlled directory fsync')
            return real(fd)
        monkeypatch.setattr(m.os,'fsync',fsync)
    m.publish(helper,p,value,0. if fault=='late' else float('inf'),seal=True)
    assert stat.S_IMODE(p.stat().st_mode)==0o400 and stat.S_IMODE(tmp_path.stat().st_mode)==0o500
    data=json.loads(p.read_bytes());assert data['status']==('pass' if fault=='none' else 'fail')
    with pytest.raises(FileExistsError):m.publish(rt,p,dict(status='pass'),float('inf'))


def native_report(binding,rev):
    rows=[]
    for i,(n,o,k,v) in enumerate(m.CASES):
        fd_expected=v in ('complete','aliases','missing','no_hoi')
        rows.append(dict(case=i,variant=v,counts=[n,o,k*k],device_tables_before='a'*64,device_tables_after='a'*64,
            distribution_fingerprint='b'*64,host_tables_after='b'*64,results=[dict(alpha=alpha,
                repeat_bit_identical=True,oracle_passed=True,fd17_and_alpha_passed=fd_expected,
                fd_skipped_no_supported_target=not fd_expected,fd_calls=36 if fd_expected else 0,
                result_sha256='1'*64,parameter_fingerprint='2'*64) for alpha in (0.,.3125)]))
    return dict(schema=m.manifest()['schema'],stage='coherent_pair_gpu_native',status='pass',phase='complete',
        producer_revision=rev,source_binding=binding,manifest=m.manifest(),image_id=m.IMAGE,
        source_rehashed_after=True,all_prepared_snapshots_rehashed_after=True,fixtures_completed=8,
        fullbank_verified=False,quality_verified=False,fit_executed=False,models_loaded=False,rgb_read=False,
        external_references_read=False,challenge_inputs_used=False,adoption=False,
        decision='TINY_GPU_ARITHMETIC_PASS_NOT_FULLBANK_QUALIFICATION',fixtures=rows,
        segment_controls=[dict(width=w,empty=e,operation=op,bytes_sha256='3'*64)
            for w in (1,9,6,2,17) for e in (False,True) for op in ('sum','max')],
        runtime=dict(torch='2.5.1+cu124',cuda='12.4',capability=[9,0],deterministic_algorithms=True,tf32=False,
            cublas_workspace_config=':4096:8',gpu_name='NVIDIA H100',gpu_total_bytes=80*1024**3,python='3.11.10',numpy='1.26.3'),
        peak_torch_allocated=100,peak_torch_reserved=200)


@pytest.mark.parametrize('fault',['none','fixture','snapshot','fd','quality','memory','source','manifest','runtime','segment','skip','case','digest'])
def test_native_complete_receipt_rejects_false_pass(fault):
    binding=dict(closure_sha256='c'*64);rev='d'*40;v=native_report(binding,rev)
    if fault=='fixture':v['fixtures'].pop()
    elif fault=='snapshot':v['fixtures'][0]['device_tables_after']='0'*64
    elif fault=='fd':v['fixtures'][0]['results'][0]['fd_calls']=34
    elif fault=='quality':v['quality_verified']=True
    elif fault=='memory':v['peak_torch_allocated']=m.MEMORY+1
    elif fault=='source':v['source_binding']={}
    elif fault=='manifest':v['manifest']['repeats']=3
    elif fault=='runtime':v.pop('runtime')
    elif fault=='segment':v['segment_controls']=[{} for _ in range(20)]
    elif fault=='skip':v['fixtures'][0]['results'][0].update(fd17_and_alpha_passed=False,fd_skipped_no_supported_target=True,fd_calls=0)
    elif fault=='case':v['fixtures'][0]['case']=1
    elif fault=='digest':v['fixtures'][0]['results'][0]['result_sha256']='not-a-sha'
    if fault=='none':m.validate_native(rt,v,binding,rev)
    else:
        with pytest.raises(ValueError):m.validate_native(rt,v,binding,rev)


@pytest.mark.parametrize('fault',['none','foreign','survivor','query','cid','renamed'])
def test_cleanup_only_exact_owned_identity_and_bounded_commands(tmp_path,monkeypatch,fault):
    path=tmp_path/'cid';cid='e'*64;rev='f'*40;name='world-reward-tiny-gpu-'+rev[:12]
    path.write_text(('g'*64 if fault=='cid' else cid)+'\n');commands=[];queries=[0]
    def control(argv,timeout=8):
        commands.append(argv)
        if argv[:2]==['docker','ps']:
            if any(x.startswith('id=') for x in argv):return (cid+'\n').encode() if fault=='renamed' else b''
            queries[0]+=1
            if fault=='query':raise ValueError('controlled Docker unavailable')
            if fault=='renamed':return b''
            return (cid+'\n').encode() if queries[0]==1 or fault=='survivor' else b''
        metadata=cid+'|'+m.IMAGE+'|/'+('other' if fault=='foreign' else name)+'|'+rev
        return metadata.encode()
    monkeypatch.setattr(m,'control',control)
    monkeypatch.setattr(m.subprocess,'run',lambda argv,**kw:commands.append((argv,kw)) or SimpleNamespace(returncode=0))
    if fault=='none':assert m.cleanup(rt,path,name,rev) and stat.S_IMODE(path.stat().st_mode)==0o400
    else:
        with pytest.raises(ValueError):m.cleanup(rt,path,name,rev)
    mutation=[x for x in commands if isinstance(x,tuple)]
    if fault in ('foreign','query','cid','renamed'):assert not mutation
    else:assert all(x[0][-1]==cid and x[1]['timeout']==5 for x in mutation)


def test_cleanup_absent_no_foreign_or_retry_mutations(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'control',lambda *a,**k:b'')
    assert m.cleanup(rt,tmp_path/'absent','fresh','f'*40)


@pytest.mark.parametrize('fault',['none','timeout','posthash','umask'])
def test_mock_host_preimport_manifest_mounts_and_final_seals(tmp_path,monkeypatch,fault):
    root=tmp_path/'root';(root/'results').mkdir(parents=True);rev='a'*40;code=root/'jobs'/rev/m.ENTRY/'code';code.mkdir(parents=True)
    binding=dict(helpers={},closure_sha256='b'*64);before=dict(source_binding=binding,image=dict(Id=m.IMAGE),manifest=m.manifest())
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'runtime',lambda *a:rt);monkeypatch.setattr(m,'leaves',lambda *a:[code/'tiny.py'])
    out=root/m.RESULT
    def proof(*args):
        if fault=='posthash' and (out/'native.json').exists():raise ValueError('controlled changed source')
        return before
    monkeypatch.setattr(m,'proof',proof);monkeypatch.setattr(m,'control',lambda *args,**kw:b'')
    monkeypatch.setattr(m.os,'chown',lambda *a:None)
    real=Path.stat
    class S:
        def __init__(self,s):self.s=s;self.st_uid=1000
        def __getattr__(self,n):return getattr(self.s,n)
    monkeypatch.setattr(Path,'stat',lambda p,*a,**kw:S(real(p,*a,**kw)) if p==out else real(p,*a,**kw))
    monkeypatch.setattr(m.signal,'signal',lambda *a:None);monkeypatch.setattr(m.signal,'setitimer',lambda *a:None)
    commands=[]
    def run(argv,**kw):
        commands.append(argv);assert (out/'proof.json').exists()
        assert json.loads((out/'proof.json').read_bytes())['manifest']==m.manifest()
        (out/'container.cid').write_text('c'*64+'\n')
        if fault=='timeout':raise subprocess.TimeoutExpired(argv,1200)
        (out/'native.json').write_text(json.dumps(native_report(binding,rev)));(out/'native.json').chmod(0o400)
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
    monkeypatch.setattr(m.subprocess,'run',run)
    old=os.umask(0o077) if fault=='umask' else None
    try:v=m.host(code,rev)
    finally:
        if old is not None:os.umask(old)
    assert v['status']==('pass' if fault in ('none','umask') else 'fail')
    assert v['owned_container_removed'] and stat.S_IMODE((out/'report.json').stat().st_mode)==0o400
    assert stat.S_IMODE(out.stat().st_mode)==0o500
    argv=commands[0];assert argv[:2]==['docker','run'] and '--gpus' in argv and '--network' in argv
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in argv and '--memory' in argv and '6g' in argv
    assert any(str(code/'tiny.py') in x and x.endswith(',readonly') for x in argv)
    assert not any('/weights' in x or '/data/' in x for x in argv)


def test_wrapper_static_syntax_lease_and_no_local_torch_execution():
    wrapper=REPO/'infra/run_coherent_pair_gpu_probe.sh'
    subprocess.run(['bash','-n',str(wrapper)],check=True,capture_output=True)
    text=wrapper.read_text();assert 'exec 9>>' in text and 'flock --nonblock 9' in text
    assert '1215s' in text and 'nvidia-smi --query-compute-apps=pid' in text and '-I -B' in text
    assert 'docker.sock' in text and 'set +x' in text


def test_explicit_profiles_change_only_host_namespace_and_schema():
    original=m.manifest();pins=dict(m.REUSED)
    try:
        m.select_profile('v2');new=m.manifest()
        assert m.ENTRY=='run_coherent_pair_gpu_probe_v2' and m.RESULT=='results/coherent-pair-gpu-probe-v2'
        assert m.HOST=='scenesmith-ncc-h100-01' and m.HELPERS[1]=='infra/run_coherent_pair_gpu_probe_v2.sh'
        assert new['schema']=='world_reward.coherent_pair_gpu_probe.v2'
        assert {k:v for k,v in original.items() if k!='schema'}=={k:v for k,v in new.items() if k!='schema'}
        assert m.REUSED==pins and m.IMAGE=='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
        with pytest.raises(ValueError):m.select_profile('vm02_fallback')
    finally:m.select_profile('v1')
    assert m.HOST=='world-reward-ncc-h100-02' and m.manifest()==original


@pytest.mark.parametrize('profile,wrong_host',[('v1','scenesmith-ncc-h100-01'),('v2','world-reward-ncc-h100-02')])
def test_each_profile_rejects_wrong_actual_host_before_any_model_or_control(tmp_path,monkeypatch,profile,wrong_host):
    root=tmp_path/'root';rev='e'*40;entry=m.PROFILES[profile][0];code=root/'jobs'/rev/entry/'code'
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'__file__',str(code/m.HELPERS[0]))
    monkeypatch.setattr(m.sys,'argv',['probe','--profile',profile,'host',str(code),rev])
    monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setattr(m.os,'geteuid',lambda:0)
    monkeypatch.setattr(m.os,'uname',lambda:SimpleNamespace(nodename=wrong_host))
    monkeypatch.setenv('DOCKER_HOST','unix://'+str(root)+'/docker.sock')
    monkeypatch.setattr(m,'host',lambda *_:pytest.fail('wrong host launched'))
    try:
        with pytest.raises(ValueError,match='frozen profile root GPU host'):m.main()
    finally:m.select_profile('v1')


def test_v2_wrapper_is_explicit_and_v1_wrapper_not_relaxed():
    v1=(REPO/'infra/run_coherent_pair_gpu_probe.sh').read_text()
    v2=REPO/'infra/run_coherent_pair_gpu_probe_v2.sh';text=v2.read_text()
    subprocess.run(['bash','-n',str(v2)],check=True,capture_output=True)
    assert '--profile v2 host' in text and 'scenesmith-ncc-h100-01' in text
    assert 'run_coherent_pair_gpu_probe_v2/code' in text and 'exec 9>>' in text and '1215s' in text
    assert '--profile' not in v1 and 'world-reward-ncc-h100-02' in v1
