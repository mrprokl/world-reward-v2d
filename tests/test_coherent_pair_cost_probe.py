"""Small procedural controls/mocked Docker lifecycle; no cloud or real FIT data."""
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

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cost_probe_test',REPO/'infra/coherent_pair_cost_probe.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(REPO/'infra'))
import mediapipe_cpu_runtime_verify as rt


def test_host_import_with_numpy_and_torch_blocked():
    script = f'''import importlib.abc,importlib.util,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('numpy','torch','transformers'):raise ImportError('numeric host forbidden')
sys.meta_path.insert(0,Block())
s=importlib.util.spec_from_file_location('cost_host',{str(REPO/'infra/coherent_pair_cost_probe.py')!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert 'numpy' not in sys.modules
'''
    subprocess.run([sys.executable,'-I','-B','-c',script],check=True,capture_output=True)


def test_fixed_configuration_and_frozen_helper_sha(tmp_path):
    cfg = json.loads((REPO/m.CONFIG).read_bytes());source = {'helpers':{name:rt.identity(REPO/name,readonly=False) for name in m.HELPERS}}
    (tmp_path/'configs').mkdir();path = tmp_path/m.CONFIG;path.write_bytes((REPO/m.CONFIG).read_bytes());path.chmod(0o444)
    assert m.configuration(rt,tmp_path,source) ==cfg
    assert cfg['object_slots'] ==3600 and cfg['budget_seconds'] ==720 and cfg['evaluations_per_fit_image'] ==1026
    source['helpers']['src/world_reward/coherent_pair_learning.py']['sha256'] = '0'*64
    with pytest.raises(ValueError,match='Frozen'):m.configuration(rt,tmp_path,source)


def test_tiny_procedural_banks_all_cartesian_rows_and_missing_without_filtering():
    banks = tuple(m.fixtures(np,object_count=8))
    for b,(n,k) in zip(banks,((2,4),(4,64))):
        e = b.evidence
        assert e.features.shape ==(n*2*8,10) and e.hoi_evidence.features.shape ==(n*2*k,15)
        assert e.route_features.shape ==(k*8,2)
        assert len(b.person_boxes) ==n and len(b.object_boxes) ==8
        assert (~e.feature_supported).any() and np.isnan(e.features[~e.feature_supported]).all()
        np.testing.assert_array_equal(e.arrays['person_slots'],np.repeat(np.arange(n),16))
        assert e.scope['hoi_native_pairs'] ==k
    with pytest.raises(ValueError):tuple(m.fixtures(np,object_count=0))


def test_real_tiny_six_measurements_and_cost_gate_without_full_optimizer():
    progress = dict(fixtures=[],measurements=[]);checks = []
    m.measure(np,lambda:checks.append(True),progress,object_count=8)
    assert len(progress['fixtures']) ==2 and len(progress['measurements']) ==6
    assert [r['name'] for r in progress['measurements']] ==['zero_geometry','dyadic_geometry','dyadic_relational']*2
    assert all(r['counts']['used'] ==1 and np.isfinite(r['gradient']).all() for r in progress['measurements'])
    assert progress['phase'] =='complete' and progress['fixture_sources_rehashed_after'] and not progress['full_fit_executed']
    assert progress['optimistic_min_pass_recipe_seconds'] ==min(r['seconds'] for r in progress['measurements'])*32*1026
    assert checks


def test_timeout_stops_without_resize_retry_or_claim_complete():
    progress = dict(fixtures=[],measurements=[]);calls = []
    def check():
        calls.append(True)
        if len(calls) ==3:raise TimeoutError('fixedstop')
    with pytest.raises(TimeoutError):m.measure(np,check,progress,object_count=8)
    assert len(progress['fixtures']) ==1 and len(progress['measurements']) ==0 and progress['phase'] =='zero_geometry'
    assert 'decision' not in progress


def native_receipt(binding,revision):
    value = dict(schema='world_reward.coherent_pair_cost_probe.v1',stage='coherent_pair_cost_native',status='pass',phase='complete',
        producer_revision=revision,source_binding=binding,image_id=m.IMAGE,measurements_completed=6,fixture_sources_rehashed_after=True,
        full_fit_executed=False,source_rehashed_after=True,gpu_used=False,models_loaded=False,rgb_read=False,
        external_references_read=False,challenge_inputs_used=False,quality_verified=False,adoption=False)
    value['fixtures'] = [dict(case=i,persons=n,objects=3600,native_pairs=k,original_candidate_rows=n*2*3600,
        original_tuple_rows=n*2*k,original_bridge_rows=k*3600,native_tokens=1500,retained_person_ids=n,retained_object_ids=3600,
        exact_person_groups=n,exact_object_groups=3600) for i,(n,k) in enumerate(((2,4),(4,64)))]
    value['measurements'] = [dict(case=i,name=name,seconds=1.,counts=dict(fit_records=1,used=1,missing_positive=0,no_alternative=0)) for i in range(2) for name in ('zero_geometry','dyadic_geometry','dyadic_relational')]
    value.update(optimistic_min_pass_recipe_seconds=32*1026.,decision='COST_RECIPE_UNQUALIFIED_NO_REAL_FIT')
    return value


@pytest.mark.parametrize('fault',['none','partial','truncated','tokens','source','baddecision','fakequality'])
def test_complete_native_validation_rejects_false_pass(fault):
    b = dict(closure_sha256='a'*64);v = native_receipt(b,'b'*40)
    if fault =='partial':v['measurements'].pop()
    elif fault =='truncated':v['fixtures'][0]['objects'] =3599
    elif fault =='tokens':v['fixtures'][1]['native_tokens'] =1499
    elif fault =='source':v['source_binding'] ={}
    elif fault =='baddecision':v['decision'] ='qualified'
    elif fault =='fakequality':v['quality_verified'] =True
    if fault =='none':m.validate_native(rt,v,b,'b'*40)
    else:
        with pytest.raises(ValueError):m.validate_native(rt,v,b,'b'*40)


@pytest.mark.parametrize('stdout,stderr,rc,expected',[(b'\n',b'error: no such object: CID\n',1,True),
    (b'[]\n',b'Error: No such object: CID\n',1,True),(b'',b'Error response from daemon: No such container: CID\n',1,True),
    (b'extra',b'error: no such object: CID\n',1,False),(b'',b'error: no such object: other\n',1,False),
    (b'',b'daemon down',1,False),(b'',b'error: no such object: CID\n',2,False)])
def test_exact_absence_streams_only(stdout,stderr,rc,expected):
    assert m.absent(SimpleNamespace(stdout=stdout,stderr=stderr,returncode=rc),'CID') is expected


def test_receipt_late_publish_demotes_pass_same_inode(tmp_path):
    value = dict(status='pass');path = tmp_path/'receipt.json'
    m.publish(rt,path,value,0.)
    assert json.loads(path.read_bytes())['status'] =='fail' and stat.S_IMODE(path.stat().st_mode) ==0o444
    with pytest.raises(FileExistsError):m.publish(rt,path,dict(status='pass'),float('inf'))


def test_native_real_narrow_source_import_and_shared_deadline_without_full_source_mount(tmp_path,monkeypatch):
    root = tmp_path/'root';rev = 'd'*40;code = root/'jobs'/rev/m.ENTRY/'code';out = root/'results'/('coherent-pair-cost-probe-'+rev)
    code.mkdir(parents=True);out.mkdir(parents=True);out.chmod(0o755)
    for name in m.HELPERS:
        path = code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((REPO/name).read_bytes());path.chmod(0o444)
    for name,value in (('revision',rev),('source-sha256','f'*64)):
        p = code.parent/name;p.write_text(value+'\n');p.chmod(0o444)
    binding = dict(producer_revision=rev,helpers={name:rt.identity(code/name) for name in m.HELPERS},
                   markers={name:rt.identity(code.parent/name) for name in ('revision','source-sha256')})
    proof = dict(source_binding=binding);(out/'proof.json').write_text(json.dumps(proof));(out/'proof.json').chmod(0o444)
    (out/'container.cid').write_text('c'*64+'\n');pin = rt.identity(out/'proof.json')
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m.sys,'platform','linux');monkeypatch.setattr(m.os,'geteuid',lambda:1000)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1');monkeypatch.setenv('WR_IMAGE_ID',m.IMAGE)
    original_iter,original_stat = Path.iterdir,Path.stat
    class OutStat:
        def __init__(self,s):self.s=s;self.st_uid=1000
        def __getattr__(self,k):return getattr(self.s,k)
    monkeypatch.setattr(Path,'stat',lambda p,*a,**kw:OutStat(original_stat(p,*a,**kw)) if p==out else original_stat(p,*a,**kw))
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')]) if p==Path('/sys/class/net') else original_iter(p))
    monkeypatch.setattr(m.signal,'signal',lambda *args:None);monkeypatch.setattr(m.signal,'setitimer',lambda *args:None)
    def small_measure(np_,check,progress):
        check();progress.update(phase='complete',measurements_completed=6,fixture_sources_rehashed_after=True,
            full_fit_executed=False,decision='COST_RECIPE_UNQUALIFIED_NO_REAL_FIT')
    monkeypatch.setattr(m,'measure',small_measure)
    # Force genuine import from the mounted source, not cached repository modules.
    cached = {k:v for k,v in sys.modules.items() if k=='world_reward' or k.startswith('world_reward.')}
    for k in cached:del sys.modules[k]
    oldpath = sys.path[:]
    try:
        result = m.native(code,rev,out,pin,m.time.monotonic()+30)
        assert result['status']=='pass' and result['source_rehashed_after'] and result['source_binding']==binding
        assert Path(sys.modules['world_reward.coherent_pair_learning'].__file__)==code/'src/world_reward/coherent_pair_learning.py'
        assert (out/'native.json').stat().st_mode&0o777==0o444
    finally:
        for k in list(sys.modules):
            if k=='world_reward' or k.startswith('world_reward.'):del sys.modules[k]
        sys.modules.update(cached);sys.path[:]=oldpath


@pytest.mark.parametrize('fault',['none','timeout','posthash','foreign','restrictive_umask'])
def test_mock_real_host_narrow_mounts_owned_cleanup_and_fail_demotions(tmp_path,monkeypatch,fault):
    root = tmp_path/'root';(root/'results').mkdir(parents=True);rev = 'b'*40;code = root/'jobs'/rev/m.ENTRY/'code';code.mkdir(parents=True)
    binding = dict(helpers={},closure_sha256='a'*64);before = dict(source_binding=binding,image={'Id':m.IMAGE})
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'runtime',lambda c:rt)
    def proof(*args):
        if fault =='posthash' and (root/'results'/('coherent-pair-cost-probe-'+rev)/'native.json').exists():raise ValueError('sourcechange')
        return before
    monkeypatch.setattr(m,'proof',proof);monkeypatch.setattr(m,'leaves',lambda c:[c/'one.py'])
    monkeypatch.setattr(m.os,'chown',lambda *args:None);uid = os.getuid();real_stat = Path.stat
    # The genuine Docker child owner is1000; local mocked test uses currentUID.
    class OutputStat:
        def __init__(self,s):self.original=s;self.st_uid=1000
        def __getattr__(self,name):return getattr(self.original,name)
    out = root/'results'/('coherent-pair-cost-probe-'+rev)
    monkeypatch.setattr(Path,'stat',lambda self,*a,**k:OutputStat(real_stat(self,*a,**k)) if self ==out else real_stat(self,*a,**k))
    monkeypatch.setattr(m.signal,'signal',lambda *args:None);monkeypatch.setattr(m.signal,'setitimer',lambda *args:None)
    argv = [];cid = 'c'*64;name = 'world-reward-cost-'+rev[:12]
    def run(args,**kw):
        argv.append(args)
        if args[:2] ==['docker','run']:
            (out/'container.cid').write_text(cid+'\n')
            if fault =='timeout':raise subprocess.TimeoutExpired(args,720)
            v = native_receipt(binding,rev);(out/'native.json').write_text(json.dumps(v));(out/'native.json').chmod(0o444)
            return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
        if fault =='foreign':return SimpleNamespace(returncode=0,stdout=(cid+'|'+m.IMAGE+'|/wrong|'+rev).encode(),stderr=b'')
        return SimpleNamespace(returncode=1,stdout=b'\n',stderr=('error: no such object: '+cid+'\n').encode())
    monkeypatch.setattr(m.subprocess,'run',run)
    oldmask = os.umask(0o077) if fault =='restrictive_umask' else None
    try:report = m.host(code,rev)
    finally:
        if oldmask is not None:os.umask(oldmask)
    assert report['status'] ==('pass' if fault in ('none','restrictive_umask') else 'fail')
    assert report['owned_container_removed'] is (fault !='foreign')
    assert not any(x[:3] ==['docker','rm','-f'] for x in argv)
    call = argv[0];assert '--gpus' not in call and '--network' in call and 'none' in call and '--user' in call
    assert '6g' in call and '4' in call and 'CUDA_VISIBLE_DEVICES=-1' in call and 'OMP_NUM_THREADS=4' in call
    mounts = [call[i+1] for i,x in enumerate(call) if x =='--mount'];assert len(mounts) ==2 and mounts[0].endswith(',readonly')
    assert (out/'report.json').stat().st_mode&0o777 ==0o444


def test_shell_fixed_cpu_budget_source_namespace_and_syntax():
    path = REPO/'infra/run_coherent_pair_cost_probe.sh';subprocess.run(['bash','-n',str(path)],check=True)
    source = path.read_text();assert '740s' in source and '$# == 0' in source and 'scenesmith-ncc-h100-01' in source
    assert '--gpus' not in source and 'DOCKER_HOST=' in source and 'python3 -I -B' in source
