"""Manufactured byte streams only: no network, models, native imports or installs."""
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess

import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location('wr_triposr_test',REPO/'infra/triposr_acquire.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Response(io.BytesIO):
    def __init__(self,raw,url,*,status=200,encoding='identity',length=None,mime='application/octet-stream'):
        super().__init__(raw);self.url=url;self.status=status;self.headers=email.message.Message()
        self.headers['Content-Encoding']=encoding;self.headers['Content-Length']=str(len(raw) if length is None else length)
        self.headers['Content-Type']=mime

    def geturl(self):return self.url

    def read(self,n=-1):
        assert 0<n<=1<<20
        return super().read(n)


class Opener:
    def __init__(self,responses):self.responses=responses;self.calls=[]

    def open(self,request,timeout):
        assert 0<timeout<=20
        assert dict(request.header_items())=={'Accept-encoding':'identity'}
        self.calls.append(request.full_url)
        return self.responses[request.full_url]


def setup(gate,tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=root/'jobs'/revision/gate.JOB/'code'
    (code/'infra').mkdir(parents=True);(code/'src/world_reward').mkdir(parents=True)
    for name in gate.HELPERS:(code/name).write_bytes((REPO/name).read_bytes())
    (code/'src/world_reward/__init__.py').write_bytes(b'')
    (code.parent/'revision').write_text(revision+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    for p in sorted(code.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    code.chmod(0o555)
    for p in ('revision','source-sha256'):(code.parent/p).chmod(0o444)
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    for n in (gate.SOURCE,gate.WEIGHTS,gate.REPORT):(root/n).parent.mkdir(parents=True,exist_ok=True)
    rows=copy.deepcopy(gate.ASSETS);payload={}
    for row in rows:
        name=row['file'];raw=('TINY_OPAQUE_'+name).encode()
        if name.endswith('triposr/LICENSE'):raw=b'MIT License\nTINY'
        if name.endswith('torchmcubes/LICENSE'):raw=b'Mozilla Public License Version 2.0\nTINY'
        if name.endswith('publisher/README.md'):raw=b'---\nlicense: mit\n---\nTiny card'
        if name.endswith('dino/config.json'):raw=json.dumps({'architectures':['ViTModel'],'hidden_size':768,'patch_size':16}).encode()
        row['bytes']=len(raw);row['sha256']=hashlib.sha256(raw).hexdigest();payload[name]=raw
    monkeypatch.setattr(gate,'ASSETS',rows)
    rt,mp=gate.helpers(code)
    def publish(a,b):os.link(a,b);a.unlink()
    # The production module uses Linux renameat2; a hard-link NOREPLACE emulates
    # its exclusive publication without compiling or invoking native code here.
    original=gate.helpers
    def mocked_helpers(p):
        a,b=original(p);b.publish=publish;return a,b
    monkeypatch.setattr(gate,'helpers',mocked_helpers)
    opener=Opener({r['url']:Response(payload[r['file']],r['url']) for r in rows})
    return root,code,revision,payload,opener


def receipt(gate,root):return json.loads((root/gate.REPORT).read_bytes())


def test_complete_tiny_acquisition_and_scope(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    result=gate.acquire(root,code,revision,opener=opener)
    assert result['status']=='pass' and len(opener.calls)==31 and len(result['artifacts'])==31
    assert result['source_rehashed_after'] and result['artifacts_rehashed_after'] and result['owned_partials_removed']
    assert result['source_binding']['helpers'][gate.HELPERS[0]]['bytes']>0
    for key in ('models_loaded','packages_installed','gpu_used','dataset_read','private_values_read','quality_verified',
                'license_eligibility_verified','training_overlap_verified','challenge_overlap_verified','official_closed_solid_requirement'):
        assert result[key] is False
    assert result['upstream_license_declarations']['TripoSR_checkpoint']=='MIT'
    for row in result['artifacts']:
        p=root/row['file'];assert p.read_bytes()==payload[row['file']] and p.stat().st_mode&0o777==0o444
        assert row['sha256']==hashlib.sha256(payload[row['file']]).hexdigest()
    for name in (gate.SOURCE,gate.WEIGHTS):
        assert (root/name).stat().st_mode&0o777==0o555
        assert all(p.stat().st_mode&0o777==0o555 for p in (root/name).rglob('*') if p.is_dir())
    assert (root/gate.REPORT).stat().st_mode&0o777==0o444
    assert result['budget_seconds']==180 and result['receipt_publication_outer_seconds']==183
    assert not list(root.rglob('*.part')) and len(json.dumps(result))<20_000


@pytest.mark.parametrize('fault',['html','pointer','encoding','status','length','hash','overflow','truncated','redirect'])
def test_transport_failures_no_resume_and_no_signed_url_logging(gate,tmp_path,monkeypatch,fault):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    row=gate.ASSETS[0];raw=payload[row['file']];options={};url=row['url']
    if fault=='html':raw=b'<html>oops';options['length']=row['bytes']
    elif fault=='pointer':raw=b'version https://git-lfs.github.com/spec/v1';options['length']=row['bytes']
    elif fault=='encoding':options['encoding']='gzip'
    elif fault=='status':options['status']=403
    elif fault=='length':options['length']=row['bytes']+1
    elif fault=='hash':raw=b'X'+raw[1:]
    elif fault=='overflow':raw+=b'X';options['length']=row['bytes']
    elif fault=='truncated':raw=raw[:-1];options['length']=row['bytes']
    else:url='https://evil.invalid/?token=NEVERPRINT'
    opener.responses[row['url']]=Response(raw,url,**options)
    with pytest.raises(ValueError,match='immutable tiny receipt'):gate.acquire(root,code,revision,opener=opener)
    value=receipt(gate,root)
    assert value['status']=='fail' and value['artifacts']==[] and value['source_rehashed_after']
    assert value['owned_partials_removed'] and not list(root.rglob('*.part'))
    assert 'NEVERPRINT' not in (root/gate.REPORT).read_text()
    with pytest.raises(ValueError,match='no resume'):gate.acquire(root,code,revision,opener=opener)
    assert len(opener.calls)==1


def test_completed_exact_assets_retained_on_late_model_failure(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    row=gate.ASSETS[-1];opener.responses[row['url']]=Response(b'BAD',row['url'],length=row['bytes'])
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    value=receipt(gate,root)
    assert len(value['artifacts'])==30 and value['artifacts_rehashed_after']
    assert not (root/row['file']).exists() and not list(root.rglob('*.part'))
    assert all((root/r['file']).read_bytes()==payload[r['file']] for r in value['artifacts'])


def test_allowlisted_signed_public_hf_redirect_without_credentials(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    row=gate.ASSETS[-1]
    opener.responses[row['url']]=Response(payload[row['file']],'https://cas-bridge.xethub.hf.co/object?signature=NOTLOGGED')
    value=gate.acquire(root,code,revision,opener=opener)
    assert value['status']=='pass' and 'NOTLOGGED' not in (root/gate.REPORT).read_text()
    for url in ('http://huggingface.co/a','https://huggingface.co@evil.invalid/a','https://u:p@huggingface.co/a',
                'https://evil.hf.co.evil/a','https://huggingface.co/a#fragment'):
        with pytest.raises(ValueError):gate.endpoint(url)


def test_zero_network_for_bad_source_or_occupied_namespace(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    (root/gate.WEIGHTS).mkdir()
    with pytest.raises(ValueError,match='no resume'):gate.acquire(root,code,revision,opener=opener)
    assert not opener.calls and not (root/gate.REPORT).exists()
    (root/gate.WEIGHTS).rmdir();(code/ gate.HELPERS[0]).chmod(0o644)
    with pytest.raises(ValueError,match='Readonly'):gate.acquire(root,code,revision,opener=opener)
    assert not opener.calls


def test_changed_source_cannot_be_pass(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    original=opener.open
    def mutate(request,timeout):
        response=original(request,timeout)
        if len(opener.calls)==len(gate.ASSETS):
            p=code/'src/world_reward/__init__.py';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
        return response
    opener.open=mutate
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    assert receipt(gate,root)['source_rehashed_after'] is False


def test_expired_budget_cannot_be_pass(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    monkeypatch.setattr(gate,'BUDGET',0)
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    value=receipt(gate,root);assert value['status']=='fail' and not opener.calls


def test_public_sealing_and_posthash_count_toward_budget(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    original=gate.binding;calls=[]
    monkeypatch.setattr(gate.time,'monotonic',lambda:0.0)
    def late(rt,r,c,v):
        result=original(rt,r,c,v);calls.append(True)
        if len(calls)==2:monkeypatch.setattr(gate.time,'monotonic',lambda:181.0)
        return result
    monkeypatch.setattr(gate,'binding',late)
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    value=receipt(gate,root)
    assert value['status']=='fail' and value['elapsed_seconds']==181 and value['source_rehashed_after']


def test_precreated_namespace_lease_exact_and_occupied_rejected(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    rt,mp=gate.helpers(code);source=gate.binding(rt,root,code,revision);rows=[]
    for name in (gate.SOURCE,gate.WEIGHTS):
        p=root/name;p.mkdir(mode=0o700);p.chmod(0o700);os.chown(p,-1,os.getgid());s=p.stat()
        rows.append(dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=os.getuid(),gid=os.getgid(),mode=0o700))
    lease=dict(schema='world_reward.fresh_namespace_lease.v1',source_closure_sha256=source['closure_sha256'],directories=rows)
    assert gate.acquire(root,code,revision,opener=opener,namespace_lease=lease)['status']=='pass'


def test_mutated_lease_no_asset_io(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    rt,mp=gate.helpers(code);source=gate.binding(rt,root,code,revision);rows=[]
    for name in (gate.SOURCE,gate.WEIGHTS):
        p=root/name;p.mkdir(mode=0o700);p.chmod(0o700);os.chown(p,-1,os.getgid());s=p.stat()
        rows.append(dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=os.getuid(),gid=os.getgid(),mode=0o700))
    rows[0]['inode']+=1
    lease=dict(schema='world_reward.fresh_namespace_lease.v1',source_closure_sha256=source['closure_sha256'],directories=rows)
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener,namespace_lease=lease)
    assert not opener.calls and receipt(gate,root)['status']=='fail'


def test_invalid_dino_contract_stops_before_checkpoint(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    row=next(r for r in gate.ASSETS if r['file'].endswith('dino/config.json'))
    raw=b'{"architectures":["ViTModel"],"hidden_size":768,"hidden_size":768,"patch_size":16}'
    row['bytes']=len(raw);row['sha256']=hashlib.sha256(raw).hexdigest()
    opener.responses[row['url']]=Response(raw,row['url'])
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    assert gate.ASSETS[-1]['url'] not in opener.calls and receipt(gate,root)['status']=='fail'


def test_no_foreign_file_chmod_on_namespace_intrusion(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    original=opener.open;foreign=root/gate.SOURCE/'foreign.txt'
    def inject(request,timeout):
        response=original(request,timeout)
        if len(opener.calls)==len(gate.ASSETS):foreign.write_bytes(b'foreign');foreign.chmod(0o640)
        return response
    opener.open=inject
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    assert foreign.read_bytes()==b'foreign' and foreign.stat().st_mode&0o777==0o640
    assert receipt(gate,root)['status']=='fail'


def test_signal_removes_only_owned_partial(gate,tmp_path,monkeypatch):
    root,code,revision,payload,opener=setup(gate,tmp_path,monkeypatch)
    row=gate.ASSETS[0];response=opener.responses[row['url']];original=response.read
    def interrupted(n):
        if response.tell():raise TimeoutError('SECRET_URL_UNLOGGED')
        return original(min(n,3))
    response.read=interrupted
    with pytest.raises(ValueError):gate.acquire(root,code,revision,opener=opener)
    value=receipt(gate,root)
    assert value['owned_partials_removed'] and 'SECRET_URL_UNLOGGED' not in (root/gate.REPORT).read_text()


def test_frozen_real_inventory_only_source_model_no_media(gate):
    assert len(gate.TRIPOSR_FILES)==12 and len(gate.MCUBES_FILES)==15 and len(gate.ASSETS)==31
    assert sum(r['bytes'] for r in gate.ASSETS)<gate.MAXIMUM
    model=gate.ASSETS[-1]
    assert model['bytes']==1_677_246_742 and model['sha256']=='429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee'
    assert not any(r['file'].endswith(('.whl','.png','.mp4')) for r in gate.ASSETS)
    assert all('/resolve/'+gate.MODEL+'/' in r['url'] for r in gate.ASSETS if r['file'].startswith(gate.WEIGHTS+'/'))


def test_wrapper_bounded_root_bootstrap_then_exact_uid_drop():
    path=REPO/'infra/run_triposr_acquire.sh'
    result=subprocess.run(['bash','-n',str(path)],capture_output=True)
    assert result.returncode==0
    raw=path.read_text()
    assert '183s python3 -I -B' in raw and '--kill-after=1s' in raw
    assert 'os.setuid(account.pw_uid)' in raw and 'os.setgid(account.pw_gid)' in raw
    assert 'create_namespace_lease([root/m.SOURCE,root/m.WEIGHTS]' in raw
    assert not any(s in raw for s in ('pip install','docker build','nvidia-smi','--gpus'))


def test_runtime_source_closure_contains_exact_helpers(gate):
    spec=importlib.util.spec_from_file_location('wr_bundle_test',REPO/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names={p.relative_to(REPO).as_posix():p.read_bytes() for p in (REPO/'infra').glob('*') if p.is_file()}
    for p in (REPO/'src').rglob('*.py'):names[p.relative_to(REPO).as_posix()]=p.read_bytes()
    for p in (REPO/'configs').glob('*.json'):names[p.relative_to(REPO).as_posix()]=p.read_bytes()
    actual=module.runtime_bundle_paths(names,'infra/run_triposr_acquire.sh')
    assert set(gate.HELPERS)<=set(actual)
