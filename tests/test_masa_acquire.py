"""Tiny manufactured streams only; no network/model/package/native execution."""
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('wr_masa_test',REPO/'infra/masa_acquire.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class Response(io.BytesIO):
    def __init__(self,raw,url,*,status=200,encoding='identity',mime='application/octet-stream',length=None):
        super().__init__(raw); self.url=url; self.status=status; self.headers=email.message.Message()
        self.headers['Content-Encoding']=encoding; self.headers['Content-Type']=mime
        self.headers['Content-Length']=str(len(raw) if length is None else length)

    def geturl(self): return self.url

    def read(self,n=-1):
        assert 0<n<=1<<20
        return super().read(n)


class Opener:
    def __init__(self,responses): self.responses=responses; self.calls=[]

    def open(self,req,timeout):
        assert dict(req.header_items())=={'Accept-encoding':'identity'} and 0<timeout<=20
        self.calls.append(req.full_url); response=self.responses[req.full_url]
        if isinstance(response,Exception): raise response
        return response


def setup(gate,tmp_path,monkeypatch):
    root=tmp_path/'root'; revision='a'*40; code=root/'jobs'/revision/gate.ENTRY/'code'
    code.mkdir(parents=True); (root/'results').mkdir()
    base=tmp_path/'data'/'masa_native_v1'; base.parent.mkdir()
    for name in gate.HELPERS:
        p=code/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((REPO/name).read_bytes())
    (code/'empty.py').write_bytes(b'')
    c=json.loads((code/gate.MANIFEST).read_bytes()); payload={}
    for r in c['assets']:
        raw=('TINY_'+r['file']).encode()
        if r['file']=='source/LICENSE': raw=b'Apache License\nVersion 2.0\n'
        if r['file']=='publisher/README.md': raw=b'---\nlibrary_name: masa\nlicense: apache-2.0\n---\nTiny card\n'
        r.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()); payload[r['file']]=raw
    (code/gate.MANIFEST).write_text(json.dumps(c))
    raw=(code/gate.MANIFEST).read_bytes()
    monkeypatch.setattr(gate,'MANIFEST_PIN',dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    monkeypatch.setattr(gate,'CHECKPOINT',{k:c['assets'][-1][k]for k in ('bytes','sha256')})
    for n,value in (('revision',revision+'\n'),('source-sha256','b'*64+'\n')):
        (code.parent/n).write_text(value); (code.parent/n).chmod(0o444)
    for p in sorted(code.rglob('*'),reverse=True): p.chmod(0o555 if p.is_dir() else 0o444)
    code.chmod(0o555)
    monkeypatch.setattr(gate,'ROOT',root); monkeypatch.setattr(gate,'BASE',base)
    monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    monkeypatch.setattr(gate.shutil,'disk_usage',lambda _:type('D',(),{'free':gate.FREE+1})())
    original=gate.helpers
    def helpers(p):
        rt,mp=original(p)
        # Test-only NOREPLACE equivalent; production uses Linux renameat2.
        def publish(a,b): os.link(a,b); a.unlink()
        mp.publish=publish
        return rt,mp
    monkeypatch.setattr(gate,'helpers',helpers)
    opener=Opener({r['url']:Response(payload[r['file']],r['url'])for r in c['assets']})
    return root,code,revision,c,payload,opener


def receipt(gate,root): return json.loads((root/gate.REPORT).read_bytes())


def test_real_manifest_identity_subset_and_helper_bytes(gate):
    raw=(REPO/gate.MANIFEST).read_bytes(); c=json.loads(raw)
    assert gate.MANIFEST_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert len(c['assets'])==18 and sum(r['bytes']for r in c['assets'])==528485634
    assert c['assets'][-1]['file']=='weights/masa_r50.pth' and c['assets'][-1]['bytes']==528391980
    assert c['source_scope'].endswith('not_whole_git_tree')
    assert c['import_boundary']['root_package_import_qualified'] is False
    assert c['import_boundary']['root_initializers_audit_only']==['masa/__init__.py','masa/models/__init__.py']
    for n,pin in gate.HELPER_PINS.items():
        raw=(REPO/n).read_bytes(); assert pin==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def test_complete_sealed_notice_first_opaque_acquisition(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    report=gate.acquire(root,code,rev,opener=opener)
    assert report['status']=='pass' and report['phase']=='complete' and len(opener.calls)==18
    assert report['source_rehashed_after'] and report['artifacts_rehashed_after'] and report['owned_partials_removed']
    assert report['source_binding']['entries']>len(gate.HELPERS) and report['public_sealed']
    assert report['upstream_license_declarations']==dict(source='Apache-2.0',publisher_card='apache-2.0',eligibility_verified=False,training_overlap_status='unknown')
    assert opener.calls==[r['url']for r in c['assets']]
    for r in report['assets']:
        p=gate.BASE/r['file']; assert p.read_bytes()==payload[r['file']] and stat.S_IMODE(p.stat().st_mode)==0o444
    assert stat.S_IMODE(gate.BASE.stat().st_mode)==0o555
    assert all(stat.S_IMODE(p.stat().st_mode)==0o555 for p in gate.BASE.rglob('*') if p.is_dir())
    assert stat.S_IMODE((root/gate.REPORT).stat().st_mode)==0o444
    assert not list(gate.BASE.rglob('*.part'))
    for n in ('models_loaded','weights_decoded','packages_installed','rgb_or_datasets_used','ground_truth_used',
              'quality_evaluated','adopted','license_eligibility_verified','training_overlap_verified'):
        assert report[n] is False


@pytest.mark.parametrize('fault',['html','pointer','gzip','status','length','hash','overflow','short','redirect','secret_exception'])
def test_transport_failure_keeps_completed_bytes_and_never_resumes(gate,tmp_path,monkeypatch,fault):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch); r=c['assets'][-1]; raw=payload[r['file']]
    kwargs={}; url=r['url']
    if fault=='html': raw=b'<html>'+raw
    if fault=='pointer': raw=b'version https://git-lfs.github.com/spec/'+raw
    if fault=='gzip': kwargs['encoding']='gzip'
    if fault=='status': kwargs['status']=403
    if fault=='length': kwargs['length']=len(raw)+1
    if fault=='hash': raw=b'X'+raw[1:]
    if fault=='overflow': raw+=b'X'; kwargs['length']=r['bytes']
    if fault=='short': raw=raw[:-1]; kwargs['length']=r['bytes']
    if fault=='redirect': url='https://attacker.invalid/?token=TOP_SECRET'
    opener.responses[r['url']]=Response(raw,url,**kwargs)
    if fault=='secret_exception': opener.responses[r['url']]=RuntimeError('token=TOP_SECRET environment password')
    with pytest.raises(ValueError,match='bounded immutable receipt'): gate.acquire(root,code,rev,opener=opener)
    report=receipt(gate,root)
    assert report['status']=='fail' and len(report['assets'])==17 and len(opener.calls)==18
    assert report['source_rehashed_after'] and report['artifacts_rehashed_after'] and report['owned_partials_removed']
    assert not (gate.BASE/r['file']).exists() and not list(gate.BASE.rglob('*.part'))
    assert 'TOP_SECRET' not in json.dumps(report) and 'https://' not in json.dumps(report)
    with pytest.raises(ValueError,match='Fresh namespace'): gate.acquire(root,code,rev,opener=opener)
    assert len(opener.calls)==18


def test_license_mismatch_prevents_weight_request(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    original=gate.notices
    def reject(*args): original(*args); raise ValueError('Declaration rejected')
    monkeypatch.setattr(gate,'notices',reject)
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    assert len(opener.calls)==17 and len(receipt(gate,root)['assets'])==17


@pytest.mark.parametrize('fault',['helper','config','source_write','extra_sibling','occupied'])
def test_fail_before_first_network_or_foreign_mutation(gate,tmp_path,monkeypatch,fault):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    if fault in ('helper','config'):
        p=code/(next(iter(gate.HELPER_PINS)) if fault=='helper' else gate.MANIFEST)
        p.chmod(0o644); p.write_bytes(p.read_bytes()+b' '); p.chmod(0o444)
    if fault=='source_write': (code/'empty.py').chmod(0o644)
    if fault=='extra_sibling': (code.parent/'foreign').write_bytes(b'untouched')
    if fault=='occupied': gate.BASE.mkdir(); (gate.BASE/'foreign').write_bytes(b'untouched')
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    assert not opener.calls
    if fault=='occupied': assert (gate.BASE/'foreign').read_bytes()==b'untouched'


def test_source_posthash_failure_demotes_pass(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch); original=gate.fetch
    def fetch(*args):
        result=original(*args)
        if args[3]['kind']=='opaque_checkpoint':
            p=code/'empty.py'; p.chmod(0o644); p.write_bytes(b'changed'); p.chmod(0o444)
        return result
    monkeypatch.setattr(gate,'fetch',fetch)
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    report=receipt(gate,root); assert report['status']=='fail' and report['error']=='cleanup_or_posthash_failed'
    assert report['public_sealed'] and not report['source_rehashed_after']


def test_completed_same_bytes_inode_replacement_fails_posthash(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch); original=gate.fetch
    def fetch(*args):
        result=original(*args)
        if args[3]['kind']=='opaque_checkpoint':
            p=gate.BASE/c['assets'][0]['file']; raw=p.read_bytes(); p.unlink(); p.write_bytes(raw); p.chmod(0o444)
        return result
    monkeypatch.setattr(gate,'fetch',fetch)
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    report=receipt(gate,root)
    assert report['status']=='fail' and report['source_rehashed_after'] and not report['artifacts_rehashed_after']


def test_replaced_partial_is_not_deleted_and_source_still_rehashed(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch); row=c['assets'][-1]
    original=opener.responses[row['url']]
    def read(n):
        part=gate.BASE/(row['file']+'.part'); part.unlink(); part.write_bytes(b'foreign')
        raise RuntimeError('No secret details')
    original.read=read
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    report=receipt(gate,root)
    assert report['status']=='fail' and not report['owned_partials_removed']
    assert report['source_rehashed_after'] and report['artifacts_rehashed_after']
    assert (gate.BASE/(row['file']+'.part')).read_bytes()==b'foreign'


@pytest.mark.parametrize('fault',['text_cap','total_cap','unsafe_path','wrong_source_revision'])
def test_manifest_contract_caps_and_namespace_paths(gate,tmp_path,monkeypatch,fault):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    if fault=='text_cap': c['assets'][2]['bytes']=gate.TEXT_MAXIMUM+1
    if fault=='total_cap': c['assets'][-1]['bytes']=gate.MAXIMUM
    if fault=='unsafe_path': c['assets'][2]['file']='source/../foreign'
    if fault=='wrong_source_revision': c['source_revision']='f'*40
    p=code/gate.MANIFEST; p.chmod(0o644); p.write_text(json.dumps(c)); p.chmod(0o444)
    raw=p.read_bytes(); monkeypatch.setattr(gate,'MANIFEST_PIN',dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    assert not opener.calls and not gate.BASE.exists()


def test_original_bootstrap_time_and_posthash_are_inclusive(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch); clock=[1000.]
    monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0]); original=gate.binding; calls=[0]
    def binding(*args):
        result=original(*args); calls[0]+=1
        if calls[0]==2: clock[0]=1501.
        return result
    monkeypatch.setattr(gate,'binding',binding)
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener,started=900.)
    report=receipt(gate,root)
    assert report['error']=='inclusive_deadline_exceeded' and report['elapsed_seconds']==601
    assert report['source_rehashed_after'] and report['artifacts_rehashed_after']


def test_low_disk_retains_failure_receipt_without_network(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    monkeypatch.setattr(gate.shutil,'disk_usage',lambda _:type('D',(),{'free':gate.FREE-1})())
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener)
    assert not opener.calls and receipt(gate,root)['status']=='fail' and not gate.BASE.exists()


def test_namespace_lease_reuses_only_empty_owned_exact_inode(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    rt,mp=gate.helpers(code); proof=gate.binding(rt,root,code,rev); gate.BASE.mkdir(mode=0o700)
    os.chown(gate.BASE,-1,os.getgid()); s=gate.BASE.stat()
    lease=dict(schema='world_reward.fresh_namespace_lease.v1',source_closure_sha256=proof['closure_sha256'],
               directories=[dict(path=str(gate.BASE),device=s.st_dev,inode=s.st_ino,uid=os.getuid(),gid=os.getgid(),mode=0o700)])
    result=gate.acquire(root,code,rev,opener=opener,namespace_lease=lease)
    assert result['namespace_lease']==lease and result['status']=='pass'


def test_bad_lease_does_not_seal_foreign_namespace(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch)
    gate.BASE.mkdir(mode=0o700); (gate.BASE/'foreign').write_bytes(b'untouched')
    with pytest.raises(ValueError): gate.acquire(root,code,rev,opener=opener,namespace_lease={})
    assert stat.S_IMODE(gate.BASE.stat().st_mode)==0o700 and (gate.BASE/'foreign').read_bytes()==b'untouched'
    assert not opener.calls and receipt(gate,root)['status']=='fail'


@pytest.mark.parametrize('url',['http://huggingface.co/x','https://u:p@huggingface.co/x','https://huggingface.co:444/x',
                              'https://evil-huggingface.co/x','https://raw.githubusercontent.com/x#secret'])
def test_public_redirect_allowlist(gate,url):
    with pytest.raises(ValueError): gate.endpoint(url)


@pytest.mark.parametrize('target,header',[
 ('https://raw.githubusercontent.com/siyuanliii/masa/hash/LICENSE',None),
 ('https://cdn-lfs.hf.co/public','Authorization'),('https://cdn-lfs.hf.co/public','Cookie')])
def test_redirect_cannot_cross_publisher_or_send_credentials(gate,target,header):
    req=gate.urllib.request.Request('https://huggingface.co/public/resolve/hash/masa_r50.pth',
                                   headers={} if header is None else {header:'TOP_SECRET'})
    with pytest.raises(ValueError): gate.PublicRedirect().redirect_request(req,None,302,'',{},target)


def test_signed_public_hf_redirect_not_logged(gate,tmp_path,monkeypatch):
    root,code,rev,c,payload,opener=setup(gate,tmp_path,monkeypatch); row=c['assets'][-1]
    opener.responses[row['url']].url='https://cas-bridge.xethub.hf.co/public?X-Amz-Signature=TOP_SECRET'
    result=gate.acquire(root,code,rev,opener=opener)
    assert result['status']=='pass' and 'TOP_SECRET' not in json.dumps(result)


def test_shell_syntax_and_actual_runtime_source_closure(gate):
    subprocess.run(['bash','-n',str(REPO/'infra/run_masa_acquire.sh')],check=True)
    spec=importlib.util.spec_from_file_location('wr_azure_job_masa',REPO/'infra/azure_job.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    files={p.relative_to(REPO).as_posix():p.read_bytes() for p in (REPO/'infra').glob('*')if p.is_file()}
    for folder in ('src','configs'):
        for p in (REPO/folder).rglob('*'):
            if p.is_file(): files[p.relative_to(REPO).as_posix()]=p.read_bytes()
    paths=m.runtime_bundle_paths(files,'infra/run_masa_acquire.sh')
    assert set(gate.HELPERS)<=set(paths)
    shell=(REPO/'infra/run_masa_acquire.sh').read_text()
    assert 'WR_ACQUISITION_STARTED' in shell and 'os.setuid(account.pw_uid)' in shell
    assert 'docker' not in shell and 'pip ' not in shell
