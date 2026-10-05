"""Tiny manufactured archives/opaque weights only; no native imports or network."""
import ast
import copy
import email.message
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location('wr_trellis_test',REPO/'infra/trellis_runtime_acquire.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Response(io.BytesIO):
    def __init__(self,b,url,*,status=200,encoding='identity',length=None,mime='application/octet-stream'):
        super().__init__(b);self.url=url;self.status=status;self.headers=email.message.Message()
        self.headers['Content-Encoding']=encoding;self.headers['Content-Length']=str(len(b)if length is None else length);self.headers['Content-Type']=mime
    def geturl(self):return self.url
    def read(self,n=-1):assert 0<n<=1<<20;return super().read(n)


class Opener:
    def __init__(self,rows):self.rows=rows;self.calls=[]
    def open(self,request,timeout):
        assert 0<timeout<=30 and dict(request.header_items())=={'Accept-encoding':'identity'}
        self.calls.append(request.full_url);return self.rows[request.full_url]


def pack(prefix,files,extra=None):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w:gz')as z:
        d=tarfile.TarInfo(prefix+'/');d.type=tarfile.DIRTYPE;z.addfile(d)
        for name,b in files.items():
            n=tarfile.TarInfo(prefix+'/'+name);n.size=len(b);z.addfile(n,io.BytesIO(b))
        if extra is not None:z.addfile(extra,io.BytesIO(b'X')if extra.isfile()else None)
    return raw.getvalue()


def setup(m,tmp_path,monkeypatch):
    root=tmp_path/'root';data=tmp_path/'data'/'trellis_v1';data.parent.mkdir();rev='a'*40;code=root/'jobs'/rev/m.JOB/'code'
    for n in m.HELPERS:
        p=code/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/n).read_bytes())
    (code.parent/'revision').write_text(rev+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n');(root/'results').mkdir()
    c=json.loads((REPO/m.PROTOCOL).read_bytes());payload={};sources=[]
    for source in c['sources']:
        files={'LICENSE'if source['name']=='trellis'else'LICENSE.txt':b'MIT License'if source['name']=='trellis'else b'Apache License Version 2.0',
               'README.md':b'TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE)',
               'code.py':b'# Tiny source never executed\n','empty.py':b''}
        row=copy.deepcopy(source);row['members']=[dict(path=n,bytes=len(b),git_blob_sha1=hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest())for n,b in files.items()];row['retained_files']=list(files)
        sources.append(row);payload[row['url']]=pack(row['archive_prefix'],files)
    c['sources']=sources
    checker='# Copyright tiny\n# Licensed under Apache License\n'+''.join('# tiny\n'for _ in range(12))+'\ndef check_tensor(tensor, shape=None):\n    return tensor is not None\n'
    n=next(n for n in ast.parse(checker).body if isinstance(n,ast.FunctionDef));seg=(ast.get_source_segment(checker,n)+'\n').encode()
    gen=('\n'.join(checker.splitlines()[:14])+'\n# Exact isolated check_tensor; no Kaolin package import.\nimport torch\n\n').encode()+seg
    c['kaolin_checker'].update(exact_function_bytes=len(seg),exact_function_sha256=hashlib.sha256(seg).hexdigest(),generated_bytes=len(gen),generated_sha256=hashlib.sha256(gen).hexdigest())
    names={r['file'].removeprefix('weights/').removesuffix('.safetensors')for r in c['assets']if r['file'].endswith('.safetensors')}
    for r in c['assets']:
        raw=b'OPAQUE_'+r['file'].encode()
        if r['file']=='weights/README.md':raw=b'---\nlicense: mit\n---\nTiny card'
        elif r['file']=='weights/pipeline.json':raw=json.dumps({'name':'TrellisImageTo3DPipeline','args':{'models':{str(i):v for i,v in enumerate(sorted(names))},'image_cond_model':'dinov2_vitl14_reg'}}).encode()
        elif r['file'].endswith('.json'):raw=b'{"name":"TinyModel","args":{}}'
        elif r['file'].endswith('kaolin_apache/LICENSE'):raw=b'Apache License Version 2.0'
        elif r['file'].endswith('testing.py'):raw=checker.encode()
        r['bytes']=len(raw);r['sha256']=hashlib.sha256(raw).hexdigest();payload[r['url']]=raw
    (code/m.PROTOCOL).write_text(json.dumps(c));monkeypatch.setattr(m,'PROTOCOL_PIN',dict(bytes=(code/m.PROTOCOL).stat().st_size,sha256=hashlib.sha256((code/m.PROTOCOL).read_bytes()).hexdigest()))
    # Use actual scope validation, except fixed production destination; keep
    # budgets and six-weight identity declarations unchanged in this tiny test.
    monkeypatch.setattr(m,'DATA',data);c['data_root']=str(data);(code/m.PROTOCOL).write_text(json.dumps(c));monkeypatch.setattr(m,'PROTOCOL_PIN',dict(bytes=(code/m.PROTOCOL).stat().st_size,sha256=hashlib.sha256((code/m.PROTOCOL).read_bytes()).hexdigest()))
    for p in sorted(code.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    code.chmod(0o555)
    for p in ('revision','source-sha256'):(code.parent/p).chmod(0o444)
    monkeypatch.setattr(m,'__file__',str(code/m.HELPERS[0]));monkeypatch.setattr(m.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(100_000_000_000,0,100_000_000_000))
    original=m.helpers
    def helpers(code):
        rt,mp=original(code)
        def publish(a,b):os.link(a,b);a.unlink()
        mp.publish=publish;return rt,mp
    monkeypatch.setattr(m,'helpers',helpers)
    return root,data,code,rev,c,payload,Opener({u:Response(b,u)for u,b in payload.items()})


def report(root):return json.loads((root/'results/trellis-runtime-acquire-v1.json').read_bytes())


def test_full_manufactured_acquisition_license_first_and_no_execution(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);v=gate.acquire(root,code,rev,opener=o)
    assert v['status']=='pass' and v['phase']=='complete' and v['first_party_license_grants_verified']
    assert len(o.calls)==18 and len(v['source_archives'])==2 and all(r['all_git_blobs_verified']for r in v['source_archives'])
    assert all(v[k]for k in('source_rehashed_after','artifacts_rehashed_after','owned_partials_removed','owned_archives_removed'))
    assert v['scope']['training_overlap_status']=='unknown' and v['scope']['license_eligibility_verified']is False
    assert not any(v['scope'][k]for k in('models_loaded','packages_installed','gpu_used','dataset_read','quality_verified','training_overlap_verified','challenge_overlap_verified'))
    assert v['future_runtime']['existing_images_modified']is False and v['future_runtime']['kaolin_package_imported']is False
    assert data.stat().st_mode&0o777==0o555 and not list(data.rglob('*.part')) and not list((data/'.archives').iterdir())
    assert (root/c['report']).stat().st_mode&0o777==0o444
    for r in v['artifacts']:assert (data/r['file']).stat().st_mode&0o777==0o444
    assert (data/c['kaolin_checker']['output']).read_bytes().startswith(b'# Copyright tiny')
    assert len(json.dumps(v))<15000
    assert (data/'source/trellis/empty.py').read_bytes()==b''


@pytest.mark.parametrize('fault',['hash','encoding','html','status','overflow','truncated','pointer','redirect'])
def test_exact_transport_failures_preserve_source_no_secret(gate,tmp_path,monkeypatch,fault):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);r=c['assets'][-1];b=payload[r['url']];kw={};url=r['url']
    if fault=='hash':b=b'X'+b[1:]
    elif fault=='encoding':kw['encoding']='gzip'
    elif fault=='html':b=b'<html>';kw['length']=r['bytes']
    elif fault=='status':kw['status']=403
    elif fault=='overflow':b+=b'X';kw['length']=r['bytes']
    elif fault=='truncated':b=b[:-1];kw['length']=r['bytes']
    elif fault=='pointer':b=b'version https://git-lfs.github.com/spec/v1';kw['length']=r['bytes']
    else:url='https://evil.invalid/?token=NEVER_LOG'
    o.rows[r['url']]=Response(b,url,**kw)
    with pytest.raises(ValueError,match='immutable receipt'):gate.acquire(root,code,rev,opener=o)
    v=report(root);assert v['status']=='fail' and v['source_rehashed_after'] and v['owned_partials_removed']
    assert not list(data.rglob('*.part')) and 'NEVER_LOG'not in(root/c['report']).read_text()
    assert len([p for p in data.rglob('*.safetensors')])==5
    with pytest.raises(ValueError,match='no resume'):gate.acquire(root,code,rev,opener=o)


@pytest.mark.parametrize('fault',['traversal','symlink','hardlink','duplicate','foreign','missing','wrong_blob'])
def test_safe_archive_inventory_rejects_before_any_weight(gate,tmp_path,monkeypatch,fault):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);r=c['sources'][0]
    files={'LICENSE':b'MIT License','README.md':b'TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE)','code.py':b'# Tiny source never executed\n','empty.py':b''}
    extra=None
    if fault=='missing':del files['code.py']
    elif fault=='wrong_blob':files['code.py']=b'X'+files['code.py'][1:]
    else:
        name='../escape'if fault=='traversal'else r['archive_prefix']+('/LICENSE'if fault=='duplicate'else'/foreign')
        extra=tarfile.TarInfo(name);extra.size=1
        if fault in('symlink','hardlink'):extra.type=tarfile.SYMTYPE if fault=='symlink'else tarfile.LNKTYPE;extra.linkname='/etc/passwd';extra.size=0
    o.rows[r['url']]=Response(pack(r['archive_prefix'],files,extra),r['url'])
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o)
    v=report(root);assert v['status']=='fail' and v['source_rehashed_after'] and v['owned_archives_removed']
    assert len(o.calls)==1 and not list(data.rglob('*.safetensors')) and not(tmp_path/'escape').exists()


def test_licence_fail_stops_before_all_model_bodies(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch)
    r=next(r for r in c['assets']if r['file']=='weights/README.md');b=b'No license here';r['bytes']=len(b);r['sha256']=hashlib.sha256(b).hexdigest()
    # Pass transport using changed manufactured manifest, then fail grant gate.
    p=code/gate.PROTOCOL;p.chmod(0o644);p.write_text(json.dumps(c));p.chmod(0o444);monkeypatch.setattr(gate,'PROTOCOL_PIN',dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));o.rows[r['url']]=Response(b,r['url'])
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o)
    v=report(root);assert v['status']=='fail' and not v['first_party_license_grants_verified'] and not list(data.rglob('*.safetensors'))


def test_wrong_source_extra_sibling_and_low_space_zero_transfer(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch)
    (code.parent/'foreign').write_bytes(b'X')
    with pytest.raises(ValueError,match='snapshot parent'):gate.acquire(root,code,rev,opener=o)
    assert not o.calls;(code.parent/'foreign').unlink()
    monkeypatch.setattr(gate.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(1,0,1))
    with pytest.raises(ValueError,match='30GB'):gate.acquire(root,code,rev,opener=o)
    assert not o.calls and not data.exists()


def test_source_mutation_late_cannot_pass(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);old=o.open
    def mutate(req,timeout):
        value=old(req,timeout)
        if len(o.calls)==18:
            p=code/gate.HELPERS[0];p.chmod(0o644);p.write_bytes(p.read_bytes()+b'\n');p.chmod(0o444)
        return value
    o.open=mutate
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o)
    assert report(root)['source_rehashed_after']is False


def test_all_posthash_sealing_inside_deadline(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);original=gate.binding;calls=[]
    def slow(*args):
        x=original(*args);calls.append(1)
        if len(calls)==2:monkeypatch.setattr(gate.time,'monotonic',lambda:10**12)
        return x
    monkeypatch.setattr(gate,'binding',slow)
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o)
    assert report(root)['status']=='fail'


def test_manifest_real_pins_and_public_redirect_contract(gate):
    raw=(REPO/gate.PROTOCOL).read_bytes();c=json.loads(raw)
    assert len(raw)==gate.PROTOCOL_PIN['bytes']and hashlib.sha256(raw).hexdigest()==gate.PROTOCOL_PIN['sha256']
    assert len([r for r in c['assets']if r['file'].endswith('.safetensors')])==6
    assert sum(r['bytes']for r in c['assets']if r['file'].endswith('.safetensors'))==3_006_922_800
    assert c['sources'][0]['revision']=='442aa1e1afb9014e80681d3bf604e8d728a86ee7'
    assert c['sources'][1]['revision']=='815e075a2a400d06c48d94c347674344ed6ae5c5'
    assert not any(p.startswith(('assets/','examples/'))for s in c['sources']for p in s['retained_files'])
    for u in('http://huggingface.co/x','https://user:pass@huggingface.co/x','https://hf.co.evil/x','https://evil.invalid/x','https://huggingface.co/x#fragment'):
        with pytest.raises(ValueError):gate.endpoint(u)
    gate.endpoint('https://cas-bridge.xethub.hf.co/object?signature=NOT_LOGGED')
    ast.parse((REPO/gate.HELPERS[0]).read_text())
    subprocess.run(['bash','-n',str(REPO/gate.HELPERS[1])],check=True)
    assert 'docker'not in(REPO/gate.HELPERS[1]).read_text()and'3630s'in(REPO/gate.HELPERS[1]).read_text()


def test_exact_pinned_submodule_empty_directory_not_materialized(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);r=c['sources'][0]
    files={'LICENSE':b'MIT License','README.md':b'TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE)',
           'code.py':b'# Tiny source never executed\n','empty.py':b''}
    extra=tarfile.TarInfo(r['archive_prefix']+'/'+r['submodules'][0]['path']);extra.type=tarfile.DIRTYPE
    o.rows[r['url']]=Response(pack(r['archive_prefix'],files,extra),r['url'])
    v=gate.acquire(root,code,rev,opener=o)
    assert v['status']=='pass' and not (data/'source/trellis'/r['submodules'][0]['path']).exists()


def test_exact_safe_directory_unknown_is_not_ignored(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);r=c['sources'][0]
    files={'LICENSE':b'MIT License','README.md':b'TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE)',
           'code.py':b'# Tiny source never executed\n','empty.py':b''}
    extra=tarfile.TarInfo(r['archive_prefix']+'/unlisted');extra.type=tarfile.DIRTYPE
    o.rows[r['url']]=Response(pack(r['archive_prefix'],files,extra),r['url'])
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o)
    assert len(o.calls)==1 and report(root)['status']=='fail'


def test_source_partial_write_failure_only_owned_removed(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch)
    old=gate.publish_bytes
    def fail(rt,mp,target,raw,owned):
        if target.name=='code.py':
            part=target.with_name(target.name+'.part');part.write_bytes(b'truncated');part.chmod(0o600)
            st=part.lstat();owned.append((part,(st.st_dev,st.st_ino,st.st_uid)))
            raise TimeoutError('no secrets')
        return old(rt,mp,target,raw,owned)
    monkeypatch.setattr(gate,'publish_bytes',fail)
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o)
    v=report(root);assert v['owned_partials_removed'] and v['owned_archives_removed'] and v['source_rehashed_after']
    assert not list(data.rglob('*.part')) and not (data/'source/trellis/code.py').exists()


def test_unprivileged_bootstrap_lease_and_public_hf_signed_redirect(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);data.mkdir(mode=0o700);os.chown(data,-1,os.getgid());st=data.lstat()
    rt,mp=gate.helpers(code);source=gate.binding(rt,root,code,rev)
    lease={'schema':'world_reward.fresh_namespace_lease.v1','source_closure_sha256':source['closure_sha256'],
           'directories':[dict(path=str(data),device=st.st_dev,inode=st.st_ino,uid=os.getuid(),gid=os.getgid(),mode=0o700)]}
    row=c['assets'][-1];o.rows[row['url']]=Response(payload[row['url']],'https://cas-bridge.xethub.hf.co/immutable?signature=NO_LOG')
    value=gate.acquire(root,code,rev,opener=o,namespace_lease=lease)
    assert value['status']=='pass' and 'NO_LOG'not in(root/c['report']).read_text()


def test_invalid_lease_rejected_before_transfer(gate,tmp_path,monkeypatch):
    root,data,code,rev,c,payload,o=setup(gate,tmp_path,monkeypatch);data.mkdir(mode=0o700);os.chown(data,-1,os.getgid());st=data.lstat()
    rt,mp=gate.helpers(code);source=gate.binding(rt,root,code,rev)
    lease={'schema':'world_reward.fresh_namespace_lease.v1','source_closure_sha256':source['closure_sha256'],
           'directories':[dict(path=str(data),device=st.st_dev,inode=st.st_ino+1,uid=os.getuid(),gid=os.getgid(),mode=0o700)]}
    with pytest.raises(ValueError):gate.acquire(root,code,rev,opener=o,namespace_lease=lease)
    assert not o.calls and report(root)['status']=='fail'
