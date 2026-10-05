"""Manufactured wheel ZIPs and mocked Docker only; no install/build/model/GPU."""
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
import zipfile

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate():
    spec=importlib.util.spec_from_file_location('wr_masa_build_test',REPO/'infra/masa_runtime_build.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def wheel(name,version,*,extra=None):
    metadata=('Metadata-Version: 2.1\nName: '+name+'\nVersion: '+version+'\nLicense: MIT\n\nTiny\n').encode()
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED)as z:
        z.writestr(name+'.dist-info/METADATA',metadata)
        z.writestr(name+'.dist-info/LICENSE',b'Tiny MIT notice\n')
        if extra:z.writestr(extra,b'invalid')
    return raw.getvalue(),dict(bytes=len(metadata),sha256=hashlib.sha256(metadata).hexdigest())


class Response(io.BytesIO):
    def __init__(self,raw,url):
        super().__init__(raw);self.url=url;self.status=200;self.headers=email.message.Message()
        self.headers['Content-Length']=str(len(raw));self.headers['Content-Type']='binary/octet-stream'
    def geturl(self):return self.url


class Opener:
    def __init__(self,payload):self.payload=payload;self.calls=[]
    def open(self,req,timeout):
        assert 0<timeout<=30 and dict(req.header_items())=={'Accept-encoding':'identity'}
        self.calls.append(req.full_url);return Response(self.payload[req.full_url],req.full_url)


def setup(gate,tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=root/'jobs'/revision/gate.ENTRY/'code'
    code.mkdir(parents=True);(root/'results').mkdir()
    for name in gate.HELPERS:
        p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((REPO/name).read_bytes())
    (code/'empty.py').write_bytes(b'')
    c=json.loads((code/gate.CONFIG).read_bytes());payload={}
    for r in c['wheels']:
        raw,m=wheel(r['name'],r['version']);r['bytes']=len(raw);r['metadata'].update(m)
        r['sha256']=None if r['name']=='mmcv'else hashlib.sha256(raw).hexdigest()
        if r['name']=='mmcv':r['publisher_md5']='65199300b098827d1ff8e6fabdb5082c'
        payload[r['url']]=raw
    # The one authentic publisher MD5 remains required by config(); fetch mock
    # below substitutes only the tiny fixture's digest for that row.
    for r in c['publisher_notices']:
        raw=b'Tiny Apache notice';r.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());payload[r['url']]=raw
    p=code/gate.CONFIG;p.write_text(json.dumps(c));raw=p.read_bytes()
    monkeypatch.setattr(gate,'CONFIG_PIN',dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    for n,v in(('revision',revision+'\n'),('source-sha256','b'*64+'\n')):
        (code.parent/n).write_text(v);(code.parent/n).chmod(0o444)
    for p in sorted(code.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
    code.chmod(0o555)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/gate.HELPERS[0]))
    monkeypatch.setattr(gate.shutil,'disk_usage',lambda _:type('Disk',(),{'free':30<<30})())
    original_helpers=gate.helpers
    def helpers(p):
        rt,mp=original_helpers(p)
        def publish(a,b):os.link(a,b);a.unlink()
        mp.publish=publish;return rt,mp
    monkeypatch.setattr(gate,'helpers',helpers)
    original_fetch=gate.fetch
    def fetch(rt,mp,row,*args):
        r=copy.deepcopy(row)
        if r.get('name')=='mmcv':r['publisher_md5']=hashlib.md5(payload[r['url']]).hexdigest()
        return original_fetch(rt,mp,r,*args)
    monkeypatch.setattr(gate,'fetch',fetch)
    opener=Opener(payload);return root,code,revision,c,opener


class Docker:
    def __init__(self,gate,c,revision,fault=None):
        self.gate=gate;self.c=c;self.rev=revision;self.fault=fault;self.calls=[];self.child=False
        self.base=dict(image_id=gate.BASE,layers=['sha256:'+'1'*64])
        self.new=dict(image_id='sha256:'+'2'*64,layers=self.base['layers']+['sha256:'+'3'*64])
    def __call__(self,args,deadline,*,log=None):
        self.calls.append(args);out=b'';err=b'';rc=0
        if args[:3]==['docker','image','inspect']:
            name=args[3]
            if name==self.gate.BASE:
                img=self.base
                if self.fault=='wrong_base':img=dict(image_id='sha256:'+'4'*64,layers=self.base['layers'])
            elif not self.child:
                return subprocess.CompletedProcess(args,1,b'\n',('error: no such image: '+name+'\n').encode())
            else:img=self.new
            out=json.dumps(dict(Id=img['image_id'],Architecture='amd64',Os='linux',RootFS=dict(Type='layers',Layers=img['layers']),
                Config=dict(Labels={'world_reward_masa_runtime_owner':self.rev}))).encode()
        elif args[:2]==['docker','run']:
            path=Path(args[args.index('--cidfile')+1]);path.write_text('c'*64+'\n')
            child=args[args.index('--name')+1].endswith('-child')
            data=(dict(versions={r['name']:r['version']for r in self.c['wheels']},python='3.11',isolated_venv=True,
              cuda_initialized=False,cuda_build='11.8',extension_imported=True,operator_executed=False,model_constructed=False)
              if child else dict(python='3.11',venv_with_pip=True,system_site_packages=False))
            if self.fault=='base_venv'and not child:data['python']='3.10'
            if self.fault=='child_import'and child:rc=1
            out=json.dumps(data).encode()
        elif args[:2]==['docker','build']:
            self.child=True
            if self.fault=='build':rc=1
        elif args[:2]==['docker','inspect']:
            rc=1;out=b'\n';err=('error: no such container: '+args[-1]+'\n').encode()
        elif args[:3]==['docker','image','rm']:self.child=False
        elif args[:2]==['docker','ps']:pass
        else:raise AssertionError(args)
        if log is not None:
            log.write_bytes(out or b'Bounded build log\n');log.chmod(0o400)
        return subprocess.CompletedProcess(args,rc,out,err)


def test_real_config_identity_full_dependency_graph(gate):
    raw=(REPO/gate.CONFIG).read_bytes();c=json.loads(raw)
    assert gate.CONFIG_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert len(c['wheels'])==52 and sum(r['bytes']for r in c['wheels'])==2683053759
    versions={r['name']:r['version']for r in c['wheels']};env=default_environment()
    env.update(python_version='3.11',python_full_version='3.11.0',platform_system='Linux',platform_machine='x86_64',sys_platform='linux',extra='')
    for r in c['wheels']:
        for text in r['metadata']['requires_dist_active_linux_cp311']:
            req=Requirement(text);name=req.name.lower().replace('_','-')
            assert req.marker is None or req.marker.evaluate(env)
            assert name in versions and req.specifier.contains(versions[name],prereleases=True)
    assert versions['torch']=='2.1.2+cu118' and versions['mmengine']=='0.10.3'
    assert not any(n.startswith('nvidia-')for n in versions) and 'torchaudio'not in versions


def test_actual_config_validator_and_exact_filenames(gate,tmp_path,monkeypatch):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);rt,_=gate.helpers(code)
    assert gate.config(rt,code)==c
    assert all(gate.endpoint(r['url'])==r['url']for r in c['wheels'])


def test_mock_actual_host_success_isolated_offline_no_base_mutation(gate,tmp_path,monkeypatch):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);docker=Docker(gate,c,rev);monkeypatch.setattr(gate,'command',docker)
    report=gate.run(code,rev,opener=opener)
    assert report['status']=='pass'and report['phase']=='complete' and len(opener.calls)==56
    assert report['source_rehashed_after']and report['artifacts_rehashed_after']and report['base_rechecked_after']
    assert report['owned_containers_removed']and report['owned_partial_cleanup']
    assert report['child_image']['image_id']!=gate.BASE
    assert not any(args[:3]==['docker','image','rm']for args in docker.calls)
    builds=[a for a in docker.calls if a[:2]==['docker','build']];assert len(builds)==1
    assert builds[0][builds[0].index('--network')+1]=='none'
    runs=[a for a in docker.calls if a[:2]==['docker','run']];assert len(runs)==2
    assert gate.VENV+'/bin/python'in runs[1] and not any('--gpus'in a for a in docker.calls)
    recipe=gate.recipe(c).decode()
    assert '-m venv '+gate.VENV in recipe and '--no-index --no-deps'in recipe and '-m pip check'in recipe
    assert 'system-site-packages'not in recipe and 'apt 'not in recipe
    out=root/'results'/('masa-runtime-build-'+rev)
    assert stat.S_IMODE(out.stat().st_mode)==0o555 and stat.S_IMODE((out/'report.json').stat().st_mode)==0o444
    assert not list(out.rglob('*.part'))


@pytest.mark.parametrize('fault',['wrong_base','base_venv','build','child_import'])
def test_mock_host_failure_never_changes_or_removes_base(gate,tmp_path,monkeypatch,fault):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);docker=Docker(gate,c,rev,fault);monkeypatch.setattr(gate,'command',docker)
    with pytest.raises(ValueError,match='immutable bounded receipt'):gate.run(code,rev,opener=opener)
    out=root/'results'/('masa-runtime-build-'+rev);report=json.loads((out/'report.json').read_bytes())
    assert report['status']=='fail'and report['source_rehashed_after']
    assert not any(gate.BASE in a and a[:3]==['docker','image','rm']for a in docker.calls)
    if fault in('wrong_base','base_venv'):assert not opener.calls
    else:assert report['owned_failed_image_removed']and not docker.child


def test_source_changed_postbuild_demotes_and_removes_only_child(gate,tmp_path,monkeypatch):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);docker=Docker(gate,c,rev)
    def command(args,*a,**kw):
        result=docker(args,*a,**kw)
        if args[:2]==['docker','run']and args[args.index('--name')+1].endswith('-child'):
            p=code/'empty.py';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
        return result
    monkeypatch.setattr(gate,'command',command)
    with pytest.raises(ValueError):gate.run(code,rev,opener=opener)
    report=json.loads((root/'results'/('masa-runtime-build-'+rev)/'report.json').read_bytes())
    assert report['status']=='fail'and not report['source_rehashed_after']and report['owned_failed_image_removed']


def test_first_notice_bad_sha_fails_before_build(gate,tmp_path,monkeypatch):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);docker=Docker(gate,c,rev);monkeypatch.setattr(gate,'command',docker)
    opener.payload[c['publisher_notices'][0]['url']]=b'corrupt'
    with pytest.raises(ValueError):gate.run(code,rev,opener=opener)
    assert len(opener.calls)==1 and not any(a[:2]==['docker','build']for a in docker.calls)


@pytest.mark.parametrize('extra',['../escape','/absolute','native\\escape'])
def test_unsafe_zip_paths_rejected_without_execution(gate,tmp_path,extra):
    # Lightweight pure require namespace, not importing model dependencies.
    class RT:
        @staticmethod
        def require(v,m):
            if not v:raise ValueError(m)
    raw,meta=wheel('tiny','1',extra=extra);p=tmp_path/'tiny.whl';p.write_bytes(raw)
    with pytest.raises(ValueError):gate.wheel_record(RT,p,dict(name='tiny',version='1',metadata=meta))


def test_cpu_child_uses_explicit_venv_executable(gate,tmp_path,monkeypatch):
    class RT:
        require=staticmethod(lambda v,m:None if v else(_ for _ in()).throw(ValueError(m)))
        strict=staticmethod(json.loads)
    calls=[]
    def command(args,deadline,log=None):calls.append(args);log.write_bytes(b'{}');return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(gate,'command',command)
    monkeypatch.setattr(gate,'namespace_absent',lambda *a:None)
    gate.cpu(RT,'sha256:'+'a'*64,tmp_path,'test','pass',123,isolated=True)
    assert gate.VENV+'/bin/python'in calls[0] and '--cap-drop'in calls[0]and '--network'in calls[0]


def test_public_byte_identity_replacement_after_import_demotes(gate,tmp_path,monkeypatch):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);docker=Docker(gate,c,rev)
    def command(args,*a,**kw):
        result=docker(args,*a,**kw)
        if args[:2]==['docker','run']and args[args.index('--name')+1].endswith('-child'):
            p=root/'results'/('masa-runtime-build-'+rev)/'wheels'/c['wheels'][0]['filename']
            p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
        return result
    monkeypatch.setattr(gate,'command',command)
    with pytest.raises(ValueError):gate.run(code,rev,opener=opener)
    report=json.loads((root/'results'/('masa-runtime-build-'+rev)/'report.json').read_bytes())
    assert report['status']=='fail'and report['source_rehashed_after']and not report['artifacts_rehashed_after']


def test_late_receipt_publication_demotes_same_inode(gate,tmp_path,monkeypatch):
    root,code,rev,c,opener=setup(gate,tmp_path,monkeypatch);docker=Docker(gate,c,rev)
    monkeypatch.setattr(gate,'command',docker);clock=[1000.]
    monkeypatch.setattr(gate.time,'monotonic',lambda:clock[0]);original_fsync=gate.os.fsync;count=[0]
    def fsync(fd):
        original_fsync(fd);count[0]+=1
        if count[0]==57:clock[0]=2801.  # 56 assets then the original receipt FD.
    monkeypatch.setattr(gate.os,'fsync',fsync)
    with pytest.raises(ValueError):gate.run(code,rev,opener=opener,started=1000.)
    p=root/'results'/('masa-runtime-build-'+rev)/'report.json';report=json.loads(p.read_bytes())
    assert report['status']=='fail'and report['error']=='inclusive_runtime_deadline_exceeded'
    assert report['elapsed_seconds']==1801 and p.stat().st_nlink==1 and stat.S_IMODE(p.stat().st_mode)==0o444


@pytest.mark.parametrize('stderr,stdout,rc',[
 ('error: no such container: '+('c'*64)+'\n',b'\n',1),
 ('Error: No such object: '+('c'*64)+'\n',b'[]\n',1)])
def test_owned_missing_container_exact_cli_variants(gate,tmp_path,monkeypatch,stderr,stdout,rc):
    class RT:
        require=staticmethod(lambda v,m:None if v else(_ for _ in()).throw(ValueError(m)))
        strict=staticmethod(json.loads)
        @staticmethod
        def identity(p,*_,**__):
            raw=p.read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    p=tmp_path/'container.cid';p.write_text('c'*64+'\n')
    monkeypatch.setattr(gate,'command',lambda *a,**kw:subprocess.CompletedProcess(a,rc,stdout,stderr.encode()))
    gate.cleanup_container(RT,p,'name','label',123)
    assert stat.S_IMODE(p.stat().st_mode)==0o444


@pytest.mark.parametrize('fault',['foreign','daemon','extra','wrong_rc'])
def test_owned_container_absence_does_not_accept_foreign_stream(gate,tmp_path,monkeypatch,fault):
    class RT:
        require=staticmethod(lambda v,m:None if v else(_ for _ in()).throw(ValueError(m)))
        strict=staticmethod(json.loads)
        identity=staticmethod(lambda p,*a,**kw:{'bytes':65,'sha256':'a'*64})
    p=tmp_path/'container.cid';p.write_text('c'*64+'\n');rc=1;stderr='error: no such object: '+'c'*64;stdout=b'\n'
    if fault=='foreign':stderr='error: no such object: '+'d'*64
    if fault=='daemon':stderr='Cannot connect to Docker daemon'
    if fault=='extra':stdout=b'extra\n'
    if fault=='wrong_rc':rc=2
    monkeypatch.setattr(gate,'command',lambda *a,**kw:subprocess.CompletedProcess(a,rc,stdout,stderr.encode()))
    with pytest.raises(ValueError):gate.cleanup_container(RT,p,'name','label',123)


def test_shell_and_runtime_source_closure(gate):
    subprocess.run(['bash','-n',str(REPO/'infra/run_masa_runtime_build.sh')],check=True)
    spec=importlib.util.spec_from_file_location('wr_masa_build_closure',REPO/'infra/azure_job.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    files={p.relative_to(REPO).as_posix():p.read_bytes()for p in(REPO/'infra').glob('*')if p.is_file()}
    for folder in('src','configs'):
        for p in(REPO/folder).rglob('*'):
            if p.is_file():files[p.relative_to(REPO).as_posix()]=p.read_bytes()
    assert set(gate.HELPERS)<=set(m.runtime_bundle_paths(files,'infra/run_masa_runtime_build.sh'))
