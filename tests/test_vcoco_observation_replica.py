"""Authored byte-only transfer controls. No photos, NPZ interpretation or HTTP."""
import base64
import copy
import hashlib
import io
from pathlib import Path
import tarfile
from types import SimpleNamespace
import time
import pytest
import vcoco_observation_replica as v

R='1'*40

def fixture_manifest(monkeypatch):
    payload={**{'inputs/image_%06d.jpg'%i:(b'JPEG authored %d'%i)for i in range(16)},
        **{'banks/image_%06d.npz'%i:(b'opaque NPZ authored %d'%i)for i in range(16)},
        'inputs/manifest.json':b'public authored',**{'banks/'+n:n.encode()for n in v.PINS}}
    monkeypatch.setattr(v,'PUBLIC',v.pin(payload['inputs/manifest.json']))
    monkeypatch.setattr(v,'PINS',{n:v.pin(payload['banks/'+n])for n in ('host.json','native.json','proof.json')})
    m=dict(schema=v.SCHEMA,export_revision=R,producer_revision=v.REV,
        original_source_declaration=dict(entries=321,closure_sha256=v.CLOSURE),original_audit_declaration=v.AUDIT,
        public_identity=v.PUBLIC,files={n:v.pin(raw)for n,raw in payload.items()})
    return m,payload


def archive_bytes(m,payload,*,members=None,tail=b'',format=tarfile.USTAR_FORMAT):
    raw=io.BytesIO()
    with tarfile.open(fileobj=raw,mode='w',format=format)as archive:
        for name,data,kind in members or [('manifest.json',v.encode(m),tarfile.REGTYPE),*[(n,payload[n],tarfile.REGTYPE)for n in sorted(payload)]]:
            row=tarfile.TarInfo(name);row.size=len(data);row.type=kind;row.mode=0o400
            if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):row.linkname='foreign'
            archive.addfile(row,io.BytesIO(data))
    return raw.getvalue()+tail


def write_readonly(path,raw):path.write_bytes(raw);path.chmod(0o400);return v.pin(raw)


def test_complete_36_original_bytes_and_slots(monkeypatch,tmp_path):
    m,payload=fixture_manifest(monkeypatch);raw=archive_bytes(m,payload);p=tmp_path/'in.tar';ap=write_readonly(p,raw)
    assert v.validate_manifest(m)==m
    assert len(m['files'])==36
    assert v.verify_archive(p,ap,v.pin(v.encode(m)),R,time.monotonic()+5)==m
    assert all(m['files'][n]==v.pin(payload[n])for n in payload)


@pytest.mark.parametrize('change',['hole','unknown','alias','size','hash','schema','source','revision','duplicatephoto','oversize'])
def test_frozen_manifest_rejects(monkeypatch,change):
    m,_=fixture_manifest(monkeypatch);m=copy.deepcopy(m)
    if change=='hole':del m['files']['banks/image_000015.npz']
    elif change=='unknown':m['files']['eval_private/reference.json']=v.pin(b'x')
    elif change=='alias':m['files']['inputs/./image_000000.jpg']=m['files'].pop('inputs/image_000000.jpg')
    elif change=='size':m['files']['banks/image_000000.npz']['bytes']=0
    elif change=='hash':m['files']['banks/image_000000.npz']['sha256']='bad'
    elif change=='schema':m['schema']='other'
    elif change=='source':m['original_source_declaration']['entries']=320
    elif change=='revision':m['export_revision']='bad'
    elif change=='duplicatephoto':m['files']['inputs/image_000016.jpg']=m['files'].pop('inputs/image_000015.jpg')
    elif change=='oversize':m['files']['banks/image_000000.npz']['bytes']=17 << 20
    with pytest.raises(ValueError):v.validate_manifest(m)


@pytest.mark.parametrize('kind',['symlink','hardlink','duplicate','unknown','escape','alias','prefix','wrongmode','payloadhash','tail','truncated','first','pax'])
def test_archive_guards(monkeypatch,tmp_path,kind):
    m,payload=fixture_manifest(monkeypatch)
    members=[('manifest.json',v.encode(m),tarfile.REGTYPE),*[(n,payload[n],tarfile.REGTYPE)for n in sorted(payload)]]
    if kind in ('symlink','hardlink'):members[1]=(members[1][0],b'',tarfile.SYMTYPE if kind=='symlink'else tarfile.LNKTYPE)
    elif kind=='duplicate':members.append(members[1])
    elif kind=='unknown':members.append(('unexpected',b'x',tarfile.REGTYPE))
    elif kind=='escape':members[1]=('../foreign',members[1][1],tarfile.REGTYPE)
    elif kind=='alias':members[1]=('banks/./host.json',members[1][1],tarfile.REGTYPE)
    elif kind=='prefix':members[1]=('p'*101+'/foreign',b'x',tarfile.REGTYPE)
    elif kind=='payloadhash':members[1]=(members[1][0],b'bad',tarfile.REGTYPE)
    elif kind=='first':members[0],members[1]=members[1],members[0]
    elif kind=='pax':members.append(('p'*101,b'x',tarfile.REGTYPE))
    raw=archive_bytes(m,payload,members=members,format=tarfile.PAX_FORMAT if kind=='pax'else tarfile.USTAR_FORMAT)
    if kind=='tail':raw+=b'notzero'+b'\0'*505
    elif kind=='truncated':raw=raw[:700]
    elif kind=='wrongmode':
        row=tarfile.TarInfo.frombuf(raw[:512],'utf-8','strict');row.mode=0o444;raw=row.tobuf(format=tarfile.USTAR_FORMAT)+raw[512:]
    p=tmp_path/'in.tar';ap=write_readonly(p,raw)
    with pytest.raises((ValueError,tarfile.HeaderError)):v.verify_archive(p,ap,v.pin(v.encode(m)),R,time.monotonic()+5)


def test_independent_archive_and_manifest_pins_before_read(monkeypatch,tmp_path):
    m,payload=fixture_manifest(monkeypatch);p=tmp_path/'in.tar';ap=write_readonly(p,archive_bytes(m,payload))
    with pytest.raises(ValueError):v.verify_archive(p,{**ap,'sha256':'0'*64},v.pin(v.encode(m)),R,time.monotonic()+5)
    with pytest.raises(ValueError):v.verify_archive(p,ap,{**v.pin(v.encode(m)),'bytes':v.MAX_MANIFEST+1},R,time.monotonic()+5)
    with pytest.raises(TimeoutError):v.verify_archive(p,ap,v.pin(v.encode(m)),R,time.monotonic()-1)


def test_source_links_rejected_before_read(tmp_path):
    p=tmp_path/'source';write_readonly(p,b'bytes');q=tmp_path/'sym';q.symlink_to(p)
    with pytest.raises(ValueError):v.rt.identity(q)
    import os
    h=tmp_path/'hard';os.link(p,h)
    with pytest.raises(ValueError):v.rt.identity(h)


def test_source_configuration_and_original_proof_are_authenticated():
    import ast
    source=Path(v.__file__).read_text();tree=ast.parse(source)
    funcs={n.name:ast.get_source_segment(source,n)for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert 'bank.configuration(code,source)'in funcs['export_inputs']
    assert 'bank.validate_report(native,recipe,REV,PINS' in funcs['export_inputs']
    assert 'rt.source(ROOT,old,REV,bank.ENTRY,bank.HELPERS)'in funcs['export_inputs']
    assert 'bank.original.qualifications'not in source and 'eval_private'not in source
    assert 'transport.Blob' in funcs['run']and 'managed_identity=True'in funcs['run']
    assert 'transport.main'not in source and 'transport.unpack'not in source
    assert 'atomic.rename_noreplace' in source and 'ctypes'not in source


def test_atomic_no_overwrite_and_owned_stage_cleanup(monkeypatch,tmp_path):
    m,payload=fixture_manifest(monkeypatch);p=tmp_path/'in.tar';write_readonly(p,archive_bytes(m,payload))
    target=tmp_path/'replica';monkeypatch.setattr(v,'DEST',target)
    def rename(stage,dest):
        if dest.exists():raise FileExistsError()
        stage.rename(dest)
    monkeypatch.setattr(v.atomic,'rename_noreplace',rename)
    v.install(p,tmp_path/'stage',m,time.monotonic()+5)
    assert {str(q.relative_to(target)):q.read_bytes()for q in target.rglob('*')if q.is_file()}==payload
    assert target.stat().st_mode&0o777==0o700
    assert all((target/n).stat().st_mode&0o777==0o500 for n in ('inputs','banks'))
    with pytest.raises(ValueError):v.install(p,tmp_path/'new-stage',m,time.monotonic()+5)
    assert not(tmp_path/'new-stage').exists()
    assert(target/'inputs/manifest.json').read_bytes()==payload['inputs/manifest.json']


def test_rename_race_cleans_only_owned_stage(monkeypatch,tmp_path):
    m,payload=fixture_manifest(monkeypatch);p=tmp_path/'in.tar';write_readonly(p,archive_bytes(m,payload))
    target=tmp_path/'replica';monkeypatch.setattr(v,'DEST',target)
    def race(stage,dest):dest.mkdir();(dest/'foreign').write_bytes(b'keep');raise FileExistsError()
    monkeypatch.setattr(v.atomic,'rename_noreplace',race)
    with pytest.raises(FileExistsError):v.install(p,tmp_path/'stage',m,time.monotonic()+5)
    assert not(tmp_path/'stage').exists()and(target/'foreign').read_bytes()==b'keep'


def test_foreign_stage_prevents_cleanup(tmp_path):
    stage=tmp_path/'stage';stage.mkdir();s=stage.stat();(stage/'foreign').write_bytes(b'keep')
    with pytest.raises(ValueError):v.clean_stage(stage,(s.st_dev,s.st_ino,s.st_uid),{})
    assert(stage/'foreign').read_bytes()==b'keep'


def report():return dict(schema=v.SCHEMA,phase='export',status='pass',producer_revision=R,original_producer_revision=v.REV,
    files=36,source_inputs_rehashed_after=True,outputs_sealed=True,blob_cleanup_verified=False,models_loaded=False,GPU_used=False,
    reference_metadata_read=False,quality_verified=False,ownership_verified=False,adoption=False,
    original_source_declaration=dict(entries=321,closure_sha256=v.CLOSURE),original_audit_declaration=v.AUDIT,blob_etag='"ABC-1"')


def test_export_receipt_canonical_base64_and_original_failure():
    r=report();raw=v.encode(r);encoded=base64.b64encode(raw).decode();expected=v.pin(raw)
    assert v.receipt_from_base64(encoded,expected,R)==(raw,r)
    for value,pin in ((encoded+'\n',expected),(encoded,{**expected,'sha256':'0'*64}),(base64.b64encode(v.encode({**r,'status':'fail'})).decode(),v.pin(v.encode({**r,'status':'fail'})))):
        with pytest.raises(ValueError):v.receipt_from_base64(value,pin,R)


@pytest.mark.parametrize('late',('deadline','delete','none'))
def test_same_fd_publication_persists_late_failure(tmp_path,monkeypatch,late):
    out=tmp_path/'receipt';out.mkdir(mode=0o700);s=out.stat();r=report();events=[]
    def callback():
        events.append(v.rt.strict((out/'report.json').read_bytes())['outputs_sealed'])
        if late=='delete':raise RuntimeError('SECRET must never be serialized')
    deadline=time.monotonic()-1 if late=='deadline'else time.monotonic()+5
    v.publish(out,r,deadline,time.monotonic(),(s.st_dev,s.st_ino,s.st_uid),set(),callback)
    saved=v.rt.strict((out/'report.json').read_bytes())
    assert saved==r and(out/'report.json').stat().st_mode&0o777==0o400
    assert out.stat().st_mode&0o777==0o500 and b'SECRET'not in(out/'report.json').read_bytes()
    if late=='none':assert saved['status']=='pass'and saved['blob_cleanup_verified']is True and events==[True]
    else:assert saved['status']=='fail'and saved['publication_failed']is True and saved['blob_cleanup_verified']is False
    if late=='deadline':assert not events


def test_owned_download_records_inode_and_rejects_second_open(tmp_path):
    owner=[];p=tmp_path/'archive';wrapper=v.OwnedDownload(p,owner)
    with wrapper.open('xb')as f:f.write(b'bytes')
    assert owner[0][:2]==v.snapshot(p)[:2]
    with pytest.raises(ValueError):wrapper.open('xb')


class Response:
    def __init__(self,status=201,headers=None,data=b''):self.status=status;self.headers=headers or {};self.data=io.BytesIO(data)
    def read(self,n=-1):return self.data.read(n)
    def __enter__(self):return self
    def __exit__(self,*_):pass


def test_blob_exact_managed_identity_security(monkeypatch):
    url='https://stworldrewardresearch26.blob.core.windows.net/private/articulated-runtime-'+R+'.tar'
    v.transport.Blob(url,R,managed_identity=True)
    for bad in (url.replace('stworldrewardresearch26','otheraccount'),url+'?token=secret',url.replace('https:','http:'),url.replace('articulated-runtime-','other-')):
        with pytest.raises(ValueError):v.transport.Blob(bad,R,managed_identity=True)
    assert v.transport.verify_azure_peer.__globals__ is v.transport.__dict__
    for phase,name,group in [('export','world-reward-ncc-h100-02','WORLD-REWARD-RESEARCH'),('import','scenesmith-ncc-h100-01','SCENESMITH-H100')]:
        class Opener:
            def open(self,*a,**kw):return Response(data=v.encode(dict(name=name,resourceGroupName=group)))
        monkeypatch.setattr(v.transport.urllib.request,'build_opener',lambda *a:Opener())
        v.transport.verify_azure_peer(phase)


def test_bounded_writer_and_exact_download(monkeypatch,tmp_path):
    requests=[]
    class Blob:
        def request(self,method,data=None,query='',headers=None):
            requests.append((method,query,headers));return Response(201)
    writer=v.transport.BlockWriter(Blob());writer.write(b'authored');assert writer.finish()==v.pin(b'authored')
    assert requests[-1][2]['If-None-Match']=='*'
    data=b'authored-transfer';p=tmp_path/'in';owner=[]
    class Getter:
        def request(self,*args):return Response(200,{'Content-Length':str(len(data))},data)
    v.transport.download(Getter(),v.OwnedDownload(p,owner),v.pin(data));assert p.read_bytes()==data
    with pytest.raises(ValueError):v.transport.download(Getter(),tmp_path/'bad',{**v.pin(data),'sha256':'0'*64})


def test_arguments_no_implicit_export_or_import_pins():
    assert v.arguments(['export']).phase=='export'
    with pytest.raises(ValueError):v.arguments(['export','--archive-bytes','3'])
    with pytest.raises(ValueError):v.arguments(['import','--export-revision',R])


def test_export_ustar_and_exclusive_blob_head(monkeypatch,tmp_path):
    m,payload=fixture_manifest(monkeypatch);paths={}
    for i,(name,raw)in enumerate(payload.items()):
        path=tmp_path/('payload-%d'%i);write_readonly(path,raw);paths[name]=path
    class Blob:
        def __init__(self):self.data={};self.archive=b'';self.heads=0
        def request(self,method,data=None,query='',headers=None):
            if query.startswith('comp=block&'):
                self.data[query]=bytes(data);return Response(201)
            if query=='comp=blocklist':
                assert headers['If-None-Match']=='*';self.archive=b''.join(self.data.values());return Response(201)
            assert method=='HEAD';self.heads+=1
            return Response(200,{'Content-Length':str(len(self.archive)),'ETag':'"ABC-1"'})
    blob=Blob();receipt=v.export_archive(blob,m,paths,time.monotonic()+5)
    assert blob.heads==1 and receipt['archive_identity']==v.pin(blob.archive)
    archive=tmp_path/'archive';write_readonly(archive,blob.archive)
    assert v.verify_archive(archive,receipt['archive_identity'],receipt['manifest_identity'],R,time.monotonic()+5)==m


def test_source_closure_auth_and_failure_postcondition_source_only():
    import ast
    source=Path(v.__file__).read_text();tree=ast.parse(source)
    f={n.name:ast.get_source_segment(source,n)for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert "if args.phase=='export':"in f['run']
    assert "original is not None and again==manifest and proof==original"in f['run']
    assert "if args.phase=='export'and original is not None"not in f['run']
    assert "snapshot(code.parent/n)"in f['authenticate']
    assert "'DELETE',headers={'If-Match':report['blob_etag']}"in f['run']
    assert "after_seal();report['blob_cleanup_verified']=True"in f['publish']
    assert 'replica_state()==replica_before'in f['run']


def test_publisher_extra_foreign_unchanged(tmp_path):
    out=tmp_path/'receipt';out.mkdir();s=out.stat();(out/'foreign').write_bytes(b'keep');r=report()
    v.publish(out,r,time.monotonic()+5,time.monotonic(),(s.st_dev,s.st_ino,s.st_uid),set())
    assert r['status']=='fail'and(out/'foreign').read_bytes()==b'keep'
    assert(out/'foreign').stat().st_mode&0o777!=0o400
    assert v.rt.strict((out/'report.json').read_bytes())['status']=='fail'


def test_complete_export_import_run_byte_only(monkeypatch,tmp_path):
    m,payload=fixture_manifest(monkeypatch);root=tmp_path/'root';(root/'results').mkdir(parents=True)
    dest=tmp_path/'DATA'/'replica';dest.parent.mkdir();monkeypatch.setattr(v,'ROOT',root);monkeypatch.setattr(v,'DEST',dest)
    monkeypatch.setattr(v.atomic,'rename_noreplace',lambda stage,target:stage.rename(target))
    monkeypatch.setattr(v,'authenticate',lambda *a:dict(source={'author_fixture':True},states='s'))
    monkeypatch.setattr(v.transport,'verify_azure_peer',lambda *a:None)
    paths={}
    for i,(name,raw)in enumerate(payload.items()):
        path=tmp_path/('raw-%d'%i);write_readonly(path,raw);paths[name]=path
    unbound=copy.deepcopy(m);unbound['export_revision']=None
    monkeypatch.setattr(v,'export_inputs',lambda *a:(copy.deepcopy(unbound),paths,{'original':'fixture'}))
    stored={'blocks':{},'data':None,'deleted':False}
    class Blob:
        def __init__(self,url,revision,managed_identity=False):
            assert url=='https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+R+'.tar'
            assert revision==R and managed_identity is True
        def request(self,method,data=None,query='',headers=None):
            if query.startswith('comp=block&'):stored['blocks'][query]=bytes(data);return Response(201)
            if query=='comp=blocklist':
                assert stored['data']is None and headers['If-None-Match']=='*'
                stored['data']=b''.join(stored['blocks'].values());return Response(201)
            if method=='HEAD':return Response(200,{'Content-Length':str(len(stored['data'])),'ETag':'"ABC-1"'})
            if method=='GET':return Response(200,{'Content-Length':str(len(stored['data']))},stored['data'])
            assert method=='DELETE'and headers=={'If-Match':'"ABC-1"'}and dest.exists()
            saved=v.rt.strict((root/f'results/vcoco-observation-replica-import-{"2"*40}'/'report.json').read_bytes())
            assert saved['outputs_sealed']is True and saved['status']=='pass'
            stored['deleted']=True;return Response(202)
    monkeypatch.setattr(v.transport,'Blob',Blob)
    export=v.run(v.arguments(['export']),tmp_path/'code',R)
    export_path=root/f'results/vcoco-observation-replica-export-{R}'/'report.json'
    assert export['status']=='pass'and v.rt.strict(export_path.read_bytes())==export
    raw=export_path.read_bytes();ip='2'*40
    args=v.arguments(['import','--export-revision',R,'--export-receipt-base64',base64.b64encode(raw).decode(),
        '--export-receipt-bytes',str(len(raw)),'--export-receipt-sha256',v.pin(raw)['sha256'],
        '--archive-bytes',str(export['archive_identity']['bytes']),'--archive-sha256',export['archive_identity']['sha256'],
        '--manifest-bytes',str(export['manifest_identity']['bytes']),'--manifest-sha256',export['manifest_identity']['sha256']])
    def replica(_):
        assert {str(p.relative_to(dest)):p.read_bytes()for p in dest.rglob('*')if p.is_file()}==payload
        return m['files']
    monkeypatch.setattr(v,'replica_identity',replica)
    imported=v.run(args,tmp_path/'code',ip)
    assert imported['status']=='pass'and imported['blob_cleanup_verified']is True and stored['deleted']is True
    saved=root/f'results/vcoco-observation-replica-import-{ip}'/'report.json'
    assert v.rt.strict(saved.read_bytes())==imported
    assert {p.name for p in saved.parent.iterdir()}=={'report.json','manifest.json','export-receipt.json'}
    assert not(saved.parent/'archive.tar').exists()


def test_stdlib_only_import_with_ml_packages_forbidden():
    import subprocess,sys
    code="""import sys,importlib.abc
sys.path[:0]=['infra','src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in ('numpy','torch','transformers','onnxruntime','PIL','cv2'):raise AssertionError('ML import forbidden')
sys.meta_path.insert(0,Deny())
import vcoco_observation_replica
print('stdlib only')
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout=='stdlib only\n'
