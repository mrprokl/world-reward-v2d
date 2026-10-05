"""Tiny Azure replica controls, entirely manufactured local bytes/no network."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    spec=importlib.util.spec_from_file_location('bank_transfer',ROOT/'infra/person_bank_transfer.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def setup(m,tmp_path,monkeypatch,action):
    root=tmp_path/'azure';(root/'results').mkdir(parents=True)
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m.os,'geteuid',lambda:0)
    monkeypatch.setattr(m.os,'uname',lambda:SimpleNamespace(sysname='Linux',nodename='scenesmith-ncc-h100-01' if action=='export' else 'world-reward-ncc-h100-02'))
    monkeypatch.setenv('WR_CODE',str(root/'code'))
    monkeypatch.setattr(m.rt,'source',lambda *a:{'helpers':{'infra/person_bank_transfer.py':{}},'frozen':True})
    return root


def test_export_originals_rehashed_archive_private_and_removed(tmp_path,monkeypatch):
    m=module();root=setup(m,tmp_path,monkeypatch,'export')
    f=root/m.BASE/'report.json';f.parent.mkdir();f.write_bytes(b'original');f.chmod(0o444)
    rows={m.BASE+'/report.json':m.identity(f)};calls=[]
    monkeypatch.setattr(m,'original_files',lambda:(rows,{'old_source':'unchanged'}))
    class Reply(io.BytesIO):status=201
    class Blob:
        def __init__(self,rev):assert rev=='a'*40
        def request(self,method,data,headers):
            assert method=='PUT' and headers=={'x-ms-blob-type':'BlockBlob','If-None-Match':'*'}
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
                assert tar.getnames()==['manifest.json',m.BASE+'/report.json']
                assert tar.extractfile(tar.getmembers()[1]).read()==b'original'
            calls.append('private_upload');return Reply()
    monkeypatch.setattr(m,'PersonBlob',Blob)
    report=m.run('export','a'*40)
    assert report['status']=='pass' and report['original_files_unchanged'] is True
    assert report['owned_archive_removed'] is True and calls==['private_upload']
    assert not (root/'results'/('person-bank-transfer-export-'+'a'*40)/'replica.tar').exists()
    assert f.read_bytes()==b'original'


def test_transfer_rejects_non_azure_host_before_creating_output(tmp_path,monkeypatch):
    m=module();root=setup(m,tmp_path,monkeypatch,'export')
    monkeypatch.setattr(m.os,'uname',lambda:SimpleNamespace(sysname='Darwin',nodename='local'))
    with pytest.raises(ValueError):m.run('export','a'*40)
    assert list((root/'results').iterdir())==[]


def test_blob_url_no_sas_token_or_public_endpoint():
    m=module();b=m.PersonBlob('a'*40)
    assert b.url=='https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/person-banks-'+('a'*40)+'.tar'
    assert b.managed_identity is True and b.token is None
    with pytest.raises(ValueError):m.PersonBlob('main')


def test_import_exact_replica_published_no_collision_and_blob_deleted(tmp_path,monkeypatch):
    m=module();root=setup(m,tmp_path,monkeypatch,'import')
    names={m.BASE+'/report.json':b'original-report',m.OLD+'/code/configs/example.json':b'code',m.OLD+'/revision':b'old-revision'}
    rows={k:dict(bytes=len(v),sha256=__import__('hashlib').sha256(v).hexdigest()) for k,v in names.items()}
    monkeypatch.setattr(m,'REPORT',rows[m.BASE+'/report.json'])
    source={'original':'frozen'};manifest=dict(schema='world_reward.person_bank_replica.v1',original_producer_revision=m.PRODUCER,original_source_binding=source,files=rows)
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        for name,data in [('manifest.json',json.dumps(manifest).encode()),*names.items()]:
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444;tar.addfile(info,io.BytesIO(data))
    archive=stream.getvalue();pin=dict(bytes=len(archive),sha256=__import__('hashlib').sha256(archive).hexdigest());calls=[]
    class Reply(io.BytesIO):
        def __init__(self,data=b'',status=200):super().__init__(data);self.status=status;self.headers={'Content-Length':str(len(data))}
    class Blob:
        def __init__(self,revision):assert revision=='a'*40
        def request(self,method):
            calls.append(method);return Reply(archive) if method=='GET' else Reply(status=202)
    def actual():
        assert all((root/k).read_bytes()==v for k,v in names.items());return rows,source
    monkeypatch.setattr(m,'PersonBlob',Blob);monkeypatch.setattr(m,'original_files',actual)
    # macOS nonroot rename of readonly directory differs from actual Azure root.
    rename=m.os.rename
    def root_rename(src,dst):
        mode=Path(src).stat().st_mode & 0o777
        if Path(src).is_dir():Path(src).chmod(0o755)
        rename(src,dst);Path(dst).chmod(mode)
    monkeypatch.setattr(m.os,'rename',root_rename)
    try:value=m.run('import','b'*40,pin,'a'*40)
    except ValueError:
        pytest.fail((root/'results'/('person-bank-transfer-import-'+'b'*40)/'report.json').read_text())
    assert value['status']=='pass' and calls==['GET','DELETE']
    assert value['private_transfer_blob_removed'] and value['owned_staging_removed'] and value['owned_archive_removed']
    assert (root/m.OLD/'code').stat().st_mode & 0o222==0


def test_final_posthash_error_demotes_prior_success(tmp_path,monkeypatch):
    m=module();root=setup(m,tmp_path,monkeypatch,'export')
    f=root/m.BASE/'report.json';f.parent.mkdir();f.write_bytes(b'original');f.chmod(0o444)
    rows={m.BASE+'/report.json':m.identity(f)};calls=[0]
    monkeypatch.setattr(m,'original_files',lambda:(rows,{'old_source':'unchanged'}))
    class Reply(io.BytesIO):status=201
    class Blob:
        def __init__(self,rev):pass
        def request(self,*a,**kw):return Reply()
    monkeypatch.setattr(m,'PersonBlob',Blob)
    def source(*args):
        calls[0]+=1
        return {'helpers':{'infra/person_bank_transfer.py':{}},'frozen':calls[0]==1}
    monkeypatch.setattr(m.rt,'source',source)
    with pytest.raises(ValueError):m.run('export','a'*40)
    report=json.loads((root/'results'/('person-bank-transfer-export-'+'a'*40)/'report.json').read_text())
    assert report['status']=='fail' and report['error_type']=='ValueError'
