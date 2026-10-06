"""Tiny authored parent/receipt tests; no Azure, Blob, archive or datasets."""
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
import vcoco_replica_parent_bootstrap as p


def ancestor_stats(monkeypatch,parent):
    original=Path.lstat
    def lstat(path):
        if path in (parent.parent,*parent.parent.parents):
            s=original(path);return SimpleNamespace(st_mode=(s.st_mode&~0o7777)|0o755,st_uid=0,st_gid=0)
        s=original(path)
        if path==parent:return SimpleNamespace(st_dev=s.st_dev,st_ino=s.st_ino,st_mode=s.st_mode,st_uid=0,st_gid=0)
        return s
    monkeypatch.setattr(Path,'lstat',lstat)


def test_only_missing_private_parent_created_and_existing_unchanged(monkeypatch,tmp_path):
    parent=tmp_path/'private';ancestor_stats(monkeypatch,parent)
    proof=p.prepare_parent(parent)
    assert proof['created']is True and parent.stat().st_mode&0o777==0o700
    p.parent_after(parent,proof);again=p.prepare_parent(parent)
    assert again['created']is False and proof['state']==again['state']


def test_existing_wrong_permissions_never_chmod(monkeypatch,tmp_path):
    parent=tmp_path/'private';parent.mkdir(mode=0o755);ancestor_stats(monkeypatch,parent)
    with pytest.raises(ValueError):p.prepare_parent(parent)
    assert parent.stat().st_mode&0o777==0o755


def test_nonempty_and_symlink_parents_rejected(monkeypatch,tmp_path):
    parent=tmp_path/'private';parent.mkdir(mode=0o700);ancestor_stats(monkeypatch,parent);(parent/'foreign').write_bytes(b'original')
    with pytest.raises(ValueError):p.prepare_parent(parent)
    assert(parent/'foreign').read_bytes()==b'original'
    other=tmp_path/'other';other.mkdir();link=tmp_path/'link';link.symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError):p.prepare_parent(link)
    assert link.is_symlink()


def test_parent_inode_and_mode_after_guard(monkeypatch,tmp_path):
    parent=tmp_path/'private';ancestor_stats(monkeypatch,parent);proof=p.prepare_parent(parent)
    parent.chmod(0o755)
    with pytest.raises(ValueError):p.parent_after(parent,proof)


def test_host_stdlib_only_no_blob_import_or_retry():
    src=Path(p.__file__).read_text()
    assert 'transport.download'not in src and 'Blob('not in src and 'install('not in src
    assert 'original.run('not in src and 'os.chmod'not in src
    code="""import sys,importlib.abc
sys.path[:0]=['infra','src']
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in ('numpy','torch','PIL','cv2','onnxruntime'):raise AssertionError('ML forbidden')
sys.meta_path.insert(0,Deny())
import vcoco_replica_parent_bootstrap
print('stdlib')
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',code],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout=='stdlib\n'


@pytest.mark.parametrize('fault',['none','sourcechange','late'])
def test_full_fake_bootstrap_publication_and_preserved_parent(monkeypatch,tmp_path,fault):
    out=tmp_path/'output';parent=tmp_path/'private';destination=parent/'replica';monkeypatch.setattr(p,'OUTPUT',out);monkeypatch.setattr(p,'PARENT',parent)
    monkeypatch.setattr(p.original,'DEST',destination);monkeypatch.setattr(p.sys,'platform','linux');monkeypatch.setattr(p.os,'geteuid',lambda:0)
    monkeypatch.setattr(p.original.transport,'verify_azure_peer',lambda phase:None);ancestor_stats(monkeypatch,parent)
    before=dict(source={'source':'current'},original_source={'source':'old'});calls=[]
    def source(*args):
        calls.append('source')
        return dict(before,states='changed')if fault=='sourcechange'and len(calls)>1 else before
    monkeypatch.setattr(p,'source',source);monkeypatch.setattr(p,'prior',lambda code:dict(original_source=before['original_source']))
    if fault=='late':monkeypatch.setattr(p,'BUDGET',0)
    r=p.run(tmp_path/'code','1'*40);saved=json.loads((out/'report.json').read_bytes())
    assert r['status']==saved['status']==('pass'if fault=='none'else 'fail')
    assert parent.exists()and parent.stat().st_mode&0o777==0o700 and not destination.exists()
    assert out.stat().st_mode&0o777==0o500 and(out/'report.json').stat().st_mode&0o777==0o400
    assert r['data_imported']is r['models_loaded']is r['blob_read']is False


def test_actual_diagnostic_is_the_pinned_metadata_only_receipt():
    file=Path(p.DIAGNOSTIC);raw=file.read_bytes();assert p.original.pin(raw)==p.DIAGNOSTIC_PIN
    d=json.loads(raw)
    assert d['parent_is_dir']is False and d['parent_canonical']is True and d['no_directory_created']is True
    assert d['original_report']==p.PINS['report.json']and d['source_entries']==327 and d['source_files']==322


def test_prior_reads_saved_diagnostic_from_authenticated_code(monkeypatch,tmp_path):
    code=tmp_path/'source';path=code/p.DIAGNOSTIC;path.parent.mkdir(parents=True)
    raw=Path(p.DIAGNOSTIC).read_bytes();path.write_bytes(raw);path.chmod(0o400)
    failed=tmp_path/'failed';failed.mkdir(mode=0o700);monkeypatch.setattr(p,'FAILED',failed)
    report=dict(schema=p.original.SCHEMA,phase='import',status='fail',producer_revision=p.REV,error_type='ValueError',elapsed_seconds=.5756366139976308,
        files=36,source_inputs_rehashed_after=True,outputs_sealed=True,archive_removed=True,blob_cleanup_verified=False,source_binding={'original':'proof'})
    for n in p.PINS:(failed/n).write_bytes(b'old');(failed/n).chmod(0o400)
    failed.chmod(0o500);requested=[];real=p.rt.pinned
    def pinned(file,pin,maximum):
        requested.append(file)
        if file==path:return real(file,pin,maximum)
        return report if file.name=='report.json'else {}
    monkeypatch.setattr(p.rt,'pinned',pinned);monkeypatch.setattr(p.original,'validate_manifest',lambda *a:None)
    monkeypatch.setattr(p.original,'receipt_from_base64',lambda *a:None)
    lstat=Path.lstat
    def stat(file):
        s=lstat(file)
        if file==failed:return SimpleNamespace(**{k:(0 if k in ('st_uid','st_gid')else getattr(s,k))for k in ('st_dev','st_ino','st_mode','st_size','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns')})
        return s
    monkeypatch.setattr(Path,'lstat',stat)
    proof=p.prior(code)
    assert path in requested and all(file!=p.ROOT/p.DIAGNOSTIC for file in requested)
    assert proof['diagnostic_is_saved_source_declaration']is True and str(path)in proof['states']
    assert p.DIAGNOSTIC in p.HELPERS
