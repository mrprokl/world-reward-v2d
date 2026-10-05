"""Tiny manufactured bytes only: no V-COCO/COCO rows or real HTTP."""
from copy import deepcopy
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import stat
import time
import zipfile

import pytest
import vcoco_metadata_acquire as m


def item(raw=b'opaque not interpreted'):
    return dict(file='opaque.json', url=m.RAW+'data/vcoco/opaque.json', bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest(), mime='text/plain')


class Response(io.BytesIO):
    def __init__(self, raw, row, *, status=200, url=None, length=None, mime=None, encoding='identity'):
        super().__init__(raw); self.status = status; self.url = url or row['url']; self.read_calls = 0
        self.headers = Message(); self.headers['Content-Length'] = str(row['bytes'] if length is None else length)
        self.headers['Content-Type'] = mime or row['mime']; self.headers['Content-Encoding'] = encoding

    def geturl(self): return self.url
    def read(self, size=-1):
        assert 0 < size <= 1 << 20
        self.read_calls += 1; return super().read(size)


class Opener:
    def __init__(self, response): self.response = response; self.calls = []
    def open(self, request, timeout):
        self.calls.append((request, timeout)); return self.response


def transfer(response=None):
    return m.Transfer(deepcopy(m.PROTOCOL), time.monotonic()+30, Opener(response))


def tiny_zip(names=('annotations/first.json', 'annotations/second.json'), mode=stat.S_IFREG|0o644):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name in names:
            info = zipfile.ZipInfo(name); info.external_attr = mode << 16
            z.writestr(info, b'opaque[]not parsed', compress_type=zipfile.ZIP_DEFLATED)
    return stream.getvalue()


def fixture_archive(tmp_path, monkeypatch, raw=None):
    raw = tiny_zip() if raw is None else raw
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        catalogue = tuple(dict(member=i.filename, compressed_bytes=i.compress_size, expanded_bytes=i.file_size,
            crc32=f'{i.CRC:08x}', external_attr=i.external_attr, compression=i.compress_type, flags=i.flag_bits)
            for i in z.infolist())
    monkeypatch.setattr(m, 'CATALOGUE', catalogue); monkeypatch.setattr(m, 'TAIL', m.vg.pin(raw))
    p = tmp_path.resolve()/'original.zip'; p.write_bytes(raw); p.chmod(0o400)
    return p


def test_stream_standard_git_blob_and_sha_opaque(tmp_path, monkeypatch):
    raw = b'not even valid JSON'; row = item(raw); response = Response(raw, row); t = transfer(response)
    monkeypatch.setattr(json, 'loads', lambda *_: pytest.fail('annotation values decoded'))
    assert t.download(row, tmp_path.resolve()) == m.vg.pin(raw)
    assert m.rt.identity(tmp_path.resolve()/row['file']) == m.vg.pin(raw)
    assert len(t.opener.calls) == 1 and 0 < t.opener.calls[0][1] <= 15
    assert t.opener.calls[0][0].get_header('Accept-encoding') == 'identity'
    assert t.received == len(raw) and (tmp_path/row['file']).stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize('bad', [dict(status=206), dict(url='https://foreign.invalid/a'), dict(length=1),
                                dict(mime='image/jpeg'), dict(encoding='gzip')])
def test_headers_before_any_body_or_leaf(tmp_path, bad):
    raw = b'opaque'; row = item(raw); response = Response(raw, row, **bad)
    with pytest.raises(ValueError): transfer(response).download(row, tmp_path.resolve())
    assert response.read_calls == 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('field', ['sha256', 'git_blob_sha1'])
def test_publisher_hash_mismatch_owned_partial_only(tmp_path, field):
    raw = b'opaque'; row = item(raw); row[field] = '0'*(64 if field == 'sha256' else 40)
    foreign = tmp_path/'userwork'; foreign.write_bytes(b'keep')
    with pytest.raises(ValueError): transfer(Response(raw, row)).download(row, tmp_path.resolve())
    assert list(tmp_path.iterdir()) == [foreign] and foreign.read_bytes() == b'keep'


@pytest.mark.parametrize('raw_change', [b'opaq', b'opaque extra'])
def test_short_and_overlong_body_fail_no_retry(tmp_path, raw_change):
    row = item(b'opaque'); t = transfer(Response(raw_change, row))
    with pytest.raises(ValueError): t.download(row, tmp_path.resolve())
    assert len(t.opener.calls) == 1 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('size', [0, -1, True, 301 << 20])
def test_invalid_size_no_request(tmp_path, size):
    row = item(); row['bytes'] = size; t = transfer()
    with pytest.raises(ValueError): t.download(row, tmp_path.resolve())
    assert not t.opener.calls


@pytest.mark.parametrize('foreign', ['file', 'symlink'])
def test_foreign_output_never_overwritten_or_removed(tmp_path, foreign):
    raw = b'opaque'; row = item(raw); p = tmp_path/row['file']; keep = tmp_path/'keep'; keep.write_bytes(b'user')
    if foreign == 'file': p.write_bytes(b'existing')
    else: p.symlink_to(keep)
    with pytest.raises((ValueError, FileExistsError)): transfer(Response(raw,row)).download(row,tmp_path.resolve())
    assert p.exists() and keep.read_bytes() == b'user'


def test_owned_fsync_failure_cleanup(tmp_path, monkeypatch):
    row = item(b'opaque')
    monkeypatch.setattr(m.vg, 'sync', lambda *_: (_ for _ in ()).throw(OSError('never serialized secret')))
    with pytest.raises(OSError): transfer(Response(b'opaque',row)).download(row,tmp_path.resolve())
    assert not list(tmp_path.iterdir())


def test_original_directory_identity_and_mode_gate(tmp_path):
    out=tmp_path.resolve()/'out';out.mkdir(mode=0o700);out.chmod(0o700);owned=out.lstat()
    m.check_directory(out,owned)
    out.chmod(0o500)
    with pytest.raises(ValueError):m.check_directory(out,owned)
    out.chmod(0o700);out.rename(tmp_path/'old');out.mkdir(mode=0o700)
    with pytest.raises(ValueError):m.check_directory(out,owned)


def test_total_budget_and_deadline_before_request(tmp_path):
    row = item(b'opaque'); t = transfer(Response(b'opaque',row)); t.received = t.cfg['max_total_compressed_bytes']
    with pytest.raises(ValueError): t.download(row,tmp_path.resolve())
    assert not list(tmp_path.iterdir()) and len(t.opener.calls) == 1
    t = transfer(); t.deadline = time.monotonic()-1
    with pytest.raises(ValueError): t.download(row,tmp_path.resolve())
    assert not t.opener.calls


def test_all_crc_streams_opaque_no_extract(tmp_path, monkeypatch):
    p = fixture_archive(tmp_path,monkeypatch); before = m.rt.identity(p); t = transfer()
    monkeypatch.setattr(json, 'loads', lambda *_: pytest.fail('values decoded'))
    result = m.qualify_archive(p,t)
    assert len(result) == 2 and all(r['crc_stream_verified'] and not r['expanded_file_written']
                                  and not r['json_values_consulted'] for r in result)
    assert t.expanded_reserved == 2*len(b'opaque[]not parsed')
    assert m.rt.identity(p) == before and list(tmp_path.iterdir()) == [p]


def test_tail_and_unexpected_catalogue_before_payload(tmp_path, monkeypatch):
    p = fixture_archive(tmp_path,monkeypatch)
    monkeypatch.setattr(zipfile.ZipFile, 'open', lambda *_: pytest.fail('invalid catalogue payload read'))
    monkeypatch.setattr(m, 'TAIL', dict(bytes=p.stat().st_size,sha256='0'*64))
    with pytest.raises(ValueError): m.qualify_archive(p,transfer())
    monkeypatch.setattr(m, 'TAIL', m.vg.pin(p.read_bytes())); monkeypatch.setattr(m,'CATALOGUE',())
    with pytest.raises(ValueError): m.qualify_archive(p,transfer())


@pytest.mark.parametrize('mode', [stat.S_IFLNK|0o777,stat.S_IFDIR|0o755,stat.S_IFIFO|0o600])
def test_special_member_before_payload(tmp_path, monkeypatch, mode):
    p = fixture_archive(tmp_path,monkeypatch,tiny_zip(mode=mode))
    monkeypatch.setattr(zipfile.ZipFile, 'open', lambda *_: pytest.fail('unsafe payload read'))
    with pytest.raises(ValueError): m.qualify_archive(p,transfer())


def test_all_expansion_reserved_before_first_payload(tmp_path, monkeypatch):
    p = fixture_archive(tmp_path,monkeypatch); t = transfer(); t.cfg['max_total_expanded_bytes'] = 1
    monkeypatch.setattr(zipfile.ZipFile, 'open', lambda *_: pytest.fail('over-budget payload read'))
    with pytest.raises(ValueError): m.qualify_archive(p,t)
    assert t.expanded_reserved == 0


def test_crc_bad_original_preserved(tmp_path, monkeypatch):
    # Stored tiny member makes corruption exact, without relying on compressed byte patterns.
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        info=zipfile.ZipInfo('annotations/first.json'); info.external_attr=(stat.S_IFREG|0o644)<<16
        z.writestr(info,b'opaque',compress_type=zipfile.ZIP_STORED)
    damaged=stream.getvalue().replace(b'opaque',b'Opaque'); p=fixture_archive(tmp_path,monkeypatch,damaged)
    before=m.rt.identity(p)
    with pytest.raises(zipfile.BadZipFile): m.qualify_archive(p,transfer())
    assert m.rt.identity(p)==before


def test_expanded_member_limit_before_payload(tmp_path,monkeypatch):
    p=fixture_archive(tmp_path,monkeypatch);t=transfer();t.cfg['max_expanded_asset_bytes']=1
    monkeypatch.setattr(zipfile.ZipFile,'open',lambda *_:pytest.fail('overlarge payload read'))
    with pytest.raises(ValueError): m.qualify_archive(p,t)
    assert t.expanded_reserved==0


def test_frozen_catalogue_asset_count_source_and_budgets():
    assert len(m.OPAQUE)==9 and len(m.NOTICES)==6 and len(m.COCO_TEXTS)==2 and len(m.CATALOGUE)==6
    assert sum(r['bytes'] for r in m.NOTICES+m.COCO_TEXTS+m.OPAQUE+(m.ARCHIVE,)) < 300 << 20
    assert sum(r['expanded_bytes'] for r in m.CATALOGUE)==844813048 < 1 << 30
    assert all(r['git_blob_sha1'] and len(r['git_blob_sha1'])==40 for r in m.OPAQUE)
    assert m.ARCHIVE['sha256'] is None and m.PROTOCOL['workers']==1 and m.PROTOCOL['retry_count']==0
    for name,pin in m.HELPER_PINS.items():
        assert m.rt.identity(Path(__file__).resolve().parents[1]/name,readonly=False)==pin
    wrapper=Path(__file__).resolve().parents[1]/m.HELPERS[1]
    assert 'ulimit -v 8388608' in wrapper.read_text() and '615s' in wrapper.read_text() and '-I -B' in wrapper.read_text()


@pytest.mark.parametrize('fault', ['none','download','source','after_pass','late_publication'])
def test_fake_host_publication_source_posthash_and_no_leaks(tmp_path,monkeypatch,capsys,fault):
    out=tmp_path.resolve()/'out'; code=Path(m.__file__).resolve().parents[1]; before={'binding':{'fixture':True}}
    source_calls=[]
    def source(*_):
        source_calls.append(1)
        return {'changed':True} if fault=='source' and len(source_calls)>1 else deepcopy(before)
    monkeypatch.setattr(m,'DATA',out); monkeypatch.setattr(m,'source',source)
    uid=m.os.geteuid(); calls=[]
    def injected_uid(): calls.append(1); return 0 if len(calls)==1 else uid
    monkeypatch.setattr(m.os,'geteuid',injected_uid); monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setattr(m.os,'uname',lambda:type('Host',(),{'nodename':'world-reward-ncc-h100-02'})())
    monkeypatch.setenv('WR_CODE',str(code)); monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    def download(self,row,output):
        if fault=='download' and row==m.ARCHIVE: raise TimeoutError('secret=should-never-leak')
        raw=b'authoredopaque'; m.rt.write(output/row['file'],raw); self.received+=len(raw); return m.vg.pin(raw)
    monkeypatch.setattr(m.Transfer,'download',download); monkeypatch.setattr(m,'qualify_archive',lambda *_:[])
    if fault=='after_pass':
        class InterruptedReport(dict):
            def update(self,*args,**kwargs):
                super().update(*args,**kwargs)
                if kwargs.get('status')=='pass': raise TimeoutError('secret=should-never-leak')
        def injected_dict(*args,**kwargs):
            return InterruptedReport(*args,**kwargs) if 'source_binding' in kwargs else dict(*args,**kwargs)
        monkeypatch.setattr(m,'dict',injected_dict,raising=False)
    if fault=='late_publication':
        original=m.vg.publish_report
        monkeypatch.setattr(m.vg,'publish_report',lambda out,report,deadline,started=None:
            original(out,report,time.monotonic()-1,started=started))
    if fault!='none':
        with pytest.raises(ValueError,match='Closed metadata acquisition'): m.run()
        report=json.loads((out/'report.json').read_bytes())
        assert report['status']=='fail'
        if fault in ('download','after_pass'): assert report['error_type']=='TimeoutError'
        if fault=='source': assert report['postcheck_failed']
        if fault=='late_publication': assert report['publication_failed']
        assert 'secret' not in (out/'report.json').read_text()
    else:
        report=m.run(); assert report['status']=='pass' and report['git_blob_sha1_verified']
    assert report['source_rehashed_after']==(fault!='source') and report['outputs_sealed'] and report['notices_retained']
    assert report['output_directory_rechecked']==(fault!='source')
    assert not report['publisher_archive_sha256_verified'] and not report['annotation_values_consulted']
    assert not report['legal_certainty_claimed'] and not report['underlying_image_rights_verified']
    assert out.stat().st_mode&0o777==0o500 and all(p.stat().st_mode&0o777==0o400 for p in out.iterdir())
    assert len(capsys.readouterr().out)<1000


def test_late_same_fd_publication_fail_closed(tmp_path):
    out=tmp_path.resolve()/'out';out.mkdir(); report={'status':'pass','outputs_sealed':False}
    with pytest.raises(ValueError): m.vg.publish_report(out,report,time.monotonic()-1)
    assert json.loads((out/'report.json').read_bytes())['status']=='fail'
