"""Manufactured ZIP/HTTP/source contracts only; no publisher asset or JSON rows."""
from copy import deepcopy
from email.message import Message
import io
import json
from pathlib import Path
import stat
import time
import zipfile

import pytest
import visual_genome_metadata_acquire as m


def cfg():
    return json.loads((Path(__file__).resolve().parents[1]/m.CONFIG).read_text())


def archive(name='objects.json', data=b'[opaque metadata; not parsed]', compression=zipfile.ZIP_DEFLATED,
            mode=stat.S_IFREG | 0o644, second=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as z:
        info = zipfile.ZipInfo(name); info.external_attr = mode << 16
        z.writestr(info, data, compress_type=compression)
        if second: z.writestr(second, b'never accepted')
    raw = output.getvalue()
    # ZipInfo's writer itself sanitizes NUL names; inject one into both saved headers.
    if '\x00' in name: raw = raw.replace(b'objects.json', b'objects.jso\x00')
    return raw


def stored(tmp_path, raw):
    p = tmp_path.resolve()/'original.zip'; p.write_bytes(raw); p.chmod(0o400); return p


def row(raw, member='objects.json'):
    return dict(url=m.PREFIX+'data/dataset/objects.json.zip', file='objects.json.zip',
                member=member, observed_bytes=len(raw), version='1.4')


class Response(io.BytesIO):
    def __init__(self, raw, item, *, status=200, url=None, length=None, kind='application/zip', encoding='identity'):
        super().__init__(raw); self.status = status; self.url = url or item['url']; self.read_calls = 0
        self.headers = Message(); self.headers['Content-Type'] = kind
        self.headers['Content-Length'] = str(len(raw) if length is None else length)
        self.headers['Content-Encoding'] = encoding

    def geturl(self): return self.url

    def read(self, size=-1):
        self.read_calls += 1; return super().read(size)


class Opener:
    def __init__(self, response): self.response = response; self.calls = []
    def open(self, request, timeout):
        self.calls.append((request, timeout)); return self.response


@pytest.mark.parametrize('compression', [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_crc_stream_not_json_parsing_or_extraction(tmp_path, monkeypatch, compression):
    raw = archive(compression=compression); p = stored(tmp_path, raw)
    monkeypatch.setattr(json, 'loads', lambda *_: pytest.fail('metadata values parsed'))
    result = m.qualify_zip(p, row(raw), cfg_without_json(), time.monotonic()+30)
    assert result['expanded_bytes'] == len(b'[opaque metadata; not parsed]')
    assert result['crc_stream_verified'] and not result['json_values_consulted']
    assert not result['expanded_file_written'] and list(tmp_path.iterdir()) == [p]


def cfg_without_json():
    return dict(max_expanded_asset_bytes=1 << 30, max_total_expanded_bytes=2 << 30)


@pytest.mark.parametrize('name', ['../objects.json', '/objects.json', 'folder/objects.json',
                                   'objects.json/', 'objects.json\x00tail', 'other.json', '..\\objects.json'])
def test_unexpected_member_name_rejected_before_payload(tmp_path, monkeypatch, name):
    raw = archive(name=name); p = stored(tmp_path, raw)
    monkeypatch.setattr(zipfile.ZipFile, 'open', lambda *_: pytest.fail('unsafe member read'))
    with pytest.raises(ValueError): m.qualify_zip(p, row(raw), cfg(), time.monotonic()+30)


@pytest.mark.parametrize('mode', [stat.S_IFLNK|0o777, stat.S_IFIFO|0o600, stat.S_IFDIR|0o755, stat.S_IFCHR|0o600])
def test_special_member_rejected(tmp_path, mode):
    raw = archive(mode=mode)
    with pytest.raises(ValueError): m.qualify_zip(stored(tmp_path, raw), row(raw), cfg(), time.monotonic()+30)


def test_second_member_rejected(tmp_path):
    raw = archive(second='__MACOSX/objects.json')
    with pytest.raises(ValueError): m.qualify_zip(stored(tmp_path, raw), row(raw), cfg(), time.monotonic()+30)


def test_crc_corruption_preserves_original_archive(tmp_path):
    raw = archive(compression=zipfile.ZIP_STORED); raw = raw.replace(b'opaque', b'Opaque'); p = stored(tmp_path, raw)
    before = m.rt.identity(p)
    with pytest.raises(zipfile.BadZipFile): m.qualify_zip(p, row(raw), cfg(), time.monotonic()+30)
    assert m.rt.identity(p) == before


def test_expanded_asset_limit_before_payload(tmp_path, monkeypatch):
    raw = archive(data=b'x'*100); config = cfg(); config['max_expanded_asset_bytes'] = 99
    monkeypatch.setattr(zipfile.ZipFile, 'open', lambda *_: pytest.fail('too-large member read'))
    with pytest.raises(ValueError): m.qualify_zip(stored(tmp_path, raw), row(raw), config, time.monotonic()+30)


def test_total_expanded_reservation_before_second_payload(tmp_path):
    raw = archive(data=b'x'*100); config = cfg(); config['max_total_expanded_bytes'] = 150
    transfer = m.Transfer(config, time.monotonic()+30, Opener(None)); p = stored(tmp_path, raw)
    m.qualify_zip(p, row(raw), config, transfer.deadline, reserve=transfer.reserve_expansion)
    with pytest.raises(ValueError): m.qualify_zip(p, row(raw), config, transfer.deadline, reserve=transfer.reserve_expansion)
    assert transfer.expanded_reserved == 100


def test_exact_https_stream_original_bytes_and_sha(tmp_path):
    raw = archive(); item = row(raw); response = Response(raw, item); opener = Opener(response)
    transfer = m.Transfer(cfg(), time.monotonic()+30, opener)
    identity = transfer.download(item, tmp_path.resolve())
    assert identity == m.pin(raw) == m.rt.identity(tmp_path.resolve()/item['file'])
    assert len(opener.calls) == 1 and opener.calls[0][1] <= 15
    assert opener.calls[0][0].get_header('Accept-encoding') == 'identity'
    assert (tmp_path/item['file']).stat().st_mode & 0o777 == 0o400
    assert transfer.received == len(raw)


@pytest.mark.parametrize('change', [dict(status=206), dict(url='https://other.example/objects.zip'),
    dict(length=1), dict(kind='image/jpeg'), dict(encoding='gzip')])
def test_response_firewall_before_any_body_or_file(tmp_path, change):
    raw = archive(); item = row(raw); response = Response(raw, item, **change)
    with pytest.raises(ValueError): m.Transfer(cfg(), time.monotonic()+30, Opener(response)).download(item, tmp_path.resolve())
    assert response.read_calls == 0 and not list(tmp_path.iterdir())


def test_pinned_publisher_text_is_not_classifier_only(tmp_path):
    raw = b'<html>genuine pinned publisher notice</html>'; item = dict(file='notice.html', url=m.PREFIX+'about.html', **m.pin(raw))
    response = Response(raw, item, kind='text/html'); transfer = m.Transfer(cfg(), time.monotonic()+30, Opener(response))
    assert transfer.download(item, tmp_path.resolve(), text=True) == m.pin(raw)


def test_changed_text_or_short_body_owned_leaf_removed(tmp_path):
    raw = b'changed'; item = dict(file='notice.html', url=m.PREFIX+'about.html', bytes=len(raw), sha256='0'*64)
    with pytest.raises(ValueError): m.Transfer(cfg(), time.monotonic()+30,
        Opener(Response(raw, item, kind='text/html'))).download(item, tmp_path.resolve(), text=True)
    assert not list(tmp_path.iterdir())


def test_overlong_stream_not_saved(tmp_path):
    raw = archive(); item = row(raw); item['observed_bytes'] -= 1
    with pytest.raises(ValueError): m.Transfer(cfg(), time.monotonic()+30,
        Opener(Response(raw, item, length=item['observed_bytes']))).download(item, tmp_path.resolve())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('foreign', ['file','symlink'])
def test_never_overwrite_or_unlink_foreign_output(tmp_path, foreign):
    raw = archive(); item = row(raw); p = tmp_path/item['file']; target = tmp_path/'foreign'
    target.write_bytes(b'userwork')
    if foreign == 'symlink': p.symlink_to(target)
    else: p.write_bytes(b'existing')
    with pytest.raises((ValueError, FileExistsError)):
        m.Transfer(cfg(), time.monotonic()+30, Opener(Response(raw,item))).download(item,tmp_path.resolve())
    assert p.exists() and target.read_bytes() == b'userwork'


def test_owned_directory_fsync_failure_removes_only_owned_leaf(tmp_path, monkeypatch):
    raw = archive(); item = row(raw); foreign = tmp_path/'userwork'; foreign.write_bytes(b'preserve')
    monkeypatch.setattr(m, 'sync', lambda *_: (_ for _ in ()).throw(OSError('manufactured fsync failure')))
    with pytest.raises(OSError): m.Transfer(cfg(),time.monotonic()+30,Opener(Response(raw,item))).download(item,tmp_path.resolve())
    assert not (tmp_path/item['file']).exists() and foreign.read_bytes() == b'preserve'


def test_total_download_budget_no_retry(tmp_path):
    raw = archive(); item = row(raw); config = cfg(); config['max_total_compressed_bytes'] = len(raw)-1
    opener = Opener(Response(raw,item))
    with pytest.raises(ValueError): m.Transfer(config,time.monotonic()+30,opener).download(item,tmp_path.resolve())
    assert len(opener.calls) == 1 and not list(tmp_path.iterdir())


def test_cancelled_or_deadline_before_request(tmp_path):
    raw = archive(); opener = Opener(Response(raw,row(raw))); transfer = m.Transfer(cfg(),time.monotonic()-1,opener)
    with pytest.raises(ValueError): transfer.download(row(raw),tmp_path.resolve())
    transfer.deadline = time.monotonic()+30; transfer.cancelled.set()
    with pytest.raises(ValueError): transfer.download(row(raw),tmp_path.resolve())
    assert not opener.calls


def test_no_redirect_handler_never_falls_back():
    with pytest.raises(ValueError): m.NoRedirect().redirect_request(None,None,302,None,None,None)


def test_exact_config_sizes_versions_and_helper_pin():
    config = cfg(); assert config['assets'] == list(m.ASSETS) and config['publisher_texts'] == list(m.TEXTS)
    assert sum(r['observed_bytes'] for r in m.ASSETS)+sum(r['bytes'] for r in m.TEXTS) < config['max_total_compressed_bytes']
    assert m.rt.identity(Path(m.rt.__file__).resolve(),readonly=False) == config['source_helper_pin']
    assert set(m.HELPERS) == {m.CONFIG,'infra/visual_genome_metadata_acquire.py',
        'infra/run_visual_genome_metadata_acquire.sh','infra/mediapipe_cpu_runtime_verify.py'}


def test_config_numeric_bool_substitution_rejected(tmp_path, monkeypatch):
    config = cfg(); config['retry_count'] = False
    monkeypatch.setattr(m.rt,'pinned',lambda *_:config)
    with pytest.raises(ValueError): m.config(tmp_path,dict(binding=dict(helpers={m.CONFIG:{}})))


@pytest.mark.parametrize('interrupt_after_pass', [False, True])
def test_complete_fake_host_gate_success_and_seals(tmp_path, monkeypatch, capsys, interrupt_after_pass):
    """Injected source/host/HTTP only; never claims this is an actual Azure receipt."""
    config = cfg(); raw = archive(data=b'opaque-no-JSON'); texts = list(m.TEXTS)
    assets = [dict(row(raw),file=f'asset_{i}.zip') for i in range(3)]
    out = tmp_path.resolve()/'output'; before = dict(binding=dict(helpers={m.CONFIG:dict(bytes=1,sha256='0'*64)}))
    monkeypatch.setattr(m,'DATA',out); monkeypatch.setattr(m,'source',lambda *_:deepcopy(before))
    monkeypatch.setattr(m,'config',lambda *_:config); monkeypatch.setattr(m,'ASSETS',assets)
    actual_uid = m.os.geteuid(); calls = []
    def injected_uid():
        calls.append(1); return 0 if len(calls) == 1 else actual_uid
    monkeypatch.setattr(m.os,'geteuid',injected_uid); monkeypatch.setattr(m.sys,'platform','linux')
    monkeypatch.setattr(m.os,'uname',lambda:type('Host',(),{'nodename':'world-reward-ncc-h100-02'})())
    monkeypatch.setenv('WR_CODE',str(Path(m.__file__).resolve().parents[1])); monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    def fake_download(self, item, output, text=False):
        data = b'<html>source proof</html>' if text else raw
        m.rt.write(output/item['file'],data); self.received += len(data); return m.pin(data)
    monkeypatch.setattr(m.Transfer,'download',fake_download)
    if interrupt_after_pass:
        class InterruptedReport(dict):
            def update(self, *args, **kwargs):
                super().update(*args, **kwargs)
                if kwargs.get('status') == 'pass':
                    raise TimeoutError('manufactured interrupt immediately after provisional PASS')
        def injected_dict(*args, **kwargs):
            return InterruptedReport(*args, **kwargs) if 'source_binding' in kwargs else dict(*args, **kwargs)
        monkeypatch.setattr(m,'dict',injected_dict,raising=False)
        with pytest.raises(ValueError,match='Closed metadata acquisition'):
            m.run()
        report = json.loads((out/'report.json').read_bytes())
        assert report['status'] == 'fail' and report['error_type'] == 'TimeoutError'
        assert report['decision'] == 'CLOSED_METADATA_ACQUISITION_NO_RGB_OR_RETRY'
    else:
        report = m.run()
        assert report['status'] == 'pass'
    assert report['source_rehashed_after'] and report['outputs_sealed']
    assert not report['annotation_values_consulted'] and not report['publisher_archive_sha256_verified']
    assert {p.name for p in out.iterdir()} == {r['file'] for r in (*texts,*assets)}|{'report.json'}
    assert out.stat().st_mode & 0o777 == 0o500
    assert all(p.stat().st_mode & 0o777 == 0o400 for p in out.iterdir())
    assert len(capsys.readouterr().out) < 1000


def test_late_report_publication_deadline_demotes_owned_receipt(tmp_path):
    out = tmp_path.resolve()/'out'; out.mkdir()
    report = dict(status='pass', outputs_sealed=False)
    with pytest.raises(ValueError): m.publish_report(out,report,time.monotonic()-1)
    actual = json.loads((out/'report.json').read_bytes())
    assert actual['status'] == 'fail' and actual['publication_failed'] and actual['outputs_sealed']
    assert out.stat().st_mode & 0o777 == 0o500
    assert (out/'report.json').stat().st_mode & 0o777 == 0o400


def test_report_fsync_failure_no_false_pass(tmp_path,monkeypatch):
    out = tmp_path.resolve()/'out'; out.mkdir(); report = dict(status='pass',outputs_sealed=False)
    monkeypatch.setattr(m,'sync',lambda *_:(_ for _ in ()).throw(OSError('controlled directory fsync failure')))
    with pytest.raises(OSError): m.publish_report(out,report,time.monotonic()+30)
    assert json.loads((out/'report.json').read_bytes())['status'] == 'fail'
