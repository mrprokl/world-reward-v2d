"""Tiny authored ZIP metadata controls; no datasets, remote archive or media."""
import io
import json
from pathlib import Path
import struct
import zipfile

import pytest

from zip_inventory import parse_directory, read_directory
from mmhoi_inventory import Ranges


def archive():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('MMHOI/', b'')
        z.writestr('MMHOI/README.md', b'authored metadata test')
        z.writestr('MMHOI/sequences/session/clip/0_00001.jpg', b'not a real JPEG')
    return stream.getvalue()


def test_only_trailer_and_complete_central_directory_requested():
    raw = archive(); reads = []
    def get(a, b):
        reads.append((a, b)); return raw[a:b + 1]
    central, layout = read_directory(get, len(raw), 4096, 10)
    rows = parse_directory(central, layout)
    assert len(reads) == 2
    assert reads[0] == (len(raw) - 22, len(raw) - 1)
    assert reads[1] == (layout['offset'], layout['offset'] + layout['bytes'] - 1)
    assert [r['name'] for r in rows] == ['MMHOI/', 'MMHOI/README.md', 'MMHOI/sequences/session/clip/0_00001.jpg']
    assert rows[0]['directory'] and rows[1]['bytes'] == 22
    assert all(b >= layout['offset'] for a, b in reads)


def test_budget_failure_before_central_body_read():
    raw = archive(); reads = []
    def get(a, b):
        reads.append((a, b)); return raw[a:b + 1]
    with pytest.raises(ValueError, match='bounds exceeded'):
        read_directory(get, len(raw), 46, 10)
    assert reads == [(len(raw) - 22, len(raw) - 1)]


def test_comment_rejected_without_tail_scan_or_member_read():
    raw = bytearray(archive()); raw[-2:] = b'\x01\x00'
    with pytest.raises(ValueError, match='comment-free'):
        read_directory(lambda a,b:bytes(raw[a:b+1]), len(raw), 4096, 10)


def test_zip64_trailer_with_sparse_virtual_archive():
    # Authored sparse address space: no GB allocation or native archive needed.
    central = struct.pack('<4s6H3I5H2I', b'PK\x01\x02', 45, 45, 0, 0, 0, 0, 0,
                          0, 0, 1, 0, 0, 0, 0, 0, 0) + b'x'
    offset = 1 << 33
    where = offset + len(central)
    record = struct.pack('<4sQ2H2I4Q', b'PK\x06\x06', 44, 45, 45, 0, 0, 1, 1, len(central), offset)
    locator = struct.pack('<4sIQI', b'PK\x06\x07', 0, where, 1)
    end = where + 76
    eocd = struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0, 65535, 65535, 4294967295, 4294967295, 0)
    wanted = {(end,end+21):eocd, (end-20,end-1):locator,
              (where,where+55):record, (offset,offset+len(central)-1):central}
    raw, layout = read_directory(lambda a,b:wanted[(a,b)], end+22, 4096, 10)
    assert layout['zip64'] is True and parse_directory(raw,layout)[0]['name'] == 'x'


@pytest.mark.parametrize('filename', ['../x', '/x', 'a//b', 'a\\b', 'a/./b', 'a\x00b'])
def test_unsafe_metadata_names_rejected_without_extracting(filename):
    name = filename.encode()
    raw = struct.pack('<4s6H3I5H2I', b'PK\x01\x02', 20,20,0,0,0,0,0,0,0,
                      len(name),0,0,0,0,0,0) + name
    with pytest.raises(ValueError, match='Unsafe'):
        parse_directory(raw, dict(bytes=len(raw),members=1,offset=1000))


def config():
    return json.loads(Path('configs/mmhoi_inventory_v1.json').read_bytes())


class Response:
    def __init__(self, cfg, start, end, *, status=206, modifications=None):
        self.status, self.reads, self.cfg = status, [], cfg
        self.headers = {'Content-Range':f'bytes {start}-{end}/{cfg["archive_bytes"]}',
                        'Content-Length':str(end-start+1), 'Last-Modified':cfg['last_modified']}
        self.headers.update(modifications or {})
    def geturl(self): return self.cfg['archive_url']
    def read(self, n): self.reads.append(n); return b'x' * (n-1)
    def __enter__(self): return self
    def __exit__(self, *args): return False


class Opener:
    def __init__(self, response): self.response, self.calls = response, []
    def open(self, request, timeout): self.calls.append((request,timeout)); return self.response


@pytest.mark.parametrize('status,modifications', [
    (200, {}), (206, {'Content-Range':'bytes 0-21/123'}),
    (206, {'Last-Modified':'different'}), (206, {'Content-Encoding':'gzip'}),
    (206, {'Content-Length':'999999999'}),
])
def test_http_identity_firewall_fails_before_any_body_read(status, modifications):
    cfg = config(); ranges = Ranges(cfg, float('inf'))
    response = Response(cfg, cfg['archive_bytes']-22, cfg['archive_bytes']-1,
                        status=status, modifications=modifications)
    ranges.opener = Opener(response)
    with pytest.raises(ValueError, match='firewall'):
        ranges.get(cfg['archive_bytes']-22, cfg['archive_bytes']-1)
    assert response.reads == [] and ranges.bytes == 0 and ranges.proofs == []
    assert ranges.rejected_headers['body_read'] is False


def test_exact206_range_is_bounded_and_pinned():
    cfg = config(); ranges = Ranges(cfg, float('inf'))
    response = Response(cfg, cfg['archive_bytes']-22, cfg['archive_bytes']-1)
    ranges.opener = Opener(response)
    raw = ranges.get(cfg['archive_bytes']-22, cfg['archive_bytes']-1)
    assert raw == b'x'*22 and response.reads == [23]
    assert ranges.bytes == 22 and len(ranges.proofs) == 1
    req, _ = ranges.opener.calls[0]
    assert req.get_header('Range') == f'bytes={cfg["archive_bytes"]-22}-{cfg["archive_bytes"]-1}'
    assert req.get_header('If-unmodified-since') == cfg['last_modified']
