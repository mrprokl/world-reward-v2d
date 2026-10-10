import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import form_hoi_external_acquire as f


def make_tar(path, members):
    with tarfile.open(path, 'w') as saved:
        for name, payload in members:
            info = tarfile.TarInfo(name); info.size = len(payload)
            saved.addfile(info, io.BytesIO(payload))
    return dict(archive_size=path.stat().st_size, member_count=len(members))


def test_pinned_existing_cohort_reserved_never_selected():
    root = Path(__file__).resolve().parents[1]
    raw = (root/f.PROTOCOL).read_bytes(); p = json.loads(raw)
    assert dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == f.PROTOCOL_PIN
    assert len(f.cohort(p, 'inventory_first')) == 1
    assert len(f.cohort(p, 'acquire_dev')) == 4
    assert all(r['split'] == 'development' for r in f.cohort(p, 'acquire_dev'))
    assert sum(r['archive_size'] for r in f.cohort(p, 'acquire_dev')) == 3261132800
    with pytest.raises(ValueError): f.cohort(p, 'reserved')


@pytest.mark.parametrize('name', ['../pose.npy', '/absolute.mp4', './videos/file', 'videos//file',
                                 'foo\\bar', 'x/../../bad', 'bad\nfile'])
def test_tar_names_fail_closed(name):
    with pytest.raises(ValueError, match='Unsafe'): f.safe_member(tarfile.TarInfo(name))


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_tar_nonregular_type_fail_closed(kind):
    m = tarfile.TarInfo('safe'); m.type = kind
    with pytest.raises(ValueError, match='Unsafe'): f.safe_member(m)


def test_inventory_first_does_not_require_guessed_rgb_layout(tmp_path):
    archive = tmp_path/'a.tar'
    row = make_tar(archive, [('unknown_layout/source.mp4', b'rgb'), ('hoi_metadata.yaml', b'object:\n  prompt: can\n')])
    rows = f.inventory(archive, row, time.monotonic()+10, qualify_rgb=False)
    assert len(rows) == 2
    with pytest.raises(ValueError, match='Exactly one'): f.inventory(archive, row, time.monotonic()+10)


def test_inventory_duplicate_or_file_ancestor_rejected(tmp_path):
    archive = tmp_path/'a.tar'
    row = make_tar(archive, [('videos/front_stereo_camera_left.mp4', b'rgb'), ('a', b'x'), ('a/b', b'x')])
    with pytest.raises(ValueError, match='ancestor'): f.inventory(archive, row, time.monotonic()+10)
    row = make_tar(archive, [('a', b'x'), ('a', b'x')])
    with pytest.raises(ValueError, match='Duplicate'): f.inventory(archive, row, time.monotonic()+10, qualify_rgb=False)


def test_inventory_only_reads_bounded_native_metadata_no_reference_payload(tmp_path):
    archive = tmp_path/'a.tar'; destination = tmp_path/'out'; destination.mkdir()
    row = make_tar(archive, [('s/videos/front_stereo_camera_left.mp4', b'rgb bytes'),
                           ('s/mhr_params_mv.pt', b'not a torch checkpoint'),
                           ('s/edex', b'not json either'),
                           ('s/hoi_metadata.yaml', b'object:\n  prompt: blue can\naction: lift can\n')])
    rows = f.inventory(archive, row, time.monotonic()+10)
    kept, meta = f.extract(archive, rows, destination, time.monotonic()+10, inventory_only=True)
    assert len(kept) == 1 and kept[0]['role'] == 'native_metadata'
    assert not tuple((destination/'inputs').iterdir())
    assert not (destination/'eval_private/s/edex').exists()
    assert not (destination/'eval_private/s/mhr_params_mv.pt').exists()
    assert meta[0]['safe_semantic_fields'] == [dict(path='object.prompt', value='blue can'), dict(path='action', value='lift can')]


def test_acquire_extract_single_rgb_opaque_private_references(tmp_path):
    archive = tmp_path/'a.tar'; destination = tmp_path/'out'; destination.mkdir()
    row = make_tar(archive, [('s/videos/front_stereo_camera_left.mp4', b'rgb bytes'),
                           ('s/videos/rear_stereo_camera_left.mp4', b'forbidden second rgb'),
                           ('s/object_masks/front_stereo_camera_left.h5', b'opaque mask'),
                           ('s/mhr_params_mv.pt', b'opaque checkpoint'),
                           ('s/edex', b'opaque calib'), ('s/ground_plane.json', b'not needed')])
    rows = f.inventory(archive, row, time.monotonic()+10)
    kept, meta = f.extract(archive, rows, destination, time.monotonic()+10)
    assert (destination/'inputs/rgb.mp4').read_bytes() == b'rgb bytes'
    assert len(list((destination/'inputs').iterdir())) == 1
    assert {r['role'] for r in kept} == {'rgb', 'reference'} and not meta
    assert len(kept) == 4 and not (destination/'eval_private/s/ground_plane.json').exists()
    assert (destination/'eval_private').stat().st_mode & 0o777 == 0o700


def test_metadata_safe_fields_do_not_publish_calibration_pose_bbox_depth():
    raw = json.dumps(dict(object=dict(prompt='blue can', id='obj42', bbox=[1,2,3,4]),
                          action='lift', person_id='p22', fps=30,
                          poses=dict(description='pose data', t=[0,0,0]),
                          calibration=dict(camera=dict(width=1536)), ground_plane=[0,1,0,0])).encode()
    result = f.metadata_schema(raw, 'hoi_metadata.json')
    assert {r['path'] for r in result['safe_semantic_fields']} == {'object.prompt','object.id','action','person_id','fps'}
    assert result['schema']['object']['bbox']['type'] == 'list'
    assert result['reference_values_published'] is False
    yaml = b'object:\n  prompt: "blue can"\n  bbox: [1,2,3,4]\nposes:\n  description: private\ncalibration:\n  camera:\n    width: 1536\naction: !unsafe constructor\n'
    assert f.metadata_schema(yaml, 'metadata.yaml')['safe_semantic_fields'] == [dict(path='object.prompt',value='blue can')]
    with pytest.raises(ValueError, match='Duplicate'): f.metadata_schema(b'{"action":"lift","action":"lower"}', 'm.json')


@pytest.mark.parametrize('url', ['http://huggingface.co/foo','https://huggingface.co.evil/foo',
                               'https://huggingface.co@evil/foo','https://evil.hf.co.evil/foo',
                               'https://huggingface.co:444/foo'])
def test_url_allowlist(url):
    with pytest.raises(ValueError, match='publisher'): f.public_url(url)


class Response(io.BytesIO):
    def __init__(self, payload, url, status=200, headers=None):
        super().__init__(payload); self.url = url; self.status = status; self.headers = headers or {}
    def geturl(self): return self.url


class Opener:
    def __init__(self, payloads): self.payloads = iter(payloads); self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, dict(request.header_items()), timeout))
        payload, status, headers = next(self.payloads)
        return Response(payload, request.full_url, status, headers)


def test_download_exact_publisher_hash_partial_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(f.time, 'sleep', lambda _: None)
    payload = b'archive bytes'; row = dict(archive_path='data/test.tar', archive_size=len(payload), archive_sha256=hashlib.sha256(payload).hexdigest())
    opener = Opener([(payload[:4], 200, {}), (payload[4:], 206, {'Content-Range':f'bytes 4-{len(payload)-1}/{len(payload)}'})])
    output = tmp_path/'source.tar'
    receipt = f.fetch(row, 'a'*40, output, opener, time.monotonic()+10)
    assert output.read_bytes() == payload and receipt['attempts'] == 2
    assert opener.calls[1][1]['Range'] == 'bytes=4-'
    assert not (tmp_path/'source.tar.part').exists()


def test_download_hash_or_range_mismatch_cleans_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(f.time, 'sleep', lambda _: None)
    payload = b'archive bytes'; row = dict(archive_path='data/test.tar', archive_size=len(payload), archive_sha256='a'*64)
    with pytest.raises(ValueError, match='SHA'):
        f.fetch(row, 'a'*40, tmp_path/'source.tar', Opener([(payload, 200, {})]), time.monotonic()+10)
    assert not tuple(tmp_path.iterdir())
    row['archive_sha256'] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='Content-Range'):
        f.fetch(row, 'a'*40, tmp_path/'source.tar', Opener([(payload[:4],200,{}),(payload[4:],206,{})]),time.monotonic()+10)
    assert not tuple(tmp_path.iterdir())


def test_receipt_atomic_readonly_collision(tmp_path):
    path = tmp_path/'receipt.json'; pin = f.seal(path, dict(stage='inventory'))
    assert f.identity(path, 1000) == pin and not path.stat().st_mode & 0o222
    with pytest.raises(ValueError, match='collision'): f.seal(path, {})
