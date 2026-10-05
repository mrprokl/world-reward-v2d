"""Tiny authored ZIPs and mock HTTPS only; no datasets/labels/model execution."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import zipfile

import pytest


@pytest.fixture
def hocap():
    p = Path(__file__).parents[1]/'infra/hocap_acquire.py'
    spec = importlib.util.spec_from_file_location('hocap_test', p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def pin(raw): return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def zip_bytes(rows):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for n, data in rows:
            archive.writestr(n, data)
    return out.getvalue()


def protocol():
    return json.loads((Path(__file__).parents[1]/'configs/hocap_acquisition_protocol_v1.json').read_bytes())


class Response(io.BytesIO):
    def __init__(self, raw, url, status=200, length=None):
        super().__init__(raw); self.status = status; self.url = url
        self.headers = {'Content-Length': str(len(raw)) if length is None else length}
    def geturl(self): return self.url


class Opener:
    def __init__(self, sources): self.sources = sources; self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        return Response(self.sources[request.full_url], request.full_url)


def manufacture(hocap, tmp_path, monkeypatch):
    p = protocol(); p['minimum_free_bytes'] = 1
    p['primary_sources'] = {'publisher.html': dict(url='https://irvlutd.github.io/HOCap/',
        **pin(b'Creative Commons Attribution 4.0 International License (CC BY 4.0) https://creativecommons.org/licenses/by/4.0/'))}
    sources = {p['primary_sources']['publisher.html']['url']: b'Creative Commons Attribution 4.0 International License (CC BY 4.0) https://creativecommons.org/licenses/by/4.0/'}
    rows = []
    for clip in p['clips']:
        base = p['subject']+'/'+clip+'/'
        rows.append((base+'meta.yaml', b'num_frames: 2\nobject_ids: [PRIVATE]\nmano_sides: PRIVATE\n'))
        for i in range(2):
            rows.append((base+p['camera']+f'/color_{i:06d}.jpg', b'manufactured RGB bytes '+bytes([i])))
            rows.append((base+p['camera']+f'/depth_{i:06d}.png', b'NEVER OPEN DEPTH'))
    for row in p['archives']:
        raw = zip_bytes(rows if row['file'] == 'subject_5.zip' else [(row['file'][:-4]+'/opaque.npz', b'NEVER PARSE REFERENCE')])
        row['bytes'] = len(raw); sources[row['url']] = raw
    p['total_archive_bytes'] = sum(row['bytes'] for row in p['archives'])
    source = dict(closure_sha256='a'*64, producer_revision='b'*40)
    monkeypatch.setattr(hocap, 'source_binding', lambda *_: (copy.deepcopy(p), copy.deepcopy(source)))
    return p, sources, source


def test_frozen_exact_primary_urls_lengths_selectors_and_claims(hocap):
    p = protocol(); raw = (Path(__file__).parents[1]/hocap.PROTOCOL).read_bytes()
    assert pin(raw) == hocap.PROTOCOL_PIN
    assert p['clips'] == ['20231027_112303', '20231027_113202'] and p['camera'] == '105322251564'
    assert p['total_archive_bytes'] == sum(r['bytes'] for r in p['archives']) == 5080226520
    assert len(p['archives']) == 5 and p['dataset_license'] == 'CC-BY-4.0'
    assert p['raw_metadata_public'] is p['predictor_source_calibration_allowed'] is p['model_loading_allowed'] is False
    assert p['primary_sources']['publisher.html']['bytes'] == 29538
    assert p['toolkit_revision'] == '576c63ebf3b84dfec8744ba0f021234213bf0dab'


def test_full_mocked_flow_public_rgb_only_and_opaque_quarantine(hocap, tmp_path, monkeypatch):
    p, sources, _ = manufacture(hocap, tmp_path, monkeypatch); target = tmp_path/'fresh'; opener = Opener(sources)
    result = hocap.acquire(None, tmp_path, 'b'*40, target, opener=opener)
    assert result['status'] == 'pass' and result['download_complete'] and result['public_inventory_qualified']
    assert result['frames'] == 4 and len(opener.calls) == 6
    public = json.loads((target/'inputs/manifest.json').read_bytes())
    assert public['annotations_public'] is public['calibration_public'] is public['raw_metadata_public'] is False
    assert len(public['images']) == 4 and [r['source_frame_id'] for r in public['images']] == [0, 1, 0, 1]
    assert 'PRIVATE' not in json.dumps(public)
    assert not list((target/'inputs').rglob('*.yaml')) and not list((target/'inputs').rglob('*.npz'))
    assert (target/'quarantine').stat().st_mode & 0o777 == 0o700
    assert (target/'report.json').stat().st_mode & 0o777 == 0o400
    assert all((target/'quarantine'/r['file']).stat().st_mode & 0o777 == 0o400 for r in p['archives'])
    assert all((target/'inputs'/r['file']).stat().st_mode & 0o777 == 0o444 for r in public['images'])
    assert (target/'inputs').stat().st_mode & 0o777 == 0o555
    assert all((target/'inputs'/r['clip']).stat().st_mode & 0o777 == 0o555 for r in public['clips'])
    assert len(result['opaque_metadata']) == 2 and result['archive_hash_basis'] == 'measured_not_publisher_checksum'
    for k in ('label_values_parsed', 'calibration_values_parsed', 'model_loaded', 'gpu_used', 'challenge_inputs_used', 'adoption'):
        assert result[k] is False


def test_failed_layout_retains_original_download_evidence_not_qualified_public(hocap, tmp_path, monkeypatch):
    p, sources, _ = manufacture(hocap, tmp_path, monkeypatch)
    row = p['archives'][0]; raw = zip_bytes([('unexpected/wrapper/meta.yaml', b'num_frames: 2\n')]); row['bytes'] = len(raw); sources[row['url']] = raw
    monkeypatch.setattr(hocap, 'source_binding', lambda *_: (p, {'closure_sha256':'a'*64}))
    result = hocap.acquire(None, tmp_path, 'b'*40, tmp_path/'failed', opener=Opener(sources))
    assert result['status'] == 'fail' and result['download_complete'] is True
    assert result['public_inventory_qualified'] is False and result['phase'] == 'public_extract'
    assert len(result['archives']) == 5 and (tmp_path/'failed/quarantine/subject_5.zip').read_bytes() == raw


def test_primary_source_mismatch_prevents_archive_downloads(hocap, tmp_path, monkeypatch):
    p, sources, _ = manufacture(hocap, tmp_path, monkeypatch)
    sources[p['primary_sources']['publisher.html']['url']] = b'x'*p['primary_sources']['publisher.html']['bytes']
    opener = Opener(sources)
    result = hocap.acquire(None, tmp_path, 'b'*40, tmp_path/'failed', opener=opener)
    assert result['status'] == 'fail' and result['phase'] == 'primary_sources'
    assert len(opener.calls) == 1 and result['archives'] == {}


def test_missing_frame_cleans_public_but_retains_prehashed_opaque_metadata(hocap, tmp_path, monkeypatch):
    p, sources, _ = manufacture(hocap, tmp_path, monkeypatch)
    row = p['archives'][0]; clip = p['clips'][0]; rawmeta = b'num_frames: 2\nignored: PRIVATE\n'
    raw = zip_bytes([(f"subject_5/{clip}/meta.yaml", rawmeta),
        (f"subject_5/{clip}/{p['camera']}/color_000000.jpg", b'manufactured')])
    row['bytes'] = len(raw); sources[row['url']] = raw
    monkeypatch.setattr(hocap, 'source_binding', lambda *_: (p, {'closure_sha256':'a'*64}))
    result = hocap.acquire(None, tmp_path, 'b'*40, tmp_path/'failed', opener=Opener(sources))
    assert result['status'] == 'fail' and result['public_inventory_qualified'] is False
    assert not tuple((tmp_path/'failed/inputs').iterdir())
    assert result['opaque_metadata'] == {clip+'-meta.yaml': pin(rawmeta)}
    assert result['source_archive_public_rehashed_after'] is True


def test_inactive_labels_depth_pose_and_models_are_never_opened(hocap, tmp_path, monkeypatch):
    _, sources, _ = manufacture(hocap, tmp_path, monkeypatch); original = zipfile.ZipFile.open; opened = []
    def permitted(self, member, *a, **k):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        assert name.endswith('meta.yaml') or '/color_' in name
        opened.append(name); return original(self, member, *a, **k)
    monkeypatch.setattr(zipfile.ZipFile, 'open', permitted)
    result = hocap.acquire(None, tmp_path, 'b'*40, tmp_path/'fresh', opener=Opener(sources))
    assert result['status'] == 'pass' and len(opened) == 6


def test_posthash_detects_tampered_original_archive_without_public_qualification(hocap, tmp_path, monkeypatch):
    _, sources, _ = manufacture(hocap, tmp_path, monkeypatch); original = hocap.identity
    def tampered(path, maximum):
        value = original(path, maximum)
        if Path(path).name == 'subject_5.zip': return {**value, 'sha256':'0'*64}
        return value
    monkeypatch.setattr(hocap, 'identity', tampered)
    result = hocap.acquire(None, tmp_path, 'b'*40, tmp_path/'fresh', opener=Opener(sources))
    assert result['status'] == 'fail' and result['post_error_type'] == 'ValueError'
    assert result.get('source_archive_public_rehashed_after') is not True


@pytest.mark.parametrize('bad', ['num_frames: 0\n', 'num_frames: 2\nnum_frames: 3\n', '  num_frames: 2\n', 'num_frames: 2.0\n', 'num_frames: !!int 2\n', 'num_frames: 100001\n', 'num_frames: 2\n"num_frames": 3\n'])
def test_only_plain_top_level_positive_numeric_metadata_constructed(hocap, bad):
    with pytest.raises(ValueError): hocap.numeric_frames(bad.encode(), 100000)
    assert hocap.numeric_frames(b'num_frames: 24 # primary count\nunknown: !!python/object:BAD\n', 100000) == 24


@pytest.mark.parametrize('name', ['/absolute', '../escape', 'a/../escape', 'a//b', 'a\\b', '.', 'a\x7fb'])
def test_zip_paths_aliases_and_special_members_fail_before_payload(hocap, tmp_path, name):
    path = tmp_path/'bad.zip'; path.write_bytes(zip_bytes([(name, b'opaque')]))
    with pytest.raises(ValueError): hocap.inventory(path, protocol(), 10**12)


@pytest.mark.parametrize('kind', [stat.S_IFLNK, stat.S_IFCHR, stat.S_IFIFO])
def test_zip_type_and_duplicate_rejection(hocap, tmp_path, kind):
    item = zipfile.ZipInfo('link'); item.create_system = 3; item.external_attr = (kind | 0o777) << 16
    path = tmp_path/'bad.zip'; path.write_bytes(zip_bytes([(item, b'opaque')]))
    with pytest.raises(ValueError): hocap.inventory(path, protocol(), 10**12)


def test_zip_duplicate_and_ancestor_collision(hocap, tmp_path):
    for rows in ([('a', b'one'), ('a', b'two')], [('a', b'file'), ('a/child', b'opaque')]):
        path = tmp_path/'bad.zip'
        with pytest.warns(UserWarning) if len(rows) == 2 and rows[0][0] == rows[1][0] else __import__('contextlib').nullcontext(): path.write_bytes(zip_bytes(rows))
        with pytest.raises(ValueError): hocap.inventory(path, protocol(), 10**12)


@pytest.mark.parametrize('key', ['maximum_zip_members', 'maximum_member_bytes', 'maximum_expanded_bytes'])
def test_inventory_limits_before_any_member_read(hocap, tmp_path, monkeypatch, key):
    path = tmp_path/'safe.zip'; path.write_bytes(zip_bytes([('one', b'abc'), ('two', b'abc')]))
    p = protocol(); p[key] = 1
    monkeypatch.setattr(zipfile.ZipFile, 'open', lambda *_a, **_k: pytest.fail('payload must not be read during inventory'))
    with pytest.raises(ValueError): hocap.inventory(path, p, 10**12)


@pytest.mark.parametrize('fault', ['short', 'long', 'length', 'redirect', 'timeout'])
def test_download_caps_failure_cleans_only_own_partial(hocap, tmp_path, fault):
    raw = b'abc'; row = dict(url='https://utdallas.box.com/index.php?public', bytes=3)
    class Wrong:
        def open(self, *_a, **_k):
            return Response(b'ab' if fault == 'short' else b'abcd' if fault == 'long' else raw,
                'https://foreign.invalid' if fault == 'redirect' else row['url'], length='5' if fault == 'length' else '3')
    with pytest.raises((ValueError, TimeoutError)):
        hocap.fetch(row, tmp_path/'owned.zip', Wrong(), 0 if fault == 'timeout' else 10**12)
    assert not (tmp_path/'owned.zip').exists() and not (tmp_path/'owned.zip.part').exists()


@pytest.mark.parametrize('url', ['http://utdallas.box.com/x', 'https://utdallas.box.com.evil/x', 'https://u:p@utdallas.box.com/x', 'https://utdallas.box.com:444/x', 'https://evilboxcloud.com/x'])
def test_no_credentials_or_nonpublisher_destinations(hocap, url):
    with pytest.raises(ValueError): hocap.public_url(url)
    assert hocap.public_url('https://dl.public.boxcloud.com/path?opaque=not-logged')


def test_existing_output_and_foreign_partial_are_never_replaced(hocap, tmp_path, monkeypatch):
    _, sources, _ = manufacture(hocap, tmp_path, monkeypatch); target = tmp_path/'existing'; target.mkdir(); original = target/'mine'; original.write_bytes(b'preserve')
    with pytest.raises(ValueError): hocap.acquire(None, tmp_path, 'b'*40, target, opener=Opener(sources))
    assert original.read_bytes() == b'preserve'
    p = tmp_path/'occupied.zip.part'; p.write_bytes(b'foreign')
    with pytest.raises(ValueError): hocap.fetch(dict(url='https://utdallas.box.com/x', bytes=1), tmp_path/'occupied.zip', Opener({}), 10**12)
    assert p.read_bytes() == b'foreign'


def test_strict_reserved_namespace_and_late_fail_receipt(hocap, tmp_path, monkeypatch):
    _, sources, source = manufacture(hocap, tmp_path, monkeypatch); target = tmp_path/'reserved'; target.mkdir(mode=0o700)
    s = target.stat(); lease = dict(device=s.st_dev, inode=s.st_ino, source_sha256=source['closure_sha256'])
    assert hocap.acquire(None, tmp_path, 'b'*40, target, opener=Opener(sources), reservation=lease)['status'] == 'pass'
    report = dict(status='pass'); hocap.write_report(tmp_path/'late.json', report, deadline=0)
    assert json.loads((tmp_path/'late.json').read_bytes())['status'] == 'fail'
    with pytest.raises(FileExistsError): hocap.write_report(tmp_path/'late.json', {'status':'pass'})


def test_source_and_shell_no_gpu_model_old_assets_or_retry(hocap):
    root = Path(__file__).parents[1]; src = Path(hocap.__file__).read_text(); shell = (root/'infra/run_hocap_acquire.sh').read_text()
    for token in ('import torch', 'import numpy', 'safe_load', '.extractall(', 'gdown', 'np.load('):
        assert token not in src
    assert "'run_hocap_acquire'" in src and 'rt.source(ROOT, code, revision, ENTRY, HELPERS)' in src
    assert 'runuser -u scenesmith' in shell and '3660s' in shell and 'nvidia' not in shell and 'docker' not in shell
